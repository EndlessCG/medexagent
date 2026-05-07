"""SFT dataset: tokenize doctor-patient conversations with assistant-only loss masking."""

import json
import os
import time
from pathlib import Path
from threading import Lock

import torch
from torch.utils.data import Dataset


def is_main_process():
    """Check if current process is the main process (rank 0) in distributed training."""
    try:
        import torch.distributed as dist
        return not dist.is_initialized() or dist.get_rank() == 0
    except Exception:
        return True  # If distributed not available, assume main process


def serialize_tool_calls(messages: list[dict]) -> list[dict]:
    """Serialize tool_calls into message content for tokenizers that don't support them natively.

    Assistant messages with tool_calls get the calls serialized as text in the content field.
    The format uses <tool_call> tags so the model learns to emit parseable tool calls.
    Tool role messages are converted to user role messages so the result is
    strictly system/assistant/user — compatible with any chat API.
    """
    out = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            parts = []
            if msg.get("content"):
                parts.append(msg["content"])
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                call_obj = json.dumps({"name": name, "arguments": args})
                parts.append(f"<tool_call>\n{call_obj}\n</tool_call>")
            out.append({"role": "assistant", "content": "\n\n".join(parts)})
        elif msg.get("role") == "tool":
            out.append({"role": "user", "content": msg.get("content", "")})
        else:
            out.append(msg)
    return out


class MedicalSFTDataset(Dataset):
    """Dataset for supervised fine-tuning on medical conversations.

    Each sample is a full multi-turn conversation in OpenAI chat format.
    Only assistant tokens (including tool_call messages) contribute to the loss;
    system / user / tool tokens are masked with label = -100.
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.conversations = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    self.conversations.append(entry["conversation"])

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        messages = self.conversations[idx]
        return self._tokenize_with_masking(messages)

    def _tokenize_with_masking(self, messages: list[dict]) -> dict:
        """Tokenize conversation and build labels with assistant-only masking.

        Strategy: tokenize incrementally — for each message i, compute the token
        delta between prefix[:i] and prefix[:i+1]. Assign labels based on the
        role of message i.
        """
        # Serialize tool_calls into content so the tokenizer can see them
        messages = serialize_tool_calls(messages)

        # Build the full input_ids using the chat template
        full_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=None,
            tokenize=True,
            add_generation_prompt=False,
        )

        # Build labels: start with all -100, then unmask assistant segments
        labels = [-100] * len(full_ids)

        # Incrementally tokenize to find each message's token boundaries
        prev_len = 0
        for i in range(len(messages)):
            prefix = messages[: i + 1]
            prefix_ids = self.tokenizer.apply_chat_template(
                prefix,
                tools=None,
                tokenize=True,
                add_generation_prompt=False,
            )
            cur_len = len(prefix_ids)

            role = messages[i].get("role", "")
            # Train on assistant messages (including tool_call messages where
            # role=assistant and content is None but tool_calls is present)
            if role == "assistant":
                for j in range(prev_len, min(cur_len, len(full_ids))):
                    labels[j] = full_ids[j]

            prev_len = cur_len

        # Truncate to max_length
        full_ids = full_ids[: self.max_length]
        labels = labels[: self.max_length]

        # Pad to max_length
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        pad_len = self.max_length - len(full_ids)
        attention_mask = [1] * len(full_ids) + [0] * pad_len
        full_ids = full_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class StreamMedicalSFTDataset(MedicalSFTDataset):
    """Streaming dataset that monitors data file for updates during training.

    This dataset loads the initial data and continues to monitor the file
    for new entries. When the trainer exhausts the current data, it checks
    for updates. If no updates arrive within max_pending_time, training stops.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 4096,
        max_pending_time: float = 1800,  # 30 minutes default
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data_path = data_path
        self.max_pending_time = max_pending_time

        self.conversations = []
        self.lock = Lock()  # Thread-safe access to conversations
        self.last_check_time = time.time()
        self.last_size = 0
        self.pending_start_time = None
        self.should_stop = False

        # Initial load
        self._load_data()
        if is_main_process():
            print(f"[StreamDataset] Initially loaded {len(self.conversations)} conversations")

    def _load_data(self):
        """Load all conversations from the data file."""
        if not os.path.exists(self.data_path):
            return

        new_conversations = []
        with open(self.data_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    new_conversations.append(entry["conversation"])

        with self.lock:
            self.conversations = new_conversations
            self.last_size = len(self.conversations)
            self.last_check_time = time.time()

    def check_for_updates(self) -> bool:
        """Check if the data file has been updated with new entries.

        Returns:
            True if new data was loaded, False otherwise
        """
        # Check if file exists and has been modified
        if not os.path.exists(self.data_path):
            return False

        # Count lines in the file
        with open(self.data_path) as f:
            num_lines = sum(1 for line in f if line.strip())

        if num_lines > self.last_size:
            # New data available
            if is_main_process():
                print(f"[StreamDataset] Detected {num_lines - self.last_size} new conversations")
            self._load_data()
            return True
        else:
            # No new data
            return False

    def __len__(self):
        with self.lock:
            return len(self.conversations)

    def __getitem__(self, idx):
        with self.lock:
            if idx >= len(self.conversations):
                # This shouldn't happen, but handle gracefully
                idx = len(self.conversations) - 1
            messages = self.conversations[idx]
        return self._tokenize_with_masking(messages)
