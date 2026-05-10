#!/usr/bin/env python3
"""Interactive CLI chat with a checkpointed medical agent model."""

import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.tool import TOOL_SCHEMAS, get_tool_info
from data.prompt import get_doctor_system_prompt
from train.sft.dataset import serialize_tool_calls


# Tool presets for interactive selection
TOOL_PRESETS = {
    "a": {
        "label": "No exams (conversation only)",
        "tools": [],
    },
    "b": {
        "label": "Basic (xray, ct, blood_test)",
        "tools": ["xray", "ct", "blood_test"],
    },
    "c": {
        "label": "Standard (10 common exams)",
        "tools": [
            "xray", "ct", "mri", "ultrasound", "blood_test",
            "cbc", "ecg", "urinalysis", "biopsy", "culture",
        ],
    },
    "d": {
        "label": "All exams (65 tools)",
        "tools": None,  # sentinel for all tools
    },
}


def _get_all_tool_infos() -> list[dict]:
    """Extract tool info dicts from TOOL_SCHEMAS for the system prompt."""
    return [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "parameters": t["function"]["parameters"],
        }
        for t in TOOL_SCHEMAS
    ]


def _get_tool_infos_by_name(names: list[str]) -> list[dict]:
    """Get tool info dicts for a list of tool names."""
    tools = []
    for name in names:
        info = get_tool_info(name)
        if info:
            tools.append(info)
    return tools


def _select_tool_preset() -> list[dict]:
    """Prompt user to select a tool preset. Returns list of tool info dicts."""
    print("Select available exams:")
    for key, preset in TOOL_PRESETS.items():
        print(f"  {key}) {preset['label']}")
    while True:
        try:
            choice = input("Choice [c]: ").strip().lower() or "c"
        except (EOFError, KeyboardInterrupt):
            print()
            choice = "c"
        if choice in TOOL_PRESETS:
            preset = TOOL_PRESETS[choice]
            if preset["tools"] is None:
                return _get_all_tool_infos()
            return _get_tool_infos_by_name(preset["tools"])
        print(f"  Invalid choice '{choice}', pick one of: {', '.join(TOOL_PRESETS)}")


def find_checkpoint(checkpoint_path: str | None) -> str:
    """Find a checkpoint directory to load."""
    if checkpoint_path:
        if not os.path.isdir(checkpoint_path):
            print(f"Error: checkpoint path '{checkpoint_path}' does not exist.")
            sys.exit(1)
        return checkpoint_path

    # Auto-discover: try final/ first, then most recent checkpoint-*
    base = "checkpoints/sft"
    final = os.path.join(base, "final")
    if os.path.isdir(final):
        return final

    if os.path.isdir(base):
        ckpts = sorted(
            [d for d in os.listdir(base) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0,
        )
        if ckpts:
            return os.path.join(base, ckpts[-1])

    print("Error: no checkpoint found. Use --checkpoint to specify a path.")
    sys.exit(1)


def _has_tokenizer_files(path: str) -> bool:
    """Return True if the path looks like it contains tokenizer artifacts."""
    candidates = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    return any(os.path.exists(os.path.join(path, fname)) for fname in candidates)


def _infer_repo_model_name() -> str | None:
    """Try to infer a default model name from config/sft_config.yaml."""
    repo_root = os.path.dirname(__file__)
    config_path = os.path.join(repo_root, "config", "sft_config.yaml")
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, "r") as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                if stripped.startswith("model_name:"):
                    value = stripped.split(":", 1)[1].strip()
                    return value.strip("'\"") if value else None
    except OSError:
        return None
    return None


def _resolve_tokenizer_source(
    checkpoint_path: str,
    tokenizer_override: str | None,
    base_model_name: str | None,
) -> str:
    if tokenizer_override:
        return tokenizer_override
    if _has_tokenizer_files(checkpoint_path):
        return checkpoint_path
    if base_model_name:
        return base_model_name
    inferred = _infer_repo_model_name()
    if inferred:
        print(f"Tokenizer not found in checkpoint. Using repo model_name: {inferred}")
        return inferred
    raise FileNotFoundError(
        "Tokenizer files not found in checkpoint. Provide --tokenizer with a model name/path."
    )


def _is_verl_rl_checkpoint(path: str) -> bool:
    """Check if path is a verl RL checkpoint (has actor/ with FSDP shards)."""
    actor_dir = os.path.join(path, "actor")
    if not os.path.isdir(actor_dir):
        return False
    # Look for model_world_size_*_rank_*.pt files
    return any(
        f.startswith("model_world_size_") and f.endswith(".pt")
        for f in os.listdir(actor_dir)
    )


def _load_verl_rl_checkpoint(checkpoint_path: str, tokenizer_override: str | None = None):
    """Load model from a verl RL checkpoint with FSDP-sharded DTensors."""
    actor_dir = os.path.join(checkpoint_path, "actor")
    hf_dir = os.path.join(actor_dir, "huggingface")

    # Discover world size and load shards
    shard_files = sorted(
        f for f in os.listdir(actor_dir)
        if f.startswith("model_world_size_") and f.endswith(".pt")
    )
    world_size = len(shard_files)
    print(f"Found {world_size} FSDP shards, consolidating...")

    shards = []
    for f in shard_files:
        d = torch.load(os.path.join(actor_dir, f), map_location="cpu", weights_only=False)
        shards.append(d)

    # Consolidate: concatenate local tensors along the shard dimension
    consolidated = {}
    for key in shards[0].keys():
        local_tensors = []
        for shard in shards:
            t = shard[key]
            # Extract local tensor from DTensor
            if hasattr(t, "_local_tensor"):
                local_t = t._local_tensor
                shard_dim = t.placements[0].dim if hasattr(t.placements[0], "dim") else 0
            else:
                local_t = t
                shard_dim = 0
            local_tensors.append(local_t)
        consolidated[key] = torch.cat(local_tensors, dim=shard_dim)

    # Free shard memory
    del shards

    # Load model architecture from huggingface config, then load consolidated weights
    from transformers import AutoConfig
    print("Loading model architecture from config...")
    config = AutoConfig.from_pretrained(hf_dir, trust_remote_code=True)
    # Cast consolidated weights to bfloat16
    for key in consolidated:
        if consolidated[key].dtype.is_floating_point:
            consolidated[key] = consolidated[key].to(torch.bfloat16)
    # Instantiate model on CPU (initializes all buffers properly), load weights, then move to GPU
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.load_state_dict(consolidated, strict=True)
    del consolidated
    model = model.to("cuda")
    print("Consolidated weights loaded.")

    # Load tokenizer
    tokenizer_source = tokenizer_override or hf_dir
    if not _has_tokenizer_files(tokenizer_source):
        inferred = _infer_repo_model_name()
        if inferred:
            tokenizer_source = inferred
        else:
            raise FileNotFoundError(
                "Tokenizer not found. Provide --tokenizer with a model name/path."
            )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("Model loaded and ready.\n")
    return model, tokenizer


def load_model(checkpoint_path: str, tokenizer_override: str | None = None):
    """Load base model + LoRA adapter and tokenizer from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path} ...")

    # Check if this is a verl RL checkpoint
    if _is_verl_rl_checkpoint(checkpoint_path):
        return _load_verl_rl_checkpoint(checkpoint_path, tokenizer_override)

    # Check if this is a LoRA adapter checkpoint
    adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
    base_model_name = None
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path) as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config["base_model_name_or_path"]
        print(f"Base model: {base_model_name}")

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load and merge LoRA
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        print("LoRA adapter merged.")
    else:
        # Full model checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    tokenizer_source = _resolve_tokenizer_source(
        checkpoint_path, tokenizer_override, base_model_name
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("Model loaded and ready.\n")
    return model, tokenizer


def parse_tool_calls(raw_output: str) -> list[dict] | None:
    """Parse tool calls from model output.

    Handles <tool_call> tags, Llama 3.1 style <|python_tag|>, or JSON function calls.
    """
    tool_calls = []

    # Pattern 0: <tool_call>...</tool_call> tags (our SFT training format)
    tc_matches = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', raw_output, re.DOTALL)
    for match in tc_matches:
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict) and "name" in parsed:
                # Normalize: use "parameters" key for consistency
                params = parsed.get("arguments", parsed.get("parameters", {}))
                tool_calls.append({"name": parsed["name"], "parameters": params})
        except json.JSONDecodeError:
            continue
    if tool_calls:
        return tool_calls

    # Pattern 1: <|python_tag|> style (Llama 3.1)
    if "<|python_tag|>" in raw_output:
        tag_content = raw_output.split("<|python_tag|>", 1)[1]
        # Remove any trailing special tokens
        for stop in ["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>"]:
            tag_content = tag_content.split(stop)[0]
        tag_content = tag_content.strip()
        # Try to parse as JSON
        try:
            parsed = json.loads(tag_content)
            if isinstance(parsed, dict) and "name" in parsed:
                tool_calls.append(parsed)
                return tool_calls
        except json.JSONDecodeError:
            pass
        # Try function call syntax: func_name(args)
        m = re.match(r"(\w+)\((.*)\)$", tag_content, re.DOTALL)
        if m:
            func_name = m.group(1)
            args_str = m.group(2).strip()
            args = {}
            if args_str:
                for pm in re.finditer(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"', args_str):
                    args[pm.group(1)] = pm.group(2)
            tool_calls.append({"name": func_name, "parameters": args})
            return tool_calls

    # Pattern 2: JSON tool call in output {"name": ..., "parameters": ...}
    json_pattern = re.findall(r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*\}', raw_output)
    for match in json_pattern:
        try:
            parsed = json.loads(match)
            if "name" in parsed:
                tool_calls.append(parsed)
        except json.JSONDecodeError:
            continue
    if tool_calls:
        return tool_calls

    # Pattern 3: function call syntax in text: func_name(key="val")
    func_pattern = re.findall(r'(\w+)\(([^)]*)\)', raw_output)
    for func_name, args_str in func_pattern:
        # Only match known tool names
        known_names = {t["function"]["name"] for t in TOOL_SCHEMAS}
        if func_name in known_names:
            args = {}
            for pm in re.finditer(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"', args_str):
                args[pm.group(1)] = pm.group(2)
            tool_calls.append({"name": func_name, "parameters": args})
    return tool_calls if tool_calls else None


def simulate_tool_result(name: str, params: dict) -> str:
    """Prompt the user to provide a tool result."""
    import readline

    exam_name = name.replace("_", " ")
    prefill = f"The {exam_name} exam shows "

    def prefill_hook():
        readline.insert_text(prefill)
        readline.redisplay()
        readline.set_pre_input_hook(None)

    readline.set_pre_input_hook(prefill_hook)
    try:
        result = input("  Tool result> ")
    except (EOFError, KeyboardInterrupt):
        result = prefill.rstrip()
    finally:
        readline.set_pre_input_hook(None)
    return result


def extract_text_before_tool(raw_output: str) -> str:
    """Extract text the assistant said before any tool call marker."""
    # Strip text after tool call markers
    if "<tool_call>" in raw_output:
        text = raw_output.split("<tool_call>")[0]
    elif "<|python_tag|>" in raw_output:
        text = raw_output.split("<|python_tag|>")[0]
    else:
        text = raw_output
    # Remove special tokens
    for tok in ["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<|begin_of_text|>"]:
        text = text.replace(tok, "")
    return text.strip()


def generate_response(model, tokenizer, messages, max_new_tokens, temperature):
    """Generate a response from the model given message history."""
    # Serialize tool_calls into content so tokenizers that don't support them
    # natively (e.g. Meditron3/Llama3 base template) can still see them.
    preprocessed = serialize_tool_calls(messages)
    # Use tools=None to match SFT training, which serializes tool_calls as
    # <tool_call> tags in content and lists tools in the system prompt text.
    # Passing tools=TOOL_SCHEMAS here would activate the tokenizer's native
    # tool-calling template (different format), causing OOD behavior.
    input_text = tokenizer.apply_chat_template(
        preprocessed,
        tools=None,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    raw_output = tokenizer.decode(new_tokens, skip_special_tokens=False)

    return raw_output


def clean_response(raw_output: str) -> str:
    """Remove special tokens from response for display."""
    text = raw_output
    for tok in [
        "<|eot_id|>", "<|eom_id|>", "<|end_of_text|>",
        "<|begin_of_text|>", "<|python_tag|>",
        "<|start_header_id|>", "<|end_header_id|>",
        "<tool_call>", "</tool_call>",
    ]:
        text = text.replace(tok, "")
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Chat with a medical agent model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint directory")
    parser.add_argument("--hf-model", type=str, default=None,
                        help="HuggingFace model name or path (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer model name or path (used when checkpoint lacks tokenizer files)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum new tokens to generate (default: 512)")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Sampling temperature (default: 0)")
    args = parser.parse_args()

    if args.hf_model:
        print(f"Loading HuggingFace model: {args.hf_model} ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_model,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        print("Model loaded and ready.\n")
    else:
        checkpoint_path = find_checkpoint(args.checkpoint)
        model, tokenizer = load_model(checkpoint_path, tokenizer_override=args.tokenizer)

    selected_tools = _select_tool_preset()
    print(f"  -> {len(selected_tools)} tools available\n")
    system_prompt = get_doctor_system_prompt(
        medical_history="",
        available_tools=selected_tools,
        include_tool_call_format=True,
    )

    messages = [{"role": "system", "content": system_prompt}]

    print("=" * 60)
    print("Medical Agent Chat")
    print("You are the patient. The model is the doctor.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("=" * 60)
    print()

    # Generate initial greeting from the doctor
    print("Doctor: ", end="", flush=True)
    raw = generate_response(model, tokenizer, messages, args.max_new_tokens, args.temperature)
    greeting = clean_response(raw)
    print(greeting)
    messages.append({"role": "assistant", "content": greeting})
    print()

    tool_call_counter = 0
    MAX_CONSECUTIVE_TOOL_CALLS = 10

    while True:
        try:
            user_input = input("Patient: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Generate doctor response (may include tool calls)
        consecutive_tool_calls = 0
        while True:
            raw = generate_response(
                model, tokenizer, messages,
                args.max_new_tokens, args.temperature,
            )

            # Check for tool calls
            tool_calls = parse_tool_calls(raw)

            if tool_calls and consecutive_tool_calls < MAX_CONSECUTIVE_TOOL_CALLS:
                consecutive_tool_calls += 1

                # Print any text before the tool call
                pre_text = extract_text_before_tool(raw)
                if pre_text:
                    print(f"\nDoctor: {pre_text}")

                # Build single assistant message with all tool calls from this generation
                tc_entries = []
                for tc in tool_calls:
                    tool_call_counter += 1
                    tc_name = tc["name"]
                    tc_params = tc.get("parameters", tc.get("arguments", {}))
                    if isinstance(tc_params, str):
                        try:
                            tc_params = json.loads(tc_params)
                        except json.JSONDecodeError:
                            tc_params = {}

                    param_str = ", ".join(f'{k}="{v}"' for k, v in tc_params.items())
                    print(f"\n  [Exam ordered: {tc_name}({param_str})]")

                    tc_id = f"call_{tool_call_counter}"
                    tc_entries.append({
                        "tc": tc,
                        "id": tc_id,
                        "name": tc_name,
                        "params": tc_params,
                    })

                # Add single assistant message with all tool calls
                messages.append({
                    "role": "assistant",
                    "content": pre_text if pre_text else None,
                    "tool_calls": [
                        {
                            "id": entry["id"],
                            "type": "function",
                            "function": {
                                "name": entry["name"],
                                "arguments": json.dumps(entry["params"]),
                            },
                        }
                        for entry in tc_entries
                    ],
                })

                # Prompt user for results and add tool result messages
                for entry in tc_entries:
                    result = simulate_tool_result(entry["name"], entry["params"])
                    print(f"  [Result: {result}]")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": entry["id"],
                        "content": result,
                    })

                # Continue generating to get the doctor's response after tool results
                continue
            else:
                if consecutive_tool_calls >= MAX_CONSECUTIVE_TOOL_CALLS:
                    print(f"\n  [Stopped: reached {MAX_CONSECUTIVE_TOOL_CALLS} consecutive tool calls]")
                # Normal text response
                response = clean_response(raw)
                print(f"\nDoctor: {response}")
                messages.append({"role": "assistant", "content": response})

                # Check for diagnosis
                diag_match = re.search(r'\[DIAGNOSIS:\s*(.+?)\]', response)
                if diag_match:
                    print(f"\n{'=' * 60}")
                    print(f"DIAGNOSIS: {diag_match.group(1)}")
                    print(f"{'=' * 60}")
                    try:
                        cont = input("\nContinue conversation? (y/n): ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        print("\nGoodbye!")
                        return
                    if cont != "y":
                        print("Goodbye!")
                        return
                break

        print()


if __name__ == "__main__":
    main()
