"""SFT training logic: HuggingFace Trainer + optional LoRA."""

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

from .dataset import MedicalSFTDataset, StreamMedicalSFTDataset

# Ensure project root is on sys.path so evaluation module can be imported
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def is_main_process():
    """Check if current process is the main process (rank 0) in distributed training."""
    import torch.distributed as dist
    return not dist.is_initialized() or dist.get_rank() == 0


def resolve_attn_implementation(config: dict) -> str:
    """Select attention implementation with a safe fallback for older GPUs."""
    attn_impl = config.get("attn_implementation", "flash_attention_2")
    if attn_impl.startswith("flash") and torch.cuda.is_available():
        # FlashAttention 2 requires compute capability >= 8.0.
        min_cc_major = min(
            torch.cuda.get_device_capability(i)[0]
            for i in range(torch.cuda.device_count())
        )
        if min_cc_major < 8:
            if is_main_process():
                print(
                    f"[train] GPU compute capability {min_cc_major}.x does not support "
                    "FlashAttention 2. Falling back to 'sdpa'."
                )
            attn_impl = "sdpa"
    return attn_impl


class StreamDatasetCallback(TrainerCallback):
    """Callback to manage streaming data updates and waiting logic."""

    def __init__(
        self,
        dataset: StreamMedicalSFTDataset,
        epochs_per_data: int = 1,
        check_interval: float = 60,  # seconds between checks while waiting
    ):
        self.dataset = dataset
        self.epochs_per_data = epochs_per_data  # How many epochs to train on each data snapshot
        self.check_interval = check_interval  # How often to check for new data (in seconds)
        self.epochs_on_current_data = 0
        self.last_dataset_size = len(dataset)

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """After each epoch, check if we should wait for new data."""
        self.epochs_on_current_data += 1
        current_size = len(self.dataset)

        if is_main_process():
            print(f"[StreamCallback] Epoch {state.epoch} completed ({self.epochs_on_current_data}/{self.epochs_per_data} on current data)")

        # If dataset size changed, reset epoch counter (new data arrived during training)
        if current_size != self.last_dataset_size:
            if is_main_process():
                print(f"[StreamCallback] Dataset size changed: {self.last_dataset_size} → {current_size}")
            self.epochs_on_current_data = 0
            self.last_dataset_size = current_size
            return control

        # If we've trained enough epochs on current data, wait for new data
        if self.epochs_on_current_data >= self.epochs_per_data:
            if is_main_process():
                print(f"[StreamCallback] Exhausted current data. Waiting for updates...")
            has_new_data = self._wait_for_new_data()

            if has_new_data:
                # New data arrived, reset counter and continue training
                self.epochs_on_current_data = 0
                self.last_dataset_size = len(self.dataset)
                if is_main_process():
                    print(f"[StreamCallback] New data loaded. Dataset size: {self.last_dataset_size}")
            elif self.dataset.should_stop:
                # Timeout reached, stop training
                if is_main_process():
                    print("[StreamCallback] Timeout reached. Stopping training.")
                control.should_training_stop = True
            else:
                # This shouldn't happen, but handle gracefully
                if is_main_process():
                    print("[StreamCallback] Unexpected state. Stopping training.")
                control.should_training_stop = True

        return control

    def _wait_for_new_data(self) -> bool:
        """Wait for new data to arrive, checking periodically.

        Returns:
            True if new data arrived, False if timeout reached
        """
        start_time = time.time()

        while True:
            has_new_data = self.dataset.check_for_updates()

            if has_new_data:
                return True

            if self.dataset.should_stop:
                return False

            # Sleep before next check
            elapsed = time.time() - start_time
            remaining_timeout = self.dataset.max_pending_time - elapsed
            sleep_time = min(self.check_interval, remaining_timeout)

            if sleep_time > 0:
                if is_main_process():
                    print(f"[StreamCallback] Sleeping {sleep_time:.0f}s before next check...")
                time.sleep(sleep_time)
            else:
                # Timeout reached
                self.dataset.should_stop = True
                return False


class MedicalTrainer(Trainer):
    """Trainer subclass that logs perplexity before wandb sees the logs."""

    def log(self, logs, *args, **kwargs):
        loss = logs.get("loss")
        if loss is not None:
            logs["perplexity"] = math.exp(min(loss, 20))  # cap to avoid overflow
        super().log(logs, *args, **kwargs)


def load_model_and_tokenizer(config: dict):
    """Load model and tokenizer, optionally applying LoRA."""
    model_name = config["model_name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use low_cpu_mem_usage for efficient FSDP sharding
    fsdp_enabled = bool(config.get("fsdp", ""))
    attn_impl = resolve_attn_implementation(config)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        low_cpu_mem_usage=fsdp_enabled,
    )

    if config.get("use_lora", True):
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=config.get("lora_r", 64),
            lora_alpha=config.get("lora_alpha", 128),
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=config.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer


def build_training_args(config: dict) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from config dict."""
    # For stream training, set epochs based on epochs_per_data * expected data updates
    # Use a large number so callback controls when to stop
    if config.get("stream_training", False):
        num_epochs = config.get("num_epochs", 999999)
    else:
        num_epochs = config.get("num_epochs", 3)

    args = dict(
        output_dir=config.get("output_dir", "checkpoints/sft"),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=config.get("batch_size", 1),
        gradient_accumulation_steps=config.get("gradient_accumulation", 16),
        learning_rate=config.get("learning_rate", 2e-5),
        lr_scheduler_type=config.get("lr_scheduler", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        bf16=True,
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 3),
        report_to=config.get("report_to", "wandb"),
        run_name=config.get("run_name", "medagent-sft"),
        dataloader_num_workers=config.get("dataloader_num_workers", 4),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )

    # FSDP settings
    if config.get("fsdp"):
        args["fsdp"] = config["fsdp"]
    if config.get("fsdp_config"):
        args["fsdp_config"] = config["fsdp_config"]

    return TrainingArguments(**args)


def _build_eval_callback(config: dict, tokenizer):
    """Build a PeriodicEvaluationCallback if eval data is configured.

    Config keys (all optional):
      eval_data_path      : path to JSONL conversation file for eval
      eval_data_paths     : list/dict of per-dataset JSONL eval files
      eval_steps          : run eval every N steps (default 500)
      max_eval_diag_samples  : max diagnosis samples per run (default 50)
      max_eval_tool_samples  : max tool call samples per run (default 100)
      eval_max_new_tokens : max tokens to generate during eval (default 256)
      eval_random_sample  : randomly subsample each run (default True)
      eval_output_dir     : directory to save per-sample eval outputs as JSONL (optional)
      eval_debug_generate_heartbeat : print per-sample generate start/end heartbeats (default False)
      judge_base_url      : OpenAI-compatible API URL for judge (optional)
      judge_model_name    : judge model name / HF path (default Qwen2.5-1.5B)

    Async eval (offloads diagnosis/conversation eval to a separate GPU):
      eval_async          : enable async eval mode (default False)
      eval_async_gpu      : GPU index for async eval subprocess (default None = auto)

    Returns None if no eval data path is set.
    """
    eval_sources: dict[str, str] = {}
    if config.get("eval_data_paths"):
        paths = config["eval_data_paths"]
        if isinstance(paths, dict):
            eval_sources = dict(paths)
        else:
            for p in paths:
                stem = Path(p).stem.lower()
                if "ddxplus" in stem:
                    label = "ddxplus"
                elif "pmc" in stem:
                    label = "pmc"
                else:
                    label = Path(p).stem
                eval_sources[label] = p
    elif config.get("eval_data_path"):
        eval_sources = {"eval": config["eval_data_path"]}

    if not eval_sources:
        return None

    from evaluation.callbacks import PeriodicEvaluationCallback

    eval_datasets: dict[str, list[dict]] = {}
    for label, path in eval_sources.items():
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            eval_datasets[label] = records

    if not eval_datasets:
        if is_main_process():
            print("[EvalCallback] eval data is empty, skipping evaluation setup.")
        return None

    judge_base_url = config.get("judge_base_url")
    judge_model_name = config.get("judge_model_name", "Qwen/Qwen2.5-1.5B-Instruct")

    if judge_base_url:
        from openai import OpenAI
        judge_model = OpenAI(base_url=judge_base_url)
        judge_tokenizer = judge_model_name  # str triggers API dispatch
        if is_main_process():
            print(f"[EvalCallback] Judge: API at {judge_base_url} ({judge_model_name})")
    else:
        if is_main_process():
            print(f"[EvalCallback] Loading local judge model: {judge_model_name}")
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        judge_tok = AutoTokenizer.from_pretrained(judge_model_name, trust_remote_code=True)
        if judge_tok.pad_token is None:
            judge_tok.pad_token = judge_tok.eos_token
        judge_model.eval()
        judge_tokenizer = judge_tok

    # Patient LLM for conversation eval (optional)
    patient_client = None
    patient_model = None
    patient_base_url = config.get("patient_base_url")
    if patient_base_url:
        from openai import OpenAI as _PatientOpenAI
        patient_client = _PatientOpenAI(base_url=patient_base_url)
        patient_model = config.get("patient_model_name", "gpt-4.1-mini")
        if is_main_process():
            print(f"[EvalCallback] Patient LLM: {patient_model} at {patient_base_url}")

    # Async eval config
    async_eval = config.get("eval_async", False)
    async_eval_gpu = config.get("eval_async_gpu")

    # In async mode, the subprocess needs string config (not live objects)
    # for judge and patient — it creates its own clients.
    async_kwargs = {}
    if async_eval:
        if not judge_base_url:
            if is_main_process():
                print("[EvalCallback] WARNING: eval_async requires judge_base_url (API judge). "
                      "Falling back to sync eval.")
            async_eval = False
        else:
            async_kwargs = dict(
                async_eval=True,
                async_eval_gpu=async_eval_gpu,
                eval_data_path=config.get("eval_data_path"),
                eval_data_paths=eval_sources,
                async_judge_base_url=judge_base_url,
                async_judge_model_name=judge_model_name,
                async_patient_base_url=patient_base_url,
                async_patient_model_name=config.get("patient_model_name", "gpt-4.1-mini"),
                async_weave_project=config.get("wandb_project"),
            )

    return PeriodicEvaluationCallback(
        eval_records=[],
        eval_datasets=eval_datasets,
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
        tokenizer=tokenizer,
        eval_steps=config.get("eval_steps", 500),
        max_diag_samples=config.get("max_eval_diag_samples", 50),
        max_conv_samples=config.get("max_eval_conv_samples", 10),
        max_new_tokens=config.get("eval_max_new_tokens", 256),
        max_conv_turns=config.get("max_eval_conv_turns", 15),
        random_sample=config.get("eval_random_sample", True),
        eval_output_dir=config.get("eval_output_dir"),
        debug_generate_heartbeat=config.get("eval_debug_generate_heartbeat", False),
        eval_before_train=config.get("eval_before_train", True),
        patient_client=patient_client,
        patient_model=patient_model,
        enable_diagnosis_eval=config.get("eval_diagnosis", True),
        enable_conversation_eval=config.get("eval_conversation", True),
        checkpoint_output_dir=config.get("output_dir"),
        best_checkpoint_top_k=config.get("top_k_checkpoints", 2),
        best_checkpoint_metrics=config.get("best_checkpoint_metrics"),
        **async_kwargs,
    )


def train(config: dict):
    """Run the full SFT training pipeline."""
    # Set wandb project before Trainer init triggers wandb.init
    if config.get("wandb_project"):
        os.environ.setdefault("WANDB_PROJECT", config["wandb_project"])

    # Initialize weave for LLM call tracing (auto-patches OpenAI client)
    try:
        import weave
        weave.init(config.get("wandb_project", "medagent-sft"))
    except ImportError:
        pass

    model, tokenizer = load_model_and_tokenizer(config)

    # Choose dataset type based on streaming mode
    stream_training = config.get("stream_training", False)
    if stream_training:
        dataset = StreamMedicalSFTDataset(
            data_path=config.get("data_path", "data/conversations/conversations.jsonl"),
            tokenizer=tokenizer,
            max_length=config.get("max_length", 4096),
            max_pending_time=config.get("max_pending_time", 1800),  # 30 min default
        )
        if is_main_process():
            print(f"[Stream Training] Loaded {len(dataset)} conversations (monitoring for updates)")
    else:
        dataset = MedicalSFTDataset(
            data_path=config.get("data_path", "data/conversations/conversations.jsonl"),
            tokenizer=tokenizer,
            max_length=config.get("max_length", 4096),
        )
        if is_main_process():
            print(f"Loaded {len(dataset)} conversations")

    training_args = build_training_args(config)

    # Add streaming callback if in stream mode
    callbacks = []
    if stream_training:
        callbacks.append(StreamDatasetCallback(
            dataset=dataset,
            epochs_per_data=config.get("epochs_per_data", 1),
            check_interval=config.get("stream_check_interval", 60),
        ))

    # Add periodic evaluation callback if eval_data_path is configured
    eval_callback = _build_eval_callback(config, tokenizer)
    if eval_callback is not None:
        callbacks.append(eval_callback)

    trainer = MedicalTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=callbacks,
    )

    def _save_resumable_checkpoint(reason: str):
        if trainer.state.global_step <= 0:
            print(f"[Training] Skipping resumable checkpoint save on {reason}: no completed steps yet.")
            return
        try:
            print(
                f"[Training] Saving resumable checkpoint at step {trainer.state.global_step} "
                f"due to {reason}..."
            )
            trainer._save_checkpoint(trainer.model, trial=None)
        except Exception as save_exc:
            print(f"[Training] Failed to save resumable checkpoint: {save_exc}")

    # Resume from checkpoint if specified
    resume_from = config.get("resume_from_checkpoint")
    if resume_from == "latest":
        # Auto-detect the latest checkpoint in output_dir
        output_dir = config.get("output_dir", "checkpoints/sft")
        if os.path.isdir(output_dir):
            ckpts = [
                os.path.join(output_dir, d)
                for d in os.listdir(output_dir)
                if d.startswith("checkpoint-")
            ]
            resume_from = max(ckpts, key=os.path.getmtime) if ckpts else None
        else:
            resume_from = None
        if resume_from:
            print(f"Auto-detected latest checkpoint: {resume_from}")
        else:
            print("No existing checkpoints found, training from scratch.")

    if resume_from and resume_from != "latest":
        print(f"Resuming training from {resume_from}")

    try:
        trainer.train(resume_from_checkpoint=resume_from)
    except KeyboardInterrupt:
        print("\n[Training] Interrupted by user.")
        _save_resumable_checkpoint("interrupt")
    except Exception as e:
        print(f"\n[Training] Error occurred: {e}")
        _save_resumable_checkpoint("error")
        raise
    finally:
        # Always save final checkpoint (whether completed, interrupted, or timed out)
        final_dir = f"{config.get('output_dir', 'checkpoints/sft')}/final"
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)
        print(f"Saved final model to {final_dir}")

        if stream_training and dataset.should_stop:
            print(f"[Stream Training] Stopped due to timeout after {config.get('max_pending_time', 1800)}s without new data")
