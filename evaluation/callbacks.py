"""Training callbacks for periodic evaluation during SFT."""

import json
import math
import os
import random
import shutil
import subprocess
import sys

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl

from .data import (
    prepare_diagnosis_eval_samples,
    prepare_conversation_eval_samples,
)
from .runners import (
    run_evaluation,
    run_diagnosis_eval,
    run_conversation_eval,
)
from .utils import _is_main_process, _get_dist_info, _is_fsdp_model


class PeriodicEvaluationCallback(TrainerCallback):
    """HF TrainerCallback for periodic evaluation during SFT training.

    Supports two modes:

    **Sync mode** (``async_eval=False``, default):
        All eval types run inline on the training GPUs.  Diagnosis and
        conversation eval run on all ranks (multi-turn, variable generate
        calls).  Tool call eval is split across ranks.

    **Async mode** (``async_eval=True``):
        Tool call eval still runs inline (fast, single-turn).  Diagnosis
        and conversation eval are offloaded to a persistent worker
        subprocess on a dedicated GPU.  The callback saves a checkpoint
        and enqueues a job; the worker processes jobs sequentially and
        never skips.  The worker continues running after training ends
        until all queued jobs are done.

    FSDP support
    ------------
    ``summon_full_params`` is used when the model is FSDP-wrapped.  In sync
    mode it enables local inference.  In async mode it enables checkpoint
    saving.  The context manager is a collective — all ranks enter and exit
    together.
    """

    def __init__(
        self,
        eval_records: list[dict],
        judge_model,
        judge_tokenizer,
        tokenizer,
        eval_datasets: dict[str, list[dict]] | None = None,
        eval_steps: int = 500,
        max_diag_samples: int = 50,
        max_conv_samples: int = 10,
        max_new_tokens: int = 256,
        max_conv_turns: int = 15,
        random_sample: bool = True,
        eval_output_dir: str | None = None,
        debug_generate_heartbeat: bool = False,
        eval_before_train: bool = True,
        patient_client=None,
        patient_model: str | None = None,
        enable_diagnosis_eval: bool = True,
        enable_conversation_eval: bool = True,
        # --- Async eval options ---
        async_eval: bool = False,
        async_eval_gpu: int | None = None,
        eval_data_path: str | None = None,
        eval_data_paths: dict[str, str] | None = None,
        async_judge_base_url: str | None = None,
        async_judge_model_name: str | None = None,
        async_patient_base_url: str | None = None,
        async_patient_model_name: str | None = None,
        async_weave_project: str | None = None,
        checkpoint_output_dir: str | None = None,
        best_checkpoint_top_k: int = 0,
        best_checkpoint_metrics: list[str] | None = None,
    ):
        self.enable_diagnosis_eval = enable_diagnosis_eval
        self.enable_conversation_eval = enable_conversation_eval
        self.eval_datasets = eval_datasets or {"eval": eval_records}

        # In async mode, diagnosis/conversation samples aren't needed inline.
        self.diag_samples_by_dataset: dict[str, list[dict]] = {}
        self.conv_samples_by_dataset: dict[str, list[dict]] = {}
        for label, records in self.eval_datasets.items():
            if async_eval:
                self.diag_samples_by_dataset[label] = []
                self.conv_samples_by_dataset[label] = []
            else:
                self.diag_samples_by_dataset[label] = (
                    prepare_diagnosis_eval_samples(records) if enable_diagnosis_eval else []
                )
                self.conv_samples_by_dataset[label] = (
                    prepare_conversation_eval_samples(records) if enable_conversation_eval else []
                )
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.max_diag_samples = max_diag_samples
        self.max_conv_samples = max_conv_samples
        self.max_new_tokens = max_new_tokens
        self.max_conv_turns = max_conv_turns
        self.random_sample = random_sample
        self.eval_output_dir = eval_output_dir
        self.debug_generate_heartbeat = debug_generate_heartbeat
        self.eval_before_train = eval_before_train
        self.patient_client = patient_client
        self.patient_model = patient_model
        self.last_eval_step = 0

        # Async eval state
        self.async_eval = async_eval
        self.async_eval_gpu = async_eval_gpu
        self.eval_data_path = eval_data_path
        self.eval_data_paths = eval_data_paths or (
            {"eval": eval_data_path} if eval_data_path else None
        )
        self.async_judge_base_url = async_judge_base_url
        self.async_judge_model_name = async_judge_model_name
        self.async_patient_base_url = async_patient_base_url
        self.async_patient_model_name = async_patient_model_name
        self.async_weave_project = async_weave_project
        self._worker_proc: subprocess.Popen | None = None
        self._worker_log_file = None
        self.checkpoint_output_dir = checkpoint_output_dir
        self.best_checkpoint_top_k = max(0, int(best_checkpoint_top_k or 0))
        self.best_checkpoint_metrics = list(best_checkpoint_metrics or [
            "diagnosis_reward",
            "conv_diagnosis_reward",
            "conv_tool_call_reward",
        ])
        self._best_ckpts: list[tuple[float, int, str]] = []

        if async_eval:
            self._queue_dir = os.path.join(
                eval_output_dir or "/tmp/medagent_async_eval", "eval_queue",
            )
            os.makedirs(os.path.join(self._queue_dir, "jobs"), exist_ok=True)
            os.makedirs(os.path.join(self._queue_dir, "results"), exist_ok=True)
            # Remove stale STOP sentinel from previous run
            stop_file = os.path.join(self._queue_dir, "STOP")
            if os.path.exists(stop_file):
                os.remove(stop_file)
        else:
            self._queue_dir = None

        self._load_best_checkpoints()

        if _is_main_process():
            enabled = []
            if enable_diagnosis_eval:
                if async_eval:
                    enabled.append(f"{len(self.eval_datasets)} diagnosis datasets (async)")
                else:
                    total = sum(len(v) for v in self.diag_samples_by_dataset.values())
                    enabled.append(f"{total} diagnosis across {len(self.eval_datasets)} dataset(s)")
            if enable_conversation_eval:
                if async_eval:
                    enabled.append(f"{len(self.eval_datasets)} conversation datasets (async)")
                else:
                    total = sum(len(v) for v in self.conv_samples_by_dataset.values())
                    enabled.append(f"{total} conversation across {len(self.eval_datasets)} dataset(s)")
            mode = "async" if async_eval else "sync"
            print(
                f"[PeriodicEvalCallback] {', '.join(enabled)} samples, "
                f"eval every {eval_steps} steps ({mode} mode)"
            )
            if self.best_checkpoint_top_k > 0 and self.checkpoint_output_dir:
                print(
                    f"[PeriodicEvalCallback] best-checkpoint tracking enabled "
                    f"(top_k={self.best_checkpoint_top_k})"
                )

    # ------------------------------------------------------------------
    # Trainer hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Run eval once at step 0 before any training."""
        if self.eval_before_train and model is not None:
            self._run_eval(model, step=0)
        return control

    def on_step_end(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        model=None,
        **kwargs,
    ):
        """Trigger evaluation every eval_steps."""
        if self.async_eval:
            self._collect_async_results()

        current_step = state.global_step
        if current_step == 0 or (current_step - self.last_eval_step) < self.eval_steps:
            return control

        self.last_eval_step = current_step
        self._run_eval(model, step=current_step)
        return control

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Signal worker to stop after finishing its queue.  Don't block."""
        if self.async_eval and self._queue_dir:
            # Collect any results that are ready
            self._collect_async_results()
            # Write STOP sentinel — worker will exit once queue is drained
            stop_file = os.path.join(self._queue_dir, "STOP")
            if _is_main_process():
                with open(stop_file, "w") as f:
                    f.write("stop\n")
                remaining = self._count_pending_jobs()
                if remaining > 0:
                    print(
                        f"[AsyncEval] STOP written. Worker will continue "
                        f"processing {remaining} remaining job(s)."
                    )
                else:
                    print("[AsyncEval] STOP written. No pending jobs.")
                if self._worker_proc and self._worker_proc.poll() is None:
                    print(
                        f"[AsyncEval] Worker (pid={self._worker_proc.pid}) "
                        f"will exit when done. Log: {self._worker_log_path()}"
                    )
        return control

    # ------------------------------------------------------------------
    # Eval dispatch
    # ------------------------------------------------------------------

    def _run_eval(self, model, step: int):
        if self.async_eval:
            self._run_eval_async(model, step)
        else:
            self._run_eval_sync(model, step)

    # ------------------------------------------------------------------
    # Sync eval (existing behavior)
    # ------------------------------------------------------------------

    def _run_eval_sync(self, model, step: int):
        """Run the full eval pass synchronously. Called on all ranks."""
        model.eval()
        rank, world_size = _get_dist_info()
        is_main = rank == 0

        is_fsdp = _is_fsdp_model(model)
        if is_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            ctx = FSDP.summon_full_params(model, writeback=False, rank0_only=False)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            if is_main:
                print(f"\n[PeriodicEval] Step {step} -- running evaluation...")

            step_results_by_dataset: dict[str, dict[str, dict]] = {}
            for ds_index, ds_label in enumerate(sorted(self.eval_datasets)):
                # --- Sample subsets ---
                diag_subset = self._sample(
                    self.diag_samples_by_dataset.get(ds_label, []),
                    self.max_diag_samples,
                    step,
                    stream_id=ds_index * 10,
                ) if self.enable_diagnosis_eval else []

                diag_output_file = None
                conv_output_file = None
                if self.eval_output_dir:
                    os.makedirs(self.eval_output_dir, exist_ok=True)
                    rank_suffix = f"_rank{rank}" if world_size > 1 else ""
                    safe_label = self._safe_dataset_label(ds_label)
                    if self.enable_diagnosis_eval and is_main:
                        diag_output_file = os.path.join(
                            self.eval_output_dir, f"step_{step}_{safe_label}_diag.jsonl"
                        )
                    if self.enable_conversation_eval and self.patient_client is not None:
                        conv_output_file = os.path.join(
                            self.eval_output_dir, f"step_{step}_{safe_label}_conv{rank_suffix}.jsonl"
                        )

                diag_results: dict = {}
                if diag_subset:
                    diag_results = run_diagnosis_eval(
                        model, self.tokenizer, diag_subset,
                        self.judge_model, self.judge_tokenizer,
                        max_new_tokens=self.max_new_tokens,
                        max_turns=self.max_conv_turns,
                        show_progress=is_main,
                        output_file=diag_output_file,
                        debug_generate_heartbeat=self.debug_generate_heartbeat,
                        debug_prefix=f"step={step} dataset={ds_label} rank={rank}",
                    )

                conv_results: dict = {}
                if self.enable_conversation_eval and self.patient_client is not None:
                    conv_subset = self._sample(
                        self.conv_samples_by_dataset.get(ds_label, []),
                        self.max_conv_samples,
                        step,
                        stream_id=ds_index * 10 + 2,
                    )
                    if conv_subset:
                        conv_results = run_conversation_eval(
                            model, self.tokenizer, conv_subset,
                            self.judge_model, self.judge_tokenizer,
                            patient_client=self.patient_client,
                            patient_model=self.patient_model,
                            max_new_tokens=self.max_new_tokens,
                            max_turns=self.max_conv_turns,
                            show_progress=is_main,
                            output_file=conv_output_file,
                            debug_generate_heartbeat=self.debug_generate_heartbeat,
                            debug_prefix=f"step={step} dataset={ds_label} rank={rank}",
                        )

                if is_main:
                    self._print_and_log(step, diag_results, conv_results, dataset_label=ds_label)
                    step_results_by_dataset[ds_label] = {
                        "diagnosis": diag_results,
                        "conversation": conv_results,
                    }

            if is_main:
                self._maybe_save_best_checkpoint_sync(model, step, step_results_by_dataset)

        model.train()

    # ------------------------------------------------------------------
    # Async eval (persistent worker)
    # ------------------------------------------------------------------

    def _run_eval_async(self, model, step: int):
        """Save checkpoint + enqueue async job for diagnosis/conversation eval."""
        model.eval()
        rank, world_size = _get_dist_info()
        is_main = rank == 0

        is_fsdp = _is_fsdp_model(model)
        if is_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            ctx = FSDP.summon_full_params(model, writeback=False, rank0_only=False)
        else:
            from contextlib import nullcontext
            ctx = nullcontext()

        with ctx:
            if is_main:
                print(f"\n[PeriodicEval] Step {step} (async mode)...")

            # --- Save checkpoint for async eval ---
            needs_async = self.enable_diagnosis_eval or self.enable_conversation_eval
            temp_dir = None
            if needs_async:
                base = self.eval_output_dir or "/tmp/medagent_async_eval"
                temp_dir = os.path.join(base, f"async_step_{step}")
                self._save_checkpoint(model, temp_dir, is_main)

        # --- Enqueue async job (rank 0, outside FSDP context) ---
        if is_main and temp_dir and needs_async:
            self._ensure_worker_running()
            self._enqueue_job(temp_dir, step)

        model.train()

    def _save_checkpoint(self, model, temp_dir: str, is_main: bool):
        """Save model checkpoint for async eval.

        Must be called inside summon_full_params context when FSDP is active.
        All ranks participate (collective), but only rank 0 writes to disk.
        """
        if not is_main:
            return
        os.makedirs(temp_dir, exist_ok=True)
        save_model = model
        while hasattr(save_model, "module"):
            save_model = save_model.module
        save_model.save_pretrained(temp_dir)
        self.tokenizer.save_pretrained(temp_dir)
        # Signal to worker that checkpoint is complete
        with open(os.path.join(temp_dir, ".ready"), "w") as f:
            f.write("ready\n")
        print(f"  [AsyncEval] Checkpoint saved to {temp_dir}")

    def _ensure_worker_running(self):
        """Spawn the persistent eval worker if it's not already running."""
        if self._worker_proc is not None and self._worker_proc.poll() is None:
            return  # still alive

        if self._worker_proc is not None:
            rc = self._worker_proc.returncode
            print(f"  [AsyncEval] Worker exited (code={rc}), respawning...")
            if self._worker_log_file:
                self._worker_log_file.close()
                self._worker_log_file = None

        cmd = [
            sys.executable, "-m", "evaluation.eval_worker",
            "--queue-dir", self._queue_dir,
        ]
        data_specs = self.eval_data_paths or {}
        for label, path in data_specs.items():
            cmd.extend(["--data", f"{label}={path}"])
        if self.async_weave_project:
            cmd.extend(["--weave-project", self.async_weave_project])

        env = os.environ.copy()
        if self.async_eval_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.async_eval_gpu)

        log_path = self._worker_log_path()
        self._worker_log_file = open(log_path, "a")

        self._worker_proc = subprocess.Popen(
            cmd, env=env, stdout=self._worker_log_file, stderr=self._worker_log_file,
        )
        gpu_info = f"gpu={self.async_eval_gpu}" if self.async_eval_gpu is not None else "gpu=auto"
        print(
            f"  [AsyncEval] Worker spawned "
            f"(pid={self._worker_proc.pid}, {gpu_info}, log={log_path})"
        )

    def _enqueue_job(self, checkpoint_dir: str, step: int):
        """Write a job file for the worker to pick up."""
        eval_types = []
        if self.enable_diagnosis_eval:
            eval_types.append("diagnosis")
        if self.enable_conversation_eval:
            eval_types.append("conversation")

        job = {
            "step": step,
            "checkpoint": checkpoint_dir,
            "eval_types": eval_types,
            "max_diag_samples": self.max_diag_samples,
            "max_conv_samples": self.max_conv_samples,
            "max_new_tokens": self.max_new_tokens,
            "max_turns": self.max_conv_turns,
            "judge_base_url": self.async_judge_base_url,
            "judge_model": self.async_judge_model_name,
            "eval_output_dir": self.eval_output_dir,
        }
        if self.async_patient_base_url:
            job["patient_base_url"] = self.async_patient_base_url
            job["patient_model"] = self.async_patient_model_name

        job_path = os.path.join(self._queue_dir, "jobs", f"step_{step:08d}.json")
        with open(job_path, "w") as f:
            json.dump(job, f, indent=2)

        pending = self._count_pending_jobs()
        print(f"  [AsyncEval] Job enqueued for step {step} ({pending} pending)")

    def _collect_async_results(self):
        """Scan for completed results and log them to wandb."""
        if not _is_main_process():
            return
        if not self._queue_dir:
            return
        results_dir = os.path.join(self._queue_dir, "results")
        if not os.path.isdir(results_dir):
            return

        for name in sorted(os.listdir(results_dir)):
            if not name.endswith(".json"):
                continue
            result_path = os.path.join(results_dir, name)

            # Use a .logged marker to avoid double-logging
            logged_marker = result_path + ".logged"
            if os.path.exists(logged_marker):
                continue

            try:
                with open(result_path) as f:
                    results = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            if "error" in results:
                print(f"  [AsyncEval] Step {results.get('step')}: error={results['error']}")
                with open(logged_marker, "w") as f:
                    f.write("logged\n")
                continue

            step = results.get("step", 0)
            elapsed = results.get("elapsed_seconds", 0)
            checkpoint = results.get("checkpoint")
            print(f"\n  [AsyncEval] Results for step {step} ({elapsed:.0f}s):")
            datasets = results.get("datasets", {})
            combined_datasets: dict[str, dict[str, dict]] = {}
            if datasets:
                for ds_label, ds_results in datasets.items():
                    combined = {
                        "diagnosis": ds_results.get("diagnosis", {}),
                        "conversation": ds_results.get("conversation", {}),
                    }
                    combined_datasets[ds_label] = combined
                    print(f"    [{ds_label}]")
                    diag = combined["diagnosis"]
                    conv = combined["conversation"]
                    if diag:
                        print(
                            f"      [Diagnosis]  reward={diag['diagnosis_reward']:.3f}  "
                            f"found_rate={diag['diagnosis_found_rate']:.1%}"
                        )
                    if conv:
                        print(
                            f"      [Conversation]  diag_reward={conv['diagnosis_reward']:.3f}  "
                            f"found_rate={conv['diagnosis_found_rate']:.1%}  "
                            f"tool_reward={conv.get('tool_call_reward', 0):.3f}"
                        )
            else:
                diag = results.get("diagnosis", {})
                conv = results.get("conversation", {})
                combined_datasets["eval"] = {
                    "diagnosis": diag,
                    "conversation": conv,
                }
                if diag:
                    print(
                        f"    [Diagnosis]  reward={diag['diagnosis_reward']:.3f}  "
                        f"found_rate={diag['diagnosis_found_rate']:.1%}"
                    )
                if conv:
                    print(
                        f"    [Conversation]  diag_reward={conv['diagnosis_reward']:.3f}  "
                        f"found_rate={conv['diagnosis_found_rate']:.1%}  "
                        f"tool_reward={conv.get('tool_call_reward', 0):.3f}"
                    )

            try:
                import wandb
                if wandb.run is not None:
                    log_dict: dict = {"eval_async/step": step}
                    if combined_datasets:
                        for ds_label, ds_results in combined_datasets.items():
                            safe_label = self._safe_dataset_label(ds_label)
                            diag = ds_results.get("diagnosis", {})
                            conv = ds_results.get("conversation", {})
                            if diag:
                                log_dict[f"eval_async/{safe_label}/diagnosis_reward"] = diag["diagnosis_reward"]
                                log_dict[f"eval_async/{safe_label}/diagnosis_found_rate"] = diag["diagnosis_found_rate"]
                            if conv:
                                log_dict[f"eval_async/{safe_label}/conv_diagnosis_reward"] = conv["diagnosis_reward"]
                                log_dict[f"eval_async/{safe_label}/conv_diagnosis_found_rate"] = conv["diagnosis_found_rate"]
                                log_dict[f"eval_async/{safe_label}/conv_tool_call_reward"] = conv.get("tool_call_reward", 0)
                                log_dict[f"eval_async/{safe_label}/conv_tool_hallucination_rate"] = conv.get("tool_hallucination_rate", 0)
                    wandb.log(log_dict)
            except ImportError:
                pass

            self._maybe_promote_async_checkpoint(step, checkpoint, combined_datasets)
            if checkpoint and os.path.isdir(checkpoint):
                shutil.rmtree(checkpoint, ignore_errors=True)

            with open(logged_marker, "w") as f:
                f.write("logged\n")

    def _count_pending_jobs(self) -> int:
        """Count jobs without corresponding results."""
        jobs_dir = os.path.join(self._queue_dir, "jobs")
        results_dir = os.path.join(self._queue_dir, "results")
        count = 0
        if os.path.isdir(jobs_dir):
            for name in os.listdir(jobs_dir):
                if name.endswith(".json") and not os.path.exists(
                    os.path.join(results_dir, name)
                ):
                    count += 1
        return count

    def _worker_log_path(self) -> str:
        return os.path.join(self._queue_dir, "worker.log")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _print_and_log(
        self, step: int, diag_results: dict, conv_results: dict,
        dataset_label: str | None = None,
    ):
        """Print results and log to wandb (rank 0 only)."""
        prefix = f"[{dataset_label}] " if dataset_label else ""
        if diag_results:
            print(
                f"  {prefix}[Diagnosis]  reward={diag_results['diagnosis_reward']:.3f}  "
                f"found_rate={diag_results['diagnosis_found_rate']:.1%}"
            )
        if conv_results:
            print(
                f"  {prefix}[Conversation]  diag_reward={conv_results['diagnosis_reward']:.3f}  "
                f"found_rate={conv_results['diagnosis_found_rate']:.1%}  "
                f"tool_reward={conv_results['tool_call_reward']:.3f}  "
                f"halluc_rate={conv_results['tool_hallucination_rate']:.3f}"
            )

        try:
            import wandb
            if wandb.run is not None:
                log_dict: dict = {"eval/step": step}
                base = f"eval/{self._safe_dataset_label(dataset_label)}" if dataset_label else "eval"
                if diag_results:
                    log_dict[f"{base}/diagnosis_reward"] = diag_results["diagnosis_reward"]
                    log_dict[f"{base}/diagnosis_found_rate"] = diag_results["diagnosis_found_rate"]
                if conv_results:
                    log_dict[f"{base}/conv_diagnosis_reward"] = conv_results["diagnosis_reward"]
                    log_dict[f"{base}/conv_diagnosis_found_rate"] = conv_results["diagnosis_found_rate"]
                    log_dict[f"{base}/conv_tool_call_reward"] = conv_results["tool_call_reward"]
                    log_dict[f"{base}/conv_tool_hallucination_rate"] = conv_results["tool_hallucination_rate"]
                wandb.log(log_dict)
        except ImportError:
            pass

    def _sample(self, samples: list[dict], max_n: int, step: int, stream_id: int) -> list[dict]:
        if not samples:
            return []
        if self.random_sample and len(samples) > max_n:
            rng = random.Random(step * 10007 + stream_id)
            indices = rng.sample(range(len(samples)), max_n)
            return [samples[i] for i in indices]
        return samples[:max_n]

    @staticmethod
    def _safe_dataset_label(label: str | None) -> str:
        if not label:
            return "eval"
        return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in label)

    def _load_best_checkpoints(self):
        if self.best_checkpoint_top_k <= 0 or not self.checkpoint_output_dir:
            return
        state_path = self._best_checkpoint_state_path()
        if not os.path.exists(state_path):
            return
        try:
            with open(state_path) as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            return
        loaded = []
        for entry in raw:
            path = entry.get("path")
            score = entry.get("score")
            step = entry.get("step")
            if path and os.path.isdir(path) and score is not None and step is not None:
                loaded.append((float(score), int(step), path))
        self._best_ckpts = sorted(loaded, key=lambda x: x[0], reverse=True)

    def _best_checkpoint_state_path(self) -> str:
        return os.path.join(self.checkpoint_output_dir, "best_checkpoints.json")

    def _compute_selection_score(self, datasets: dict[str, dict[str, dict]]) -> tuple[float | None, dict[str, float]]:
        metric_values: dict[str, list[float]] = {name: [] for name in self.best_checkpoint_metrics}

        for ds_results in datasets.values():
            diag = ds_results.get("diagnosis", {})
            conv = ds_results.get("conversation", {})

            if diag and "diagnosis_reward" in metric_values:
                metric_values["diagnosis_reward"].append(float(diag["diagnosis_reward"]))
            if conv and "conv_diagnosis_reward" in metric_values:
                metric_values["conv_diagnosis_reward"].append(float(conv["diagnosis_reward"]))
            if conv and "conv_tool_call_reward" in metric_values:
                metric_values["conv_tool_call_reward"].append(float(conv["tool_call_reward"]))

        averaged = {
            name: sum(values) / len(values)
            for name, values in metric_values.items()
            if values
        }
        if not averaged:
            return None, {}
        return sum(averaged.values()) / len(averaged), averaged

    def _should_keep_checkpoint(self, step: int, score: float) -> bool:
        if self.best_checkpoint_top_k <= 0 or not self.checkpoint_output_dir:
            return False
        if any(existing_step == step for _, existing_step, _ in self._best_ckpts):
            return False
        return (
            len(self._best_ckpts) < self.best_checkpoint_top_k
            or score > self._best_ckpts[-1][0]
        )

    def _record_best_checkpoint(self, score: float, step: int, path: str, metric_summary: dict[str, float]) -> None:
        self._best_ckpts.append((score, step, path))
        self._best_ckpts.sort(key=lambda x: x[0], reverse=True)

        while len(self._best_ckpts) > self.best_checkpoint_top_k:
            _, _, worst_path = self._best_ckpts.pop()
            if os.path.isdir(worst_path):
                shutil.rmtree(worst_path, ignore_errors=True)

        os.makedirs(self.checkpoint_output_dir, exist_ok=True)
        with open(self._best_checkpoint_state_path(), "w") as f:
            json.dump(
                [{"score": s, "step": st, "path": p} for s, st, p in self._best_ckpts],
                f,
                indent=2,
            )

        print(
            f"[BestCheckpoint] step={step} score={score:.4f} metrics={metric_summary} "
            f"top_k={[(round(s, 4), st) for s, st, _ in self._best_ckpts]}"
        )

    def _maybe_save_best_checkpoint_sync(self, model, step: int, datasets: dict[str, dict[str, dict]]) -> None:
        score, metric_summary = self._compute_selection_score(datasets)
        if score is None or not self._should_keep_checkpoint(step, score):
            return

        ckpt_path = os.path.join(self.checkpoint_output_dir, f"best_eval_step_{step}")
        if not os.path.isdir(ckpt_path):
            self._save_checkpoint(model, ckpt_path, is_main=True)
        self._record_best_checkpoint(score, step, ckpt_path, metric_summary)

    def _maybe_promote_async_checkpoint(
        self,
        step: int,
        checkpoint: str | None,
        datasets: dict[str, dict[str, dict]],
    ) -> None:
        score, metric_summary = self._compute_selection_score(datasets)
        if (
            score is None
            or not checkpoint
            or not os.path.isdir(checkpoint)
            or not self._should_keep_checkpoint(step, score)
        ):
            return

        dst = os.path.join(self.checkpoint_output_dir, f"best_eval_step_{step}")
        if os.path.abspath(checkpoint) != os.path.abspath(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst, ignore_errors=True)
            tmp_dst = dst + ".tmp"
            if os.path.isdir(tmp_dst):
                shutil.rmtree(tmp_dst, ignore_errors=True)
            shutil.copytree(checkpoint, tmp_dst)
            os.replace(tmp_dst, dst)
            shutil.rmtree(checkpoint, ignore_errors=True)
        self._record_best_checkpoint(score, step, dst, metric_summary)


class EvaluationCallback(TrainerCallback):
    """Callback to run custom evaluation during training (legacy, multi-turn)."""

    def __init__(
        self,
        eval_data: list[dict],
        judge_model,
        judge_tokenizer,
        tokenizer,
        eval_steps: int = 500,
        max_eval_samples: int = 50,
        max_new_tokens: int = 1024,
        random_sample: bool = True,
    ):
        self.eval_data = eval_data
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.max_eval_samples = max_eval_samples
        self.max_new_tokens = max_new_tokens
        self.random_sample = random_sample
        self.last_eval_step = 0

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Run evaluation every eval_steps."""
        current_step = state.global_step

        if current_step - self.last_eval_step >= self.eval_steps and current_step > 0:
            if _is_main_process():
                print(f"\n[Evaluation] Running evaluation at step {current_step}...")

                if self.random_sample and len(self.eval_data) > self.max_eval_samples:
                    eval_subset = random.sample(self.eval_data, self.max_eval_samples)
                else:
                    eval_subset = self.eval_data[:self.max_eval_samples]

                model.eval()
                results = run_evaluation(
                    model=model,
                    tokenizer=self.tokenizer,
                    eval_data=eval_subset,
                    judge_model=self.judge_model,
                    judge_tokenizer=self.judge_tokenizer,
                    max_new_tokens=self.max_new_tokens,
                    show_progress=True,
                )
                model.train()

                print(f"[Evaluation] Step {current_step}:")
                print(f"  Tool Call Reward: {results['tool_call_reward']:.3f}")
                print(f"  Diagnosis Reward: {results['diagnosis_reward']:.3f}")

                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            "eval/tool_call_reward": results["tool_call_reward"],
                            "eval/diagnosis_reward": results["diagnosis_reward"],
                            "eval/step": current_step,
                        })
                except ImportError:
                    pass

            self.last_eval_step = current_step

        return control
