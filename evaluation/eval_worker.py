"""Persistent evaluation worker subprocess.

Watches a queue directory for eval job files, processes them sequentially,
and writes results.  Continues running after training ends until all
queued jobs are processed.

Queue directory structure::

    <queue_dir>/
        jobs/step_<N>.json      -- job definitions (written by callback)
        results/step_<N>.json   -- eval results (written by worker)
        STOP                    -- sentinel: exit after all pending jobs

Job file format::

    {
        "step": 100,
        "checkpoint": "/path/to/async_step_100",
        "eval_types": ["diagnosis", "conversation"],
        "max_diag_samples": 50,
        "max_conv_samples": 10,
        "max_new_tokens": 512,
        "max_turns": 15,
        "judge_base_url": "https://api.openai.com/v1",
        "judge_model": "gpt-4.1-mini",
        "patient_base_url": "...",
        "patient_model": "gpt-4.1-mini",
        "eval_output_dir": "..."
    }

Usage (typically spawned by the callback, not manually)::

    CUDA_VISIBLE_DEVICES=4 python -m evaluation.eval_worker \\
        --queue-dir logs/eval_queue \\
        --data ddxplus=data/ddxplus_eval.jsonl \\
        --data pmc=data/pmc_eval.jsonl \\
        --poll-interval 10
"""

import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def find_pending_jobs(queue_dir: str) -> list[str]:
    """Return job file paths without corresponding results, sorted by name."""
    jobs_dir = os.path.join(queue_dir, "jobs")
    results_dir = os.path.join(queue_dir, "results")
    if not os.path.isdir(jobs_dir):
        return []
    pending = []
    for name in sorted(os.listdir(jobs_dir)):
        if not name.endswith(".json"):
            continue
        if not os.path.exists(os.path.join(results_dir, name)):
            pending.append(os.path.join(jobs_dir, name))
    return pending


def load_model_weights(model, checkpoint: str):
    """Reload model weights from a checkpoint directory (in-place)."""
    import torch

    safetensors_files = sorted(glob.glob(os.path.join(checkpoint, "*.safetensors")))
    if safetensors_files:
        from safetensors.torch import load_file
        state_dict = {}
        for sf in safetensors_files:
            state_dict.update(load_file(sf, device="cpu"))
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        torch.cuda.empty_cache()
        return

    bin_files = sorted(glob.glob(os.path.join(checkpoint, "*.bin")))
    if bin_files:
        state_dict = {}
        for bf in bin_files:
            state_dict.update(torch.load(bf, map_location="cpu", weights_only=True))
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        torch.cuda.empty_cache()
        return

    raise FileNotFoundError(f"No model weight files found in {checkpoint}")


def wait_for_ready(checkpoint: str, timeout: int = 300) -> bool:
    """Wait for the ``.ready`` marker written by the callback after saving."""
    ready_file = os.path.join(checkpoint, ".ready")
    start = time.time()
    while not os.path.exists(ready_file):
        if time.time() - start > timeout:
            return False
        time.sleep(2)
    return True


# ------------------------------------------------------------------
# Job processing
# ------------------------------------------------------------------

def process_job(
    job: dict,
    model,
    tokenizer,
    diag_pools: dict[str, list[dict]],
    conv_pools: dict[str, list[dict]],
) -> dict:
    """Run a single eval job.  Returns results dict."""
    step = job["step"]
    start_time = time.time()
    results: dict = {"step": step, "checkpoint": job.get("checkpoint")}

    from openai import OpenAI
    judge_client = OpenAI(base_url=job["judge_base_url"], timeout=1800.0, max_retries=2)
    judge_model_name = job["judge_model"]

    results["datasets"] = {}
    patient_client = None
    if "conversation" in job.get("eval_types", []) and job.get("patient_base_url"):
        patient_client = OpenAI(base_url=job["patient_base_url"], timeout=1800.0, max_retries=2)

    for ds_index, ds_label in enumerate(sorted(diag_pools)):
        ds_results: dict = {}

        if "diagnosis" in job.get("eval_types", []):
            from evaluation.runners import run_diagnosis_eval

            diag_pool = diag_pools.get(ds_label, [])
            max_n = job.get("max_diag_samples", 50)
            if max_n and len(diag_pool) > max_n:
                diag_samples = random.Random(step * 10007 + ds_index * 10).sample(diag_pool, max_n)
            else:
                diag_samples = diag_pool

            diag_output = None
            if job.get("eval_output_dir"):
                os.makedirs(job["eval_output_dir"], exist_ok=True)
                diag_output = os.path.join(
                    job["eval_output_dir"], f"step_{step}_{ds_label}_diag_async.jsonl",
                )

            if diag_samples:
                print(f"[EvalWorker] {ds_label} diagnosis eval ({len(diag_samples)} samples)...")
                dr = run_diagnosis_eval(
                    model, tokenizer, diag_samples,
                    judge_client, judge_model_name,
                    max_new_tokens=job.get("max_new_tokens", 512),
                    max_turns=job.get("max_turns", 15),
                    show_progress=True,
                    output_file=diag_output,
                )
                ds_results["diagnosis"] = dr
                print(
                    f"[EvalWorker] {ds_label} diagnosis: reward={dr['diagnosis_reward']:.3f}  "
                    f"found_rate={dr['diagnosis_found_rate']:.1%}"
                )

        if "conversation" in job.get("eval_types", []) and patient_client is not None:
            from evaluation.runners import run_conversation_eval

            conv_pool = conv_pools.get(ds_label, [])
            max_n = job.get("max_conv_samples", 10)
            if max_n and len(conv_pool) > max_n:
                conv_samples = random.Random(step * 10007 + ds_index * 10 + 2).sample(conv_pool, max_n)
            else:
                conv_samples = conv_pool

            conv_output = None
            if job.get("eval_output_dir"):
                conv_output = os.path.join(
                    job["eval_output_dir"], f"step_{step}_{ds_label}_conv_async.jsonl",
                )

            if conv_samples:
                print(f"[EvalWorker] {ds_label} conversation eval ({len(conv_samples)} samples)...")
                cr = run_conversation_eval(
                    model, tokenizer, conv_samples,
                    judge_client, judge_model_name,
                    patient_client=patient_client,
                    patient_model=job.get("patient_model", "gpt-4.1-mini"),
                    max_new_tokens=job.get("max_new_tokens", 512),
                    max_turns=job.get("max_turns", 15),
                    show_progress=True,
                    output_file=conv_output,
                )
                ds_results["conversation"] = cr
                print(
                    f"[EvalWorker] {ds_label} conversation: "
                    f"diag_reward={cr['diagnosis_reward']:.3f}"
                )

        results["datasets"][ds_label] = ds_results

    results["elapsed_seconds"] = time.time() - start_time
    return results


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Persistent eval worker")
    parser.add_argument("--queue-dir", required=True)
    parser.add_argument(
        "--data",
        required=True,
        nargs="+",
        help="Eval data JSONL path(s). Format: path or name=path.",
    )
    parser.add_argument("--poll-interval", type=int, default=10,
                        help="Seconds between queue polls when idle")
    parser.add_argument("--weave-project", default=None)
    args = parser.parse_args()

    print(f"[EvalWorker] Starting persistent eval worker")
    print(f"[EvalWorker] Queue: {args.queue_dir}")
    print(f"[EvalWorker] Data: {args.data}")
    print(
        f"[EvalWorker] CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}"
    )

    if args.weave_project:
        try:
            import weave
            weave.init(args.weave_project)
        except ImportError:
            pass

    # Create queue structure
    os.makedirs(os.path.join(args.queue_dir, "jobs"), exist_ok=True)
    os.makedirs(os.path.join(args.queue_dir, "results"), exist_ok=True)

    data_sources: dict[str, str] = {}
    for spec in args.data:
        if "=" in spec:
            label, path = spec.split("=", 1)
        else:
            path = spec
            label = Path(path).stem
        data_sources[label] = path

    dataset_records: dict[str, list[dict]] = {}
    for label, path in data_sources.items():
        records: list[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        dataset_records[label] = records
        print(f"[EvalWorker] Loaded {len(records)} records for {label} from {path}")

    from evaluation.data import (
        prepare_diagnosis_eval_samples,
        prepare_conversation_eval_samples,
    )
    diag_pools = {
        label: prepare_diagnosis_eval_samples(records)
        for label, records in dataset_records.items()
    }
    conv_pools = {
        label: prepare_conversation_eval_samples(records)
        for label, records in dataset_records.items()
    }
    for label in sorted(dataset_records):
        print(
            f"[EvalWorker] Sample pools for {label}: "
            f"{len(diag_pools[label])} diagnosis, {len(conv_pools[label])} conversation"
        )

    model = None
    tokenizer = None
    stop_file = os.path.join(args.queue_dir, "STOP")

    while True:
        pending = find_pending_jobs(args.queue_dir)

        if pending:
            job_path = pending[0]  # oldest first
            with open(job_path) as f:
                job = json.load(f)

            checkpoint = job["checkpoint"]
            step = job["step"]

            # Wait for checkpoint to be fully written
            if not wait_for_ready(checkpoint):
                print(
                    f"[EvalWorker] Timeout waiting for checkpoint at step {step}, "
                    f"skipping"
                )
                results_path = os.path.join(
                    args.queue_dir, "results", os.path.basename(job_path),
                )
                with open(results_path, "w") as f:
                    json.dump({"step": step, "error": "checkpoint_timeout"}, f)
                continue

            print(f"\n[EvalWorker] ======== Step {step} ========")

            if model is None:
                # First job: full model load
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(
                    checkpoint, trust_remote_code=True,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                model = AutoModelForCausalLM.from_pretrained(
                    checkpoint,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                model.eval()
                print(f"[EvalWorker] Model loaded on {next(model.parameters()).device}")
            else:
                # Subsequent jobs: reload weights in-place
                load_model_weights(model, checkpoint)
                model.eval()
                print(f"[EvalWorker] Weights reloaded for step {step}")

            results = process_job(job, model, tokenizer, diag_pools, conv_pools)

            # Write results
            results_path = os.path.join(
                args.queue_dir, "results", os.path.basename(job_path),
            )
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[EvalWorker] Results -> {results_path}")

        elif os.path.exists(stop_file):
            print("[EvalWorker] STOP sentinel found, no pending jobs. Exiting.")
            break
        else:
            time.sleep(args.poll_interval)

    print("[EvalWorker] Worker finished.")


if __name__ == "__main__":
    main()
