"""Async evaluation subprocess for diagnosis and conversation eval.

Launched by PeriodicEvaluationCallback during training.  Loads a saved
checkpoint on a dedicated GPU, runs evaluation, writes results to a JSON
file, and optionally traces with weave.

The parent training process spawns this with CUDA_VISIBLE_DEVICES set to
a single GPU so it doesn't interfere with FSDP training.

Usage (typically invoked by the callback, not manually):
    CUDA_VISIBLE_DEVICES=7 python -m evaluation.async_eval \
        --checkpoint /tmp/medagent_async_eval/step_100 \
        --data data/conversations/eval.jsonl \
        --output /tmp/medagent_async_eval/step_100/results.json \
        --step 100 \
        --eval-type diagnosis \
        --judge-base-url https://api.openai.com/v1 \
        --judge-model gpt-4.1-mini
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Async evaluation subprocess")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path (default: checkpoint)")
    parser.add_argument("--data", required=True, help="Path to eval data JSONL")
    parser.add_argument("--output", required=True, help="Path to write results JSON")
    parser.add_argument("--step", type=int, required=True, help="Training step")
    parser.add_argument("--eval-type", nargs="+", default=["diagnosis"],
                        choices=["diagnosis", "conversation"],
                        help="Eval types to run")
    parser.add_argument("--max-diag-samples", type=int, default=50)
    parser.add_argument("--max-conv-samples", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Added to step for sampling seed")
    # Judge
    parser.add_argument("--judge-base-url", required=True)
    parser.add_argument("--judge-model", default="gpt-4.1-mini")
    # Patient (conversation mode)
    parser.add_argument("--patient-base-url", default=None)
    parser.add_argument("--patient-model", default="gpt-4.1-mini")
    # Output
    parser.add_argument("--eval-output-dir", default=None,
                        help="Directory for per-sample JSONL output")
    # Weave
    parser.add_argument("--weave-project", default=None)

    args = parser.parse_args()

    start_time = time.time()
    print(f"[AsyncEval] Starting async eval for step {args.step}")
    print(f"[AsyncEval] Checkpoint: {args.checkpoint}")
    print(f"[AsyncEval] Eval types: {args.eval_type}")
    print(f"[AsyncEval] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # Init weave
    if args.weave_project:
        try:
            import weave
            weave.init(args.weave_project)
        except ImportError:
            pass

    # Load model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = args.tokenizer or args.checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[AsyncEval] Model loaded on {next(model.parameters()).device}")

    # Judge
    from openai import OpenAI
    judge_model = OpenAI(base_url=args.judge_base_url, timeout=1800.0, max_retries=2)
    judge_tokenizer = args.judge_model

    # Load records
    records = []
    with open(args.data) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[AsyncEval] Loaded {len(records)} records from {args.data}")

    results = {"step": args.step}

    # --- Diagnosis eval ---
    if "diagnosis" in args.eval_type:
        from evaluation.data import prepare_diagnosis_eval_samples
        from evaluation.runners import run_diagnosis_eval

        diag_samples = prepare_diagnosis_eval_samples(records)
        if args.max_diag_samples and len(diag_samples) > args.max_diag_samples:
            rng = random.Random(args.step * 10007 + args.seed_offset)
            diag_samples = rng.sample(diag_samples, args.max_diag_samples)

        diag_output = None
        if args.eval_output_dir:
            os.makedirs(args.eval_output_dir, exist_ok=True)
            diag_output = os.path.join(
                args.eval_output_dir, f"step_{args.step}_diag_async.jsonl"
            )

        if diag_samples:
            print(f"[AsyncEval] Running diagnosis eval ({len(diag_samples)} samples)...")
            diag_results = run_diagnosis_eval(
                model, tokenizer, diag_samples,
                judge_model, judge_tokenizer,
                max_new_tokens=args.max_new_tokens,
                max_turns=args.max_turns,
                show_progress=True,
                output_file=diag_output,
            )
            results["diagnosis"] = diag_results
            print(
                f"[AsyncEval] Diagnosis: reward={diag_results['diagnosis_reward']:.3f} "
                f"found_rate={diag_results['diagnosis_found_rate']:.1%}"
            )
        else:
            print("[AsyncEval] No diagnosis samples found")

    # --- Conversation eval ---
    if "conversation" in args.eval_type and args.patient_base_url:
        from evaluation.data import prepare_conversation_eval_samples
        from evaluation.runners import run_conversation_eval

        conv_samples = prepare_conversation_eval_samples(records)
        if args.max_conv_samples and len(conv_samples) > args.max_conv_samples:
            rng = random.Random(args.step * 10007 + 2 + args.seed_offset)
            conv_samples = rng.sample(conv_samples, args.max_conv_samples)

        conv_output = None
        if args.eval_output_dir:
            os.makedirs(args.eval_output_dir, exist_ok=True)
            conv_output = os.path.join(
                args.eval_output_dir, f"step_{args.step}_conv_async.jsonl"
            )

        if conv_samples:
            patient_client = OpenAI(base_url=args.patient_base_url, timeout=1800.0, max_retries=2)
            print(f"[AsyncEval] Running conversation eval ({len(conv_samples)} samples)...")
            conv_results = run_conversation_eval(
                model, tokenizer, conv_samples,
                judge_model, judge_tokenizer,
                patient_client=patient_client,
                patient_model=args.patient_model,
                max_new_tokens=args.max_new_tokens,
                max_turns=args.max_turns,
                show_progress=True,
                output_file=conv_output,
            )
            results["conversation"] = conv_results
            print(
                f"[AsyncEval] Conversation: "
                f"diag_reward={conv_results['diagnosis_reward']:.3f} "
                f"found_rate={conv_results['diagnosis_found_rate']:.1%}"
            )
        else:
            print("[AsyncEval] No conversation samples found")

    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed

    # Write results
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[AsyncEval] Done in {elapsed:.1f}s. Results: {args.output}")


if __name__ == "__main__":
    main()
