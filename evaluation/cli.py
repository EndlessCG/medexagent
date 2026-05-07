"""CLI entry point for standalone evaluation.

Default usage (only --model required):
    python -m evaluation --model /path/to/checkpoint

This loads the two default test sets (ddxplus + pmc), runs all eval types
(diagnosis, conversation) on each, reports per-dataset per-eval-type
results, and logs everything to wandb under the "eval" tag.

All defaults can be overridden via CLI flags.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch

from .data import (
    prepare_conversation_eval_samples,
)
from .weave_eval import _APIBackend, _LocalBackend, run_weave_eval


# ── Default test sets ──────────────────────────────────────────────────────
DEFAULT_TEST_SETS = {
    "ddxplus": "data/conversations/processed/ddxplus_test_conversations_clean.jsonl",
    "pmc": "data/conversations/processed/pmc_test_conversations_clean.jsonl",
}

# ── Default eval parameters ───────────────────────────────────────────────
DEFAULT_MAX_DIAG_SAMPLES = None
DEFAULT_MAX_CONV_SAMPLES = None
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_MAX_TURNS = 15
DEFAULT_JUDGE_MODEL = "gpt-4.1-mini"
DEFAULT_JUDGE_BASE_URL = "https://api.openai.com/v1"
DEFAULT_PATIENT_MODEL = "gpt-4.1-mini"
DEFAULT_PATIENT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_WANDB_PROJECT = "medagent-eval-gold"


def _load_records(path: str) -> list[dict]:
    """Load raw conversation records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _sample(items: list, max_n: int | None, seed: int = 42) -> list:
    if max_n is None or len(items) <= max_n:
        return items
    rng = random.Random(seed)
    return rng.sample(items, max_n)


def _model_label(model_path: str) -> str:
    """Derive a short label from the model path for wandb."""
    return os.path.basename(os.path.normpath(model_path))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate medical diagnosis model on a local checkpoint or API model.",
    )
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str,
                             help="Path to local model checkpoint or HuggingFace model name")
    model_group.add_argument("--api-model", type=str,
                             help="OpenAI-compatible API model name")
    parser.add_argument("--api-base-url", type=str, default="https://api.openai.com/v1",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer (local model only; defaults to --model)")
    parser.add_argument("--data", type=str, nargs="+", default=None,
                        help="Override default test sets: path(s) to eval JSONL/CSV files. "
                             "Format: path or name=path (e.g. mytest=data/test.jsonl)")

    # Eval type toggles
    parser.add_argument("--eval-types", type=str, nargs="+",
                        default=["all"],
                        choices=["diagnosis", "conversation", "all"],
                        help="Eval types to run (default: all)")

    # Sample limits
    parser.add_argument("--max-diag-samples", type=int, default=DEFAULT_MAX_DIAG_SAMPLES)
    parser.add_argument("--max-conv-samples", type=int, default=DEFAULT_MAX_CONV_SAMPLES)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)

    # Judge
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--judge-base-url", type=str, default=DEFAULT_JUDGE_BASE_URL)

    # Patient LLM
    parser.add_argument("--patient-model", type=str, default=DEFAULT_PATIENT_MODEL)
    parser.add_argument("--patient-base-url", type=str, default=DEFAULT_PATIENT_BASE_URL)
    parser.add_argument("--persona", action=argparse.BooleanOptionalAction, default=False,
                        help="Inject sampled persona into the patient system prompt. "
                             "Default: off. Use --persona to enable, --no-persona to force-disable.")
    parser.add_argument("--patient-style", type=str, default="default",
                        choices=["default", "agentclinic"],
                        help="Patient prompt style. 'agentclinic' mirrors AgentClinic's "
                             "PatientAgent prompt (no bias) for paper-comparable eval.")

    # Wandb
    parser.add_argument("--wandb-project", type=str, default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")

    # Weave
    parser.add_argument("--weave-project", type=str, default="medagent-eval-gold",
                        help="Weave project name")
    parser.add_argument("--weave-model-name", type=str, default=None,
                        help="Override model name shown in Weave UI (defaults to --api-model or checkpoint basename)")
    parser.add_argument("--weave-dataset-ref", type=str, default=None,
                        help="Reuse existing Weave dataset object(s) by base ref/name")

    # Concurrency / retries
    parser.add_argument("--max-concurrency", type=int, default=4,
                        help="Max parallel in-flight eval trials (sets WEAVE_PARALLELISM). "
                             "Lower this if you see 429s from rate-limited APIs.")
    parser.add_argument("--api-max-retries", type=int, default=6,
                        help="Max retries for doctor/judge/patient API clients on 429/5xx. "
                             "OpenAI SDK handles exponential backoff + Retry-After.")
    parser.add_argument("--disable-thinking", action="store_true",
                        help="For Qwen3-family API models: pass "
                             "extra_body={'chat_template_kwargs': {'enable_thinking': False}} "
                             "to disable the default thinking mode.")

    parser.add_argument("--diagagent", action="store_true",
                        help="Wrap the API backend with the DiagAgent adapter "
                             "(Henrychur/DiagAgent-{7B,8B,14B}, served externally "
                             "via vLLM). Uses DiagAgent's verbatim system prompt + "
                             "EHR input format + temperature=0.0. Requires "
                             "--api-model and --api-base-url. The judge client "
                             "doubles as a free-text exam -> tool name matcher.")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for results")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    os.environ["WEAVE_PARALLELISM"] = str(args.max_concurrency)
    print(f"Weave parallelism: {args.max_concurrency}, API max_retries: {args.api_max_retries}")

    if args.diagagent and not args.api_model:
        parser.error("--diagagent requires --api-model + --api-base-url "
                     "(serve Henrychur/DiagAgent-* via vLLM and point at it).")

    if "all" in args.eval_types:
        args.eval_types = ["diagnosis", "conversation"]

    # ── Resolve data sources ───────────────────────────────────────────
    # Each entry is (label, path)
    data_sources: list[tuple[str, str]] = []
    if args.data:
        for spec in args.data:
            if "=" in spec:
                label, path = spec.split("=", 1)
            else:
                label = Path(spec).stem
                # Shorten common prefixes
                for prefix in ("ddxplus", "pmc"):
                    if prefix in label.lower():
                        label = prefix
                        break
            data_sources.append((label, path if "=" in spec else spec))
    else:
        for label, path in DEFAULT_TEST_SETS.items():
            if os.path.exists(path):
                data_sources.append((label, path))
            else:
                print(f"Warning: default test set not found: {path}")

    if not data_sources:
        parser.error("No test data found. Provide --data or ensure default test sets exist.")
    if args.weave_dataset_ref and len(data_sources) != 1:
        parser.error("--weave-dataset-ref requires exactly one dataset source")

    # ── Init wandb ─────────────────────────────────────────────────────
    wandb_run = None
    if not args.no_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=f"eval-{_model_label(args.model) if args.model else args.api_model}",
                tags=["eval"],
                config={
                    "model": args.model or args.api_model,
                    "api_base_url": args.api_base_url if args.api_model else None,
                    "eval_types": args.eval_types,
                    "datasets": [label for label, _ in data_sources],
                    "max_diag_samples": args.max_diag_samples,
                    "max_conv_samples": args.max_conv_samples,
                    "max_new_tokens": args.max_new_tokens,
                    "max_turns": args.max_turns,
                    "judge_model": args.judge_model,
                    "patient_model": args.patient_model,
                },
            )
        except ImportError:
            print("wandb not installed, skipping logging.")

    # ── Init weave ─────────────────────────────────────────────────────
    try:
        import weave
        weave.init(args.weave_project)
    except ImportError as exc:
        raise ImportError(
            "weave is required for `python -m evaluation`. "
            "Install it or use a different entrypoint."
        ) from exc

    # ── Load generation backend ────────────────────────────────────────
    if args.model:
        backend = _LocalBackend(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=0.1,
        )
    else:
        extra_body = None
        if args.disable_thinking:
            extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
        backend = _APIBackend(
            api_base_url=args.api_base_url,
            model_name=args.api_model,
            max_tokens=args.max_new_tokens,
            temperature=0.1,
            max_retries=args.api_max_retries,
            extra_body=extra_body,
        )
    backend.load()

    # ── Init judge ─────────────────────────────────────────────────────
    needs_judge = "diagnosis" in args.eval_types or "conversation" in args.eval_types
    if needs_judge:
        from openai import OpenAI
        judge_model = OpenAI(base_url=args.judge_base_url, timeout=1800.0, max_retries=args.api_max_retries)
        judge_tokenizer = args.judge_model
        print(f"Judge: {args.judge_model} at {args.judge_base_url}")
    else:
        judge_model = None
        judge_tokenizer = None

    # ── DiagAgent adapter ──────────────────────────────────────────────
    if args.diagagent:
        from .diagagent_backend import DiagAgentBackend
        # Reuse the judge client as the free-text -> tool name matcher; if
        # eval-types skipped diagnosis/conversation we still need a client
        # for the matcher.
        if judge_model is None:
            from openai import OpenAI as _OpenAI
            matcher_client = _OpenAI(base_url=args.judge_base_url, timeout=1800.0,
                                     max_retries=args.api_max_retries)
        else:
            matcher_client = judge_model
        backend = DiagAgentBackend(
            inner_backend=backend,
            matcher_client=matcher_client,
            matcher_model=args.judge_model,
        )
        print(f"DiagAgent adapter: matcher={args.judge_model}, "
              f"temp={backend.inner.temperature}, max_tokens={backend.inner.max_tokens}")

    # ── Init patient LLM ───────────────────────────────────────────────
    patient_client = None
    if "conversation" in args.eval_types:
        from openai import OpenAI as _OpenAI
        patient_client = _OpenAI(base_url=args.patient_base_url, timeout=1800.0, max_retries=args.api_max_retries)
        print(f"Patient LLM: {args.patient_model} at {args.patient_base_url}")

    # ── Run evaluation per dataset × eval type ─────────────────────────
    all_results: dict[str, dict[str, dict]] = {}  # dataset_label -> eval_type -> metrics

    for ds_label, ds_path in data_sources:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds_label} ({ds_path})")
        print(f"{'=' * 60}")

        records = _load_records(ds_path)
        print(f"  Loaded {len(records)} records")
        if not records:
            print("  No records, skipping.")
            all_results[ds_label] = {"error": "no records"}
            continue

        if "conversation" in args.eval_types:
            conv_samples = prepare_conversation_eval_samples(records)
            conv_samples = _sample(conv_samples, args.max_conv_samples)
            print(f"  [Conversation] {len(conv_samples)} candidate samples")

        weave_dataset_name = args.weave_dataset_ref or ds_label
        weave_model_name = args.weave_model_name or (_model_label(args.model) if args.model else args.api_model)
        requested = set(args.eval_types)
        ds_results: dict[str, dict] = {}
        serialize = args.model is not None

        if "diagnosis" in requested:
            diag_results = run_weave_eval(
                backend,
                records,
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                mode="diagnosis",
                max_diag_samples=args.max_diag_samples,
                serialize=serialize,
                model_name=weave_model_name,
                dataset_name=weave_dataset_name,
                dataset_ref=args.weave_dataset_ref,
            )
            ds_results.update(diag_results)

        if "conversation" in requested:
            conv_results = run_weave_eval(
                backend,
                records,
                judge_model=judge_model,
                judge_tokenizer=judge_tokenizer,
                mode="conversation",
                max_conversation_samples=args.max_conv_samples,
                serialize=serialize,
                model_name=weave_model_name,
                dataset_name=weave_dataset_name,
                dataset_ref=args.weave_dataset_ref,
                patient_client=patient_client,
                patient_model=args.patient_model,
                max_turns=args.max_turns,
                patient_include_persona=args.persona,
                patient_style=args.patient_style,
            )
            ds_results.update(conv_results)

        all_results[ds_label] = ds_results

        # ── Log to wandb ───────────────────────────────────────────
        if wandb_run is not None:
            log_dict = {}
            for eval_type, metrics in ds_results.items():
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            log_dict[f"eval/{ds_label}/{eval_type}/{k}"] = v
                        elif isinstance(v, dict) and "mean" in v:
                            log_dict[f"eval/{ds_label}/{eval_type}/{k}"] = v["mean"]
                    # Compute effective rewards for wandb
                    flat = {k: (v["mean"] if isinstance(v, dict) and "mean" in v else v)
                            for k, v in metrics.items() if isinstance(v, (int, float, dict))}
                    diag_r = flat.get("diagnosis_reward")
                    diag_f = flat.get("diagnosis_found")
                    if isinstance(diag_r, (int, float)) and isinstance(diag_f, (int, float)) and diag_f > 0:
                        log_dict[f"eval/{ds_label}/{eval_type}/diagnosis_effective_reward"] = diag_r / diag_f
            if log_dict:
                wandb_run.log(log_dict)

    # ── Print summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Model: {args.model or args.api_model}")
    print()

    for ds_label, ds_results in all_results.items():
        if isinstance(ds_results, dict) and "error" in ds_results:
            print(f"  {ds_label}: {ds_results['error']}")
            continue

        print(f"  {ds_label}:")
        for eval_type, metrics in ds_results.items():
            if not isinstance(metrics, dict):
                continue
            # Extract mean values from Weave summary structure {key: {"mean": val}}
            flat = {}
            for k, v in metrics.items():
                if isinstance(v, float):
                    flat[k] = v
                elif isinstance(v, dict) and "mean" in v:
                    flat[k] = v["mean"]

            # Compute effective rewards (mean over valid samples only)
            diag_r = flat.get("diagnosis_reward")
            diag_found = flat.get("diagnosis_found")
            if diag_r is not None and diag_found and diag_found > 0:
                flat["diagnosis_effective_reward"] = diag_r / diag_found

            parts = [f"{k}={v:.4f}" for k, v in flat.items() if isinstance(v, (int, float))]
            print(f"    {eval_type}: {', '.join(parts)}")
        print()

    print(f"{'=' * 70}")

    # ── Log summary table to wandb ─────────────────────────────────────
    if wandb_run is not None:
        import wandb
        summary_data = []
        for ds_label, ds_results in all_results.items():
            if isinstance(ds_results, dict) and "error" in ds_results:
                continue
            for eval_type, metrics in ds_results.items():
                if not isinstance(metrics, dict):
                    continue
                row = {"dataset": ds_label, "eval_type": eval_type}
                row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
                summary_data.append(row)
        if summary_data:
            table = wandb.Table(
                columns=list(summary_data[0].keys()),
                data=[list(row.values()) for row in summary_data],
            )
            wandb_run.log({"eval/summary_table": table})
        wandb_run.finish()

    # ── Save to file ───────────────────────────────────────────────────
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")
