"""
Unified CLI for the data generation pipeline.

Usage:
    python -m data.pipeline extract [OPTIONS]
    python -m data.pipeline filter  [OPTIONS]
    python -m data.pipeline generate [OPTIONS]
    python -m data.pipeline all     [OPTIONS]   # run all three steps sequentially

The ``all`` subcommand exposes shared flags (--base-url, --model, --batch-size,
--max-workers, --total-size) that apply to every stage.  Per-stage I/O paths
are kept separate because each stage reads/writes different files.
"""

import argparse
import sys

from data.extract import run_extract, DEFAULT_INPUT as EXT_INPUT, DEFAULT_OUTPUT as EXT_OUTPUT, \
    DEFAULT_CHECKPOINT as EXT_CKPT, DEFAULT_BASE_URL as EXT_URL, DEFAULT_MODEL as EXT_MODEL, \
    DEFAULT_BATCH_SIZE as EXT_BATCH, DEFAULT_MAX_WORKERS as EXT_WORKERS
from data.filter import run_filter, DEFAULT_INPUT as FIL_INPUT, DEFAULT_OUTPUT as FIL_OUTPUT, \
    DEFAULT_CHECKPOINT as FIL_CKPT, DEFAULT_BASE_URL as FIL_URL, DEFAULT_MODEL as FIL_MODEL, \
    DEFAULT_BATCH_SIZE as FIL_BATCH, DEFAULT_MAX_WORKERS as FIL_WORKERS
from data.generate import run_pipeline as run_generate, DEFAULT_INPUT as GEN_INPUT, \
    DEFAULT_OUTPUT as GEN_OUTPUT, DEFAULT_CHECKPOINT as GEN_CKPT, DEFAULT_BASE_URL as GEN_URL, \
    DEFAULT_MODEL as GEN_MODEL, DEFAULT_BATCH_SIZE as GEN_BATCH, DEFAULT_MAX_WORKERS as GEN_WORKERS, \
    DEFAULT_MAX_TURNS as GEN_TURNS, DEFAULT_MAX_INQUIRY_TURNS as GEN_INQUIRY_TURNS, \
    DEFAULT_DIAGNOSIS_MATCH_RETRIES as GEN_DIAG_RETRIES, \
    DEFAULT_DIAGNOSIS_ALIASES_PATH as GEN_DIAG_ALIASES


def _add_extract_args(parser: argparse.ArgumentParser):
    parser.add_argument("--input", default=EXT_INPUT, help="Input JSON path")
    parser.add_argument("--output", default=EXT_OUTPUT, help="Output CSV path")
    parser.add_argument("--checkpoint", default=EXT_CKPT, help="Checkpoint JSON path")
    parser.add_argument("--base-url", default=EXT_URL, help="API base URL")
    parser.add_argument("--model", default=EXT_MODEL, help="Model name")
    parser.add_argument("--batch-size", type=int, default=EXT_BATCH, help="Batch size")
    parser.add_argument("--max-workers", type=int, default=EXT_WORKERS, help="Parallel workers")
    parser.add_argument(
        "--total-size",
        type=int,
        default=None,
        help="Target total number of extracted records after checkpointing",
    )


def _add_filter_args(parser: argparse.ArgumentParser):
    parser.add_argument("--input", default=FIL_INPUT, help="Input CSV path")
    parser.add_argument("--output", default=FIL_OUTPUT, help="Output CSV path")
    parser.add_argument("--checkpoint", default=FIL_CKPT, help="Checkpoint JSON path")
    parser.add_argument("--base-url", default=FIL_URL, help="API base URL")
    parser.add_argument("--model", default=FIL_MODEL, help="Model name")
    parser.add_argument("--batch-size", type=int, default=FIL_BATCH, help="Batch size")
    parser.add_argument("--max-workers", type=int, default=FIL_WORKERS, help="Parallel workers")
    parser.add_argument("--start", type=int, default=None, help="Start row index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End row index (exclusive)")
    parser.add_argument("--rl-fraction", type=float, default=0.0, help="Fraction of rows to hold out for RL training")
    parser.add_argument("--eval-fraction", type=float, default=0.0, help="Fraction of rows to hold out for evaluation")
    parser.add_argument("--use-batch-api", action="store_true",
                        help="Use OpenAI Batch API for LLM calls (~50%% cost savings; requires api.openai.com)")
    parser.add_argument("--batch-poll-interval", type=int, default=60,
                        help="Seconds between Batch API status polls (default: 60)")


def _add_generate_args(parser: argparse.ArgumentParser):
    parser.add_argument("--input", default=GEN_INPUT, help="Input CSV path")
    parser.add_argument("--output", default=GEN_OUTPUT, help="Output JSONL path")
    parser.add_argument("--checkpoint", default=GEN_CKPT, help="Checkpoint JSON path")
    parser.add_argument("--base-url", default=GEN_URL, help="API base URL")
    parser.add_argument("--model", default=GEN_MODEL, help="Model name")
    parser.add_argument("--batch-size", type=int, default=GEN_BATCH, help="Batch size")
    parser.add_argument("--max-workers", type=int, default=GEN_WORKERS, help="Parallel workers")
    parser.add_argument("--max-turns", type=int, default=GEN_TURNS, help="Max conversation turns")
    parser.add_argument("--max-inquiry-turns", type=int, default=GEN_INQUIRY_TURNS, help="Max symptom inquiry turns")
    parser.add_argument("--diagnosis-match-retries", type=int, default=GEN_DIAG_RETRIES,
                        help="Retry final diagnosis turn until the diagnosis tag matches the gold label")
    parser.add_argument("--diagnosis-aliases-path", default=GEN_DIAG_ALIASES,
                        help="JSON file of manually approved diagnosis aliases/equivalences")
    parser.add_argument("--total-size", type=int, default=None, help="Max patients to process")
    parser.add_argument("--patient-noise-level", type=float, default=0.0,
                        help="Patient noise level 0.0-1.0 (probability of noise per turn/symptom)")
    parser.add_argument("--exam-noise-level", type=float, default=0.0,
                        help="Exam noise level 0.0-1.0 (probability of noise per exam finding)")
    parser.add_argument("--noise-seed", type=int, default=None,
                        help="Random seed for reproducible noise injection")


def main():
    parser = argparse.ArgumentParser(
        description="Data generation pipeline: extract -> filter -> generate",
    )
    subparsers = parser.add_subparsers(dest="step", help="Pipeline step to run")

    # extract
    sp_extract = subparsers.add_parser("extract", help="Step 1: Extract patient info from PMC JSON via LLM")
    _add_extract_args(sp_extract)

    # filter
    sp_filter = subparsers.add_parser("filter", help="Step 2: Filter CSV and classify exams to tool calls")
    _add_filter_args(sp_filter)

    # generate
    sp_generate = subparsers.add_parser("generate", help="Step 3: Generate doctor-patient conversations")
    _add_generate_args(sp_generate)

    # all — shared params that apply to every stage
    sp_all = subparsers.add_parser("all", help="Run all three steps sequentially")

    shared = sp_all.add_argument_group("shared (apply to all stages)")
    shared.add_argument("--base-url", default=None,
                        help="API base URL for all stages (overrides per-stage defaults)")
    shared.add_argument("--model", default=None,
                        help="Model name for all stages (overrides per-stage defaults)")
    shared.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for all stages (overrides per-stage defaults)")
    shared.add_argument("--max-workers", type=int, default=None,
                        help="Parallel workers for all stages")
    shared.add_argument("--total-size", type=int, default=None,
                        help="Max records to process for extract & generate stages")

    ext_grp = sp_all.add_argument_group("extract stage I/O")
    ext_grp.add_argument("--extract-input", default=EXT_INPUT, help="Extract: input JSON")
    ext_grp.add_argument("--extract-output", default=EXT_OUTPUT, help="Extract: output CSV")
    ext_grp.add_argument("--extract-checkpoint", default=EXT_CKPT, help="Extract: checkpoint")

    fil_grp = sp_all.add_argument_group("filter stage I/O")
    fil_grp.add_argument("--filter-input", default=FIL_INPUT, help="Filter: input CSV")
    fil_grp.add_argument("--filter-output", default=FIL_OUTPUT, help="Filter: output CSV")
    fil_grp.add_argument("--filter-checkpoint", default=FIL_CKPT, help="Filter: checkpoint")
    fil_grp.add_argument("--filter-start", type=int, default=None, help="Filter: start row index")
    fil_grp.add_argument("--filter-end", type=int, default=None, help="Filter: end row index")
    fil_grp.add_argument("--rl-fraction", type=float, default=0.0, help="Fraction of rows to hold out for RL training")
    fil_grp.add_argument("--eval-fraction", type=float, default=0.0, help="Fraction of rows to hold out for evaluation")
    fil_grp.add_argument("--filter-use-batch-api", action="store_true",
                         help="Filter: use OpenAI Batch API for LLM calls (~50%% cost savings)")
    fil_grp.add_argument("--filter-batch-poll-interval", type=int, default=60,
                         help="Filter: seconds between Batch API polls (default: 60)")

    gen_grp = sp_all.add_argument_group("generate stage I/O")
    gen_grp.add_argument("--generate-input", default=GEN_INPUT, help="Generate: input CSV")
    gen_grp.add_argument("--generate-output", default=GEN_OUTPUT, help="Generate: output JSONL")
    gen_grp.add_argument("--generate-checkpoint", default=GEN_CKPT, help="Generate: checkpoint")
    gen_grp.add_argument("--generate-max-turns", type=int, default=GEN_TURNS, help="Max conversation turns")
    gen_grp.add_argument("--generate-max-inquiry-turns", type=int, default=GEN_INQUIRY_TURNS, help="Max symptom inquiry turns")
    gen_grp.add_argument("--generate-diagnosis-match-retries", type=int, default=GEN_DIAG_RETRIES,
                         help="Retry final diagnosis turn until the diagnosis tag matches the gold label")
    gen_grp.add_argument("--generate-diagnosis-aliases-path", default=GEN_DIAG_ALIASES,
                         help="JSON file of manually approved diagnosis aliases/equivalences")
    gen_grp.add_argument("--generate-patient-noise-level", type=float, default=0.0,
                         help="Patient noise level 0.0-1.0")
    gen_grp.add_argument("--generate-exam-noise-level", type=float, default=0.0,
                         help="Exam noise level 0.0-1.0")
    gen_grp.add_argument("--generate-noise-seed", type=int, default=None,
                         help="Random seed for noise injection")

    args = parser.parse_args()

    if args.step is None:
        parser.print_help()
        sys.exit(1)

    if args.step == "extract":
        run_extract(args)
    elif args.step == "filter":
        run_filter(args)
    elif args.step == "generate":
        run_generate(args)
    elif args.step == "all":
        # Helper: pick shared value if set, otherwise per-stage default
        def _or(shared_val, default):
            return shared_val if shared_val is not None else default

        extract_args = argparse.Namespace(
            input=args.extract_input,
            output=args.extract_output,
            checkpoint=args.extract_checkpoint,
            base_url=_or(args.base_url, EXT_URL),
            model=_or(args.model, EXT_MODEL),
            batch_size=_or(args.batch_size, EXT_BATCH),
            max_workers=_or(args.max_workers, EXT_WORKERS),
            total_size=args.total_size,
        )
        filter_args = argparse.Namespace(
            input=args.filter_input,
            output=args.filter_output,
            checkpoint=args.filter_checkpoint,
            base_url=_or(args.base_url, FIL_URL),
            model=_or(args.model, FIL_MODEL),
            batch_size=_or(args.batch_size, FIL_BATCH),
            max_workers=_or(args.max_workers, FIL_WORKERS),
            start=args.filter_start,
            end=args.filter_end,
            rl_fraction=args.rl_fraction,
            eval_fraction=args.eval_fraction,
            use_batch_api=args.filter_use_batch_api,
            batch_poll_interval=args.filter_batch_poll_interval,
        )
        generate_args = argparse.Namespace(
            input=args.generate_input,
            output=args.generate_output,
            checkpoint=args.generate_checkpoint,
            base_url=_or(args.base_url, GEN_URL),
            model=_or(args.model, GEN_MODEL),
            batch_size=_or(args.batch_size, GEN_BATCH),
            max_workers=_or(args.max_workers, GEN_WORKERS),
            max_turns=args.generate_max_turns,
            max_inquiry_turns=args.generate_max_inquiry_turns,
            diagnosis_match_retries=args.generate_diagnosis_match_retries,
            diagnosis_aliases_path=args.generate_diagnosis_aliases_path,
            total_size=args.total_size,
            patient_noise_level=args.generate_patient_noise_level,
            exam_noise_level=args.generate_exam_noise_level,
            noise_seed=args.generate_noise_seed,
        )

        print("=" * 60)
        print("Step 1/3: Extract")
        print("=" * 60)
        run_extract(extract_args)

        print("\n" + "=" * 60)
        print("Step 2/3: Filter")
        print("=" * 60)
        run_filter(filter_args)

        print("\n" + "=" * 60)
        print("Step 3/3: Generate")
        print("=" * 60)
        run_generate(generate_args)

        print("\n" + "=" * 60)
        print("Pipeline complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
