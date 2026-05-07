"""
Evaluation module for medical diagnosis SFT model.

Evaluates:
1. Tool call accuracy (name + parameters)
2. Diagnosis accuracy (using LLM-as-a-judge)
3. Full conversation (LLM patient from scratch, scoring both diagnosis and tool calls)

Supports multiple modes:
- Prefill mode (JSONL): loads pre-existing conversations, starts mid-conversation
- Full conversation mode (CSV): model drives the entire consultation with an LLM patient
- Diagnosis-only mode: model sees full conversation context, must produce diagnosis
- Conversation mode (weave): full multi-turn consultation from scratch with LLM patient

Data sources: local JSONL, local CSV, HuggingFace datasets (multiple supported).

Can be used:
- As a standalone script: python -m evaluation --model /path/to/checkpoint
  (defaults: both test sets, all eval types, results logged to wandb with "eval" tag)
- As a callback during training: PeriodicEvaluationCallback
- Via weave: python -m evaluation.weave_eval --model ... --data ... --mode conversation
"""

# Re-export all public symbols for backward compatibility
from .data import (
    extract_diagnosis,
    parse_available_tools_from_system_prompt,
    prepare_eval_sample,
    prepare_diagnosis_only_sample,
    prepare_diagnosis_eval_samples,
    prepare_tool_call_eval_samples,
    prepare_conversation_eval_samples,
    load_eval_data_jsonl,
    load_eval_data_csv,
    load_eval_data,
    load_hf_eval_data,
    load_and_split_data,
)
from .generation import (
    generate_response,
    generate_continuation,
    extract_content_before_tool_call,
    get_tool_response,
    parse_tool_calls_from_output,
    parse_tool_call_tag_from_output,
    get_patient_response,
)
from .metrics import (
    jaccard_similarity,
    compute_tool_call_reward,
    compute_diagnosis_reward,
    judge_diagnosis_score,
    judge_diagnosis_score_api,
)
from .runners import (
    run_evaluation,
    run_diagnosis_only_evaluation,
    run_diagnosis_eval,
    run_tool_call_format_eval,
    run_conversation_eval,
)
from .callbacks import (
    PeriodicEvaluationCallback,
    EvaluationCallback,
)
from .utils import (
    _is_main_process,
    _get_dist_info,
    _is_fsdp_model,
)
try:
    from .cli import DEFAULT_TEST_SETS
except ImportError:
    DEFAULT_TEST_SETS = {}

# weave_eval is imported on-demand (requires weave)
