"""Standalone weave.Evaluation for evaluating checkpoints or API-served models.

Usage:
    # Evaluate a local checkpoint (diagnosis)
    python -m evaluation.weave_eval --model path/to/checkpoint --data eval.jsonl

    # Evaluate an API model
    python -m evaluation.weave_eval --api-model gpt-4.1-mini --api-base-url https://api.openai.com/v1 --data eval.jsonl

    # Both eval types with HuggingFace dataset
    python -m evaluation.weave_eval --model path/to/checkpoint --hf-dataset org/dataset --mode both

    # Full conversation eval (LLM patient, from scratch)
    python -m evaluation.weave_eval --model path/to/checkpoint --data eval.jsonl \\
        --mode conversation --patient-base-url https://api.openai.com/v1 --patient-model gpt-4.1-mini

    # All eval types
    python -m evaluation.weave_eval --model path/to/checkpoint --data eval.jsonl \\
        --mode all --patient-base-url https://api.openai.com/v1
"""

import argparse
import asyncio
import json
import os

try:
    import weave
except ImportError:
    raise ImportError("weave is required for weave_eval. Install with: pip install weave")

from .data import (
    extract_diagnosis,
    prepare_diagnosis_eval_samples,
    prepare_conversation_eval_samples,
)
from .generation import (
    parse_tool_call_tag_from_output,
    get_tool_response,
    get_patient_response,
)
from .metrics import compute_tool_call_reward, compute_diagnosis_reward
from .trace_format import format_conversation, serialize_for_trace


_ACTIVE_BACKEND = None
_ACTIVE_PATIENT_CLIENT = None
_ACTIVE_PATIENT_MODEL = None
_ACTIVE_MAX_TURNS = 15
_ACTIVE_JUDGE_MODEL = None
_ACTIVE_JUDGE_TOKENIZER = None
_ACTIVE_PATIENT_INCLUDE_PERSONA = True
_ACTIVE_PATIENT_STYLE = "default"


# ---------------------------------------------------------------------------
# Weave Model wrapper — name field controls the Weave UI label; predict
# source code is constant so same name = same content hash = same version.
# ---------------------------------------------------------------------------

class MedAgentModel(weave.Model):
    """Thin wrapper so Weave shows a meaningful model name."""

    @weave.op()
    def predict(
        self,
        # diagnosis eval
        input_messages: list[dict] | None = None,
        # conversation eval
        patient_kwargs: dict | None = None,
        available_tools: list[dict] | None = None,
        gt_tool_findings: list[dict] | None = None,
        available_tool_names: list[str] | None = None,
        **kwargs,
    ) -> dict:
        return _model_predict(
            input_messages=input_messages,
            patient_kwargs=patient_kwargs,
            available_tools=available_tools,
            gt_tool_findings=gt_tool_findings,
            available_tool_names=available_tool_names,
        )


# ---------------------------------------------------------------------------
# Generation backends (plain helpers, not weave.Model)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Generation backends (plain helpers, not weave.Model)
# ---------------------------------------------------------------------------

class _LocalBackend:
    """Loads an HF checkpoint and generates via the evaluation generation module."""

    def __init__(self, model_path, tokenizer_path=None, max_new_tokens=4096, temperature=0.1):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._model = None
        self._tokenizer = None

    def load(self):
        from chat import load_model

        self._model, self._tokenizer = load_model(
            self.model_path,
            tokenizer_override=self.tokenizer_path,
        )
        print(f"Model loaded on {next(self._model.parameters()).device}")

    def generate(self, messages, max_new_tokens=None):
        from .generation import generate_response
        return generate_response(
            self._model, self._tokenizer, messages,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=self.temperature,
        )

    def generate_continuation(self, messages, prefix_text, max_new_tokens=None):
        from .generation import generate_continuation
        return generate_continuation(
            self._model, self._tokenizer, messages, prefix_text,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=self.temperature,
        )


class _APIBackend:
    """Wraps an OpenAI-compatible API for generation."""

    def __init__(self, api_base_url, model_name, max_tokens=4096, temperature=0.1,
                 max_retries=6, extra_body=None):
        self.api_base_url = api_base_url
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.extra_body = extra_body
        self._client = None

    def load(self):
        from openai import OpenAI
        api_key = None
        if "anthropic.com" in (self.api_base_url or ""):
            api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self._client = OpenAI(
            base_url=self.api_base_url, api_key=api_key,
            timeout=1800.0, max_retries=self.max_retries,
        )
        print(f"API model: {self.model_name} at {self.api_base_url}")

    @staticmethod
    def _ensure_alternating_roles(messages):
        """Ensure messages alternate user/assistant after system for vllm compat."""
        out = []
        for msg in messages:
            if msg["role"] == "system":
                out.append(msg)
                continue
            if msg["role"] == "tool":
                out.append(msg)
                continue
            if out and out[-1]["role"] == msg["role"]:
                # Merge consecutive same-role messages
                out[-1] = dict(out[-1])
                out[-1]["content"] = (out[-1].get("content") or "") + "\n" + (msg.get("content") or "")
            elif not out or (out[-1]["role"] == "system" and msg["role"] == "assistant"):
                # First non-system message is assistant — prepend a user message
                out.append({"role": "user", "content": "[Begin consultation]"})
                out.append(msg)
            else:
                out.append(msg)
        # Anthropic (and some providers) reject payloads with only system messages.
        # Conversation mode's first turn hits this on a fresh stream.
        if not any(m["role"] != "system" for m in out):
            out.append({"role": "user", "content": "[Begin consultation]"})
        return out

    @staticmethod
    def _structure_tool_calls(messages):
        """Rewrite <tool_call> text tags into OpenAI structured tool_calls.

        The internal message stream uses ``<tool_call>{...}</tool_call>`` in
        assistant content followed by ``role: tool`` messages — the format the
        trained model emits and the local chat template renders. OpenAI's API
        requires the assistant message to carry a structured ``tool_calls``
        field whose ids match the following tool messages'. Translate only at
        the API boundary; upstream data structures stay unchanged.
        """
        import re

        out = []
        pending_ids: list[str] = []
        pending_cursor = 0
        counter = 0

        for msg in messages:
            role = msg.get("role")
            if role == "assistant":
                content = msg.get("content") or ""
                tool_calls_struct = []
                for tag in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', content, re.DOTALL):
                    try:
                        obj = json.loads(tag.group(1))
                    except (json.JSONDecodeError, ValueError):
                        continue
                    name = obj.get("name")
                    if not name:
                        continue
                    args = obj.get("arguments", obj.get("parameters", {}))
                    if not isinstance(args, str):
                        args = json.dumps(args if isinstance(args, dict) else {}, ensure_ascii=False)
                    counter += 1
                    call_id = f"call_{counter:04d}"
                    tool_calls_struct.append({
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    })
                if tool_calls_struct:
                    stripped = re.sub(r'<tool_call>.*?</tool_call>', '', content, flags=re.DOTALL).strip()
                    out.append({
                        "role": "assistant",
                        "content": stripped,
                        "tool_calls": tool_calls_struct,
                    })
                    pending_ids = [tc["id"] for tc in tool_calls_struct]
                    pending_cursor = 0
                else:
                    out.append(msg)
                    pending_ids = []
                    pending_cursor = 0
                continue
            if role == "tool":
                new_msg = dict(msg)
                if pending_cursor < len(pending_ids):
                    new_msg["tool_call_id"] = pending_ids[pending_cursor]
                    pending_cursor += 1
                out.append(new_msg)
                continue
            if role == "user" and pending_cursor < len(pending_ids):
                # SFT-serialized prefix: train/sft/dataset.py:serialize_tool_calls
                # rewrites role:tool -> role:user. Revert it at the API boundary
                # so OpenAI sees a structurally valid tool_calls/tool pairing.
                out.append({
                    "role": "tool",
                    "tool_call_id": pending_ids[pending_cursor],
                    "content": msg.get("content", ""),
                })
                pending_cursor += 1
                continue
            pending_ids = []
            pending_cursor = 0
            out.append(msg)
        return out

    def _needs_structured_tool_calls(self) -> bool:
        """OpenAI and Anthropic APIs strictly require assistant.tool_calls +
        role:tool pairing. vllm/sglang/etc. accept the pass-through format and
        may render structured tool_calls differently via their chat template,
        so keep their behavior unchanged.
        """
        url = self.api_base_url or ""
        return "openai.com" in url or "anthropic.com" in url

    def generate(self, messages, max_tokens=None):
        msgs = self._structure_tool_calls(messages) if self._needs_structured_tool_calls() else messages
        msgs = self._ensure_alternating_roles(msgs)
        req_kwargs = dict(
            model=self.model_name,
            messages=msgs,
            max_completion_tokens=max_tokens or self.max_tokens,
            temperature=self.temperature,
        )
        if self.extra_body:
            req_kwargs["extra_body"] = self.extra_body
        resp = self._client.chat.completions.create(**req_kwargs)
        msg = resp.choices[0].message
        content = msg.content or ""
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            # Sonnet via Anthropic's OpenAI-compat returns tool_use as
            # message.tool_calls with empty content. Reconstruct the text-tag
            # form so downstream parse_tool_call_tag_from_output sees them.
            parts = [content] if content else []
            for tc in tool_calls:
                fn = tc.function
                raw_args = fn.arguments
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except (json.JSONDecodeError, TypeError):
                    args = {}
                call_obj = json.dumps({"name": fn.name, "arguments": args}, ensure_ascii=False)
                parts.append(f"<tool_call>\n{call_obj}\n</tool_call>")
            return "\n\n".join(parts)
        return content

    def generate_continuation(self, messages, prefix_text, max_tokens=None):
        msgs = [dict(m) for m in messages]
        if prefix_text:
            if msgs and msgs[-1]["role"] == "assistant":
                msgs[-1]["content"] = (msgs[-1].get("content") or "") + prefix_text
            else:
                msgs.append({"role": "assistant", "content": prefix_text})
        return self.generate(msgs, max_tokens)


# ---------------------------------------------------------------------------
# Model ops — defined at module level so Weave content-hashes their source
# code consistently across runs (no closure-captured objects that change hash).
# All runtime state is accessed via module-level _ACTIVE_* globals.
# ---------------------------------------------------------------------------

@weave.op(name="model_predict")
def _model_predict(
    # diagnosis eval
    input_messages: list[dict] | None = None,
    # conversation eval
    patient_kwargs: dict | None = None,
    available_tools: list[dict] | None = None,
    gt_tool_findings: list[dict] | None = None,
    available_tool_names: list[str] | None = None,
    # common (passed through but unused by model)
    **kwargs,
) -> dict:
    from evaluation import weave_eval as _we
    from evaluation.diagagent_backend import set_active_patient_kwargs

    if input_messages is not None:
        # Clear any stale conv-mode patient_kwargs so the DiagAgent adapter
        # falls back to parsing demographics from the prefilled system prompt.
        set_active_patient_kwargs(None)
        working_messages = [msg.copy() for msg in input_messages]
        gt_findings_remaining = [entry.copy() for entry in (gt_tool_findings or [])]
        available_names = set(available_tool_names or [])
        predicted = None
        call_counter = 0
        output = ""

        for _turn in range(_we._ACTIVE_MAX_TURNS):
            output = _we._ACTIVE_BACKEND.generate(working_messages)
            diagnosis = extract_diagnosis(output)
            if diagnosis:
                predicted = diagnosis
                working_messages.append({"role": "assistant", "content": output})
                break

            tool_calls = parse_tool_call_tag_from_output(output)
            if not tool_calls:
                break

            working_messages.append({"role": "assistant", "content": output})
            for tc in tool_calls:
                call_counter += 1
                response_text, matched_idx = get_tool_response(
                    tc, gt_findings_remaining, available_names,
                )
                if matched_idx is not None:
                    gt_findings_remaining.pop(matched_idx)
                working_messages.append({
                    "role": "tool",
                    "tool_call_id": f"eval_{call_counter:04d}",
                    "content": response_text,
                })

        return {
            "output": output,
            "predicted_diagnosis": predicted,
        }

    if patient_kwargs is not None and available_tools is not None:
        from data.prompt import get_doctor_system_prompt

        # Plumb to DiagAgent adapter (no-op for other backends): they need
        # demographics + symptoms + history to seed the EHR template on turn
        # 1, before any patient utterance has arrived in the message stream.
        set_active_patient_kwargs(patient_kwargs)

        system_prompt = get_doctor_system_prompt(
            medical_history=patient_kwargs.get("medical_history", ""),
            available_tools=available_tools,
            include_tool_call_format=True,
            age=patient_kwargs.get("age", ""),
            gender=patient_kwargs.get("gender", ""),
            race=patient_kwargs.get("race", ""),
            additional_demographics=patient_kwargs.get("additional_demographics", ""),
        )
        messages = [{"role": "system", "content": system_prompt}]

        available_names = set(available_tool_names or [])
        findings_remaining = [f.copy() for f in (gt_tool_findings or [])]
        all_predicted_tools = []
        predicted_diagnosis = None
        call_counter = 0

        for _turn in range(_we._ACTIVE_MAX_TURNS):
            output = _we._ACTIVE_BACKEND.generate(messages)
            diagnosis = extract_diagnosis(output)
            if diagnosis:
                predicted_diagnosis = diagnosis
                messages.append({"role": "assistant", "content": output})
                break

            tool_calls = parse_tool_call_tag_from_output(output)
            if tool_calls:
                messages.append({"role": "assistant", "content": output})
                for tc in tool_calls:
                    call_counter += 1
                    all_predicted_tools.append(tc)
                    response_text, matched_idx = get_tool_response(
                        tc, findings_remaining, available_names,
                    )
                    if matched_idx is not None:
                        findings_remaining.pop(matched_idx)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": f"conv_{call_counter:04d}",
                        "content": response_text,
                    })
                continue

            messages.append({"role": "assistant", "content": output})
            if _we._ACTIVE_PATIENT_CLIENT:
                patient_call_kwargs = dict(patient_kwargs)
                patient_call_kwargs["include_persona"] = _we._ACTIVE_PATIENT_INCLUDE_PERSONA
                patient_call_kwargs["patient_style"] = _we._ACTIVE_PATIENT_STYLE
                patient_reply = get_patient_response(
                    _we._ACTIVE_PATIENT_CLIENT, _we._ACTIVE_PATIENT_MODEL,
                    messages, patient_call_kwargs,
                )
                messages.append({"role": "user", "content": patient_reply})
            else:
                break

        return {
            "output": format_conversation(messages, include_system=False),
            "conversation": serialize_for_trace(messages),
            "predicted_diagnosis": predicted_diagnosis,
            "predicted_tools": all_predicted_tools,
            "num_turns": sum(1 for m in messages if m["role"] == "assistant"),
        }

    provided = sorted(
        k for k, v in {
            "input_messages": input_messages,
            "prior_messages": prior_messages,
            "patient_kwargs": patient_kwargs,
        }.items() if v is not None
    )
    raise ValueError(f"Unsupported evaluation input combination: {provided}")


# ---------------------------------------------------------------------------
# Scorers — module-level ops for stable Weave content hashing.
# ---------------------------------------------------------------------------

@weave.op(name="diagnosis_scorer")
def _diagnosis_scorer(output: dict, ground_truth_diagnosis: str, **kwargs) -> dict:
    from evaluation import weave_eval as _we
    predicted = output["predicted_diagnosis"]
    diag = compute_diagnosis_reward(
        predicted, ground_truth_diagnosis, _we._ACTIVE_JUDGE_MODEL, _we._ACTIVE_JUDGE_TOKENIZER,
        return_details=True,
    )
    return {
        "diagnosis_reward": diag["reward"],
        "diagnosis_accuracy_strict": diag["accuracy_strict"],
        "diagnosis_accuracy_lenient": diag["accuracy_lenient"],
        "diagnosis_found": 1.0 if predicted is not None else 0.0,
    }



@weave.op(name="conversation_scorer")
def _conversation_scorer(
    output: dict,
    ground_truth_diagnosis: str,
    ground_truth_tools: list[dict],
    available_tool_names: list[str],
    **kwargs,
) -> dict:
    from evaluation import weave_eval as _we
    predicted_diag = output["predicted_diagnosis"]
    predicted_tools = output["predicted_tools"]

    diag = compute_diagnosis_reward(
        predicted_diag, ground_truth_diagnosis, _we._ACTIVE_JUDGE_MODEL, _we._ACTIVE_JUDGE_TOKENIZER,
        return_details=True,
    )

    tool_reward = compute_tool_call_reward(predicted_tools, ground_truth_tools)

    available = set(available_tool_names)
    hallucinated = sum(
        1 for tc in predicted_tools if tc.get("name", "") not in available
    )
    hallucination_rate = hallucinated / len(predicted_tools) if predicted_tools else 0.0

    return {
        "diagnosis_reward": diag["reward"],
        "diagnosis_accuracy_strict": diag["accuracy_strict"],
        "diagnosis_accuracy_lenient": diag["accuracy_lenient"],
        "diagnosis_found": 1.0 if predicted_diag else 0.0,
        "tool_call_reward": tool_reward,
        "tool_hallucination_rate": hallucination_rate,
        "num_tool_calls": float(len(predicted_tools)),
        "num_turns": float(output["num_turns"]),
    }


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def _load_records(data_paths=None, hf_datasets=None):
    """Load raw conversation records from local files and/or HF datasets."""
    records = []

    if data_paths:
        for path in data_paths:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

    if hf_datasets:
        from datasets import load_dataset
        for ds_name in hf_datasets:
            ds = load_dataset(ds_name, split="train")
            records.extend(dict(r) for r in ds)

    return records


def _prepare_diagnosis_dataset(records, max_samples=None, serialize=True):
    """Prepare weave-compatible dataset for diagnosis evaluation."""
    # Always serialize: tool/assistant(tool_calls) roles cause 'roles must
    # alternate' errors on vllm and other OpenAI-compatible servers.
    samples = prepare_diagnosis_eval_samples(records, serialize=True)
    if max_samples and len(samples) > max_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, max_samples)

    return [
        {
            "input_messages": s["input_messages"],
            "ground_truth_diagnosis": s["ground_truth_diagnosis"],
            "gt_tool_findings": s.get("gt_tool_findings", []),
            "available_tool_names": list(s.get("available_tool_names", [])),
        }
        for s in samples
    ]



def _prepare_conversation_dataset(records, max_samples=None):
    """Prepare weave-compatible dataset for full conversation evaluation."""
    samples = prepare_conversation_eval_samples(records)
    if max_samples and len(samples) > max_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, max_samples)
    return samples


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------

def run_weave_eval(
    backend,
    records: list[dict],
    judge_model=None,
    judge_tokenizer=None,
    mode: str = "diagnosis",
    max_diag_samples: int | None = None,
    max_conversation_samples: int | None = None,
    serialize: bool = True,
    model_name: str | None = None,
    dataset_name: str | None = None,
    dataset_ref: str | None = None,
    patient_client=None,
    patient_model: str | None = None,
    max_turns: int = 15,
    patient_include_persona: bool = False,
    patient_style: str = "default",
) -> dict:
    """Run weave.Evaluation on the given records.

    Returns:
        dict with evaluation summaries
    """
    results = {}

    run_diagnosis = mode in ("diagnosis", "all")
    run_conversation = mode in ("conversation", "all")

    global _ACTIVE_BACKEND, _ACTIVE_PATIENT_CLIENT, _ACTIVE_PATIENT_MODEL, _ACTIVE_MAX_TURNS
    global _ACTIVE_JUDGE_MODEL, _ACTIVE_JUDGE_TOKENIZER, _ACTIVE_PATIENT_INCLUDE_PERSONA
    global _ACTIVE_PATIENT_STYLE
    _ACTIVE_BACKEND = backend
    _ACTIVE_PATIENT_CLIENT = patient_client
    _ACTIVE_PATIENT_MODEL = patient_model or "gpt-4.1-mini"
    _ACTIVE_MAX_TURNS = max_turns
    _ACTIVE_JUDGE_MODEL = judge_model
    _ACTIVE_JUDGE_TOKENIZER = judge_tokenizer
    _ACTIVE_PATIENT_INCLUDE_PERSONA = patient_include_persona
    _ACTIVE_PATIENT_STYLE = patient_style

    def _resolve_dataset(rows, default_name):
        if dataset_ref:
            # Full Weave URI — fetch exactly that version.
            if str(dataset_ref).startswith("weave:///"):
                dataset_obj = weave.get(dataset_ref)
                if not isinstance(dataset_obj, weave.Dataset):
                    raise TypeError(
                        f"Dataset ref {dataset_ref!r} did not resolve to weave.Dataset"
                    )
                return dataset_obj
            # Short name — try to fetch existing; if not found, create new.
            try:
                dataset_obj = weave.get(default_name)
                if isinstance(dataset_obj, weave.Dataset):
                    return dataset_obj
            except (ValueError, Exception):
                pass
        return weave.Dataset(name=default_name, rows=rows)

    # Wrap in MedAgentModel so Weave UI shows a meaningful model name.
    # Same model_label + same predict source = same content hash = same version.
    model_obj = MedAgentModel(name=model_name or "model")

    if run_diagnosis:
        if judge_model is None:
            raise ValueError("judge_model required for diagnosis evaluation")

        dataset = _prepare_diagnosis_dataset(records, max_diag_samples, serialize=serialize)
        if dataset:
            print(f"\nRunning diagnosis evaluation ({len(dataset)} samples)...")
            diag_dataset_name = f"{dataset_name}_diagnosis" if dataset_name else "diagnosis_eval"
            evaluation = weave.Evaluation(
                name="diagnosis_eval",
                dataset=_resolve_dataset(dataset, diag_dataset_name),
                scorers=[_diagnosis_scorer],
            )
            summary = asyncio.run(evaluation.evaluate(model_obj))
            results["diagnosis"] = summary
            print("Diagnosis eval complete.")
        else:
            print("No diagnosis samples found.")

    if run_conversation:
        if judge_model is None:
            raise ValueError("judge_model required for conversation evaluation")
        if patient_client is None:
            raise ValueError(
                "patient_client required for conversation evaluation. "
                "Use --patient-base-url and --patient-model."
            )

        dataset = _prepare_conversation_dataset(records, max_conversation_samples)
        if dataset:
            print(f"\nRunning conversation evaluation ({len(dataset)} samples)...")
            conv_dataset_name = f"{dataset_name}_conversation" if dataset_name else "conversation_eval"
            evaluation = weave.Evaluation(
                name="conversation_eval",
                dataset=_resolve_dataset(dataset, conv_dataset_name),
                scorers=[_conversation_scorer],
            )
            summary = asyncio.run(evaluation.evaluate(model_obj))
            results["conversation"] = summary
            print("Conversation eval complete.")
        else:
            print("No conversation samples found (records may lack patient demographics).")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run weave.Evaluation on a checkpoint or API model",
    )

    # Model source (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", type=str,
                             help="Path to local HF model checkpoint")
    model_group.add_argument("--api-model", type=str,
                             help="API model name (requires --api-base-url)")

    parser.add_argument("--api-base-url", type=str, default="https://api.openai.com/v1",
                        help="OpenAI-compatible API base URL")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer (defaults to --model)")

    # Data sources
    parser.add_argument("--data", type=str, nargs="+", default=None,
                        help="Path(s) to eval data JSONL files")
    parser.add_argument("--hf-dataset", type=str, nargs="+", default=None,
                        help="HuggingFace dataset name(s)")

    # Eval config
    
    parser.add_argument("--mode", type=str, default="diagnosis",
                        choices=["diagnosis", "conversation", "all"],
                        help="Evaluation mode (conversation requires --patient-base-url)")
    parser.add_argument("--max-diag-samples", type=int, default=None,
                        help="Max diagnosis eval samples")
    parser.add_argument("--max-conversation-samples", type=int, default=None,
                        help="Max conversation eval samples")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="Max new tokens for generation")
    parser.add_argument("--max-turns", type=int, default=15,
                        help="Max turns per conversation (conversation mode)")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature")

    # Patient LLM (conversation mode)
    parser.add_argument("--patient-base-url", type=str, default=None,
                        help="OpenAI-compatible API URL for patient LLM (conversation mode)")
    parser.add_argument("--patient-model", type=str, default="gpt-4.1-mini",
                        help="Model name for patient LLM (conversation mode)")
    parser.add_argument("--persona", action=argparse.BooleanOptionalAction, default=False,
                        help="Inject sampled persona into the patient system prompt. "
                             "Default: off. Use --persona to enable, --no-persona to force-disable.")
    parser.add_argument("--patient-style", type=str, default="default",
                        choices=["default", "agentclinic"],
                        help="Patient prompt style. 'agentclinic' mirrors AgentClinic's "
                             "PatientAgent prompt (no bias) for paper-comparable eval.")

    # Judge model
    parser.add_argument("--judge-model", type=str, default="gpt-4.1-mini",
                        help="Judge model name or path")
    parser.add_argument("--judge-base-url", type=str, default="https://api.openai.com/v1",
                        help="Judge API base URL")

    # Weave config
    parser.add_argument("--weave-project", type=str, default="medagent-eval-gold",
                        help="Weave project name")
    parser.add_argument("--weave-model-name", type=str, default=None,
                        help="Override model name shown in Weave UI (defaults to --api-model or checkpoint basename)")
    parser.add_argument("--weave-dataset-ref", type=str, default=None,
                        help="Reuse existing Weave dataset object(s) by base ref/name")

    # Concurrency
    parser.add_argument("--max-concurrency", type=int, default=8,
                        help="Max parallel in-flight eval trials (sets WEAVE_PARALLELISM)")

    # Output
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path for summary")

    args = parser.parse_args()

    if not args.data and not args.hf_dataset:
        parser.error("Must provide either --data or --hf-dataset")

    if args.api_model and not args.api_base_url:
        parser.error("--api-model requires --api-base-url")

    os.environ["WEAVE_PARALLELISM"] = str(args.max_concurrency)
    print(f"Weave parallelism: {args.max_concurrency}")

    # Init weave
    weave.init(args.weave_project)

    # Build backend
    if args.model:
        backend = _LocalBackend(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    else:
        backend = _APIBackend(
            api_base_url=args.api_base_url,
            model_name=args.api_model,
            # max_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    backend.load()

    # Build judge
    needs_judge = args.mode in ("diagnosis", "conversation", "all")
    if needs_judge:
        from openai import OpenAI
        judge_model = OpenAI(base_url=args.judge_base_url, timeout=1800.0, max_retries=2)
        judge_tokenizer = args.judge_model
        print(f"Judge: {args.judge_model} at {args.judge_base_url}")
    else:
        judge_model = None
        judge_tokenizer = None

    # Build patient LLM client (conversation mode)
    patient_client = None
    if args.mode in ("conversation", "all"):
        if args.patient_base_url is None:
            parser.error("--patient-base-url is required for conversation mode")
        from openai import OpenAI as _PatientOpenAI
        patient_client = _PatientOpenAI(base_url=args.patient_base_url, timeout=1800.0, max_retries=2)
        print(f"Patient LLM: {args.patient_model} at {args.patient_base_url}")

    # Load data
    records = _load_records(args.data, args.hf_dataset)
    print(f"Loaded {len(records)} records")

    if not records:
        print("No records found, exiting.")
        return

    # Run evaluation
    # API models use native OpenAI tool calling format (no serialization);
    # local models use <tool_call> tag serialization.
    serialize = args.model is not None
    import os
    model_name = args.weave_model_name or args.api_model or os.path.basename(os.path.normpath(args.model))
    # Build dataset name from file names (without extension) or HF dataset names
    dataset_parts = []
    if args.data:
        dataset_parts.extend(args.data)
    if args.hf_dataset:
        dataset_parts.extend(args.hf_dataset)
    dataset_name = args.weave_dataset_ref or ("+".join(dataset_parts) if dataset_parts else None)
    results = run_weave_eval(
        backend, records,
        judge_model=judge_model,
        judge_tokenizer=judge_tokenizer,
        mode=args.mode,
        max_diag_samples=args.max_diag_samples,
        max_conversation_samples=args.max_conversation_samples,
        serialize=serialize,
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_ref=args.weave_dataset_ref,
        patient_client=patient_client,
        patient_model=args.patient_model,
        max_turns=args.max_turns,
        patient_include_persona=args.persona,
        patient_style=args.patient_style,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    for eval_type, summary in results.items():
        print(f"\n  {eval_type}:")
        if isinstance(summary, dict):
            for k, v in summary.items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        if isinstance(sv, dict) and "mean" in sv:
                            print(f"    {k}/{sk}: {sv['mean']:.4f}")
                        elif isinstance(sv, (int, float)):
                            print(f"    {k}/{sk}: {sv}")
    print("=" * 60)

    if args.output:
        # Serialize summary (weave summaries may have non-serializable types)
        def _make_serializable(obj):
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_make_serializable(x) for x in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            return str(obj)

        with open(args.output, "w") as f:
            json.dump(_make_serializable(results), f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
