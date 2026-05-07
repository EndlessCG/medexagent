"""Data loading and preparation for evaluation."""

import json
import random
import re

from train.sft.dataset import serialize_tool_calls


def extract_diagnosis(text: str) -> str | None:
    """Extract diagnosis from [DIAGNOSIS: ...] format."""
    if text is None:
        return None
    match = re.search(r'\[DIAGNOSIS:\s*(.+?)\]', text, re.IGNORECASE)
    return match.group(1).strip() if match else None


def parse_available_tools_from_system_prompt(system_content: str) -> set[str]:
    """Parse available tool names from the system prompt's AVAILABLE EXAMINATIONS section.

    The system prompt lists tools as '- Name: tool_name'. This includes both
    ground-truth tools and distractor tools assigned to the case.
    """
    tool_names = set()
    for match in re.finditer(r'- Name:\s*(\w+)', system_content):
        tool_names.add(match.group(1))
    return tool_names


def prepare_eval_sample(record: dict) -> dict | None:
    """Extract evaluation input and ground truth from a conversation record.

    Returns:
        {
            "input_messages": [...],  # Messages up to first tool call
            "ground_truth_tools": [{"name": ..., "arguments": {...}}, ...],
            "ground_truth_diagnosis": "...",
            "gt_tool_findings": [{"name": ..., "arguments": {...}, "findings": ...}, ...],
            "available_tool_names": set(...),
        }
    """
    conversation = record.get("conversation", [])
    if not conversation:
        return None

    # Extract input messages (stop before first tool call)
    input_messages = []
    for msg in conversation:
        if msg["role"] == "system":
            input_messages.append(msg)
        elif msg["role"] == "user":
            input_messages.append(msg)
        elif msg["role"] == "assistant":
            if msg.get("tool_calls"):
                break
            else:
                input_messages.append(msg)
        elif msg["role"] == "tool":
            break

    # Build tool_call_id → findings mapping from tool response messages
    findings_by_id = {}
    for msg in conversation:
        if msg["role"] == "tool":
            findings_by_id[msg["tool_call_id"]] = msg.get("content", "")

    # Extract ground truth tools with their findings
    ground_truth_tools = []
    gt_tool_findings = []
    for msg in conversation:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name", "")
                arguments = json.loads(func.get("arguments", "{}"))
                ground_truth_tools.append({
                    "name": name,
                    "arguments": arguments,
                })
                gt_tool_findings.append({
                    "name": name,
                    "arguments": arguments,
                    "findings": findings_by_id.get(tc.get("id", ""), ""),
                })

    # Parse available tools from system prompt (includes GT + distractors)
    available_tool_names = set()
    for msg in conversation:
        if msg["role"] == "system":
            available_tool_names = parse_available_tools_from_system_prompt(
                msg.get("content", "")
            )
            break
    if not available_tool_names:
        available_tool_names = {t["name"] for t in ground_truth_tools}

    # Extract ground truth diagnosis
    ground_truth_diagnosis = record.get("diagnosis")
    if not ground_truth_diagnosis:
        for msg in conversation:
            content = msg.get("content") or ""
            diag = extract_diagnosis(content)
            if diag:
                ground_truth_diagnosis = diag
                break

    if not ground_truth_tools or not ground_truth_diagnosis:
        return None

    return {
        "input_messages": input_messages,
        "ground_truth_tools": ground_truth_tools,
        "ground_truth_diagnosis": ground_truth_diagnosis,
        "gt_tool_findings": gt_tool_findings,
        "available_tool_names": available_tool_names,
        "patient_id": record.get("patient_id"),
        "patient_kwargs": {
            "age": record.get("demographics", {}).get("age", ""),
            "gender": record.get("demographics", {}).get("gender", ""),
            "race": record.get("demographics", {}).get("race", ""),
            "additional_demographics": record.get("demographics", {}).get("additional_demographics", ""),
            "medical_history": record.get("medical_history", ""),
            "self_reported_symptoms": record.get("self_reported_symptoms", ""),
            "patient_persona": record.get("patient_persona"),
        } if any([
            record.get("demographics"),
            record.get("medical_history"),
            record.get("self_reported_symptoms"),
            record.get("patient_persona"),
        ]) else None,
    }


def prepare_diagnosis_only_sample(record: dict) -> dict | None:
    """Prepare a sample for diagnosis-only evaluation.

    Truncates the conversation right before the assistant message that
    contains [DIAGNOSIS:], so the model sees all prior context.
    """
    conversation = record.get("conversation", [])
    if not conversation:
        return None

    ground_truth_diagnosis = record.get("diagnosis")
    if not ground_truth_diagnosis:
        return None

    diag_idx = None
    for i, msg in enumerate(conversation):
        content = msg.get("content") or ""
        if msg["role"] == "assistant" and re.search(r'\[DIAGNOSIS:', content, re.IGNORECASE):
            diag_idx = i

    if diag_idx is None:
        return None

    input_messages = serialize_tool_calls(conversation[:diag_idx])
    if not input_messages:
        return None

    # Build tool_call_id -> findings mapping
    findings_by_id = {}
    for msg in conversation:
        if msg["role"] == "tool":
            findings_by_id[msg["tool_call_id"]] = msg.get("content", "")

    ground_truth_tools = []
    gt_tool_findings = []
    for msg in conversation:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func = tc.get("function", {})
                name = func.get("name", "")
                arguments = json.loads(func.get("arguments", "{}"))
                ground_truth_tools.append({
                    "name": name,
                    "arguments": arguments,
                })
                gt_tool_findings.append({
                    "name": name,
                    "arguments": arguments,
                    "findings": findings_by_id.get(tc.get("id", ""), ""),
                })

    available_tool_names = set()
    for msg in conversation:
        if msg["role"] == "system":
            available_tool_names = parse_available_tools_from_system_prompt(
                msg.get("content", "")
            )
            break
    if not available_tool_names:
        available_tool_names = {t["name"] for t in ground_truth_tools}

    return {
        "input_messages": input_messages,
        "ground_truth_tools": ground_truth_tools,
        "ground_truth_diagnosis": ground_truth_diagnosis,
        "gt_tool_findings": gt_tool_findings,
        "available_tool_names": available_tool_names,
        "patient_id": record.get("patient_id"),
    }


def prepare_diagnosis_eval_samples(
    records: list[dict], serialize: bool = True,
) -> list[dict]:
    """Prepare eval samples for diagnosis prediction.

    For each conversation, find the last assistant message containing
    [DIAGNOSIS: ...].  Input = all messages before that message (no prefill
    from the diagnosis turn).  The model must generate the diagnosis from
    scratch, optionally making tool calls along the way.

    Args:
        serialize: If True, inline tool_calls as <tool_call> tags (for local
            models). If False, keep native OpenAI format (for API models).
    """
    samples = []
    for record in records:
        conversation = record.get("conversation", [])
        if not conversation:
            continue

        diag_msg_idx = None
        ground_truth_diagnosis = None
        for i, msg in enumerate(conversation):
            if msg.get("role") == "assistant":
                content = msg.get("content", "") or ""
                diag = extract_diagnosis(content)
                if diag:
                    diag_msg_idx = i
                    ground_truth_diagnosis = diag

        if diag_msg_idx is None or not ground_truth_diagnosis:
            continue

        input_messages = list(conversation[:diag_msg_idx])
        if not input_messages:
            continue

        if serialize:
            input_messages = serialize_tool_calls(input_messages)

        # Build tool_call_id -> findings mapping from tool response messages
        findings_by_id = {}
        for msg in conversation:
            if msg["role"] == "tool":
                findings_by_id[msg["tool_call_id"]] = msg.get("content", "")

        # Extract ground truth tools with their findings
        ground_truth_tools = []
        gt_tool_findings = []
        for msg in conversation:
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    arguments = json.loads(func.get("arguments", "{}"))
                    ground_truth_tools.append({
                        "name": name,
                        "arguments": arguments,
                    })
                    gt_tool_findings.append({
                        "name": name,
                        "arguments": arguments,
                        "findings": findings_by_id.get(tc.get("id", ""), ""),
                    })

        # Parse available tools from system prompt (includes GT + distractors)
        available_tool_names = set()
        for msg in conversation:
            if msg["role"] == "system":
                available_tool_names = parse_available_tools_from_system_prompt(
                    msg.get("content", "")
                )
                break
        if not available_tool_names:
            available_tool_names = {t["name"] for t in ground_truth_tools}

        samples.append({
            "input_messages": input_messages,
            "ground_truth_diagnosis": ground_truth_diagnosis,
            "ground_truth_tools": ground_truth_tools,
            "gt_tool_findings": gt_tool_findings,
            "available_tool_names": available_tool_names,
            "gt_conversation": conversation,
        })

    return samples


def prepare_tool_call_eval_samples(
    records: list[dict], serialize: bool = True,
) -> list[dict]:
    """Prepare eval samples for tool call format prediction.

    For each assistant turn that contains tool calls, build a sample where:
    - prior_messages  = all messages before this assistant turn
    - text_prefix     = text content of this turn *before* any tool call
    - ground_truth_tool = the first tool call made in that turn

    Args:
        serialize: If True, inline tool_calls as <tool_call> tags (for local
            models). If False, keep native OpenAI format (for API models).
    """
    samples = []
    for record in records:
        conversation = record.get("conversation", [])
        if not conversation:
            continue

        available_tool_names = set()
        for msg in conversation:
            if msg["role"] == "system":
                available_tool_names = parse_available_tools_from_system_prompt(
                    msg.get("content", "")
                )
                break

        for i, msg in enumerate(conversation):
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                continue

            if serialize:
                prior_messages = serialize_tool_calls(list(conversation[:i]))
            else:
                prior_messages = list(conversation[:i])

            raw_content = msg.get("content", "") or ""
            text_prefix = re.sub(
                r'<tool_call>.*?</tool_call>', '', raw_content, flags=re.DOTALL
            ).strip()

            tc = tool_calls[0]
            func = tc.get("function", {})
            name = func.get("name", "")
            args_raw = func.get("arguments", "{}")
            try:
                arguments = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except (json.JSONDecodeError, TypeError):
                arguments = {}

            if not name:
                continue

            samples.append({
                "prior_messages": prior_messages,
                "text_prefix": text_prefix,
                "ground_truth_tool": {"name": name, "arguments": arguments},
                "available_tool_names": available_tool_names,
                "gt_conversation": conversation,
            })

    return samples


def prepare_conversation_eval_samples(
    records: list[dict],
) -> list[dict]:
    """Prepare eval samples for full conversation evaluation.

    Extracts patient profiles from records so the model starts a fresh
    consultation from scratch (no pre-existing conversation context).
    Only includes records with patient demographics/symptoms needed
    for the patient LLM.

    Each sample contains everything needed to run and score a full
    multi-turn consultation: patient info, available tools, ground truth
    tools/findings, and ground truth diagnosis.
    """
    from data.tool import TOOL_SCHEMA_MAP

    samples = []
    for record in records:
        base = prepare_eval_sample(record)
        if base is None:
            continue
        if base.get("patient_kwargs") is None:
            continue

        # Build tool info list for the doctor's system prompt
        available_tools = []
        for name in sorted(base["available_tool_names"]):
            schema = TOOL_SCHEMA_MAP.get(name)
            if schema:
                func = schema["function"]
                available_tools.append({
                    "name": func["name"],
                    "description": func["description"],
                    "parameters": func["parameters"],
                })

        if not available_tools:
            continue

        samples.append({
            "patient_kwargs": base["patient_kwargs"],
            "ground_truth_tools": base["ground_truth_tools"],
            "gt_tool_findings": base["gt_tool_findings"],
            "available_tool_names": list(base["available_tool_names"]),
            "available_tools": available_tools,
            "ground_truth_diagnosis": base["ground_truth_diagnosis"],
            "patient_id": base.get("patient_id"),
        })

    return samples


def load_eval_data_jsonl(
    data_path: str, begin: int = 0, end: int | None = None,
) -> list[dict]:
    """Load and prepare evaluation samples from JSONL file."""
    samples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample = prepare_eval_sample(record)
            if sample:
                samples.append(sample)
    return samples[begin:end]


def load_eval_data_csv(
    data_path: str, begin: int = 0, end: int | None = None,
) -> list[dict]:
    """Load evaluation samples from CSV patient profiles (full conversation mode)."""
    from train.rl.dataset import load_patient_profiles

    profiles = load_patient_profiles(data_path)
    profiles = profiles[begin:end]

    samples = []
    for profile in profiles:
        ground_truth_tools = []
        gt_tool_findings = []
        for exam_str, info in profile["exam_map"].items():
            parsed = info["parsed"]
            ground_truth_tools.append({
                "name": parsed["name"],
                "arguments": parsed["arguments"],
            })
            gt_tool_findings.append({
                "name": parsed["name"],
                "arguments": parsed["arguments"],
                "findings": info["findings"],
            })

        if not ground_truth_tools or not profile["diagnosis"]:
            continue

        samples.append({
            "input_messages": [
                {"role": "system", "content": profile["system_prompt"]},
            ],
            "ground_truth_tools": ground_truth_tools,
            "ground_truth_diagnosis": profile["diagnosis"],
            "gt_tool_findings": gt_tool_findings,
            "available_tool_names": set(profile["available_tools"]),
            "patient_kwargs": {
                "age": profile["demographics"]["age"],
                "gender": profile["demographics"]["gender"],
                "race": profile["demographics"]["race"],
                "additional_demographics": profile["demographics"]["additional_demographics"],
                "medical_history": profile["medical_history"],
                "self_reported_symptoms": profile["self_reported_symptoms"],
                "patient_persona": profile.get("patient_persona"),
            },
        })

    return samples


def load_eval_data(
    data_path: str,
    begin: int = 0,
    end: int | None = None,
    max_samples: int | None = None,
) -> list[dict]:
    """Load evaluation samples, dispatching by file extension."""
    if max_samples is not None and end is None:
        end = begin + max_samples

    if data_path.endswith(".csv"):
        return load_eval_data_csv(data_path, begin, end)
    else:
        return load_eval_data_jsonl(data_path, begin, end)


def load_hf_eval_data(
    dataset_name: str,
    max_samples: int | None = None,
    diagnosis_only: bool = False,
) -> list[dict]:
    """Load evaluation samples from a HuggingFace dataset."""
    from datasets import load_dataset
    ds = load_dataset(dataset_name, split="train")

    prepare_fn = prepare_diagnosis_only_sample if diagnosis_only else prepare_eval_sample
    samples = []
    for record in ds:
        record_dict = dict(record)
        sample = prepare_fn(record_dict)
        if sample:
            samples.append(sample)
        if max_samples and len(samples) >= max_samples:
            break
    return samples


def load_and_split_data(
    data_path: str,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Load data and split into train/eval sets."""
    records = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    random.seed(seed)
    shuffled = records.copy()
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - eval_ratio))
    train_records = shuffled[:split_idx]
    eval_records = shuffled[split_idx:]

    eval_samples = []
    for record in eval_records:
        sample = prepare_eval_sample(record)
        if sample:
            eval_samples.append(sample)

    return train_records, eval_samples
