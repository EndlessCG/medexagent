"""Model generation and tool call parsing utilities for evaluation."""

import json
import re
import time

import torch

from data.tool import TOOL_SCHEMAS
from data.prompt import (
    format_exam_notification_for_patient,
    get_agentclinic_patient_system_prompt,
    get_patient_system_prompt,
)

from .utils import _get_dist_info


def generate_response(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    debug_generate_heartbeat: bool = False,
    debug_label: str | None = None,
) -> str:
    """Generate a response from the model."""
    try:
        input_text = tokenizer.apply_chat_template(
            messages,
            tools=TOOL_SCHEMAS,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        input_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_tokens = inputs["input_ids"].shape[1]
    rank, world_size = _get_dist_info()
    hb_label = debug_label or "generate_response"
    start_time = time.perf_counter()
    if debug_generate_heartbeat:
        print(
            f"[EvalHB][rank {rank}/{world_size}] {hb_label} start "
            f"input_tokens={input_tokens} max_new_tokens={max_new_tokens}"
        )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            synced_gpus=(world_size > 1),
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    if debug_generate_heartbeat:
        elapsed = time.perf_counter() - start_time
        print(
            f"[EvalHB][rank {rank}/{world_size}] {hb_label} done "
            f"generated_tokens={new_tokens.shape[0]} elapsed_s={elapsed:.2f}"
        )
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def generate_continuation(
    model,
    tokenizer,
    messages: list[dict],
    prefix_text: str,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    debug_generate_heartbeat: bool = False,
    debug_label: str | None = None,
) -> str:
    """Generate a continuation from a partial assistant message.

    Applies the chat template to *messages* with add_generation_prompt=True,
    then appends *prefix_text* so the model continues from that partial output.
    """
    try:
        base_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        base_text = "\n".join(
            f"{m['role']}: {m.get('content', '')}" for m in messages
        ) + "\nassistant: "

    input_text = base_text + (prefix_text or "")
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_tokens = inputs["input_ids"].shape[1]
    rank, world_size = _get_dist_info()
    hb_label = debug_label or "generate_continuation"
    start_time = time.perf_counter()
    if debug_generate_heartbeat:
        print(
            f"[EvalHB][rank {rank}/{world_size}] {hb_label} start "
            f"input_tokens={input_tokens} max_new_tokens={max_new_tokens}"
        )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            synced_gpus=(world_size > 1),
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    if debug_generate_heartbeat:
        elapsed = time.perf_counter() - start_time
        print(
            f"[EvalHB][rank {rank}/{world_size}] {hb_label} done "
            f"generated_tokens={new_tokens.shape[0]} elapsed_s={elapsed:.2f}"
        )
    return tokenizer.decode(new_tokens, skip_special_tokens=False)


def extract_content_before_tool_call(output: str) -> str:
    """Extract the assistant's natural language content before any <tool_call> tag."""
    match = re.search(r'<tool_call>', output)
    if match:
        return output[:match.start()].strip()
    return output.strip()


def get_tool_response(
    predicted_tool: dict,
    gt_tool_findings: list[dict],
    available_tool_names: set[str],
) -> tuple[str, int | None]:
    """Determine the tool response for a predicted tool call.

    Logic:
    1. Tool not in available tools -> "This exam is not available."
    2. Exact match against remaining GT tool calls -> return real findings
    3. Tool in available tools but no GT match -> "No significant findings."
    """
    pred_name = predicted_tool.get("name", "")
    pred_args = predicted_tool.get("arguments", {})

    if pred_name not in available_tool_names:
        return "This exam is not available.", None

    for i, gt in enumerate(gt_tool_findings):
        if gt["name"] != pred_name:
            continue
        gt_args = gt.get("arguments", {})
        if set(gt_args.keys()) != set(pred_args.keys()):
            continue
        all_match = all(
            str(gt_args[k]).lower() == str(pred_args.get(k, "")).lower()
            for k in gt_args
        )
        if all_match:
            return gt["findings"], i

    return "No significant findings.", None


def parse_tool_calls_from_output(output: str) -> list[dict]:
    """Parse tool calls from model output (JSON or function call syntax)."""
    tool_calls = []

    json_pattern = re.findall(r'\{[^{}]*"name"\s*:\s*"([^"]+)"[^{}]*"(?:parameters|arguments)"\s*:\s*(\{[^{}]*\})[^{}]*\}', output)
    for name, args_str in json_pattern:
        try:
            args = json.loads(args_str)
            tool_calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            tool_calls.append({"name": name, "arguments": {}})

    if tool_calls:
        return tool_calls

    func_pattern = re.findall(r'(\w+)\(([^)]*)\)', output)
    known_names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    for func_name, args_str in func_pattern:
        if func_name in known_names:
            args = {}
            for match in re.finditer(r'(\w+)\s*=\s*"([^"]*)"', args_str):
                args[match.group(1)] = match.group(2)
            tool_calls.append({"name": func_name, "arguments": args})

    return tool_calls


def parse_tool_call_tag_from_output(output: str) -> list[dict]:
    """Parse tool calls from <tool_call>...</tool_call> tags in model output.

    Falls back to parse_tool_calls_from_output for other formats.
    """
    results = []

    for tag_match in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', output, re.DOTALL):
        try:
            obj = json.loads(tag_match.group(1))
            name = obj.get("name", "")
            args = obj.get("arguments", obj.get("parameters", {}))
            if name:
                results.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
        except (json.JSONDecodeError, ValueError):
            continue

    if results:
        return results

    return parse_tool_calls_from_output(output)


# ---------------------------------------------------------------------------
# LLM patient for full conversation mode
# ---------------------------------------------------------------------------

def _convert_to_patient_perspective(
    patient_system_prompt: str, messages: list[dict],
) -> list[dict]:
    """Convert doctor-perspective conversation to patient-perspective."""
    patient_messages = [{"role": "system", "content": patient_system_prompt}]

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            continue
        elif role == "assistant":
            if content:
                patient_messages.append({"role": "user", "content": content})
        elif role == "user":
            patient_messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            patient_messages.append({
                "role": "user",
                "content": "[The doctor has reviewed some test results.]",
            })

    return patient_messages


def get_patient_response(
    client, model: str, messages: list[dict], patient_kwargs: dict,
    max_retries: int = 3,
) -> str:
    """Get a patient response from an external LLM via OpenAI-compatible API."""
    kwargs = dict(patient_kwargs)
    patient_style = kwargs.pop("patient_style", "default")
    if patient_style == "agentclinic":
        # Drop persona/persona-only fields the AgentClinic prompt doesn't consume.
        for k in ("patient_persona", "include_persona"):
            kwargs.pop(k, None)
        patient_system_prompt = get_agentclinic_patient_system_prompt(**kwargs)
    else:
        patient_system_prompt = get_patient_system_prompt(**kwargs)
    patient_messages = _convert_to_patient_perspective(
        patient_system_prompt, messages,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=patient_messages,
                max_tokens=256,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Patient LLM request failed after {max_retries} attempts: {e}")
                return "I'm not sure what to say, Doctor."
