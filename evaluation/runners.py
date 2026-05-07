"""Evaluation runner functions."""

import json
from collections.abc import Awaitable, Callable

from tqdm import tqdm

try:
    import weave
except ImportError:
    weave = None

from .data import extract_diagnosis
from .generation import (
    generate_response,
    generate_continuation,
    get_tool_response,
    get_patient_response,
    parse_tool_calls_from_output,
    parse_tool_call_tag_from_output,
)
from .metrics import compute_tool_call_reward, compute_diagnosis_reward
from .trace_format import format_conversation, serialize_for_trace


async def eval_diagnosis_sample_async(
    *,
    messages: list[dict],
    ground_truth_diagnosis: str,
    available_tool_names,
    gt_tool_findings,
    gt_conversation,
    max_turns: int,
    generate_response_fn: Callable[[list[dict]], Awaitable[str]],
    diagnosis_reward_fn: Callable[[str | None, str], Awaitable[float]],
) -> dict:
    """Async diagnosis eval matching standalone multi-turn diagnosis semantics."""
    working_messages = [msg.copy() for msg in messages]
    gt_findings_remaining = [entry.copy() for entry in (gt_tool_findings or [])]
    available_names = available_tool_names or set()
    predicted = None
    call_counter = 0
    turns_used = 0
    tool_calls_made = []
    stop_reason = "max_turns"
    output = ""

    for turn in range(max_turns):
        turns_used = turn + 1
        output = await generate_response_fn(working_messages)
        diagnosis = extract_diagnosis(output)
        if diagnosis:
            predicted = diagnosis
            working_messages.append({"role": "assistant", "content": output})
            stop_reason = "diagnosis"
            break

        tool_calls = parse_tool_call_tag_from_output(output)
        if not tool_calls:
            stop_reason = "no_tool_no_diagnosis"
            break

        working_messages.append({"role": "assistant", "content": output})
        for tc in tool_calls:
            call_counter += 1
            tool_calls_made.append(tc.get("name", "?"))
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

    reward = await diagnosis_reward_fn(predicted, ground_truth_diagnosis)
    return {
        "output": output,
        "conversation": serialize_for_trace(working_messages),
        "conversation_text": format_conversation(working_messages),
        "ground_truth_conversation": serialize_for_trace(gt_conversation or []),
        "predicted_diagnosis": predicted,
        "ground_truth_diagnosis": ground_truth_diagnosis,
        "stop_reason": stop_reason,
        "turns_used": turns_used,
        "tool_calls_made": tool_calls_made,
        "diagnosis_reward": reward,
        "diagnosis_found": 1.0 if predicted is not None else 0.0,
    }


async def eval_tool_call_sample_async(
    *,
    messages: list[dict],
    text_prefix: str,
    ground_truth_tool: dict,
    available_tool_names,
    gt_conversation,
    generate_continuation_fn: Callable[[list[dict], str], Awaitable[str]],
) -> dict:
    """Async tool-call eval matching standalone tool-call semantics."""
    output = await generate_continuation_fn(messages, text_prefix)
    parsed_calls = parse_tool_call_tag_from_output(output)
    num_parsed_calls = len(parsed_calls)
    format_correct = 1.0 if num_parsed_calls == 1 else 0.0
    multiple_tool_calls = 1.0 if num_parsed_calls > 1 else 0.0

    reward = 0.0
    hallucinated = 0.0
    if num_parsed_calls == 1:
        reward = compute_tool_call_reward(parsed_calls, [ground_truth_tool])
        if available_tool_names:
            pred_name = parsed_calls[0].get("name", "")
            hallucinated = 0.0 if pred_name in set(available_tool_names) else 1.0

    return {
        "output": output,
        "conversation": serialize_for_trace(messages + [{"role": "assistant", "content": f"{text_prefix}{output}"}]),
        "conversation_text": format_conversation(
            messages + [{"role": "assistant", "content": f"{text_prefix}{output}"}]
        ),
        "ground_truth_conversation": serialize_for_trace(gt_conversation or []),
        "parsed_calls": parsed_calls,
        "num_parsed_calls": num_parsed_calls,
        "ground_truth_tool": ground_truth_tool,
        "tool_call_found": format_correct,
        "tool_call_format_correct": format_correct,
        "multiple_tool_calls": multiple_tool_calls,
        "tool_call_reward": reward,
        "tool_hallucinated": hallucinated,
    }


async def eval_conversation_sample_async(
    *,
    patient_kwargs: dict,
    available_tools: list[dict],
    gt_tool_findings,
    available_tool_names,
    ground_truth_diagnosis: str,
    ground_truth_tools,
    gt_conversation,
    max_turns: int,
    generate_response_fn: Callable[[list[dict]], Awaitable[str]],
    diagnosis_reward_fn: Callable[[str | None, str], Awaitable[float]],
    patient_response_fn: Callable[[list[dict], dict], Awaitable[str]],
) -> dict:
    """Async conversation eval matching standalone full-conversation semantics."""
    from data.prompt import get_doctor_system_prompt

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

    available_names = set(available_tool_names)
    findings_remaining = [f.copy() for f in gt_tool_findings]
    all_predicted_tools = []
    predicted_diagnosis = None
    call_counter = 0
    stop_reason = "max_turns"
    output = ""

    for _turn in range(max_turns):
        output = await generate_response_fn(messages)

        diagnosis = extract_diagnosis(output)
        if diagnosis:
            predicted_diagnosis = diagnosis
            messages.append({"role": "assistant", "content": output})
            stop_reason = "diagnosis"
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
            stop_reason = "tool_call"
            continue

        messages.append({"role": "assistant", "content": output})
        patient_reply = await patient_response_fn(messages, patient_kwargs)
        messages.append({"role": "user", "content": patient_reply})
        stop_reason = "patient_reply"

    diag_reward = await diagnosis_reward_fn(predicted_diagnosis, ground_truth_diagnosis)
    tool_reward = compute_tool_call_reward(all_predicted_tools, ground_truth_tools)
    hallucinated = sum(
        1 for tc in all_predicted_tools if tc.get("name", "") not in available_names
    )
    halluc_rate = hallucinated / len(all_predicted_tools) if all_predicted_tools else 0.0

    return {
        "output": output,
        "conversation": serialize_for_trace(messages),
        "conversation_text": format_conversation(messages),
        "ground_truth_conversation": serialize_for_trace(gt_conversation or []),
        "predicted_diagnosis": predicted_diagnosis,
        "predicted_tools": all_predicted_tools,
        "diagnosis_reward": diag_reward,
        "diagnosis_found": 1.0 if predicted_diagnosis is not None else 0.0,
        "tool_call_reward": tool_reward,
        "tool_hallucination_rate": halluc_rate,
        "num_turns": sum(1 for m in messages if m["role"] == "assistant"),
        "stop_reason": stop_reason,
    }


# ---------------------------------------------------------------------------
# Weave helpers
# ---------------------------------------------------------------------------

def _weave_op(name: str):
    """Decorator: wraps fn with weave.op(name=…) if weave is available."""
    def decorator(fn):
        if weave is not None:
            return weave.op(fn, name=name)
        return fn
    return decorator


def _call_weave_op(weave_fn, *args, **kwargs):
    """Call a weave-traced function; attach numeric results as feedback scores."""
    if weave is not None and hasattr(weave_fn, 'call'):
        result, call = weave_fn.call(*args, **kwargs)
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    call.feedback.add(key, {"value": value})
        return result
    return weave_fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# Traced per-sample eval functions
# ---------------------------------------------------------------------------

@_weave_op("diagnosis_eval")
def _eval_diagnosis_sample(
    model, tokenizer, messages, ground_truth_diagnosis,
    judge_model, judge_tokenizer, max_new_tokens,
    debug_generate_heartbeat=False, debug_label="",
    max_turns=15,
    available_tool_names=None,
    gt_tool_findings=None,
    gt_conversation=None,
):
    """Evaluate a single diagnosis sample. Traced by weave."""
    working_messages = [msg.copy() for msg in messages]
    gt_findings_remaining = [entry.copy() for entry in (gt_tool_findings or [])]
    available_names = available_tool_names or set()
    predicted = None
    call_counter = 0
    turns_used = 0
    tool_calls_made = []
    stop_reason = "max_turns"
    output = ""

    for turn in range(max_turns):
        turns_used = turn + 1
        output = generate_response(
            model, tokenizer, working_messages,
            max_new_tokens=max_new_tokens,
            debug_generate_heartbeat=debug_generate_heartbeat,
            debug_label=f"{debug_label} turn={turn}".strip(),
        )
        diagnosis = extract_diagnosis(output)
        if diagnosis:
            predicted = diagnosis
            working_messages.append({"role": "assistant", "content": output})
            stop_reason = "diagnosis"
            break

        tool_calls = parse_tool_call_tag_from_output(output)
        if not tool_calls:
            stop_reason = "no_tool_no_diagnosis"
            break

        working_messages.append({"role": "assistant", "content": output})
        for tc in tool_calls:
            call_counter += 1
            tool_calls_made.append(tc.get("name", "?"))
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

    diag = compute_diagnosis_reward(
        predicted, ground_truth_diagnosis, judge_model, judge_tokenizer,
        return_details=True,
    )
    return {
        "output": output,
        "conversation": serialize_for_trace(working_messages),
        "conversation_text": format_conversation(working_messages),
        "ground_truth_conversation": serialize_for_trace(gt_conversation or []),
        "predicted_diagnosis": predicted,
        "ground_truth_diagnosis": ground_truth_diagnosis,
        "stop_reason": stop_reason,
        "turns_used": turns_used,
        "tool_calls_made": tool_calls_made,
        "diagnosis_reward": diag["reward"],
        "diagnosis_accuracy_strict": diag["accuracy_strict"],
        "diagnosis_accuracy_lenient": diag["accuracy_lenient"],
        "diagnosis_found": 1.0 if predicted is not None else 0.0,
    }


@_weave_op("tool_call_eval")
def _eval_tool_call_sample(
    model, tokenizer, messages, text_prefix,
    ground_truth_tool, available_tool_names,
    max_new_tokens,
    debug_generate_heartbeat=False, debug_label="",
    gt_conversation=None,
):
    """Evaluate a single tool call sample. Traced by weave."""
    output = generate_continuation(
        model, tokenizer, messages, text_prefix,
        max_new_tokens=max_new_tokens,
        debug_generate_heartbeat=debug_generate_heartbeat,
        debug_label=debug_label,
    )
    parsed_calls = parse_tool_call_tag_from_output(output)
    num_parsed_calls = len(parsed_calls)
    format_correct = 1.0 if num_parsed_calls == 1 else 0.0
    multiple_tool_calls = 1.0 if num_parsed_calls > 1 else 0.0

    reward = 0.0
    hallucinated = 0.0
    if num_parsed_calls == 1:
        reward = compute_tool_call_reward(parsed_calls, [ground_truth_tool])
        if available_tool_names:
            pred_name = parsed_calls[0].get("name", "")
            hallucinated = 0.0 if pred_name in set(available_tool_names) else 1.0

    return {
        "output": output,
        "conversation": serialize_for_trace(messages + [{"role": "assistant", "content": f"{text_prefix}{output}"}]),
        "conversation_text": format_conversation(
            messages + [{"role": "assistant", "content": f"{text_prefix}{output}"}]
        ),
        "ground_truth_conversation": serialize_for_trace(gt_conversation or []),
        "parsed_calls": parsed_calls,
        "num_parsed_calls": num_parsed_calls,
        "ground_truth_tool": ground_truth_tool,
        "tool_call_found": format_correct,
        "tool_call_format_correct": format_correct,
        "multiple_tool_calls": multiple_tool_calls,
        "tool_call_reward": reward,
        "tool_hallucinated": hallucinated,
    }


@_weave_op("conversation_eval")
def _eval_conversation_sample(
    model,
    tokenizer,
    patient_kwargs,
    available_tools,
    gt_tool_findings,
    available_tool_names,
    ground_truth_diagnosis,
    ground_truth_tools,
    judge_model,
    judge_tokenizer,
    patient_client=None,
    patient_model: str | None = None,
    max_new_tokens: int = 4096,
    max_turns: int = 15,
    debug_generate_heartbeat: bool = False,
    debug_label: str = "",
    gt_conversation=None,
):
    """Evaluate a single full conversation sample. Traced by weave."""
    from data.prompt import get_doctor_system_prompt

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

    available_names = set(available_tool_names)
    findings_remaining = [f.copy() for f in gt_tool_findings]
    all_predicted_tools = []
    predicted_diagnosis = None
    call_counter = 0
    stop_reason = "max_turns"
    output = ""

    for turn in range(max_turns):
        output = generate_response(
            model, tokenizer, messages,
            max_new_tokens=max_new_tokens,
            debug_generate_heartbeat=debug_generate_heartbeat,
            debug_label=f"{debug_label} turn={turn}".strip(),
        )

        diagnosis = extract_diagnosis(output)
        if diagnosis:
            predicted_diagnosis = diagnosis
            messages.append({"role": "assistant", "content": output})
            stop_reason = "diagnosis"
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
            stop_reason = "tool_call"
            continue

        messages.append({"role": "assistant", "content": output})
        if patient_client:
            patient_reply = get_patient_response(
                patient_client, patient_model,
                messages, patient_kwargs,
            )
            messages.append({"role": "user", "content": patient_reply})
            stop_reason = "patient_reply"
        else:
            stop_reason = "no_patient_client"
            break

    diag = compute_diagnosis_reward(
        predicted_diagnosis, ground_truth_diagnosis, judge_model, judge_tokenizer,
        return_details=True,
    )
    tool_reward = compute_tool_call_reward(all_predicted_tools, ground_truth_tools)
    hallucinated = sum(
        1 for tc in all_predicted_tools if tc.get("name", "") not in available_names
    )
    halluc_rate = hallucinated / len(all_predicted_tools) if all_predicted_tools else 0.0

    return {
        "output": output,
        "conversation": serialize_for_trace(messages),
        "conversation_text": format_conversation(messages),
        "ground_truth_conversation": serialize_for_trace(gt_conversation or []),
        "predicted_diagnosis": predicted_diagnosis,
        "predicted_tools": all_predicted_tools,
        "diagnosis_reward": diag["reward"],
        "diagnosis_accuracy_strict": diag["accuracy_strict"],
        "diagnosis_accuracy_lenient": diag["accuracy_lenient"],
        "diagnosis_found": 1.0 if predicted_diagnosis is not None else 0.0,
        "tool_call_reward": tool_reward,
        "tool_hallucination_rate": halluc_rate,
        "num_turns": sum(1 for m in messages if m["role"] == "assistant"),
        "stop_reason": stop_reason,
    }


# ---------------------------------------------------------------------------
# Multi-turn evaluation (full conversation mode)
# ---------------------------------------------------------------------------

def run_evaluation(
    model,
    tokenizer,
    eval_data: list[dict],
    judge_model,
    judge_tokenizer,
    max_new_tokens: int = 4096,
    max_turns: int = 15,
    show_progress: bool = True,
    patient_client=None,
    patient_model: str | None = None,
) -> dict:
    """Run multi-turn evaluation on the given samples.

    For each sample, the model generates responses in a loop:
    - If the model outputs a tool call, we return the appropriate findings
      and let the model continue.
    - If the model outputs [DIAGNOSIS: ...], we stop and score.
    - If the model outputs neither:
      - Prefill mode (no patient_client): stop.
      - Full conversation mode: get LLM patient response and continue.

    Returns:
        {
            "tool_call_reward": float,   # mean reward in [0, 1]
            "diagnosis_reward": float,   # mean reward in [0, 1]
        }
    """
    tool_rewards = []
    diag_rewards = []
    acc_strict_list = []
    acc_lenient_list = []
    iterator = tqdm(eval_data, desc="Evaluating", leave=False) if show_progress else eval_data

    for sample in iterator:
        messages = [msg.copy() for msg in sample["input_messages"]]
        gt_findings_remaining = [entry.copy() for entry in sample["gt_tool_findings"]]
        available_names = sample["available_tool_names"]

        all_predicted_tools = []
        predicted_diagnosis = None
        call_counter = 0
        turn_bar = tqdm(range(max_turns), desc="  turns", leave=False) if show_progress else range(max_turns)

        for turn in turn_bar:
            output = generate_response(
                model, tokenizer, messages,
                max_new_tokens=max_new_tokens,
            )

            # Check for diagnosis
            diagnosis = extract_diagnosis(output)
            if diagnosis:
                predicted_diagnosis = diagnosis
                if show_progress and hasattr(turn_bar, 'set_postfix'):
                    turn_bar.set_postfix(status="diagnosed")
                break

            # Parse tool calls
            tool_calls = parse_tool_calls_from_output(output)
            if not tool_calls:
                if sample.get("patient_kwargs") is None or patient_client is None:
                    if show_progress and hasattr(turn_bar, 'set_postfix'):
                        turn_bar.set_postfix(status="no_tool_no_diag")
                    break
                messages.append({"role": "assistant", "content": output})
                patient_reply = get_patient_response(
                    patient_client, patient_model, messages,
                    sample["patient_kwargs"],
                )
                messages.append({"role": "user", "content": patient_reply})
                if show_progress and hasattr(turn_bar, 'set_postfix'):
                    turn_bar.set_postfix(status="patient_reply")
                continue

            all_predicted_tools.extend(tool_calls)
            if show_progress and hasattr(turn_bar, 'set_postfix'):
                names = ",".join(tc.get("name", "?") for tc in tool_calls)
                turn_bar.set_postfix(tools=names[:40])

            messages.append({
                "role": "assistant",
                "content": output,
            })

            for tc in tool_calls:
                call_counter += 1
                response_text, matched_idx = get_tool_response(
                    tc, gt_findings_remaining, available_names,
                )
                if matched_idx is not None:
                    gt_findings_remaining.pop(matched_idx)

                messages.append({
                    "role": "tool",
                    "tool_call_id": f"eval_{call_counter:04d}",
                    "content": response_text,
                })

        tool_reward = compute_tool_call_reward(
            all_predicted_tools,
            sample["ground_truth_tools"],
        )
        tool_rewards.append(tool_reward)

        diag = compute_diagnosis_reward(
            predicted_diagnosis,
            sample["ground_truth_diagnosis"],
            judge_model,
            judge_tokenizer,
            return_details=True,
        )
        diag_reward = diag["reward"]
        diag_rewards.append(diag_reward)
        acc_strict_list.append(diag["accuracy_strict"])
        acc_lenient_list.append(diag["accuracy_lenient"])
        if show_progress and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(
                tool_r=f"{tool_reward:.2f}",
                diag_r=f"{diag_reward:.2f}",
                pred=(predicted_diagnosis or "none")[:25],
            )
        print('Predicted diagnosis:', predicted_diagnosis)
        print('Ground truth diagnosis:', sample["ground_truth_diagnosis"])

    n = len(tool_rewards)
    if n == 0:
        return {
            "tool_call_reward": 0.0, "diagnosis_reward": 0.0,
            "diagnosis_accuracy_strict": 0.0, "diagnosis_accuracy_lenient": 0.0,
        }

    return {
        "tool_call_reward": sum(tool_rewards) / n,
        "diagnosis_reward": sum(diag_rewards) / n,
        "diagnosis_accuracy_strict": sum(acc_strict_list) / n,
        "diagnosis_accuracy_lenient": sum(acc_lenient_list) / n,
    }


# ---------------------------------------------------------------------------
# Diagnosis-only evaluation (multi-turn with tool calls)
# ---------------------------------------------------------------------------

def run_diagnosis_only_evaluation(
    model,
    tokenizer,
    eval_data: list[dict],
    judge_model,
    judge_tokenizer,
    max_new_tokens: int = 4096,
    max_turns: int = 15,
    show_progress: bool = True,
    verbose: bool = False,
    verbose_file=None,
) -> dict:
    """Run diagnosis-only evaluation with multi-turn tool call support.

    Returns:
        {
            "diagnosis_reward": float,
            "num_diagnosed": int,
            "num_total": int,
        }
    """
    def vprint(*a, **kw):
        if not verbose:
            return
        if verbose_file is not None:
            print(*a, **kw, file=verbose_file)
        else:
            print(*a, **kw)

    diag_rewards = []
    acc_strict_list = []
    acc_lenient_list = []
    num_diagnosed = 0
    iterator = tqdm(eval_data, desc="Evaluating (diagnosis-only)", leave=False) if show_progress else eval_data

    for i, sample in enumerate(iterator):
        messages = [msg.copy() for msg in sample["input_messages"]]
        gt_findings_remaining = [entry.copy() for entry in sample["gt_tool_findings"]]
        available_names = sample["available_tool_names"]

        vprint(f"\n=== Sample {i} (patient_id={sample.get('patient_id')}) ===")
        vprint(f"GT diagnosis: {sample['ground_truth_diagnosis']}")
        vprint(f"GT tools: {[t['name'] for t in sample['ground_truth_tools']]}")
        vprint(f"Available tools: {available_names}")
        vprint(f"\n--- Input conversation ({len(messages)} messages) ---")
        if verbose:
            for j, msg in enumerate(messages):
                role = msg["role"]
                content = (msg.get("content") or "")
                tc = msg.get("tool_calls")
                vprint(f"  [{j}] {role}: {content[:200]}{'...' if len(content) > 200 else ''}")
                if tc:
                    vprint(f"       tool_calls: {tc}")
            vprint()

        predicted_diagnosis = None
        call_counter = 0

        for turn in range(max_turns):
            output = generate_response(
                model, tokenizer, messages,
                max_new_tokens=max_new_tokens,
            )

            vprint(f"--- Turn {turn} ---")
            vprint(f"Model output: {output}")

            diagnosis = extract_diagnosis(output)
            if diagnosis:
                predicted_diagnosis = diagnosis
                vprint(f"DIAGNOSIS found: {diagnosis}")
                break

            tool_calls = parse_tool_calls_from_output(output)
            if not tool_calls:
                vprint("No tool call and no diagnosis, stopping.")
                break

            messages.append({"role": "assistant", "content": output})

            for tc in tool_calls:
                call_counter += 1
                response_text, matched_idx = get_tool_response(
                    tc, gt_findings_remaining, available_names,
                )
                if matched_idx is not None:
                    gt_findings_remaining.pop(matched_idx)

                vprint(f"Tool call: {tc['name']}({tc.get('arguments', {})}) -> {response_text[:150]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": f"eval_{call_counter:04d}",
                    "content": response_text,
                })

            vprint()

        if predicted_diagnosis:
            num_diagnosed += 1

        diag = compute_diagnosis_reward(
            predicted_diagnosis,
            sample["ground_truth_diagnosis"],
            judge_model,
            judge_tokenizer,
            return_details=True,
        )
        diag_reward = diag["reward"]
        diag_rewards.append(diag_reward)
        acc_strict_list.append(diag["accuracy_strict"])
        acc_lenient_list.append(diag["accuracy_lenient"])

        vprint(f"Predicted diagnosis: {predicted_diagnosis}")
        vprint(f"Diagnosis reward: {diag_reward:.3f}")
        vprint("=" * 60)

    n = len(diag_rewards)
    if n == 0:
        return {
            "diagnosis_reward": 0.0,
            "diagnosis_accuracy_strict": 0.0,
            "diagnosis_accuracy_lenient": 0.0,
            "num_diagnosed": 0, "num_total": 0,
        }

    return {
        "diagnosis_reward": sum(diag_rewards) / n,
        "diagnosis_accuracy_strict": sum(acc_strict_list) / n,
        "diagnosis_accuracy_lenient": sum(acc_lenient_list) / n,
        "num_diagnosed": num_diagnosed,
        "num_total": n,
    }


# ---------------------------------------------------------------------------
# Lightweight eval (used by PeriodicEvaluationCallback)
# ---------------------------------------------------------------------------

def run_diagnosis_eval(
    model,
    tokenizer,
    samples: list[dict],
    judge_model,
    judge_tokenizer,
    max_new_tokens: int = 4096,
    max_turns: int = 15,
    show_progress: bool = False,
    output_file: str | None = None,
    debug_generate_heartbeat: bool = False,
    debug_prefix: str = "",
) -> dict:
    """Evaluate model's ability to predict diagnosis (multi-turn with tool calls).

    For each sample the model generates responses in a loop:
    - If the model outputs a tool call, execute it against GT findings and continue.
    - If the model outputs [DIAGNOSIS: ...], stop and score.
    - Otherwise, stop.

    Returns:
        {"diagnosis_reward": float, "diagnosis_found_rate": float}
    """
    if not samples:
        return {"diagnosis_reward": 0.0, "diagnosis_effective_reward": 0.0, "diagnosis_found_rate": 0.0}

    total = len(samples)
    diag_rewards = []
    diag_found_rewards = []
    found_count = 0
    iterator = tqdm(samples, desc="Diagnosis eval", leave=False) if show_progress else samples

    for idx, sample in enumerate(iterator):
        hb_label = f"{debug_prefix} diag {idx + 1}/{total}".strip()
        result = _call_weave_op(
            _eval_diagnosis_sample,
            model,
            tokenizer,
            sample["input_messages"],
            sample["ground_truth_diagnosis"],
            judge_model,
            judge_tokenizer,
            max_new_tokens,
            debug_generate_heartbeat=debug_generate_heartbeat,
            debug_label=hb_label,
            max_turns=max_turns,
            available_tool_names=sample.get("available_tool_names", set()),
            gt_tool_findings=sample.get("gt_tool_findings", []),
            gt_conversation=sample.get("gt_conversation"),
        )

        predicted_diagnosis = result["predicted_diagnosis"]
        stop_reason = result["stop_reason"]
        turns_used = result["turns_used"]
        tool_calls_made = result["tool_calls_made"]
        last_output = result["output"]

        reward = result["diagnosis_reward"]

        if predicted_diagnosis is not None:
            found_count += 1
            diag_found_rewards.append(reward)
        diag_rewards.append(reward)

        if output_file is not None:
            with open(output_file, "a") as f:
                f.write(json.dumps({
                    "gt_diagnosis": sample["ground_truth_diagnosis"],
                    "predicted_diagnosis": predicted_diagnosis,
                    "stop_reason": stop_reason,
                    "turns_used": turns_used,
                    "tool_calls_made": tool_calls_made,
                    "last_output": last_output[:500],
                    "gt_conversation": sample.get("gt_conversation", []),
                    "diagnosis_reward": reward,
                    "diagnosis_found": predicted_diagnosis is not None,
                }, ensure_ascii=False) + "\n")

        if show_progress and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(
                reward=f"{reward:.2f}",
                pred=(predicted_diagnosis or "none")[:25],
            )

    n = len(diag_rewards)
    return {
        "diagnosis_reward": sum(diag_rewards) / n if n else 0.0,
        "diagnosis_effective_reward": sum(diag_found_rewards) / len(diag_found_rewards) if diag_found_rewards else 0.0,
        "diagnosis_found_rate": found_count / n if n else 0.0,
    }


def run_tool_call_format_eval(
    model,
    tokenizer,
    samples: list[dict],
    max_new_tokens: int = 4096,
    show_progress: bool = False,
    output_file: str | None = None,
    debug_generate_heartbeat: bool = False,
    debug_prefix: str = "",
    num_real: int | None = None,
) -> dict:
    """Evaluate model's ability to generate a correctly-formatted tool call.

    Uses weave tracing via _call_weave_op when weave is available,
    attaching per-sample scores as feedback.

    Args:
        num_real: If set, only the first *num_real* samples count towards
            metrics and output.  Remaining samples are still evaluated
            (needed for FSDP sync) but their results are discarded.

    Returns:
        {"tool_call_format_rate": float, "tool_call_reward": float,
         "tool_hallucination_rate": float, "multiple_tool_call_rate": float}
    """
    if not samples:
        return {"tool_call_format_rate": 0.0, "tool_call_reward": 0.0,
                "tool_hallucination_rate": 0.0, "multiple_tool_call_rate": 0.0}

    if num_real is None:
        num_real = len(samples)

    total = len(samples)
    format_scores = []
    tool_rewards = []
    hallucination_scores = []
    multiple_tool_scores = []
    iterator = tqdm(samples, desc="Tool call eval", leave=False) if show_progress else samples

    for idx, sample in enumerate(iterator):
        hb_label = f"{debug_prefix} tool {idx + 1}/{total}".strip()

        result = _call_weave_op(
            _eval_tool_call_sample,
            model, tokenizer,
            sample["prior_messages"],
            sample["text_prefix"],
            sample["ground_truth_tool"],
            sample.get("available_tool_names", set()),
            max_new_tokens,
            debug_generate_heartbeat=debug_generate_heartbeat,
            debug_label=hb_label,
            gt_conversation=sample.get("gt_conversation"),
        )

        # Skip padding samples for metrics / output.
        if idx >= num_real:
            continue

        parsed_calls = result["parsed_calls"]
        num_parsed_calls = result.get("num_parsed_calls", len(parsed_calls))
        reward = result["tool_call_reward"]
        hallucinated = result["tool_hallucinated"]
        format_correct = (num_parsed_calls == 1)
        multiple_tool = (num_parsed_calls > 1)

        format_scores.append(1.0 if format_correct else 0.0)
        multiple_tool_scores.append(1.0 if multiple_tool else 0.0)

        if format_correct:
            tool_rewards.append(reward)
            if sample.get("available_tool_names"):
                hallucination_scores.append(hallucinated)

        if output_file is not None:
            with open(output_file, "a") as f:
                f.write(json.dumps({
                    "gt_tool": sample["ground_truth_tool"],
                    "predicted_tool": parsed_calls[0] if format_correct else None,
                    "predicted_tools": parsed_calls,
                    "gt_conversation": sample.get("gt_conversation", []),
                    "model_output": result["output"],
                    "tool_call_format_found": bool(format_correct),
                    "tool_call_format_correct": bool(format_correct),
                    "num_parsed_calls": num_parsed_calls,
                    "multiple_tool_calls": bool(multiple_tool),
                    "tool_call_reward": reward,
                    "tool_hallucinated": bool(hallucinated),
                }, ensure_ascii=False) + "\n")

    n = len(format_scores)
    if n == 0:
        return {"tool_call_format_rate": 0.0, "tool_call_reward": 0.0,
                "tool_call_effective_reward": 0.0,
                "tool_hallucination_rate": 0.0, "multiple_tool_call_rate": 0.0}
    tool_eff = sum(tool_rewards) / len(tool_rewards) if tool_rewards else 0.0
    return {
        "tool_call_format_rate": sum(format_scores) / n,
        "tool_call_reward": tool_eff,
        "tool_call_effective_reward": tool_eff,
        "tool_hallucination_rate": sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else 0.0,
        "multiple_tool_call_rate": sum(multiple_tool_scores) / n,
    }


def run_conversation_eval(
    model,
    tokenizer,
    samples: list[dict],
    judge_model,
    judge_tokenizer,
    patient_client=None,
    patient_model: str | None = None,
    max_new_tokens: int = 4096,
    max_turns: int = 15,
    show_progress: bool = False,
    output_file: str | None = None,
    debug_generate_heartbeat: bool = False,
    debug_prefix: str = "",
) -> dict:
    """Run full conversation evaluation (from scratch with LLM patient).

    For each sample the model starts a fresh consultation: greeting,
    symptom inquiry, tool calls (executed against GT findings), and
    diagnosis.  Patient responses come from an external LLM.

    All ranks must run the same samples (no splitting) so that every rank
    makes the same number of ``generate()`` calls — turn counts depend on
    model output which is identical across ranks given the same input and
    weights.  Only rank 0 should use the returned metrics.

    Returns:
        {"diagnosis_reward": float, "diagnosis_found_rate": float,
         "tool_call_reward": float, "tool_hallucination_rate": float}
    """
    empty = {
        "diagnosis_reward": 0.0, "diagnosis_effective_reward": 0.0,
        "diagnosis_found_rate": 0.0,
        "tool_call_reward": 0.0, "tool_hallucination_rate": 0.0,
    }
    if not samples:
        return empty

    diag_rewards = []
    diag_found_rewards = []
    tool_rewards = []
    halluc_scores = []
    found_count = 0
    total = len(samples)
    iterator = tqdm(samples, desc="Conversation eval", leave=False) if show_progress else samples

    for idx, sample in enumerate(iterator):
        hb_prefix = f"{debug_prefix} conv {idx + 1}/{total}".strip()
        result = _call_weave_op(
            _eval_conversation_sample,
            model,
            tokenizer,
            sample["patient_kwargs"],
            sample["available_tools"],
            sample["gt_tool_findings"],
            sample["available_tool_names"],
            sample["ground_truth_diagnosis"],
            sample["ground_truth_tools"],
            judge_model,
            judge_tokenizer,
            patient_client=patient_client,
            patient_model=patient_model,
            max_new_tokens=max_new_tokens,
            max_turns=max_turns,
            debug_generate_heartbeat=debug_generate_heartbeat,
            debug_label=hb_prefix,
            gt_conversation=sample.get("gt_conversation"),
        )

        predicted_diagnosis = result["predicted_diagnosis"]
        all_predicted_tools = result["predicted_tools"]
        diag_reward = result["diagnosis_reward"]
        tool_reward = result["tool_call_reward"]
        halluc_rate = result["tool_hallucination_rate"]
        messages = result["conversation"]

        diag_rewards.append(diag_reward)
        tool_rewards.append(tool_reward)
        halluc_scores.append(halluc_rate)
        if predicted_diagnosis is not None:
            found_count += 1
            diag_found_rewards.append(diag_reward)

        if output_file is not None:
            with open(output_file, "a") as f:
                f.write(json.dumps({
                    "gt_diagnosis": sample["ground_truth_diagnosis"],
                    "predicted_diagnosis": predicted_diagnosis,
                    "gt_tools": sample["ground_truth_tools"],
                    "predicted_tools": all_predicted_tools,
                    "conversation": messages,
                    "diagnosis_reward": diag_reward,
                    "tool_call_reward": tool_reward,
                    "tool_hallucination_rate": halluc_rate,
                    "num_turns": sum(1 for m in messages if m["role"] == "assistant"),
                }, ensure_ascii=False) + "\n")

        if show_progress and hasattr(iterator, 'set_postfix'):
            iterator.set_postfix(
                diag_r=f"{diag_reward:.2f}",
                tool_r=f"{tool_reward:.2f}",
                pred=(predicted_diagnosis or "none")[:25],
            )

    n = len(diag_rewards)
    if n == 0:
        return empty
    return {
        "diagnosis_reward": sum(diag_rewards) / n,
        "diagnosis_effective_reward": sum(diag_found_rewards) / len(diag_found_rewards) if diag_found_rewards else 0.0,
        "diagnosis_found_rate": found_count / n,
        "tool_call_reward": sum(tool_rewards) / n,
        "tool_hallucination_rate": sum(halluc_scores) / n,
    }
