"""
Conversation generation pipeline.

Generates synthetic doctor-patient diagnostic conversations from
structured EHR data. The doctor and patient are both LLM-driven,
with tool calls (exams) scripted deterministically from the dataset.

Usage:
    python -m data.pipeline generate --total-size 5
    python -m data.generate --input data/ehr/pmc_patients_post_processed.csv --output conversations.jsonl
"""

import argparse
import csv
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI, RateLimitError

from data.utils import rate_limit_sleep
from tqdm import tqdm

from data.tool import (
    TOOL_SCHEMAS,
    format_findings,
    parse_exam_string,
    select_tools_for_case,
)
from data.prompt import (
    get_doctor_base_prompt,
    get_doctor_phase_prompt,
    get_doctor_system_prompt,
    get_doctor_training_prompt,
    get_patient_system_prompt,
    sample_patient_persona,
)
from data.noise import (
    NoiseConfig,
    PATIENT_NOISE_TYPES,
    EXAM_NOISE_TYPES,
    check_patient_noise_eligibility,
    check_exam_noise_eligibility,
    plan_patient_noise,
    plan_dataset_noise_assignments,
    get_patient_turn_hint,
    apply_exam_noise,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT = "data/ehr/pmc_patients_post_processed.csv"
DEFAULT_OUTPUT = "data/conversations/conversations.jsonl"
DEFAULT_CHECKPOINT = "data/conversations/checkpoint.json"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
# DEFAULT_BASE_URL = "http://localhost:8000/v1" 
# DEFAULT_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_MAX_WORKERS = 64
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_TURNS = 99  # safety limit on total conversation turns
DEFAULT_MAX_INQUIRY_TURNS = 10  # max turns for symptom inquiry before forcing exam transition
MAX_RETRIES = 3
DEFAULT_DIAGNOSIS_MATCH_RETRIES = 10
DEFAULT_DIAGNOSIS_ALIASES_PATH = "config/diagnosis_aliases.json"

# Sentinel returned by process_patient when an exception occurs (as opposed to
# intentional None, which means the conversation was filtered out on purpose).
_ERROR_SENTINEL = object()


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def llm_call(
    client: OpenAI,
    model: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """Make an LLM chat completion call with retries.

    Rate-limit errors (429) are retried indefinitely, respecting the
    ``retry-after`` header from the server.  Other errors are retried up to
    ``max_retries`` times with exponential back-off.

    Returns the assistant message as a dict with 'content' and optionally 'tool_calls'.
    """
    attempt = 0
    while True:
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
            }
            if tools:
                kwargs["tools"] = tools

            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message

            result = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            return result
        except RateLimitError as e:
            rate_limit_sleep(e, attempt)
            attempt += 1
            # Never give up on rate limits — just keep waiting
        except Exception as e:
            attempt += 1
            if attempt >= max_retries:
                raise RuntimeError(f"LLM call failed after {attempt} attempts: {e}") from e
            time.sleep(2 ** (attempt - 1))


# ---------------------------------------------------------------------------
# Patient data extraction
# ---------------------------------------------------------------------------

def extract_exams(row: dict) -> list[tuple[str, str]]:
    """Extract (exam_string, findings) pairs from a CSV row, skipping empty slots."""
    exams = []
    for i in range(1, 11):
        exam = (row.get(f"exam{i}") or "").strip()
        findings = (row.get(f"exam{i}_findings") or "").strip()
        if exam:
            exams.append((exam, findings))
    return exams


def parse_patient_row(row: dict, idx: int) -> dict:
    """Parse a CSV row into a structured patient dict."""
    return {
        "patient_id": idx,
        "age": (row.get("age") or "").strip(),
        "gender": (row.get("gender") or "").strip(),
        "race": (row.get("race") or "").strip(),
        "additional_demographics": (row.get("additional_demographics") or "").strip(),
        "medical_history": (row.get("medical_history") or "").strip(),
        "diagnosis": (row.get("diagnosis") or "").strip(),
        "self_reported_symptoms": (row.get("self_reported_symptoms") or "").strip(),
        "exams": extract_exams(row),
    }


# ---------------------------------------------------------------------------
# Noise pre-assignment (dataset-level)
# ---------------------------------------------------------------------------
# Implementation lives in data/noise.py; kept as an alias for callers that
# still reference the underscored name.

_plan_noise_assignments = plan_dataset_noise_assignments


# ---------------------------------------------------------------------------
# Conversation generation (per patient)
# ---------------------------------------------------------------------------

_EXAM_HUMAN_NAMES = {
    "xray": lambda a: f"an X-ray of the {a.get('region', 'body')}",
    "ct": lambda a: f"a CT scan of the {a.get('region', 'body')}" + (" with contrast" if a.get("contrast") == "yes" else ""),
    "mri": lambda a: f"an MRI of the {a.get('region', 'body')}",
    "ultrasound": lambda a: f"an ultrasound of the {a.get('region', 'body')}",
    "blood_test": lambda a: f"a blood test ({a.get('test', 'general panel')})",
    "cbc": lambda _: "a complete blood count",
    "bmp": lambda _: "a basic metabolic panel",
    "cmp": lambda _: "a comprehensive metabolic panel",
    "ecg": lambda _: "an electrocardiogram (ECG)",
    "echocardiogram": lambda _: "an echocardiogram",
    "eeg": lambda _: "an EEG",
    "endoscopy": lambda a: f"an endoscopy ({a.get('type', 'upper')})",
    "biopsy": lambda a: f"a biopsy of the {a.get('site', 'tissue')}",
    "lumbar_puncture": lambda _: "a lumbar puncture",
    "culture": lambda a: f"a culture from {a.get('source', 'a sample')}",
    "pet_scan": lambda _: "a PET scan",
    "stress_test": lambda _: "a cardiac stress test",
    "pulmonary_function_test": lambda _: "pulmonary function testing",
    "bone_density_scan": lambda _: "a bone density scan",
    "mammogram": lambda _: "a mammogram",
    "colonoscopy": lambda _: "a colonoscopy",
    "bronchoscopy": lambda _: "a bronchoscopy",
    "angiography": lambda a: f"angiography of the {a.get('region', 'body')}",
    "emg": lambda _: "an electromyography (EMG)",
    "spirometry": lambda _: "spirometry",
    "allergy_test": lambda _: "allergy testing",
    "audiometry": lambda _: "an audiometry test",
    "visual_acuity_test": lambda _: "a visual acuity test",
    "fundoscopy": lambda _: "a fundoscopy",
    "slit_lamp_exam": lambda _: "a slit lamp examination",
    "tonometry": lambda _: "tonometry",
    "blood_gas": lambda a: f"{'an arterial' if a.get('type') == 'arterial' else 'a venous'} blood gas",
    "urinalysis": lambda _: "a urinalysis",
    "coagulation_panel": lambda _: "a coagulation panel",
    "liver_function_test": lambda _: "liver function tests",
    "thyroid_function_test": lambda _: "thyroid function tests",
    "lipid_panel": lambda _: "a lipid panel",
    "hemoglobin_a1c": lambda _: "a hemoglobin A1c test",
    "troponin": lambda _: "a troponin test",
    "d_dimer": lambda _: "a D-dimer test",
    "procalcitonin": lambda _: "a procalcitonin test",
    "crp": lambda _: "a C-reactive protein test",
    "esr": lambda _: "an erythrocyte sedimentation rate test",
    "bnp": lambda _: "a BNP test",
}


def _humanize_exam_name(exam_str: str) -> str:
    """Convert raw exam string to human-readable description."""
    parsed = parse_exam_string(exam_str)
    if parsed is None:
        name = exam_str.split("(")[0].replace("_", " ") if "(" in exam_str else exam_str.replace("_", " ")
        return f"a {name}"
    func = _EXAM_HUMAN_NAMES.get(parsed["name"])
    if func:
        return func(parsed["arguments"])
    return f"a {parsed['name'].replace('_', ' ')}"


def _llm_is_clean(client: OpenAI, model: str, content: str, phase: str) -> bool:
    """Use the LLM to check if the doctor's text is appropriate for the phase.

    Asks the model a yes/no question about whether the text contains
    ordering language or references to exam results.  Falls back to True
    (assume clean) if the call fails for any reason.
    """
    resp = llm_call(client, model, [
        {
            "role": "system",
            "content": (
                "You are a strict content classifier. Answer only YES or NO.\n"
                "Does the following doctor message do ANY of these?\n"
                "- Mention ordering, running, scheduling, or performing any examination, test, scan, or lab work\n"
                "- Promise or suggest that exams/tests will be done (e.g. 'let me check', 'we should look into', 'I'll run')\n"
                "- Reference results from any examination or test\n"
                "Answer YES if it does any of the above, NO if it only asks about symptoms/history/lifestyle."
            ),
        },
        {"role": "user", "content": content},
    ])
    answer = (resp.get("content") or "").strip().upper()
    # Accept anything starting with "NO" as clean
    return answer.startswith("NO")


def _doctor_turn_is_clean(
    content: str,
    phase: str,
    client: OpenAI,
    model: str,
) -> bool:
    """Check if doctor's text is appropriate for the current phase."""
    if not content:
        return True
    if phase not in ("greeting", "inquiry"):
        return True
    return _llm_is_clean(client, model, content, phase)


_DIAGNOSIS_TAG_RE = re.compile(r"\[DIAGNOSIS:\s*(.*?)\]", re.IGNORECASE | re.DOTALL)


def extract_diagnosis_tag(content: str) -> str | None:
    """Extract the diagnosis text from a [DIAGNOSIS: ...] tag."""
    if not content:
        return None
    match = _DIAGNOSIS_TAG_RE.search(content)
    if not match:
        return None
    diagnosis = " ".join(match.group(1).split()).strip()
    return diagnosis or None


def _normalize_diagnosis_text(text: str) -> str:
    return " ".join((text or "").split()).strip().casefold()


_PAREN_CONTENT_RE = re.compile(r"\(([^)]*)\)")


def _canonicalize_diagnosis_delimiters(text: str) -> str:
    """Normalize common multi-diagnosis delimiters without changing word order."""
    normalized = _normalize_diagnosis_text(text)
    normalized = normalized.replace(";", ",")
    normalized = re.sub(r"\s+(?:and|&)\s+", ", ", normalized)
    normalized = re.sub(r"\s*,\s*", ", ", normalized)
    normalized = re.sub(r",\s*,+", ", ", normalized)
    return normalized.strip(" ,")


def _diagnosis_variants(text: str) -> set[str]:
    """Build lenient variants for diagnosis matching.

    This is intentionally conservative: it handles parenthetical expansions and
    separator differences that appear frequently in PMC labels, without trying
    to infer semantic equivalence from unrelated diagnoses.
    """
    raw = (text or "").strip()
    if not raw:
        return set()

    variants = {
        _normalize_diagnosis_text(raw),
        _canonicalize_diagnosis_delimiters(raw),
    }

    # Remove parenthetical content, e.g. "acute liver failure (ALF)".
    no_paren = re.sub(r"\s*\([^)]*\)", "", raw).strip()
    if no_paren:
        variants.add(_normalize_diagnosis_text(no_paren))
        variants.add(_canonicalize_diagnosis_delimiters(no_paren))

    # Add parenthetical expansions themselves, e.g. "DWS (Dandy-Walker Syndrome)".
    for match in _PAREN_CONTENT_RE.finditer(raw):
        inside = match.group(1).strip()
        if inside:
            variants.add(_normalize_diagnosis_text(inside))
            variants.add(_canonicalize_diagnosis_delimiters(inside))

    return {v for v in variants if v}


def load_diagnosis_aliases(path: str) -> dict[str, set[str]]:
    """Load manually approved diagnosis equivalences from JSON."""
    if not path or not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    aliases: dict[str, set[str]] = {}
    for gold, accepted in raw.items():
        norm_gold = _normalize_diagnosis_text(gold)
        values = set(_diagnosis_variants(gold))
        if isinstance(accepted, list):
            for item in accepted:
                values.update(_diagnosis_variants(item))
        aliases[norm_gold] = {v for v in values if v}
    return aliases


def diagnosis_matches_gold(
    content: str,
    gold_diagnosis: str,
    approved_aliases: dict[str, set[str]] | None = None,
) -> bool:
    """Return True when the final diagnosis tag matches the gold label exactly."""
    predicted = extract_diagnosis_tag(content)
    if predicted is None:
        return False
    norm_gold = _normalize_diagnosis_text(gold_diagnosis)
    gold_variants = _diagnosis_variants(gold_diagnosis)
    pred_variants = _diagnosis_variants(predicted)
    if pred_variants & gold_variants:
        return True
    if approved_aliases and pred_variants & approved_aliases.get(norm_gold, set()):
        return True

    # Fallback: accept if the first comma-separated item in the predicted diagnosis
    # matches the gold label or one of its aliases (LLM often appends secondary
    # findings to the primary diagnosis).
    norm_pred = _canonicalize_diagnosis_delimiters(predicted)
    first_pred = norm_pred.split(",")[0].strip()
    if first_pred and first_pred != norm_pred:
        if first_pred in gold_variants:
            return True
        if approved_aliases and first_pred in approved_aliases.get(norm_gold, set()):
            return True
    return False


def _generate_doctor_turn(client, model, doctor_messages, phase, max_regen=10):
    """Generate a doctor turn with post-validation. Retries with correction hint if dirty."""
    for attempt in range(max_regen):
        msg = llm_call(client, model, doctor_messages)
        content = msg.get("content") or ""
        if _doctor_turn_is_clean(content, phase, client, model):
            return msg
        if attempt < max_regen - 1:
            correction = {
                "role": "system",
                "content": (
                    "CORRECTION: Your previous response mentioned ordering tests or "
                    "referenced test results. In this phase, you should ONLY ask "
                    "about symptoms and medical history. Do NOT mention any exams, "
                    "tests, results, or ordering. Try again."
                ),
            }
            doctor_messages.append(correction)
    return None  # all attempts unclean — caller should drop this data point


def generate_conversation(
    patient: dict,
    client: OpenAI,
    model: str,
    max_turns: int = DEFAULT_MAX_TURNS,
    max_inquiry_turns: int = DEFAULT_MAX_INQUIRY_TURNS,
    noise_config: NoiseConfig | None = None,
    patient_noise_types: set[str] | None = None,
    exam_noise_plan: dict[int, str] | None = None,
) -> dict:
    """Generate a full doctor-patient conversation for one patient.

    Returns a dict ready to be written as one JSONL line.
    """
    exams = patient["exams"]

    # Plan noise for this conversation
    noise_cfg = noise_config or NoiseConfig()
    rng = random.Random(
        noise_cfg.seed + patient["patient_id"] if noise_cfg.seed is not None else None
    )
    patient_noise_spec = plan_patient_noise(
        patient["self_reported_symptoms"],
        noise_cfg.patient_noise_level if patient_noise_types is None else 0.0,
        rng,
        forced_noise_types=patient_noise_types,
    )
    max_planned_patient_turn = max(patient_noise_spec.noise_turns.values(), default=-1)
    patient_turn_index = 0
    exam_noise_records = []
    exam_strings = [e[0] for e in exams]

    # Extract tool names from exam strings for tool selection
    ground_truth_tool_names = []
    for exam_str in exam_strings:
        parsed = parse_exam_string(exam_str)
        if parsed:
            ground_truth_tool_names.append(parsed["name"])

    # Select available tools (ground truth + 5-15 distractors)
    num_distractors = rng.randint(5, 15)
    available_tools = select_tools_for_case(
        ground_truth_tools=ground_truth_tool_names,
        num_distractors=num_distractors,
        shuffle=True,
    )

    # ---- Build system prompts ----
    demo = dict(
        age=patient.get("age") or "",
        gender=patient.get("gender") or "",
        race=patient.get("race") or "",
        additional_demographics=patient.get("additional_demographics") or "",
    )
    doctor_base = get_doctor_base_prompt(
        medical_history=patient["medical_history"],
        available_tools=available_tools,
        hidden_canonical_diagnosis=patient["diagnosis"],
        **demo,
    )
    patient_persona = sample_patient_persona(rng)
    patient_system = get_patient_system_prompt(
        age=patient["age"],
        gender=patient["gender"],
        race=patient["race"],
        additional_demographics=patient["additional_demographics"],
        medical_history=patient["medical_history"],
        self_reported_symptoms=patient["self_reported_symptoms"],
        patient_persona=patient_persona,
    )

    # Training output uses get_doctor_training_prompt (no phase labels)
    doctor_system_training = get_doctor_training_prompt(
        medical_history=patient["medical_history"],
        available_tools=available_tools,
        **demo,
    )
    conversation: list[dict] = [{"role": "system", "content": doctor_system_training}]

    # The generation LLM sees only the base prompt; phase prompts are injected
    # one at a time at transitions to improve adherence.
    doctor_messages: list[dict] = [{"role": "system", "content": doctor_base}]
    patient_messages: list[dict] = [{"role": "system", "content": patient_system}]

    turn_count = 0
    patient_noise_log: list[dict] = []  # tracks {noise_type, conv_index} for each applied hint

    # Decide how many inquiry turns before transitioning to exams
    target_inquiry_turns = rng.randint(3, 6)

    # ================================================================
    # Phase 1: Greeting
    # ================================================================
    # Inject greeting phase prompt (generation LLM only)
    doctor_messages.append({"role": "system", "content": get_doctor_phase_prompt("greeting")})

    # Doctor greets (validated — no ordering/result language)
    doctor_msg = _generate_doctor_turn(client, model, doctor_messages, phase="greeting")
    if doctor_msg is None:
        return None  # drop this data point
    # Clean up any CORRECTION hints appended by _generate_doctor_turn
    while doctor_messages and doctor_messages[-1].get("role") == "system" and "CORRECTION" in (doctor_messages[-1].get("content") or ""):
        doctor_messages.pop()
    conversation.append({"role": "assistant", "content": doctor_msg["content"]})
    doctor_messages.append(doctor_msg)
    patient_messages.append({"role": "user", "content": doctor_msg["content"]})
    turn_count += 1

    # Patient responds with chief complaint (with optional noise hint)
    hint_result = get_patient_turn_hint(patient_noise_spec, patient_turn_index, doctor_msg["content"], rng)
    patient_hint, patient_hint_type = hint_result if hint_result else (None, None)
    if patient_hint:
        patient_messages.append({"role": "system", "content": patient_hint})
    patient_msg = llm_call(client, model, patient_messages)
    if patient_hint:
        patient_messages.pop()  # remove hint
    conversation.append({"role": "user", "content": patient_msg["content"]})
    if patient_hint_type:
        patient_noise_log.append({"noise_type": patient_hint_type, "conv_index": len(conversation) - 1})
    patient_messages.append({"role": "assistant", "content": patient_msg["content"]})
    doctor_messages.append({"role": "user", "content": patient_msg["content"]})
    patient_turn_index += 1
    turn_count += 1

    # ================================================================
    # Phase 2: Symptom inquiry (dynamic Q&A turns)
    # ================================================================
    # Inject inquiry phase prompt (generation LLM only)
    doctor_messages.append({"role": "system", "content": get_doctor_phase_prompt("inquiry")})

    inquiry_turns = 0
    while turn_count < max_turns:
        # Check if we should end inquiry phase
        should_end = (inquiry_turns >= target_inquiry_turns or inquiry_turns >= max_inquiry_turns)
        # But keep going if planned noise turns haven't been consumed yet
        if should_end and patient_turn_index <= max_planned_patient_turn:
            should_end = False
        if should_end:
            break

        # Doctor asks about symptoms (validated — no ordering/result language)
        doctor_msg = _generate_doctor_turn(client, model, doctor_messages, phase="inquiry")
        if doctor_msg is None:
            return None  # drop this data point
        # Clean up any CORRECTION hints appended by _generate_doctor_turn
        while doctor_messages and doctor_messages[-1].get("role") == "system" and "CORRECTION" in (doctor_messages[-1].get("content") or ""):
            doctor_messages.pop()
        conversation.append({"role": "assistant", "content": doctor_msg["content"]})
        doctor_messages.append(doctor_msg)
        if doctor_msg.get("content"):
            patient_messages.append({"role": "user", "content": doctor_msg["content"]})
        turn_count += 1
        inquiry_turns += 1

        # Patient answers (with optional noise hint)
        hint_result = get_patient_turn_hint(patient_noise_spec, patient_turn_index, doctor_msg.get("content", ""), rng)
        patient_hint, patient_hint_type = hint_result if hint_result else (None, None)
        if patient_hint:
            patient_messages.append({"role": "system", "content": patient_hint})
        patient_msg = llm_call(client, model, patient_messages)
        if patient_hint:
            patient_messages.pop()  # remove hint
        conversation.append({"role": "user", "content": patient_msg["content"]})
        if patient_hint_type:
            patient_noise_log.append({"noise_type": patient_hint_type, "conv_index": len(conversation) - 1})
        patient_messages.append({"role": "assistant", "content": patient_msg["content"]})
        doctor_messages.append({"role": "user", "content": patient_msg["content"]})
        patient_turn_index += 1
        turn_count += 1

    # ================================================================
    # Phase 3: Exams (tool calls with commentary between each)
    # ================================================================
    if exams:
        # Inject exam phase prompt (generation LLM only, no tool_call format)
        doctor_messages.append({"role": "system", "content": get_doctor_phase_prompt("exam", include_tool_call_format=False)})

    tool_call_counter = 0

    for exam_idx, (exam_str, findings) in enumerate(exams):
        if turn_count >= max_turns:
            break

        parsed = parse_exam_string(exam_str)
        if parsed is None:
            continue

        # Generate a unique tool call ID
        tool_call_id = f"call_{tool_call_counter:04d}"
        tool_call_counter += 1

        # Generate doctor commentary about ordering this exam
        human_name = _humanize_exam_name(exam_str)
        commentary_hint = {
            "role": "system",
            "content": (
                f"Explain to the patient that you'd like to order {human_name}. "
                f"Keep it concise (1-2 sentences)."
            ),
        }
        doctor_messages.append(commentary_hint)
        doctor_commentary = llm_call(client, model, doctor_messages)
        doctor_messages.pop()  # remove hint
        commentary_content = doctor_commentary.get("content") or ""

        # Build assistant message with tool call and commentary as content
        assistant_tool_msg = {
            "role": "assistant",
            "content": commentary_content,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": parsed["name"],
                        "arguments": json.dumps(parsed["arguments"]),
                    },
                }
            ],
        }

        # Apply exam noise before formatting
        exam_noise_meta = None
        forced_exam_noise_type = exam_noise_plan.get(exam_idx) if exam_noise_plan else None
        if forced_exam_noise_type is not None:
            findings, exam_noise_meta = apply_exam_noise(
                parsed["name"], findings, 1.0, rng,
                forced_noise_type=forced_exam_noise_type,
            )
        exam_noise_records.append(exam_noise_meta)

        # Build the tool response message
        formatted_result = format_findings(parsed["name"], parsed["arguments"], findings)
        tool_response_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": formatted_result,
        }

        # Add to conversation
        conversation.append(assistant_tool_msg)
        conversation.append(tool_response_msg)
        if exam_noise_meta is not None:
            exam_noise_meta["conv_index"] = len(conversation) - 1  # index of tool response message

        # Add to doctor's context
        doctor_messages.append(assistant_tool_msg)
        doctor_messages.append(tool_response_msg)
        turn_count += 1

    # ================================================================
    # Phase 4: Diagnosis and Closure
    # ================================================================
    # No separate transition message — the diagnosis prompt instructs the
    # doctor to summarize findings and state the diagnosis in one turn,
    # avoiding consecutive assistant messages.
    has_parseable_exams = any(parse_exam_string(exam_str) is not None for exam_str, _ in exams)
    diagnosis_phase = "diagnosis" if has_parseable_exams else "diagnosis_no_exams"
    doctor_messages.append({"role": "system", "content": get_doctor_phase_prompt(diagnosis_phase)})

    doctor_msg = llm_call(client, model, doctor_messages)
    doctor_diagnosis_content = doctor_msg.get("content") or ""
    conversation.append({"role": "assistant", "content": doctor_diagnosis_content})

    # ---- Check if conversation hit max turns (discard if so) ----
    has_tool_calls = any(m.get("tool_calls") for m in conversation)
    if turn_count >= max_turns and not has_tool_calls:
        return None

    # ---- Build output record ----
    record = {
        "patient_id": patient["patient_id"],
        "diagnosis": patient["diagnosis"],
        "demographics": {
            "age": patient["age"],
            "gender": patient["gender"],
            "race": patient["race"],
            "additional_demographics": patient["additional_demographics"],
        },
        "medical_history": patient["medical_history"],
        "self_reported_symptoms": patient["self_reported_symptoms"],
        "patient_persona": patient_persona,
        "conversation": conversation,
    }

    # Include noise metadata if any noise was configured
    if noise_cfg.patient_noise_level > 0 or noise_cfg.exam_noise_level > 0:
        record["noise_metadata"] = {
            "patient_noise_level": noise_cfg.patient_noise_level,
            "exam_noise_level": noise_cfg.exam_noise_level,
            "patient_noise": {
                "body_part_swaps": patient_noise_spec.body_part_swaps,
                "symptom_confusions": patient_noise_spec.symptom_confusions,
                "severity_changes": patient_noise_spec.severity_changes,
                "temporal_changes": patient_noise_spec.temporal_changes,
                "omitted_symptoms": patient_noise_spec.omitted_symptoms,
                "self_diagnoses": patient_noise_spec.self_diagnoses,
                "vague_answer_assigned": patient_noise_spec.vague_answer_assigned,
                "vague_turns": patient_noise_spec.vague_turns,
                "noise_log": patient_noise_log,
            },
            "exam_noise": exam_noise_records,
        }

    return record


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> set:
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
            return set(data.get("processed_ids", []))
    return set()


def save_checkpoint(processed_ids: set, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"processed_ids": sorted(processed_ids)}, f)


def append_jsonl(records: list[dict], path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_diagnosis_review_jsonl(records: list[dict], output_path: str):
    """Write exhausted diagnosis mismatches to a sidecar review file."""
    if not records:
        return
    base, _ = os.path.splitext(output_path)
    review_path = base + "_diagnosis_review.jsonl"
    os.makedirs(os.path.dirname(review_path) or ".", exist_ok=True)
    with open(review_path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def process_patient(
    patient: dict,
    client: OpenAI,
    model: str,
    max_turns: int,
    max_inquiry_turns: int,
    noise_config: NoiseConfig | None = None,
    patient_noise_types: set[str] | None = None,
    exam_noise_plan: dict[int, str] | None = None,
    diagnosis_match_retries: int = DEFAULT_DIAGNOSIS_MATCH_RETRIES,
    approved_diagnosis_aliases: dict[str, set[str]] | None = None,
) -> dict | None:
    """Wrapper with error handling and diagnosis-match retry for ThreadPoolExecutor.

    If the final [DIAGNOSIS: ...] tag does not exactly match the gold label,
    the entire conversation is regenerated (up to ``diagnosis_match_retries``
    attempts). If all attempts fail, the data point is emitted to the
    diagnosis-review sidecar and treated as processed so future reruns move on
    to later patients instead of retrying the same mismatch indefinitely.

    Returns:
        dict  — successful conversation record
        None  — conversation intentionally filtered
        _ERROR_SENTINEL — LLM/network error occurred; patient should be retried
    """
    gold = patient["diagnosis"].strip()
    attempts = max(1, diagnosis_match_retries)
    for attempt in range(attempts):
        try:
            record = generate_conversation(
                patient, client, model,
                max_turns=max_turns, max_inquiry_turns=max_inquiry_turns,
                noise_config=noise_config,
                patient_noise_types=patient_noise_types,
                exam_noise_plan=exam_noise_plan,
            )
        except Exception as e:
            print(f"[ERROR] Patient {patient['patient_id']}: {e}")
            return _ERROR_SENTINEL

        if record is None:
            # Conversation was intentionally filtered (e.g. hit max turns)
            return None

        # Check if final diagnosis matches gold
        last_asst = None
        for msg in reversed(record["conversation"]):
            if msg["role"] == "assistant":
                last_asst = msg.get("content") or ""
                break

        if last_asst is not None and diagnosis_matches_gold(last_asst, gold, approved_diagnosis_aliases):
            return record

        predicted = extract_diagnosis_tag(last_asst or "")
        if attempt < attempts - 1:
            print(
                f"[DIAGNOSIS MISMATCH] Patient {patient['patient_id']} "
                f"(attempt {attempt + 1}/{attempts}): "
                f"gold={gold!r}, predicted={predicted!r}. Retrying whole conversation."
            )

    # All attempts exhausted — queue for manual review and move on.
    print(
        f"[REVIEW] Patient {patient['patient_id']}: diagnosis never matched "
        f"after {attempts} attempts. gold={gold!r}, last predicted={predicted!r}. "
        "Queued for later review."
    )
    return {
        "_diagnosis_review": True,
        "patient_id": patient["patient_id"],
        "gold_diagnosis": gold,
        "predicted_diagnosis": predicted,
        "attempts": attempts,
        "conversation_record": record,
    }


def _collect_noise_stats(record: dict, counts: dict, indices: dict):
    """Update noise stats from a single conversation record."""
    meta = record.get("noise_metadata")
    if not meta:
        return

    pid = record["patient_id"]
    pn = meta.get("patient_noise", {})

    # Patient noise counts — based on what was planned/detected
    patient_checks = {
        "body_part_swap": bool(pn.get("body_part_swaps")),
        "symptom_confusion": bool(pn.get("symptom_confusions")),
        "severity_change": bool(pn.get("severity_changes")),
        "temporal_change": bool(pn.get("temporal_changes")),
        "omission": bool(pn.get("omitted_symptoms")),
        "self_diagnosis": bool(pn.get("self_diagnoses")),
        "vague_answer": bool(pn.get("vague_answer_assigned", pn.get("vague_turns", 0) > 0)),
    }
    for noise_type, active in patient_checks.items():
        if active:
            counts[f"patient_{noise_type}"] = counts.get(f"patient_{noise_type}", 0) + 1

    # Patient noise indices — built from noise_log (actual applied hints with conv_index)
    for entry in pn.get("noise_log", []):
        nt = entry["noise_type"]
        indices["patient_noise"].setdefault(nt, []).append(
            {"pid": pid, "conv_index": entry["conv_index"]}
        )

    # Exam noise types — list of dicts or Nones
    for entry in meta.get("exam_noise", []):
        if entry and entry.get("noise_type"):
            nt = entry["noise_type"]
            counts[f"exam_{nt}"] = counts.get(f"exam_{nt}", 0) + 1
            indices["exam_noise"].setdefault(nt, []).append(
                {"pid": pid, "conv_index": entry.get("conv_index")}
            )


def _print_noise_summary(counts: dict):
    """Print a summary table of noise injection stats."""
    print("\nNoise injection summary:")

    patient_types = [
        ("body_part_swap", "patients"),
        ("symptom_confusion", "patients"),
        ("severity_change", "patients"),
        ("temporal_change", "patients"),
        ("omission", "patients"),
        ("self_diagnosis", "patients"),
        ("vague_answer", "patients"),
    ]
    exam_types = [
        ("body_part_swap", "exams"),
        ("omission", "exams"),
        ("ambiguity", "exams"),
    ]

    print("  Patient noise:")
    for nt, unit in patient_types:
        c = counts.get(f"patient_{nt}", 0)
        print(f"    {nt + ':':<25s}{c} {unit}")

    print("  Exam noise:")
    for nt, unit in exam_types:
        c = counts.get(f"exam_{nt}", 0)
        print(f"    {nt + ':':<25s}{c} {unit}")


def _print_noise_plan_summary(
    patient_assignments: dict[int, set[str]],
    exam_assignments: dict[int, dict[int, str]],
    patient_eligible: dict[str, int],
    exam_eligible: dict[str, int],
    noise_config: NoiseConfig,
):
    """Print planned noise counts before generation starts."""
    print("\nNoise plan summary (before generation):")

    if noise_config.patient_noise_level > 0:
        total_with_noise = len(patient_assignments)
        print(f"  Patient noise: {total_with_noise} patients selected "
              f"({noise_config.patient_noise_level*100:.0f}% of total)")
        print("  Per-type breakdown (eligible -> planned):")
        for nt in PATIENT_NOISE_TYPES:
            eligible = patient_eligible.get(nt, 0)
            planned = sum(1 for types in patient_assignments.values() if nt in types)
            pct = f"{planned/eligible*100:.1f}%" if eligible else "n/a"
            print(f"    {nt + ':':<25s}{eligible} eligible -> {planned} planned ({pct})")

    if noise_config.exam_noise_level > 0:
        total_exam_slots = sum(len(plan) for plan in exam_assignments.values())
        print(f"  Exam noise: {total_exam_slots} exam slots selected "
              f"({noise_config.exam_noise_level*100:.0f}% of total)")
        print("  Per-type breakdown (eligible -> planned):")
        for nt in EXAM_NOISE_TYPES:
            eligible = exam_eligible.get(nt, 0)
            planned = sum(1 for plan in exam_assignments.values() for t in plan.values() if t == nt)
            pct = f"{planned/eligible*100:.1f}%" if eligible else "n/a"
            print(f"    {nt + ':':<25s}{eligible} eligible -> {planned} planned ({pct})")


def _save_noise_index(indices: dict, output_path: str):
    """Save noise index JSON alongside the output JSONL."""
    base, _ = os.path.splitext(output_path)
    index_path = base + "_noise_index.json"
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    with open(index_path, "w") as f:
        json.dump(indices, f, indent=2)
    print(f"  Noise index saved to: {index_path}")


def run_pipeline(args):
    """Main pipeline entry point."""
    # Load data
    print(f"Loading data from {args.input} ...")
    print(f"Using API base URL: {args.base_url}")
    print(f"Using model: {args.model}")
    patients = []
    with open(args.input, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            patients.append(parse_patient_row(row, idx))
    print(f"Loaded {len(patients)} patient records.")

    # Filter by checkpoint
    processed_ids = load_checkpoint(args.checkpoint)
    remaining = [p for p in patients if p["patient_id"] not in processed_ids]
    print(f"Already processed: {len(processed_ids)}. Remaining: {len(remaining)}.")

    if args.total_size is not None:
        remaining = remaining[: args.total_size]
        print(f"Processing {len(remaining)} patients this run (--total-size {args.total_size}).")

    if not remaining:
        print("Nothing to process.")
        return

    # Init client
    client = OpenAI(base_url=args.base_url, timeout=180.0)
    approved_diagnosis_aliases = load_diagnosis_aliases(
        getattr(args, "diagnosis_aliases_path", DEFAULT_DIAGNOSIS_ALIASES_PATH)
    )
    if approved_diagnosis_aliases:
        print(
            f"Loaded manual diagnosis aliases for {len(approved_diagnosis_aliases)} gold labels "
            f"from {getattr(args, 'diagnosis_aliases_path', DEFAULT_DIAGNOSIS_ALIASES_PATH)}"
        )

    # Build noise config from CLI args
    noise_config = NoiseConfig(
        patient_noise_level=getattr(args, "patient_noise_level", 0.0),
        exam_noise_level=getattr(args, "exam_noise_level", 0.0),
        seed=getattr(args, "noise_seed", None),
    )
    if noise_config.patient_noise_level > 0 or noise_config.exam_noise_level > 0:
        print(f"Noise injection enabled: patient={noise_config.patient_noise_level}, "
              f"exam={noise_config.exam_noise_level}, seed={noise_config.seed}")

    patient_assignments: dict[int, set[str]] = {}
    exam_assignments: dict[int, dict[int, str]] = {}
    if noise_config.patient_noise_level > 0 or noise_config.exam_noise_level > 0:
        patient_assignments, exam_assignments, patient_eligible, exam_eligible = \
            _plan_noise_assignments(remaining, noise_config)
        _print_noise_plan_summary(
            patient_assignments=patient_assignments,
            exam_assignments=exam_assignments,
            patient_eligible=patient_eligible,
            exam_eligible=exam_eligible,
            noise_config=noise_config,
        )

    # Noise tracking
    noise_type_counts = {}
    noise_type_indices = {"patient_noise": {}, "exam_noise": {}}

    # Process in batches
    total_batches = (len(remaining) + args.batch_size - 1) // args.batch_size
    total_saved = 0
    total_filtered = 0

    for batch_start in tqdm(range(0, len(remaining), args.batch_size), total=total_batches, desc="Batches"):
        batch = remaining[batch_start : batch_start + args.batch_size]
        results: list[dict] = []
        diagnosis_review_records: list[dict] = []
        batch_filtered = 0

        error_ids: set[int] = set()
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    process_patient,
                    p,
                    client,
                    args.model,
                    args.max_turns,
                    args.max_inquiry_turns,
                    noise_config,
                    patient_assignments.get(p["patient_id"], set()) if noise_config.patient_noise_level > 0 else None,
                    exam_assignments.get(p["patient_id"]),
                    args.diagnosis_match_retries,
                    approved_diagnosis_aliases,
                ): p
                for p in batch
            }
            for future in as_completed(futures):
                p = futures[future]
                result = future.result()
                if result is _ERROR_SENTINEL:
                    error_ids.add(p["patient_id"])
                    batch_filtered += 1
                elif isinstance(result, dict) and result.get("_diagnosis_review"):
                    diagnosis_review_records.append(result)
                    batch_filtered += 1
                elif result is not None:
                    results.append(result)
                else:
                    batch_filtered += 1

        # Collect noise stats
        for rec in results:
            _collect_noise_stats(rec, noise_type_counts, noise_type_indices)

        # Save results
        if results:
            append_jsonl(results, args.output)
        if diagnosis_review_records:
            append_diagnosis_review_jsonl(diagnosis_review_records, args.output)

        # Update checkpoint — only exclude true execution errors.
        # LLM/network failures are retried on the next run. Intentionally
        # filtered patients and exhausted diagnosis-review cases are marked
        # processed so later reruns can move on to new data points.
        batch_errors = len(error_ids)
        for p in batch:
            if p["patient_id"] not in error_ids:
                processed_ids.add(p["patient_id"])
        save_checkpoint(processed_ids, args.checkpoint)

        total_saved += len(results)
        total_filtered += batch_filtered
        error_note = f", {batch_errors} errors (will retry)" if batch_errors else ""
        print(f"  Batch done. Saved {len(results)}/{len(batch)} conversations ({batch_filtered - batch_errors} filtered out{error_note}).")

    print(f"\nPipeline complete. Output: {args.output}")
    print(f"Total: {total_saved} saved, {total_filtered} filtered out of {len(remaining)} patients.")

    if noise_config.patient_noise_level > 0 or noise_config.exam_noise_level > 0:
        _print_noise_summary(noise_type_counts)
        _save_noise_index(noise_type_indices, args.output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate doctor-patient conversations from EHR data.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint JSON path")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for checkpointing")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel workers")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="Max conversation turns")
    parser.add_argument("--max-inquiry-turns", type=int, default=DEFAULT_MAX_INQUIRY_TURNS, help="Max symptom inquiry turns before forcing exam transition")
    parser.add_argument(
        "--diagnosis-match-retries",
        type=int,
        default=DEFAULT_DIAGNOSIS_MATCH_RETRIES,
        help="How many times to retry the final diagnosis turn when the [DIAGNOSIS: ...] tag does not exactly match the gold label",
    )
    parser.add_argument(
        "--diagnosis-aliases-path",
        default=DEFAULT_DIAGNOSIS_ALIASES_PATH,
        help="JSON file of manually approved diagnosis aliases/equivalences",
    )
    parser.add_argument("--total-size", type=int, default=None, help="Max patients to process this run")
    parser.add_argument("--patient-noise-level", type=float, default=0.0,
                        help="Patient noise level 0.0-1.0 (probability of planning each patient noise type)")
    parser.add_argument("--exam-noise-level", type=float, default=0.0,
                        help="Exam noise level 0.0-1.0 (probability of noise per exam finding)")
    parser.add_argument("--noise-seed", type=int, default=None,
                        help="Random seed for reproducible noise injection")
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
