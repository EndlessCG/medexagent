"""
Convert DDxPlus dataset into the pipeline CSV format used by data/generate.py.

Decodes evidence codes into structured data, uses LLM to convert into
natural patient descriptions, balances across conditions, and optionally
synthesizes plausible exams via LLM.

Usage:
    python -m data.database.process_ddxplus \
        --input data/database/release_train_patients.csv \
        --evidences data/database/release_evidences.json \
        --output data/database/processed_datasets/ddxplus_converted.csv \
        --target-size 20000 \
        --base-url http://localhost:8000/v1 \
        --model DEFAULT \
        --max-workers 64 \
        --batch-size 100 \
        [--skip-synthesis] \
        [--start-index 0] \
        [--end-index 5000]
"""

import argparse
import ast
import json
import os
import re
import time

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, RateLimitError

from data.utils import rate_limit_sleep
from tqdm import tqdm

from data.tool import (
    TOOL_PARAM_VALUES,
    parse_exam_string,
    normalize_param_value,
    rebuild_tool_call_string,
)
from data.filter import TOOL_TAXONOMY


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT = "data/database/release_train_patients.csv"
DEFAULT_EVIDENCES = "data/database/release_evidences.json"
DEFAULT_OUTPUT = "data/database/processed_datasets/ddxplus_converted.csv"
DEFAULT_CHECKPOINT = "data/database/processed_datasets/ddxplus_checkpoint.json"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_WORKERS = 64
DEFAULT_TARGET_SIZE = 20000
MAX_RETRIES = 3

# CSV columns matching the pipeline format (9 exam slots, no exam10)
CSV_COLUMNS = [
    "age", "gender", "race", "additional_demographics", "medical_history",
    "diagnosis", "self_reported_symptoms",
]
for _i in range(1, 10):
    CSV_COLUMNS.append(f"exam{_i}")
    CSV_COLUMNS.append(f"exam{_i}_findings")


# ---------------------------------------------------------------------------
# Evidence decoder (mechanical: codes -> structured data)
# ---------------------------------------------------------------------------
_SKIP_M_VALUES = {"nowhere", "na", "n", "none"}
_SKIP_C_VALUES = {0, "0"}


def load_evidences(path: str) -> dict:
    """Load release_evidences.json."""
    with open(path, "r") as f:
        return json.load(f)


def _decode_evidence_raw(code: str, evidences: dict) -> dict | None:
    """Decode a single evidence code into a structured dict.

    Returns:
        {"question": str, "value": str|None, "is_antecedent": bool}
        or None if the evidence should be skipped.
    """
    parts = code.split("_@_")
    ev_id = parts[0]

    if ev_id not in evidences:
        return None

    ev = evidences[ev_id]
    is_antecedent = ev.get("is_antecedent", False)
    question = ev.get("question_en", "")
    data_type = ev.get("data_type", "B")
    default_value = ev.get("default_value")

    if data_type == "B":
        # Binary: presence means "yes" to this question
        return {"question": question, "value": "yes", "is_antecedent": is_antecedent}

    elif data_type == "M" and len(parts) == 2:
        value_code = parts[1]
        if value_code == default_value:
            return None
        value_meaning = ev.get("value_meaning", {})
        if value_code in value_meaning:
            meaning = value_meaning[value_code].get("en", "")
            if meaning.lower().strip() in _SKIP_M_VALUES:
                return None
            return {"question": question, "value": meaning, "is_antecedent": is_antecedent}
        return None

    elif data_type == "C" and len(parts) == 2:
        raw_value = parts[1]
        if raw_value in _SKIP_C_VALUES or raw_value == str(default_value):
            return None
        value_meaning = ev.get("value_meaning", {})
        if raw_value in value_meaning:
            meaning = value_meaning[raw_value].get("en", raw_value)
        else:
            meaning = raw_value
        return {"question": question, "value": meaning, "is_antecedent": is_antecedent}

    return None


def extract_raw_evidences(
    evidences_str: str,
    initial_evidence: str,
    evidences_db: dict,
) -> tuple[list[dict], list[dict]]:
    """Extract structured evidence data for a patient.

    Returns:
        (symptom_evidences, history_evidences) — each is a list of
        {"question": str, "value": str} dicts.
    """
    try:
        evidence_codes = ast.literal_eval(evidences_str)
    except (ValueError, SyntaxError):
        return [], []

    symptoms = []
    history = []
    seen_questions = set()

    # Initial evidence first (chief complaint)
    if initial_evidence and initial_evidence in evidences_db:
        decoded = _decode_evidence_raw(initial_evidence, evidences_db)
        if decoded:
            symptoms.append({"question": decoded["question"], "value": decoded["value"]})
            seen_questions.add(decoded["question"])

    for code in evidence_codes:
        base_code = code.split("_@_")[0]
        # Skip bare initial evidence (already added), but allow M/C variants
        if base_code == initial_evidence and "_@_" not in code:
            continue

        decoded = _decode_evidence_raw(code, evidences_db)
        if decoded is None:
            continue

        entry = {"question": decoded["question"], "value": decoded["value"]}

        if decoded["is_antecedent"]:
            history.append(entry)
        else:
            # Avoid exact duplicates
            key = (decoded["question"], decoded["value"])
            if key not in seen_questions:
                seen_questions.add(key)
                symptoms.append(entry)

    return symptoms, history


# ---------------------------------------------------------------------------
# LLM evidence-to-text conversion
# ---------------------------------------------------------------------------
EVIDENCE_CONVERSION_PROMPT = """You are a medical data converter. You will be given structured evidence data from a patient record (question-answer pairs from a medical questionnaire) and the patient's demographics.

Convert these into natural, concise clinical descriptions as a doctor would write in a patient chart.

## Rules:
1. For "self_reported_symptoms": Write a semicolon-separated list of symptoms as a patient would describe them. Combine related items (e.g., pain location + intensity + character into one phrase). Be concise and natural.
2. For "medical_history": Write a semicolon-separated list of past medical conditions, risk factors, and relevant background. Be concise and natural.
3. Do NOT include the diagnosis in the output.
4. Do NOT fabricate information not present in the evidence data.
5. Convert questionnaire-style language into natural clinical language.

## Output format:
Return ONLY a JSON object:
{"self_reported_symptoms": "fever; productive cough with yellow sputum; sharp chest pain on the right side, 7/10 severity", "medical_history": "smoking history; previous pneumonia; COPD"}

Empty strings are valid if no relevant evidences exist:
{"self_reported_symptoms": "headache", "medical_history": ""}"""


def convert_evidences_llm(
    age: str,
    gender: str,
    diagnosis: str,
    symptom_evidences: list[dict],
    history_evidences: list[dict],
    client: OpenAI,
    model: str,
    max_retries: int = MAX_RETRIES,
) -> tuple[str, str]:
    """Use LLM to convert raw evidence data into natural text.

    Returns (symptoms_text, history_text).
    """
    # Format raw evidences for the LLM
    symptom_lines = []
    for ev in symptom_evidences:
        if ev["value"] == "yes":
            symptom_lines.append(f"- Q: {ev['question']} → Yes")
        else:
            symptom_lines.append(f"- Q: {ev['question']} → {ev['value']}")

    history_lines = []
    for ev in history_evidences:
        if ev["value"] == "yes":
            history_lines.append(f"- Q: {ev['question']} → Yes")
        else:
            history_lines.append(f"- Q: {ev['question']} → {ev['value']}")

    user_msg = (
        f"Patient: {age}, {gender}\n\n"
        f"Symptom evidences:\n{chr(10).join(symptom_lines) or 'None'}\n\n"
        f"Medical history evidences:\n{chr(10).join(history_lines) or 'None'}"
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EVIDENCE_CONVERSION_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.3,
            )
            raw = response.choices[0].message.content
            cleaned = _clean_response(raw)
            result = json.loads(cleaned)
            symptoms = result.get("self_reported_symptoms", "")
            history = result.get("medical_history", "")
            return symptoms, history
        except RateLimitError as e:
            rate_limit_sleep(e, attempt)
            if attempt == max_retries - 1:
                print(f"Evidence conversion failed after {max_retries} attempts (rate limited): {e}")
                return _fallback_evidence_text(symptom_evidences, history_evidences)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Evidence conversion failed after {max_retries} attempts: {e}")
                # Fallback: simple concatenation
                return _fallback_evidence_text(symptom_evidences, history_evidences)

    return _fallback_evidence_text(symptom_evidences, history_evidences)


def _fallback_evidence_text(
    symptom_evidences: list[dict],
    history_evidences: list[dict],
) -> tuple[str, str]:
    """Simple fallback if LLM fails: join question/value pairs."""
    symptoms = "; ".join(
        ev["question"].rstrip("?") if ev["value"] == "yes"
        else f"{ev['question'].rstrip('?')}: {ev['value']}"
        for ev in symptom_evidences
    )
    history = "; ".join(
        ev["question"].rstrip("?") if ev["value"] == "yes"
        else f"{ev['question'].rstrip('?')}: {ev['value']}"
        for ev in history_evidences
    )
    return symptoms, history


# ---------------------------------------------------------------------------
# Balanced sampling
# ---------------------------------------------------------------------------
def balanced_sample(df: pd.DataFrame, target_size: int, random_state: int = 42) -> pd.DataFrame:
    """Sample approximately target_size patients, balanced across conditions."""
    conditions = df["PATHOLOGY"].unique()
    per_condition = target_size // len(conditions)
    remainder = target_size % len(conditions)

    samples = []
    for i, cond in enumerate(sorted(conditions)):
        cond_df = df[df["PATHOLOGY"] == cond]
        n = per_condition + (1 if i < remainder else 0)
        n = min(n, len(cond_df))
        samples.append(cond_df.sample(n=n, random_state=random_state))

    result = pd.concat(samples, ignore_index=True)
    result = result.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# LLM exam synthesis
# ---------------------------------------------------------------------------
def _build_compact_tool_reference() -> str:
    """Build a compact tool reference from TOOL_TAXONOMY with enum values."""
    lines = []
    for func_name, info in TOOL_TAXONOMY.items():
        if func_name == "other":
            continue
        params = info["params"]
        if params:
            param_parts = []
            for k in params:
                if func_name in TOOL_PARAM_VALUES and k in TOOL_PARAM_VALUES[func_name]:
                    allowed = TOOL_PARAM_VALUES[func_name][k]
                    param_parts.append(f'{k}=[{"|".join(allowed)}]')
                else:
                    param_parts.append(f'{k}="..."')
            sig = f"{func_name}({', '.join(param_parts)})"
        else:
            sig = f"{func_name}()"
        lines.append(f"  {sig}")
    return "\n".join(lines)


COMPACT_TOOL_REF = _build_compact_tool_reference()

EXAM_SYNTHESIS_PROMPT = f"""You are a medical exam planner. Given a patient's demographics, symptoms, medical history, and final diagnosis, suggest realistic diagnostic examinations a doctor would order.

## Available tool calls (use ONLY these, with the exact parameter values shown):
{COMPACT_TOOL_REF}

## Rules:
1. Return 0-9 exams. Zero is valid if diagnosis is obvious from symptoms alone.
2. Each exam must use one of the tool calls above with valid parameter values from the brackets.
3. Findings should be plausible for the given diagnosis — what the exam would realistically show.
4. Keep findings concise (1-2 sentences).
5. Order exams from most basic/common to most specialized.
6. Do NOT include exams that wouldn't realistically be ordered for this presentation.

## Output format:
Return ONLY a JSON object:
{{"exams": [{{"tool_call": "xray(region=\\"chest\\")", "findings": "Bilateral infiltrates consistent with pneumonia."}}, ...]}}

An empty list is valid: {{"exams": []}}"""


def _clean_response(text: str) -> str:
    """Remove thinking tags and extract JSON."""
    patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
    ]
    for p in patterns:
        text = re.sub(p, '', text, flags=re.DOTALL | re.IGNORECASE)
    text = text.strip()

    # Extract JSON from code blocks
    m = re.search(r'```(?:json)?\s*\n?({.*?})\s*\n?```', text, re.DOTALL)
    if m:
        return m.group(1)

    # Find raw JSON
    m = re.search(r'(\{.*\})', text, re.DOTALL)
    if m:
        return m.group(1)

    return text


def synthesize_exams(
    patient_info: dict,
    client: OpenAI,
    model: str,
    max_retries: int = MAX_RETRIES,
) -> list[dict]:
    """Call LLM to synthesize exams for a patient. Returns list of {tool_call, findings}."""
    user_msg = (
        f"Patient: {patient_info['age']}, {patient_info['gender']}\n"
        f"Symptoms: {patient_info['symptoms']}\n"
        f"Medical history: {patient_info['history'] or 'None reported'}\n"
        f"Diagnosis: {patient_info['diagnosis']}"
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": EXAM_SYNTHESIS_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.7,
            )
            raw = response.choices[0].message.content
            cleaned = _clean_response(raw)
            result = json.loads(cleaned)
            exams = result.get("exams", [])

            # Validate each exam
            validated = []
            for exam in exams:
                if len(validated) >= 9:
                    break
                tc_str = exam.get("tool_call", "")
                findings = exam.get("findings", "")
                parsed = parse_exam_string(tc_str)
                if parsed is None:
                    continue
                if parsed["name"] == "other":
                    continue

                # Normalize parameters
                new_args = {}
                valid = True
                for param, val in parsed["arguments"].items():
                    norm = normalize_param_value(parsed["name"], param, val)
                    if norm is not None:
                        new_args[param] = norm
                    else:
                        if parsed["name"] in TOOL_PARAM_VALUES and param in TOOL_PARAM_VALUES[parsed["name"]]:
                            valid = False
                            break
                        else:
                            new_args[param] = val

                if not valid:
                    continue

                normalized_tc = rebuild_tool_call_string(parsed["name"], new_args)
                validated.append({"tool_call": normalized_tc, "findings": findings})

            return validated

        except RateLimitError as e:
            rate_limit_sleep(e, attempt)
            if attempt == max_retries - 1:
                return []
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return []

    return []


# ---------------------------------------------------------------------------
# Row processing
# ---------------------------------------------------------------------------
def process_patient_row(
    row: pd.Series,
    evidences_db: dict,
    client: OpenAI,
    model: str,
    skip_synthesis: bool,
) -> dict:
    """Process a single DDxPlus patient row into pipeline CSV format."""
    # Demographics
    age = f"{int(row['AGE'])} years"
    gender = "Male" if row["SEX"] == "M" else "Female"
    diagnosis = row["PATHOLOGY"]

    # Extract raw structured evidences
    symptom_evidences, history_evidences = extract_raw_evidences(
        str(row["EVIDENCES"]),
        str(row["INITIAL_EVIDENCE"]),
        evidences_db,
    )

    # Convert to natural text via LLM
    symptoms, history = convert_evidences_llm(
        age, gender, diagnosis,
        symptom_evidences, history_evidences,
        client, model,
    )

    result = {
        "age": age,
        "gender": gender,
        "race": "",
        "additional_demographics": "",
        "medical_history": history,
        "diagnosis": diagnosis,
        "self_reported_symptoms": symptoms,
    }

    # Initialize empty exam slots
    for i in range(1, 10):
        result[f"exam{i}"] = ""
        result[f"exam{i}_findings"] = ""

    # Synthesize exams if not skipping
    if not skip_synthesis:
        exams = synthesize_exams(
            {
                "age": age,
                "gender": gender,
                "symptoms": symptoms,
                "history": history,
                "diagnosis": diagnosis,
            },
            client,
            model,
        )
        for i, exam in enumerate(exams, start=1):
            result[f"exam{i}"] = exam["tool_call"]
            result[f"exam{i}_findings"] = exam["findings"]

    return result


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
def save_checkpoint(processed_indices: set, checkpoint_file: str):
    os.makedirs(os.path.dirname(checkpoint_file) or ".", exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump({"processed_indices": sorted(processed_indices)}, f)


def load_checkpoint(checkpoint_file: str) -> set:
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
            return set(data.get("processed_indices", []))
    return set()


def append_to_csv(results: list[dict], output_file: str):
    df = pd.DataFrame(results)
    df = df[[c for c in CSV_COLUMNS if c in df.columns]]
    write_header = not os.path.exists(output_file)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df.to_csv(output_file, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def process_batch(
    batch: list[tuple[int, pd.Series]],
    evidences_db: dict,
    client: OpenAI,
    model: str,
    skip_synthesis: bool,
    max_workers: int,
) -> list[tuple[int, dict]]:
    """Process a batch of patients in parallel."""
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_patient_row, row, evidences_db, client, model, skip_synthesis
            ): idx
            for idx, row in batch
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
                results.append((idx, result))
            except Exception as e:
                print(f"Error processing row {idx}: {e}")

    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run(args):
    """Run the DDxPlus conversion pipeline."""
    # Load evidences
    print(f"Loading evidences from {args.evidences} ...")
    evidences_db = load_evidences(args.evidences)
    print(f"Loaded {len(evidences_db)} evidence definitions")
    print(f"Using API base URL: {args.base_url}")

    # Load patient data — either from a pre-sampled CSV or by sampling from raw
    if args.sampled_input:
        print(f"Loading pre-sampled data from {args.sampled_input} ...")
        sampled = pd.read_csv(args.sampled_input)
        print(f"Loaded {len(sampled)} pre-sampled patients")
    else:
        print(f"Loading patient data from {args.input} ...")
        df = pd.read_csv(args.input)
        print(f"Loaded {len(df)} patient records, {df['PATHOLOGY'].nunique()} conditions")

        target_size = args.target_size
        if args.end_index is not None and args.end_index > target_size:
            target_size = args.end_index
            print(f"Adjusted target-size to {target_size} to cover --end-index {args.end_index}")

        sampled = balanced_sample(df, target_size)
        print(f"Sampled {len(sampled)} patients ({target_size} target, balanced across conditions)")

        # If --sample-only, write the sampled pool and exit
        if args.sample_only:
            if os.path.exists(args.output):
                existing = pd.read_csv(args.output)
                print(f"Sampled pool already exists at {args.output} ({len(existing)} patients). Skipping.")
                return
            sampled.to_csv(args.output, index=False)
            print(f"Wrote sampled pool ({len(sampled)} patients) to {args.output}")
            return

    # Apply manual start/end index slicing
    start_idx = args.start_index if args.start_index is not None else 0
    end_idx = args.end_index if args.end_index is not None else len(sampled)
    start_idx = max(0, start_idx)
    end_idx = min(len(sampled), end_idx)
    if start_idx > 0 or end_idx < len(sampled):
        sampled = sampled.iloc[start_idx:end_idx].reset_index(drop=True)
        print(f"Sliced to index range [{start_idx}, {end_idx}): {len(sampled)} patients")

    # Setup LLM client (always required for evidence conversion)
    client = OpenAI(base_url=args.base_url, timeout=180.0)
    model_name = args.model
    if model_name == "DEFAULT":
        try:
            models = client.models.list()
            available = [m.id for m in models.data]
            if available:
                model_name = available[0]
                print(f"Using model: {model_name}")
            else:
                print("No models available. Please specify --model.")
                return
        except Exception as e:
            print(f"Could not fetch models: {e}")
            print("Please set --model manually.")
            return
    else:
        print(f"Using model: {model_name}")

    # Load checkpoint
    processed_indices = load_checkpoint(args.checkpoint)
    print(f"Found {len(processed_indices)} already processed rows")

    # Build list of remaining rows
    remaining = [
        (idx, sampled.iloc[idx])
        for idx in range(len(sampled))
        if idx not in processed_indices
    ]
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("All records have been processed!")
        return

    # Process in batches
    batch_size = args.batch_size
    total_batches = (len(remaining) + batch_size - 1) // batch_size
    processed_this_run = 0

    for i in tqdm(range(0, len(remaining), batch_size),
                  total=total_batches, desc="Processing batches"):
        batch = remaining[i:i + batch_size]
        results = process_batch(
            batch, evidences_db, client, model_name,
            args.skip_synthesis, args.max_workers,
        )

        # Sort by index for consistent output
        results.sort(key=lambda x: x[0])
        rows = [r for _, r in results]
        indices = [idx for idx, _ in results]

        append_to_csv(rows, args.output)

        processed_indices.update(indices)
        save_checkpoint(processed_indices, args.checkpoint)

        processed_this_run += len(batch)

    print(f"\nProcessing complete!")
    print(f"Processed this run: {processed_this_run}")
    print(f"Total records processed: {len(processed_indices)}")
    print(f"Output saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert DDxPlus dataset to pipeline CSV format."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input DDxPlus CSV")
    parser.add_argument("--evidences", default=DEFAULT_EVIDENCES, help="Evidence definitions JSON")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint JSON")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Moƒdel name (DEFAULT = auto-detect)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for checkpointing")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel workers")
    parser.add_argument("--target-size", type=int, default=DEFAULT_TARGET_SIZE, help="Target number of patients")
    parser.add_argument("--skip-synthesis", action="store_true", help="Skip LLM exam synthesis (still uses LLM for evidence conversion)")
    parser.add_argument("--start-index", type=int, default=None, help="Start index (inclusive) into the sampled dataset")
    parser.add_argument("--end-index", type=int, default=None, help="End index (exclusive) into the sampled dataset")
    parser.add_argument("--sample-only", action="store_true", help="Only generate the balanced sample CSV and exit (no LLM processing)")
    parser.add_argument("--sampled-input", type=str, default=None, help="Pre-sampled CSV to use instead of sampling from --input (guarantees no overlap between splits)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
