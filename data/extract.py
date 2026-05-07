"""
Step 1: Extract structured patient information from PMC-Patients JSON via LLM.

Reads PMC-Patients-V2.json, sends each patient record through an LLM to
extract demographics, symptoms, exams, and diagnosis, then saves to CSV
with checkpoint support.

Usage:
    python -m data.pipeline extract --total-size 100
    python -m data.extract  # runs directly
"""

import argparse
import json
import os
import re
import time

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, RateLimitError

from data.utils import rate_limit_sleep
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT = "data/ehr/PMC-Patients-V2.json"
DEFAULT_OUTPUT = "data/ehr/pmc_patients_extracted.csv"
DEFAULT_CHECKPOINT = "data/ehr/processing_checkpoint.json"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "DEFAULT"
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_WORKERS = 64
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """You are a medical information extraction assistant. Your task is to extract specific information from a patient case report.

IMPORTANT INSTRUCTIONS:
1. Only extract information from the INITIAL DIAGNOSIS process - the patient's first presentation and workup.
2. DO NOT include any examinations or tests performed DURING treatment.
3. DO NOT include any information from follow-up visits, readmissions, or post-treatment evaluations.
4. For examinations, only include diagnostic tests done during the initial workup to establish the diagnosis.
5. If information is not explicitly mentioned or cannot be determined, leave the field empty.
6. Extract up to 10 examinations/tests with their findings.
7. When recording exam names, include the type of exam (e.g., "CT scan", "blood test") and the body part if specified (e.g., "chest X-ray", "abdominal ultrasound"), as well as other information if specified (e.g. "AP X-ray").

Extract the following information in the exact JSON format below:

```json
{
    "age": "<age with unit, e.g., '34 years', '6 months', or empty if not mentioned>",
    "gender": "<Male/Female/Other or empty if not mentioned>",
    "race": "<race/ethnicity or empty if not mentioned>",
    "additional_demographics": "<any other demographic info like occupation, location, etc., or empty>",
    "medical_history": "<past medical history, comorbidities, prior conditions, or empty>",
    "diagnosis": "<the initial/primary diagnosis established, or empty>",
    "self_reported_symptoms": "<symptoms the patient reported/complained of at presentation, or empty>",
    "exam1": "<name & other info of first examination/test during initial workup, or empty>",
    "exam1_findings": "<findings/results of first exam, or empty>",
    "exam2": "<name & other info of second examination/test during initial workup, or empty>",
    "exam2_findings": "<findings/results of second exam, or empty>",
    "exam3": "<name & other info of third examination/test during initial workup, or empty>",
    "exam3_findings": "<findings/results of third exam, or empty>",
    "exam4": "<name & other info of fourth examination/test during initial workup, or empty>",
    "exam4_findings": "<findings/results of fourth exam, or empty>",
    "exam5": "<name & other info of fifth examination/test during initial workup, or empty>",
    "exam5_findings": "<findings/results of fifth exam, or empty>",
    "exam6": "<name & other info of sixth examination/test during initial workup, or empty>",
    "exam6_findings": "<findings/results of sixth exam, or empty>",
    "exam7": "<name & other info of seventh examination/test during initial workup, or empty>",
    "exam7_findings": "<findings/results of seventh exam, or empty>",
    "exam8": "<name & other info of eighth examination/test during initial workup, or empty>",
    "exam8_findings": "<findings/results of eighth exam, or empty>",
    "exam9": "<name & other info of ninth examination/test during initial workup, or empty>",
    "exam9_findings": "<findings/results of ninth exam, or empty>",
    "exam10": "<name & other info of tenth examination/test during initial workup, or empty>",
    "exam10_findings": "<findings/results of tenth exam, or empty>"
}
```

Patient Case Report:
{patient_text}

Extract the information and respond ONLY with the JSON object, no other text."""


# ---------------------------------------------------------------------------
# Empty result template
# ---------------------------------------------------------------------------
_EMPTY_RESULT = {
    "age": "", "gender": "", "race": "",
    "additional_demographics": "", "medical_history": "",
    "diagnosis": "", "self_reported_symptoms": "",
}
for _i in range(1, 11):
    _EMPTY_RESULT[f"exam{_i}"] = ""
    _EMPTY_RESULT[f"exam{_i}_findings"] = ""

_CSV_COLUMNS = list(_EMPTY_RESULT.keys()) + ["original_text"]


# ---------------------------------------------------------------------------
# Response cleaning
# ---------------------------------------------------------------------------
def clean_thinking_response(response_text: str) -> str:
    """Remove thinking tags from LLM response."""
    patterns = [
        r'<think>.*?</think>',
        r'<thinking>.*?</thinking>',
        r'<reasoning>.*?</reasoning>',
        r'\[thinking\].*?\[/thinking\]',
    ]
    cleaned = response_text
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def extract_json_from_response(response_text: str) -> str:
    """Extract JSON object from the LLM response."""
    cleaned = clean_thinking_response(response_text)

    # Try to find JSON in code blocks
    json_match = re.search(r'```(?:json)?\s*\n?({.*?})\s*\n?```', cleaned, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # Try to find raw JSON object
    json_match = re.search(r'({\s*"age".*?})', cleaned, re.DOTALL)
    if json_match:
        return json_match.group(1)

    return cleaned


def parse_extraction_result(response_text: str) -> dict:
    """Parse the LLM response into a structured dictionary."""
    empty_result = dict(_EMPTY_RESULT)
    try:
        json_str = extract_json_from_response(response_text)
        result = json.loads(json_str)
        for key in empty_result:
            if key not in result or result[key] is None:
                result[key] = ""
        return result
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Failed to parse response: {e}")
        return empty_result


# ---------------------------------------------------------------------------
# Single-record extraction
# ---------------------------------------------------------------------------
def extract_patient_info(patient_record: dict, client: OpenAI, model_name: str, max_retries: int = MAX_RETRIES) -> dict:
    """Extract information from a single patient record using the LLM."""
    patient_text = patient_record.get('patient', '')

    if not patient_text:
        print(f"No patient text found for record {patient_record.get('patient_id', 'unknown')}")
        result = dict(_EMPTY_RESULT)
        result['original_text'] = ''
        return result

    prompt = EXTRACTION_PROMPT.replace("{patient_text}", patient_text)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = response.choices[0].message.content
            result = parse_extraction_result(response_text)
            result['original_text'] = patient_text
            return result
        except RateLimitError as e:
            rate_limit_sleep(e, attempt)
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"All retries failed (rate limited) for patient {patient_record.get('patient_id', 'unknown')}"
                ) from e
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise RuntimeError(
                    f"All retries failed for patient {patient_record.get('patient_id', 'unknown')}"
                ) from e


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------
def save_checkpoint(processed_ids: set, checkpoint_file: str):
    os.makedirs(os.path.dirname(checkpoint_file) or ".", exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump({'processed_ids': list(processed_ids)}, f)


def load_checkpoint(checkpoint_file: str) -> set:
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            return set(data.get('processed_ids', []))
    return set()


def append_to_csv(results: list[dict], output_file: str):
    df = pd.DataFrame(results)
    df = df[[c for c in _CSV_COLUMNS if c in df.columns]]
    write_header = not os.path.exists(output_file)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df.to_csv(output_file, mode='a', header=write_header, index=False)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def process_batch(
    patients_batch: list[dict], client: OpenAI, model_name: str, max_workers: int,
) -> tuple[list[dict], set]:
    """Process a batch of patients. Returns (results, error_patient_ids)."""
    results = []
    error_ids: set = set()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_patient = {
            executor.submit(extract_patient_info, patient, client, model_name): patient
            for patient in patients_batch
        }
        for future in as_completed(future_to_patient):
            patient = future_to_patient[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing patient {patient.get('patient_id', 'unknown')}: {e}")
                error_ids.add(patient.get('patient_id'))
    return results, error_ids


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_extract(args):
    """Run the extraction pipeline."""
    # Load data
    print(f"Loading patient data from {args.input} ...")
    print(f"Using API base URL: {args.base_url}")
    with open(args.input, 'r') as f:
        patients_data = json.load(f)
    print(f"Loaded {len(patients_data)} patient records")

    # Resolve model name
    client = OpenAI(base_url=args.base_url, timeout=180.0)
    model_name = args.model

    if model_name == 'DEFAULT':
        try:
            models = client.models.list()
            available_models = [m.id for m in models.data]
            print(f"Available models: {available_models}")
            if available_models:
                model_name = available_models[0]
                print(f"Using model: {model_name}")
        except Exception as e:
            print(f"Could not fetch models: {e}")
            print("Please set --model manually")
            return
    else:
        print(f"Using specified model: {model_name}")

    # Load checkpoint
    processed_ids = load_checkpoint(args.checkpoint)
    print(f"Found {len(processed_ids)} already processed records")

    # Filter remaining
    remaining_patients = [
        p for p in patients_data
        if p.get('patient_id') not in processed_ids
    ]
    print(f"Remaining to process: {len(remaining_patients)}")

    if not remaining_patients:
        print("All records have been processed!")
        return

    if args.total_size is not None:
        remaining_budget = max(0, args.total_size - len(processed_ids))
        if remaining_budget == 0:
            print(
                f"Target total_size={args.total_size} already satisfied by checkpoint "
                f"({len(processed_ids)} processed). Nothing to do."
            )
            return
        remaining_patients = remaining_patients[:remaining_budget]
        print(
            f"Processing {len(remaining_patients)} patients this run "
            f"to reach total_size={args.total_size}"
        )

    # Process in batches
    batch_size = args.batch_size
    total_batches = (len(remaining_patients) + batch_size - 1) // batch_size
    processed_this_run = 0

    for i in tqdm(range(0, len(remaining_patients), batch_size),
                  total=total_batches, desc="Processing batches"):
        batch = remaining_patients[i:i + batch_size]
        results, error_ids = process_batch(batch, client, model_name, args.max_workers)

        if results:
            append_to_csv(results, args.output)

        for patient in batch:
            pid = patient.get('patient_id')
            if pid not in error_ids:
                processed_ids.add(pid)
        save_checkpoint(processed_ids, args.checkpoint)

        processed_this_run += len(results)
        error_note = f", {len(error_ids)} errors (will retry)" if error_ids else ""
        print(f"Processed batch {i // batch_size + 1}/{total_batches}, "
              f"This run: {processed_this_run}, Total processed: {len(processed_ids)}{error_note}")

    print(f"\nProcessing complete!")
    print(f"Processed this run: {processed_this_run}")
    print(f"Total records processed: {len(processed_ids)}")
    print(f"Output saved to: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Extract structured patient info from PMC-Patients JSON via LLM.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSON path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint JSON path")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name (DEFAULT = auto-detect)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for checkpointing")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel workers")
    parser.add_argument(
        "--total-size",
        type=int,
        default=None,
        help="Target total number of extracted patients after checkpointing, not an additional per-run amount",
    )
    args = parser.parse_args()
    run_extract(args)


if __name__ == "__main__":
    main()
