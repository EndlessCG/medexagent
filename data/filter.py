"""
Step 2: Filter extracted CSV and classify exam names to tool call strings.

Reads the extracted CSV from Step 1, filters out rows without diagnosis or
symptoms, drops rows with exam10 (too many exams), classifies each unique
exam name into a tool call function via LLM, removes rows containing
``other()`` exams, and saves the post-processed CSV.

Usage:
    python -m data.pipeline filter
    python -m data.filter  # runs directly
"""

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from openai import OpenAI, RateLimitError

from data.utils import rate_limit_sleep
from tqdm import tqdm

from data.tool import (
    TOOL_PARAM_VALUES,
    parse_exam_string,
    normalize_param_value,
    rebuild_tool_call_string,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_INPUT = "data/ehr/pmc_patients_extracted.csv"
DEFAULT_OUTPUT = "data/ehr/pmc_patients_post_processed.csv"
DEFAULT_CHECKPOINT = "data/ehr/exam_tool_call_checkpoint.json"
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_WORKERS = 64


# ---------------------------------------------------------------------------
# Tool taxonomy (function name -> description + params)
# ---------------------------------------------------------------------------
TOOL_TAXONOMY = {
    # ===== IMAGING (11) =====
    "xray": {
        "description": "X-ray / radiograph imaging",
        "params": {"region": "body region imaged (e.g. chest, abdomen, knee, spine, hand)"}
    },
    "ct": {
        "description": "CT scan / computed tomography",
        "params": {"region": "body region (e.g. head, chest, abdomen, pelvis)", "contrast": "with or without contrast (yes/no/unspecified)"}
    },
    "mri": {
        "description": "MRI / magnetic resonance imaging",
        "params": {"region": "body region (e.g. brain, spine, knee, liver)", "contrast": "with or without contrast (yes/no/unspecified)"}
    },
    "ultrasound": {
        "description": "Ultrasound / sonography / doppler ultrasound",
        "params": {"region": "body region or organ (e.g. abdomen, thyroid, carotid, fetal, pelvic)"}
    },
    "pet_scan": {
        "description": "PET scan / PET-CT",
        "params": {"region": "body region or whole-body", "tracer": "tracer used if specified (e.g. FDG)"}
    },
    "nuclear_scan": {
        "description": "Nuclear medicine scan / scintigraphy (bone scan, thyroid scan, renal scan, HIDA, V/Q scan, etc.)",
        "params": {"type": "type of nuclear scan (e.g. bone scan, thyroid scan, renal scan, HIDA, V/Q, MIBG, gallium)", "region": "body region if applicable"}
    },
    "angiography": {
        "description": "Angiography / arteriography / venography (conventional, CT angiography, MR angiography)",
        "params": {"type": "type (conventional, CT angiography, MR angiography)", "region": "vascular territory (e.g. coronary, cerebral, pulmonary, peripheral)"}
    },
    "fluoroscopy": {
        "description": "Fluoroscopy / barium study / swallow study / contrast study",
        "params": {"type": "type of fluoroscopy study (e.g. barium swallow, barium enema, voiding cystourethrogram, upper GI series)"}
    },
    "mammography": {
        "description": "Mammography / breast imaging",
        "params": {"type": "screening or diagnostic"}
    },
    "dexa_scan": {
        "description": "DEXA scan / bone densitometry",
        "params": {"region": "body region (e.g. hip, spine, whole body)"}
    },

    # ===== CARDIAC (5) =====
    "ecg": {
        "description": "Electrocardiogram / ECG / EKG",
        "params": {"type": "type (standard 12-lead, rhythm strip, etc.)"}
    },
    "echocardiogram": {
        "description": "Echocardiogram / cardiac ultrasound (transthoracic, transesophageal, stress echo)",
        "params": {"type": "type (transthoracic/TTE, transesophageal/TEE, stress echo)"}
    },
    "cardiac_catheterization": {
        "description": "Cardiac catheterization / coronary angiography / hemodynamic study",
        "params": {"type": "type (left heart, right heart, coronary angiography)"}
    },
    "holter_monitor": {
        "description": "Holter monitor / ambulatory ECG monitoring / event recorder",
        "params": {"duration": "monitoring duration if specified (e.g. 24h, 48h, 7d)"}
    },
    "stress_test": {
        "description": "Cardiac stress test (exercise, pharmacologic, nuclear stress test)",
        "params": {"type": "type (exercise, pharmacologic/dobutamine/adenosine, nuclear)"}
    },

    # ===== LAB - BLOOD (16) =====
    "cbc": {
        "description": "Complete blood count / full blood count / hemogram / blood count with differential",
        "params": {"type": "type (CBC, CBC with differential, peripheral smear)"}
    },
    "bmp": {
        "description": "Basic metabolic panel (sodium, potassium, chloride, bicarbonate, BUN, creatinine, glucose, calcium)",
        "params": {}
    },
    "cmp": {
        "description": "Comprehensive metabolic panel / chemistry panel / electrolytes panel",
        "params": {}
    },
    "liver_function_test": {
        "description": "Liver function tests / hepatic panel (AST, ALT, ALP, bilirubin, albumin, GGT)",
        "params": {}
    },
    "renal_function_test": {
        "description": "Renal function tests / kidney function (BUN, creatinine, GFR, electrolytes focused on renal)",
        "params": {}
    },
    "coagulation_panel": {
        "description": "Coagulation studies (PT, INR, PTT/aPTT, fibrinogen, D-dimer, bleeding time)",
        "params": {"type": "specific test if mentioned (e.g. PT/INR, aPTT, D-dimer, fibrinogen)"}
    },
    "blood_gas": {
        "description": "Arterial or venous blood gas analysis (ABG / VBG)",
        "params": {"type": "arterial or venous"}
    },
    "thyroid_function_test": {
        "description": "Thyroid function tests (TSH, T3, T4, free T4, thyroid antibodies)",
        "params": {}
    },
    "tumor_marker": {
        "description": "Tumor markers (CEA, AFP, CA-125, CA 19-9, PSA, beta-hCG, LDH as tumor marker, etc.)",
        "params": {"marker": "specific marker(s) if mentioned"}
    },
    "inflammatory_marker": {
        "description": "Inflammatory markers (CRP, ESR, procalcitonin, ferritin as inflammatory marker)",
        "params": {"marker": "specific marker(s) if mentioned (e.g. CRP, ESR, procalcitonin)"}
    },
    "cardiac_enzyme": {
        "description": "Cardiac enzymes / cardiac biomarkers (troponin, CK-MB, BNP, NT-proBNP, myoglobin)",
        "params": {"marker": "specific marker(s) if mentioned"}
    },
    "lipid_panel": {
        "description": "Lipid panel / cholesterol test (total cholesterol, LDL, HDL, triglycerides)",
        "params": {}
    },
    "hormone_level": {
        "description": "Hormone level tests (cortisol, ACTH, growth hormone, insulin, estrogen, testosterone, PTH, prolactin, catecholamines, etc.)",
        "params": {"hormone": "specific hormone(s) tested"}
    },
    "drug_level": {
        "description": "Therapeutic drug level / drug monitoring / toxicology screen",
        "params": {"drug": "specific drug or substance"}
    },
    "iron_studies": {
        "description": "Iron studies (serum iron, ferritin, TIBC, transferrin saturation)",
        "params": {}
    },
    "blood_test": {
        "description": "Generic / other blood test not fitting above categories (e.g. blood type, Coombs test, hemoglobin electrophoresis, autoantibody panels like ANA/dsDNA/ANCA, complement levels, immunoglobulin levels, vitamin levels, HbA1c, glucose tolerance test)",
        "params": {"test": "specific test name or description"}
    },

    # ===== LAB - MICRO (3) =====
    "culture": {
        "description": "Microbiological culture (blood culture, urine culture, wound culture, sputum culture, CSF culture, etc.)",
        "params": {"source": "specimen source (e.g. blood, urine, wound, sputum, CSF, stool, tissue)"}
    },
    "serology": {
        "description": "Serological test / antibody test (hepatitis panel, HIV test, RPR/VDRL, viral titers, autoimmune antibodies, etc.)",
        "params": {"test": "specific serological test or pathogen"}
    },
    "pcr_test": {
        "description": "PCR / molecular test / nucleic acid amplification test",
        "params": {"target": "target pathogen or gene"}
    },

    # ===== LAB - URINE (2) =====
    "urinalysis": {
        "description": "Urinalysis / urine dipstick / urine microscopy",
        "params": {"type": "type (dipstick, microscopy, full urinalysis)"}
    },
    "urine_test": {
        "description": "Other urine test (24h urine collection, urine electrolytes, urine protein, urine drug screen, urine culture is under culture)",
        "params": {"test": "specific urine test"}
    },

    # ===== LAB - OTHER (2) =====
    "csf_analysis": {
        "description": "CSF analysis / cerebrospinal fluid analysis (cell count, protein, glucose, gram stain, culture is under culture)",
        "params": {}
    },
    "stool_test": {
        "description": "Stool test / fecal test (occult blood, stool culture is under culture, ova and parasites, calprotectin, elastase, fat)",
        "params": {"test": "specific stool test"}
    },

    # ===== PATHOLOGY (5) =====
    "biopsy": {
        "description": "Biopsy (tissue sampling for histological examination - skin biopsy, liver biopsy, kidney biopsy, lymph node biopsy, etc.)",
        "params": {"site": "biopsy site/organ", "type": "biopsy method if specified (excisional, incisional, needle/core, punch, shave)"}
    },
    "histopathology": {
        "description": "Histopathological examination / surgical pathology (examination of excised tissue, tumor grading)",
        "params": {"specimen": "tissue/specimen type"}
    },
    "cytology": {
        "description": "Cytology / Pap smear / fine needle aspiration cytology (FNAC) / fluid cytology",
        "params": {"type": "type (Pap smear, FNAC, fluid cytology, brushing)", "site": "site if applicable"}
    },
    "immunohistochemistry": {
        "description": "Immunohistochemistry (IHC) staining",
        "params": {"markers": "IHC markers tested if specified"}
    },
    "flow_cytometry": {
        "description": "Flow cytometry / immunophenotyping",
        "params": {"specimen": "specimen type (blood, bone marrow, tissue)"}
    },

    # ===== PROCEDURES (6) =====
    "lumbar_puncture": {
        "description": "Lumbar puncture / spinal tap",
        "params": {}
    },
    "paracentesis": {
        "description": "Paracentesis / abdominal tap / ascitic fluid analysis",
        "params": {}
    },
    "thoracentesis": {
        "description": "Thoracentesis / pleural tap / pleural fluid analysis",
        "params": {}
    },
    "bone_marrow_biopsy": {
        "description": "Bone marrow biopsy / bone marrow aspiration",
        "params": {"type": "aspiration, biopsy, or both"}
    },
    "endoscopy": {
        "description": "Endoscopy (upper GI endoscopy/EGD, colonoscopy, bronchoscopy, cystoscopy, sigmoidoscopy, ERCP, EUS, laryngoscopy, rhinoscopy, arthroscopy, hysteroscopy, colposcopy)",
        "params": {"type": "type of endoscopy (EGD, colonoscopy, bronchoscopy, cystoscopy, sigmoidoscopy, ERCP, EUS, laryngoscopy, arthroscopy, etc.)", "region": "region if not clear from type"}
    },
    "arthrocentesis": {
        "description": "Arthrocentesis / joint aspiration / synovial fluid analysis",
        "params": {"joint": "which joint"}
    },

    # ===== NEURO (3) =====
    "eeg": {
        "description": "Electroencephalogram (EEG) / continuous EEG monitoring",
        "params": {"type": "type (routine, continuous/cEEG, video-EEG, sleep-deprived)"}
    },
    "emg": {
        "description": "Electromyography (EMG) / nerve conduction study (NCS) / electroneurography",
        "params": {"region": "body region tested"}
    },
    "evoked_potentials": {
        "description": "Evoked potentials (visual, auditory/BAER, somatosensory)",
        "params": {"type": "type (visual/VEP, auditory/BAER, somatosensory/SSEP)"}
    },

    # ===== PHYSICAL EXAM (1) =====
    "physical_examination": {
        "description": "Physical examination (general exam, neurological exam, cardiovascular exam, respiratory exam, abdominal exam, musculoskeletal exam, skin exam, eye exam via physical methods, ENT exam, rectal exam, breast exam, etc.)",
        "params": {"type": "type of examination (general, neurological, cardiovascular, respiratory, abdominal, musculoskeletal, dermatological, ophthalmologic, ENT, rectal, breast, vital signs, etc.)", "region": "specific body region if applicable"}
    },

    # ===== PULMONARY (1) =====
    "pulmonary_function_test": {
        "description": "Pulmonary function test / spirometry / lung function test / methacholine challenge",
        "params": {"type": "type if specified (spirometry, full PFT, DLCO, methacholine challenge, bronchoprovocation)"}
    },

    # ===== GENETIC (1) =====
    "genetic_test": {
        "description": "Genetic test / karyotyping / FISH / gene sequencing / chromosomal microarray / mutation analysis",
        "params": {"test": "specific genetic test or gene/mutation"}
    },

    # ===== EYE (1) =====
    "eye_exam": {
        "description": "Eye examination (fundoscopy, slit lamp, tonometry, visual acuity, visual field, OCT, fluorescein angiography, retinal exam)",
        "params": {"type": "type of eye exam (fundoscopy, slit lamp, tonometry, visual acuity, visual field, OCT, fluorescein angiography)"}
    },

    # ===== SKIN / ALLERGY (2) =====
    "skin_test": {
        "description": "Skin test (tuberculin/PPD/Mantoux test, patch test, prick test for specific diagnosis, skin scraping, KOH prep, Wood's lamp, dermoscopy)",
        "params": {"type": "type of skin test"}
    },
    "allergy_test": {
        "description": "Allergy test (skin prick test panel, IgE levels, RAST, patch test for allergies)",
        "params": {"type": "type of allergy test", "allergens": "specific allergens if mentioned"}
    },

    # ===== CATCH-ALL (1) =====
    "other": {
        "description": "Any exam/test/procedure not fitting the above categories (e.g. audiometry, sleep study/polysomnography, urodynamic study, manometry, pH study, electrophysiology study, tilt table test)",
        "params": {"description": "description of the exam/test/procedure"}
    },
}


# ---------------------------------------------------------------------------
# Inject allowed enum values into TOOL_TAXONOMY param descriptions
# (must happen before _build_classification_prompt is called)
# ---------------------------------------------------------------------------
for _tool_name, _info in TOOL_TAXONOMY.items():
    if _tool_name in TOOL_PARAM_VALUES:
        for _param_name in _info["params"]:
            if _param_name in TOOL_PARAM_VALUES[_tool_name]:
                _allowed = ", ".join(TOOL_PARAM_VALUES[_tool_name][_param_name])
                _base_desc = _info["params"][_param_name].split("(")[0].split(".")[0].strip()
                _info["params"][_param_name] = f"{_base_desc}. MUST be one of: {_allowed}"


# ---------------------------------------------------------------------------
# Classification prompt builder
# ---------------------------------------------------------------------------
def _build_tool_reference() -> str:
    """Build a formatted string listing all tool functions with their signatures."""
    lines = []
    for func_name, info in TOOL_TAXONOMY.items():
        params = info["params"]
        if params:
            param_str = ", ".join(f'{k}="{v}"' for k, v in params.items())
            sig = f"{func_name}({param_str})"
        else:
            sig = f"{func_name}()"
        lines.append(f"- {sig}: {info['description']}")
    return "\n".join(lines)


def _build_classification_prompt() -> str:
    tool_reference = _build_tool_reference()
    return f"""You are a medical exam classifier. You will be given a list of medical exam/test names extracted from clinical case reports.

For each exam, you must classify it into EXACTLY ONE of the predefined tool functions below and fill in the parameters based on what's described in the exam name.

## Available Tool Functions (you MUST pick from this list, do NOT invent new function names):

{tool_reference}

## Rules:
1. You MUST use EXACTLY ONE function from the list above for each exam. Do NOT create new function names.
2. Fill parameter values based on what is described in the exam name. Use short, lowercase values.
3. Omit parameters that cannot be determined from the exam name (don't include them in the output). NEVER use "unspecified", "unknown", or "not specified" as parameter values — simply omit the parameter instead.
4. If an exam genuinely doesn't fit any specific category, use other(description="...").
5. For physical examination variants (vital signs, temperature, blood pressure, general inspection, palpation, auscultation, percussion, neurological exam, skin exam, etc.), use physical_examination(type="...", region="...").
6. For generic lab work / blood work / biochemistry / "laboratory data" without specifics, use blood_test(test="general laboratory panel").

## Output format:
Return a JSON object mapping each exam name (exactly as given) to its tool call string.
Example:
{{
  "chest X-ray": "xray(region=\\"chest\\")",
  "Complete blood count": "cbc()",
  "Lumbar puncture": "lumbar_puncture()",
  "Temperature": "physical_examination(type=\\"vital signs\\", region=\\"general\\")"
}}

Return ONLY the JSON object, no other text."""


CLASSIFICATION_SYSTEM_PROMPT = _build_classification_prompt()


# ---------------------------------------------------------------------------
# LLM-based exam classification
# ---------------------------------------------------------------------------
def convert_exams_batch(
    exam_names: list[str],
    client: OpenAI,
    model: str,
    max_retries: int = 3,
) -> dict[str, str]:
    """Send a batch of exam names to the LLM and get tool call classifications."""
    user_msg = "Classify these exams:\n" + json.dumps(exam_names)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            result_text = response.choices[0].message.content
            result = json.loads(result_text)

            validated = {}
            for exam in exam_names:
                if exam in result:
                    validated[exam] = result[exam]
                else:
                    for k, v in result.items():
                        if k.lower().strip() == exam.lower().strip():
                            validated[exam] = v
                            break
                    else:
                        if attempt == max_retries - 1:
                            print(f"  WARNING: Missing classification for '{exam}', using other()")
                            validated[exam] = f'other(description="{exam}")'
                        else:
                            print(f"  WARNING: Missing classification for '{exam}', retrying...")
            return validated

        except RateLimitError as e:
            rate_limit_sleep(e, attempt)
            if attempt == max_retries - 1:
                return {exam: f'other(description="{exam}")' for exam in exam_names}
        except Exception as e:
            print(f"  Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {exam: f'other(description="{exam}")' for exam in exam_names}


def classify_all_exams(
    all_exam_names: list[str],
    client: OpenAI,
    model: str,
    batch_size: int,
    checkpoint_path: str,
    max_workers: int = 1,
    use_batch: bool = False,
    poll_interval: int = 60,
) -> dict[str, str]:
    """Process all exam names in batches with checkpointing."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            exam_to_tool_call = json.load(f)
        print(f"Loaded checkpoint with {len(exam_to_tool_call)} existing mappings")
    else:
        exam_to_tool_call = {}

    remaining = [e for e in all_exam_names if e not in exam_to_tool_call]
    print(f"Total unique exams: {len(all_exam_names)}, already processed: {len(exam_to_tool_call)}, remaining: {len(remaining)}")

    if not remaining:
        print("All exams already classified!")
        return exam_to_tool_call

    batches = [remaining[i:i + batch_size] for i in range(0, len(remaining), batch_size)]

    if use_batch:
        from data.batch import build_chat_request, run_batch_job
        requests = [
            build_chat_request(
                custom_id=f"cls_{i}",
                messages=[
                    {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": "Classify these exams:\n" + json.dumps(b)},
                ],
                model=model,
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            for i, b in enumerate(batches)
        ]
        state_file = checkpoint_path.replace(".json", "_batch_state.json")
        print(f"Submitting {len(requests)} classification chunks via Batch API ({len(remaining)} exams)...")
        raw = run_batch_job(client, requests, "exam-classification", poll_interval, state_file)
        for i, b in enumerate(batches):
            content = raw.get(f"cls_{i}")
            try:
                result = json.loads(content) if content else {}
            except json.JSONDecodeError:
                result = {}
            for exam in b:
                if exam in result:
                    exam_to_tool_call[exam] = result[exam]
                else:
                    for k, v in result.items():
                        if k.lower().strip() == exam.lower().strip():
                            exam_to_tool_call[exam] = v
                            break
                    else:
                        print(f"  WARNING: Missing classification for '{exam}', using other()")
                        exam_to_tool_call[exam] = f'other(description="{exam}")'
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        with open(checkpoint_path, "w") as f:
            json.dump(exam_to_tool_call, f, indent=2)
        print(f"Done! Total mappings: {len(exam_to_tool_call)}")
        return exam_to_tool_call

    lock = threading.Lock()

    def _process_batch(batch):
        batch_result = convert_exams_batch(batch, client, model)
        with lock:
            exam_to_tool_call.update(batch_result)
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            with open(checkpoint_path, "w") as f:
                json.dump(exam_to_tool_call, f, indent=2)
        return batch_result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_batch, batch): batch for batch in batches}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Classifying exam batches"):
            future.result()

    print(f"Done! Total mappings: {len(exam_to_tool_call)}")
    return exam_to_tool_call


# ---------------------------------------------------------------------------
# Unspecified argument detection and inference
# ---------------------------------------------------------------------------
_UNSPECIFIED_VALUES = {"unspecified", "unknown", "not specified"}


def _has_unspecified(tool_call_str: str) -> bool:
    """Check if a tool call string contains 'unspecified'/'unknown'/'not specified' argument values."""
    parsed = parse_exam_string(tool_call_str)
    if parsed is None:
        return False
    for val in parsed["arguments"].values():
        if val.lower().strip() in _UNSPECIFIED_VALUES:
            return True
    return False


def _build_inference_prompt() -> str:
    return """You are a medical data analyst. You will be given a list of tool call strings paired with exam findings. Each tool call has one or more argument values set to "unspecified", "unknown", or "not specified".

Your task: Infer the correct argument value from the exam findings. If the findings do not provide enough information, return "CANNOT_INFER".

## Rules:
1. Use short, lowercase values for inferred arguments.
2. Only replace arguments that are "unspecified"/"unknown"/"not specified".
3. Keep all other arguments unchanged.
4. Return the full corrected tool call string, or "CANNOT_INFER" if any unspecified argument cannot be determined.

## Output format:
Return a JSON object mapping each input key to either the corrected tool call string or "CANNOT_INFER".
Example:
{
  "0": "ct(region=\\"head\\", contrast=\\"yes\\")",
  "1": "CANNOT_INFER"
}

Return ONLY the JSON object, no other text."""


def _infer_batch(
    batch: list[tuple[str, str]],
    global_offset: int,
    client: OpenAI,
    model: str,
    system_prompt: str,
    max_retries: int = 3,
) -> dict[int, str | None]:
    """Process a single batch for unspecified arg inference."""
    batch_dict = {}
    for j, (tool_call_str, findings) in enumerate(batch):
        batch_dict[str(global_offset + j)] = {
            "tool_call": tool_call_str,
            "findings": findings,
        }

    user_msg = json.dumps(batch_dict, indent=2)
    batch_results: dict[int, str | None] = {}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            for key, val in result.items():
                idx = int(key)
                if val == "CANNOT_INFER":
                    batch_results[idx] = None
                else:
                    batch_results[idx] = val
            return batch_results
        except RateLimitError as e:
            rate_limit_sleep(e, attempt)
            if attempt == max_retries - 1:
                for j in range(len(batch)):
                    batch_results[global_offset + j] = None
        except Exception as e:
            print(f"  Inference error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                for j in range(len(batch)):
                    batch_results[global_offset + j] = None

    return batch_results


def infer_unspecified_args(
    items: list[tuple[str, str]],
    client: OpenAI,
    model: str,
    batch_size: int = 30,
    max_workers: int = 1,
    max_retries: int = 3,
    use_batch: bool = False,
    poll_interval: int = 60,
) -> dict[int, str | None]:
    """Infer unspecified argument values from exam findings via LLM.

    Args:
        items: List of (tool_call_str, findings_str) pairs.
        client: OpenAI client.
        model: Model name.
        batch_size: Batch size for LLM calls.
        max_workers: Parallel workers for LLM calls.
        max_retries: Max retries per batch.
        use_batch: Use OpenAI Batch API (async, 50% cost savings).
        poll_interval: Seconds between Batch API status polls.

    Returns:
        Dict mapping item index -> corrected tool call string, or None if inference failed.
    """
    if not items:
        return {}

    system_prompt = _build_inference_prompt()
    results: dict[int, str | None] = {}
    batches = [(items[i:i + batch_size], i) for i in range(0, len(items), batch_size)]

    if use_batch:
        from data.batch import build_chat_request, run_batch_job
        requests = []
        for batch, offset in batches:
            batch_dict = {
                str(offset + j): {"tool_call": tc, "findings": f}
                for j, (tc, f) in enumerate(batch)
            }
            requests.append(build_chat_request(
                custom_id=f"infer_{offset}",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(batch_dict, indent=2)},
                ],
                model=model,
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
            ))
        print(f"  Submitting {len(requests)} inference chunks via Batch API...")
        raw = run_batch_job(client, requests, "infer-unspecified-args", poll_interval)
        for batch, offset in batches:
            content = raw.get(f"infer_{offset}")
            try:
                parsed = json.loads(content) if content else {}
            except json.JSONDecodeError:
                parsed = {}
            for j in range(len(batch)):
                idx = offset + j
                val = parsed.get(str(idx))
                results[idx] = None if (val is None or val == "CANNOT_INFER") else val
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_infer_batch, batch, offset, client, model, system_prompt, max_retries): offset
            for batch, offset in batches
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Inferring unspecified args"):
            results.update(future.result())

    return results


# ---------------------------------------------------------------------------
# Enum normalization + validation
# ---------------------------------------------------------------------------

def _normalize_single_tool_call(tool_call_str: str) -> tuple[str, bool]:
    """Normalize param values in a tool call string deterministically.

    Returns (normalized_string, all_resolved) where all_resolved indicates
    whether every param with an enum constraint was successfully normalized.
    """
    parsed = parse_exam_string(tool_call_str)
    if parsed is None:
        return tool_call_str, True  # can't parse -> pass through

    func_name = parsed["name"]
    if func_name not in TOOL_PARAM_VALUES:
        return tool_call_str, True  # no enum constraints for this tool

    new_args = {}
    all_resolved = True
    for param, val in parsed["arguments"].items():
        if param in TOOL_PARAM_VALUES.get(func_name, {}):
            normalized = normalize_param_value(func_name, param, val)
            if normalized is not None:
                new_args[param] = normalized
            else:
                new_args[param] = val  # keep original, mark as unresolved
                all_resolved = False
        else:
            new_args[param] = val  # param not enum-constrained

    return rebuild_tool_call_string(func_name, new_args), all_resolved


def _build_enum_normalization_prompt() -> str:
    """Build a prompt for LLM-based enum normalization of remaining failures."""
    lines = []
    for tool_name, params in TOOL_PARAM_VALUES.items():
        for param_name, allowed in params.items():
            lines.append(f"  {tool_name}.{param_name}: {', '.join(allowed)}")
    enum_ref = "\n".join(lines)

    return f"""You are a medical data normalizer. You will be given tool call strings with parameter values that need to be mapped to predefined enum values.

## Allowed enum values:
{enum_ref}

## Rules:
1. Map each parameter value to the closest matching allowed enum value.
2. If a value truly cannot be mapped, return "CANNOT_NORMALIZE" for that entry.
3. Return the full corrected tool call string with normalized values.

## Output format:
Return a JSON object mapping each input key to either the corrected tool call string or "CANNOT_NORMALIZE".
Example:
{{
  "0": "ct(region=\\"chest\\", contrast=\\"yes\\")",
  "1": "CANNOT_NORMALIZE"
}}

Return ONLY the JSON object, no other text."""


def _normalize_batch_llm(
    batch: list[tuple[str, str]],
    global_offset: int,
    client,
    model: str,
    system_prompt: str,
    max_retries: int = 3,
) -> dict[int, str | None]:
    """Send a batch of unresolved tool calls to LLM for normalization."""
    batch_dict = {}
    for j, (tool_call_str, findings) in enumerate(batch):
        batch_dict[str(global_offset + j)] = {
            "tool_call": tool_call_str,
            "findings": findings,
        }

    user_msg = json.dumps(batch_dict, indent=2)
    results: dict[int, str | None] = {}

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content)
            for key, val in result.items():
                idx = int(key)
                if isinstance(val, dict):
                    val = val.get("tool_call", val.get("corrected", "CANNOT_NORMALIZE"))
                if val == "CANNOT_NORMALIZE":
                    results[idx] = None
                elif isinstance(val, str):
                    results[idx] = val
                else:
                    results[idx] = None
            return results
        except RateLimitError as e:
            rate_limit_sleep(e, attempt)
            if attempt == max_retries - 1:
                for j in range(len(batch)):
                    results[global_offset + j] = None
        except Exception as e:
            print(f"  Normalization error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                for j in range(len(batch)):
                    results[global_offset + j] = None

    return results


def normalize_tool_call_params(
    df: pd.DataFrame,
    client,
    model: str,
    batch_size: int = 30,
    max_workers: int = 1,
    use_batch: bool = False,
    poll_interval: int = 60,
) -> pd.DataFrame:
    """Normalize tool call parameter values to predefined enums.

    Phase 1: Deterministic normalization (direct match, alias, substring).
    Phase 2: LLM fallback for remaining failures.
    """
    # Phase 1: deterministic
    det_resolved = 0
    det_unresolved_items = []
    det_unresolved_locs = []

    for idx in df.index:
        for i in range(1, 10):
            col = f"exam{i}"
            val = df.at[idx, col]
            if pd.isna(val):
                continue
            normalized, resolved = _normalize_single_tool_call(str(val))
            df.at[idx, col] = normalized
            if resolved and normalized != str(val):
                det_resolved += 1
            elif not resolved:
                findings_col = f"exam{i}_findings"
                findings = str(df.at[idx, findings_col]) if findings_col in df.columns and pd.notna(df.at[idx, findings_col]) else ""
                det_unresolved_items.append((normalized, findings))
                det_unresolved_locs.append((idx, i))

    print(f"  Deterministic normalization: {det_resolved} tool calls updated")
    print(f"  Remaining unresolved: {det_unresolved_items.__len__()} tool calls")

    # Phase 2: LLM fallback
    if det_unresolved_items:
        print(f"  Sending {len(det_unresolved_items)} unresolved tool calls to LLM...")
        system_prompt = _build_enum_normalization_prompt()
        results: dict[int, str | None] = {}
        batches = [
            (det_unresolved_items[i:i + batch_size], i)
            for i in range(0, len(det_unresolved_items), batch_size)
        ]

        if use_batch:
            from data.batch import build_chat_request, run_batch_job
            requests = []
            for batch, offset in batches:
                batch_dict = {
                    str(offset + j): {"tool_call": tc, "findings": f}
                    for j, (tc, f) in enumerate(batch)
                }
                requests.append(build_chat_request(
                    custom_id=f"norm_{offset}",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": json.dumps(batch_dict, indent=2)},
                    ],
                    model=model,
                    temperature=0.0,
                    max_tokens=4096,
                    response_format={"type": "json_object"},
                ))
            print(f"  Submitting {len(requests)} normalization chunks via Batch API...")
            raw = run_batch_job(client, requests, "enum-normalization", poll_interval)
            for batch, offset in batches:
                content = raw.get(f"norm_{offset}")
                try:
                    parsed = json.loads(content) if content else {}
                except json.JSONDecodeError:
                    parsed = {}
                for j in range(len(batch)):
                    idx = offset + j
                    val = parsed.get(str(idx))
                    if isinstance(val, dict):
                        val = val.get("tool_call", val.get("corrected", "CANNOT_NORMALIZE"))
                    results[idx] = None if (not val or val == "CANNOT_NORMALIZE") else val
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _normalize_batch_llm, batch, offset, client, model,
                        system_prompt,
                    ): offset
                    for batch, offset in batches
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="LLM enum normalization"):
                    results.update(future.result())

        llm_fixed = 0
        for item_idx, (row_idx, col_num) in enumerate(det_unresolved_locs):
            corrected = results.get(item_idx)
            if corrected is not None:
                df.at[row_idx, f"exam{col_num}"] = corrected
                llm_fixed += 1
        print(f"  LLM normalization: {llm_fixed}/{len(det_unresolved_items)} resolved")

    return df


def validate_tool_call_enums(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any exam has a parameter value outside the allowed enums."""
    before = len(df)
    invalid_count = 0
    rows_to_drop = set()

    for idx in df.index:
        for i in range(1, 10):
            val = df.at[idx, f"exam{i}"]
            if pd.isna(val):
                continue
            parsed = parse_exam_string(str(val))
            if parsed is None:
                continue
            func_name = parsed["name"]
            if func_name not in TOOL_PARAM_VALUES:
                continue
            for param, pval in parsed["arguments"].items():
                allowed = TOOL_PARAM_VALUES[func_name].get(param)
                if allowed is not None and pval not in allowed:
                    rows_to_drop.add(idx)
                    invalid_count += 1
                    break

    df = df.drop(index=rows_to_drop)
    print(f"  Enum validation: dropped {len(rows_to_drop)} rows ({invalid_count} invalid params), {len(df)} rows remain")
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run_filter(args):
    """Run the filtering + classification pipeline."""
    # Load extracted CSV
    print(f"Loading data from {args.input} ...")
    print(f"Using API base URL: {args.base_url}")
    print(f"Using model: {args.model}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")

    # Optional row range slicing
    start = getattr(args, "start", None)
    end = getattr(args, "end", None)
    if start is not None or end is not None:
        df = df.iloc[start:end]
        print(f"Sliced to rows [{start}:{end}]: {len(df)} rows")

    # --- Step A: Drop rows with exam10 (too many exams) ---
    df = df[df['exam10'].isna()]
    print(f"After dropping exam10 rows: {len(df)} rows")

    # --- Step B: Drop rows without diagnosis or self_reported_symptoms ---
    df = df[df['diagnosis'].notna()]
    df = df[df['self_reported_symptoms'].notna()]
    print(f"After requiring diagnosis + symptoms: {len(df)} rows")

    # --- Step C: Collect all unique exam names ---
    all_exams_list = []
    for i in range(1, 10):
        all_exams_list.extend(df[f'exam{i}'].dropna().tolist())
    all_exam_names = sorted(set(all_exams_list))
    print(f"Unique exam names: {len(all_exam_names)}")

    # --- Step D: Classify exams via LLM ---
    client = OpenAI(base_url=args.base_url, timeout=180.0)
    use_batch = getattr(args, "use_batch_api", False)
    poll_interval = getattr(args, "batch_poll_interval", 60)
    exam_to_tool_call = classify_all_exams(
        all_exam_names, client, args.model, args.batch_size, args.checkpoint,
        max_workers=args.max_workers,
        use_batch=use_batch,
        poll_interval=poll_interval,
    )

    # --- Step E: Map exam columns to tool call strings ---
    for i in range(1, 10):
        col = f"exam{i}"
        df[col] = df[col].map(exam_to_tool_call)

    # --- Step E2: Normalize param values to enums ---
    print("Normalizing tool call parameter values to enums...")
    df = normalize_tool_call_params(
        df, client, args.model, args.batch_size,
        max_workers=args.max_workers,
        use_batch=use_batch,
        poll_interval=poll_interval,
    )

    # --- Step E3: Validate + filter non-enum values ---
    print("Validating enum compliance...")
    df = validate_tool_call_enums(df)

    # --- Step F: Fix "unspecified" argument values ---
    unspec_items = []  # list of (tool_call_str, findings_str)
    unspec_locs = []   # list of (row_idx, exam_col_num) for in-place update
    for idx in df.index:
        for i in range(1, 10):
            tool_call_str = df.at[idx, f"exam{i}"]
            if pd.notna(tool_call_str) and _has_unspecified(str(tool_call_str)):
                findings_col = f"exam{i}_findings"
                findings = str(df.at[idx, findings_col]) if findings_col in df.columns and pd.notna(df.at[idx, findings_col]) else ""
                unspec_items.append((str(tool_call_str), findings))
                unspec_locs.append((idx, i))

    if unspec_items:
        print(f"Found {len(unspec_items)} exam entries with 'unspecified' argument values, attempting inference...")
        inferred = infer_unspecified_args(
            unspec_items, client, args.model, args.batch_size,
            max_workers=args.max_workers,
            use_batch=use_batch,
            poll_interval=poll_interval,
        )

        fixed_count = 0
        for item_idx, (row_idx, col_num) in enumerate(unspec_locs):
            corrected = inferred.get(item_idx)
            if corrected is not None and not _has_unspecified(corrected):
                df.at[row_idx, f"exam{col_num}"] = corrected
                fixed_count += 1
        print(f"  Successfully inferred {fixed_count}/{len(unspec_items)} entries")

        # Filter out rows that still have unspecified values
        before = len(df)
        for i in range(1, 10):
            mask = df[f"exam{i}"].apply(lambda x: _has_unspecified(str(x)) if pd.notna(x) else False)
            df = df[~mask]
        print(f"  Removed {before - len(df)} rows still containing 'unspecified' values")
    else:
        print("No 'unspecified' argument values found")

    # --- Step F2: Final enum validation (safety net after F) ---
    print("Final enum validation after unspecified-arg inference...")
    df = validate_tool_call_enums(df)

    # --- Step G: Remove rows containing other() exams ---
    for i in range(1, 10):
        df = df[~df[f'exam{i}'].fillna("").astype(str).str.startswith("other")]
    print(f"After removing other() exams: {len(df)} rows")

    # --- Step H: Split into train / rl / eval sets ---
    rl_fraction = float(getattr(args, "rl_fraction", 0.0))
    eval_fraction = float(getattr(args, "eval_fraction", 0.0))
    holdout_fraction = rl_fraction + eval_fraction

    if rl_fraction < 0 or eval_fraction < 0:
        raise ValueError("rl_fraction and eval_fraction must be non-negative.")
    if holdout_fraction >= 1.0:
        print(f"WARNING: rl_fraction ({rl_fraction}) + eval_fraction ({eval_fraction}) >= 1.0. "
              f"Skipping split — saving everything to {args.output}.")
    elif holdout_fraction > 0:
        # Shuffle before splitting for a random split
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        total_rows = len(df)
        eval_size = int(total_rows * eval_fraction)
        rl_size = int(total_rows * rl_fraction)
        holdout = eval_size + rl_size

        if holdout == 0:
            print(f"WARNING: fractions are too small for dataset size ({total_rows} rows). "
                  f"Skipping split — saving everything to {args.output}.")
        else:
            eval_df = df.iloc[:eval_size]
            rl_df = df.iloc[eval_size:eval_size + rl_size]
            train_df = df.iloc[eval_size + rl_size:]

            output_dir = os.path.dirname(args.output) or "."
            base, ext = os.path.splitext(os.path.basename(args.output))

            if eval_size > 0:
                eval_path = os.path.join(output_dir, f"{base}_eval{ext}")
                eval_df.to_csv(eval_path, index=False)
                print(f"Saved eval split to {eval_path} ({len(eval_df)} rows, fraction={eval_fraction})")

            if rl_size > 0:
                rl_path = os.path.join(output_dir, f"{base}_rl{ext}")
                rl_df.to_csv(rl_path, index=False)
                print(f"Saved RL split to {rl_path} ({len(rl_df)} rows, fraction={rl_fraction})")

            # Save train split as the main output
            df = train_df
            print(f"Train split: {len(df)} rows")

    # --- Save ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description="Filter extracted CSV and classify exams to tool calls.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint JSON path for exam classification")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible API base URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for LLM classification")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Parallel workers")
    parser.add_argument("--start", type=int, default=None, help="Start row index (inclusive) to process")
    parser.add_argument("--end", type=int, default=None, help="End row index (exclusive) to process")
    parser.add_argument("--rl-fraction", type=float, default=0.0, help="Fraction of rows to hold out for RL training")
    parser.add_argument("--eval-fraction", type=float, default=0.0, help="Fraction of rows to hold out for evaluation")
    parser.add_argument("--use-batch-api", action="store_true",
                        help="Use OpenAI Batch API for LLM calls (async, ~50%% cost savings; requires api.openai.com)")
    parser.add_argument("--batch-poll-interval", type=int, default=60,
                        help="Seconds between Batch API status polls (default: 60)")
    args = parser.parse_args()
    run_filter(args)


if __name__ == "__main__":
    main()
