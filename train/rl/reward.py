"""Reward function for DAPO-based RL training.

Rewards:
    R = w_diagnosis * R_diagnosis + w_tool_match * R_tool_match
        - w_cost * R_cost - w_format * R_format_penalty
    R_diagnosis: cosine similarity (clipped to [0, 1]) between predicted and ground-truth
                 diagnosis embeddings (MedEmbed), or UMLS CUI match, or LLM judge
    R_tool_match: ToolRL-style three-level scoring (name Jaccard + param name Jaccard
                  + param value exact match), normalized to [0, 1]
    R_cost: normalized cost penalty for unnecessary exams (not in ground truth),
            based on exam cost and patient discomfort levels
    R_format_penalty: fraction of tool-call turns with format errors (malformed JSON
                      args or multiple tool calls in a single turn), in [0, 1]

Diagnosis methods:
    - "embedding": cosine similarity via MedEmbed (default)
    - "umls": map both to UMLS CUI codes via QuickUMLS, binary match
    - "llm_judge": LLM-as-judge via OpenAI-compatible API, Jaccard score
"""

import json
import logging
import os
import re
from collections import Counter

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

logger = logging.getLogger(__name__)

_model = None
_umls_matcher = None
_umls_cache: dict[str, str] = {}
_llm_client = None

# ---------------------------------------------------------------------------
# Exam cost & patient discomfort levels (Low=1, Medium=2, High=3)
# Combined score per exam = cost_level + discomfort_level (range: 2-6)
# ---------------------------------------------------------------------------
EXAM_COST = {
    # Physical Exam
    "physical_examination":     (1, 1),
    # Imaging
    "xray":                     (1, 1),
    "ultrasound":               (2, 1),
    "mammography":              (1, 2),
    "dexa_scan":                (1, 1),
    "fluoroscopy":              (2, 2),
    "ct":                       (3, 1),
    "mri":                      (3, 2),
    "angiography":              (3, 3),
    "nuclear_scan":             (3, 2),
    "pet_scan":                 (3, 1),
    # Cardiac
    "ecg":                      (1, 1),
    "holter_monitor":           (2, 1),
    "echocardiogram":           (2, 1),
    "echocardiography":         (2, 1),
    "stress_test":              (2, 2),
    "cardiac_catheterization":  (3, 3),
    # Lab - Blood
    "cbc":                      (1, 1),
    "bmp":                      (1, 1),
    "cmp":                      (1, 1),
    "lipid_panel":              (1, 1),
    "liver_function_test":      (1, 1),
    "renal_function_test":      (1, 1),
    "thyroid_function_test":    (1, 1),
    "iron_studies":             (1, 1),
    "coagulation_panel":        (1, 1),
    "blood_gas":                (2, 2),
    "inflammatory_marker":      (1, 1),
    "cardiac_enzyme":           (1, 1),
    "hormone_level":            (2, 1),
    "tumor_marker":             (2, 1),
    "drug_level":               (2, 1),
    "blood_test":               (1, 1),
    # Lab - Microbiology
    "culture":                  (2, 1),
    "serology":                 (2, 1),
    "pcr_test":                 (2, 1),
    # Lab - Urine
    "urinalysis":               (1, 1),
    "urine_test":               (1, 1),
    "urine_cytology":           (2, 1),
    # Lab - Other
    "csf_analysis":             (2, 3),
    "stool_test":               (1, 1),
    # Pathology
    "cytology":                 (2, 2),
    "histopathology":           (2, 1),
    "biopsy":                   (3, 3),
    "immunohistochemistry":     (2, 1),
    "flow_cytometry":           (2, 1),
    # Procedures
    "lumbar_puncture":          (3, 3),
    "paracentesis":             (3, 3),
    "thoracentesis":            (3, 3),
    "arthrocentesis":           (2, 2),
    "bone_marrow_biopsy":       (3, 3),
    "endoscopy":                (3, 2),
    "amniocentesis":            (3, 3),
    "hysterosalpingogram":      (2, 2),
    # Neuro
    "eeg":                      (2, 1),
    "emg":                      (2, 2),
    "evoked_potentials":        (2, 1),
    # Pulmonary
    "pulmonary_function_test":  (2, 1),
    # Genetic
    "genetic_test":             (3, 1),
    "molecular_test":           (3, 1),
    # Eye
    "eye_exam":                 (1, 1),
    # Skin / Allergy
    "skin_test":                (1, 1),
    "allergy_test":             (2, 1),
    # Other
    "rheumatoid_factor":        (1, 1),
    "viral_load":               (2, 1),
    "other":                    (2, 2),
}

# Maximum possible combined score for a single exam (cost=3 + discomfort=3)
_MAX_EXAM_SCORE = 6


def _get_embed_model():
    """Lazy-load the medical embedding model (singleton)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("abhinand/MedEmbed-base-v0.1")
    return _model


def _get_umls_matcher(quickumls_db: str):
    """Lazy-load QuickUMLS matcher (singleton)."""
    global _umls_matcher
    if _umls_matcher is None:
        try:
            from quickumls import QuickUMLS
        except ImportError as exc:
            raise RuntimeError(
                "quickumls is not installed. Install it with: pip install quickumls"
            ) from exc
        _umls_matcher = QuickUMLS(quickumls_db, overlapping_criteria="length")
        logger.info("Loaded QuickUMLS matcher from %s", quickumls_db)
    return _umls_matcher


def _map_to_cui(text: str, quickumls_db: str) -> str:
    """Map a diagnosis string to a UMLS CUI. Returns empty string if no match."""
    global _umls_cache
    if text in _umls_cache:
        return _umls_cache[text]
    matcher = _get_umls_matcher(quickumls_db)
    matches = matcher.match(text, best_match=True)
    cui = ""
    if matches:
        best_score = (-1.0, -1)
        for span_matches in matches:
            for candidate in span_matches:
                similarity = float(candidate.get("similarity", 0.0))
                ngram_len = len(str(candidate.get("ngram", "")))
                score = (similarity, ngram_len)
                if score > best_score:
                    best_score = score
                    cui = str(candidate.get("cui", ""))
    _umls_cache[text] = cui
    return cui


def compute_diagnosis_umls(
    diagnosis: str,
    ground_truth: str,
    quickumls_db: str = "data/database/umls/db",
) -> float:
    """Compute diagnosis reward via UMLS CUI matching.

    Maps both strings to UMLS CUIs and returns 1.0 if they match, 0.0 otherwise.
    If either string fails to map to a CUI, returns 0.0.
    """
    pred_cui = _map_to_cui(diagnosis, quickumls_db)
    gt_cui = _map_to_cui(ground_truth, quickumls_db)
    if not pred_cui or not gt_cui:
        return 0.0
    return 1.0 if pred_cui == gt_cui else 0.0


def _get_llm_client():
    """Lazy-load OpenAI client for LLM judge (singleton)."""
    global _llm_client
    if _llm_client is None:
        from openai import OpenAI
        _llm_client = OpenAI()  # uses OPENAI_API_KEY, OPENAI_BASE_URL env vars
    return _llm_client


_LLM_JUDGE_PROMPT = """\
You are a medical expert evaluating a diagnosis prediction.

Ground truth diagnosis: {ground_truth}
Predicted diagnosis: {prediction}

Instructions:
1. Identify the individual medical conditions in the ground truth. Note that a comma \
may be part of a single condition name (e.g. "seminoma, classic type" is ONE condition, \
"Follicular lymphoma, grade 2" is ONE condition). Semicolons or "and" typically separate \
distinct conditions.
2. Identify the individual medical conditions in the prediction, using the same logic.
3. For each ground truth condition, check if any predicted condition refers to the same \
disease. Consider synonyms (e.g. "heart attack" = "myocardial infarction"), abbreviations, \
and minor wording differences.
4. Count how many ground truth conditions have a match in the predictions.

You MUST respond in exactly this format (numbers only):
gt_count: <number of ground truth conditions>
pred_count: <number of predicted conditions>
matched: <number of matched conditions>"""


def _parse_judge_response(response: str) -> tuple[int, int, int]:
    """Parse structured judge response into (gt_count, pred_count, matched)."""
    gt_count, pred_count, matched = 1, 1, 0
    for line in response.split("\n"):
        line = line.strip().lower()
        if line.startswith("gt_count:"):
            try:
                gt_count = max(1, int(line.split(":")[1].strip()))
            except (ValueError, IndexError):
                pass
        elif line.startswith("pred_count:"):
            try:
                pred_count = max(1, int(line.split(":")[1].strip()))
            except (ValueError, IndexError):
                pass
        elif line.startswith("matched:"):
            try:
                matched = max(0, int(line.split(":")[1].strip()))
            except (ValueError, IndexError):
                pass
    matched = min(matched, gt_count, pred_count)
    return gt_count, pred_count, matched


def compute_diagnosis_llm_judge(
    diagnosis: str,
    ground_truth: str,
    judge_model: str = "gpt-4.1-mini",
) -> float:
    """Compute diagnosis reward via LLM-as-judge with Jaccard scoring.

    Uses the same prompt and scoring as evaluation/metrics.py to ensure
    train/eval consistency. Returns Jaccard score in [0, 1].
    Falls back to 0.0 on API errors.
    """
    client = _get_llm_client()
    try:
        response = client.chat.completions.create(
            model=judge_model,
            messages=[
                {
                    "role": "user",
                    "content": _LLM_JUDGE_PROMPT.format(
                        ground_truth=ground_truth,
                        prediction=diagnosis,
                    ),
                }
            ],
            temperature=0.7,
            max_tokens=50,
        )
        content = response.choices[0].message.content.strip()
        gt_count, pred_count, matched = _parse_judge_response(content)
        union = gt_count + pred_count - matched
        return matched / union if union > 0 else 0.0
    except Exception:
        logger.warning(
            "LLM judge failed for diagnosis=%r vs ground_truth=%r",
            diagnosis,
            ground_truth,
            exc_info=True,
        )
        return 0.0


def _jaccard(set1: set, set2: set) -> float:
    """Jaccard similarity. Returns 1.0 if both empty."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def _multiset_jaccard(items1: list[str], items2: list[str]) -> float:
    """Multiset Jaccard similarity over item counts.

    J(A, B) = sum_x min(cA(x), cB(x)) / sum_x max(cA(x), cB(x)).
    Returns 1.0 if both multisets are empty.
    """
    c1 = Counter(items1)
    c2 = Counter(items2)
    if not c1 and not c2:
        return 1.0

    keys = set(c1) | set(c2)
    intersection = sum(min(c1[k], c2[k]) for k in keys)
    union = sum(max(c1[k], c2[k]) for k in keys)
    return intersection / union if union > 0 else 1.0


def compute_tool_reward(
    predicted: list[dict],
    ground_truth: list[dict],
) -> float:
    """ToolRL-style three-level tool call reward.

    Level 1: Tool name Jaccard (multiset-level, count-aware)
    Level 2: Param name Jaccard (per matched GT tool, greedy matched to pred)
    Level 3: Param value exact match count (case-insensitive, str-cast)

    Returns reward in [0, 1].
    """
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0

    # Level 1: Tool name Jaccard (count-aware; duplicates are penalized)
    pred_names = [t["name"] for t in predicted]
    gt_names = [t["name"] for t in ground_truth]
    name_jaccard = _multiset_jaccard(pred_names, gt_names)

    # Greedy match: for each GT tool, find best matching predicted tool by name
    pred_available = list(predicted)
    param_score = 0.0
    max_param_score = 0.0

    for gt_tool in ground_truth:
        gt_params = gt_tool.get("arguments", {})
        max_param_score += 1.0 + len(gt_params)  # param_name_jaccard(=1) + all values match

        best_idx = None
        best_score = -1.0
        for i, pred_tool in enumerate(pred_available):
            if pred_tool["name"] != gt_tool["name"]:
                continue
            pred_params = pred_tool.get("arguments", {})
            # Level 2: param name Jaccard
            pn_jaccard = _jaccard(set(gt_params.keys()), set(pred_params.keys()))
            # Level 3: param value exact match
            val_matches = sum(
                1 for k in gt_params
                if k in pred_params and str(gt_params[k]).lower() == str(pred_params.get(k, "")).lower()
            )
            score = pn_jaccard + val_matches
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx is not None:
            param_score += best_score
            pred_available.pop(best_idx)

    total_score = name_jaccard + param_score
    max_possible = 1.0 + max_param_score
    return total_score / max_possible if max_possible > 0 else 0.0


def compute_cost_penalty(
    predicted_tools: list[dict],
    gt_tools: list[dict],
    uniform_exam_cost: bool = False,
) -> float:
    """Compute cost penalty for unnecessary exams.

    Only penalizes exams that are NOT in the ground truth set (by name).
    Exams that match ground truth incur zero penalty regardless of cost,
    so the model is not discouraged from ordering necessary expensive exams.
    Each GT exam name grants one free call; duplicate calls of the same
    GT exam name (e.g. ct with different params) are penalized.

    Returns total (cost + discomfort) of unnecessary exams divided by the
    max per-exam score. Not bounded to [0, 1] — scales linearly with both
    the number and costliness of unnecessary exams. Use w_cost to control
    the magnitude in the final reward.

    If ``uniform_exam_cost`` is True, every unnecessary exam contributes a
    flat 1 regardless of its EXAM_COST entry. This isolates "penalize any
    unnecessary exam" from "penalize expensive/invasive ones more" —
    useful as an ablation of the cost differentiation itself.
    """
    if not predicted_tools:
        return 0.0

    # Count how many times each GT exam name can be called for free
    gt_name_budget = {}
    for t in gt_tools:
        gt_name_budget[t["name"]] = gt_name_budget.get(t["name"], 0) + 1

    total_penalty = 0.0
    used_budget = {}
    for tool in predicted_tools:
        name = tool["name"]
        budget = gt_name_budget.get(name, 0)
        used = used_budget.get(name, 0)
        if used < budget:
            # Free call — matches a GT exam
            used_budget[name] = used + 1
        else:
            if uniform_exam_cost:
                total_penalty += 1.0
            else:
                cost, discomfort = EXAM_COST.get(name, (2, 2))
                total_penalty += cost + discomfort

    return total_penalty / _MAX_EXAM_SCORE


def compute_tool_cost_totals(
    predicted_tools: list[dict],
    gt_tools: list[dict],
) -> dict[str, float]:
    """Compute raw tool-cost totals for logging.

    Returns:
        Dict containing:
            total_tool_cost: Sum of cost+discomfort for all predicted tools.
            unnecessary_tool_cost: Sum of cost+discomfort for predicted tools not in GT by name.
            gt_tool_cost: Sum of cost+discomfort for GT tools.
    """
    gt_names = {t["name"] for t in gt_tools}

    total_tool_cost = 0.0
    unnecessary_tool_cost = 0.0
    for tool in predicted_tools:
        name = tool["name"]
        cost, discomfort = EXAM_COST.get(name, (2, 2))
        exam_cost = float(cost + discomfort)
        total_tool_cost += exam_cost
        if name not in gt_names:
            unnecessary_tool_cost += exam_cost

    gt_tool_cost = 0.0
    for tool in gt_tools:
        name = tool["name"]
        cost, discomfort = EXAM_COST.get(name, (2, 2))
        gt_tool_cost += float(cost + discomfort)

    return {
        "total_tool_cost": total_tool_cost,
        "unnecessary_tool_cost": unnecessary_tool_cost,
        "gt_tool_cost": gt_tool_cost,
    }


def compute_format_penalty(
    tool_call_turn_attempts: int,
    tool_call_turn_format_errors: int,
) -> float:
    """Compute format penalty as fraction of tool-call turns with errors.

    A tool-call turn has a format error if:
    - The arguments JSON failed to parse, OR
    - Multiple tool calls were emitted in a single turn.

    Returns a value in [0, 1]. 0 = all turns formatted correctly,
    1 = every turn had a format error.  If no tool-call turns were
    attempted, returns 0 (no penalty).
    """
    if tool_call_turn_attempts <= 0:
        return 0.0
    return tool_call_turn_format_errors / tool_call_turn_attempts


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    w_diagnosis: float = 1.0,
    w_tool_match: float = 0.5,
    w_cost: float = 0.0,
    w_format: float = 0.0,
    uniform_exam_cost: bool = False,
    diagnosis_clamp_enable: bool = False,
    diagnosis_clamp_threshold: float = 0.7,
    diagnosis_method: str = "embedding",
    diagnosis_judge_model: str = "gpt-4.1-mini",
    diagnosis_quickumls_db: str = "data/database/umls/db",
) -> float:
    """Compute reward for a completed rollout.

    Args:
        data_source: Dataset identifier (unused, kept for verl API compatibility).
        solution_str: The model's full output (last message or full conversation).
        ground_truth: The correct diagnosis string.
        extra_info: Dict with tool stats. Must contain "predicted_tools" and "gt_tools"
            (lists of {"name": ..., "arguments": {...}}).
            Also reads "tool_call_turn_attempts" and "tool_call_turn_format_errors"
            for format penalty computation.
            Individual reward components are written back here for logging.
        w_diagnosis: Weight for diagnosis reward.
        w_tool_match: Weight for tool match reward.
        w_cost: Weight for cost penalty (subtracted). Default 0.0 (disabled).
        w_format: Weight for format penalty (subtracted). Default 0.0 (disabled).
        diagnosis_clamp_enable: If True, clamp r_diagnosis to 0 or 1.
            Only applies to "embedding" method.
        diagnosis_clamp_threshold: Threshold for clamping (1 if sim > threshold, else 0).
        diagnosis_method: How to compute R_diagnosis. One of:
            "embedding" - cosine similarity via MedEmbed (default, continuous [0,1])
            "umls" - UMLS CUI code match (binary 0/1)
            "llm_judge" - LLM-as-judge (Jaccard scoring, continuous [0,1])
        diagnosis_judge_model: Model name for LLM judge (only used if method="llm_judge").
        diagnosis_quickumls_db: Path to QuickUMLS database (only used if method="umls").

    Returns:
        Float reward value.
    """
    extra_info = extra_info or {}
    diagnosis = extract_diagnosis(solution_str)

    # --- R_diagnosis ---
    if diagnosis is None:
        r_diagnosis = 0.0
    elif diagnosis_method == "umls":
        r_diagnosis = compute_diagnosis_umls(
            diagnosis, ground_truth, quickumls_db=diagnosis_quickumls_db,
        )
    elif diagnosis_method == "llm_judge":
        r_diagnosis = compute_diagnosis_llm_judge(
            diagnosis, ground_truth, judge_model=diagnosis_judge_model,
        )
    else:
        # Default: embedding cosine similarity
        model = _get_embed_model()
        embs = model.encode([diagnosis, ground_truth], normalize_embeddings=True)
        similarity = float(cos_sim(embs[0], embs[1]).item())
        r_diagnosis = max(0.0, similarity)
        # Optionally clamp to binary 0/1 (only for embedding method)
        if diagnosis_clamp_enable:
            r_diagnosis = 1.0 if r_diagnosis > diagnosis_clamp_threshold else 0.0

    # --- R_tool_match (ToolRL three-level) ---
    predicted_tools = extra_info.get("predicted_tools", [])
    gt_tools = extra_info.get("gt_tools", [])
    r_tool_match = compute_tool_reward(predicted_tools, gt_tools)

    # --- R_cost (penalty for unnecessary expensive/invasive exams) ---
    r_cost = compute_cost_penalty(predicted_tools, gt_tools, uniform_exam_cost=uniform_exam_cost)

    # --- R_format (penalty for malformed tool-call turns) ---
    r_format = compute_format_penalty(
        extra_info.get("tool_call_turn_attempts", 0),
        extra_info.get("tool_call_turn_format_errors", 0),
    )

    total = (
        w_diagnosis * r_diagnosis
        + w_tool_match * r_tool_match
        - w_cost * r_cost
        - w_format * r_format
    )

    # Write components back for logging
    extra_info["r_diagnosis"] = r_diagnosis
    extra_info["r_tool_match"] = r_tool_match
    extra_info["r_cost"] = r_cost
    extra_info["r_format"] = r_format

    return total


def compute_similarity(diagnosis: str, ground_truth: str) -> float:
    """Compute raw cosine similarity between diagnosis and ground truth embeddings.

    Returns a float in [-1.0, 1.0] (typically [0.0, 1.0] for medical terms).
    """
    model = _get_embed_model()
    embs = model.encode([diagnosis, ground_truth], normalize_embeddings=True)
    return float(cos_sim(embs[0], embs[1]).item())


def extract_diagnosis(text: str) -> str | None:
    """Extract diagnosis from [DIAGNOSIS: ...] tag in text."""
    match = re.search(r"\[DIAGNOSIS:\s*(.+?)\]", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def match_diagnosis(predicted: str, ground_truth: str) -> bool:
    """Check if predicted diagnosis matches ground truth.

    Uses case-insensitive bidirectional substring matching:
    the prediction matches if it is a substring of the ground truth
    or the ground truth is a substring of the prediction.
    """
    pred_lower = predicted.lower().strip()
    gt_lower = ground_truth.lower().strip()
    return pred_lower in gt_lower or gt_lower in pred_lower
