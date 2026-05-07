"""
Noise injection for conversation generation.

Two categories:
- Patient noise (Phases 1-2): misreported symptoms, body part confusion, vague answers, etc.
- Exam noise (Phase 3): modified findings in tool responses

Noise is planned once per conversation, then applied per-turn via system hints
injected into patient_messages before each LLM call.
"""

import random
import re
from dataclasses import dataclass, field
from random import Random


# ---------------------------------------------------------------------------
# Dictionaries
# ---------------------------------------------------------------------------

BODY_PART_ADJACENCY = {
    "stomach": ["liver", "spleen", "intestines", "pancreas"],
    "liver": ["stomach", "gallbladder", "spleen", "right kidney"],
    "chest": ["upper abdomen", "shoulder", "back"],
    "head": ["neck", "face", "scalp"],
    "knee": ["thigh", "shin", "calf"],
    "lower back": ["hip", "upper back", "buttock"],
    "shoulder": ["neck", "upper arm", "chest"],
    "abdomen": ["stomach", "pelvis", "lower back"],
    "throat": ["neck", "mouth", "esophagus"],
    "hip": ["lower back", "groin", "thigh"],
    "neck": ["shoulder", "throat", "head"],
    "ankle": ["foot", "lower leg"],
    "wrist": ["hand", "forearm"],
    "elbow": ["upper arm", "forearm"],
    "eye": ["forehead", "temple", "cheek"],
    "ear": ["jaw", "temple", "neck"],
    "groin": ["hip", "lower abdomen", "thigh"],
    "pelvis": ["lower abdomen", "hip", "groin"],
    "forearm": ["wrist", "elbow", "upper arm"],
    "thigh": ["hip", "knee", "groin"],
    "calf": ["shin", "ankle", "knee"],
    "back": ["chest", "shoulder", "lower back"],
    "lung": ["chest", "bronchus", "pleura"],
    "heart": ["chest", "left shoulder", "upper back"],
    "kidney": ["lower back", "abdomen", "bladder"],
    "bladder": ["lower abdomen", "kidney", "pelvis"],
}

SYMPTOM_CONFUSIONS = {
    "burning": ["stabbing", "tingling", "stinging"],
    "sharp": ["dull", "throbbing", "aching"],
    "dull": ["sharp", "aching", "heavy"],
    "lump": ["tumor", "swelling", "mass", "bump"],
    "nausea": ["dizziness", "lightheadedness"],
    "fatigue": ["weakness", "exhaustion", "drowsiness"],
    "cramping": ["spasming", "tightening", "aching"],
    "itching": ["tingling", "burning", "prickling"],
    "numbness": ["tingling", "weakness", "heaviness"],
    "throbbing": ["pulsating", "pounding", "aching"],
    "stiffness": ["tightness", "soreness", "aching"],
    "swelling": ["bloating", "puffiness", "lump"],
    "bleeding": ["bruising", "spotting", "discharge"],
    "cough": ["clearing throat", "choking", "wheezing"],
    "fever": ["chills", "sweating", "hot flashes"],
}

VAGUE_RESPONSES = [
    "I'm not really sure about that.",
    "I can't quite remember.",
    "It's hard to say, honestly.",
    "I don't know, it's been kind of a blur.",
    "I'm not sure, maybe? I can't recall clearly.",
    "I'd rather not get into that.",
    "I don't think I can answer that right now.",
]

SELF_DIAGNOSIS_PHRASES = [
    "I think I might have {condition}.",
    "My neighbor had something similar and said it was {condition}.",
    "I looked it up online and it seems like {condition}.",
    "My friend who's a nurse thinks it could be {condition}.",
]

COMMON_SELF_DIAGNOSES = [
    "a cold", "the flu", "allergies", "food poisoning",
    "a pulled muscle", "stress", "anxiety", "a virus",
    "a pinched nerve", "an infection",
]

AMBIGUITY_TEMPLATES = [
    "Findings are equivocal — {original}. Cannot definitively rule out alternative interpretation.",
    "Results show {original}. However, findings are not entirely clear and may warrant further evaluation.",
    "{original}. Note: image quality/sample quality limits definitive interpretation.",
]

# Severity word mappings for perturbation
_SEVERITY_SWAPS = {
    "severe": "moderate",
    "moderate": "mild",
    "mild": "barely noticeable",
    "intense": "moderate",
    "extreme": "moderate",
    "excruciating": "severe",
    "unbearable": "significant",
    "significant": "mild",
    "terrible": "moderate",
    "awful": "moderate",
}

# Duration pattern: matches "N days/weeks/months/hours/years"
_DURATION_RE = re.compile(r"\b(\d+)\s*(days?|weeks?|months?|hours?|years?)\b", re.IGNORECASE)

PATIENT_NOISE_TYPES = [
    "body_part_swap",
    "symptom_confusion",
    "severity_change",
    "temporal_change",
    "omission",
    "self_diagnosis",
    "vague_answer",
]

EXAM_NOISE_TYPES = ["body_part_swap", "omission", "ambiguity"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NoiseConfig:
    patient_noise_level: float = 0.0  # 0.0-1.0
    exam_noise_level: float = 0.0     # 0.0-1.0
    seed: int | None = None


@dataclass
class PatientNoiseSpec:
    """Planned noise for one patient conversation. Created once, consumed per-turn."""
    body_part_swaps: dict[str, str] = field(default_factory=dict)
    symptom_confusions: dict[str, str] = field(default_factory=dict)
    severity_changes: list[str] = field(default_factory=list)
    temporal_changes: list[str] = field(default_factory=list)
    omitted_symptoms: list[str] = field(default_factory=list)
    self_diagnoses: list[str] = field(default_factory=list)
    vague_answer_assigned: bool = False
    vague_answer_turn: int | None = None
    vague_turns: int = 0  # actual count of turns where a vague hint was injected
    self_diagnosis_turn: int | None = None
    noise_turns: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def check_patient_noise_eligibility(symptoms: str) -> dict[str, bool]:
    """Check which noise types a patient is eligible for based on their symptoms.

    Returns a dict mapping noise type names to booleans.
    """
    symptom_list = [s.strip().lower() for s in symptoms.split(",") if s.strip()]

    body_part_eligible = any(
        body_part in sym
        for sym in symptom_list
        for body_part in BODY_PART_ADJACENCY
    )
    symptom_confusion_eligible = any(
        descriptor in sym
        for sym in symptom_list
        for descriptor in SYMPTOM_CONFUSIONS
    )
    omission_eligible = len(symptom_list) > 1

    return {
        "body_part_swap": body_part_eligible,
        "symptom_confusion": symptom_confusion_eligible,
        "severity_change": True,
        "temporal_change": True,
        "omission": omission_eligible,
        "self_diagnosis": True,
        "vague_answer": True,
    }


def check_exam_noise_eligibility(findings: str) -> dict[str, bool]:
    """Check which exam noise types are applicable to a findings string."""
    if not findings.strip():
        return {nt: False for nt in EXAM_NOISE_TYPES}
    findings_lower = findings.lower()
    body_part_eligible = any(bp in findings_lower for bp in BODY_PART_ADJACENCY)
    parts = [p.strip() for p in re.split(r"[;,]", findings) if p.strip()]
    return {
        "body_part_swap": body_part_eligible,
        "omission": len(parts) > 1,
        "ambiguity": True,
    }


def plan_patient_noise(
    symptoms: str,
    noise_level: float,
    rng: Random,
    forced_noise_types: set[str] | None = None,
) -> PatientNoiseSpec:
    """Plan all noise for one patient conversation based on their symptoms.

    Args:
        forced_noise_types: If provided, exactly these noise types are applied
            (pre-assigned by dataset-level planning). noise_level is ignored.
    """
    if noise_level <= 0 and not forced_noise_types:
        return PatientNoiseSpec()

    def _enabled(noise_type: str) -> bool:
        if forced_noise_types is not None:
            return noise_type in forced_noise_types
        return rng.random() < noise_level

    spec = PatientNoiseSpec()
    symptom_list = [s.strip().lower() for s in symptoms.split(",") if s.strip()]

    # Body part swaps: scan symptoms for adjacency keys
    if _enabled("body_part_swap"):
        for sym in symptom_list:
            for body_part, adjacents in BODY_PART_ADJACENCY.items():
                if body_part in sym:
                    spec.body_part_swaps[body_part] = rng.choice(adjacents)
                    break  # one swap is enough
            if spec.body_part_swaps:
                break

    # Symptom confusion: scan for confusion keys
    if _enabled("symptom_confusion"):
        for sym in symptom_list:
            for descriptor, confusions in SYMPTOM_CONFUSIONS.items():
                if descriptor in sym:
                    spec.symptom_confusions[descriptor] = rng.choice(confusions)
                    break
            if spec.symptom_confusions:
                break

    # Severity changes
    if _enabled("severity_change"):
        for sym in symptom_list:
            for word, replacement in _SEVERITY_SWAPS.items():
                if word in sym:
                    spec.severity_changes.append(
                        f"report the pain/symptom as {replacement} instead of {word}"
                    )
                    break
            if spec.severity_changes:
                break
        # If no severity word found in symptoms, add a generic one
        if not spec.severity_changes:
            spec.severity_changes.append("downplay the severity of your symptoms slightly")

    # Temporal changes: look for duration patterns
    if _enabled("temporal_change"):
        for sym in symptom_list:
            match = _DURATION_RE.search(sym)
            if match:
                original_num = int(match.group(1))
                unit = match.group(2)
                # Perturb by ±30-100%
                factor = rng.uniform(0.3, 2.0)
                new_num = max(1, int(original_num * factor))
                if new_num != original_num:
                    spec.temporal_changes.append(
                        f"say {new_num} {unit} instead of {original_num} {unit}"
                    )
                    break
        # If no duration found, add a generic temporal perturbation
        if not spec.temporal_changes:
            spec.temporal_changes.append(
                "be vague about when symptoms started — say 'a while ago' or 'recently' instead of giving a specific time"
            )

    # Omission: pick one symptom to omit (skip if only 1)
    if _enabled("omission") and len(symptom_list) > 1:
        spec.omitted_symptoms.append(rng.choice(symptom_list))

    # Self-diagnosis
    if _enabled("self_diagnosis"):
        phrase = rng.choice(SELF_DIAGNOSIS_PHRASES).format(
            condition=rng.choice(COMMON_SELF_DIAGNOSES)
        )
        spec.self_diagnoses.append(phrase)

    # Vague answer assignment flag (turn is set later in unified turn planning).
    if _enabled("vague_answer"):
        spec.vague_answer_assigned = True

    # Unified turn planning: every planned patient noise type is assigned
    # exactly one random patient turn.
    selected_types: list[str] = []
    if spec.body_part_swaps:
        selected_types.append("body_part_swap")
    if spec.symptom_confusions:
        selected_types.append("symptom_confusion")
    if spec.severity_changes:
        selected_types.append("severity_change")
    if spec.temporal_changes:
        selected_types.append("temporal_change")
    if spec.omitted_symptoms:
        selected_types.append("omission")
    if spec.self_diagnoses:
        selected_types.append("self_diagnosis")
    if spec.vague_answer_assigned:
        selected_types.append("vague_answer")

    if selected_types:
        rng.shuffle(selected_types)
        # Use a small early-turn window so planned noise appears during inquiry.
        turn_pool = list(range(max(7, len(selected_types))))
        rng.shuffle(turn_pool)
        for idx, noise_type in enumerate(selected_types):
            turn = turn_pool[idx]
            spec.noise_turns[noise_type] = turn
            if noise_type == "self_diagnosis":
                spec.self_diagnosis_turn = turn
            elif noise_type == "vague_answer":
                spec.vague_answer_turn = turn

    return spec


# ---------------------------------------------------------------------------
# Per-turn hint generation
# ---------------------------------------------------------------------------

def get_patient_turn_hint(
    noise_spec: PatientNoiseSpec,
    turn_index: int,
    doctor_message: str,
    rng: Random,
) -> tuple[str, str] | None:
    """Generate a system hint for one patient turn.

    Returns:
        (hint_text, noise_type) tuple, or None if no noise applies.
    """
    if not doctor_message:
        return None

    planned_type = next(
        (nt for nt, planned_turn in noise_spec.noise_turns.items() if planned_turn == turn_index),
        None,
    )
    if planned_type is None:
        return None

    if planned_type == "vague_answer":
        vague = rng.choice(VAGUE_RESPONSES)
        noise_spec.vague_turns += 1
        return (
            f"For this response, be vague and evasive. Say something like "
            f"'{vague}' Do not provide specific details for this question.",
            "vague_answer",
        )

    if planned_type == "omission" and noise_spec.omitted_symptoms:
        omitted = noise_spec.omitted_symptoms[0]
        return (
            f"In this response, do NOT mention {omitted}. "
            f"Talk only about your other symptoms or details.",
            "omission",
        )

    if planned_type == "self_diagnosis" and noise_spec.self_diagnoses:
        phrase = noise_spec.self_diagnoses[0]
        return (
            f"Work the following into your response naturally: '{phrase}' "
            f"Then continue answering the doctor's question.",
            "self_diagnosis",
        )

    if planned_type == "body_part_swap" and noise_spec.body_part_swaps:
        original, swapped = next(iter(noise_spec.body_part_swaps.items()))
        return (
            f"When answering, refer to your {original} symptom as being "
            f"in your {swapped} area instead.",
            "body_part_swap",
        )

    if planned_type == "symptom_confusion" and noise_spec.symptom_confusions:
        original, confused = next(iter(noise_spec.symptom_confusions.items()))
        return (
            f"When describing the sensation, say '{confused}' "
            f"instead of '{original}'.",
            "symptom_confusion",
        )

    if planned_type == "severity_change" and noise_spec.severity_changes:
        return (
            f"When answering: {noise_spec.severity_changes[0]}.",
            "severity_change",
        )

    if planned_type == "temporal_change" and noise_spec.temporal_changes:
        return (
            f"When answering: {noise_spec.temporal_changes[0]}.",
            "temporal_change",
        )

    return None


# ---------------------------------------------------------------------------
# Exam noise
# ---------------------------------------------------------------------------

def apply_exam_noise(
    exam_name: str,
    raw_findings: str,
    noise_level: float,
    rng: Random,
    forced_noise_type: str | None = None,
) -> tuple[str, dict | None]:
    """Apply noise to exam findings.

    If forced_noise_type is set, that type is applied deterministically and
    noise_level/probability gating is skipped.
    Returns (possibly_modified_findings, metadata_or_None).
    """
    if forced_noise_type is None:
        if noise_level <= 0 or rng.random() >= noise_level:
            return (raw_findings, None)
        noise_type = rng.choice(EXAM_NOISE_TYPES)
    else:
        if forced_noise_type not in EXAM_NOISE_TYPES:
            raise ValueError(f"Unknown forced exam noise type: {forced_noise_type}")
        noise_type = forced_noise_type

    if not raw_findings.strip():
        return (raw_findings, None)

    original = raw_findings

    if noise_type == "body_part_swap":
        # Scan findings for body part adjacency keys, replace one
        findings_lower = raw_findings.lower()
        for body_part, adjacents in BODY_PART_ADJACENCY.items():
            if body_part in findings_lower:
                replacement = rng.choice(adjacents)
                # Case-insensitive replacement of first occurrence
                pattern = re.compile(re.escape(body_part), re.IGNORECASE)
                modified = pattern.sub(replacement, raw_findings, count=1)
                if modified != raw_findings:
                    return (modified, {
                        "noise_type": "body_part_swap",
                        "original": original,
                        "modified": modified,
                    })
        noise_type = "ambiguity"

    if noise_type == "omission":
        # Split findings by comma or semicolon, drop one clause
        separators = re.split(r"[;,]", raw_findings)
        separators = [s.strip() for s in separators if s.strip()]
        if len(separators) > 1:
            drop_idx = rng.randrange(len(separators))
            remaining = [s for i, s in enumerate(separators) if i != drop_idx]
            modified = "; ".join(remaining)
            return (modified, {
                "noise_type": "omission",
                "original": original,
                "modified": modified,
            })
        noise_type = "ambiguity"

    if noise_type == "ambiguity":
        template = rng.choice(AMBIGUITY_TEMPLATES)
        modified = template.format(original=raw_findings)
        return (modified, {
            "noise_type": "ambiguity",
            "original": original,
            "modified": modified,
        })

    return (raw_findings, None)


# ---------------------------------------------------------------------------
# Dataset-level noise pre-assignment
# ---------------------------------------------------------------------------

def plan_dataset_noise_assignments(
    patients: list[dict],
    noise_config: NoiseConfig,
) -> tuple[dict[int, set[str]], dict[int, dict[int, str]], dict[str, int], dict[str, int]]:
    """Pre-assign noise types to eligible patients and exam slots.

    ``patient_noise_level`` controls the total fraction of patients that receive
    any noise (not per-type). Selected patients are each assigned 1-3 eligible
    types at random. ``exam_noise_level`` controls the fraction of valid exam
    slots across the entire dataset that get noise; each selected slot is
    assigned one eligible noise type.

    Args:
        patients: List of dicts with keys ``patient_id``, ``self_reported_symptoms``,
            and ``exams`` (list of ``(exam_str, findings)`` tuples).
        noise_config: NoiseConfig with patient_noise_level, exam_noise_level, seed.

    Returns:
        patient_assignments: pid -> set of noise types to apply
        exam_assignments:    pid -> {exam_idx -> noise_type}
        patient_eligible:    noise_type -> eligible count (for reporting)
        exam_eligible:       noise_type -> eligible count (for reporting)
    """
    # Imported lazily to avoid a hard dependency on data.tool at import time.
    from data.tool import parse_exam_string

    patient_assignments: dict[int, set[str]] = {}
    exam_assignments: dict[int, dict[int, str]] = {}
    patient_eligible: dict[str, int] = {nt: 0 for nt in PATIENT_NOISE_TYPES}
    exam_eligible: dict[str, int] = {nt: 0 for nt in EXAM_NOISE_TYPES}
    seed = noise_config.seed

    if noise_config.patient_noise_level > 0:
        eligibility_by_pid: dict[int, list[str]] = {}
        for patient in patients:
            pid = patient["patient_id"]
            elig = check_patient_noise_eligibility(patient["self_reported_symptoms"])
            eligible_types = [nt for nt, ok in elig.items() if ok]
            if eligible_types:
                eligibility_by_pid[pid] = eligible_types
            for nt, ok in elig.items():
                if ok:
                    patient_eligible[nt] += 1

        rng = random.Random(seed)
        all_pids = list(eligibility_by_pid.keys())
        rng.shuffle(all_pids)
        n_selected = round(len(patients) * noise_config.patient_noise_level)
        selected_pids = all_pids[:n_selected]

        for pid in selected_pids:
            eligible_types = eligibility_by_pid[pid]
            rng.shuffle(eligible_types)
            n_types = min(rng.randint(1, 3), len(eligible_types))
            patient_assignments[pid] = set(eligible_types[:n_types])

    if noise_config.exam_noise_level > 0:
        all_exam_slots: list[tuple[int, int, list[str]]] = []
        for patient in patients:
            pid = patient["patient_id"]
            for exam_idx, (exam_str, findings) in enumerate(patient["exams"]):
                if parse_exam_string(exam_str) is None:
                    continue
                elig = check_exam_noise_eligibility(findings)
                eligible_types = [nt for nt, ok in elig.items() if ok]
                if eligible_types:
                    all_exam_slots.append((pid, exam_idx, eligible_types))
                for nt, ok in elig.items():
                    if ok:
                        exam_eligible[nt] += 1

        rng_exam = random.Random(None if seed is None else seed + 1)
        rng_exam.shuffle(all_exam_slots)
        n_selected = round(len(all_exam_slots) * noise_config.exam_noise_level)
        for pid, exam_idx, eligible_types in all_exam_slots[:n_selected]:
            noise_type = rng_exam.choice(eligible_types)
            exam_assignments.setdefault(pid, {})[exam_idx] = noise_type

    return patient_assignments, exam_assignments, patient_eligible, exam_eligible
