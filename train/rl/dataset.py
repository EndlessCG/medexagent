"""RL dataset: load patient profiles from CSV, build initial prompts for rollouts."""

import csv
import json
import random
import sys
from pathlib import Path

import pandas as pd

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.noise import NoiseConfig, plan_dataset_noise_assignments
from data.tool import TOOL_SCHEMAS, parse_exam_string, select_tools_for_case
from data.prompt import format_tools_for_prompt, sample_patient_persona


def _format_demographics(age: str, gender: str, race: str, additional_demographics: str) -> str:
    """Format patient demographics, omitting any empty/None fields."""
    parts = []
    if age:
        parts.append(f"Age: {age}")
    if gender:
        parts.append(f"Gender: {gender}")
    if race:
        parts.append(f"Race/Ethnicity: {race}")
    if additional_demographics:
        parts.append(f"Other: {additional_demographics}")
    return "\n".join(parts) if parts else "Not specified"


def build_doctor_system_prompt_rl(
    medical_history: str,
    available_tools: list[dict],
    age: str = "",
    gender: str = "",
    race: str = "",
    additional_demographics: str = "",
) -> str:
    """Build doctor system prompt for RL — no exam plan provided.

    Unlike the SFT doctor prompt, the RL prompt does NOT include a pre-specified
    exam plan. The model must learn to choose exams independently.

    Args:
        medical_history: Patient's medical history.
        available_tools: List of tool info dicts (from select_tools_for_case).
        age, gender, race, additional_demographics: Patient demographics; any
            empty/None field is silently omitted from the prompt.
    """
    demographics_str = _format_demographics(age, gender, race, additional_demographics)
    history_str = medical_history if medical_history else "No known past medical history."
    tools_str = format_tools_for_prompt(available_tools)

    return f"""You are an experienced physician conducting a medical consultation. Your goal is to gather information from the patient, order appropriate examinations, interpret results, and arrive at a diagnosis.

PATIENT DEMOGRAPHICS:
{demographics_str}

PATIENT'S MEDICAL HISTORY:
{history_str}

AVAILABLE EXAMINATIONS:
{tools_str}

TOOL CALL FORMAT:
To order an examination, include a tool call in your message using the following format:
<tool_call>
{{"name": "<tool_name>", "arguments": {{"<param>": "<value>", ...}}}}
</tool_call>

For tools with no required parameters, use an empty arguments object:
<tool_call>
{{"name": "<tool_name>", "arguments": {{}}}}
</tool_call>

You may include a brief message to the patient before the tool call. For example, to order a chest CT with contrast, you would write:

Based on your symptoms, I'd like to order a CT scan of your chest to get a better look.
<tool_call>
{{"name": "ct", "arguments": {{"region": "chest", "contrast": "yes"}}}}
</tool_call>

You MUST use only the allowed values listed for each parameter above. Do NOT invent new parameter values.

CONSULTATION STRUCTURE:
1. GREETING: Greet the patient and ask the patient what brings them in today.
2. SYMPTOM INQUIRY: Ask about the patient's symptoms, their onset, duration, severity, and any associated factors. Keep this phase brief (3-5 questions) and ask at most 1-2 short questions per turn. Once you have enough information, move on to examinations.
3. EXAMINATIONS: Based on your clinical assessment, order appropriate examinations one at a time using the <tool_call> format. Order only exams that are clinically indicated — avoid unnecessary tests. After each exam result comes back, briefly discuss the findings with the patient (2-3 sentences) and explain why you'd like to order the next examination before ordering it.
4. DIAGNOSIS: After receiving exam results, state your final diagnosis using the exact format: [DIAGNOSIS: <your diagnosis>]

INSTRUCTIONS:
- Only output what you would say directly to the patient. Do NOT include your internal thoughts / reasoning / plan.
- Pace gently and do NOT overwhelm the patient with too many questions at once.
- Ask at most 1-2 short questions per turn.
- Order exams one at a time, do NOT order multiple exams in one turn.
- After receiving exam results, always discuss the findings before ordering the next exam.
- You MUST end with [DIAGNOSIS: ...] to conclude the consultation.
"""


def load_patient_profiles(data_path: str) -> list[dict]:
    """Load patient profiles from CSV and build structured records.

    Each record contains:
        - demographics: age, gender, race, etc.
        - medical_history: past conditions
        - self_reported_symptoms: patient's reported symptoms
        - exam_map: {exam_string: findings} for all available exams
        - diagnosis: ground truth diagnosis
        - system_prompt: doctor system prompt for RL
    """
    df = pd.read_csv(data_path)
    profiles = []

    for row_idx, (_, row) in enumerate(df.iterrows()):
        # Build exam map from exam1..exam10 columns
        exam_map = {}
        for i in range(1, 11):
            exam_str = str(row.get(f"exam{i}", "") or "").strip()
            findings = str(row.get(f"exam{i}_findings", "") or "").strip()
            if exam_str and findings:
                parsed = parse_exam_string(exam_str)
                if parsed:
                    exam_map[exam_str] = {
                        "parsed": parsed,
                        "findings": findings,
                    }

        # Build available tools list (ground truth + distractors)
        ground_truth_tool_names = [
            info["parsed"]["name"] for info in exam_map.values()
        ]
        available_tools = select_tools_for_case(
            ground_truth_tools=ground_truth_tool_names,
            num_distractors=5,
            shuffle=True,
        )

        medical_history = str(row.get("medical_history", "") or "")
        patient_persona = sample_patient_persona(random.Random(row_idx))
        profile = {
            "demographics": {
                "age": str(row.get("age", "") or ""),
                "gender": str(row.get("gender", "") or ""),
                "race": str(row.get("race", "") or ""),
                "additional_demographics": str(row.get("additional_demographics", "") or ""),
            },
            "medical_history": medical_history,
            "self_reported_symptoms": str(row.get("self_reported_symptoms", "") or ""),
            "patient_persona": patient_persona,
            "exam_map": exam_map,
            "available_tools": [t["name"] for t in available_tools],
            "diagnosis": str(row.get("diagnosis", "") or ""),
            "source": str(row.get("source", "") or ""),
            "system_prompt": build_doctor_system_prompt_rl(
                medical_history,
                available_tools,
                age=str(row.get("age", "") or ""),
                gender=str(row.get("gender", "") or ""),
                race=str(row.get("race", "") or ""),
                additional_demographics=str(row.get("additional_demographics", "") or ""),
            ),
        }
        profiles.append(profile)

    return profiles


def build_verl_dataset(
    data_path: str,
    output_path: str,
    patient_noise_level: float = 0.0,
    exam_noise_level: float = 0.0,
    noise_seed: int | None = None,
):
    """Build a parquet dataset compatible with verl's expectations.

    Output columns (verl schema):
        prompt: list of message dicts (the initial conversation prompt)
        reward_model: dict with {"ground_truth": "..."} for the reward function
        data_source: string identifier
        extra_info: dict with patient metadata for the rollout environment
    """
    profiles = load_patient_profiles(data_path)

    # Dataset-level noise pre-assignment (matches data generation semantics).
    # patient_noise_level selects a fraction of patients who each receive 1-3
    # eligible noise types; exam_noise_level selects a fraction of eligible
    # exam slots dataset-wide and assigns each one a noise type.
    patient_assignments: dict[int, set[str]] = {}
    exam_assignments: dict[int, dict[int, str]] = {}
    if patient_noise_level > 0 or exam_noise_level > 0:
        patient_records = [
            {
                "patient_id": i,
                "self_reported_symptoms": p["self_reported_symptoms"],
                "exams": [(k, v["findings"]) for k, v in p["exam_map"].items()],
            }
            for i, p in enumerate(profiles)
        ]
        patient_assignments, exam_assignments, p_elig, e_elig = plan_dataset_noise_assignments(
            patient_records,
            NoiseConfig(
                patient_noise_level=patient_noise_level,
                exam_noise_level=exam_noise_level,
                seed=noise_seed,
            ),
        )
        print(
            f"Noise plan: {len(patient_assignments)}/{len(profiles)} patients noisy; "
            f"{sum(len(v) for v in exam_assignments.values())} exam slots noisy"
        )

    records = []

    for i, profile in enumerate(profiles):
        # Initial prompt: system message only — verl tokenizes this as the prompt
        prompt = [
            {"role": "system", "content": profile["system_prompt"]},
        ]

        # Noise assignments for this patient (possibly empty).
        assigned_types = sorted(patient_assignments.get(i, set()))
        # Map exam_idx -> noise_type to exam_str -> noise_type for ToolAgent lookup.
        exam_idx_to_str = list(profile["exam_map"].keys())
        exam_noise_map = {
            exam_idx_to_str[idx]: nt
            for idx, nt in exam_assignments.get(i, {}).items()
            if idx < len(exam_idx_to_str)
        }

        # Metadata for the rollout environment
        source = profile.get("source", "")
        extra_info = {
            "patient_index": i,
            "data_source": source,
            "exam_map": json.dumps({k: v for k, v in profile["exam_map"].items()}),
            "available_tools": json.dumps(profile["available_tools"]),
            "ground_truth": profile["diagnosis"],
            "demographics": json.dumps(profile["demographics"]),
            "medical_history": profile["medical_history"],
            "self_reported_symptoms": profile["self_reported_symptoms"],
            "patient_persona": json.dumps(profile["patient_persona"]),
            "exam_noise_map": json.dumps(exam_noise_map),
            "interaction_kwargs": json.dumps({
                "name": "patient",
                "age": profile["demographics"]["age"],
                "gender": profile["demographics"]["gender"],
                "race": profile["demographics"]["race"],
                "additional_demographics": profile["demographics"]["additional_demographics"],
                "medical_history": profile["medical_history"],
                "self_reported_symptoms": profile["self_reported_symptoms"],
                "patient_persona": profile["patient_persona"],
                "patient_index": i,
                "assigned_noise_types": assigned_types,
            }),
        }

        records.append({
            "prompt": prompt,
            "reward_model": {"ground_truth": profile["diagnosis"]},
            "data_source": f"medagent_{profile['source']}" if profile.get("source") else "medagent",
            "extra_info": extra_info,
        })

    df = pd.DataFrame(records)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(records)} records to {output_path}")
    return df


def build_eval_parquet(
    jsonl_path: str,
    output_path: str | None = None,
    max_diag_samples: int | None = None,
    max_tool_samples: int | None = None,
    max_conv_samples: int | None = None,
    name: str = "",
):
    """Build a verl-compatible parquet for in-loop RL evaluation.

    Reads JSONL conversation records and creates two types of eval rows:

    1. Diagnosis eval (data_source="medagent_eval_diag"):
       prompt = conversation cut before final [DIAGNOSIS: ...] turn.
       extra_info["eval_type"] = "diagnosis"
       extra_info["ground_truth"] = ground-truth diagnosis string.

    2. Tool call format eval (data_source="medagent_eval_tool"):
       prompt = messages before a tool-call turn.
       extra_info["eval_type"] = "tool_call"
       extra_info["ground_truth_tool"] = JSON-serialised {"name":..., "arguments":...}
       extra_info["text_prefix"] = assistant text before the <tool_call> block.

    The MedicalAgentLoop._run_eval_mode() dispatches on extra_info["eval_type"]
    to run single-turn generation and score with MedEmbed / compute_tool_reward.
    """
    from evaluation.data import (
        prepare_conversation_eval_samples,
        prepare_diagnosis_eval_samples,
        prepare_tool_call_eval_samples,
    )

    # Load raw records
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    diag_samples = prepare_diagnosis_eval_samples(records)
    tool_samples = prepare_tool_call_eval_samples(records)
    conv_samples = prepare_conversation_eval_samples(records)

    if max_diag_samples and len(diag_samples) > max_diag_samples:
        diag_samples = diag_samples[:max_diag_samples]
    if max_tool_samples and len(tool_samples) > max_tool_samples:
        tool_samples = tool_samples[:max_tool_samples]
    if max_conv_samples and len(conv_samples) > max_conv_samples:
        conv_samples = conv_samples[:max_conv_samples]

    suffix = f"_{name}" if name else ""
    rows = []

    for sample in diag_samples:
        diag_source = f"medagent_eval_diag{suffix}"
        rows.append({
            "prompt": sample["input_messages"],
            "reward_model": {"ground_truth": sample["ground_truth_diagnosis"]},
            "data_source": diag_source,
            "extra_info": {
                "eval_type": "diagnosis",
                "data_source": diag_source,
                "ground_truth": sample["ground_truth_diagnosis"],
                "gt_conversation": json.dumps(sample.get("gt_conversation", [])),
            },
        })

    for sample in tool_samples:
        tool_source = f"medagent_eval_tool{suffix}"
        rows.append({
            "prompt": sample["prior_messages"],
            "reward_model": {"ground_truth": ""},
            "data_source": tool_source,
            "extra_info": {
                "eval_type": "tool_call",
                "data_source": tool_source,
                "ground_truth_tool": json.dumps(sample["ground_truth_tool"]),
                "text_prefix": sample["text_prefix"],
                "gt_conversation": json.dumps(sample.get("gt_conversation", [])),
            },
        })

    for sample in conv_samples:
        conv_source = f"medagent_eval_conversation{suffix}"
        rows.append({
            "prompt": [{"role": "system", "content": ""}],
            "reward_model": {"ground_truth": sample["ground_truth_diagnosis"]},
            "data_source": conv_source,
            "extra_info": {
                "eval_type": "conversation",
                "data_source": conv_source,
                "ground_truth": sample["ground_truth_diagnosis"],
                "patient_kwargs": json.dumps(sample["patient_kwargs"]),
                "available_tools": json.dumps(sample["available_tools"]),
                "available_tool_names": json.dumps(sample["available_tool_names"]),
                "ground_truth_tools": json.dumps(sample["ground_truth_tools"]),
                "gt_tool_findings": json.dumps(sample["gt_tool_findings"]),
            },
        })

    df = pd.DataFrame(rows)
    if output_path:
        df.to_parquet(output_path, index=False)
        print(
            f"Saved eval dataset [{name or 'default'}]: {len(diag_samples)} diagnosis + "
            f"{len(tool_samples)} tool-call + {len(conv_samples)} conversation samples -> {output_path}"
        )
    return df


def build_multi_eval_parquet(
    sources: dict[str, str],
    output_path: str,
    max_diag_samples: int | None = None,
    max_tool_samples: int | None = None,
    max_conv_samples: int | None = None,
) -> pd.DataFrame:
    """Build a merged eval parquet from one or more named JSONL sources.

    Args:
        sources: Mapping of {name: jsonl_path}. Use name="" for a single unnamed source
            (data_source will have no suffix). Use distinct names for multiple sources
            so verl can log metrics per set (e.g. "medagent_eval_diag_ddxplus").
        output_path: Where to write the merged parquet.
        max_diag_samples: Cap per source (not total).
        max_tool_samples: Cap per source (not total).
    """
    all_dfs = []
    for name, path in sources.items():
        df = build_eval_parquet(
            path,
            output_path=None,
            max_diag_samples=max_diag_samples,
            max_tool_samples=max_tool_samples,
            max_conv_samples=max_conv_samples,
            name=name,
        )
        all_dfs.append(df)

    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_parquet(output_path, index=False)
    print(
        f"Saved merged eval dataset: {len(merged)} rows "
        f"({len(sources)} source(s)) -> {output_path}"
    )
    return merged


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build RL dataset")
    parser.add_argument("--data_path", default="data/ehr/pmc_patients_post_processed.csv")
    parser.add_argument("--output_path", default="data/rl_dataset.parquet")
    parser.add_argument("--patient_noise_level", type=float, default=0.0)
    parser.add_argument("--exam_noise_level", type=float, default=0.0)
    parser.add_argument("--noise_seed", type=int, default=None)
    args = parser.parse_args()
    build_verl_dataset(
        args.data_path,
        args.output_path,
        patient_noise_level=args.patient_noise_level,
        exam_noise_level=args.exam_noise_level,
        noise_seed=args.noise_seed,
    )
