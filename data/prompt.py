"""
System prompts for the doctor and patient LLM agents.
"""

import random


def _format_demographics(
    age: str,
    gender: str,
    race: str,
    additional_demographics: str,
) -> str:
    """Format patient demographics into a compact multi-line string.

    Omits any field that is empty/None. Returns "Not specified" if all are absent.
    """
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


PATIENT_PERSONA_OPTIONS = {
    "personality": [
        {
            "key": "verbose",
            "label": "Verbose",
            "instruction": "You are extremely talkative. You give excessively detailed, rambling answers and often go off-topic. You share unnecessary personal anecdotes or daily routines before finally getting to the medical point.",
        },
        {
            "key": "pleasing",
            "label": "Pleasing",
            "instruction": "You are highly cooperative, polite, and eager to please the doctor. You agree with the doctor easily, apologize often for taking up their time, and try not to be a burden.",
        },
        {
            "key": "impatient",
            "label": "Impatient",
            "instruction": "You are annoyed, rushed, and easily frustrated. You want quick answers, give short or curt replies, complain about waiting times, and frequently ask how much longer this will take.",
        },
        {
            "key": "distrust",
            "label": "Distrust",
            "instruction": "You are skeptical of the doctor and the medical system. You constantly question why the doctor needs certain personal information, doubt diagnoses or treatments, and are defensive and hesitant to share details.",
        },
        {
            "key": "overanxious",
            "label": "Overanxious",
            "instruction": "You are extremely worried about your symptoms. You fear the worst-case scenario, ask for excessive reassurance, and maintain a panicked or nervous tone.",
        },
        {
            "key": "plain",
            "label": "Plain",
            "instruction": "You are a normal, cooperative patient. Answer questions straightforwardly and naturally without any extreme emotional bias.",
        },
    ],
    "language_proficiency": [
        {
            "key": "level_a",
            "label": "Level A (Low)",
            "instruction": "You struggle with the language. Use very simple, broken sentences with basic vocabulary. Make grammatical errors, misunderstand complex medical terms, and occasionally struggle to find the right word.",
        },
        {
            "key": "level_b",
            "label": "Level B (Moderate)",
            "instruction": "You are conversational but not entirely fluent. You use slightly awkward phrasing and occasionally ask the doctor to explain or simplify advanced medical jargon.",
        },
        {
            "key": "level_c",
            "label": "Level C (Native)",
            "instruction": "You speak naturally and fluently. You can articulate your symptoms clearly like a native speaker.",
        },
    ],
    "medical_history_recall": [
        {
            "key": "low",
            "label": "Low",
            "instruction": "Your memory of your past health is fuzzy. You only remember vague details and frequently use phrases like I think, maybe, or I do not remember exactly.",
        },
        {
            "key": "high",
            "label": "High",
            "instruction": "You have an excellent memory. You accurately provide specific names of diseases, past surgeries, and exact medication names seamlessly based on your true medical history.",
        },
    ],
}

_PATIENT_PERSONA_OPTION_MAP = {
    category: {option["key"]: option for option in options}
    for category, options in PATIENT_PERSONA_OPTIONS.items()
}


def sample_patient_persona(
    rng: random.Random | None = None,
) -> dict[str, dict[str, str]]:
    """Uniformly sample one persona item from each patient-persona category."""
    chooser = rng if rng is not None else random
    return {
        category: dict(chooser.choice(options))
        for category, options in PATIENT_PERSONA_OPTIONS.items()
    }


def _resolve_patient_persona(
    patient_persona: dict[str, dict[str, str] | str] | None,
    rng: random.Random | None = None,
) -> dict[str, dict[str, str]]:
    """Normalize a persona payload and fill any missing categories by sampling."""
    chooser = rng if rng is not None else random
    resolved: dict[str, dict[str, str]] = {}

    for category, options in PATIENT_PERSONA_OPTIONS.items():
        selected = patient_persona.get(category) if patient_persona else None

        if isinstance(selected, str):
            selected = _PATIENT_PERSONA_OPTION_MAP[category].get(selected)
        elif isinstance(selected, dict):
            selected_key = selected.get("key")
            if selected_key in _PATIENT_PERSONA_OPTION_MAP[category]:
                selected = _PATIENT_PERSONA_OPTION_MAP[category][selected_key]

        if selected is None:
            selected = chooser.choice(options)

        resolved[category] = dict(selected)

    return resolved


def _format_patient_persona(persona: dict[str, dict[str, str]]) -> str:
    """Render the selected persona items as a compact prompt section."""
    return "\n".join([
        f"Personality: [{persona['personality']['label']}] {persona['personality']['instruction']}",
        f"Language Proficiency: [{persona['language_proficiency']['label']}] {persona['language_proficiency']['instruction']}",
        f"Medical History Recall: [{persona['medical_history_recall']['label']}] {persona['medical_history_recall']['instruction']}",
    ])


def get_patient_system_prompt(
    age: str,
    gender: str,
    race: str,
    additional_demographics: str,
    medical_history: str,
    self_reported_symptoms: str,
    patient_persona: dict[str, dict[str, str] | str] | None = None,
    rng: random.Random | None = None,
    include_persona: bool = True,
) -> str:
    """Build the patient agent's system prompt.

    The patient knows their demographics, symptoms, and medical history.
    They cooperate with the doctor but do not reveal the diagnosis.

    When include_persona=False, omit the persona section and persona-specific
    instructions entirely (baseline mode).
    """
    demographics_str = _format_demographics(age, gender, race, additional_demographics)
    history_str = medical_history if medical_history else "No significant past medical history."
    symptoms_str = self_reported_symptoms if self_reported_symptoms else "General discomfort."

    if include_persona:
        persona = _resolve_patient_persona(patient_persona, rng=rng)
        persona_str = _format_patient_persona(persona)
        persona_section = f"\n\nSELECTED PATIENT PERSONA:\n{persona_str}"
        instructions = """- Follow the selected persona naturally, but do NOT mention the persona labels to the doctor.
- Describe your symptoms using everyday, non-medical language when possible.
- Pace gently. Keep each reply brief and natural, usually 1-3 spoken sentences.
- Do NOT volunteer all your symptoms or history at once. Share information gradually as the doctor asks.
- Even if your selected personality is Verbose, do NOT dominate the conversation or dump long monologues unless the doctor explicitly asks for more detail.
- Only output spoken words. Do NOT include thoughts, reasoning, narration, stage directions, bracketed text, labels, or body language.
- Stay faithful to your true demographics and current symptoms. For past medical history, follow your selected recall setting exactly."""
    else:
        persona_section = ""
        instructions = """- Describe your symptoms using everyday, non-medical language when possible.
- Pace gently. Keep each reply brief and natural, usually 1-3 spoken sentences.
- Do NOT volunteer all your symptoms or history at once. Share information gradually as the doctor asks.
- Only output spoken words. Do NOT include thoughts, reasoning, narration, stage directions, bracketed text, labels, or body language.
- Stay faithful to your true demographics and current symptoms."""

    return f"""You are a patient visiting a doctor. You should role-play as a real patient would behave during a medical consultation.

YOUR DEMOGRAPHICS:
{demographics_str}

YOUR MEDICAL HISTORY:
{history_str}

YOUR CURRENT SYMPTOMS:
{symptoms_str}{persona_section}

INSTRUCTIONS:
{instructions}
"""


def get_agentclinic_patient_system_prompt(
    age: str,
    gender: str,
    race: str,
    additional_demographics: str,
    medical_history: str,
    self_reported_symptoms: str,
) -> str:
    """Patient system prompt mirroring AgentClinic (Schmidgall et al.).

    Reproduces the base prompt from AgentClinic's PatientAgent.system_prompt
    (agentclinic.py:439-445), without bias injection. The "information" block
    concatenates demographics + history + symptoms to approximate
    AgentClinic's patient_information() dict dump.
    """
    demographics_str = _format_demographics(age, gender, race, additional_demographics)
    history_str = medical_history if medical_history else "No significant past medical history."
    symptoms_str = self_reported_symptoms if self_reported_symptoms else "General discomfort."
    info_block = (
        f"Demographics:\n{demographics_str}\n\n"
        f"Medical History:\n{history_str}\n\n"
        f"Current Symptoms:\n{symptoms_str}"
    )
    return (
        "You are a patient in a clinic who only responds in the form of dialogue. "
        "You are being inspected by a doctor who will ask you questions and will "
        "perform exams on you in order to understand your disease. Your answer "
        "will only be 1-3 sentences in length."
        f"\n\nBelow is all of your information. {info_block}.\n\n "
        "Remember, you must not reveal your disease explicitly but may only "
        "convey the symptoms you have in the form of dialogue if you are asked."
    )


def format_tools_for_prompt(tools: list[dict]) -> str:
    """Format tools list into ToolRL-style Available Tools section.

    Args:
        tools: List of tool info dicts with name, description, parameters.

    Returns:
        Formatted string for the system prompt.
    """
    lines = []
    for tool in tools:
        lines.append(f"- Name: {tool['name']}")
        lines.append(f"  Description: {tool['description']}")

        params = tool.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        if properties:
            lines.append("  Parameters:")
            for pname, pinfo in properties.items():
                pdesc = pinfo.get("description", "")
                req_marker = " (required)" if pname in required else ""
                enum = pinfo.get("enum")
                if enum:
                    lines.append(f"    - {pname}{req_marker}: {pdesc}")
                    lines.append(f"      Allowed values: {', '.join(enum)}")
                else:
                    lines.append(f"    - {pname}{req_marker}: {pdesc}")
        else:
            lines.append("  Parameters: None")
        lines.append("")  # blank line between tools

    return "\n".join(lines)


def get_doctor_base_prompt(
    medical_history: str,
    available_tools: list[dict],
    age: str = "",
    gender: str = "",
    race: str = "",
    additional_demographics: str = "",
    hidden_canonical_diagnosis: str = "",
) -> str:
    """Build the doctor agent's base system prompt (static context only).

    Contains role statement, patient demographics, medical history, available
    examinations, and general output instructions. No phase-specific behavioral
    instructions — those are injected via ``get_doctor_phase_prompt``.
    """
    demographics_str = _format_demographics(age, gender, race, additional_demographics)
    history_str = medical_history if medical_history else "No known past medical history."
    tools_str = format_tools_for_prompt(available_tools)
    hidden_diagnosis_block = (
        f"\nHIDDEN CANONICAL DIAGNOSIS (for training/consistency checking only; do NOT reveal to patient):\n{hidden_canonical_diagnosis}\n"
        if hidden_canonical_diagnosis
        else ""
    )
    hidden_diagnosis_instruction_str = (
        "- The hidden canonical diagnosis is ground truth for this case. Use it to guide your reasoning and keep the conversation clinically consistent, but do NOT reveal it to the patient before you have enough information and appropriate exam results.\n"
        "- When you give the final diagnosis, it must match the hidden canonical diagnosis exactly.\n"
        if hidden_canonical_diagnosis
        else ""
    )

    return f"""You are an experienced physician conducting a medical consultation. Your goal is to gather information from the patient, order appropriate examinations, interpret results, and arrive at a diagnosis.

PATIENT DEMOGRAPHICS:
{demographics_str}

PATIENT'S MEDICAL HISTORY:
{history_str}
{hidden_diagnosis_block}

AVAILABLE EXAMINATIONS:
{tools_str}

INSTRUCTIONS:
- Only output what you would say directly to the patient. Do NOT include your internal thoughts / reasoning / plan.
- Pace gently, do NOT overwhelm the patient with too many questions at once.
- Ask at most 1-2 short questions per turn.
- Order exams one at a time, do NOT order multiple exams in one turn.
{hidden_diagnosis_instruction_str}- If no hidden canonical diagnosis is provided, reason from the conversation and exam results alone.
- You MUST end with [DIAGNOSIS: ...] to conclude the consultation.
"""


def get_doctor_phase_prompt(
    phase: str,
    include_tool_call_format: bool = False,
) -> str:
    """Return a phase-specific system message for the doctor agent.

    Args:
        phase: One of ``"greeting"``, ``"inquiry"``, ``"exam"``, ``"diagnosis"``,
            ``"diagnosis_no_exams"``.
        include_tool_call_format: If True, include ``<tool_call>`` format
            instructions in the ``"exam"`` phase (used for training data; the
            data-generation LLM does not need them).

    Returns:
        The system-message content string for the requested phase.
    """
    if phase == "greeting":
        return (
            "CURRENT PHASE — GREETING:\n"
            "Greet the patient warmly and ask what brings them in today. "
            "Do NOT hallucinate the patient's name or your own name.\n"
            "Do NOT mention any examinations, tests, or lab work. Do NOT mention ordering anything."
        )

    if phase == "inquiry":
        return (
            "CURRENT PHASE — SYMPTOM INQUIRY:\n"
            "Ask about the patient's symptoms, their onset, duration, severity, "
            "and any associated factors. Keep this phase brief (3-5 questions).\n"
            "STRICT RULES:\n"
            "- Ask at most 1-2 short questions in each turn. Do NOT ask long multi-part question lists.\n"
            "- Do NOT mention ordering, running, or performing any examinations or tests.\n"
            "- Do NOT reference any test results, lab results, or imaging findings.\n"
            "- Do NOT say phrases like 'I'd like to order', 'let me check', 'let's run', etc.\n"
            "- ONLY ask questions about symptoms, history, and lifestyle.\n"
            "- Once you have a reasonable clinical picture, simply say you have enough information to proceed.\n"
            "- Do NOT include [DIAGNOSIS: ...] tags during this phase."
        )

    if phase == "exam":
        tool_call_section = ""
        if include_tool_call_format:
            tool_call_section = (
                "\n\nTOOL CALL FORMAT:\n"
                "To order an examination, include a tool call in your message using the following format:\n"
                "<tool_call>\n"
                '{"name": "<tool_name>", "arguments": {"<param>": "<value>", ...}}\n'
                "</tool_call>\n\n"
                "For tools with no required parameters, use an empty arguments object:\n"
                "<tool_call>\n"
                '{"name": "<tool_name>", "arguments": {}}\n'
                "</tool_call>\n\n"
                "You may include a brief message to the patient before the tool call. "
                "For example, to order a chest CT with contrast, you would write:\n\n"
                "Based on your symptoms, I'd like to order a CT scan of your chest to get a better look.\n"
                "<tool_call>\n"
                '{"name": "ct", "arguments": {"region": "chest", "contrast": "yes"}}\n'
                "</tool_call>\n\n"
                "You MUST use only the allowed values listed for each parameter above. "
                "Do NOT invent new parameter values."
            )

        if include_tool_call_format:
            order_instruction = (
                "Order examinations using the <tool_call> format above. "
                "You may include a brief message to the patient before each tool call."
            )
        else:
            order_instruction = (
                "Order examinations by using a phrase like \"I'd like to order...\" "
                "or \"Let's run...\". The examination will be performed and results "
                "provided to you automatically. Do NOT simply describe exam results in text."
            )

        return (
            "CURRENT PHASE — EXAMINATIONS:\n"
            f"{order_instruction} Order one examination per turn.\n"
            "Do NOT skip the examination phase. Do NOT narrate or fabricate exam "
            "results — you must order exams and wait for the actual results.\n"
            "Do NOT say you need to wait for results. If you order the exam, it "
            "will be assumed that the results have been received by the next turn, "
            "and you can interpret them immediately.\n"
            "Do NOT include [DIAGNOSIS: ...] tags during this phase — the diagnosis "
            "phase comes later."
            f"{tool_call_section}"
        )

    if phase == "diagnosis":
        return (
            "CURRENT PHASE — DIAGNOSIS:\n"
            "All examinations are complete. Summarize your findings, interpret "
            "the results for the patient, and state your final diagnosis using "
            "the exact format: [DIAGNOSIS: <your diagnosis>]. If you have "
            "multiple diagnoses, separate them with commas. Also explain the "
            "next steps or treatment plan."
        )

    if phase == "diagnosis_no_exams":
        return (
            "CURRENT PHASE — DIAGNOSIS:\n"
            "No examinations were performed for this case. Do NOT mention test "
            "results, imaging findings, lab values, or any completed examination. "
            "Base your explanation only on the history and symptoms discussed in "
            "the conversation, and state your final diagnosis using the exact "
            "format: [DIAGNOSIS: <your diagnosis>]. If you have multiple "
            "diagnoses, separate them with commas. Also explain the next steps "
            "or treatment plan."
        )

    raise ValueError(f"Unknown doctor prompt phase: {phase!r}")


def get_doctor_system_prompt(
    medical_history: str,
    available_tools: list[dict],
    include_tool_call_format: bool = True,
    age: str = "",
    gender: str = "",
    race: str = "",
    additional_demographics: str = "",
) -> str:
    """Build the full doctor agent system prompt (backward-compatible wrapper).

    Concatenates the base prompt with all four phase prompts so that callers
    outside ``generate.py`` (e.g. ``agent_loop.py``, ``fix_system_prompts.py``)
    continue to work unchanged.
    """
    base = get_doctor_base_prompt(
        medical_history=medical_history,
        available_tools=available_tools,
        age=age, gender=gender, race=race,
        additional_demographics=additional_demographics,
    )

    phases = "\n\n".join([
        get_doctor_phase_prompt("greeting"),
        get_doctor_phase_prompt("inquiry"),
        get_doctor_phase_prompt("exam", include_tool_call_format=include_tool_call_format),
        get_doctor_phase_prompt("diagnosis"),
    ])

    return base + "\n" + phases


def get_doctor_training_prompt(
    medical_history: str,
    available_tools: list[dict],
    age: str = "",
    gender: str = "",
    race: str = "",
    additional_demographics: str = "",
) -> str:
    """Build the doctor system prompt used in training data (JSONL output).

    Unlike ``get_doctor_system_prompt`` (which concatenates all phase prompts),
    this prompt contains only the base prompt, tool call format instructions,
    and output conventions — with **no** phase labels like ``CURRENT PHASE —``.
    """
    base = get_doctor_base_prompt(
        medical_history=medical_history,
        available_tools=available_tools,
        age=age, gender=gender, race=race,
        additional_demographics=additional_demographics,
    )

    tool_call_format = (
        "TOOL CALL FORMAT:\n"
        "To order an examination, include a tool call in your message using the following format:\n"
        "<tool_call>\n"
        '{"name": "<tool_name>", "arguments": {"<param>": "<value>", ...}}\n'
        "</tool_call>\n\n"
        "For tools with no required parameters, use an empty arguments object:\n"
        "<tool_call>\n"
        '{"name": "<tool_name>", "arguments": {}}\n'
        "</tool_call>\n\n"
        "You may include a brief message to the patient before the tool call.\n\n"
        "You MUST use only the allowed values listed for each parameter above. "
        "Do NOT invent new parameter values."
    )

    output_conventions = (
        "OUTPUT CONVENTIONS:\n"
        "- Only output what you would say directly to the patient. "
        "Do NOT include your internal thoughts / reasoning / plan.\n"
        "- Pace gently. Ask at most 1-2 short questions per turn; do not overwhelm the patient.\n"
        "- When you are ready to give a diagnosis, use the exact format: "
        "[DIAGNOSIS: <your diagnosis>]\n"
        "- Order exams one at a time. Do NOT order multiple exams in one turn.\n"
        "- Do NOT fabricate or narrate exam results. Order the exam and wait for results."
    )

    return base + "\n" + tool_call_format + "\n\n" + output_conventions


def format_exam_notification_for_patient(exam_name: str, exam_args: dict) -> str:
    """Create a brief notification for the patient about an exam being performed.

    This is inserted as a system-style note so the patient knows an exam happened
    without seeing raw tool call messages.
    """
    # Build a human-readable description
    descriptions = {
        "xray": lambda a: f"an X-ray of your {a.get('region', 'body')}",
        "ct": lambda a: f"a CT scan of your {a.get('region', 'body')}",
        "mri": lambda a: f"an MRI of your {a.get('region', 'body')}",
        "ultrasound": lambda a: f"an ultrasound of your {a.get('region', 'body')}",
        "blood_test": lambda a: f"a blood test ({a.get('test', 'general')})",
        "cbc": lambda _: "a complete blood count",
        "bmp": lambda _: "a basic metabolic panel",
        "cmp": lambda _: "a comprehensive metabolic panel",
        "ecg": lambda _: "an ECG",
        "echocardiogram": lambda _: "an echocardiogram",
        "eeg": lambda _: "an EEG",
        "endoscopy": lambda a: f"an endoscopy ({a.get('type', '')})",
        "biopsy": lambda a: f"a biopsy of your {a.get('site', 'tissue')}",
        "lumbar_puncture": lambda _: "a lumbar puncture",
        "culture": lambda a: f"a culture from {a.get('source', 'a sample')}",
        "pet_scan": lambda _: "a PET scan",
    }

    func = descriptions.get(exam_name)
    if func:
        desc = func(exam_args)
    else:
        # Generic fallback
        desc = f"a {exam_name.replace('_', ' ')}"

    return f"[The doctor has ordered {desc}. The results have been received.]"
