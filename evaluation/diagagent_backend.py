"""DiagAgent baseline adapter for the medagent evaluation harness.

Wraps a standard backend (typically `_APIBackend` pointing at a vLLM-served
``Henrychur/DiagAgent-{7B,8B,14B}`` checkpoint) so DiagAgent can be evaluated
side-by-side with medagent checkpoints in both the diagnosis eval and the
conversation eval.

Fairness: every DiagAgent-facing knob is taken verbatim from their HF model
card (``https://huggingface.co/Henrychur/DiagAgent-14B``) — same system
prompt, same EHR input template, ``temperature=0.0``, ``max_new_tokens=1024``.
The adapter only translates between medagent's chat-with-tool-calls protocol
and DiagAgent's free-text exam-recommendation protocol; it does not modify
DiagAgent's prompts or decoding behavior in any other way.

What the adapter does on each ``generate(messages)``:
  1. Render an EHR-style user message (Patient Information / CC / HPI / PMH /
     Allergy / Examination Results) from the medagent message stream and any
     plumbed ``patient_kwargs`` (set by the conv-mode harness).
  2. Call the inner backend with ``[DIAGAGENT_SYSTEM_PROMPT, ehr_user_msg]``.
  3. Translate output:
        ``Diagnosis: X``                    -> append ``[DIAGNOSIS: X]``
        ``... should be performed: Y``      -> append ``<tool_call>{"name":...}</tool_call>``
     The tool name is picked from the case's available_tool_names by an LLM
     matcher using the same client as the judge (default ``gpt-4.1-mini``).
     This matcher is part of the harness, not the model — DiagAgent never
     sees medagent's tool taxonomy.

What it does NOT do: change DiagAgent's prompts, change its decoding, hide
exam findings, or feed it conversational small-talk. Patient turns (rare
under DiagAgent's flow) get folded into HPI; everything else stays pristine.
"""

import json
import re

from .data import parse_available_tools_from_system_prompt


# Verbatim from Henrychur/DiagAgent-14B model card (HF, accessed 2026-04).
DIAGAGENT_SYSTEM_PROMPT = """You are a medical AI assistant. Analyze patient information, suggest relevant tests, and provide a final diagnosis when sufficient information is available.

RESPONSE FORMAT:
If more information is needed:
```
Current diagnosis: <your current best diagnosis>
Based on the patient's initial presentation, the following investigation(s) should be performed: <one additional test>
Reason: <reason for the test>
```

If sufficient information exists for diagnosis:
```
The available information is sufficient to make a diagnosis.
Diagnosis: <final diagnosis>
Reason: <brief justification>
```"""

DIAGAGENT_TEMPERATURE = 0.0
DIAGAGENT_MAX_NEW_TOKENS = 1024


# Detection patterns for parsing DiagAgent output.
_RE_DIAG_SUFFICIENT = re.compile(r"sufficient", re.IGNORECASE)
_RE_DIAG_LINE = re.compile(r"(?im)^\s*Diagnosis\s*:\s*(.+?)\s*$")
_RE_EXAM_REQ = re.compile(
    r"investigation\(?s?\)?\s+should\s+be\s+performed\s*:\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)
_RE_TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_RE_AGE_LINE = re.compile(r"(?im)^\s*Age\s*:\s*([^\n]+)")
_RE_GENDER_LINE = re.compile(r"(?im)^\s*Gender\s*:\s*([^\n]+)")
_RE_MED_HISTORY_BLOCK = re.compile(
    r"(?ims)PATIENT'?S?\s+MEDICAL\s+HISTORY\s*:\s*(.+?)(?:\n\s*\n|\n\s*[A-Z][A-Z ]+:)"
)


class DiagAgentBackend:
    """Adapter wrapping an inner backend that talks to a served DiagAgent."""

    def __init__(self, inner_backend, matcher_client=None, matcher_model="gpt-4.1-mini"):
        self.inner = inner_backend
        self.matcher_client = matcher_client
        self.matcher_model = matcher_model
        self._matcher_cache: dict[tuple[str, tuple[str, ...]], str] = {}

        # DiagAgent's HF card specifies temperature=0.0, max_new_tokens=1024.
        if hasattr(self.inner, "temperature"):
            self.inner.temperature = DIAGAGENT_TEMPERATURE
        for attr in ("max_tokens", "max_new_tokens"):
            if hasattr(self.inner, attr):
                setattr(self.inner, attr, DIAGAGENT_MAX_NEW_TOKENS)

    def load(self):
        self.inner.load()

    # ---- backend interface ---------------------------------------------------

    def generate(self, messages, max_new_tokens=None, max_tokens=None):
        ehr_user_msg = self._render_ehr(messages)
        diag_messages = [
            {"role": "system", "content": DIAGAGENT_SYSTEM_PROMPT},
            {"role": "user", "content": ehr_user_msg},
        ]
        raw = self._call_inner(diag_messages, max_new_tokens, max_tokens)
        return self._translate_output(raw, messages)

    def generate_continuation(self, messages, prefix_text, max_new_tokens=None, max_tokens=None):
        # Tool-call-format eval prefills assistant text and asks the model to
        # continue. DiagAgent doesn't speak that protocol — fall back to a
        # full generate so the harness still gets a structurally valid output.
        return self.generate(messages, max_new_tokens=max_new_tokens, max_tokens=max_tokens)

    def _call_inner(self, messages, max_new_tokens, max_tokens):
        # Inner backends differ on kwarg name; try in order.
        for kw in ({"max_tokens": max_tokens or DIAGAGENT_MAX_NEW_TOKENS},
                   {"max_new_tokens": max_new_tokens or DIAGAGENT_MAX_NEW_TOKENS},
                   {}):
            try:
                return self.inner.generate(messages, **kw)
            except TypeError:
                continue
        return self.inner.generate(messages)

    # ---- input rendering -----------------------------------------------------

    def _render_ehr(self, messages):
        """Build DiagAgent's expected EHR-style user message.

        Two information sources, used together when available:
          - patient_kwargs (plumbed via _ACTIVE_PATIENT_KWARGS in conv mode):
            authoritative for demographics, CC, HPI seed, PMH.
          - the message stream itself: parsed for already-revealed exam
            findings (tool messages) and any patient utterances that arrived
            after the seed (rare under DiagAgent's flow).
        """
        patient_kwargs = _get_active_patient_kwargs() or {}

        system_content = ""
        first_user = ""
        extra_user_turns: list[str] = []
        exam_entries: list[tuple[str, str]] = []  # (label, findings)
        last_tool_call_name: str | None = None

        for msg in messages:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role == "system":
                system_content = content
            elif role == "user":
                if not first_user:
                    first_user = content
                else:
                    extra_user_turns.append(content)
            elif role == "assistant":
                m = _RE_TOOL_CALL_TAG.search(content)
                if m:
                    try:
                        obj = json.loads(m.group(1))
                        last_tool_call_name = obj.get("name") or None
                    except (json.JSONDecodeError, ValueError):
                        last_tool_call_name = None
            elif role == "tool":
                exam_entries.append((last_tool_call_name or "exam", content))
                last_tool_call_name = None

        # Demographics: prefer patient_kwargs, fall back to system-prompt parse.
        age = patient_kwargs.get("age", "") or ""
        gender = patient_kwargs.get("gender", "") or ""
        if not (age or gender) and system_content:
            age, gender = self._parse_demographics(system_content)
        sex_letter = self._sex_letter(gender)
        age_str = self._age_str(age, sex_letter)

        # Chief Complaint + HPI.
        symptoms = (patient_kwargs.get("self_reported_symptoms") or "").strip()
        if symptoms:
            cc = symptoms.splitlines()[0].strip() or symptoms
            hpi_lines = [symptoms]
            if first_user and first_user not in symptoms:
                hpi_lines.append(first_user)
            hpi_lines.extend(extra_user_turns)
        else:
            cc = first_user or "Not stated"
            hpi_lines = ([first_user] if first_user else []) + extra_user_turns
            if not hpi_lines:
                hpi_lines = ["Not stated"]
        hpi = "\n".join(line for line in hpi_lines if line)

        # Past medical history.
        pmh = (patient_kwargs.get("medical_history") or "").strip()
        if not pmh and system_content:
            pmh = self._parse_medical_history(system_content)
        if not pmh:
            pmh = "Not stated"

        sections: list[str] = []
        if age_str:
            sections.append(f"- Patient Information: {age_str}")
        sections.append(f"- Chief Complaint: {cc}")
        sections.append(f"- HPI: {hpi}")
        sections.append(f"- PMH: {pmh}")
        sections.append("- Allergy: Not stated")

        if exam_entries:
            results_block = ["", "Examination Results:"]
            for label, findings in exam_entries:
                pretty = label.replace("_", " ")
                results_block.append(f"- {pretty}: {findings}")
            sections.append("\n".join(results_block))

        return "\n".join(sections)

    # ---- output translation --------------------------------------------------

    def _translate_output(self, raw, original_messages):
        if not raw:
            return raw or ""

        if _RE_DIAG_SUFFICIENT.search(raw):
            m = _RE_DIAG_LINE.search(raw)
            if m:
                diag = m.group(1).strip().rstrip(".").strip()
                if diag:
                    return f"{raw}\n\n[DIAGNOSIS: {diag}]"

        m = _RE_EXAM_REQ.search(raw)
        if m:
            exam_text = m.group(1).strip().rstrip(".").strip()
            tool_name = self._match_exam_to_tool(exam_text, original_messages)
            if tool_name:
                obj = {"name": tool_name, "arguments": {}}
                return f"{raw}\n\n<tool_call>\n{json.dumps(obj, ensure_ascii=False)}\n</tool_call>"

        return raw

    def _match_exam_to_tool(self, exam_text, original_messages):
        avail = set()
        for msg in original_messages:
            if msg.get("role") == "system":
                avail = parse_available_tools_from_system_prompt(msg.get("content") or "")
                break
        if not avail:
            return None
        avail_sorted = tuple(sorted(avail))
        cache_key = (exam_text.lower(), avail_sorted)
        if cache_key in self._matcher_cache:
            return self._matcher_cache[cache_key]

        if self.matcher_client is not None:
            picked = self._llm_match(exam_text, avail_sorted)
            if picked:
                self._matcher_cache[cache_key] = picked
                return picked
        picked = self._heuristic_match(exam_text, avail_sorted)
        if picked:
            self._matcher_cache[cache_key] = picked
        return picked

    def _llm_match(self, exam_text, available_names):
        prompt = (
            "Map the free-text medical examination request to ONE tool name from "
            "the list. Reply with only the tool name (no quotes, no explanation). "
            "If nothing fits, reply with the closest single match.\n\n"
            f"Free-text request: {exam_text}\n\n"
            f"Available tools: {', '.join(available_names)}\n\n"
            "Tool name:"
        )
        try:
            resp = self.matcher_client.chat.completions.create(
                model=self.matcher_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_completion_tokens=32,
            )
            picked = (resp.choices[0].message.content or "").strip()
        except Exception:
            return None
        picked = picked.split()[0] if picked else ""
        picked = picked.strip(".,:;`'\"")
        return picked if picked in available_names else None

    @staticmethod
    def _heuristic_match(exam_text, available_names):
        # Tokenize on alphanumerics: tool names like "ct_chest" must split into
        # {"ct", "chest"} so they overlap with free-text "CT scan of the chest".
        token_re = re.compile(r"[a-z0-9]+")
        et_tokens = set(token_re.findall(exam_text.lower()))
        if not et_tokens:
            return None
        best = None
        best_score = 0
        for name in available_names:
            nt = set(token_re.findall(name.lower()))
            score = len(et_tokens & nt)
            if score > best_score:
                best_score = score
                best = name
        return best

    # ---- system-prompt parsing helpers --------------------------------------

    @staticmethod
    def _parse_demographics(system_content):
        age_m = _RE_AGE_LINE.search(system_content)
        gen_m = _RE_GENDER_LINE.search(system_content)
        age = age_m.group(1).strip() if age_m else ""
        gender = gen_m.group(1).strip() if gen_m else ""
        return age, gender

    @staticmethod
    def _parse_medical_history(system_content):
        m = _RE_MED_HISTORY_BLOCK.search(system_content)
        if m:
            return m.group(1).strip()
        return ""

    @staticmethod
    def _sex_letter(gender):
        g = (gender or "").strip().lower()
        if g.startswith("f"):
            return "F"
        if g.startswith("m"):
            return "M"
        return ""

    @staticmethod
    def _age_str(age, sex_letter):
        age = (age or "").strip()
        if age and sex_letter:
            return f"{age} y/o {sex_letter}"
        if age:
            return f"{age} y/o"
        if sex_letter:
            return sex_letter
        return ""


# ---------------------------------------------------------------------------
# Plumbing: the conv-mode eval sets _ACTIVE_PATIENT_KWARGS per sample so the
# adapter can populate Demographics / CC / HPI / PMH on turn 1, before any
# patient utterance has arrived. weave_eval._model_predict imports & sets
# this when patient_kwargs is provided. Diag-mode leaves it None and the
# adapter falls back to parsing the system prompt.
# ---------------------------------------------------------------------------

_ACTIVE_PATIENT_KWARGS: dict | None = None


def set_active_patient_kwargs(patient_kwargs: dict | None) -> None:
    global _ACTIVE_PATIENT_KWARGS
    _ACTIVE_PATIENT_KWARGS = patient_kwargs


def _get_active_patient_kwargs() -> dict | None:
    return _ACTIVE_PATIENT_KWARGS
