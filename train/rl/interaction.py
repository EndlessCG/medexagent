"""Patient interaction for multi-turn RL rollouts.

Calls a separate LLM server (or rule-based fallback) to simulate the patient
side of the medical consultation.
"""

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Optional

import aiohttp

from verl.interactions.base import BaseInteraction

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.noise import PatientNoiseSpec, get_patient_turn_hint, plan_patient_noise
from data.prompt import (
    format_exam_notification_for_patient,
    get_patient_system_prompt,
)
from train.rl.patient_agent import RuleBasedPatientAgent

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class PatientInteraction(BaseInteraction):
    """Patient interaction via external LLM server or rule-based fallback."""

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.server_url = config.get("server_url", "rule_based")
        self.model_name = config.get("model_name", "")
        self.max_tokens = config.get("max_tokens", 256)
        self.temperature = config.get("temperature", 0.7)
        self.api_key = config.get("api_key") or os.environ.get("OPENAI_API_KEY", "")

        # Per-instance state: instance_id -> {patient_system_prompt, rule_based_agent, ...}
        self._instances: dict[str, dict] = {}

    async def start_interaction(self, instance_id: Optional[str] = None, **kwargs) -> str:
        instance_id = await super().start_interaction(instance_id)

        patient_system_prompt = get_patient_system_prompt(
            age=kwargs.get("age", ""),
            gender=kwargs.get("gender", ""),
            race=kwargs.get("race", ""),
            additional_demographics=kwargs.get("additional_demographics", ""),
            medical_history=kwargs.get("medical_history", ""),
            self_reported_symptoms=kwargs.get("self_reported_symptoms", ""),
            patient_persona=kwargs.get("patient_persona"),
        )

        # Optional: plan patient noise for this rollout using pre-assigned
        # noise types from dataset-level planning. Matches data generation
        # semantics (see data/generate.py:_plan_noise_assignments): only a
        # fraction of patients receive noise, each selected patient gets a
        # specific set of types.
        # Seed by patient_index so all group_size rollouts of the same
        # patient see the same noise realization (reduces advantage variance).
        assigned_types_raw = kwargs.get("assigned_noise_types")
        patient_index = int(kwargs.get("patient_index", 0) or 0)
        symptoms = kwargs.get("self_reported_symptoms", "") or ""
        assigned_types: set[str] | None = None
        if assigned_types_raw:
            if isinstance(assigned_types_raw, str):
                try:
                    parsed = json.loads(assigned_types_raw)
                    assigned_types = set(parsed) if parsed else None
                except (json.JSONDecodeError, TypeError):
                    assigned_types = None
            elif isinstance(assigned_types_raw, (list, tuple, set)):
                assigned_types = set(assigned_types_raw) if assigned_types_raw else None

        noise_rng = random.Random(patient_index * 1_000_003 + 7)
        if assigned_types and symptoms:
            patient_noise_spec = plan_patient_noise(
                symptoms, 0.0, noise_rng, forced_noise_types=assigned_types,
            )
        else:
            patient_noise_spec = PatientNoiseSpec()

        state: dict[str, Any] = {
            "patient_system_prompt": patient_system_prompt,
            "patient_noise_spec": patient_noise_spec,
            "noise_rng": noise_rng,
            "patient_turn_index": 0,
            "noise_counts": {},
        }

        if self.server_url == "rule_based":
            state["rule_based_agent"] = RuleBasedPatientAgent(
                demographics={
                    "age": kwargs.get("age", ""),
                    "gender": kwargs.get("gender", ""),
                    "race": kwargs.get("race", ""),
                    "additional_demographics": kwargs.get("additional_demographics", ""),
                },
                medical_history=kwargs.get("medical_history", ""),
                self_reported_symptoms=kwargs.get("self_reported_symptoms", ""),
            )

        self._instances[instance_id] = state
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        state = self._instances.get(instance_id, {})

        if self.server_url == "rule_based":
            return await self._rule_based_response(state, messages)
        else:
            return await self._llm_response(state, messages)

    async def _rule_based_response(
        self, state: dict, messages: list[dict[str, Any]]
    ) -> tuple[bool, str, float, dict[str, Any]]:
        agent: RuleBasedPatientAgent = state["rule_based_agent"]

        # Find the last assistant message to respond to
        last_assistant_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if content:
                    last_assistant_msg = content
                    break

        response = agent.respond(last_assistant_msg)
        return False, response, 0.0, {}

    async def _llm_response(
        self, state: dict, messages: list[dict[str, Any]]
    ) -> tuple[bool, str, float, dict[str, Any]]:
        patient_messages = self._convert_to_patient_perspective(
            state["patient_system_prompt"], messages
        )

        # Inject per-turn noise hint (if any) as an extra system message before
        # the patient LLM is queried. Mirrors data/generate.py's generation path.
        noise_type_applied: str | None = None
        spec: PatientNoiseSpec = state.get("patient_noise_spec") or PatientNoiseSpec()
        if spec.noise_turns:
            turn_idx = state.get("patient_turn_index", 0)
            # Find the most recent doctor message (becomes a user msg in patient view)
            last_doctor_msg = ""
            for m in reversed(patient_messages):
                if m.get("role") == "user":
                    last_doctor_msg = m.get("content", "") or ""
                    break
            if last_doctor_msg:
                hint = get_patient_turn_hint(
                    spec, turn_idx, last_doctor_msg, state["noise_rng"],
                )
                if hint is not None:
                    hint_text, noise_type_applied = hint
                    patient_messages.append({"role": "system", "content": hint_text})
                    state["noise_counts"][noise_type_applied] = (
                        state["noise_counts"].get(noise_type_applied, 0) + 1
                    )
        state["patient_turn_index"] = state.get("patient_turn_index", 0) + 1

        url = f"{self.server_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": patient_messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                    # Handle error responses first; OpenAI/edge proxies may return text/plain.
                    if resp.status >= 400:
                        body = await resp.text()
                        logger.error(
                            "Patient LLM HTTP error: status=%s content_type=%s body=%s",
                            resp.status,
                            resp.headers.get("Content-Type", ""),
                            body[:500],
                        )
                        return False, "I'm not sure what to say, Doctor.", 0.0, {}

                    # Prefer JSON, but tolerate unexpected content-types.
                    try:
                        result = await resp.json(content_type=None)
                    except Exception:
                        body = await resp.text()
                        logger.error(
                            "Patient LLM invalid JSON response: status=%s content_type=%s body=%s",
                            resp.status,
                            resp.headers.get("Content-Type", ""),
                            body[:500],
                        )
                        return False, "I'm not sure what to say, Doctor.", 0.0, {}

                    if "error" in result:
                        logger.error(f"Patient LLM API error: {result['error']}")
                        return False, "I'm not sure what to say, Doctor.", 0.0, {}
                    response_text = result["choices"][0]["message"]["content"]
                    return False, response_text, 0.0, {}
        except Exception as e:
            logger.error(f"Patient LLM request failed: {e}")
            return False, "I'm not sure what to say, Doctor.", 0.0, {}

    def _convert_to_patient_perspective(
        self, patient_system_prompt: str, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert doctor-perspective conversation to patient-perspective.

        Swap assistant/user roles and convert tool messages to notifications.
        """
        patient_messages = [{"role": "system", "content": patient_system_prompt}]

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                # Skip doctor's system prompt
                continue
            elif role == "assistant":
                # Doctor's message becomes user message for patient
                if content:
                    patient_messages.append({"role": "user", "content": content})
            elif role == "user":
                # Patient's previous response becomes assistant message
                patient_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                # Convert tool results to a notification for the patient
                tool_call_id = msg.get("tool_call_id", "")
                # Find the corresponding tool call in the previous assistant message
                exam_name = ""
                exam_args = {}
                for prev_msg in reversed(messages):
                    if prev_msg.get("role") == "assistant" and prev_msg.get("tool_calls"):
                        for tc in prev_msg["tool_calls"]:
                            if tc.get("id") == tool_call_id:
                                exam_name = tc["function"]["name"]
                                args = tc["function"].get("arguments", "{}")
                                exam_args = json.loads(args) if isinstance(args, str) else args
                                break
                        break
                if exam_name:
                    notification = format_exam_notification_for_patient(exam_name, exam_args)
                    patient_messages.append({"role": "user", "content": notification})

        return patient_messages

    def get_noise_stats(self, instance_id: str) -> dict[str, int]:
        """Return per-type patient-noise injection counts for this rollout."""
        state = self._instances.get(instance_id)
        if not state:
            return {}
        return dict(state.get("noise_counts", {}))

    async def finalize_interaction(self) -> None:
        self._instances.clear()
