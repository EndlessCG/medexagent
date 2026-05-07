"""Rule-based patient simulator for RL rollouts.

No LLM inference — uses templates and keyword matching to respond to doctor
questions based on the patient profile. Fast enough for RL rollout scale.
"""

import re


class RuleBasedPatientAgent:
    """Simulates patient responses during RL rollouts."""

    def __init__(self, demographics: dict, medical_history: str, self_reported_symptoms: str):
        self.demographics = demographics
        self.medical_history = medical_history
        self.symptoms = [s.strip() for s in self_reported_symptoms.split(",") if s.strip()]
        self.turn = 0
        self.revealed_symptoms = set()

    def respond(self, doctor_message: str) -> str:
        """Generate a patient response based on the doctor's message."""
        self.turn += 1
        doctor_lower = doctor_message.lower()

        # Turn 1: Report chief complaint (first 1-2 symptoms)
        if self.turn == 1:
            return self._chief_complaint()

        # Check if doctor is asking about medical history
        if self._asking_about_history(doctor_lower):
            return self._report_history()

        # Check if doctor is asking about demographics
        if self._asking_about_demographics(doctor_lower):
            return self._report_demographics()

        # Check if doctor ordered an exam (acknowledge it)
        if self._is_exam_notification(doctor_lower):
            return self._acknowledge_exam()

        # Check if doctor is asking about symptoms
        if self._asking_about_symptoms(doctor_lower):
            return self._reveal_symptoms()

        # Default: reveal next unrevealed symptoms
        return self._reveal_symptoms()

    def _chief_complaint(self) -> str:
        """Report the first 1-2 symptoms as chief complaint."""
        initial = self.symptoms[:2]
        self.revealed_symptoms.update(initial)
        if len(initial) == 1:
            return f"Hello, Doctor. I've been having {initial[0]}. It's been bothering me quite a bit."
        return (
            f"Hello, Doctor. I've been experiencing {initial[0]} and {initial[1]}. "
            "It's been really concerning me."
        )

    def _reveal_symptoms(self) -> str:
        """Reveal the next batch of unrevealed symptoms."""
        unrevealed = [s for s in self.symptoms if s not in self.revealed_symptoms]
        if not unrevealed:
            return "I think I've told you about all my symptoms, Doctor."

        # Reveal 1-2 at a time
        batch = unrevealed[:2]
        self.revealed_symptoms.update(batch)
        if len(batch) == 1:
            return f"I've also been experiencing {batch[0]}."
        return f"I've also noticed {batch[0]} and {batch[1]}."

    def _report_history(self) -> str:
        """Report medical history."""
        if self.medical_history:
            return f"Regarding my medical history, I have {self.medical_history}."
        return "I don't have any significant medical history that I know of."

    def _report_demographics(self) -> str:
        """Report demographics."""
        parts = []
        if self.demographics.get("age"):
            parts.append(f"I'm {self.demographics['age']} old")
        if self.demographics.get("gender"):
            parts.append(f"{self.demographics['gender'].lower()}")
        if parts:
            return f"I'm a {', '.join(parts)}."
        return "I'd rather not go into personal details right now."

    def _acknowledge_exam(self) -> str:
        """Acknowledge an exam being performed."""
        return "Okay, Doctor. I understand."

    def _asking_about_history(self, msg: str) -> bool:
        keywords = ["medical history", "past medical", "previous condition",
                     "prior condition", "pre-existing", "chronic", "medication",
                     "allergies", "allergy", "surgical history"]
        return any(k in msg for k in keywords)

    def _asking_about_demographics(self, msg: str) -> bool:
        keywords = ["how old", "your age", "gender", "ethnicity", "race"]
        return any(k in msg for k in keywords)

    def _asking_about_symptoms(self, msg: str) -> bool:
        keywords = ["symptom", "feeling", "pain", "discomfort", "experience",
                     "notice", "other complaint", "anything else", "what else",
                     "how long", "when did", "severity", "worse", "better",
                     "onset", "duration"]
        return any(k in msg for k in keywords)

    def _is_exam_notification(self, msg: str) -> bool:
        keywords = ["order", "x-ray", "blood test", "scan", "exam", "test",
                     "biopsy", "ultrasound", "mri", "ct scan", "ecg"]
        return any(k in msg for k in keywords)
