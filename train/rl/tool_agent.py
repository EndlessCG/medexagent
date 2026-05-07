"""Tool executor for RL rollouts.

Looks up exam results from the patient profile's exam map.
Reuses parse_exam_string and format_findings from data/tool.py.
"""

import json
import random
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.noise import apply_exam_noise
from data.tool import format_findings, parse_exam_string


class ToolAgent:
    """Execute tool calls against a patient's exam profile."""

    def __init__(
        self,
        exam_map: dict,
        available_tools: list[str] | None = None,
        exam_noise_map: dict[str, str] | None = None,
        noise_rng: random.Random | None = None,
    ):
        """Initialize with exam_map and optional available_tools list.

        Args:
            exam_map: {exam_string: {"parsed": {...}, "findings": str}}.
                If values are plain strings (just findings), they are re-parsed.
            available_tools: List of tool function names available to the doctor.
                If provided, calls to tools not in this list return an unavailable message.
            exam_noise_map: Optional {exam_string: noise_type} pre-assigned by
                dataset-level noise planning. When an exam match is found, the
                corresponding noise type is applied deterministically.
            noise_rng: RNG used by apply_exam_noise to pick concrete substitutions.
        """
        self.exam_map = exam_map
        self.available_tools = set(available_tools) if available_tools else None
        self.noise_rng = noise_rng if noise_rng is not None else random.Random(0)
        self.call_count = 0
        self.call_types = []
        self.predicted_tools: list[dict] = []  # {"name": ..., "arguments": {...}}
        self.empty_results = 0
        self.unavailable_results = 0
        self.informative_results: set = set()
        self.exam_noise_counts: dict[str, int] = {}

        # Build a normalized lookup: (func_name, frozenset(arg_values)) -> findings
        self._lookup = {}
        self._noise_by_key: dict[tuple, str] = {}
        noise_by_str = exam_noise_map or {}
        for exam_str, info in exam_map.items():
            if isinstance(info, dict) and "parsed" in info:
                parsed = info["parsed"]
                findings = info["findings"]
            else:
                parsed = parse_exam_string(exam_str)
                findings = info if isinstance(info, str) else str(info)
                if parsed is None:
                    continue

            key = self._make_key(parsed["name"], parsed["arguments"])
            self._lookup[key] = (parsed["name"], parsed["arguments"], findings)
            if exam_str in noise_by_str:
                self._noise_by_key[key] = noise_by_str[exam_str]

    def execute(self, function_name: str, arguments: dict | str) -> str:
        """Execute a tool call and return formatted findings.

        Args:
            function_name: The exam function name (e.g., "xray").
            arguments: The arguments dict or JSON string.

        Returns:
            Formatted findings string, or "No significant findings" if no match.
        """
        arguments = self._normalize_arguments(arguments)

        self.call_count += 1
        self.call_types.append(function_name)
        self.predicted_tools.append({"name": function_name, "arguments": arguments.copy()})

        # Check if the tool is in the available tools list
        if self.available_tools is not None and function_name not in self.available_tools:
            self.unavailable_results += 1
            return "The requested exam is not available."

        # Look up by exact function name + case-insensitive argument values
        key = self._make_key(function_name, arguments)
        match = self._lookup.get(key)

        if match:
            exam_name, exam_args, raw_findings = match
            self.informative_results.add(key)
            raw_findings = self._maybe_apply_noise(key, exam_name, raw_findings)
            return format_findings(exam_name, exam_args, raw_findings)

        # Fallback: match by function name only (if only one exam of that type)
        name_matches = [
            (k, v) for k, v in self._lookup.items() if k[0] == function_name
        ]
        if len(name_matches) == 1:
            matched_key, (exam_name, exam_args, raw_findings) = name_matches[0]
            self.informative_results.add(matched_key)
            raw_findings = self._maybe_apply_noise(matched_key, exam_name, raw_findings)
            return format_findings(exam_name, exam_args, raw_findings)

        self.empty_results += 1
        return "No significant findings."

    def _maybe_apply_noise(self, key: tuple, exam_name: str, raw_findings: str) -> str:
        """Apply dataset-assigned exam noise (if any) to the raw findings."""
        noise_type = self._noise_by_key.get(key)
        if not noise_type:
            return raw_findings
        modified, meta = apply_exam_noise(
            exam_name=exam_name,
            raw_findings=raw_findings,
            noise_level=0.0,
            rng=self.noise_rng,
            forced_noise_type=noise_type,
        )
        if meta is not None:
            applied = meta.get("noise_type", noise_type)
            self.exam_noise_counts[applied] = self.exam_noise_counts.get(applied, 0) + 1
        return modified

    @staticmethod
    def _normalize_arguments(arguments: dict | str) -> dict:
        """Best-effort normalization to a plain dict for robust execution."""
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return {}

        if isinstance(arguments, Mapping):
            return dict(arguments)

        # Some parser outputs can be list/tuple-like; accept key-value pairs if valid.
        try:
            return dict(arguments)
        except (TypeError, ValueError):
            return {}

    def get_stats(self) -> dict:
        """Return tool usage statistics."""
        # Ground truth tool names (from exam_map)
        gt_tool_names = {k[0] for k in self._lookup}
        # Unique tool names the model actually called
        used_tool_names = set(self.call_types)
        # IoU: |intersection| / |union|
        intersection = gt_tool_names & used_tool_names
        union = gt_tool_names | used_tool_names
        tool_iou = len(intersection) / len(union) if union else 0.0
        # Ground truth tools as list of dicts (for ToolRL-style scoring)
        gt_tools = [
            {"name": name, "arguments": dict(args)}
            for name, args, _findings in self._lookup.values()
        ]
        return {
            "total_calls": self.call_count,
            "call_types": self.call_types,
            "predicted_tools": self.predicted_tools,
            "gt_tools": gt_tools,
            "empty_results": self.empty_results,
            "unavailable_results": self.unavailable_results,
            "informative_exams": len(self.informative_results),
            "total_available_exams": len(self._lookup),
            "tool_iou": tool_iou,
            "gt_tool_names": gt_tool_names,
            "used_tool_names": used_tool_names,
            "exam_noise_counts": dict(self.exam_noise_counts),
        }

    @staticmethod
    def _make_key(func_name: str, arguments: dict) -> tuple:
        """Create a normalized lookup key from function name and arguments."""
        try:
            normalized_args = frozenset(
                (k, v.lower().strip() if isinstance(v, str) else str(v))
                for k, v in arguments.items()
                if isinstance(v, (str, int, float, bool))
            )
            return (func_name, normalized_args)
        except Exception:
            print(f"Warning: Failed to normalize arguments for {func_name} with {arguments}")
            return (func_name, frozenset())
