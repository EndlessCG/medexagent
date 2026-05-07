"""Helpers for attaching full conversations to evaluation traces."""

from __future__ import annotations

import json
from typing import Any


def serialize_for_trace(value: Any) -> Any:
    """Return a JSON-safe deep copy for Weave trace payloads."""
    return json.loads(json.dumps(value, ensure_ascii=False))


def format_conversation(messages: list[dict[str, Any]], include_system: bool = True) -> str:
    """Render a full conversation transcript without truncating message content."""
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "?")).upper()
        if not include_system and role == "SYSTEM":
            continue

        content = msg.get("content", "") or ""
        lines.append(f"[{role}] {content}")

        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            arguments = fn.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            lines.append(f"[TOOL_CALL] {name} {json.dumps(arguments, ensure_ascii=False, sort_keys=True)}")

    return "\n".join(lines)
