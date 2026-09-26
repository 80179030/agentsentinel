"""Minimal observability layer: record every decision for later audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import Decision, ToolCall


@dataclass
class TraceEntry:
    call: ToolCall
    decision: Decision


class TraceLogger:
    """Append-only log of guardrail decisions, ready for audit or analytics."""

    def __init__(self) -> None:
        self.entries: list[TraceEntry] = []

    def record(self, call: ToolCall, decision: Decision) -> None:
        self.entries.append(TraceEntry(call=call, decision=decision))

    def as_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "tool": entry.call.name,
                "arguments": dict(entry.call.arguments),
                "verdict": entry.decision.verdict.value,
                "reasons": entry.decision.reasons,
                "gaps": [gap.value for gap in entry.decision.gaps],
            }
            for entry in self.entries
        ]
