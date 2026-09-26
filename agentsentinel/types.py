"""Core data types for AgentSentinel."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Verdict(str, Enum):
    """What a guardrail decided about a proposed tool call."""

    ALLOW = "allow"
    DENY = "deny"
    ABSTAIN = "abstain"


class GapKind(str, Enum):
    """Why abstention was required (the three-gap taxonomy)."""

    SPECIFICATION = "specification"  # missing information needed to proceed safely
    VERIFICATION = "verification"    # world state / precondition cannot be confirmed
    AUTHORITY = "authority"          # explicit permission was not granted


@dataclass
class ToolCall:
    """A normalized tool invocation produced by an LLM."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """A tool as registered in the closed-world registry."""

    name: str
    description: str = ""
    # JSON Schema for the arguments, e.g. {"type": "object", "properties": {...}, "required": [...]}
    parameters: Mapping[str, Any] = field(default_factory=dict)
    # Tools that should never run without an explicit grant (e.g. send_email, delete_record).
    requires_approval: bool = False


@dataclass
class Decision:
    """The outcome of a guardrail check, with structured reasoning."""

    verdict: Verdict
    reasons: list[str] = field(default_factory=list)
    gaps: list[GapKind] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.verdict is Verdict.ALLOW
