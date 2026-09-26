"""Policy controls for AgentSentinel guardrails."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Pattern


@dataclass
class Policy:
    """Runtime rules layered on top of the static tool registry."""

    denylist: set[str] = field(default_factory=set)
    approved_tools: set[str] = field(default_factory=set)
    # Dangerous patterns are ON by default (safety-first). Pass [] to opt out.
    sensitive_patterns: list[Pattern[str]] = field(
        default_factory=lambda: list(DANGEROUS_PATTERNS)
    )

    def deny(self, tool_name: str) -> None:
        """Block a tool entirely, regardless of arguments."""
        self.denylist.add(tool_name)

    def approve(self, tool_name: str) -> None:
        """Grant approval for a tool that requires it."""
        self.approved_tools.add(tool_name)

    def approved(self, tool_name: str) -> bool:
        return tool_name in self.approved_tools


DANGEROUS_PATTERNS: list[Pattern[str]] = [
    re.compile(r"rm\s+-rf"),
    re.compile(r"DROP\s+TABLE", re.IGNORECASE),
    re.compile(r"DROP\s+DATABASE", re.IGNORECASE),
    re.compile(r"DELETE\s+FROM", re.IGNORECASE),
    re.compile(r"mkfs\."),
    re.compile(r"shutdown", re.IGNORECASE),
]


def default_policy() -> Policy:
    """Return a Policy with the default dangerous patterns enabled."""
    return Policy()
