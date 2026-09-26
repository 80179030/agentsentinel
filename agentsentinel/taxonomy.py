"""Production failure taxonomy for agent trajectories."""

from __future__ import annotations

from enum import Enum


class FailureCategory(str, Enum):
    """Six top-level failure families seen in deployed agentic systems."""

    DRIFT = "drift"                    # gradual departure from the original intent
    STATE = "state"                    # memory / context / world-state errors
    COORDINATION = "coordination"      # multi-agent handoff or race conditions
    TERMINATION = "termination"        # premature stop, loops, or budget exhaustion
    ADVERSARIAL = "adversarial"        # injection, reward hacking, or unauthorized action
    TOOL_INTERFACE = "tool_interface"  # hallucinated tool, bad args, or execution error


CATEGORY_DESCRIPTIONS = {
    FailureCategory.DRIFT: "gradual departure from the original intent",
    FailureCategory.STATE: "memory, context, or world-state management error",
    FailureCategory.COORDINATION: "multi-agent handoff or race condition",
    FailureCategory.TERMINATION: "premature stop, infinite loop, or budget exhaustion",
    FailureCategory.ADVERSARIAL: "prompt injection, reward hacking, or unauthorized action",
    FailureCategory.TOOL_INTERFACE: "hallucinated tool, bad arguments, or execution error",
}


def describe(category: FailureCategory) -> str:
    return CATEGORY_DESCRIPTIONS.get(category, category.value)
