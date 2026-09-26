"""Adapters for OpenAI-style function calling (and similar tool-call formats)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..guard import SafeStep
from ..registry import ToolRegistry
from ..types import Decision, ToolCall


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Normalize tool arguments from a dict or a JSON string."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        return json.loads(raw)
    raise TypeError(f"cannot parse arguments from {type(raw).__name__}")


def to_tool_call(obj: Any) -> ToolCall:
    """Convert a dict or an OpenAI-SDK tool-call object to a ToolCall."""
    if isinstance(obj, dict):
        return ToolCall(
            name=str(obj.get("name", "")),
            arguments=parse_arguments(obj.get("arguments")),
        )
    fn = getattr(obj, "function", None)
    if fn is not None:
        return ToolCall(
            name=getattr(fn, "name", ""),
            arguments=parse_arguments(getattr(fn, "arguments", "")),
        )
    raise TypeError("unsupported tool-call object; pass a dict or an OpenAI tool call")


def openai_tools(registry: ToolRegistry) -> list[dict[str, Any]]:
    """Build the OpenAI ``tools`` array from a ToolRegistry."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": dict(tool.parameters),
            },
        }
        for tool in (registry.get(name) for name in registry.names())
        if tool is not None
    ]


@dataclass
class DispatchResult:
    decision: Decision
    result: Any = None

    @property
    def blocked(self) -> bool:
        return not self.decision.allowed


class SafeDispatcher:
    """Guard a tool call, then run the registered function only if allowed."""

    def __init__(
        self,
        registry: ToolRegistry,
        guard: SafeStep,
        functions: Mapping[str, Callable[[dict[str, Any]], Any]],
    ) -> None:
        self.registry = registry
        self.guard = guard
        self.functions = dict(functions)

    def dispatch(self, call: ToolCall) -> DispatchResult:
        decision = self.guard.check(call)
        if not decision.allowed:
            return DispatchResult(decision=decision)
        fn = self.functions[call.name]
        return DispatchResult(decision=decision, result=fn(dict(call.arguments)))
