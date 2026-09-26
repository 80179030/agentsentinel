"""Closed-world tool registry."""

from __future__ import annotations

from typing import Iterable, Optional

from .types import ToolDefinition


class ToolRegistry:
    """Holds the set of tools an agent is actually allowed to call.

    Anything not in this registry is treated as a hallucinated tool.
    """

    def __init__(self, tools: Optional[Iterable[ToolDefinition]] = None) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools.keys())
