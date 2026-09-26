"""Adapter for LangChain tools and tool calls (lazy, no hard dependency)."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Optional

from ..registry import ToolRegistry
from ..types import ToolCall, ToolDefinition
from .openai import parse_arguments


def from_langchain_tool(tool: Any) -> ToolDefinition:
    """Convert a LangChain tool (BaseTool) into a ToolDefinition.

    Reads ``name``, ``description``, and ``args`` (dict) or ``args_schema``
    (Pydantic model) without importing LangChain.
    """
    name = str(getattr(tool, "name", "") or getattr(tool, "__name__", ""))
    description = str(getattr(tool, "description", "") or "")

    parameters: dict[str, Any] = {}
    schema_obj = getattr(tool, "args_schema", None)
    if schema_obj is not None:
        parameters = _schema_to_json(schema_obj)
    if not parameters:
        raw_args = getattr(tool, "args", None)
        if callable(raw_args):
            try:
                raw_args = raw_args()
            except Exception:
                raw_args = None
        if isinstance(raw_args, dict):
            parameters = {"type": "object", "properties": raw_args}

    return ToolDefinition(name=name, description=description, parameters=parameters)


def langchain_tool_call(obj: Any) -> ToolCall:
    """Convert a LangChain tool call (dict or dataclass) into a ToolCall.

    LangChain's ``AIMessage.tool_calls`` entries use ``args`` (not
    ``arguments``), which this function handles directly.
    """
    if isinstance(obj, dict):
        name = obj.get("name", "")
        raw_args = obj.get("args", obj.get("arguments"))
    else:
        name = getattr(obj, "name", "")
        raw_args = getattr(obj, "args", getattr(obj, "arguments", {}))
    return ToolCall(name=str(name), arguments=parse_arguments(raw_args))


def to_langchain_tools(
    registry: ToolRegistry,
    functions: Optional[Mapping[str, Callable[[dict[str, Any]], Any]]] = None,
) -> list[Any]:
    """Build LangChain ``Tool`` objects from a ToolRegistry.

    Requires ``langchain-core``; raises ImportError with a hint if missing.
    Arguments are passed to the wrapped function as a JSON string, matching the
    legacy ``Tool`` interface.
    """
    try:
        from langchain_core.tools import Tool
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "langchain-core is required for to_langchain_tools; "
            "install with `pip install langchain-core`"
        ) from exc

    functions = dict(functions or {})
    tools: list[Any] = []
    for name in registry.names():
        tool = registry.get(name)
        if tool is None:
            continue
        func = functions.get(name)
        if func is None:
            func = lambda _args, _name=name: {"tool": _name, "unimplemented": True}
        tools.append(
            Tool(
                name=tool.name,
                description=tool.description,
                func=_string_wrapper(func),
            )
        )
    return tools


def _string_wrapper(func: Callable[[dict[str, Any]], Any]) -> Callable[[str], Any]:
    def runner(input_text: str) -> Any:
        if isinstance(input_text, str) and input_text.strip():
            args = json.loads(input_text)
        else:
            args = {}
        return func(args)

    return runner


def _schema_to_json(schema_obj: Any) -> dict[str, Any]:
    for method in ("model_json_schema", "schema"):
        fn = getattr(schema_obj, method, None)
        if callable(fn):
            try:
                result = fn()
            except Exception:
                continue
            if isinstance(result, dict):
                return result
    return {}
