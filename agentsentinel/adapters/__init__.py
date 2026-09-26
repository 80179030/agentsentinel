"""Framework adapters for AgentSentinel."""

from .langchain import from_langchain_tool, langchain_tool_call, to_langchain_tools
from .openai import (
    DispatchResult,
    SafeDispatcher,
    openai_tools,
    parse_arguments,
    to_tool_call,
)

__all__ = [
    "DispatchResult",
    "SafeDispatcher",
    "openai_tools",
    "parse_arguments",
    "to_tool_call",
    "from_langchain_tool",
    "langchain_tool_call",
    "to_langchain_tools",
]
