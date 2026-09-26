"""Tests for the OpenAI adapter. Run: python tests/test_adapters.py"""

from agentsentinel import SafeStep, ToolCall, ToolDefinition, ToolRegistry
from agentsentinel.adapters import (
    SafeDispatcher,
    openai_tools,
    parse_arguments,
    to_tool_call,
)


def _registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(
                name="add",
                parameters={
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            ),
            ToolDefinition(
                name="danger",
                parameters={
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                },
                requires_approval=True,
            ),
        ]
    )


class _FakeFn:
    """Mimic the shape of an OpenAI-SDK tool call object."""

    def __init__(self, name, arguments):
        self.function = _FakeFunction(name, arguments)


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


def test_parse_arguments_dict():
    assert parse_arguments({"a": 1}) == {"a": 1}


def test_parse_arguments_json_string():
    assert parse_arguments('{"a": 1}') == {"a": 1}


def test_parse_arguments_empty():
    assert parse_arguments("") == {}
    assert parse_arguments(None) == {}


def test_to_tool_call_from_dict():
    call = to_tool_call({"name": "add", "arguments": '{"a": 1, "b": 2}'})
    assert call.name == "add"
    assert call.arguments == {"a": 1, "b": 2}


def test_to_tool_call_from_sdk_object():
    call = to_tool_call(_FakeFn("add", '{"a": 1, "b": 2}'))
    assert call.name == "add"
    assert call.arguments == {"a": 1, "b": 2}


def test_openai_tools_shape():
    tools = openai_tools(_registry())
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "add"


def test_dispatcher_runs_allowed():
    reg = _registry()
    guard = SafeStep(reg)
    dispatcher = SafeDispatcher(reg, guard, {"add": lambda args: args["a"] + args["b"]})
    result = dispatcher.dispatch(ToolCall("add", {"a": 3, "b": 4}))
    assert not result.blocked
    assert result.result == 7


def test_dispatcher_blocks_unapproved():
    reg = _registry()
    guard = SafeStep(reg)
    called = []
    dispatcher = SafeDispatcher(reg, guard, {"danger": lambda args: called.append(args)})
    result = dispatcher.dispatch(ToolCall("danger", {"cmd": "rm -rf /"}))
    assert result.blocked
    assert called == []


def test_dispatcher_blocks_unknown_tool():
    reg = _registry()
    guard = SafeStep(reg)
    dispatcher = SafeDispatcher(reg, guard, {})
    result = dispatcher.dispatch(ToolCall("ghost", {}))
    assert result.blocked


if __name__ == "__main__":
    import sys

    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    sys.exit(1 if failed else 0)
