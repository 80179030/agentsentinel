"""Tests for the LangChain adapter. Run: python tests/test_langchain.py"""

from agentsentinel import ToolDefinition, ToolRegistry
from agentsentinel.adapters import (
    from_langchain_tool,
    langchain_tool_call,
    to_langchain_tools,
)


class _PydanticLike:
    def model_json_schema(self):
        return {
            "type": "object",
            "properties": {"q": {"type": "string"}},
            "required": ["q"],
        }


class _SchemaTool:
    name = "search"
    description = "search the web"
    args_schema = _PydanticLike()


class _ArgsDictTool:
    name = "echo"
    description = "echo back"
    args = {"text": {"type": "string"}}


def test_from_langchain_tool_with_args_schema():
    tool = from_langchain_tool(_SchemaTool())
    assert tool.name == "search"
    assert tool.parameters["properties"]["q"]["type"] == "string"


def test_from_langchain_tool_with_args_dict():
    tool = from_langchain_tool(_ArgsDictTool())
    assert tool.name == "echo"
    assert tool.parameters["properties"]["text"]["type"] == "string"


def test_langchain_tool_call_from_dict():
    call = langchain_tool_call({"name": "search", "args": {"q": "hi"}, "id": "call_1"})
    assert call.name == "search"
    assert call.arguments == {"q": "hi"}


def test_langchain_tool_call_from_object():
    class _TC:
        name = "search"
        args = {"q": "hi"}

    call = langchain_tool_call(_TC())
    assert call.name == "search"
    assert call.arguments == {"q": "hi"}


def test_to_langchain_tools_builds_or_import_errors():
    registry = ToolRegistry(
        [
            ToolDefinition(
                name="add",
                parameters={
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            )
        ]
    )
    try:
        tools = to_langchain_tools(registry, {"add": lambda a: a["a"] + a["b"]})
    except ImportError:
        return  # langchain-core not installed; skip
    assert len(tools) == 1
    assert tools[0].name == "add"


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
