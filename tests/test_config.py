"""Tests for config loading. Run: python tests/test_config.py"""

import json
import os
import tempfile

from agentsentinel import (
    SafeStep,
    ToolCall,
    Verdict,
    load_config,
    load_config_file,
)

_CONFIG = {
    "tools": [
        {
            "name": "search",
            "description": "search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "run_shell",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
            "requires_approval": True,
        },
    ],
    "policy": {"denylist": ["admin_panel"], "sensitive_patterns": ["DROP\\s+TABLE"]},
}


def test_load_config_builds_registry_and_guard():
    cfg = load_config(_CONFIG)
    assert "search" in cfg.registry
    assert isinstance(cfg.guard, SafeStep)
    decision = cfg.guard.check(ToolCall("search", {"query": "hi"}))
    assert decision.verdict is Verdict.ALLOW


def test_load_config_requires_approval_flag():
    cfg = load_config(_CONFIG)
    decision = cfg.guard.check(ToolCall("run_shell", {"command": "ls"}))
    assert decision.verdict is Verdict.ABSTAIN


def test_load_config_denylist():
    cfg = load_config(_CONFIG)
    decision = cfg.guard.check(ToolCall("admin_panel", {}))
    assert decision.verdict is Verdict.DENY


def test_load_config_extra_sensitive_pattern():
    cfg = load_config(_CONFIG)
    decision = cfg.guard.check(ToolCall("run_shell", {"command": "DROP TABLE users"}))
    assert decision.verdict is Verdict.ABSTAIN
    assert any("sensitive" in r for r in decision.reasons)


def test_load_json_file():
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(_CONFIG, f)
        path = f.name
    try:
        cfg = load_config_file(path)
        assert "search" in cfg.registry
    finally:
        os.unlink(path)


def test_load_yaml_file_or_skip():
    content = (
        "tools:\n"
        "  - name: search\n"
        "    parameters:\n"
        "      type: object\n"
        "      properties:\n"
        "        query: {type: string}\n"
        "      required: [query]\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(content)
        path = f.name
    try:
        try:
            cfg = load_config_file(path)
        except ImportError:
            return  # PyYAML not installed; skip
        assert "search" in cfg.registry
    finally:
        os.unlink(path)


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
