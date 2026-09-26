"""Tests for SafeStep. Run with pytest or directly: python tests/test_guard.py"""

from agentsentinel import (
    GapKind,
    SafeStep,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    Verdict,
)


def _build() -> SafeStep:
    registry = ToolRegistry(
        [
            ToolDefinition(
                name="search",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                        "lang": {"type": "string", "enum": ["en", "zh"]},
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="pay",
                parameters={
                    "type": "object",
                    "properties": {"amount": {"type": "number"}},
                    "required": ["amount"],
                },
                requires_approval=True,
            ),
            ToolDefinition(
                name="run_shell",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                requires_approval=True,
            ),
        ]
    )
    return SafeStep(registry)


def test_allow_valid_call():
    guard = _build()
    decision = guard.check(ToolCall("search", {"query": "hello", "limit": 3, "lang": "en"}))
    assert decision.verdict is Verdict.ALLOW
    assert decision.allowed


def test_deny_unknown_tool():
    guard = _build()
    decision = guard.check(ToolCall("send_email", {}))
    assert decision.verdict is Verdict.DENY


def test_deny_unknown_parameter():
    guard = _build()
    decision = guard.check(ToolCall("search", {"query": "hi", "q": "typo"}))
    assert decision.verdict is Verdict.DENY


def test_deny_bad_enum():
    guard = _build()
    decision = guard.check(ToolCall("search", {"query": "hi", "lang": "de"}))
    assert decision.verdict is Verdict.DENY


def test_deny_bad_type():
    guard = _build()
    decision = guard.check(ToolCall("search", {"query": "hi", "limit": "three"}))
    assert decision.verdict is Verdict.DENY


def test_abstain_missing_required():
    guard = _build()
    decision = guard.check(ToolCall("search", {"limit": 3}))
    assert decision.verdict is Verdict.ABSTAIN
    assert GapKind.SPECIFICATION in decision.gaps


def test_abstain_requires_approval():
    guard = _build()
    decision = guard.check(ToolCall("pay", {"amount": 100}))
    assert decision.verdict is Verdict.ABSTAIN
    assert GapKind.AUTHORITY in decision.gaps


def test_allow_after_approval():
    guard = _build()
    guard.policy.approve("pay")
    decision = guard.check(ToolCall("pay", {"amount": 100}))
    assert decision.verdict is Verdict.ALLOW


def test_denylist():
    guard = _build()
    guard.policy.deny("search")
    decision = guard.check(ToolCall("search", {"query": "hi"}))
    assert decision.verdict is Verdict.DENY


def test_sensitive_pattern_abstains():
    guard = _build()
    decision = guard.check(ToolCall("run_shell", {"command": "rm -rf /var/data"}))
    assert decision.verdict is Verdict.ABSTAIN
    assert any("sensitive" in reason for reason in decision.reasons)


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
