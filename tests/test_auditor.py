"""Tests for the failure taxonomy and TrajectoryAuditor. Run: python tests/test_auditor.py"""

from agentsentinel import (
    Decision,
    FailureCategory,
    GapKind,
    Step,
    TrajectoryAuditor,
    Verdict,
)


def test_deny_maps_to_tool_interface():
    auditor = TrajectoryAuditor()
    step = Step(0, "ghost", decision=Decision(Verdict.DENY, reasons=["hallucinated tool"]))
    failure = auditor.first_failure([step])
    assert failure is not None
    assert failure.category is FailureCategory.TOOL_INTERFACE
    assert failure.index == 0


def test_abstain_specification_maps_to_state():
    auditor = TrajectoryAuditor()
    step = Step(
        0, "pay",
        decision=Decision(Verdict.ABSTAIN, reasons=["missing amount"], gaps=[GapKind.SPECIFICATION]),
    )
    failure = auditor.first_failure([step])
    assert failure.category is FailureCategory.STATE


def test_abstain_authority_maps_to_adversarial():
    auditor = TrajectoryAuditor()
    step = Step(
        0, "run_shell",
        decision=Decision(Verdict.ABSTAIN, reasons=["needs approval"], gaps=[GapKind.AUTHORITY]),
    )
    failure = auditor.first_failure([step])
    assert failure.category is FailureCategory.ADVERSARIAL


def test_error_maps_to_tool_interface():
    auditor = TrajectoryAuditor()
    step = Step(0, "run_shell", error="command not found")
    failure = auditor.first_failure([step])
    assert failure.category is FailureCategory.TOOL_INTERFACE
    assert failure.mode == "execution_error"


def test_timeout_maps_to_termination():
    auditor = TrajectoryAuditor()
    step = Step(0, "run_shell", error="timeout after 30s")
    failure = auditor.first_failure([step])
    assert failure.category is FailureCategory.TERMINATION


def test_observation_failure_maps():
    auditor = TrajectoryAuditor()
    step = Step(0, "search", observation="search failed with error 500")
    failure = auditor.first_failure([step])
    assert failure.category is FailureCategory.TOOL_INTERFACE


def test_first_failure_returns_earliest():
    auditor = TrajectoryAuditor()
    steps = [
        Step(0, "a", observation="ok"),
        Step(1, "b", error="boom"),
        Step(2, "c", error="also broken"),
    ]
    failure = auditor.first_failure(steps)
    assert failure is not None
    assert failure.index == 1


def test_repetition_detected_as_drift():
    auditor = TrajectoryAuditor()
    steps = [
        Step(0, "run_shell", arguments={"cmd": "rm -rf /x"}, error="boom"),
        Step(1, "run_shell", arguments={"cmd": "rm -rf /x"}, error="boom"),
    ]
    modes = [failure.mode for failure in auditor.audit(steps)]
    assert "repetition_without_recovery" in modes


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
