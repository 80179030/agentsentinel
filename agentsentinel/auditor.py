"""A deterministic, structured-signal locator for the first mistake in a run."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .taxonomy import FailureCategory
from .types import Decision, GapKind, Verdict


@dataclass
class Step:
    """One step of an agent trajectory."""

    index: int
    tool: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    decision: Optional[Decision] = None   # from SafeStep, if the step was guarded
    observation: Any = None               # result/observation returned by the environment
    error: Optional[str] = None           # exception text, if the step raised


@dataclass
class Failure:
    index: int
    category: FailureCategory
    mode: str
    reason: str


_ERROR_KEYWORDS = ("error", "failed", "invalid", "denied", "exception", "traceback")


class TrajectoryAuditor:
    """Classify each step's failure mode and locate the first mistake.

    This is a *deterministic, structured-signal* locator: it reasons over
    guardrail verdicts, execution errors, failed observations, and repeated
    calls. It does not claim to be a learned verifier, but it recovers the
    single most useful fact a human operator wants — "where did this run first
    go wrong?" — with no model and no training.
    """

    def audit(self, steps: list[Step]) -> list[Failure]:
        failures: list[Failure] = []
        prev_sig: Optional[str] = None
        prev_failed = False

        for step in steps:
            failure = self._classify_step(step)
            sig = self._signature(step)

            if failure is not None:
                failures.append(failure)

            if prev_failed and prev_sig == sig and failure is not None:
                failures.append(
                    Failure(
                        step.index,
                        FailureCategory.DRIFT,
                        "repetition_without_recovery",
                        f"retried the identical call '{step.tool}' after it already failed",
                    )
                )

            prev_sig = sig
            prev_failed = failure is not None

        return failures

    def first_failure(self, steps: list[Step]) -> Optional[Failure]:
        failures = self.audit(steps)
        return failures[0] if failures else None

    def _classify_step(self, step: Step) -> Optional[Failure]:
        decision = step.decision
        if decision is not None:
            if decision.verdict is Verdict.DENY:
                return Failure(
                    step.index,
                    FailureCategory.TOOL_INTERFACE,
                    "invalid_tool_or_arguments",
                    "; ".join(decision.reasons),
                )
            if decision.verdict is Verdict.ABSTAIN:
                if GapKind.AUTHORITY in decision.gaps:
                    return Failure(
                        step.index,
                        FailureCategory.ADVERSARIAL,
                        "unauthorized_action",
                        "; ".join(decision.reasons),
                    )
                if GapKind.SPECIFICATION in decision.gaps:
                    return Failure(
                        step.index,
                        FailureCategory.STATE,
                        "missing_information",
                        "; ".join(decision.reasons),
                    )
                if GapKind.VERIFICATION in decision.gaps:
                    return Failure(
                        step.index,
                        FailureCategory.STATE,
                        "unverified_precondition",
                        "; ".join(decision.reasons),
                    )

        if step.error:
            lowered = step.error.lower()
            if any(k in lowered for k in ("timeout", "budget", "rate limit")):
                return Failure(
                    step.index, FailureCategory.TERMINATION, "budget_exhaustion", step.error
                )
            return Failure(
                step.index, FailureCategory.TOOL_INTERFACE, "execution_error", step.error
            )

        obs = step.observation
        if isinstance(obs, str) and any(k in obs.lower() for k in _ERROR_KEYWORDS):
            return Failure(
                step.index, FailureCategory.TOOL_INTERFACE, "failed_outcome", obs
            )
        if isinstance(obs, dict) and obs.get("success") is False:
            return Failure(
                step.index,
                FailureCategory.TOOL_INTERFACE,
                "failed_outcome",
                str(obs.get("error", "task reported failure")),
            )

        return None

    @staticmethod
    def _signature(step: Step) -> str:
        args = json.dumps(dict(step.arguments), sort_keys=True, default=str)
        return f"{step.tool}:{args}"
