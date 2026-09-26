"""SafeStep: a closed-world tool-call guardrail with abstention."""

from __future__ import annotations

import json
from typing import Any, Optional

from .policy import Policy, default_policy
from .registry import ToolRegistry
from .types import Decision, GapKind, ToolCall, ToolDefinition, Verdict


class SafeStep:
    """Validate an agent's tool calls before they touch the outside world.

    The check is *closed-world*: every tool name and every argument must be
    known up front. Three outcomes map onto the "three-gap" safety taxonomy:

    * ``ALLOW``   - the call conforms to the registered schema.
    * ``DENY``    - the call is malformed or hallucinated (unknown tool, unknown
                    parameter, wrong type). Blocking is the right answer.
    * ``ABSTAIN`` - the call is well-formed but should not run yet: a required
                    argument is missing (specification gap) or an approval /
                    sensitive action is unconfirmed (authority gap). The right
                    answer is to pause and ask, not to guess.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        policy: Optional[Policy] = None,
    ) -> None:
        self.registry = registry
        self.policy = policy if policy is not None else default_policy()

    def check(self, call: ToolCall) -> Decision:
        tool = self.registry.get(call.name)

        if tool is None:
            return Decision(
                verdict=Verdict.DENY,
                reasons=[f"tool '{call.name}' is not registered (hallucinated tool or typo)"],
            )

        if call.name in self.policy.denylist:
            return Decision(
                verdict=Verdict.DENY,
                reasons=[f"tool '{call.name}' is denylisted by policy"],
            )

        missing = self._missing_required(tool, call.arguments)
        if missing:
            return Decision(
                verdict=Verdict.ABSTAIN,
                reasons=[f"'{call.name}' is missing required arguments: " + ", ".join(missing)],
                gaps=[GapKind.SPECIFICATION],
            )

        arg_errors = self._validate_arguments(tool, call.arguments)
        if arg_errors:
            return Decision(
                verdict=Verdict.DENY,
                reasons=[f"'{call.name}' arguments do not match schema: " + "; ".join(arg_errors)],
            )

        sensitive = self._detect_sensitive(call.arguments)
        if sensitive and not self.policy.approved(call.name):
            return Decision(
                verdict=Verdict.ABSTAIN,
                reasons=[
                    f"'{call.name}' matched sensitive pattern(s): " + ", ".join(sensitive)
                ],
                gaps=[GapKind.AUTHORITY],
            )

        if tool.requires_approval and not self.policy.approved(call.name):
            return Decision(
                verdict=Verdict.ABSTAIN,
                reasons=[f"'{call.name}' requires explicit approval before execution"],
                gaps=[GapKind.AUTHORITY],
            )

        return Decision(verdict=Verdict.ALLOW, reasons=["arguments conform to schema"])

    @staticmethod
    def _missing_required(tool: ToolDefinition, args: Mapping[str, Any]) -> list[str]:
        required = (tool.parameters or {}).get("required", [])
        return [r for r in required if r not in args]

    def _validate_arguments(
        self, tool: ToolDefinition, args: Mapping[str, Any]
    ) -> list[str]:
        errors: list[str] = []
        schema = tool.parameters or {}
        properties: dict[str, Any] = schema.get("properties", {})

        allowed = set(properties)
        if allowed:
            extra = sorted(k for k in args if k not in allowed)
            if extra:
                errors.append("unknown parameter(s): " + ", ".join(extra))

        for key, value in args.items():
            prop = properties.get(key)
            if prop is None:
                continue
            type_error = self._check_type(value, prop.get("type"))
            if type_error:
                errors.append(f"'{key}': {type_error}")
            elif "enum" in prop and value not in prop["enum"]:
                errors.append(f"'{key}' must be one of {prop['enum']}")

        return errors

    @staticmethod
    def _check_type(value: Any, expected: Optional[str]) -> Optional[str]:
        if expected is None:
            return None
        mapping = {
            "string": isinstance(value, str),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "array": isinstance(value, list),
            "object": isinstance(value, dict),
        }
        ok = mapping.get(expected)
        if ok is None:
            return None
        if not ok:
            return f"expected type '{expected}', got '{type(value).__name__}'"
        return None

    def _detect_sensitive(self, args: Mapping[str, Any]) -> list[str]:
        blob = json.dumps(args, ensure_ascii=False, default=str)
        hits: list[str] = []
        for pattern in self.policy.sensitive_patterns:
            if pattern.search(blob):
                hits.append(pattern.pattern)
        return hits
