"""Walk through SafeStep on a handful of real-looking tool calls."""

from agentsentinel import (
    Policy,
    SafeStep,
    ToolCall,
    ToolDefinition,
    ToolRegistry,
    TraceLogger,
)


def main() -> None:
    registry = ToolRegistry(
        [
            ToolDefinition(
                name="search_web",
                description="Search the web for information.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "lang": {"type": "string", "enum": ["en", "zh"]},
                    },
                    "required": ["query"],
                },
            ),
            ToolDefinition(
                name="run_shell",
                description="Run a shell command.",
                parameters={
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
                requires_approval=True,
            ),
        ]
    )

    policy = Policy()
    guard = SafeStep(registry, policy)
    log = TraceLogger()

    calls = [
        ToolCall("search_web", {"query": "best agent papers 2026", "lang": "zh"}),
        ToolCall("search_web", {"lang": "en"}),
        ToolCall("search_web", {"query": "hi", "lang": "de"}),
        ToolCall("send_email", {}),
        ToolCall("run_shell", {"command": "ls -la"}),
        ToolCall("run_shell", {"command": "rm -rf /tmp/cache"}),
    ]

    for call in calls:
        decision = guard.check(call)
        log.record(call, decision)
        print(
            f"{call.name}{str(dict(call.arguments)):<56} -> "
            f"{decision.verdict.value:>7}  {decision.reasons[0]}"
        )

    print("\nGrant approval to run_shell and retry the safe command:")
    policy.approve("run_shell")
    decision = guard.check(ToolCall("run_shell", {"command": "ls -la"}))
    print(f"run_shell {{'command': 'ls -la'}} -> {decision.verdict.value}")

    print("\nAudit trail:")
    for row in log.as_dicts():
        print(f"  {row['tool']:<12} {row['verdict']:<8} gaps={row['gaps']}")


if __name__ == "__main__":
    main()
