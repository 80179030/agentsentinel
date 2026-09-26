# AgentSentinel

Zero-training, drop-in safety guardrails and observability for LLM agents.
Block hallucinated tool calls, pause on missing info or missing approval, and
keep an audit trail of every decision — with no model training, no GPU, and no
framework lock-in.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Zero deps](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](pyproject.toml)

## Why this exists

As soon as an LLM is allowed to *call tools*, it starts making a specific class
of mistakes that capability benchmarks never catch:

- **Tool hallucination** — calling a tool that does not exist, or inventing a
  parameter the tool never had.
- **Compliance bias** — proceeding to act even when key information or explicit
  permission is missing, because "just try it" is the rewarded default.
- **Unchecked destructive actions** — the model writes `rm -rf` into a shell
  call and nothing upstream notices.

Recent research keeps landing on the same gap: the leading agent benchmarks
spend almost nothing on safety, and the reliable fix is *not* a bigger model —
it is a **closed-world check before the call goes out**. AgentSentinel turns
that idea into a small, composable library you can slot in front of any agent.

## What it does

`SafeStep` validates a proposed tool call and returns one of three verdicts:

| Verdict  | Meaning                                                        | Example                                |
|----------|----------------------------------------------------------------|----------------------------------------|
| `allow`  | Call conforms to the registered schema.                        | `search_web({"query": "..."})`         |
| `deny`   | Call is malformed or hallucinated — block it.                  | `send_email({})` when not registered   |
| `abstain`| Well-formed, but should pause and ask first (don't guess).     | missing required arg, needs approval   |

`abstain` carries a `gap` that says *why* it paused, using the three-gap
taxonomy: `specification` (missing info), `verification` (can't confirm state),
or `authority` (no permission granted). That structured reason is what lets a
downstream layer turn a silent failure into an informed question.

## Quickstart

```bash
pip install agentsentinel   # not yet on PyPI — clone and `pip install -e .` for now
```

```python
from agentsentinel import (
    Policy, SafeStep, ToolCall, ToolDefinition, ToolRegistry, TraceLogger,
)

registry = ToolRegistry([
    ToolDefinition(
        name="search_web",
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
        parameters={"type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"]},
        requires_approval=True,
    ),
])

guard = SafeStep(registry)
log = TraceLogger()

for call in [
    ToolCall("search_web", {"query": "agent papers", "lang": "zh"}),  # allow
    ToolCall("search_web", {"lang": "en"}),                           # abstain (missing query)
    ToolCall("search_web", {"query": "hi", "lang": "de"}),            # deny (bad enum)
    ToolCall("send_email", {}),                                       # deny (unknown tool)
    ToolCall("run_shell", {"command": "rm -rf /tmp/cache"}),          # abstain (sensitive)
]:
    decision = guard.check(call)
    log.record(call, decision)
    print(call.name, "->", decision.verdict.value, "|", decision.reasons[0])
```

Dangerous patterns (`rm -rf`, `DROP TABLE`, `DELETE FROM`, …) are **on by
default**. Grant approval for a tool with `guard.policy.approve("run_shell")`.

## How it works

```
LLM proposes ToolCall
        │
        ▼
┌─────────────────────────────┐
│ 1. tool in registry?        │──no──▶ deny (tool hallucination)
└─────────────────────────────┘
        │ yes
        ▼
┌─────────────────────────────┐
│ 2. required args present?   │──no──▶ abstain (specification gap)
└─────────────────────────────┘
        │ yes
        ▼
┌─────────────────────────────┐
│ 3. args match schema?       │──no──▶ deny (unknown param / bad type)
└─────────────────────────────┘
        │ yes
        ▼
┌─────────────────────────────┐
│ 4. sensitive / needs approval?│──yes─▶ abstain (authority gap)
└─────────────────────────────┘
        │ no
        ▼
      allow
```

The validator is deliberately small: `type`, `required`, `enum`, and unknown-key
detection. That covers the failure modes that actually bite in production while
staying predictable and auditable.

## OpenAI adapter

`agentsentinel.adapters` bridges OpenAI-style function calling to the guard:

```python
from agentsentinel import SafeStep, ToolCall, ToolRegistry
from agentsentinel.adapters import SafeDispatcher, openai_tools, to_tool_call

registry = ToolRegistry([...])
guard = SafeStep(registry)

# 1. send these to the model so it only ever sees known tools
tools = openai_tools(registry)

# 2. after the model replies, convert its raw tool call and dispatch safely
raw = {"name": "run_shell", "arguments": '{"command": "ls"}'}   # or an SDK object
dispatcher = SafeDispatcher(registry, guard, functions={"run_shell": my_shell_fn})

result = dispatcher.dispatch(to_tool_call(raw))
if result.blocked:
    print(result.decision.reasons)   # tell the model why it was paused
else:
    print(result.result)
```

`SafeDispatcher` runs the registered function only when the guard allows the
call — blocked or abstained calls never touch the real function.

## LangChain adapter

The same guard works with LangChain tools — with **no hard dependency** on
LangChain. `from_langchain_tool` and `langchain_tool_call` read a tool's
`name` / `description` / `args_schema` (or `args`) by duck typing, so they run
without installing anything:

```python
from agentsentinel.adapters import from_langchain_tool, langchain_tool_call

tool_def = from_langchain_tool(my_langchain_tool)      # -> ToolDefinition
call = langchain_tool_call(ai_message.tool_calls[0])   # -> ToolCall (handles "args")
```

`to_langchain_tools(registry, functions)` builds LangChain `Tool` objects when
`langchain-core` is installed (it raises a clear hint if not).

## Trajectory auditor

`TrajectoryAuditor` answers the question operators actually ask: *where did
this run first go wrong?* It classifies each step into the six production
failure families — `drift`, `state`, `coordination`, `termination`,
`adversarial`, `tool_interface` — using only structured signals (guardrail
verdicts, execution errors, failed observations, repeated calls), no model and
no training:

```python
from agentsentinel import Decision, Step, TrajectoryAuditor, Verdict

steps = [
    Step(0, "search", observation="ok"),
    Step(1, "run_shell", arguments={"command": "rm -rf /x"},
         decision=Decision(Verdict.ABSTAIN, gaps=["authority"])),
]

first = TrajectoryAuditor().first_failure(steps)
print(first.category.value, "->", first.mode)   # adversarial -> unauthorized_action
```

This is the deterministic cousin of the learned "failure localizer" in the
research — cheap to run on every trajectory today, and a clean foundation to
swap in a trained verifier later.

## Policy as config

Declare tools and rules in a YAML or JSON file instead of code. YAML needs
PyYAML; JSON works with no extra dependency.

```yaml
# policy.yaml
tools:
  - name: search_web
    parameters:
      type: object
      properties:
        query: {type: string}
      required: [query]
  - name: run_shell
    parameters:
      type: object
      properties: {command: {type: string}}
      required: [command]
    requires_approval: true

policy:
  denylist: [admin_panel]
  sensitive_patterns: ["DROP\\s+TABLE"]
```

```python
from agentsentinel import load_config_file

cfg = load_config_file("policy.yaml")
cfg.guard.check(...)          # ready to use
cfg.registry, cfg.policy      # or reach into the parts
```

Config builds on the safe defaults: dangerous patterns stay on, and your
`denylist` / `approved_tools` / `sensitive_patterns` are merged on top.

## Project layout

```
agentsentinel/
├── agentsentinel/
│   ├── types.py          # Verdict, GapKind, ToolCall, ToolDefinition, Decision
│   ├── registry.py       # closed-world tool registry
│   ├── policy.py         # denylist, approvals, sensitive patterns
│   ├── guard.py          # SafeStep — the check pipeline
│   ├── config.py         # load_config / load_config_file (YAML or JSON)
│   ├── taxonomy.py       # six failure families + descriptions
│   ├── auditor.py        # TrajectoryAuditor — first-mistake locator
│   ├── observability.py  # TraceLogger — audit trail
│   └── adapters/         # openai.py, langchain.py
├── examples/             # quickstart.py, policy.yaml
└── tests/                # 38 tests, stdlib only
```

## Roadmap

- [x] `SafeStep` closed-world validation + abstention (v0.1)
- [x] OpenAI adapter — parse tool calls, build `tools` array, guarded dispatch
- [x] LangChain adapter — duck-typed tool/call conversion, optional `langchain-core`
- [x] Failure taxonomy + deterministic first-mistake locator (`TrajectoryAuditor`)
- [x] Policy-as-config (YAML / JSON)
- [ ] Per-user approval flows (interactive pause-and-ask)
- [ ] Learned trajectory verifier (trained to locate the first mistake)

## License

[MIT](LICENSE)
