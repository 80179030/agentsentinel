<p align="center">
  <h1 align="center">AgentSentinel</h1>
  <p align="center"><strong>Stop LLM agents from calling the wrong thing — before it leaves your process.</strong></p>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python 3.9+" src="https://img.shields.io/badge/Python-3.9%2B-blue"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="#"><img alt="Zero dependencies" src="https://img.shields.io/badge/dependencies-none-brightgreen.svg"></a>
  <a href="#"><img alt="Pure stdlib" src="https://img.shields.io/badge/pure-stdlib-blueviolet.svg"></a>
</p>

**AgentSentinel** is a zero-training, zero-dependency guardrail that validates an
agent's tool calls *before* they touch the outside world. One check in front of
any agent — no model, no GPU, no framework lock-in.

---

## The problem

The moment an LLM is allowed to call tools, a new class of failure appears that
capability benchmarks never measure:

- **Tool hallucination** — the model invents a tool that doesn't exist, or a
  parameter the tool never had.
- **Compliance bias** — it acts even when a required input or explicit
  permission is missing, because "just try it" is the rewarded default.
- **Unchecked destructive actions** — `rm -rf`, `DROP TABLE`, an unapproved
  payment — and nothing upstream stops it.

The fix is not a bigger model. It's a **closed-world check before the call
leaves your process**. AgentSentinel is that check, packaged as a small library
you can read top to bottom in ten minutes.

## What you get

- **`SafeStep`** — validates every tool call against a registry and returns one
  of three verdicts: `allow`, `deny`, or `abstain` (pause and ask, don't guess).
- **Structured abstention** — every pause carries a `gap` that says *why*:
  `specification` (missing info), `verification` (unconfirmed state), or
  `authority` (no permission).
- **`SafeDispatcher`** — runs the real function only after the guard approves.
- **Adapters** — OpenAI-style function calling and LangChain tools, with no hard
  dependency on either.
- **`TrajectoryAuditor`** — locates the *first* mistake in a run across six
  failure families, using structured signals and no model.
- **Policy-as-config** — declare tools and rules in YAML or JSON.

## Demo

```python
from agentsentinel import Policy, SafeStep, ToolCall, ToolDefinition, ToolRegistry

registry = ToolRegistry([
    ToolDefinition("search_web", parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "lang": {"type": "string", "enum": ["en", "zh"]},
        },
        "required": ["query"],
    }),
    ToolDefinition("run_shell", parameters={
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    }, requires_approval=True),
])

guard = SafeStep(registry)

for call in [
    ToolCall("search_web", {"query": "best agent papers", "lang": "zh"}),
    ToolCall("search_web", {"lang": "en"}),
    ToolCall("search_web", {"query": "hi", "lang": "de"}),
    ToolCall("send_email", {}),
    ToolCall("run_shell", {"command": "rm -rf /tmp/cache"}),
]:
    d = guard.check(call)
    print(f"{call.name:<12} {d.verdict.value:<8} {d.reasons[0]}")
```

```
search_web   allow    arguments conform to schema
search_web   abstain  'search_web' is missing required arguments: query
search_web   deny     'search_web' arguments do not match schema: 'lang' must be one of ['en', 'zh']
send_email   deny     tool 'send_email' is not registered (hallucinated tool or typo)
run_shell    abstain  'run_shell' matched sensitive pattern(s): rm\s+-rf
```

The destructive command was stopped **before it ran** — and the reason why is
structured, so your agent (or your user) can ask the right follow-up instead of
silently guessing.

## Quickstart

```bash
pip install agentsentinel          # or: git clone … && pip install -e .
```

```python
from agentsentinel import SafeStep, ToolCall, ToolDefinition, ToolRegistry

guard = SafeStep(ToolRegistry([
    ToolDefinition("search", parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }),
]))

decision = guard.check(ToolCall("search", {"query": "hello"}))
print(decision.verdict.value)   # allow
```

Dangerous patterns (`rm -rf`, `DROP TABLE`, `DELETE FROM`, …) are **on by
default**. Grant approval for a specific tool with
`guard.policy.approve("run_shell")`.

## How it works

```
LLM proposes ToolCall
        │
        ▼
┌─────────────────────────────┐
│ 1. tool in registry?        │──no──▶ deny   (tool hallucination)
└─────────────────────────────┘
        │ yes
        ▼
┌─────────────────────────────┐
│ 2. required args present?   │──no──▶ abstain (specification gap)
└─────────────────────────────┘
        │ yes
        ▼
┌─────────────────────────────┐
│ 3. args match schema?       │──no──▶ deny   (unknown param / bad type)
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

| Verdict   | Meaning                                                    | Example                              |
|-----------|------------------------------------------------------------|--------------------------------------|
| `allow`   | Call conforms to the registered schema.                    | `search({"query": "..."})`           |
| `deny`    | Call is malformed or hallucinated — block it.              | unregistered tool, unknown parameter |
| `abstain` | Well-formed, but pause and ask first — don't guess.        | missing arg, sensitive action        |

The validator is deliberately small — `type`, `required`, `enum`, and
unknown-key detection — which covers the failure modes that actually bite in
production while staying predictable and auditable.

## Integrations

**OpenAI / function calling** — parse raw tool calls and dispatch safely:

```python
from agentsentinel import SafeStep, ToolRegistry
from agentsentinel.adapters import SafeDispatcher, openai_tools, to_tool_call

registry = ToolRegistry([...])
guard = SafeStep(registry)

tools = openai_tools(registry)        # send to the model — it only sees known tools
result = SafeDispatcher(registry, guard, functions={"run_shell": my_shell_fn}) \
    .dispatch(to_tool_call(raw_tool_call))

if result.blocked:
    print(result.decision.reasons)    # tell the model why it was paused
```

**LangChain** — duck-typed conversion with no hard dependency:

```python
from agentsentinel.adapters import from_langchain_tool, langchain_tool_call

tool = from_langchain_tool(my_langchain_tool)          # -> ToolDefinition
call = langchain_tool_call(ai_message.tool_calls[0])   # -> ToolCall (handles "args")
```

## Failure localization

`TrajectoryAuditor` answers the question operators actually ask: *where did this
run first go wrong?* It classifies each step into six failure families — `drift`,
`state`, `coordination`, `termination`, `adversarial`, `tool_interface` — using
only structured signals (guardrail verdicts, execution errors, failed
observations, repeated calls), no model and no training:

```python
from agentsentinel import Decision, GapKind, Step, TrajectoryAuditor, Verdict

steps = [
    Step(0, "search", observation="ok"),
    Step(1, "run_shell", arguments={"command": "rm -rf /x"},
         decision=Decision(Verdict.ABSTAIN, gaps=[GapKind.AUTHORITY])),
]

first = TrajectoryAuditor().first_failure(steps)
print(first.category.value, "->", first.mode)   # adversarial -> unauthorized_action
```

Recent research shows frontier LLM judges struggle to locate the *first* mistake
in a long agent run; structured signals recover it cheaply and deterministically
today — and leave a clean interface to swap in a trained verifier later.

## Policy as config

Declare tools and rules in YAML or JSON instead of code. YAML needs PyYAML;
JSON works with no extra dependency.

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
```

Config builds on the safe defaults: dangerous patterns stay on, and your
`denylist` / `approved_tools` / `sensitive_patterns` are merged on top.

## Why AgentSentinel

- **Not a prompt.** Prompts can be talked out of. A registry check cannot.
- **Not a heavyweight framework.** No server, no model in the loop, no YAML
  sprawl. The core is a few hundred lines of stdlib you can audit.
- **Closed-world by default.** Anything not registered doesn't run.
- **Deterministic and observable.** Every decision is reproducible and logged,
  so a "no" always has a reason.
- **Sits in front of any agent.** Framework-agnostic by design.

## Project layout

```
agentsentinel/
├── agentsentinel/
│   ├── types.py          # Verdict, GapKind, ToolCall, ToolDefinition, Decision
│   ├── registry.py       # closed-world tool registry
│   ├── policy.py         # denylist, approvals, sensitive patterns
│   ├── guard.py          # SafeStep — the check pipeline
│   ├── config.py         # load_config / load_config_file (YAML or JSON)
│   ├── taxonomy.py       # six failure families
│   ├── auditor.py        # TrajectoryAuditor — first-mistake locator
│   ├── observability.py  # TraceLogger — audit trail
│   └── adapters/         # openai.py, langchain.py
├── examples/             # quickstart.py, policy.yaml
└── tests/                # 38 tests, stdlib only
```

## Roadmap

- [x] `SafeStep` closed-world validation + abstention
- [x] OpenAI and LangChain adapters
- [x] Failure taxonomy + deterministic first-mistake locator
- [x] Policy-as-config (YAML / JSON)
- [ ] Per-user approval flows (interactive pause-and-ask)
- [ ] Learned trajectory verifier (trained to locate the first mistake)

## Contributing

Issues and PRs welcome. The library is intentionally small and dependency-free —
keep it that way unless there's a strong reason.

## License

[MIT](LICENSE)
