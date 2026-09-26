"""AgentSentinel: zero-training safety guardrails and observability for LLM agents."""

from .auditor import Failure, Step, TrajectoryAuditor
from .config import GuardConfig, load_config, load_config_file
from .guard import SafeStep
from .observability import TraceEntry, TraceLogger
from .policy import Policy, default_policy
from .registry import ToolRegistry
from .taxonomy import FailureCategory, describe
from .types import Decision, GapKind, ToolCall, ToolDefinition, Verdict

__version__ = "0.1.0"

__all__ = [
    "SafeStep",
    "Policy",
    "default_policy",
    "ToolRegistry",
    "ToolCall",
    "ToolDefinition",
    "Decision",
    "GapKind",
    "Verdict",
    "TraceEntry",
    "TraceLogger",
    "FailureCategory",
    "describe",
    "Step",
    "Failure",
    "TrajectoryAuditor",
    "GuardConfig",
    "load_config",
    "load_config_file",
]
