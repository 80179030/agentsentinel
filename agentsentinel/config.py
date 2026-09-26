"""Load guardrail configuration from YAML or JSON."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Union

from .guard import SafeStep
from .policy import Policy, default_policy
from .registry import ToolRegistry
from .types import ToolDefinition


@dataclass
class GuardConfig:
    """The three objects built from a config file, ready to use."""

    registry: ToolRegistry
    policy: Policy
    guard: SafeStep


def load_config(data: Mapping[str, Any]) -> GuardConfig:
    """Build a GuardConfig from an already-parsed dict (YAML/JSON structure)."""
    registry = ToolRegistry()
    for spec in data.get("tools", []):
        name = str(spec.get("name", "")).strip()
        if not name:
            raise ValueError(f"tool entry is missing a 'name': {spec!r}")
        registry.register(
            ToolDefinition(
                name=name,
                description=str(spec.get("description", "")),
                parameters=dict(spec.get("parameters", {})),
                requires_approval=bool(spec.get("requires_approval", False)),
            )
        )

    # Start from the safe default (dangerous patterns on), then merge user rules.
    policy = default_policy()
    policy_cfg = data.get("policy", {})
    policy.denylist.update(str(t) for t in policy_cfg.get("denylist", []))
    policy.approved_tools.update(str(t) for t in policy_cfg.get("approved_tools", []))
    for pattern in policy_cfg.get("sensitive_patterns", []):
        policy.sensitive_patterns.append(re.compile(pattern))

    return GuardConfig(registry=registry, policy=policy, guard=SafeStep(registry, policy))


def load_config_file(path: Union[str, Path]) -> GuardConfig:
    """Load a config from a ``.yaml``/``.yml`` or ``.json`` file.

    YAML requires PyYAML; JSON needs no extra dependency.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "PyYAML is required to load .yaml configs; install with "
                "`pip install pyyaml` (or use a .json config, which needs no "
                "extra dependency)"
            ) from exc
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported config format: {suffix!r} (use .yaml or .json)")

    return load_config(data)
