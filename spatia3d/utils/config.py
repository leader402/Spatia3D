"""YAML experiment-config loading, saving and deep-merging.

Experiments are config-driven (PLAN.md 4): a base YAML plus optional overrides, all returned as
plain dicts so they are trivial to log, hash, and diff.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

__all__ = ["load_config", "save_config", "merge_configs"]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict (empty file -> empty dict)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at top level, got {type(data).__name__}")
    return data


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Write a config dict to YAML, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, default_flow_style=False)


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins on conflicts)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
