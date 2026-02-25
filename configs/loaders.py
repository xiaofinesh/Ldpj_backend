"""YAML configuration loaders for Ldpj_backend."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

_BASE_DIR = Path(__file__).resolve().parent


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return its content as a dictionary.

    If *path* is relative it is resolved against the configs/ directory.
    Returns an empty dict on any error so the caller can always proceed
    with safe defaults.
    """
    p = Path(path)
    if not p.is_absolute():
        p = _BASE_DIR / p
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def load_plc_config() -> Dict[str, Any]:
    return load_yaml("plc.yaml")


def load_runtime_config() -> Dict[str, Any]:
    return load_yaml("runtime.yaml")


def load_models_config() -> Dict[str, Any]:
    return load_yaml("models.yaml")


def load_health_config() -> Dict[str, Any]:
    return load_yaml("health.yaml")


def load_ipc_config() -> Dict[str, Any]:
    return load_yaml("ipc.yaml")
