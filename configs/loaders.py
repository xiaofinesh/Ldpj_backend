"""YAML configuration loaders for Ldpj_backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from core.cycle_profile import (
    CycleProfile,
    load_active_cycle_profile as _load_active_cycle_profile_from_dict,
)

_BASE_DIR = Path(__file__).resolve().parent


def load_yaml(path: str | Path) -> Dict[str, Any]:
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


def load_active_cycle_profile() -> CycleProfile:
    """Load runtime.yaml and extract the active CycleProfile (v2.6).

    Convenience wrapper that reads runtime.yaml then dispatches to
    core.cycle_profile.load_active_cycle_profile().
    """
    return _load_active_cycle_profile_from_dict(load_runtime_config())


# ── Cabin V_cabin calibration (v2.6) ──────────────────────────────────

def load_cabins_config(path: str | Path = "cabins.yaml") -> Dict[str, Any]:
    """Load V_cabin calibration values for all cabins.

    Path is resolved relative to the configs/ directory if not absolute.

    Returns
    -------
    dict with keys: calibration_date, calibrator, cabins, default

    Raises
    ------
    FileNotFoundError if the file is missing.
    """
    p = Path(path)
    if not p.is_absolute():
        p = _BASE_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"cabins config not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_v_cabin(cabins_cfg: Dict[str, Any], cabin_id: int) -> Tuple[float, float]:
    """Look up (v_cabin, u_v_cabin) in m³ for a specific cabin.

    Falls back to the ``default`` block when the cabin has no calibrated
    entry. Returns numeric defaults (3.5e-4, 1e-5) if even ``default`` is
    missing, so that downstream physics never sees None.
    """
    entry = cabins_cfg.get("cabins", {}).get(cabin_id)
    if entry and "v_cabin" in entry:
        return float(entry["v_cabin"]), float(entry.get("u_v_cabin", 0.0))
    default = cabins_cfg.get("default", {}) or {}
    return float(default.get("v_cabin", 3.5e-4)), float(default.get("u_v_cabin", 1.0e-5))


def is_cabin_calibrated(cabins_cfg: Dict[str, Any], cabin_id: int) -> bool:
    """Check whether ``cabin_id`` has been measured (vs using a placeholder).

    Heuristic: an entry whose ``notes`` contains '占位' is treated as
    uncalibrated, even if a numeric ``v_cabin`` is present.
    """
    entry = cabins_cfg.get("cabins", {}).get(cabin_id, {})
    if not entry or entry.get("v_cabin") is None:
        return False
    notes = entry.get("notes", "") or ""
    return "占位" not in notes


# ── Product configuration (v2.6) ──────────────────────────────────────

def load_products_config(path: str | Path = "products.yaml") -> Dict[str, Any]:
    """Load product configuration (Q_threshold per product).

    Path is resolved relative to the configs/ directory if not absolute.

    Returns
    -------
    dict with keys: default_product_id, products

    Raises
    ------
    FileNotFoundError if the file is missing.
    """
    p = Path(path)
    if not p.is_absolute():
        p = _BASE_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"products config not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_product(products_cfg: Dict[str, Any], product_id: str) -> Dict[str, Any]:
    """Look up a product by id, falling back to the configured default.

    Returns an empty dict if neither the requested product nor the default
    can be resolved (so callers always get a Mapping back).
    """
    products = products_cfg.get("products", {}) or {}
    if product_id in products:
        return products[product_id]
    default_id = products_cfg.get("default_product_id", "")
    if default_id and default_id in products:
        return products[default_id]
    return {}
