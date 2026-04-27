"""Feature computation for Ldpj_backend.

v2.6 (current): 43-dimensional feature contract — pressure curve is
segmented into 6 sections by angle, 7 statistics per section, plus
cavity_id as the 43rd feature. See ``core.feature_spec`` for the canonical
ordering.

v2.5 (deprecated, retained for legacy callers and migration tests):
the 7-dim contract ``[max, min, difference, average, variance, trend_slope,
cavity_id]``.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from core.cycle_profile import CycleProfile, SECTION_NAMES
from core.curve_segmenter import segment_by_angle
from core.feature_spec import FEATURE_ORDER_43D, SECTION_SUB_FEATURES

logger = logging.getLogger(__name__)


# ── v2.6 (current) ─────────────────────────────────────────────────────

def _compute_section_stats(pressures: List[float]) -> Dict[str, float]:
    """Compute the 7 sub-features for one section.

    Returns all-zero stats with the actual ``count`` for sections too short
    for slope/variance estimates (n < 2). Empty sections give count=0.
    """
    n = len(pressures)
    if n < 2:
        return {
            "max": 0.0, "min": 0.0, "difference": 0.0,
            "average": 0.0, "variance": 0.0, "trend_slope": 0.0,
            "count": float(n),
        }

    arr = np.asarray(pressures, dtype=np.float64)
    p_max = float(np.max(arr))
    p_min = float(np.min(arr))
    p_diff = p_max - p_min
    p_avg = float(np.mean(arr))
    p_var = float(np.var(arr))

    try:
        slope = float(np.polyfit(np.arange(n), arr, 1)[0])
    except Exception:
        slope = 0.0

    return {
        "max": round(p_max, 3),
        "min": round(p_min, 3),
        "difference": round(p_diff, 3),
        "average": round(p_avg, 3),
        "variance": round(p_var, 3),
        "trend_slope": round(slope, 6),
        "count": float(n),
    }


def compute_features_v26(
    pressures: List[float],
    angles: List[float],
    cavity_id: int,
    profile: CycleProfile,
) -> Dict[str, float]:
    """Compute the 43-dim feature dict for one cycle (v2.6).

    Parameters
    ----------
    pressures : list of float
        Pressure samples (typically 70 points).
    angles : list of float
        Corresponding angles in degrees (same length as ``pressures``).
    cavity_id : int
        Cabin index (1..25).
    profile : CycleProfile
        Defines section boundaries for segmentation.

    Returns
    -------
    dict with 43 keys (see ``FEATURE_ORDER_43D``).
    """
    if len(pressures) < 2 or len(angles) < 2:
        feats = {name: 0.0 for name in FEATURE_ORDER_43D}
        feats["cavity_id"] = float(cavity_id)
        return feats

    sections = segment_by_angle(pressures, angles, profile)

    feats: Dict[str, float] = {}
    for section_name in SECTION_NAMES:
        stats = _compute_section_stats(sections.get(section_name, []))
        for sub_name, value in stats.items():
            feats[f"{section_name}_{sub_name}"] = value

    feats["cavity_id"] = float(cavity_id)
    return feats


def features_to_vector(feats: Dict[str, float], mode: str = "43d") -> List[float]:
    """Convert a feature dict into a fixed-order vector.

    v2.6 uses ``mode="43d"`` (default).
    The legacy v2.5 modes ``"7d"`` / ``"6d"`` are retained for callers that
    have not yet migrated (notably ``pipeline.processing_loop`` until Task 8
    rewires it). Once Task 8 ships, the legacy modes can be removed.
    """
    if mode == "43d":
        return [feats.get(k, 0.0) for k in FEATURE_ORDER_43D]
    if mode == "7d":
        order = ["max", "min", "difference", "average",
                 "variance", "trend_slope", "cavity_id"]
        return [feats.get(k, 0.0) for k in order]
    if mode == "6d":
        order = ["max", "min", "difference", "average", "variance", "trend_slope"]
        return [feats.get(k, 0.0) for k in order]
    raise ValueError(f"Unsupported feature mode: {mode}. Use '43d' (or legacy '7d'/'6d').")


# ── v2.5 (deprecated) ──────────────────────────────────────────────────
# Kept for ``pipeline.processing_loop`` until Task 8 rewires it to use
# compute_features_v26. New code MUST use ``compute_features_v26`` instead.

def compute_features(pressures: List[float], cavity_id: int) -> Dict[str, float]:
    """[DEPRECATED v2.5] Flat 7-dim feature computation.

    Use ``compute_features_v26(pressures, angles, cavity_id, profile)`` instead.
    """
    if not pressures or len(pressures) < 2:
        return {
            "max": 0.0, "min": 0.0, "difference": 0.0, "average": 0.0,
            "variance": 0.0, "trend_slope": 0.0, "cavity_id": float(cavity_id),
        }

    arr = np.asarray(pressures, dtype=np.float64)
    p_max = float(np.max(arr))
    p_min = float(np.min(arr))
    p_diff = p_max - p_min
    p_avg = float(np.mean(arr))
    p_var = float(np.var(arr))

    try:
        slope = float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    except Exception:
        slope = 0.0

    return {
        "max": round(p_max, 3), "min": round(p_min, 3),
        "difference": round(p_diff, 3), "average": round(p_avg, 3),
        "variance": round(p_var, 3), "trend_slope": round(slope, 6),
        "cavity_id": float(cavity_id),
    }
