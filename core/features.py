"""Feature computation for Ldpj_backend.

v2.6 (current): 43-dimensional feature contract — pressure curve is
segmented into 6 sections by angle, 7 statistics per section, plus
cavity_id as the 43rd feature. See ``core.feature_spec`` for the canonical
ordering.

v2.5 (deprecated, retained for legacy callers and migration tests):
the 7-dim contract ``[max, min, difference, average, variance, trend_slope,
cavity_id]``.

Performance notes
-----------------
The hot path (``compute_features_v26``) is called once per cabin per
completed cycle (~25 cabins × 1 cycle/7s = ~3.5 calls/sec at steady state,
but bursts can be 25 calls in the same poll tick). It must finish in
well under 10 ms across all cabins to leave headroom for the rest of the
loop.

Optimizations applied vs the literal v2.6 formulation:
- Single ``np.asarray`` per cycle (the whole ``pressures`` list), not
  per section. Sections are taken as views via boolean masks.
- Closed-form slope ``(n·∑xy − ∑x·∑y) / (n·∑x² − (∑x)²)`` instead of
  ``np.polyfit`` (≈ 7× faster for n=70).
- Single-pass max/min via ``np.minmax``-equivalent reductions; one
  ``arr.sum()`` reused for mean and slope.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Sequence

import numpy as np

from core.cycle_profile import CycleProfile, SECTION_NAMES
from core.curve_segmenter import segment_by_angle
from core.feature_spec import FEATURE_ORDER_43D

logger = logging.getLogger(__name__)


# Pre-built tuple so list comprehensions in features_to_vector skip the
# attribute lookup on every call.
_FEATURE_ORDER_43D_T: tuple = tuple(FEATURE_ORDER_43D)
_LEGACY_7D_ORDER: tuple = (
    "max", "min", "difference", "average",
    "variance", "trend_slope", "cavity_id",
)
_LEGACY_6D_ORDER: tuple = _LEGACY_7D_ORDER[:-1]


# ── v2.6 (current) ─────────────────────────────────────────────────────

def _zero_section_stats(n: int) -> Dict[str, float]:
    return {
        "max": 0.0, "min": 0.0, "difference": 0.0,
        "average": 0.0, "variance": 0.0, "trend_slope": 0.0,
        "count": float(n),
    }


def _compute_section_stats_arr(arr: np.ndarray) -> Dict[str, float]:
    """Compute the 7 sub-features for one section's ndarray slice.

    For n<2 returns zeros + actual count. The closed-form slope avoids
    np.polyfit (which builds a Vandermonde matrix and runs lstsq).
    """
    n = arr.size
    if n < 2:
        return _zero_section_stats(int(n))

    # Reductions — each is a single SIMD pass.
    p_max = float(arr.max())
    p_min = float(arr.min())
    total = float(arr.sum())
    mean = total / n

    # Population variance: E[(X − E[X])²]. One extra pass.
    diff = arr - mean
    var = float((diff * diff).sum() / n)

    # Closed-form slope on x = arange(n):
    #   ∑x  = n(n−1)/2
    #   ∑x² = n(n−1)(2n−1)/6
    #   slope = (n·∑xy − ∑x·∑y) / (n·∑x² − (∑x)²)
    sx = (n - 1) * n * 0.5
    sxx = (n - 1) * n * (2 * n - 1) / 6.0
    sxy = float(np.dot(np.arange(n, dtype=np.float64), arr))
    denom = n * sxx - sx * sx
    slope = (n * sxy - sx * total) / denom if denom > 0 else 0.0

    return {
        "max": round(p_max, 3),
        "min": round(p_min, 3),
        "difference": round(p_max - p_min, 3),
        "average": round(mean, 3),
        "variance": round(var, 3),
        "trend_slope": round(slope, 6),
        "count": float(n),
    }


def _compute_section_stats(pressures: Sequence[float]) -> Dict[str, float]:
    """Backwards-compatible wrapper (accepts Python list).

    Keep this for the deprecated v2.5 path and any external caller; the
    new internal entry is ``_compute_section_stats_arr``.
    """
    n = len(pressures)
    if n < 2:
        return _zero_section_stats(n)
    arr = np.asarray(pressures, dtype=np.float64)
    return _compute_section_stats_arr(arr)


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
    n = min(len(pressures), len(angles))
    if n < 2:
        # Surface this rare-but-silent failure mode. Rate-limited so a
        # stuck cabin (no rotation, no data) doesn't drown the log.
        from core.rate_limit import warn_throttled
        warn_throttled(
            "compute_features_v26.degenerate",
            "compute_features_v26 received n=%d (< 2) for cabin %d; "
            "returning zero feats. Likely a comms hiccup or a stuck cabin.",
            n, cavity_id,
        )
        feats = {name: 0.0 for name in FEATURE_ORDER_43D}
        feats["cavity_id"] = float(cavity_id)
        return feats

    # Convert ONCE; section slicing is then a boolean mask on the same buffer.
    pressures_arr = np.asarray(pressures[:n], dtype=np.float64)
    angles_arr = np.asarray(angles[:n], dtype=np.float64)

    feats: Dict[str, float] = {}
    for section_name in SECTION_NAMES:
        start, end = profile.sections[section_name]
        # Half-open [start, end), as documented for segment_by_angle.
        mask = (angles_arr >= start) & (angles_arr < end)
        section_arr = pressures_arr[mask]
        stats = _compute_section_stats_arr(section_arr)
        for sub_name, value in stats.items():
            feats[f"{section_name}_{sub_name}"] = value

    feats["cavity_id"] = float(cavity_id)
    return feats


def features_to_vector(feats: Dict[str, float], mode: str = "43d") -> List[float]:
    """Convert a feature dict into a fixed-order vector.

    v2.6 uses ``mode="43d"`` (default).
    Legacy v2.5 modes ``"7d"``/``"6d"`` are retained for migration tests
    that haven't been rewritten.
    """
    if mode == "43d":
        # ``feats[k]`` is faster than ``feats.get(k, 0.0)`` and v2.6
        # always populates all 43 keys via _compute_section_stats_arr.
        return [feats[k] for k in _FEATURE_ORDER_43D_T]
    if mode == "7d":
        # Legacy callers may pass a v2.5 dict; .get keeps the contract
        # (missing keys → 0.0) so v2.5 unit tests still pass unchanged.
        return [feats.get(k, 0.0) for k in _LEGACY_7D_ORDER]
    if mode == "6d":
        return [feats.get(k, 0.0) for k in _LEGACY_6D_ORDER]
    raise ValueError(
        f"Unsupported feature mode: {mode}. Use '43d' (or legacy '7d'/'6d')."
    )


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
