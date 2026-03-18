"""Feature computation module – implements the 7-dimension feature contract.

Feature contract (order matters):
    ['max', 'min', 'difference', 'average', 'variance', 'trend_slope', 'cavity_id']
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np


def compute_features(pressures: List[float], cavity_id: int) -> Dict[str, float]:
    if not pressures or len(pressures) < 2:
        return {"max": 0.0, "min": 0.0, "difference": 0.0, "average": 0.0,
                "variance": 0.0, "trend_slope": 0.0, "cavity_id": float(cavity_id)}

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


def features_to_vector(feats: Dict[str, float], mode: str = "7d") -> List[float]:
    if mode == "6d":
        order = ["max", "min", "difference", "average", "variance", "trend_slope"]
    else:
        order = ["max", "min", "difference", "average", "variance", "trend_slope", "cavity_id"]
    return [feats.get(k, 0.0) for k in order]
