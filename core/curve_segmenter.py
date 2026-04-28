"""Curve segmentation by angle (v2.6).

Splits a (pressures, angles) sequence into 6 named sections according to
a CycleProfile's section boundaries. Each section is a list of points
whose angle falls within [start_angle, end_angle).

Notes
-----
- Section boundaries are half-open [start, end): a point exactly at
  end_angle belongs to the *next* section.
- Points whose angle falls outside all 6 sections are silently dropped.
- Order of points within a section follows their original index order
  (segmentation is index-preserving, not sorted by angle).
"""

from __future__ import annotations

import logging
from typing import Dict, List

from core.cycle_profile import CycleProfile, SECTION_NAMES

logger = logging.getLogger(__name__)


def segment_by_angle(
    pressures: List[float],
    angles: List[float],
    profile: CycleProfile,
) -> Dict[str, List[float]]:
    """Split a pressure sequence into 6 named sections by angle.

    Parameters
    ----------
    pressures : list of float
    angles : list of float
        Same length as ``pressures``; each angle in degrees.
    profile : CycleProfile
        Defines the half-open section boundaries.

    Returns
    -------
    dict[str, list[float]]
        Mapping section name → pressures in that section. All 6 standard
        sections are always present (empty list if no points fall in range).
    """
    if len(pressures) != len(angles):
        logger.warning(
            "segment_by_angle: length mismatch (pressures=%d, angles=%d), truncating",
            len(pressures), len(angles),
        )
    n = min(len(pressures), len(angles))

    result: Dict[str, List[float]] = {name: [] for name in SECTION_NAMES}

    for i in range(n):
        a = angles[i]
        for name in SECTION_NAMES:
            start, end = profile.sections[name]
            if start <= a < end:
                result[name].append(pressures[i])
                break

    return result


def segment_indices_by_angle(
    angles: List[float],
    profile: CycleProfile,
) -> Dict[str, List[int]]:
    """Same as segment_by_angle but returns indices, not values.

    Useful when the caller needs to slice multiple parallel arrays
    (pressures, angles, timestamps, ...).
    """
    result: Dict[str, List[int]] = {name: [] for name in SECTION_NAMES}
    for i, a in enumerate(angles):
        for name in SECTION_NAMES:
            start, end = profile.sections[name]
            if start <= a < end:
                result[name].append(i)
                break
    return result
