"""Per-cycle data-quality bit flags (v2.6).

Each completed cycle gets a single 32-bit integer in the DB whose bits
flag silently-degraded computations (short sections, clamped formulas,
etc.). Downstream ML data analysis can filter rows where any of these
bits are set instead of treating all 70-point cycles as equal-quality.

Bit layout (1 << N):
    0  QF_DEGENERATE_INPUT     full cycle had < 2 points
    1  QF_SHORT_BASELINE_PRE   section had < 2 points
    2  QF_SHORT_EVAC
    3  QF_SHORT_STABLE
    4  QF_SHORT_HOLD           ← M1 critical: hold_trend_slope unreliable
    5  QF_SHORT_RELEASE
    6  QF_SHORT_BASELINE_POST
    7  QF_CD_CLAMPED           q_d_conversion C_d hit [0.5, 0.95] limit
                                (set externally by ResultSender / API
                                 path that calls q_to_d_choked)
"""

from __future__ import annotations

from typing import Dict, Mapping

QF_DEGENERATE_INPUT      = 1 << 0
QF_SHORT_BASELINE_PRE    = 1 << 1
QF_SHORT_EVAC            = 1 << 2
QF_SHORT_STABLE          = 1 << 3
QF_SHORT_HOLD            = 1 << 4
QF_SHORT_RELEASE         = 1 << 5
QF_SHORT_BASELINE_POST   = 1 << 6
QF_CD_CLAMPED            = 1 << 7

# Section name → bit. Order MUST match SECTION_NAMES in cycle_profile.py.
_SECTION_BITS: Dict[str, int] = {
    "baseline_pre":  QF_SHORT_BASELINE_PRE,
    "evac":          QF_SHORT_EVAC,
    "stable":        QF_SHORT_STABLE,
    "hold":          QF_SHORT_HOLD,
    "release":       QF_SHORT_RELEASE,
    "baseline_post": QF_SHORT_BASELINE_POST,
}


def compute_quality_flags(feats: Mapping[str, float]) -> int:
    """Inspect a 43-dim feats dict and return the section-quality bitmask.

    Looks at each section's ``*_count`` field; bit set if the section had
    fewer than 2 points (slope/variance unreliable in that section).

    A fully degenerate input (no useful data anywhere) is detected by
    *all* section counts being zero — every short-section bit will be set
    and we additionally OR in ``QF_DEGENERATE_INPUT`` so callers can
    cheaply check "any data at all" without scanning each section.

    The caller is expected to OR additional bits (e.g. QF_CD_CLAMPED)
    that come from outside the feature pipeline.
    """
    flags = 0
    nonzero_section = False
    for section, bit in _SECTION_BITS.items():
        n = feats.get(f"{section}_count", 0)
        if n < 2:
            flags |= bit
        if n > 0:
            nonzero_section = True
    if not nonzero_section:
        flags |= QF_DEGENERATE_INPUT
    return flags
