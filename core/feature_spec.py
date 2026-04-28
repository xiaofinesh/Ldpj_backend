"""Feature specification for v2.6.1 (36-dimensional).

The 36-dim feature vector is:

    [
      # Section 1: baseline_pre   (7)
      baseline_pre_max, baseline_pre_min, baseline_pre_difference,
      baseline_pre_average, baseline_pre_variance,
      baseline_pre_trend_slope, baseline_pre_count,

      # Section 2: evac           (7)
      evac_max, ..., evac_count,

      # Section 3: hold           (7)  ← primary section, M1 reads hold_trend_slope
      hold_max, ..., hold_count,

      # Section 4: release        (7)
      release_max, ..., release_count,

      # Section 5: baseline_post  (7)
      baseline_post_max, ..., baseline_post_count,

      # 36th dim
      cavity_id
    ]

History: v2.6 had 6 sections (43 dims), with a `stable` segment between
evac and hold. Analysis of 131808 / 112936 production data showed
`stable` was an artifact — the trend slope is continuous from the
moment vacuum is reached (~90°) all the way to release (~290°). The
segment was dropped in v2.6.1; M1 still reads `hold_trend_slope`.

Note: 'count' is the 7th sub-feature (replaces v2.5's per-row cavity_id),
so a too-short or empty section yields count=0 alongside the zeroed stats.
"""

from __future__ import annotations

from typing import List

from core.cycle_profile import SECTION_NAMES


# Per-section sub-features. Order matters — used to assemble FEATURE_ORDER_36D.
SECTION_SUB_FEATURES: List[str] = [
    "max",
    "min",
    "difference",
    "average",
    "variance",
    "trend_slope",
    "count",
]


# Full 36-dim feature names in canonical order
FEATURE_ORDER_36D: List[str] = [
    f"{section}_{sub}"
    for section in SECTION_NAMES
    for sub in SECTION_SUB_FEATURES
] + ["cavity_id"]

assert len(FEATURE_ORDER_36D) == 36, (
    f"Expected 36 features, got {len(FEATURE_ORDER_36D)}"
)


def primary_trend_slope_index(primary_section: str = "hold") -> int:
    """Return the index of <primary_section>_trend_slope in FEATURE_ORDER_36D.

    Used by M1 to extract a single feature without recomputing.
    """
    target = f"{primary_section}_trend_slope"
    return FEATURE_ORDER_36D.index(target)
