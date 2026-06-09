"""Unit tests for core.quality_flags (v2.6.1, 5 sections)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from core.cycle_profile import CycleProfile
from core.features import compute_features_v26
from core.quality_flags import (
    QF_CD_CLAMPED,
    QF_DEGENERATE_INPUT,
    QF_NONFINITE,
    QF_SHORT_BASELINE_POST,
    QF_SHORT_BASELINE_PRE,
    QF_SHORT_EVAC,
    QF_SHORT_HOLD,
    QF_SHORT_RELEASE,
    compute_quality_flags,
)


def test_nonfinite_feature_sets_flag():
    """A NaN/inf feature value must set QF_NONFINITE (the most dangerous
    silent corruption — a NaN Q would read as OK against any threshold)."""
    feats = {f"{s}_count": 10.0 for s in
             ("baseline_pre", "evac", "hold", "release", "baseline_post")}
    feats["hold_trend_slope"] = float("nan")
    assert compute_quality_flags(feats) & QF_NONFINITE

    feats["hold_trend_slope"] = float("inf")
    assert compute_quality_flags(feats) & QF_NONFINITE


def test_finite_features_no_nonfinite_flag():
    feats = {f"{s}_count": 10.0 for s in
             ("baseline_pre", "evac", "hold", "release", "baseline_post")}
    feats["hold_trend_slope"] = -0.14
    assert not (compute_quality_flags(feats) & QF_NONFINITE)


@pytest.fixture
def profile():
    return CycleProfile(
        profile_id="bph_13000",
        bph=13000,
        cycle_total_ms=6900,
        sections={
            "baseline_pre":  (0.0,   75.0),
            "evac":          (75.0,  90.0),
            "hold":          (90.0,  290.0),
            "release":       (290.0, 304.0),
            "baseline_post": (304.0, 360.0),
        },
        trigger_angle=0.0,
        collection_points=70,
        collection_interval_s=0.1,
        collection_timeout_s=10.0,
        primary_section="hold",
    )


class TestComputeQualityFlags:
    def test_normal_70_point_cycle_flags_zero(self, profile):
        """A well-formed 70-point cycle: every section has plenty of points."""
        rng = np.random.default_rng(42)
        # 70 points evenly spaced 0..360°
        angles = np.linspace(0, 360, 70, endpoint=False).tolist()
        pressures = (np.linspace(0, 600, 70) + rng.normal(0, 0.5, 70)).tolist()
        feats = compute_features_v26(pressures, angles, cavity_id=1, profile=profile)
        assert compute_quality_flags(feats) == 0

    def test_short_hold_section_sets_qf_short_hold(self, profile):
        """All angles outside the hold range → hold_count=0 → QF_SHORT_HOLD set."""
        # Put all 10 points in baseline_pre only
        angles = [10.0] * 10
        pressures = [100.0 + i for i in range(10)]
        feats = compute_features_v26(pressures, angles, cavity_id=1, profile=profile)
        flags = compute_quality_flags(feats)
        assert flags & QF_SHORT_HOLD
        # Other empty sections also flagged
        assert flags & QF_SHORT_EVAC
        assert flags & QF_SHORT_RELEASE
        assert flags & QF_SHORT_BASELINE_POST
        # baseline_pre HAS data so its bit is NOT set
        assert not (flags & QF_SHORT_BASELINE_PRE)
        # Some section had data → not fully degenerate
        assert not (flags & QF_DEGENERATE_INPUT)

    def test_completely_degenerate_input_sets_all_section_bits_and_qf_degenerate(self, profile):
        """compute_features_v26 with n<2 returns all-zero feats — every
        section count is 0, so all 5 short-section bits + QF_DEGENERATE_INPUT
        are set."""
        feats = compute_features_v26([], [], cavity_id=1, profile=profile)
        flags = compute_quality_flags(feats)
        assert flags & QF_DEGENERATE_INPUT
        for bit in (QF_SHORT_BASELINE_PRE, QF_SHORT_EVAC,
                    QF_SHORT_HOLD, QF_SHORT_RELEASE, QF_SHORT_BASELINE_POST):
            assert flags & bit

    def test_cd_clamped_bit_is_independent(self):
        """QF_CD_CLAMPED is set externally; compute_quality_flags never sets it."""
        # Synthetic feats with all section counts >= 2 and the bit explicitly OR'd in
        feats = {f"{s}_count": 10 for s in (
            "baseline_pre", "evac", "hold", "release", "baseline_post"
        )}
        # compute_quality_flags doesn't touch this bit
        assert compute_quality_flags(feats) == 0
        # The bit is reserved for callers (e.g. q_d_conversion path) to OR in.
        combined = compute_quality_flags(feats) | QF_CD_CLAMPED
        assert combined & QF_CD_CLAMPED
