"""Unit tests for core.features module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.cycle_profile import CycleProfile
from core.feature_spec import (
    FEATURE_ORDER_36D,
    SECTION_SUB_FEATURES,
    primary_trend_slope_index,
)
from core.features import (
    compute_features,
    compute_features_v26,
    features_to_vector,
)


# ── v2.5 (deprecated path, kept for back-compat) ───────────────────────

class TestComputeFeatures:
    def test_basic(self):
        pressures = [100.0, 200.0, 300.0, 400.0, 500.0]
        feats = compute_features(pressures, cavity_id=2)
        assert feats["max"] == 500.0
        assert feats["min"] == 100.0
        assert feats["difference"] == 400.0
        assert feats["average"] == 300.0
        assert feats["cavity_id"] == 2.0
        assert "variance" in feats
        assert "trend_slope" in feats

    def test_empty_input(self):
        feats = compute_features([], cavity_id=0)
        assert feats["max"] == 0.0
        assert feats["min"] == 0.0

    def test_single_point(self):
        feats = compute_features([42.0], cavity_id=1)
        assert feats["max"] == 0.0  # < 2 points returns zeros

    def test_constant_pressure(self):
        pressures = [500.0] * 100
        feats = compute_features(pressures, cavity_id=3)
        assert feats["difference"] == 0.0
        assert feats["variance"] == 0.0
        assert abs(feats["trend_slope"]) < 1e-6


class TestFeaturesToVector:
    def test_7d_order(self):
        feats = compute_features([10.0, 20.0, 30.0], cavity_id=5)
        vec = features_to_vector(feats, mode="7d")
        assert len(vec) == 7
        assert vec[-1] == 5.0  # cavity_id is last

    def test_6d_order(self):
        feats = compute_features([10.0, 20.0, 30.0], cavity_id=5)
        vec = features_to_vector(feats, mode="6d")
        assert len(vec) == 6

    def test_unknown_mode_raises(self):
        feats = compute_features([10.0, 20.0, 30.0], cavity_id=5)
        with pytest.raises(ValueError, match="Unsupported feature mode"):
            features_to_vector(feats, mode="99d")


# ── v2.6.1 (current 36-dim path, 5 sections) ─────────────────────────

@pytest.fixture
def profile():
    """Test profile with simple round-number boundaries (5 sections)."""
    return CycleProfile(
        profile_id="test",
        bph=13000,
        cycle_total_ms=6900,
        sections={
            "baseline_pre":  (0.0,   60.0),
            "evac":          (60.0,  100.0),
            "hold":          (100.0, 280.0),
            "release":       (280.0, 310.0),
            "baseline_post": (310.0, 360.0),
        },
        trigger_angle=0.0,
        collection_points=70,
        collection_interval_s=0.1,
        collection_timeout_s=10.0,
        primary_section="hold",
    )


class TestFeatureSpec:
    def test_36_features_total(self):
        assert len(FEATURE_ORDER_36D) == 36

    def test_cavity_id_is_last(self):
        assert FEATURE_ORDER_36D[-1] == "cavity_id"

    def test_seven_sub_features_per_section(self):
        assert len(SECTION_SUB_FEATURES) == 7
        assert "trend_slope" in SECTION_SUB_FEATURES
        assert "count" in SECTION_SUB_FEATURES

    def test_primary_trend_slope_index_hold(self):
        idx = primary_trend_slope_index("hold")
        assert FEATURE_ORDER_36D[idx] == "hold_trend_slope"

    def test_primary_trend_slope_index_other_section(self):
        idx = primary_trend_slope_index("evac")
        assert FEATURE_ORDER_36D[idx] == "evac_trend_slope"


class TestComputeFeaturesV26:
    def test_36_dim_output(self, profile):
        # 70 points spanning 0–360°
        pressures = [600.0 + i for i in range(70)]
        angles = [i * 360.0 / 70 for i in range(70)]
        feats = compute_features_v26(pressures, angles, cavity_id=5, profile=profile)
        for key in FEATURE_ORDER_36D:
            assert key in feats, f"missing key: {key}"
        assert feats["cavity_id"] == 5.0

    def test_features_to_vector_36d(self, profile):
        pressures = [600.0] * 70
        angles = [i * 360.0 / 70 for i in range(70)]
        feats = compute_features_v26(pressures, angles, 5, profile)
        vec = features_to_vector(feats, mode="36d")
        assert len(vec) == 36
        assert vec[-1] == 5.0  # cavity_id is last

    def test_empty_section_yields_zero_count(self, profile):
        """All angles in hold → other sections' count == 0."""
        pressures = [600.0] * 10
        angles = [200.0] * 10
        feats = compute_features_v26(pressures, angles, cavity_id=1, profile=profile)
        assert feats["baseline_pre_count"] == 0.0
        assert feats["baseline_pre_max"] == 0.0
        assert feats["evac_count"] == 0.0
        assert feats["release_count"] == 0.0
        assert feats["hold_count"] == 10.0

    def test_primary_trend_slope_consistency(self, profile):
        """hold_trend_slope ≈ slope of pressure sequence in hold section."""
        # 20 points at integer slope 1.0 within hold (100°–280°)
        pressures = [600.0 + i for i in range(20)]
        angles = [100.0 + i * 8.0 for i in range(20)]  # 100, 108, ..., 252 — all hold
        feats = compute_features_v26(pressures, angles, cavity_id=1, profile=profile)
        assert abs(feats["hold_trend_slope"] - 1.0) < 0.01
        idx = primary_trend_slope_index("hold")
        vec = features_to_vector(feats, mode="36d")
        assert abs(vec[idx] - 1.0) < 0.01

    def test_short_input_returns_zeros(self, profile):
        feats = compute_features_v26([], [], cavity_id=7, profile=profile)
        for key in FEATURE_ORDER_36D:
            if key == "cavity_id":
                assert feats[key] == 7.0
            else:
                assert feats[key] == 0.0

    def test_single_point_returns_zeros(self, profile):
        feats = compute_features_v26([100.0], [50.0], cavity_id=2, profile=profile)
        # < 2 points: short-circuited zeros (cavity_id still set)
        assert feats["cavity_id"] == 2.0
        assert feats["baseline_pre_max"] == 0.0
        assert feats["hold_count"] == 0.0

    def test_section_with_one_point_is_zeroed_but_count_one(self, profile):
        """Section with exactly 1 point: stats = 0, count = 1 (not enough for slope)."""
        # 5 points: 1 in baseline_pre, 4 in hold (test profile: hold = [100, 280))
        pressures = [10.0, 100.0, 200.0, 300.0, 400.0]
        angles = [10.0, 150.0, 200.0, 250.0, 270.0]
        feats = compute_features_v26(pressures, angles, cavity_id=1, profile=profile)
        # baseline_pre has 1 point → count=1, but stats are all 0 (n < 2 short-circuit)
        assert feats["baseline_pre_count"] == 1.0
        assert feats["baseline_pre_max"] == 0.0
        assert feats["baseline_pre_trend_slope"] == 0.0
        # hold has 4 points → real stats
        assert feats["hold_count"] == 4.0
        assert feats["hold_max"] == 400.0
        assert feats["hold_min"] == 100.0
        assert feats["hold_trend_slope"] > 0  # increasing

    def test_real_runtime_profile_compatible(self):
        """Sanity-check against the production profile (90/290/etc., 5 sections)."""
        from configs.loaders import load_active_cycle_profile
        prof = load_active_cycle_profile()
        # 70 evenly spaced points — all sections should pick some up
        pressures = [500.0 + i for i in range(70)]
        angles = [i * 360.0 / 70 for i in range(70)]
        feats = compute_features_v26(pressures, angles, cavity_id=3, profile=prof)
        assert len(feats) == 36
        # hold section is the largest (90°→290°, 200° wide), should have most points
        assert feats["hold_count"] > feats["evac_count"]
        assert feats["hold_count"] > feats["release_count"]
        # All counts sum to == 70 (boundaries are half-open, no points dropped)
        total = sum(feats[f"{s}_count"] for s in [
            "baseline_pre", "evac", "hold", "release", "baseline_post"
        ])
        assert total == 70
