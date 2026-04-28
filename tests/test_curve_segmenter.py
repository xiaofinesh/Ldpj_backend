"""Tests for core.curve_segmenter (v2.6)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.cycle_profile import CycleProfile, SECTION_NAMES
from core.curve_segmenter import segment_by_angle, segment_indices_by_angle


@pytest.fixture
def profile():
    """Test profile with simple round-number boundaries (vs production 57.6/115/etc)."""
    return CycleProfile(
        profile_id="test",
        bph=13000,
        cycle_total_ms=6900,
        sections={
            "baseline_pre":  (0.0,   60.0),
            "evac":          (60.0,  100.0),
            "stable":        (100.0, 120.0),
            "hold":          (120.0, 280.0),
            "release":       (280.0, 310.0),
            "baseline_post": (310.0, 360.0),
        },
        trigger_angle=0.0,
        collection_points=70,
        collection_interval_s=0.1,
        collection_timeout_s=10.0,
        primary_section="hold",
    )


class TestSegmentByAngle:
    def test_one_point_per_section(self, profile):
        pressures = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        angles = [10.0, 70.0, 110.0, 200.0, 290.0, 320.0]
        result = segment_by_angle(pressures, angles, profile)
        assert result["baseline_pre"] == [10.0]
        assert result["evac"] == [20.0]
        assert result["stable"] == [30.0]
        assert result["hold"] == [40.0]
        assert result["release"] == [50.0]
        assert result["baseline_post"] == [60.0]

    def test_boundary_inclusive_left(self, profile):
        """A point exactly at start_angle belongs to that section."""
        result = segment_by_angle([100.0], [60.0], profile)
        assert result["evac"] == [100.0]
        assert result["baseline_pre"] == []

    def test_boundary_exclusive_right(self, profile):
        """A point exactly at end_angle belongs to the NEXT section."""
        result = segment_by_angle([100.0], [100.0], profile)
        assert result["stable"] == [100.0]
        assert result["evac"] == []

    def test_out_of_range_dropped(self, profile):
        """Points with angle >= 360 (or < 0) are silently dropped."""
        result = segment_by_angle([50.0, 100.0], [180.0, 400.0], profile)
        assert result["hold"] == [50.0]
        total = sum(len(v) for v in result.values())
        assert total == 1  # 400° dropped

    def test_all_sections_present_even_when_empty(self, profile):
        result = segment_by_angle([10.0], [200.0], profile)
        assert set(result.keys()) == set(SECTION_NAMES)
        assert result["baseline_pre"] == []
        assert result["evac"] == []
        assert result["hold"] == [10.0]

    def test_preserves_index_order_within_section(self, profile):
        """Segmentation is stable (same order as input)."""
        pressures = [1.0, 2.0, 3.0, 4.0]
        angles = [150.0, 200.0, 250.0, 130.0]  # all in hold
        result = segment_by_angle(pressures, angles, profile)
        assert result["hold"] == [1.0, 2.0, 3.0, 4.0]

    def test_length_mismatch_truncates_to_shorter(self, profile):
        result = segment_by_angle([1.0, 2.0, 3.0], [150.0], profile)
        # Only the first pair (1.0 @ 150°) is processed
        assert result["hold"] == [1.0]
        total = sum(len(v) for v in result.values())
        assert total == 1

    def test_empty_input(self, profile):
        result = segment_by_angle([], [], profile)
        for name in SECTION_NAMES:
            assert result[name] == []


class TestSegmentIndicesByAngle:
    def test_returns_indices(self, profile):
        angles = [10.0, 70.0, 110.0, 200.0, 290.0, 320.0]
        idx = segment_indices_by_angle(angles, profile)
        assert idx["baseline_pre"] == [0]
        assert idx["evac"] == [1]
        assert idx["stable"] == [2]
        assert idx["hold"] == [3]
        assert idx["release"] == [4]
        assert idx["baseline_post"] == [5]

    def test_useful_for_parallel_arrays(self, profile):
        """Demonstrate the documented use case: slicing parallel arrays."""
        angles = [10.0, 50.0, 200.0, 250.0, 320.0]
        timestamps = [0.0, 0.1, 0.2, 0.3, 0.4]
        idx = segment_indices_by_angle(angles, profile)
        hold_ts = [timestamps[i] for i in idx["hold"]]
        assert hold_ts == [0.2, 0.3]
