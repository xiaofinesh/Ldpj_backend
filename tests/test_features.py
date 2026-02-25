"""Unit tests for core.features module."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.features import compute_features, features_to_vector


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
