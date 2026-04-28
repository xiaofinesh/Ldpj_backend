"""Tests for M1 linear regression model (v2.6)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from models.linear_regression_m1 import LinearRegressionM1


def _write_coef_file(tmp_path: Path, *, name: str = "m1_coef.json",
                     primary_section: str = "hold",
                     cabins: dict | None = None,
                     version: str = "test_v1") -> Path:
    if cabins is None:
        cabins = {
            "1": {"beta": 10.0, "alpha": 0.001, "r_squared": 0.99,
                  "u_beta": 0.5, "u_alpha": 0.0001, "n_samples": 150},
            "2": {"beta": 12.0, "alpha": 0.002, "r_squared": 0.995,
                  "u_beta": 0.4, "u_alpha": 0.0002, "n_samples": 150},
        }
    f = tmp_path / name
    f.write_text(json.dumps({
        "version": version,
        "trained_at": "2026-05-22T00:00:00",
        "feature": f"{primary_section}_trend_slope",
        "target": "Q (Pa·m³/s)",
        "primary_section": primary_section,
        "cabins": cabins,
    }), encoding="utf-8")
    return f


def _cfg_for(coef_file: Path, tmp_path: Path) -> dict:
    """Build a models_cfg dict that points to coef_file via a relative path."""
    return {"m1": {
        "coefficients_path": str(coef_file.relative_to(tmp_path)),
        "version": "test_v1",
    }}


class TestLoad:
    def test_load_basic(self, tmp_path):
        f = _write_coef_file(tmp_path)
        m1 = LinearRegressionM1(_cfg_for(f, tmp_path), base_dir=tmp_path)
        m1.load()
        assert m1.loaded is True
        assert m1.version == "test_v1"
        assert m1.primary_section == "hold"
        assert m1.calibrated_cabins == [1, 2]

    def test_load_missing_file_raises(self, tmp_path):
        cfg = {"m1": {"coefficients_path": "nonexistent.json"}}
        m1 = LinearRegressionM1(cfg, base_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            m1.load()
        assert m1.loaded is False

    def test_default_paths_when_cfg_missing(self):
        m1 = LinearRegressionM1({})
        # Should not raise; just isn't loaded yet
        assert m1.loaded is False
        assert m1.primary_section == "hold"

    def test_primary_section_override(self, tmp_path):
        f = _write_coef_file(tmp_path, primary_section="evac",
                             cabins={"1": {"beta": 5.0, "alpha": 0.0,
                                           "u_beta": 0.1, "u_alpha": 0.0,
                                           "n_samples": 50}})
        m1 = LinearRegressionM1(_cfg_for(f, tmp_path), base_dir=tmp_path)
        m1.load()
        assert m1.primary_section == "evac"


class TestPredictCalibrated:
    def test_basic_arithmetic(self, tmp_path):
        f = _write_coef_file(tmp_path)
        m1 = LinearRegressionM1(_cfg_for(f, tmp_path), base_dir=tmp_path)
        m1.load()
        # cabin 1: β=10, α=0.001, slope=0.5 → Q = 5.001
        result = m1.predict(primary_trend_slope=0.5, cabin_id=1)
        assert result["q_est"] == pytest.approx(5.001)
        assert result["cabin_calibrated"] is True
        assert result["uncertainty"] > 0

    def test_zero_slope_returns_alpha(self, tmp_path):
        f = _write_coef_file(tmp_path)
        m1 = LinearRegressionM1(_cfg_for(f, tmp_path), base_dir=tmp_path)
        m1.load()
        result = m1.predict(primary_trend_slope=0.0, cabin_id=1)
        # At slope=0, Q == α
        assert result["q_est"] == pytest.approx(0.001)
        # Uncertainty at slope=0 is just u_alpha
        assert result["uncertainty"] == pytest.approx(0.0001)

    def test_uncertainty_combine_formula(self, tmp_path):
        """u_Q² = (u_β·slope)² + u_α²"""
        f = _write_coef_file(tmp_path, cabins={
            "5": {"beta": 10.0, "alpha": 0.0, "u_beta": 0.5,
                  "u_alpha": 0.001, "n_samples": 100, "r_squared": 0.99},
        })
        m1 = LinearRegressionM1(_cfg_for(f, tmp_path), base_dir=tmp_path)
        m1.load()
        result = m1.predict(primary_trend_slope=2.0, cabin_id=5)
        expected_u = ((0.5 * 2.0) ** 2 + 0.001 ** 2) ** 0.5
        assert result["uncertainty"] == pytest.approx(expected_u)

    def test_relative_uncertainty_when_q_nonzero(self, tmp_path):
        f = _write_coef_file(tmp_path)
        m1 = LinearRegressionM1(_cfg_for(f, tmp_path), base_dir=tmp_path)
        m1.load()
        result = m1.predict(primary_trend_slope=1.0, cabin_id=2)
        # Q = 12.0 * 1.0 + 0.002 = 12.002
        # u = sqrt((0.4*1.0)² + 0.0002²) ≈ 0.4
        # rel_unc ≈ 0.4 / 12.002 ≈ 0.033
        assert result["q_est"] == pytest.approx(12.002, rel=1e-4)
        assert result["relative_uncertainty"] == pytest.approx(
            result["uncertainty"] / abs(result["q_est"])
        )


class TestPredictUncalibratedFallback:
    def test_unknown_cabin_uses_mean(self, tmp_path):
        f = _write_coef_file(tmp_path)
        m1 = LinearRegressionM1(_cfg_for(f, tmp_path), base_dir=tmp_path)
        m1.load()
        result = m1.predict(primary_trend_slope=0.5, cabin_id=99)
        assert result["cabin_calibrated"] is False
        # mean β = 11.0, mean α = 0.0015 → 11*0.5 + 0.0015 = 5.5015
        assert result["q_est"] == pytest.approx(5.5015, rel=1e-3)
        # Conservative 30% relative uncertainty for fallback
        assert result["relative_uncertainty"] == pytest.approx(0.30)

    def test_empty_table_returns_zero_q(self, tmp_path):
        """When no cabins are calibrated, fallback returns Q=0 with inf uncertainty."""
        f = _write_coef_file(tmp_path, cabins={})
        m1 = LinearRegressionM1(_cfg_for(f, tmp_path), base_dir=tmp_path)
        m1.load()
        result = m1.predict(primary_trend_slope=0.5, cabin_id=1)
        assert result["q_est"] == 0.0
        assert result["uncertainty"] == float("inf")
        assert result["cabin_calibrated"] is False
