"""Tests for train.train_m1 (v2.6 Task 10)."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.linear_regression_m1 import LinearRegressionM1
from tests.fixtures.generate_mock_q_data import generate
from train.train_m1 import fit_one_cabin, main as train_main


class TestFitOneCabin:
    def test_perfect_linear_relation_recovers_coefs(self):
        """Synthetic Q = 2.0 · slope + 0.5 → fit must recover (β, α)."""
        slopes = np.linspace(-1, 1, 50)
        q_values = 2.0 * slopes + 0.5
        result = fit_one_cabin(slopes, q_values, bootstrap_samples=100, seed=0)
        assert result is not None
        assert result["beta"] == pytest.approx(2.0, rel=1e-6)
        assert result["alpha"] == pytest.approx(0.5, rel=1e-6)
        assert result["r_squared"] == pytest.approx(1.0, abs=1e-9)
        assert result["n_samples"] == 50

    def test_noisy_data_lowers_r2(self):
        rng = np.random.default_rng(42)
        slopes = np.linspace(-1, 1, 100)
        q = 2.0 * slopes + 0.5 + rng.normal(0, 0.5, 100)  # heavy noise
        result = fit_one_cabin(slopes, q, bootstrap_samples=100, seed=0)
        assert 0.0 < result["r_squared"] < 1.0
        # u_β > 0 for noisy data
        assert result["u_beta"] > 0

    def test_too_few_samples_returns_none(self):
        result = fit_one_cabin(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                               bootstrap_samples=10, seed=0)
        assert result is None

    def test_bootstrap_reproducible_with_seed(self):
        """Same seed must give the same uncertainty estimates."""
        rng = np.random.default_rng(42)
        slopes = np.linspace(-1, 1, 50)
        q = 2.0 * slopes + 0.5 + rng.normal(0, 0.1, 50)
        a = fit_one_cabin(slopes, q, bootstrap_samples=200, seed=123)
        b = fit_one_cabin(slopes, q, bootstrap_samples=200, seed=123)
        assert a["u_beta"] == b["u_beta"]
        assert a["u_alpha"] == b["u_alpha"]


class TestCli:
    @pytest.fixture
    def labeled_csv(self, tmp_path):
        df = generate(n_cabins=5, n_rounds=12, seed=42)
        # train_m1 reads slope from features JSON or a direct column.
        # The mock fixture already populates 'features' (43-key JSON).
        csv = tmp_path / "labeled.csv"
        df.to_csv(csv, index=False)
        return csv

    def test_train_writes_loadable_coefficients(self, tmp_path, labeled_csv):
        out_json = tmp_path / "m1_coefficients.json"
        rc = train_main([
            "--data", str(labeled_csv),
            "--output", str(out_json),
            "--version", "test_v1",
            "--bootstrap-samples", "100",     # cheap for tests
            "--min-samples-per-cabin", "5",   # mock has only 12 rounds
        ])
        assert rc == 0
        assert out_json.exists()

        # Round-trip through LinearRegressionM1
        cfg = {"m1": {"coefficients_path": str(out_json.relative_to(tmp_path)),
                      "version": "test_v1"}}
        m1 = LinearRegressionM1(cfg, base_dir=tmp_path)
        m1.load()
        assert m1.loaded
        assert m1.version == "test_v1"
        assert m1.primary_section == "hold"
        # All 5 cabins should be calibrated
        assert m1.calibrated_cabins == [1, 2, 3, 4, 5]

        # JSON content sanity
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["feature"] == "hold_trend_slope"
        assert "n_cabins_calibrated" in data
        assert data["n_cabins_calibrated"] == 5

        # v2.6.3: trainer must stamp the operating_point fingerprint from the
        # active profile, and it must load back into M1.
        assert "operating_point" in data
        op = data["operating_point"]
        assert op["profile_id"] == "bph_13000"
        assert op["interval_s"] == 0.1
        assert op["hold_window_deg"] == [93.0, 283.0]
        assert m1.operating_point is not None
        assert m1.operating_point.profile_id == "bph_13000"
        for cid in [1, 2, 3, 4, 5]:
            entry = data["cabins"][str(cid)]
            assert "beta" in entry and "alpha" in entry
            assert "u_beta" in entry and "u_alpha" in entry
            assert entry["n_samples"] == 12  # 12 rounds per cabin

    def test_skip_cabin_with_too_few_samples(self, tmp_path):
        """When --min-samples-per-cabin is high, sparse cabins are skipped."""
        df = generate(n_cabins=3, n_rounds=4, seed=7)
        csv = tmp_path / "small.csv"
        df.to_csv(csv, index=False)
        out_json = tmp_path / "m1.json"
        rc = train_main([
            "--data", str(csv),
            "--output", str(out_json),
            "--version", "skipped",
            "--bootstrap-samples", "50",
            "--min-samples-per-cabin", "100",  # higher than any cabin has
        ])
        assert rc == 0
        data = json.loads(out_json.read_text(encoding="utf-8"))
        assert data["n_cabins_calibrated"] == 0

    def test_rejected_cabin_excluded_from_output(self, tmp_path):
        """Cabins whose R² falls below --min-r2 must NOT be written to the
        output JSON. At inference time, predict() on such a cabin must
        return cabin_calibrated=False so the processing loop's F011 path
        kicks in (instead of using a low-quality fit)."""
        # Build a dataset where cabin 3 has heavily corrupted q_measured
        # so its (slope, q_measured) pairs no longer linear-fit cleanly.
        df = generate(n_cabins=5, n_rounds=20, seed=42)
        # Inject random scatter into cabin 3's q_measured to wreck R²
        rng = np.random.default_rng(0)
        mask = df["cavity_id"] == 3
        df.loc[mask, "q_measured"] = rng.uniform(0, 1e-2, size=mask.sum())
        csv = tmp_path / "tainted.csv"
        df.to_csv(csv, index=False)

        out_json = tmp_path / "m1.json"
        rc = train_main([
            "--data", str(csv),
            "--output", str(out_json),
            "--version", "reject_test",
            "--min-r2", "0.99",
            "--bootstrap-samples", "50",
            "--min-samples-per-cabin", "5",
        ])
        assert rc == 0

        data = json.loads(out_json.read_text(encoding="utf-8"))
        # Cabin 3 should be rejected; tracked in the failed-acceptance metric
        assert "3" not in data["cabins"]
        assert data["n_cabins_failed_acceptance"] >= 1
        # The other cabins should still be present (rejection is per-cabin,
        # not all-or-nothing)
        assert "1" in data["cabins"] or "2" in data["cabins"]

        # Inference round-trip: cabin 3 falls back to global mean
        cfg = {"m1": {"coefficients_path": str(out_json.relative_to(tmp_path))}}
        m1 = LinearRegressionM1(cfg, base_dir=tmp_path)
        m1.load()
        assert 3 not in m1.calibrated_cabins
        result = m1.predict(primary_trend_slope=-1.0, cabin_id=3)
        assert result["cabin_calibrated"] is False  # F011 path

    def test_predict_round_trip(self, tmp_path, labeled_csv):
        """Train M1, then call .predict() and check Q_est is finite + positive
        for the same slopes the trainer saw."""
        out_json = tmp_path / "m1.json"
        rc = train_main([
            "--data", str(labeled_csv),
            "--output", str(out_json),
            "--bootstrap-samples", "100",
            "--min-samples-per-cabin", "5",
        ])
        assert rc == 0

        cfg = {"m1": {"coefficients_path": str(out_json.relative_to(tmp_path))}}
        m1 = LinearRegressionM1(cfg, base_dir=tmp_path)
        m1.load()

        # Pull the actual hold_trend_slope of one row from labeled CSV
        df = pd.read_csv(labeled_csv)
        row = df.iloc[0]
        slope = json.loads(row["features"])["hold_trend_slope"]
        result = m1.predict(slope, cabin_id=int(row["cavity_id"]))
        assert result["cabin_calibrated"] is True
        assert result["q_est"] > 0
