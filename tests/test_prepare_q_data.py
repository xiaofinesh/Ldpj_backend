"""Tests for train.prepare_q_data (v2.6 Task 10)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.fixtures.generate_mock_q_data import generate
from train.prepare_q_data import compute_q_for_row, main as prepare_main
from core.cycle_profile import CycleProfile


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


@pytest.fixture
def cabins_cfg():
    return {
        "cabins": {
            1: {"v_cabin": 3.50e-4, "u_v_cabin": 7e-6, "notes": "calibrated"},
            2: {"v_cabin": 3.45e-4, "u_v_cabin": 6e-6, "notes": "calibrated"},
        },
        "default": {"v_cabin": 3.50e-4, "u_v_cabin": 1e-5},
    }


class TestComputeQForRow:
    def test_q_is_positive_for_decaying_curve(self, profile, cabins_cfg):
        df = generate(n_cabins=2, n_rounds=1, seed=42)
        for _, row in df.iterrows():
            q, dp = compute_q_for_row(row, profile, cabins_cfg)
            # Synthetic curves all have negative slope in hold → |dp/dt| > 0
            assert q is not None
            assert q > 0
            # dp/dt is negative (vacuum decay)
            assert dp is not None
            assert dp < 0

    def test_short_section_returns_none(self, profile, cabins_cfg):
        """Curves with too few hold-section points should be skipped."""
        bad_row = pd.Series({
            "pressure_data": json.dumps([600.0, 599.0]),
            "angle_data": json.dumps([200.0, 201.0]),
            "cavity_id": 1,
        })
        q, dp = compute_q_for_row(bad_row, profile, cabins_cfg)
        assert q is None
        assert dp is None

    def test_invalid_json_returns_none(self, profile, cabins_cfg):
        bad_row = pd.Series({
            "pressure_data": "not json",
            "angle_data": "[1,2,3]",
            "cavity_id": 1,
        })
        q, dp = compute_q_for_row(bad_row, profile, cabins_cfg)
        assert q is None and dp is None

    def test_initial_slope_window_isolates_initial_decay(self, profile, cabins_cfg):
        """README §5: first-N window captures the steep initial decay slope.

        Curve = steep linear drop for 30 samples, then flat. The full-window
        fit averages in the flat tail (shallower |dp/dt|); the first-30 window
        recovers the true initial slope (steeper |dp/dt|).
        """
        n = 60
        angles = [90.0 + 3.0 * i for i in range(n)]   # all inside hold [90,290)
        pressures = [600.0 - 2.0 * i if i < 30 else 600.0 - 2.0 * 29 for i in range(n)]
        row = pd.Series({
            "pressure_data": json.dumps(pressures),
            "angle_data": json.dumps(angles),
            "cavity_id": 1,
        })
        _, dp_full = compute_q_for_row(row, profile, cabins_cfg)
        _, dp_init = compute_q_for_row(row, profile, cabins_cfg,
                                       initial_slope_points=30)
        # Initial window sees the true -2 mbar/sample → -2000 Pa/s (interval 0.1s)
        assert dp_init == pytest.approx(-2000.0, rel=1e-6)
        # Full window is diluted by the flat tail → shallower magnitude
        assert abs(dp_init) > abs(dp_full)

    def test_per_row_interval_resolution(self, profile, cabins_cfg):
        """interval_s is resolved per row from cycle_profile_id, not the active
        profile — a row collected at a finer interval gets a 2× larger dp/dt."""
        n = 40
        angles = [90.0 + 4.0 * i for i in range(n)]   # inside hold [90,290)
        pressures = [600.0 - 1.0 * i for i in range(n)]
        base = {"pressure_data": json.dumps(pressures),
                "angle_data": json.dumps(angles), "cavity_id": 1}
        interval_by_profile = {"bph_13000": 0.1, "bph_fast": 0.05}

        row_slow = pd.Series({**base, "cycle_profile_id": "bph_13000"})
        row_fast = pd.Series({**base, "cycle_profile_id": "bph_fast"})
        _, dp_slow = compute_q_for_row(row_slow, profile, cabins_cfg,
                                       interval_by_profile=interval_by_profile)
        _, dp_fast = compute_q_for_row(row_fast, profile, cabins_cfg,
                                       interval_by_profile=interval_by_profile)
        # Same curve, half the interval → twice the dp/dt
        assert dp_fast == pytest.approx(2.0 * dp_slow, rel=1e-9)

    def test_missing_profile_id_falls_back_loudly(self, profile, cabins_cfg, caplog):
        """A row whose cycle_profile_id is absent from the map falls back to the
        active interval AND warns once."""
        import logging
        n = 20
        row = pd.Series({
            "pressure_data": json.dumps([600.0 - i for i in range(n)]),
            "angle_data": json.dumps([90.0 + 4.0 * i for i in range(n)]),
            "cavity_id": 1, "cycle_profile_id": "deleted_profile",
        })
        warned: set = set()
        with caplog.at_level(logging.WARNING):
            compute_q_for_row(row, profile, cabins_cfg,
                              interval_by_profile={"bph_13000": 0.1},
                              warned_ids=warned)
        assert "deleted_profile" in warned
        assert any("deleted_profile" in r.message for r in caplog.records)

    def test_q_scales_linearly_with_v_cabin(self, profile, cabins_cfg):
        """Doubling V_cabin must double Q for the same curve."""
        df = generate(n_cabins=1, n_rounds=1, seed=99)
        row = df.iloc[0]
        q1, _ = compute_q_for_row(row, profile, cabins_cfg)
        # Double V_cabin
        cfg2 = {
            "cabins": {1: {"v_cabin": 2 * cabins_cfg["cabins"][1]["v_cabin"],
                           "u_v_cabin": 0, "notes": ""}},
            "default": cabins_cfg["default"],
        }
        q2, _ = compute_q_for_row(row, profile, cfg2)
        assert q2 == pytest.approx(2 * q1, rel=1e-4)


class TestCli:
    def test_end_to_end_writes_q_column(self, tmp_path):
        """Run prepare_q_data on a small synthetic CSV; verify q_measured col."""
        # Generate raw-style CSV (with pressure_data, angle_data, cavity_id)
        df = generate(n_cabins=3, n_rounds=2, seed=42)
        # Drop q_measured / features so prepare_q_data has to compute q itself
        raw = df.drop(columns=["q_measured", "features"])
        raw_csv = tmp_path / "raw.csv"
        raw.to_csv(raw_csv, index=False)

        out_csv = tmp_path / "prepared.csv"

        rc = prepare_main([
            "--raw-csv", str(raw_csv),
            "--cabins-config", str(PROJECT_ROOT / "configs" / "cabins.yaml"),
            "--runtime-config", str(PROJECT_ROOT / "configs" / "runtime.yaml"),
            "--output", str(out_csv),
        ])
        assert rc == 0
        assert out_csv.exists()

        out = pd.read_csv(out_csv)
        assert "q_measured" in out.columns
        assert "dp_dt_pa_per_s" in out.columns
        assert (out["q_measured"] > 0).all()
        # All 6 rows kept (synthetic curves are well-formed)
        assert len(out) == 6
