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
            "baseline_pre":  (0.0,   57.6),
            "evac":          (57.6,  93.0),
            "stable":        (93.0,  115.0),
            "hold":          (115.0, 273.6),
            "release":       (273.6, 302.4),
            "baseline_post": (302.4, 360.0),
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
