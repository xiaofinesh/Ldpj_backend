"""Integration tests for the v2.6 ProcessingLoop.

These tests run the loop on synthetic CycleData (built directly, no
PollingEngine) and verify the dual-track Q regression fuses correctly,
labels are assigned by Q_threshold, and the right faults are raised.
"""

import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.cycle_fsm import CabinFSM, CycleFSMManager, CycleState
from core.cycle_profile import CycleProfile
from core.polling_engine import CabinFrame
from health.fault_reporter import FaultReporter
from models.linear_regression_m1 import LinearRegressionM1
from models.xgb_regressor_m2 import XGBRegressorM2
from pipeline.processing_loop import (
    LABEL_LEAK, LABEL_NA, LABEL_NO_BOTTLE, LABEL_OK, ProcessingLoop,
)
from storage.database_logger import DatabaseLogger


# ── Test helpers ─────────────────────────────────────────────────────

def _profile() -> CycleProfile:
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


def _runtime_cfg(*, no_bottle=50.0, a_resolution=1e-5,
                 m_disagree=0.20, loop_interval=0.0):
    return {
        "loop_interval": loop_interval,
        "no_bottle_threshold": no_bottle,
        "model_inference": {
            "a_resolution": a_resolution,
            "m_disagreement_threshold": m_disagree,
        },
    }


def _products_cfg(q_threshold=1.0e-3, default_id="TEST"):
    return {
        "default_product_id": default_id,
        "products": {default_id: {"q_threshold": q_threshold,
                                  "flow_regime": "laminar", "l_ref_mm": 0.5}},
    }


def _cabins_cfg(calibrated_ids=(1, 2, 3, 4, 5)):
    cabins = {}
    for cid in calibrated_ids:
        cabins[cid] = {"v_cabin": 3.50e-4, "u_v_cabin": 7e-6,
                       "notes": "calibrated"}
    return {"cabins": cabins, "default": {"v_cabin": 3.5e-4, "u_v_cabin": 1e-5}}


def _seed_fsm_with_cycle(fsm: CabinFSM, *, slope_per_sample: float,
                         hold_baseline: float = 600.0,
                         n_points: int = 40) -> None:
    """Force an FSM into PROCESSING with synthetic hold-section data.

    Bypasses the trigger by directly populating ``_data`` and setting state.
    """
    angles = np.linspace(116.0, 270.0, n_points)  # all in hold (115..273.6)
    pressures = hold_baseline + slope_per_sample * np.arange(n_points)
    ts0 = time.time()
    timestamps = ts0 + np.arange(n_points) * 0.1

    fsm._state = CycleState.PROCESSING
    fsm._data.start_time = ts0
    fsm._data.end_angle = float(angles[-1])
    fsm._data.cycle_profile_id = fsm._profile.profile_id
    fsm._data.leak_valve_status = False
    fsm._data.pressures = pressures.tolist()
    fsm._data.angles = angles.tolist()
    fsm._data.timestamps = timestamps.tolist()
    fsm._data.ai_values = [0] * n_points
    fsm._data.positions = [0] * n_points


def _seed_no_bottle(fsm: CabinFSM, n_points: int = 30) -> None:
    """Seed FSM with low-pressure data simulating an empty position."""
    angles = np.linspace(116.0, 270.0, n_points)
    pressures = np.zeros(n_points) + np.random.uniform(-1, 1, n_points)
    ts0 = time.time()
    fsm._state = CycleState.PROCESSING
    fsm._data.start_time = ts0
    fsm._data.end_angle = float(angles[-1])
    fsm._data.cycle_profile_id = fsm._profile.profile_id
    fsm._data.leak_valve_status = False
    fsm._data.pressures = pressures.tolist()
    fsm._data.angles = angles.tolist()
    fsm._data.timestamps = (ts0 + np.arange(n_points) * 0.1).tolist()
    fsm._data.ai_values = [0] * n_points
    fsm._data.positions = [0] * n_points


def _train_tiny_m1(tmp_path: Path) -> Path:
    """Write a small m1_coefficients.json covering cabins 1..5."""
    data = {
        "version": "test_v1",
        "primary_section": "hold",
        "feature": "hold_trend_slope",
        "target": "Q (Pa·m³/s)",
        "cabins": {
            "1": {"beta": -1e-3, "alpha": 0.0, "u_beta": 1e-5,
                  "u_alpha": 1e-7, "n_samples": 20, "r_squared": 0.995},
            "2": {"beta": -1e-3, "alpha": 0.0, "u_beta": 1e-5,
                  "u_alpha": 1e-7, "n_samples": 20, "r_squared": 0.995},
            "3": {"beta": -1e-3, "alpha": 0.0, "u_beta": 1e-5,
                  "u_alpha": 1e-7, "n_samples": 20, "r_squared": 0.995},
            "4": {"beta": -1e-3, "alpha": 0.0, "u_beta": 1e-5,
                  "u_alpha": 1e-7, "n_samples": 20, "r_squared": 0.995},
            "5": {"beta": -1e-3, "alpha": 0.0, "u_beta": 1e-5,
                  "u_alpha": 1e-7, "n_samples": 20, "r_squared": 0.995},
        },
    }
    f = tmp_path / "m1.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    return f


@pytest.fixture
def temp_db(tmp_path):
    db = DatabaseLogger(tmp_path / "ldpj.db")
    yield db
    db.close()


@pytest.fixture
def loaded_m1(tmp_path):
    coef_path = _train_tiny_m1(tmp_path)
    cfg = {"m1": {"coefficients_path": str(coef_path.relative_to(tmp_path))}}
    m1 = LinearRegressionM1(cfg, base_dir=tmp_path)
    m1.load()
    return m1


@pytest.fixture
def unloaded_m1():
    return LinearRegressionM1({}, base_dir=".")  # no .load() call


@pytest.fixture
def unloaded_m2():
    return XGBRegressorM2({}, base_dir=".")


def _make_loop(*, m1, m2, runtime, products, cabins,
               db, fault_reporter):
    """Build a ProcessingLoop with mock collaborators where appropriate."""
    profile = _profile()
    polling = MagicMock()
    polling.drain_frames_since = MagicMock(return_value=[])
    polling.buffer_length = 0
    polling.stats = {}

    fsm_manager = CycleFSMManager(26, profile, active_start=1, active_end=5)
    result_sender = MagicMock()
    alarm_pusher = MagicMock()
    health_checker = MagicMock()
    health_checker.report_inference_latency = MagicMock()

    return ProcessingLoop(
        runtime_cfg=runtime,
        profile=profile,
        cabins_cfg=cabins,
        products_cfg=products,
        polling_engine=polling,
        fsm_manager=fsm_manager,
        m1_model=m1,
        m2_model=m2,
        db_logger=db,
        result_sender=result_sender,
        alarm_pusher=alarm_pusher,
        health_checker=health_checker,
        fault_reporter=fault_reporter,
    )


# ── Tests ────────────────────────────────────────────────────────────

class TestQJudgment:
    def test_high_q_labels_leak(self, loaded_m1, unloaded_m2, temp_db):
        """Steep negative slope → large |dp/dt| → Q above threshold → LEAK."""
        rep = FaultReporter()
        loop = _make_loop(
            m1=loaded_m1, m2=unloaded_m2,
            runtime=_runtime_cfg(),
            products=_products_cfg(q_threshold=1e-4),  # tight threshold
            cabins=_cabins_cfg(), db=temp_db, fault_reporter=rep,
        )
        # β=1e-3, slope=-5 → q=5e-3 > 1e-4 → LEAK
        _seed_fsm_with_cycle(loop._fsm.fsms[1], slope_per_sample=-5.0)

        loop._process_cabin(1)

        loop._sender.write_result.assert_called_once()
        cabin_id, label, q = loop._sender.write_result.call_args[0]
        assert cabin_id == 1
        assert label == LABEL_LEAK
        assert q > 0
        loop._alarm.push_leak_alarm.assert_called_once()

    def test_low_q_labels_ok(self, loaded_m1, unloaded_m2, temp_db):
        """Tiny slope → Q above resolution but below product threshold → OK."""
        rep = FaultReporter()
        loop = _make_loop(
            m1=loaded_m1, m2=unloaded_m2,
            runtime=_runtime_cfg(a_resolution=1e-8),
            products=_products_cfg(q_threshold=1e-2),  # loose threshold
            cabins=_cabins_cfg(), db=temp_db, fault_reporter=rep,
        )
        # β=1e-3, slope=-0.5 → q=5e-4 (< q_threshold but > A_resolution)
        _seed_fsm_with_cycle(loop._fsm.fsms[2], slope_per_sample=-0.5)

        loop._process_cabin(2)

        cabin_id, label, q = loop._sender.write_result.call_args[0]
        assert label == LABEL_OK
        loop._alarm.push_leak_alarm.assert_not_called()

    def test_no_bottle_skips_inference_and_writeback(self, loaded_m1, unloaded_m2, temp_db):
        rep = FaultReporter()
        loop = _make_loop(
            m1=loaded_m1, m2=unloaded_m2,
            runtime=_runtime_cfg(no_bottle=50.0),
            products=_products_cfg(),
            cabins=_cabins_cfg(), db=temp_db, fault_reporter=rep,
        )
        _seed_no_bottle(loop._fsm.fsms[3])
        loop._process_cabin(3)

        # No PLC write, no alarm, but DB record exists with NO_BOTTLE label
        loop._sender.write_result.assert_not_called()
        loop._alarm.push_leak_alarm.assert_not_called()

        with temp_db._lock:
            cur = temp_db._conn.execute(
                "SELECT label FROM test_records WHERE cavity_id = 3"
            )
            label = cur.fetchone()[0]
        assert label == LABEL_NO_BOTTLE


class TestM1Unloaded:
    def test_no_m1_yields_label_na(self, unloaded_m1, unloaded_m2, temp_db):
        """Without M1, every inference yields N/A and skips write-back."""
        rep = FaultReporter()
        loop = _make_loop(
            m1=unloaded_m1, m2=unloaded_m2,
            runtime=_runtime_cfg(),
            products=_products_cfg(),
            cabins=_cabins_cfg(), db=temp_db, fault_reporter=rep,
        )
        # Constructor raises F002 since M1 isn't loaded
        assert "F002" in rep.active_faults

        _seed_fsm_with_cycle(loop._fsm.fsms[1], slope_per_sample=-1.0)
        loop._process_cabin(1)

        # No write-back, no alarm — N/A doesn't reach PLC
        loop._sender.write_result.assert_not_called()
        loop._alarm.push_leak_alarm.assert_not_called()


class TestF012BelowResolution:
    def test_q_below_a_raises_f012(self, loaded_m1, unloaded_m2, temp_db):
        rep = FaultReporter()
        loop = _make_loop(
            m1=loaded_m1, m2=unloaded_m2,
            runtime=_runtime_cfg(a_resolution=1.0),  # huge A so any Q is below
            products=_products_cfg(),
            cabins=_cabins_cfg(), db=temp_db, fault_reporter=rep,
        )
        _seed_fsm_with_cycle(loop._fsm.fsms[1], slope_per_sample=-0.1)
        loop._process_cabin(1)

        assert "F012" in rep.active_faults
        # No write-back when judgment is N/A
        loop._sender.write_result.assert_not_called()


class TestF011UncalibratedCabin:
    def test_cabin_not_in_m1_table_and_cabins_yaml_raises_f011(
        self, loaded_m1, unloaded_m2, temp_db
    ):
        rep = FaultReporter()
        # cabin 99 is in neither M1's table nor cabins.yaml
        loop = _make_loop(
            m1=loaded_m1, m2=unloaded_m2,
            runtime=_runtime_cfg(),
            products=_products_cfg(q_threshold=1e-4),
            cabins=_cabins_cfg(calibrated_ids=()),  # nothing calibrated
            db=temp_db, fault_reporter=rep,
        )
        # FSMManager only covers 1..5, so we have to inject directly
        from core.cycle_fsm import CabinFSM
        loop._fsm.fsms[99] = CabinFSM(99, _profile())
        _seed_fsm_with_cycle(loop._fsm.fsms[99], slope_per_sample=-1.0)

        loop._process_cabin(99)

        assert "F011" in rep.active_faults


class TestM1M2Disagreement:
    def test_disagreement_above_threshold_raises_f010(
        self, loaded_m1, temp_db, tmp_path
    ):
        """Force M2 to return a value far from M1's; verify F010."""
        rep = FaultReporter()

        # Build a stub M2 that returns a fixed Q wildly different from M1's
        m2 = MagicMock(spec=XGBRegressorM2)
        m2.loaded = True
        m2.version = "stub_v2"
        m2.predict = MagicMock(return_value={"q_est": 1.0, "valid": True})

        loop = _make_loop(
            m1=loaded_m1, m2=m2,
            runtime=_runtime_cfg(m_disagree=0.10),
            products=_products_cfg(q_threshold=1e-4),
            cabins=_cabins_cfg(), db=temp_db, fault_reporter=rep,
        )
        # M1: β=1e-3, slope=-1 → m1_q ≈ 1e-3. M2 returns 1.0. Disagreement ~1000×.
        _seed_fsm_with_cycle(loop._fsm.fsms[1], slope_per_sample=-1.0)
        loop._process_cabin(1)

        assert "F010" in rep.active_faults

    def test_no_disagreement_when_close(self, loaded_m1, temp_db):
        rep = FaultReporter()
        m2 = MagicMock(spec=XGBRegressorM2)
        m2.loaded = True
        m2.version = "stub_v2"
        # M1 will give ~1e-3; M2 returns 1.05e-3 (5% off) — within 20% threshold
        m2.predict = MagicMock(return_value={"q_est": 1.05e-3, "valid": True})

        loop = _make_loop(
            m1=loaded_m1, m2=m2,
            runtime=_runtime_cfg(m_disagree=0.20),
            products=_products_cfg(q_threshold=1e-4),
            cabins=_cabins_cfg(), db=temp_db, fault_reporter=rep,
        )
        _seed_fsm_with_cycle(loop._fsm.fsms[2], slope_per_sample=-1.0)
        loop._process_cabin(2)
        assert "F010" not in rep.active_faults


class TestProductSwitch:
    def test_set_active_product(self, loaded_m1, unloaded_m2, temp_db):
        rep = FaultReporter()
        loop = _make_loop(
            m1=loaded_m1, m2=unloaded_m2,
            runtime=_runtime_cfg(),
            products={
                "default_product_id": "TEST",
                "products": {
                    "TEST": {"q_threshold": 1e-3},
                    "P001": {"q_threshold": 5e-4},
                },
            },
            cabins=_cabins_cfg(), db=temp_db, fault_reporter=rep,
        )
        assert loop._current_product_id == "TEST"
        assert loop.set_active_product("P001") is True
        assert loop._current_product_id == "P001"

    def test_set_unknown_product_returns_false(self, loaded_m1, unloaded_m2, temp_db):
        rep = FaultReporter()
        loop = _make_loop(
            m1=loaded_m1, m2=unloaded_m2,
            runtime=_runtime_cfg(),
            products=_products_cfg(),
            cabins=_cabins_cfg(), db=temp_db, fault_reporter=rep,
        )
        assert loop.set_active_product("DOES_NOT_EXIST") is False


class TestDiagnostics:
    def test_diagnostics_reports_v26_fields(self, loaded_m1, unloaded_m2, temp_db):
        rep = FaultReporter()
        loop = _make_loop(
            m1=loaded_m1, m2=unloaded_m2,
            runtime=_runtime_cfg(),
            products=_products_cfg(),
            cabins=_cabins_cfg(), db=temp_db, fault_reporter=rep,
        )
        d = loop.get_diagnostics()
        assert d["m1_loaded"] is True
        assert d["m2_loaded"] is False
        assert d["m1_version"] == "test_v1"
        assert d["profile_id"] == "bph_13000"
        assert d["a_resolution"] == 1e-5
        assert d["current_product_id"] == "TEST"
        assert "cabin_states" in d


class TestDBPersistence:
    def test_db_record_includes_q_columns(self, loaded_m1, unloaded_m2, temp_db):
        rep = FaultReporter()
        loop = _make_loop(
            m1=loaded_m1, m2=unloaded_m2,
            runtime=_runtime_cfg(a_resolution=1e-12),  # ensure not below
            products=_products_cfg(q_threshold=1e-4),
            cabins=_cabins_cfg(), db=temp_db, fault_reporter=rep,
        )
        _seed_fsm_with_cycle(loop._fsm.fsms[1], slope_per_sample=-1.0)
        loop._process_cabin(1)

        with temp_db._lock:
            cur = temp_db._conn.execute(
                "SELECT label, q_est, q_threshold, m1_q, m2_q, "
                "m_disagreement, product_id, cycle_profile_id, quality_flags "
                "FROM test_records WHERE cavity_id = 1"
            )
            row = cur.fetchone()
        assert row is not None
        (label, q_est, q_threshold, m1_q, m2_q, disagreement,
         product_id, profile_id, quality_flags) = row
        assert q_est is not None and q_est > 0
        assert q_threshold == pytest.approx(1e-4)
        assert m1_q == pytest.approx(q_est, rel=1e-9)  # M1 is the primary
        assert m2_q is None  # M2 not loaded
        assert product_id == "TEST"
        assert profile_id == "bph_13000"
        # The _seed_fsm_with_cycle helper places ALL points in hold (angles
        # 116°–270°), so the other 5 sections legitimately have count=0
        # and their bits are set. What MUST NOT be set is the hold bit
        # itself — the hold section had 40 points, plenty for a stable
        # slope estimate, which is the whole point of this happy path.
        from core.quality_flags import QF_SHORT_HOLD, QF_DEGENERATE_INPUT
        assert quality_flags is not None
        assert not (quality_flags & QF_SHORT_HOLD)
        assert not (quality_flags & QF_DEGENERATE_INPUT)
