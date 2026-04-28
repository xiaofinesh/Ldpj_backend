"""Unit tests for v2.6 cycle FSM."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.cycle_fsm import CabinFSM, CycleFSMManager, CycleState
from core.cycle_profile import CycleProfile
from core.polling_engine import CabinFrame


def _make_profile(*, trigger_angle=0.0, collection_points=10,
                  collection_interval_s=0.0, collection_timeout_s=8.0):
    """Build a small CycleProfile suitable for tests."""
    return CycleProfile(
        profile_id="test",
        bph=13000,
        cycle_total_ms=6900,
        sections={
            "baseline_pre":  (0.0, 57.6),
            "evac":          (57.6, 93.0),
            "stable":        (93.0, 115.0),
            "hold":          (115.0, 273.6),
            "release":       (273.6, 302.4),
            "baseline_post": (302.4, 360.0),
        },
        trigger_angle=trigger_angle,
        collection_points=collection_points,
        collection_interval_s=collection_interval_s,
        collection_timeout_s=collection_timeout_s,
        primary_section="hold",
    )


def _frame(ci, p, a=0.0, ts=None):
    return CabinFrame(
        cabin_index=ci, rt_ai=0, rt_pressure=p,
        rt_position=0, rt_angle=a,
        leak_valve_status=False,
        timestamp=ts if ts is not None else time.time(),
    )


class TestTriggerLogic:
    def test_initial_state_idle(self):
        fsm = CabinFSM(1, _make_profile())
        assert fsm.state == CycleState.IDLE

    def test_trigger_zero_via_wrap_around(self):
        """trigger_angle=0° must fire on 358° -> 2° transition."""
        fsm = CabinFSM(1, _make_profile(trigger_angle=0.0))
        fsm.update(_frame(1, 0, 358.0))
        assert fsm.state == CycleState.IDLE
        fsm.update(_frame(1, 0, 2.0))
        assert fsm.state == CycleState.COLLECTING

    def test_trigger_zero_does_not_fire_mid_cycle(self):
        """Updates from middle of cycle should NOT trigger when trigger_angle=0°."""
        fsm = CabinFSM(1, _make_profile(trigger_angle=0.0))
        fsm.update(_frame(1, 600, 150.0))
        fsm.update(_frame(1, 600, 200.0))
        assert fsm.state == CycleState.IDLE

    def test_trigger_nonzero_rising_edge(self):
        """trigger_angle=100° fires when last < 100 <= current."""
        fsm = CabinFSM(1, _make_profile(trigger_angle=100.0))
        fsm.update(_frame(1, 600, 95.0))
        fsm.update(_frame(1, 600, 105.0))
        assert fsm.state == CycleState.COLLECTING

    def test_trigger_nonzero_no_crossing(self):
        fsm = CabinFSM(1, _make_profile(trigger_angle=100.0))
        fsm.update(_frame(1, 600, 150.0))
        fsm.update(_frame(1, 600, 200.0))
        assert fsm.state == CycleState.IDLE

    def test_first_update_alone_does_not_trigger(self):
        """Need a previous angle to detect a crossing."""
        fsm = CabinFSM(1, _make_profile(trigger_angle=0.0))
        fsm.update(_frame(1, 0, 2.0))
        # last_angle was None, no crossing detected
        assert fsm.state == CycleState.IDLE


class TestCollection:
    def test_collect_to_target_points(self):
        """Reach exactly target_points and transition to PROCESSING."""
        fsm = CabinFSM(1, _make_profile(trigger_angle=0.0, collection_points=10))
        fsm.update(_frame(1, 0, 358.0))
        fsm.update(_frame(1, 0, 2.0))    # trigger -> 1 point
        for ang in [10, 50, 80, 100, 150, 200, 250, 280, 320]:
            fsm.update(_frame(1, 600, ang))
        assert fsm.state == CycleState.PROCESSING
        assert fsm.point_count == 10

    def test_records_cycle_profile_id(self):
        fsm = CabinFSM(1, _make_profile(trigger_angle=0.0))
        fsm.update(_frame(1, 0, 358.0))
        fsm.update(_frame(1, 0, 2.0))
        assert fsm.data.cycle_profile_id == "test"

    def test_records_end_angle(self):
        fsm = CabinFSM(1, _make_profile(trigger_angle=0.0, collection_points=3))
        fsm.update(_frame(1, 0, 358.0))
        fsm.update(_frame(1, 0, 2.0))      # 1 point
        fsm.update(_frame(1, 600, 50.0))   # 2
        fsm.update(_frame(1, 600, 100.0))  # 3 -> PROCESSING
        assert fsm.state == CycleState.PROCESSING
        assert fsm.data.end_angle == 100.0

    def test_records_leak_valve_status_at_start(self):
        """leak_valve_status sampled once at trigger frame."""
        fsm = CabinFSM(1, _make_profile(trigger_angle=0.0))
        fsm.update(_frame(1, 0, 358.0))
        f = CabinFrame(cabin_index=1, rt_ai=0, rt_pressure=0,
                       rt_position=0, rt_angle=2.0,
                       leak_valve_status=True, timestamp=time.time())
        fsm.update(f)
        assert fsm.data.leak_valve_status is True

    def test_interval_throttles_sampling(self):
        """When interval > 0, frames closer than interval are dropped."""
        prof = _make_profile(trigger_angle=0.0, collection_points=5,
                             collection_interval_s=0.1)
        fsm = CabinFSM(1, prof)
        t0 = 1000.0
        fsm.update(_frame(1, 0, 358.0, ts=t0))
        fsm.update(_frame(1, 600, 2.0, ts=t0 + 0.001))   # trigger -> 1 point
        # Burst at 5ms intervals — none should be sampled
        for i in range(1, 10):
            fsm.update(_frame(1, 600, 5.0 + i, ts=t0 + 0.001 + i * 0.005))
        assert fsm.point_count == 1

        # Now jump 100ms ahead — should sample one
        fsm.update(_frame(1, 600, 50.0, ts=t0 + 0.2))
        assert fsm.point_count == 2


class TestEndConditions:
    def test_wrap_back_below_threshold_does_not_end(self):
        """If wrap-back happens with < 70% of points, do NOT end (still COLLECTING)."""
        # target=10, 70% = 7. Collect only 3 then wrap-back.
        prof = _make_profile(trigger_angle=0.0, collection_points=10,
                             collection_timeout_s=100.0)
        fsm = CabinFSM(1, prof)
        fsm.update(_frame(1, 0, 358.0))
        fsm.update(_frame(1, 0, 2.0))      # 1
        fsm.update(_frame(1, 600, 50.0))   # 2
        fsm.update(_frame(1, 600, 100.0))  # 3
        # Now wrap-back: last=100, current=5 — NOT a wrap-from-330
        fsm.update(_frame(1, 600, 5.0))    # 4
        assert fsm.state == CycleState.COLLECTING

    def test_wrap_back_safety_net_above_threshold(self):
        """If wrap-back happens after >= 70% of points, accept and PROCESS."""
        prof = _make_profile(trigger_angle=0.0, collection_points=10,
                             collection_timeout_s=100.0)
        fsm = CabinFSM(1, prof)
        fsm.update(_frame(1, 0, 358.0))
        fsm.update(_frame(1, 0, 2.0))                                 # 1
        for ang in [50, 100, 150, 200, 250, 300, 340]:                # 8 total
            fsm.update(_frame(1, 600, ang))
        # 8 >= 0.7 * 10 = 7, last_angle=340 (>= 330), now wrap-back to 5
        fsm.update(_frame(1, 600, 5.0))
        assert fsm.state == CycleState.PROCESSING
        assert fsm.point_count >= 7

    def test_timeout_to_fault(self):
        """Slow data → timeout → FAULT."""
        prof = _make_profile(trigger_angle=0.0, collection_points=10,
                             collection_timeout_s=2.0)
        fsm = CabinFSM(1, prof)
        t0 = 1000.0
        fsm.update(_frame(1, 0, 358.0, ts=t0))
        fsm.update(_frame(1, 0, 2.0, ts=t0 + 0.001))
        fsm.update(_frame(1, 600, 50.0, ts=t0 + 5.0))  # 5s elapsed
        assert fsm.state == CycleState.FAULT

    def test_clear_fault_returns_to_idle(self):
        prof = _make_profile(trigger_angle=0.0, collection_timeout_s=1.0)
        fsm = CabinFSM(1, prof)
        t0 = 1000.0
        fsm.update(_frame(1, 0, 358.0, ts=t0))
        fsm.update(_frame(1, 0, 2.0, ts=t0))
        fsm.update(_frame(1, 600, 50.0, ts=t0 + 5.0))
        assert fsm.state == CycleState.FAULT
        fsm.clear_fault()
        assert fsm.state == CycleState.IDLE
        assert fsm.point_count == 0


class TestReset:
    def test_reset_clears_state(self):
        fsm = CabinFSM(1, _make_profile(trigger_angle=0.0, collection_points=3))
        fsm.update(_frame(1, 0, 358.0))
        fsm.update(_frame(1, 0, 2.0))
        fsm.update(_frame(1, 600, 50.0))
        fsm.update(_frame(1, 600, 100.0))    # PROCESSING
        assert fsm.state == CycleState.PROCESSING
        fsm.reset()
        assert fsm.state == CycleState.IDLE
        assert fsm.point_count == 0
        assert fsm.data.cycle_profile_id == ""


class TestSamplingTimingNoDrift:
    """v2.6 perf-fix D: per-frame jitter must NOT accumulate.

    The v2.5 logic (``last_sample_ts = ts; since_last >= interval``)
    compounded every late frame into the next sample's target. Across 70
    samples this became hundreds of ms of drift. The fix advances an
    absolute target (``next_target_ts += interval``).
    """

    def test_jittery_frames_do_not_accumulate(self):
        """Trigger + 4 samples at 100 ms target with +20 ms jitter each.

        Last sample's frame ts should still be the actually-jittery 0.420 s,
        not the drifted 0.520 s the v2.5 logic would produce.
        """
        profile = _make_profile(
            trigger_angle=0.0, collection_points=5,
            collection_interval_s=0.100, collection_timeout_s=10.0,
        )
        fsm = CabinFSM(1, profile)
        # Trigger at ts=10.0
        fsm.update(_frame(1, 600, 358.0, ts=10.0))
        fsm.update(_frame(1, 600, 2.0, ts=10.0))  # sample 0 at trigger
        # Each subsequent frame is +20 ms past its ideal target slot.
        for off in [0.120, 0.220, 0.320, 0.420]:
            fsm.update(_frame(1, 600, 100.0 + off * 100, ts=10.0 + off))
        assert fsm.point_count == 5
        assert fsm.state == CycleState.PROCESSING
        # Drift-free: last ts is the actually-jittery 10.420, not 10.520
        assert fsm.data.timestamps[-1] == pytest.approx(10.420, abs=1e-3)


class TestCycleFSMManager:
    def test_active_range(self):
        mgr = CycleFSMManager(26, _make_profile(),
                              active_start=1, active_end=25)
        assert len(mgr.fsms) == 25
        assert 0 not in mgr.fsms
        assert 25 in mgr.fsms

    def test_default_active_end(self):
        mgr = CycleFSMManager(26, _make_profile(), active_start=1)
        # Default: cabin_count - 1
        assert max(mgr.fsms.keys()) == 25
