"""Unit tests for core.cycle_fsm module."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.cycle_fsm import CabinFSM, CycleFSMManager, CycleState
from core.polling_engine import CabinFrame


def _frame(cabin_index: int, pressure: float, angle: float = 0.0) -> CabinFrame:
    return CabinFrame(
        cabin_index=cabin_index,
        rt_ai=0,
        rt_pressure=pressure,
        rt_position=0,
        rt_angle=angle,
        timestamp=time.time(),
    )


class TestCabinFSM:
    CFG = {
        "start_pressure_drop": 50.0,
        "end_pressure_rise": 50.0,
        "min_collection_points": 5,
        "max_collection_points": 100,
        "max_collection_duration_s": 10,
        "collection_timeout_s": 15,
        "idle_pressure_min": 900.0,
    }

    def test_initial_state(self):
        fsm = CabinFSM(0, self.CFG)
        assert fsm.state == CycleState.IDLE

    def test_idle_to_collecting(self):
        fsm = CabinFSM(0, self.CFG)
        # First frame sets baseline
        fsm.update(_frame(0, 1000.0))
        assert fsm.state == CycleState.IDLE
        # Big drop triggers COLLECTING
        fsm.update(_frame(0, 940.0))
        assert fsm.state == CycleState.COLLECTING

    def test_collecting_to_processing_on_max_points(self):
        cfg = dict(self.CFG, max_collection_points=10, min_collection_points=3)
        fsm = CabinFSM(0, cfg)
        fsm.update(_frame(0, 1000.0))
        fsm.update(_frame(0, 940.0))  # triggers COLLECTING
        for _ in range(10):
            fsm.update(_frame(0, 500.0))
        assert fsm.state == CycleState.PROCESSING

    def test_harvest_and_reset(self):
        cfg = dict(self.CFG, max_collection_points=5, min_collection_points=2)
        fsm = CabinFSM(0, cfg)
        fsm.update(_frame(0, 1000.0))
        fsm.update(_frame(0, 940.0))
        for _ in range(5):
            fsm.update(_frame(0, 500.0))
        assert fsm.state == CycleState.PROCESSING
        data = fsm.harvest()
        assert len(data.pressures) > 0
        fsm.reset()
        assert fsm.state == CycleState.IDLE

    def test_force_fault_and_clear(self):
        fsm = CabinFSM(0, self.CFG)
        fsm.force_fault("test")
        assert fsm.state == CycleState.FAULT
        fsm.clear_fault()
        assert fsm.state == CycleState.IDLE


class TestCycleFSMManager:
    def test_manager_creation(self):
        mgr = CycleFSMManager(6, TestCabinFSM.CFG)
        assert len(mgr.fsms) == 6
        assert mgr.get_processing_cabins() == []
        assert mgr.get_fault_cabins() == []
