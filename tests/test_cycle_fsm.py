"""Unit tests for core.cycle_fsm module (v2.5 fixed-count collection)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.cycle_fsm import CabinFSM, CycleFSMManager, CycleState
from core.polling_engine import CabinFrame

def _frame(ci, p, a=0.0):
    return CabinFrame(cabin_index=ci, rt_ai=0, rt_pressure=p, rt_position=0, rt_angle=a, timestamp=time.time())

class TestCabinFSM:
    CFG = {"start_angle":100.0,"end_angle":276.0,"collection_points":36,"collection_interval_s":0.0,"collection_timeout_s":8.0}

    def test_initial_state(self):
        assert CabinFSM(1, self.CFG).state == CycleState.IDLE

    def test_trigger_on_crossing(self):
        fsm = CabinFSM(1, self.CFG)
        fsm.update(_frame(1, 600, 95)); fsm.update(_frame(1, 600, 105))
        assert fsm.state == CycleState.COLLECTING

    def test_no_trigger_without_crossing(self):
        fsm = CabinFSM(1, self.CFG)
        fsm.update(_frame(1, 600, 150)); fsm.update(_frame(1, 600, 200))
        assert fsm.state == CycleState.IDLE

    def test_fixed_count_collection(self):
        cfg = dict(self.CFG, collection_points=5)
        fsm = CabinFSM(1, cfg)
        fsm.update(_frame(1, 600, 95)); fsm.update(_frame(1, 600, 105))
        for a in [110,120,130,140]:
            fsm.update(_frame(1, 595, a))
        assert fsm.state == CycleState.PROCESSING
        assert fsm.point_count == 5

    def test_backup_end_angle(self):
        cfg = dict(self.CFG, collection_points=100)
        fsm = CabinFSM(1, cfg)
        fsm.update(_frame(1, 600, 95)); fsm.update(_frame(1, 600, 105))
        fsm.update(_frame(1, 595, 200)); fsm.update(_frame(1, 590, 280))
        assert fsm.state == CycleState.PROCESSING

    def test_harvest_reset(self):
        cfg = dict(self.CFG, collection_points=3)
        fsm = CabinFSM(1, cfg)
        fsm.update(_frame(1, 600, 95)); fsm.update(_frame(1, 600, 105))
        fsm.update(_frame(1, 595, 150)); fsm.update(_frame(1, 590, 200))
        assert fsm.state == CycleState.PROCESSING
        assert len(fsm.harvest().pressures) == 3
        fsm.reset(); assert fsm.state == CycleState.IDLE

class TestCycleFSMManager:
    def test_active_range(self):
        mgr = CycleFSMManager(26, TestCabinFSM.CFG, active_start=1, active_end=25)
        assert len(mgr.fsms) == 25
        assert 0 not in mgr.fsms
