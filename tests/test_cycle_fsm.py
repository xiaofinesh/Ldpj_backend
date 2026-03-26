"""Unit tests for core.cycle_fsm module (v2.5 fixed-count collection)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.cycle_fsm import CabinFSM, CycleFSMManager, CycleState
from core.polling_engine import CabinFrame

def _frame(ci, p, a=0.0):
    return CabinFrame(cabin_index=ci, rt_ai=0, rt_pressure=p, rt_position=0, rt_angle=a, leak_valve_status=False, timestamp=time.time())

class TestCabinFSM:
    CFG = {"start_angle":100.0,"collection_points":12,"collection_interval_s":0.0,"collection_timeout_s":8.0}

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

    def test_end_angle_recorded(self):
        """end_angle should record the angle when collection finishes."""
        cfg = dict(self.CFG, collection_points=3)
        fsm = CabinFSM(1, cfg)
        fsm.update(_frame(1, 600, 95)); fsm.update(_frame(1, 600, 105))
        fsm.update(_frame(1, 595, 150)); fsm.update(_frame(1, 590, 200))
        assert fsm.state == CycleState.PROCESSING
        assert fsm.data.end_angle == 200.0

    def test_must_collect_all_points(self):
        """Even past 276°, must collect all target points."""
        cfg = dict(self.CFG, collection_points=5)
        fsm = CabinFSM(1, cfg)
        fsm.update(_frame(1, 600, 95)); fsm.update(_frame(1, 600, 105))
        fsm.update(_frame(1, 595, 200)); fsm.update(_frame(1, 590, 280))
        # Only 3 points so far, need 5
        assert fsm.state == CycleState.COLLECTING
        fsm.update(_frame(1, 585, 300)); fsm.update(_frame(1, 580, 320))
        assert fsm.state == CycleState.PROCESSING
        assert fsm.point_count == 5

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
