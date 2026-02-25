"""Unit tests for core.polling_engine module (mock mode)."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.polling_engine import MockS7Connection, PollingEngine


class TestMockConnection:
    def test_connect_disconnect(self):
        conn = MockS7Connection(cabin_count=6)
        conn.connect()
        assert conn.connected
        conn.disconnect()
        assert not conn.connected

    def test_db_read(self):
        conn = MockS7Connection(cabin_count=6, cabin_size=12)
        conn.connect()
        data = conn.db_read(9, 0, 72)
        assert len(data) == 72  # 6 cabins * 12 bytes


class TestPollingEngine:
    def test_start_stop(self, plc_cfg):
        engine = PollingEngine(plc_cfg, mode="mock")
        engine.start()
        assert engine.is_running
        time.sleep(0.2)
        assert engine.buffer_length > 0
        engine.stop()
        assert not engine.is_running

    def test_get_latest_frame(self, plc_cfg):
        engine = PollingEngine(plc_cfg, mode="mock")
        engine.start()
        time.sleep(0.2)
        frame = engine.get_latest_frame()
        assert frame is not None
        assert len(frame.cabins) == 6
        engine.stop()

    def test_drain_frames(self, plc_cfg):
        engine = PollingEngine(plc_cfg, mode="mock")
        engine.start()
        ts = time.time()
        time.sleep(0.3)
        frames = engine.drain_frames_since(ts)
        assert len(frames) > 0
        engine.stop()
