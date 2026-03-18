"""Unit tests for core.polling_engine module (mock mode).

v2.1 修正: 测试用例适配 18 字节 CabinParam 和新增的 3 个字段.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.polling_engine import MockS7Connection, PollingEngine


class TestMockConnection:
    def test_connect_disconnect(self):
        conn = MockS7Connection(cabin_count=26)
        conn.connect()
        assert conn.connected
        conn.disconnect()
        assert not conn.connected

    def test_db_read_size(self):
        conn = MockS7Connection(cabin_count=26, cabin_size=18)
        conn.connect()
        data = conn.db_read(9, 0, 26 * 18)
        assert len(data) == 468  # 26 cabins * 18 bytes

    def test_db_read_structure(self):
        """Verify that each 18-byte chunk can be parsed."""
        import struct
        conn = MockS7Connection(cabin_count=2, cabin_size=18)
        conn.connect()
        data = conn.db_read(9, 0, 36)
        # Parse first cabin
        rt_ai = struct.unpack_from(">h", data, 0)[0]
        rt_pressure = struct.unpack_from(">f", data, 2)[0]
        rt_position = struct.unpack_from(">h", data, 6)[0]
        rt_angle = struct.unpack_from(">f", data, 8)[0]
        bool_byte = data[12]
        health = struct.unpack_from(">f", data, 14)[0]
        assert isinstance(rt_ai, int)
        assert isinstance(rt_pressure, float)
        assert 0.0 <= health <= 1.0


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
        assert len(frame.cabins) == 26
        # Verify new fields exist
        cabin = frame.cabins[0]
        assert hasattr(cabin, "leak_result_ai")
        assert hasattr(cabin, "leak_result_plc")
        assert hasattr(cabin, "cabin_health_status")
        assert isinstance(cabin.leak_result_ai, bool)
        assert isinstance(cabin.cabin_health_status, float)
        engine.stop()

    def test_drain_frames(self, plc_cfg):
        engine = PollingEngine(plc_cfg, mode="mock")
        engine.start()
        ts = time.time()
        time.sleep(0.3)
        frames = engine.drain_frames_since(ts)
        assert len(frames) > 0
        engine.stop()
