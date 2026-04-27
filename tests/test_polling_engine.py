"""Unit tests for core.polling_engine module (mock mode).

v2.5: 20-byte CabinParam (RT_AI/RT_Pressure/RT_Position/RT_Angle +
      flags + cabinHealthStatus + leakValveStatus byte).
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
        conn = MockS7Connection(cabin_count=26, cabin_size=20)
        conn.connect()
        data = conn.db_read(9, 0, 26 * 20)
        assert len(data) == 520  # 26 cabins * 20 bytes

    def test_db_read_structure(self):
        """Verify that each 20-byte chunk can be parsed."""
        import struct
        conn = MockS7Connection(cabin_count=2, cabin_size=20)
        conn.connect()
        data = conn.db_read(9, 0, 40)
        # Parse first cabin (20 bytes)
        rt_ai = struct.unpack_from(">h", data, 0)[0]
        rt_pressure = struct.unpack_from(">f", data, 2)[0]
        rt_position = struct.unpack_from(">h", data, 6)[0]
        rt_angle = struct.unpack_from(">f", data, 8)[0]
        bool_byte_12 = data[12]                                # AI flags
        cabin_health = struct.unpack_from(">f", data, 14)[0]   # REAL
        bool_byte_18 = data[18]                                # leakValveStatus
        assert isinstance(rt_ai, int)
        assert isinstance(rt_pressure, float)
        assert isinstance(rt_angle, float)
        assert isinstance(cabin_health, float)
        # flags byte is one of 0x00 / 0x01 / 0x02 / 0x03
        assert 0 <= bool_byte_12 <= 0x03
        assert 0 <= bool_byte_18 <= 0x03


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
        cabin = frame.cabins[0]
        # Verify CabinFrame schema (v2.5)
        assert hasattr(cabin, "rt_pressure")
        assert hasattr(cabin, "rt_angle")
        assert hasattr(cabin, "leak_valve_status")
        assert isinstance(cabin.leak_valve_status, bool)
        assert isinstance(cabin.rt_pressure, float)
        engine.stop()

    def test_drain_frames(self, plc_cfg):
        engine = PollingEngine(plc_cfg, mode="mock")
        engine.start()
        ts = time.time()
        time.sleep(0.3)
        frames = engine.drain_frames_since(ts)
        assert len(frames) > 0
        engine.stop()
