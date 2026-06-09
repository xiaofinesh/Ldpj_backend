"""Lock-in tests for round-2 concurrency/timing/robustness fixes."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.exceptions import PLCReadError
from core.polling_engine import PollingEngine
from storage.database_logger import DatabaseLogger


class _FakeDownConn:
    """A connection that always fails db_read, and records connect/disconnect."""
    def __init__(self):
        self.connected = False
        self.connect_calls = 0
        self.disconnect_calls = 0

    def connect(self):
        self.connect_calls += 1
        self.connected = True

    def disconnect(self):
        self.disconnect_calls += 1
        self.connected = False

    def db_read(self, *a):
        raise PLCReadError("PLC down")

    def db_write(self, *a):
        pass


def test_reconnect_aborts_on_shutdown_no_resurrect():
    """stop() must abort an in-flight reconnect wait so the poll thread does
    NOT re-open a live PLC connection after stop() already disconnected."""
    cfg = {
        "connection": {"reconnect_interval_s": 0.5},
        "polling": {"interval_ms": 10, "buffer_size": 1000},
        "cabin_array": {"db_number": 9, "start_offset": 0,
                        "cabin_count": 2, "cabin_size_bytes": 20},
    }
    engine = PollingEngine(cfg, mode="mock")
    fake = _FakeDownConn()
    engine._conn = fake          # inject a perpetually-down connection

    engine.start()               # connect() #1, then db_read fails → reconnect wait(0.5)
    time.sleep(0.15)             # ensure the poll thread is inside the reconnect wait
    engine.stop()                # stop_event aborts the wait before it reconnects
    time.sleep(0.6)              # well past the 0.5s reconnect interval

    assert fake.connected is False          # not resurrected after shutdown
    assert fake.connect_calls == 1          # the aborted wait did NOT reconnect


def test_get_full_record_legacy_json_curve(tmp_path):
    """get_full_record must surface curves for legacy v2.5 rows that store the
    pressure/angle data as JSON text (no compressed BLOB)."""
    db = DatabaseLogger(tmp_path / "legacy.db")
    try:
        db._conn.execute(
            "INSERT INTO test_records "
            "(cavity_id, timestamp, pressure_data, angle_data, label) "
            "VALUES (?,?,?,?,?)",
            (1, "2026-01-01T00:00:00",
             json.dumps([1.0, 2.0, 3.0]), json.dumps([90.0, 180.0, 270.0]), 1),
        )
        db._conn.commit()
        rid = db._conn.execute("SELECT id FROM test_records").fetchone()[0]
        rec = db.get_full_record(rid)
        assert rec["pressures"] == [1.0, 2.0, 3.0]
        assert rec["angles"] == [90.0, 180.0, 270.0]
    finally:
        db.close()
