"""Tests for integration.result_sender (v3 dual-REAL writeback: Q + d)."""

import struct
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from integration.result_sender import ResultSender

# v3 layout: 26-byte CabinParam; leakRate @+14, leakHoleDiameter @+18.
CABIN_SIZE = 26
RATE_OFF = 14


def _rate_addr(cabin_id: int) -> int:
    return cabin_id * CABIN_SIZE + RATE_OFF


def _plc_cfg() -> dict:
    return {
        "cabin_array": {
            "db_number": 9, "start_offset": 0, "cabin_size_bytes": CABIN_SIZE,
            "active_start": 1, "active_end": 25,
        },
        "write_back": {"leak_rate_offset": 14, "leak_hole_diameter_offset": 18},
        "fault_write": {"db_number": 9, "byte_offset": 676},
    }


class FakeS7Conn:
    def __init__(self, write_delay_s: float = 0.0):
        self._db_state: dict[int, bytearray] = {}
        self.reads: list[tuple[int, int, int]] = []
        self.writes: list[tuple[int, int, bytearray]] = []
        self.write_delay_s = write_delay_s
        self._lock = threading.Lock()

    def db_read(self, db, start, size):
        with self._lock:
            self.reads.append((db, start, size))
            buf = self._db_state.setdefault(db, bytearray(2048))
            return bytearray(buf[start:start + size])

    def db_write(self, db, start, data):
        if self.write_delay_s > 0:
            time.sleep(self.write_delay_s)
        with self._lock:
            self.writes.append((db, start, bytearray(data)))
            buf = self._db_state.setdefault(db, bytearray(2048))
            buf[start:start + len(data)] = data


def _make_sender(write_delay_s: float = 0.0) -> tuple[ResultSender, FakeS7Conn]:
    conn = FakeS7Conn(write_delay_s=write_delay_s)
    poller = MagicMock()
    poller.connection = conn
    poller._conn = conn
    return ResultSender(_plc_cfg(), poller), conn


def _wait_for(predicate, timeout_s=2.0, poll_s=0.01):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


# ── Sync mode ────────────────────────────────────────────────────────

class TestSyncWriteResult:
    def test_writes_both_q_and_d(self):
        sender, conn = _make_sender()
        sender.write_result(cabin_id=1, q_est=1.5e-3, d_est=42.0)
        assert len(conn.writes) == 1
        db, start, data = conn.writes[0]
        assert (db, start) == (9, _rate_addr(1))     # leakRate addr (+14)
        assert len(data) == 8                          # two contiguous REALs
        q, d = struct.unpack_from(">ff", data, 0)
        assert q == pytest.approx(1.5e-3, rel=1e-6)
        assert d == pytest.approx(42.0, rel=1e-6)

    def test_only_writes_the_8_byte_block(self):
        """No verdict bit (+12) and no health (+22) are written — just Q+d."""
        sender, conn = _make_sender()
        sender.write_result(2, 5e-4, 20.0)
        assert [s for _, s, _ in conn.writes] == [_rate_addr(2)]
        # never touches the bool byte (+12) or health (+22)
        assert all(s == _rate_addr(2) for _, s, _ in conn.writes)

    def test_skip_outside_active_range(self):
        sender, conn = _make_sender()
        sender.write_result(0, 1e-3, 10.0)    # reserved
        sender.write_result(99, 1e-3, 10.0)   # past end
        assert conn.writes == []


# ── Async mode ───────────────────────────────────────────────────────

class TestAsyncWriteResult:
    def test_enable_idempotent(self):
        sender, _ = _make_sender()
        sender.enable_async()
        t1 = sender._thread
        sender.enable_async()
        assert sender._thread is t1
        sender.shutdown()

    def test_async_returns_quickly(self):
        sender, conn = _make_sender(write_delay_s=0.05)
        sender.enable_async()
        try:
            t0 = time.perf_counter()
            for cid in range(1, 6):
                sender.write_result(cid, 1e-3, 10.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            assert elapsed_ms < 20
        finally:
            sender.shutdown()

    def test_async_eventually_writes(self):
        sender, conn = _make_sender()
        sender.enable_async()
        try:
            sender.write_result(1, 2.5e-3, 33.0)
            assert _wait_for(lambda: len(conn.writes) >= 1)
        finally:
            sender.shutdown()
        _, _, data = conn.writes[0]
        q, d = struct.unpack_from(">ff", data, 0)
        assert q == pytest.approx(2.5e-3, rel=1e-6)
        assert d == pytest.approx(33.0, rel=1e-6)

    def test_per_cabin_coalesce(self):
        sender, conn = _make_sender(write_delay_s=0.05)
        sender.enable_async()
        try:
            sender.write_result(1, 1.0e-3, 10.0)
            sender.write_result(1, 2.0e-3, 20.0)
            sender.write_result(1, 3.0e-3, 30.0)   # latest
            sender.shutdown()
        except Exception:
            sender.shutdown()
            raise
        cabin1 = [w for w in conn.writes if w[1] == _rate_addr(1)]
        assert len(cabin1) >= 1
        _, _, last = cabin1[-1]
        q, d = struct.unpack_from(">ff", last, 0)
        assert q == pytest.approx(3.0e-3, rel=1e-6)
        assert d == pytest.approx(30.0, rel=1e-6)
        assert sender.stats["queued"] >= 3
        assert sender.stats["coalesced"] >= 1

    def test_shutdown_flushes_pending(self):
        sender, conn = _make_sender()
        sender.enable_async()
        with sender._pending_lock:
            sender._pending[5] = (1e-3, 15.0)
            sender._pending[6] = (2e-3, 25.0)
        sender.shutdown()
        starts = sorted(w[1] for w in conn.writes)
        assert _rate_addr(5) in starts
        assert _rate_addr(6) in starts

    def test_writer_survives_individual_failure(self):
        sender, conn = _make_sender()
        original = conn.db_write
        def selective_fail(db, start, data):
            if start == _rate_addr(5):
                raise RuntimeError("simulated PLC error")
            return original(db, start, data)
        conn.db_write = selective_fail
        sender.enable_async()
        try:
            sender.write_result(5, 1e-3, 10.0)
            sender.write_result(6, 1e-3, 10.0)
            sender.write_result(7, 1e-3, 10.0)
            assert _wait_for(lambda: len(conn.writes) >= 2)
        finally:
            sender.shutdown()
        assert sender.stats["failed"] >= 1
        starts = {w[1] for w in conn.writes}
        assert _rate_addr(6) in starts
        assert _rate_addr(7) in starts

    def test_pending_count(self):
        sender, conn = _make_sender(write_delay_s=0.05)
        sender.enable_async()
        try:
            sender.write_result(1, 1e-3, 10.0)
            sender.write_result(2, 1e-3, 10.0)
            sender.write_result(3, 1e-3, 10.0)
            assert 0 <= sender.pending_count <= 3
        finally:
            sender.shutdown()


class TestContention:
    def test_async_does_not_block_caller_under_load(self):
        sender, _conn = _make_sender(write_delay_s=0.05)
        sender.enable_async()
        try:
            t0 = time.perf_counter()
            for cid in range(1, 26):
                sender.write_result(cid, 1e-3, 10.0)
            enqueue_ms = (time.perf_counter() - t0) * 1000
            assert enqueue_ms < 100
        finally:
            sender.shutdown()
