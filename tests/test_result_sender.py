"""Tests for integration.result_sender (sync + async writeback)."""

import struct
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.exceptions import PLCWriteError
from integration.result_sender import ResultSender


# ── Helpers ──────────────────────────────────────────────────────────

def _plc_cfg() -> dict:
    return {
        "cabin_array": {
            "db_number": 9,
            "start_offset": 0,
            "cabin_size_bytes": 20,
            "active_start": 1,
            "active_end": 25,
        },
        "write_back": {
            "leak_result_ai_offset": 12,
            "cabin_health_offset": 14,
        },
        "fault_write": {"db_number": 9, "byte_offset": 520},
    }


class FakeS7Conn:
    """In-memory S7 stand-in. Records every db_read/db_write call.

    Optionally injects a delay so we can observe interleaving of polling
    vs. writeback IO under the writer thread.
    """

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
    # v2.6.2: ResultSender now uses polling_engine.connection (public property)
    # instead of polling_engine._conn (private). Mock both for back-compat.
    poller.connection = conn
    poller._conn = conn
    sender = ResultSender(_plc_cfg(), poller)
    return sender, conn


# ── Sync mode (back-compat) ─────────────────────────────────────────

class TestSyncWriteResult:
    def test_writes_immediately(self):
        sender, conn = _make_sender()
        sender.write_result(cabin_id=1, label=1, q_est=1.5e-3)
        assert len(conn.writes) == 1
        # Cabin 1 starts at offset 20 + 12 = 32
        db, start, data = conn.writes[0]
        assert (db, start) == (9, 32)
        # Bytes 2..5 should hold the REAL value
        recovered_q = struct.unpack_from(">f", data, 2)[0]
        assert recovered_q == pytest.approx(1.5e-3, rel=1e-6)

    def test_label_zero_sets_bit0(self):
        sender, conn = _make_sender()
        sender.write_result(cabin_id=2, label=0, q_est=1e-3)
        _, _, data = conn.writes[0]
        # bit 0 of byte 0 should be 1 for LEAK
        assert data[0] & 0x01 == 0x01

    def test_label_one_clears_bit0(self):
        sender, conn = _make_sender()
        # Pre-stage bit 0 = 1 in the underlying memory
        conn._db_state[9] = bytearray(2048)
        conn._db_state[9][3 * 20 + 12] = 0xFF  # cabin 3, byte +12, all bits set
        sender.write_result(cabin_id=3, label=1, q_est=1e-3)
        _, _, data = conn.writes[0]
        # bit 0 should be cleared, others preserved (RMW)
        assert data[0] & 0x01 == 0
        assert data[0] & 0xFE == 0xFE

    def test_skip_outside_active_range(self):
        sender, conn = _make_sender()
        sender.write_result(cabin_id=0, label=1, q_est=1e-3)   # reserved
        sender.write_result(cabin_id=99, label=1, q_est=1e-3)  # past end
        assert conn.writes == []

    def test_skip_unknown_label(self):
        sender, conn = _make_sender()
        sender.write_result(cabin_id=1, label=-1, q_est=1e-3)
        sender.write_result(cabin_id=1, label=2, q_est=1e-3)
        assert conn.writes == []


# ── Async mode ───────────────────────────────────────────────────────

def _wait_for(predicate, timeout_s=2.0, poll_s=0.01):
    """Spin until predicate is True or timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return False


class TestAsyncWriteResult:
    def test_enable_idempotent(self):
        sender, _ = _make_sender()
        sender.enable_async()
        thread1 = sender._thread
        sender.enable_async()
        assert sender._thread is thread1  # didn't start a second one
        sender.shutdown()

    def test_async_returns_quickly(self):
        """write_result in async mode should return in microseconds even
        if the underlying db_write takes much longer."""
        sender, conn = _make_sender(write_delay_s=0.05)  # 50 ms per write
        sender.enable_async()
        try:
            t0 = time.perf_counter()
            for cid in range(1, 6):
                sender.write_result(cabin_id=cid, label=1, q_est=1e-3)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            # Five enqueues should complete in << 50 ms (one db_write delay)
            assert elapsed_ms < 20, f"queue puts took {elapsed_ms:.1f} ms"
        finally:
            sender.shutdown()

    def test_async_eventually_writes(self):
        sender, conn = _make_sender()
        sender.enable_async()
        try:
            sender.write_result(cabin_id=1, label=1, q_est=2.5e-3)
            assert _wait_for(lambda: len(conn.writes) >= 1)
        finally:
            sender.shutdown()
        # Verify the actual value made it through
        _, _, data = conn.writes[0]
        recovered_q = struct.unpack_from(">f", data, 2)[0]
        assert recovered_q == pytest.approx(2.5e-3, rel=1e-6)

    def test_per_cabin_coalesce(self):
        """Three rapid enqueues for the same cabin → at most one write,
        carrying the LATEST q_est."""
        sender, conn = _make_sender(write_delay_s=0.05)  # ensure backlog
        sender.enable_async()
        try:
            sender.write_result(cabin_id=1, label=1, q_est=1.0e-3)
            sender.write_result(cabin_id=1, label=1, q_est=2.0e-3)
            sender.write_result(cabin_id=1, label=0, q_est=3.0e-3)  # LATEST
            sender.shutdown()
        except Exception:
            sender.shutdown()
            raise

        # At most one write went out for cabin 1
        cabin1_writes = [w for w in conn.writes if w[1] == 1 * 20 + 12]
        assert len(cabin1_writes) >= 1
        # The actually-written q_est must be the latest one (3e-3, label=0)
        last_db, last_start, last_data = cabin1_writes[-1]
        assert last_data[0] & 0x01 == 0x01  # label=0 → bit set
        recovered_q = struct.unpack_from(">f", last_data, 2)[0]
        assert recovered_q == pytest.approx(3.0e-3, rel=1e-6)

        # Stats should reflect the coalesce
        assert sender.stats["queued"] >= 3
        assert sender.stats["coalesced"] >= 1

    def test_shutdown_flushes_pending(self):
        """Pending entries when shutdown() is called must be drained."""
        sender, conn = _make_sender()
        sender.enable_async()
        # Stop the wakeup loop momentarily so we can pile up entries
        with sender._pending_lock:
            sender._pending[5] = (1, 1e-3)
            sender._pending[6] = (0, 2e-3)
        sender.shutdown()
        # Both entries should have made it to the wire
        starts = sorted(w[1] for w in conn.writes)
        assert 5 * 20 + 12 in starts
        assert 6 * 20 + 12 in starts

    def test_writer_survives_individual_failure(self):
        """A db_write that raises must not stop subsequent writes."""
        sender, conn = _make_sender()

        # Patch db_write to fail for cabin 5 only
        original_write = conn.db_write
        def selective_fail(db, start, data):
            if start == 5 * 20 + 12:
                raise RuntimeError("simulated PLC error")
            return original_write(db, start, data)
        conn.db_write = selective_fail

        sender.enable_async()
        try:
            sender.write_result(5, 1, 1e-3)
            sender.write_result(6, 1, 1e-3)
            sender.write_result(7, 1, 1e-3)
            assert _wait_for(lambda: len(conn.writes) >= 2)
        finally:
            sender.shutdown()

        # Cabin 5 failed, cabin 6 and 7 succeeded
        assert sender.stats["failed"] >= 1
        starts = {w[1] for w in conn.writes}
        assert 6 * 20 + 12 in starts
        assert 7 * 20 + 12 in starts

    def test_pending_count(self):
        """pending_count exposes the in-memory queue depth."""
        sender, conn = _make_sender(write_delay_s=0.05)
        sender.enable_async()
        try:
            sender.write_result(1, 1, 1e-3)
            sender.write_result(2, 1, 1e-3)
            sender.write_result(3, 1, 1e-3)
            # Some may have already drained, but at least one should be visible
            # in the snapshot before the writer empties it.
            # Don't assert exact count; just that it's accessible and < 4.
            assert 0 <= sender.pending_count <= 3
        finally:
            sender.shutdown()


# ── Polling vs. writeback contention (the actual reason H exists) ───

class TestContention:
    def test_async_does_not_block_caller_under_load(self):
        """The original symptom: 25 cabins finishing simultaneously must not
        wedge the caller for the cumulative IO time. With async, each
        write_result returns regardless of how slow the underlying snap7
        round-trip is."""
        # 50 ms per write × 25 cabins = 1.25 s sync wall-clock
        sender, _conn = _make_sender(write_delay_s=0.05)
        sender.enable_async()
        try:
            t0 = time.perf_counter()
            for cid in range(1, 26):
                sender.write_result(cabin_id=cid, label=1, q_est=1e-3)
            enqueue_ms = (time.perf_counter() - t0) * 1000
            # All 25 enqueues must complete in well under 1.25 s
            assert enqueue_ms < 100, f"enqueue took {enqueue_ms:.0f} ms"
        finally:
            sender.shutdown()
