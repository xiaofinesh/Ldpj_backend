"""Result sender — writes inference results and fault codes back to PLC.

Each cabin's result is written into its own CabinParam UDT within DB9:
  - leakTestResult_AI  (Bool at cabin_base + 12, bit 0): AI leak detection result
  - cabinHealthStatus  (REAL at cabin_base + 14):        v2.5 probability / v2.6 Q_est

v2.6 architecture: optional async write-back
--------------------------------------------
``write_result`` defaults to synchronous (drop-in v2.5 behavior). Calling
``enable_async()`` flips the path: ``write_result`` becomes a non-blocking
enqueue and a dedicated writer thread does the snap7 RMW.

Why this matters: when 25 cabins finish PROCESSING in the same crank
position, 25 sync RMWs through ``_io_lock`` previously blocked the
polling thread for 250–500 ms. The async path **never** blocks the
processing loop on PLC IO; the writer thread fairly contends with
polling for the lock instead of monopolizing the path.

Per-cabin coalesce: pending writes are stored in a ``dict`` keyed by
``cabin_id``; queueing a new result for a cabin overwrites the old one.
This guarantees the PLC sees the freshest Q_est, never a stale earlier
verdict from the same cabin. Memory bound: 25 entries.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from typing import Any, Dict, Optional, Tuple

from core.exceptions import PLCWriteError

logger = logging.getLogger(__name__)


class ResultSender:
    """Writes inference results and fault codes to PLC data blocks.

    Parameters
    ----------
    plc_cfg : dict
        The full content of ``plc.yaml``.
    polling_engine : Any
        The PollingEngine instance (used to access the S7 connection).
    """

    # How long the writer thread waits between drains when no work is queued.
    _IDLE_POLL_INTERVAL_S = 0.1
    # Bound on shutdown.flush() — don't wedge the main thread on a dead PLC.
    _SHUTDOWN_FLUSH_TIMEOUT_S = 2.0

    def __init__(self, plc_cfg: Dict[str, Any], polling_engine: Any):
        self._polling_engine = polling_engine
        self._write_lock = threading.Lock()

        cabin_cfg = plc_cfg.get("cabin_array", {})
        self._db_number = cabin_cfg.get("db_number", 9)
        self._start_offset = cabin_cfg.get("start_offset", 0)
        self._cabin_size = cabin_cfg.get("cabin_size_bytes", 20)
        self._active_start = cabin_cfg.get("active_start", 1)
        self._active_end = cabin_cfg.get("active_end", 25)

        wb = plc_cfg.get("write_back", {})
        self._result_ai_offset = wb.get("leak_result_ai_offset", 12)
        self._health_offset = wb.get("cabin_health_offset", 14)

        fw = plc_cfg.get("fault_write", {})
        self._fw_db = fw.get("db_number", 9)
        self._fw_offset = fw.get("byte_offset", 520)

        # Async writeback state
        self._async_enabled = False
        self._pending: Dict[int, Tuple[int, float]] = {}     # cabin_id -> (label, q_est)
        self._pending_lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Stats for diagnostics
        self._stats = {
            "queued": 0,             # times write_result was called (async only)
            "coalesced": 0,          # times an enqueue replaced a still-pending entry
            "written": 0,
            "failed": 0,
        }

    # ── Lifecycle ─────────────────────────────────────────────────────

    def enable_async(self) -> None:
        """Switch into async writeback mode. Idempotent."""
        if self._async_enabled:
            return
        self._async_enabled = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="result-sender",
        )
        self._thread.start()
        logger.info("ResultSender: async writeback enabled")

    def shutdown(self) -> None:
        """Flush pending writes and stop the writer thread.

        Bounded by ``_SHUTDOWN_FLUSH_TIMEOUT_S`` to avoid hanging on a dead PLC.
        """
        if not self._async_enabled:
            return
        self._stop_event.set()
        self._wakeup.set()
        if self._thread is not None:
            self._thread.join(timeout=self._SHUTDOWN_FLUSH_TIMEOUT_S)
            if self._thread.is_alive():
                logger.warning("ResultSender: writer thread did not stop in time")
        self._async_enabled = False

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    @property
    def pending_count(self) -> int:
        with self._pending_lock:
            return len(self._pending)

    # ── Public API (sync + async dual-mode) ──────────────────────────

    def write_result(self, cabin_id: int, label: int, q_est: float) -> None:
        """Write the inference result to PLC.

        Sync mode (default): performs the RMW immediately, blocks until done.
        Async mode (``enable_async()`` was called): enqueues the result and
        returns in O(1). The writer thread serialises the actual snap7 IO.

        v2.6 semantic note (cabinHealthStatus REAL field):
            v2.5: probability ∈ [0, 1]
            v2.6: Q_est in Pa·m³/s
        Bytes-on-wire format unchanged; HMI display logic must match the
        backend version. See module docstring for details.
        """
        if cabin_id < self._active_start or cabin_id > self._active_end:
            logger.debug("Cabin %d: outside active range [%d..%d], skip",
                         cabin_id, self._active_start, self._active_end)
            return
        if label not in (0, 1):
            logger.debug("Cabin %d: label=%d (unknown), skip", cabin_id, label)
            return

        if self._async_enabled:
            self._enqueue(cabin_id, label, q_est)
        else:
            self._write_result_sync(cabin_id, label, q_est)

    def write_fault_code(self, plc_value: int) -> None:
        """Write a fault code integer to PLC for HMI display.

        Always synchronous — fault codes are rare and high-priority.
        """
        with self._write_lock:
            try:
                data = struct.pack(">h", plc_value)
                self._polling_engine._conn.db_write(
                    self._fw_db, self._fw_offset, bytearray(data),
                )
                logger.debug("Fault code written to PLC: %d", plc_value)
            except Exception as exc:
                logger.error("Failed to write fault code to PLC: %s", exc)
                raise PLCWriteError(f"write_fault_code failed: {exc}") from exc

    # ── Internals ─────────────────────────────────────────────────────

    def _cabin_base(self, cabin_id: int) -> int:
        return self._start_offset + cabin_id * self._cabin_size

    def _enqueue(self, cabin_id: int, label: int, q_est: float) -> None:
        """Add (or replace) a pending writeback for ``cabin_id`` and wake the writer."""
        with self._pending_lock:
            replaced = cabin_id in self._pending
            self._pending[cabin_id] = (label, q_est)
        self._stats["queued"] += 1
        if replaced:
            self._stats["coalesced"] += 1
        self._wakeup.set()

    def _drain_pending_snapshot(self) -> Dict[int, Tuple[int, float]]:
        """Atomically take the pending dict and clear it."""
        with self._pending_lock:
            snap = self._pending
            self._pending = {}
        return snap

    def _writer_loop(self) -> None:
        """Drain pending writes until shutdown is requested."""
        while not self._stop_event.is_set():
            self._wakeup.wait(timeout=self._IDLE_POLL_INTERVAL_S)
            self._wakeup.clear()
            self._flush_once()

        # Final flush on shutdown (best-effort)
        self._flush_once()

    def _flush_once(self) -> None:
        """Drain the current snapshot and write each entry synchronously.

        Failures on individual cabins are logged but do not stop the loop;
        the writer keeps going so a transient comm error on one cabin
        doesn't starve the others.
        """
        snap = self._drain_pending_snapshot()
        if not snap:
            return
        # Iterate in deterministic order so logs are readable and tests stable.
        for cabin_id in sorted(snap.keys()):
            label, q_est = snap[cabin_id]
            try:
                self._write_result_sync(cabin_id, label, q_est)
                self._stats["written"] += 1
            except PLCWriteError as exc:
                self._stats["failed"] += 1
                logger.warning("Async writeback failed for cabin %d: %s",
                               cabin_id, exc)

    def _write_result_sync(self, cabin_id: int, label: int, q_est: float) -> None:
        """Read-modify-write the cabin's 8-byte writeback block.

        See class docstring for the byte layout. This is the same logic
        v2.5 used; the only v2.6 change is the semantic of the third arg.
        """
        base = self._cabin_base(cabin_id)
        write_addr = base + self._result_ai_offset

        with self._write_lock:
            conn = self._polling_engine._conn
            try:
                current = conn.db_read(self._db_number, write_addr, 8)
                buf = bytearray(current)

                # Byte 0 (+12): set/clear bit 0 (leakTestResult_AI)
                if label == 0:
                    buf[0] = buf[0] | 0x01
                else:
                    buf[0] = buf[0] & ~0x01

                # Bytes 2..5 (+14..+17): cabinHealthStatus (REAL)
                struct.pack_into(">f", buf, 2, q_est)

                conn.db_write(self._db_number, write_addr, buf)

                logger.debug(
                    "Cabin %d: write-back OK (addr=%d, label=%d, q=%.4e)",
                    cabin_id, write_addr, label, q_est,
                )
            except Exception as exc:
                logger.error("Cabin %d: PLC write-back failed: %s", cabin_id, exc)
                raise PLCWriteError(
                    f"write_result failed for cabin {cabin_id}: {exc}"
                ) from exc
