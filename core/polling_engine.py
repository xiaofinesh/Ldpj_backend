"""High-frequency PLC polling engine.

Runs in a dedicated daemon thread and continuously reads the Cabin array
from DB9 at a configurable interval, storing timestamped frames in a
thread-safe ring buffer for the main processing loop to consume.
"""

from __future__ import annotations

import logging
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from core.exceptions import PLCConnectionError, PLCReadError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CabinFrame:
    """A single snapshot of one cabin's sensor readings."""
    cabin_index: int
    rt_ai: int
    rt_pressure: float
    rt_position: int
    rt_angle: float
    timestamp: float  # time.time()


@dataclass
class PollFrame:
    """A complete snapshot of all cabins at one polling instant."""
    timestamp: float
    cabins: List[CabinFrame] = field(default_factory=list)


# ---------------------------------------------------------------------------
# S7 connection wrapper (thin abstraction over snap7)
# ---------------------------------------------------------------------------

class S7Connection:
    """Manages a single snap7 TCP connection to a Siemens S7 PLC."""

    def __init__(self, ip: str, rack: int, slot: int):
        self.ip = ip
        self.rack = rack
        self.slot = slot
        self._client: Any = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        try:
            import snap7
            self._client = snap7.client.Client()
            self._client.connect(self.ip, self.rack, self.slot)
            self._connected = True
            logger.info("PLC connected: %s rack=%d slot=%d", self.ip, self.rack, self.slot)
        except Exception as exc:
            self._connected = False
            raise PLCConnectionError(f"Cannot connect to PLC {self.ip}: {exc}") from exc

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
        self._connected = False

    def db_read(self, db_number: int, start: int, size: int) -> bytearray:
        if not self._connected:
            raise PLCConnectionError("PLC not connected")
        try:
            return self._client.db_read(db_number, start, size)
        except Exception as exc:
            self._connected = False
            raise PLCReadError(f"db_read failed: {exc}") from exc

    def db_write(self, db_number: int, start: int, data: bytearray) -> None:
        if not self._connected:
            raise PLCConnectionError("PLC not connected")
        try:
            self._client.db_write(db_number, start, data)
        except Exception as exc:
            self._connected = False
            raise PLCConnectionError(f"db_write failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Mock connection for development / testing
# ---------------------------------------------------------------------------

class MockS7Connection:
    """Generates synthetic PLC data for offline development."""

    def __init__(self, cabin_count: int = 25, cabin_size: int = 12):
        self._cabin_count = cabin_count
        self._cabin_size = cabin_size
        self._connected = True
        self._tick = 0

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True
        logger.info("MockPLC connected (cabin_count=%d)", self._cabin_count)

    def disconnect(self) -> None:
        self._connected = False

    def db_read(self, db_number: int, start: int, size: int) -> bytearray:
        import random
        self._tick += 1
        buf = bytearray()
        for i in range(self._cabin_count):
            ai = i * 100 + random.randint(0, 10)
            # Simulate a pressure curve: high when idle, drops during test
            pressure = 950.0 + random.uniform(-5, 5)
            position = self._tick % 360
            angle = float(position) + random.uniform(-0.5, 0.5)
            buf += struct.pack(">h", ai)
            buf += struct.pack(">f", pressure)
            buf += struct.pack(">h", position)
            buf += struct.pack(">f", angle)
        return buf

    def db_write(self, db_number: int, start: int, data: bytearray) -> None:
        logger.debug("MockPLC db_write db=%d start=%d len=%d", db_number, start, len(data))


# ---------------------------------------------------------------------------
# Polling engine
# ---------------------------------------------------------------------------

class PollingEngine:
    """Background thread that polls PLC at a fixed interval.

    Parameters
    ----------
    plc_cfg : dict
        The full content of ``plc.yaml``.
    mode : str
        ``"s7"`` for real PLC, ``"mock"`` for synthetic data.
    """

    def __init__(self, plc_cfg: Dict[str, Any], mode: str = "mock"):
        conn_cfg = plc_cfg.get("connection", {})
        poll_cfg = plc_cfg.get("polling", {})
        cabin_cfg = plc_cfg.get("cabin_array", {})

        self._interval = poll_cfg.get("interval_ms", 10) / 1000.0
        self._buffer_size = poll_cfg.get("buffer_size", 10000)
        self._db_number = cabin_cfg.get("db_number", 9)
        self._start_offset = cabin_cfg.get("start_offset", 0)
        self._cabin_count = cabin_cfg.get("cabin_count", 25)
        self._cabin_size = cabin_cfg.get("cabin_size_bytes", 12)
        self._reconnect_interval = conn_cfg.get("reconnect_interval_s", 5)

        self._total_read_size = self._cabin_count * self._cabin_size

        if mode == "s7":
            self._conn = S7Connection(
                ip=conn_cfg.get("ip", "192.168.0.10"),
                rack=conn_cfg.get("rack", 0),
                slot=conn_cfg.get("slot", 1),
            )
        else:
            self._conn = MockS7Connection(self._cabin_count, self._cabin_size)

        # Thread-safe ring buffer
        self._buffer: Deque[PollFrame] = deque(maxlen=self._buffer_size)
        self._lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stats = {"total_polls": 0, "errors": 0, "reconnects": 0}

    # -- public properties ---------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def plc_connected(self) -> bool:
        return self._conn.connected

    @property
    def buffer_length(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    # -- buffer access -------------------------------------------------------

    def get_latest_frame(self) -> Optional[PollFrame]:
        """Return the most recent frame without removing it."""
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def drain_frames_since(self, since_ts: float) -> List[PollFrame]:
        """Return all frames with timestamp > *since_ts* (non-destructive)."""
        with self._lock:
            return [f for f in self._buffer if f.timestamp > since_ts]

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._conn.connect()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="plc-poller")
        self._thread.start()
        logger.info("PollingEngine started (interval=%.1fms, buffer=%d)", self._interval * 1000, self._buffer_size)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._conn.disconnect()
        logger.info("PollingEngine stopped")

    # -- internal ------------------------------------------------------------

    def _poll_loop(self) -> None:
        while self._running:
            t0 = time.perf_counter()
            try:
                if not self._conn.connected:
                    self._try_reconnect()
                    if not self._conn.connected:
                        time.sleep(self._reconnect_interval)
                        continue

                raw = self._conn.db_read(self._db_number, self._start_offset, self._total_read_size)
                frame = self._parse_frame(raw)
                with self._lock:
                    self._buffer.append(frame)
                self._stats["total_polls"] += 1

            except (PLCConnectionError, PLCReadError) as exc:
                self._stats["errors"] += 1
                logger.warning("Poll error: %s", exc)
                self._try_reconnect()

            except Exception as exc:
                self._stats["errors"] += 1
                logger.error("Unexpected poll error: %s", exc, exc_info=True)

            elapsed = time.perf_counter() - t0
            sleep_time = max(0, self._interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _try_reconnect(self) -> None:
        try:
            self._conn.disconnect()
            self._conn.connect()
            self._stats["reconnects"] += 1
            logger.info("PLC reconnected successfully")
        except PLCConnectionError:
            pass

    def _parse_frame(self, raw: bytearray) -> PollFrame:
        ts = time.time()
        cabins: List[CabinFrame] = []
        for i in range(self._cabin_count):
            offset = i * self._cabin_size
            chunk = raw[offset: offset + self._cabin_size]
            if len(chunk) < self._cabin_size:
                break
            rt_ai = struct.unpack_from(">h", chunk, 0)[0]
            rt_pressure = struct.unpack_from(">f", chunk, 2)[0]
            rt_position = struct.unpack_from(">h", chunk, 6)[0]
            rt_angle = struct.unpack_from(">f", chunk, 8)[0]
            cabins.append(CabinFrame(
                cabin_index=i,
                rt_ai=rt_ai,
                rt_pressure=rt_pressure,
                rt_position=rt_position,
                rt_angle=rt_angle,
                timestamp=ts,
            ))
        return PollFrame(timestamp=ts, cabins=cabins)
