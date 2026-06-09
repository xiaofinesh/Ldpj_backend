"""Polling engine – reads PLC data at high frequency.

v2.5: MockS7Connection generates realistic pressure/angle data
      based on cleaned_data.csv statistical distributions.
      - OK cabins: pressure ~634, nearly flat (slope ~ -0.05)
      - LEAK cabins: pressure starts ~634, drops to ~618 (slope ~ -0.45)
      - NO_BOTTLE cabins: pressure ~0 (max < 50)
      - Angle: increments ~0.51° per 10ms tick (51°/s, matching real device)
"""

from __future__ import annotations

import logging
import math
import random
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

from core.exceptions import PLCConnectionError, PLCReadError

logger = logging.getLogger(__name__)


@dataclass
class CabinFrame:
    cabin_index: int
    rt_ai: int
    rt_pressure: float
    rt_position: int
    rt_angle: float
    leak_valve_status: bool
    timestamp: float           # wall-clock (time.time()) — for DB persistence
    monotonic: float = 0.0     # time.monotonic() — for FSM scheduling, NTP-jump safe


@dataclass
class PollFrame:
    timestamp: float           # wall-clock
    cabins: List[CabinFrame] = field(default_factory=list)
    seq: int = 0               # monotonically increasing across the engine's lifetime
    monotonic: float = 0.0     # time.monotonic() captured at the same call as `timestamp`


# ---------------------------------------------------------------------------
# S7 Connection wrapper
# ---------------------------------------------------------------------------

class S7Connection:
    """Thread-safe S7 connection with built-in lock.
    
    All db_read/db_write calls are serialized through a single lock,
    preventing snap7 'Job pending' errors when polling and write-back
    happen concurrently from different threads.
    """

    def __init__(self, ip: str, rack: int = 0, slot: int = 1):
        self._ip = ip
        self._rack = rack
        self._slot = slot
        self._client: Any = None
        self._connected = False
        self._io_lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        with self._io_lock:
            try:
                import snap7
                self._client = snap7.client.Client()
                self._client.connect(self._ip, self._rack, self._slot)
                self._connected = True
                logger.info("S7 connected to %s (rack=%d, slot=%d)", self._ip, self._rack, self._slot)
            except Exception as exc:
                self._connected = False
                raise PLCConnectionError(f"S7 connect failed: {exc}") from exc

    def disconnect(self) -> None:
        with self._io_lock:
            if self._client:
                try:
                    self._client.disconnect()
                except Exception:
                    pass
            self._connected = False

    def db_read(self, db_number: int, start: int, size: int) -> bytearray:
        with self._io_lock:
            try:
                return bytearray(self._client.db_read(db_number, start, size))
            except Exception as exc:
                self._connected = False
                raise PLCReadError(f"db_read failed: {exc}") from exc

    def db_write(self, db_number: int, start: int, data: bytearray) -> None:
        with self._io_lock:
            try:
                self._client.db_write(db_number, start, data)
            except Exception as exc:
                raise PLCReadError(f"db_write failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Mock S7 Connection — realistic data simulation
# ---------------------------------------------------------------------------

class MockS7Connection:
    """Generates realistic 20-byte CabinParam frames.

    Simulates a rotating machine with 26 cabins (Cabin[0] reserved).
    Each cabin cycles through 0→360° angle. When angle crosses ~100°,
    the downstream FSM will start collecting data.

    Cabin types (configurable):
      - OK:        pressure ~634 mbar, nearly flat
      - LEAK:      pressure starts ~634, drops ~0.45 mbar/sample-point (total ~16 over 36 points)
      - NO_BOTTLE: pressure near 0
    """

    # Distributions from cleaned_data.csv analysis
    # NOTE: slopes are per-sample-point (100ms). Since db_read is called
    # every tick (10ms), we divide by 10 to get per-tick slope.
    OK_PRESSURE_BASE = 634.0
    OK_PRESSURE_STD = 20.0
    OK_SLOPE_PER_TICK = -0.005       # -0.05 per sample / 10 ticks
    OK_NOISE = 1.5

    LEAK_PRESSURE_BASE = 634.0
    LEAK_PRESSURE_STD = 18.0
    LEAK_SLOPE_PER_TICK = -0.045     # -0.45 per sample / 10 ticks
    LEAK_NOISE = 2.0

    NO_BOTTLE_PRESSURE = 2.0
    NO_BOTTLE_NOISE = 1.5

    def __init__(self, cabin_count: int = 26, cabin_size: int = 20):
        self._cabin_count = cabin_count
        self._cabin_size = cabin_size
        self._connected = True
        self._tick = 0

        # Angle state per cabin (each cabin offset by 14.4° = 360/25)
        self._angles = [0.0] * cabin_count
        for i in range(1, cabin_count):
            self._angles[i] = (i * 14.4) % 360.0

        # Cabin type assignment:
        # odd cabins = LEAK, even cabins = OK, cabin 0 = NO_BOTTLE
        # Every 5th cabin = NO_BOTTLE (simulates empty positions)
        self._cabin_types: List[str] = ["NO_BOTTLE"]  # Cabin[0]
        for i in range(1, cabin_count):
            if i % 5 == 0:
                self._cabin_types.append("NO_BOTTLE")
            elif i % 2 == 0:
                self._cabin_types.append("LEAK")
            else:
                self._cabin_types.append("OK")

        # Per-cabin pressure base (varies per cycle)
        self._pressure_bases = [0.0] * cabin_count
        self._cycle_point = [0] * cabin_count
        self._in_test = [False] * cabin_count

        self._reset_cycle_params()

        # Backing store for write read-back
        self._db: Dict[int, bytearray] = {}

        logger.info("MockPLC: %d cabins, types=%s",
                     cabin_count, {t: self._cabin_types.count(t) for t in set(self._cabin_types)})

    def _reset_cycle_params(self) -> None:
        for i in range(self._cabin_count):
            ct = self._cabin_types[i]
            if ct == "OK":
                self._pressure_bases[i] = self.OK_PRESSURE_BASE + random.gauss(0, self.OK_PRESSURE_STD)
            elif ct == "LEAK":
                self._pressure_bases[i] = self.LEAK_PRESSURE_BASE + random.gauss(0, self.LEAK_PRESSURE_STD)
            else:
                self._pressure_bases[i] = 0.0
            self._cycle_point[i] = 0
            self._in_test[i] = False

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True
        logger.info("MockPLC connected (cabin_count=%d)", self._cabin_count)

    def disconnect(self) -> None:
        self._connected = False

    def db_read(self, db_number: int, start: int, size: int) -> bytearray:
        self._tick += 1

        # Small reads: return from backing store (for bool read-modify-write)
        if size < self._cabin_size and db_number in self._db:
            stored = self._db[db_number]
            if start + size <= len(stored):
                return bytearray(stored[start: start + size])
            return bytearray(size)

        # Full array read: generate data
        buf = bytearray()
        for i in range(self._cabin_count):
            # Advance angle (~0.51°/tick at 10ms poll = 51°/s, matching real device)
            # Real device: ~178° in 3.5s = 50.8°/s → 0.508°/tick at 10ms
            self._angles[i] = (self._angles[i] + 0.51 + random.uniform(-0.03, 0.03)) % 360.0
            angle = self._angles[i]

            # Detect test zone — match v2.6.1 hold section [90°, 290°)
            if 90.0 <= angle < 290.0:
                if not self._in_test[i]:
                    self._in_test[i] = True
                    self._cycle_point[i] = 0
                    # New cycle: randomize pressure base
                    ct = self._cabin_types[i]
                    if ct == "OK":
                        self._pressure_bases[i] = self.OK_PRESSURE_BASE + random.gauss(0, self.OK_PRESSURE_STD)
                    elif ct == "LEAK":
                        self._pressure_bases[i] = self.LEAK_PRESSURE_BASE + random.gauss(0, self.LEAK_PRESSURE_STD)
                self._cycle_point[i] += 1
            else:
                self._in_test[i] = False

            # Generate pressure
            ct = self._cabin_types[i]
            cp = self._cycle_point[i]
            if ct == "NO_BOTTLE":
                pressure = max(0.0, self.NO_BOTTLE_PRESSURE + random.gauss(0, self.NO_BOTTLE_NOISE))
            elif ct == "LEAK":
                pressure = self._pressure_bases[i] + self.LEAK_SLOPE_PER_TICK * cp + random.gauss(0, self.LEAK_NOISE)
            else:  # OK
                pressure = self._pressure_bases[i] + self.OK_SLOPE_PER_TICK * cp + random.gauss(0, self.OK_NOISE)

            # Approximate AI channel value, clamped to signed-16-bit range so a
            # rare high-σ pressure outlier can't raise struct.error on ">h".
            ai = max(-32768, min(32767, int(pressure * 42.5)))
            position = int(angle / 14.4)

            # +12.0: leakValveStatus — LEAK cabins have their verification valve open
            bool_12 = 0x01 if ct == "LEAK" else 0x00

            # 26-byte CabinParam (v3 / DB_Global 2024)
            buf += struct.pack(">h", ai)            # +0  RT_AI
            buf += struct.pack(">f", pressure)      # +2  RT_Pressure
            buf += struct.pack(">h", position)      # +6  RT_Position
            buf += struct.pack(">f", angle)         # +8  RT_Angle
            buf += bytes([bool_12, 0x00])           # +12 bool flags (bit0 valve) + pad(+13)
            buf += struct.pack(">f", 0.0)           # +14 leakRate (edge writes)
            buf += struct.pack(">f", 0.0)           # +18 leakHoleDiameter (edge writes)
            buf += struct.pack(">f", 0.0)           # +22 cabinHealthStatus

        return buf

    def db_write(self, db_number: int, start: int, data: bytearray) -> None:
        if db_number not in self._db:
            self._db[db_number] = bytearray(1024)
        stored = self._db[db_number]
        if start + len(data) > len(stored):
            stored.extend(bytearray(start + len(data) - len(stored)))
        stored[start: start + len(data)] = data
        logger.debug("MockPLC db_write db=%d start=%d len=%d", db_number, start, len(data))


# ---------------------------------------------------------------------------
# Polling engine
# ---------------------------------------------------------------------------

class PollingEngine:
    def __init__(self, plc_cfg: Dict[str, Any], mode: str = "mock"):
        conn_cfg = plc_cfg.get("connection", {})
        poll_cfg = plc_cfg.get("polling", {})
        cabin_cfg = plc_cfg.get("cabin_array", {})

        self._interval = poll_cfg.get("interval_ms", 10) / 1000.0
        self._buffer_size = poll_cfg.get("buffer_size", 10000)
        self._db_number = cabin_cfg.get("db_number", 9)
        self._start_offset = cabin_cfg.get("start_offset", 0)
        self._cabin_count = cabin_cfg.get("cabin_count", 26)
        self._cabin_size = cabin_cfg.get("cabin_size_bytes", 20)
        self._reconnect_interval = conn_cfg.get("reconnect_interval_s", 5)
        self._total_read_size = self._cabin_count * self._cabin_size

        if mode == "s7":
            self._conn = S7Connection(
                ip=conn_cfg.get("ip", "192.168.0.10"),
                rack=conn_cfg.get("rack", 0),
                slot=conn_cfg.get("slot", 1))
        else:
            self._conn = MockS7Connection(self._cabin_count, self._cabin_size)

        self._buffer: Deque[PollFrame] = deque(maxlen=self._buffer_size)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False
        # Set by stop() to make the reconnect wait interruptible — otherwise an
        # in-flight _try_reconnect would sleep the full interval and then
        # re-open a live PLC connection AFTER stop() already disconnected.
        self._stop_event = threading.Event()
        self._next_seq = 0   # monotonic counter assigned to PollFrame.seq
        self._stats = {"total_polls": 0, "errors": 0, "reconnects": 0,
                       "behind_ticks": 0}

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def plc_connected(self) -> bool:
        return self._conn.connected

    @property
    def connection(self):
        """Public accessor for the underlying S7 / Mock connection.

        ResultSender (and any other PLC writer) uses this for db_write.
        Replaces ``polling_engine._conn`` private access.
        """
        return self._conn

    @property
    def buffer_length(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def get_latest_frame(self) -> Optional[PollFrame]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    @property
    def latest_seq(self) -> int:
        """Highest seq assigned so far (== self._next_seq - 1, or -1 if no
        frames yet). Useful for callers initializing their cursor."""
        return self._next_seq - 1

    def drain_frames_since_seq(self, last_seq: int) -> List[PollFrame]:
        """Return frames with seq > last_seq, oldest-first.

        O(K) where K is the number of new frames since the caller's last
        drain (typically 5 frames at 50 ms loop / 10 ms poll). Replaces
        the v2.5 timestamp-scan that walked the entire deque (10–25k frames).
        """
        with self._lock:
            if not self._buffer:
                return []
            # If the caller is fully caught up, exit fast.
            if self._buffer[-1].seq <= last_seq:
                return []
            # Walk backwards from the newest until we find one already seen,
            # then reverse to deliver in oldest-first order.
            result: List[PollFrame] = []
            for frame in reversed(self._buffer):
                if frame.seq <= last_seq:
                    break
                result.append(frame)
            result.reverse()
            return result

    def drain_frames_since(self, since_ts: float) -> List[PollFrame]:
        """[Deprecated] Timestamp-based drain. Kept for any external caller;
        new code should use ``drain_frames_since_seq``.
        """
        with self._lock:
            return [f for f in self._buffer if f.timestamp > since_ts]

    def start(self) -> None:
        if self._running:
            return
        self._stop_event.clear()
        self._conn.connect()
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="plc-poller")
        self._thread.start()
        logger.info("PollingEngine started (interval=%dms)", int(self._interval * 1000))

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()   # abort any in-flight reconnect wait
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._conn.disconnect()
        logger.info("PollingEngine stopped")

    def _poll_loop(self) -> None:
        """Tick-aligned polling loop.

        Uses ``time.monotonic()`` and a target-tick clock so the actual
        period equals the configured interval regardless of how long
        ``db_read`` takes. ``time.sleep(interval)`` after IO would
        accumulate the IO time as drift (each cycle ends up slower than
        the previous), pushing the effective polling rate well below the
        configured frequency on real S7 connections (typical RTT 5–15 ms
        per read on a 10 ms target → effective rate halved).
        """
        next_tick = time.monotonic()
        while self._running:
            try:
                raw = self._conn.db_read(
                    self._db_number, self._start_offset, self._total_read_size,
                )
                frame = self._parse_frame(raw)
                frame.seq = self._next_seq
                self._next_seq += 1
                with self._lock:
                    self._buffer.append(frame)
                self._stats["total_polls"] += 1
            except PLCReadError as exc:
                self._stats["errors"] += 1
                logger.warning("Poll error: %s", exc)
                self._try_reconnect()
                next_tick = time.monotonic()  # resync after reconnect window
                continue
            except Exception as exc:
                self._stats["errors"] += 1
                logger.error("Unexpected poll error: %s", exc, exc_info=True)

            next_tick += self._interval
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                # Behind schedule. If we've fallen far behind (> 100 ms),
                # log it and reset the clock so we don't burn the CPU
                # trying to "catch up" by issuing a flood of reads back-to-back.
                self._stats["behind_ticks"] += 1
                if delay < -0.1:
                    logger.warning("Polling fell behind by %.0f ms; resyncing",
                                   -delay * 1000)
                    next_tick = time.monotonic()

    def _try_reconnect(self) -> None:
        # Interruptible: wait() returns True immediately if stop() fired, so we
        # neither sleep the full interval nor reconnect during/after shutdown.
        if self._stop_event.wait(self._reconnect_interval):
            return
        if not self._running:
            return
        try:
            self._conn.connect()
            self._stats["reconnects"] += 1
            logger.info("PLC reconnected successfully")
        except PLCConnectionError:
            pass

    def _parse_frame(self, raw: bytearray) -> PollFrame:
        # Capture both wall-clock (for DB) and monotonic (for FSM scheduling).
        # FSM uses monotonic so NTP slewing / jumps don't bork mid-cycle
        # collection — the wall-clock can be discontinuous, monotonic isn't.
        ts = time.time()
        mono = time.monotonic()
        cabins: List[CabinFrame] = []
        for i in range(self._cabin_count):
            offset = i * self._cabin_size
            chunk = raw[offset: offset + self._cabin_size]
            if len(chunk) < self._cabin_size:
                break
            # +12.0: leakValveStatus (v3 layout — moved from +18 in v2.x)
            bool_byte_12 = chunk[12] if len(chunk) > 12 else 0
            leak_valve_status = bool(bool_byte_12 & 0x01)  # bit 0
            cabins.append(CabinFrame(
                cabin_index=i,
                rt_ai=struct.unpack_from(">h", chunk, 0)[0],
                rt_pressure=struct.unpack_from(">f", chunk, 2)[0],
                rt_position=struct.unpack_from(">h", chunk, 6)[0],
                rt_angle=struct.unpack_from(">f", chunk, 8)[0],
                leak_valve_status=leak_valve_status,
                timestamp=ts,
                monotonic=mono,
            ))
        return PollFrame(timestamp=ts, cabins=cabins, monotonic=mono)
