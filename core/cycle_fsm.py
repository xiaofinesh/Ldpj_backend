"""Finite State Machine for test-cycle detection per cabin.

v2.5: Fixed-count collection with angle trigger.
      - IDLE → COLLECTING: when RT_Angle crosses start_angle (rising edge)
      - COLLECTING: accumulate exactly collection_points samples (default 36)
        with minimum collection_interval_s (100ms) between samples
      - COLLECTING → PROCESSING: collected enough points
      - Backup end: angle reaches end_angle (276°) → early PROCESSING
      - Timeout → FAULT
      - Cabin[0] reserved; active range controlled by CycleFSMManager.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from core.polling_engine import CabinFrame

logger = logging.getLogger(__name__)


class CycleState(enum.Enum):
    IDLE = "IDLE"
    COLLECTING = "COLLECTING"
    PROCESSING = "PROCESSING"
    FAULT = "FAULT"


@dataclass
class CycleData:
    """Accumulated data for one test cycle."""
    pressures: List[float] = field(default_factory=list)
    angles: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    ai_values: List[int] = field(default_factory=list)
    positions: List[int] = field(default_factory=list)
    start_time: float = 0.0


class CabinFSM:
    """State machine for a single cabin (fixed-count collection).

    Parameters
    ----------
    cabin_id : int
        Cabin index (1-based, Cabin[0] is reserved).
    cfg : dict
        The ``cycle_detection`` section from ``runtime.yaml``.
    """

    def __init__(self, cabin_id: int, cfg: Dict[str, Any]):
        self.cabin_id = cabin_id
        self._state = CycleState.IDLE
        self._data = CycleData()
        self._last_angle: Optional[float] = None
        self._last_sample_ts: float = 0.0

        # ── Configuration ─────────────────────────────────────────
        self._start_angle = cfg.get("start_angle", 100.0)
        self._end_angle = cfg.get("end_angle", 276.0)
        self._target_points = cfg.get("collection_points", 36)
        self._sample_interval = cfg.get("collection_interval_s", 0.1)
        self._timeout = cfg.get("collection_timeout_s", 8.0)

    # -- public interface ----------------------------------------------------

    @property
    def state(self) -> CycleState:
        return self._state

    @property
    def data(self) -> CycleData:
        return self._data

    @property
    def point_count(self) -> int:
        return len(self._data.pressures)

    def update(self, frame: CabinFrame) -> CycleState:
        """Feed a new data point and return the (possibly updated) state."""
        angle = frame.rt_angle
        ts = frame.timestamp

        if self._state == CycleState.IDLE:
            self._handle_idle(angle, ts, frame)
        elif self._state == CycleState.COLLECTING:
            self._handle_collecting(angle, ts, frame)

        self._last_angle = angle
        return self._state

    def harvest(self) -> CycleData:
        """Return collected data. Should only be called when PROCESSING."""
        return self._data

    def reset(self) -> None:
        """Reset to IDLE and clear accumulated data."""
        self._state = CycleState.IDLE
        self._data = CycleData()
        self._last_angle = None
        self._last_sample_ts = 0.0
        logger.debug("Cabin %d FSM reset to IDLE", self.cabin_id)

    def force_fault(self, reason: str = "") -> None:
        self._state = CycleState.FAULT
        logger.warning("Cabin %d: forced FAULT (%s)", self.cabin_id, reason)

    def clear_fault(self) -> None:
        self._state = CycleState.IDLE
        self._data = CycleData()
        self._last_angle = None
        self._last_sample_ts = 0.0
        logger.info("Cabin %d: FAULT cleared -> IDLE", self.cabin_id)

    # -- internal state handlers ---------------------------------------------

    def _handle_idle(self, angle: float, ts: float, frame: CabinFrame) -> None:
        """Detect start: angle crosses start_angle upward."""
        if self._last_angle is not None:
            if self._last_angle < self._start_angle <= angle:
                self._state = CycleState.COLLECTING
                self._data = CycleData(start_time=ts)
                self._append(frame)
                self._last_sample_ts = ts
                logger.info(
                    "Cabin %d: IDLE -> COLLECTING (angle %.1f° crossed %.1f°)",
                    self.cabin_id, angle, self._start_angle,
                )

    def _handle_collecting(self, angle: float, ts: float, frame: CabinFrame) -> None:
        """Collect fixed number of samples at regular intervals."""
        elapsed = ts - self._data.start_time

        # ── Sample at interval ────────────────────────────────────
        since_last = ts - self._last_sample_ts
        if since_last >= self._sample_interval:
            self._append(frame)
            self._last_sample_ts = ts

        # ── End condition 1: reached target point count ───────────
        if len(self._data.pressures) >= self._target_points:
            self._transition_to_processing(
                f"collected {self._target_points} points"
            )
            return

        # ── End condition 2: angle reached end_angle (backup) ─────
        if angle >= self._end_angle and len(self._data.pressures) >= 2:
            self._transition_to_processing(
                f"angle {angle:.1f}° reached end {self._end_angle:.1f}° "
                f"({len(self._data.pressures)} points)"
            )
            return

        # ── Fault: timeout ────────────────────────────────────────
        if elapsed >= self._timeout:
            self._state = CycleState.FAULT
            logger.warning(
                "Cabin %d: COLLECTING -> FAULT (timeout %.1fs, %d points)",
                self.cabin_id, elapsed, len(self._data.pressures),
            )

    def _transition_to_processing(self, reason: str) -> None:
        self._state = CycleState.PROCESSING
        logger.info(
            "Cabin %d: COLLECTING -> PROCESSING (%s, %.3fs)",
            self.cabin_id, reason,
            time.time() - self._data.start_time if self._data.start_time else 0,
        )

    def _append(self, frame: CabinFrame) -> None:
        self._data.pressures.append(frame.rt_pressure)
        self._data.angles.append(frame.rt_angle)
        self._data.timestamps.append(frame.timestamp)
        self._data.ai_values.append(frame.rt_ai)
        self._data.positions.append(frame.rt_position)


class CycleFSMManager:
    """Manages FSM instances for all active cabins.

    Parameters
    ----------
    cabin_count : int
        Total cabins in PLC array (including reserved Cabin[0]).
    cycle_cfg : dict
        The ``cycle_detection`` section from ``runtime.yaml``.
    active_start : int
        First active cabin index (default 1).
    active_end : int | None
        Last active cabin index inclusive.
    """

    def __init__(
        self,
        cabin_count: int,
        cycle_cfg: Dict[str, Any],
        active_start: int = 1,
        active_end: Optional[int] = None,
    ):
        if active_end is None:
            active_end = cabin_count - 1
        self.fsms: Dict[int, CabinFSM] = {
            i: CabinFSM(i, cycle_cfg)
            for i in range(active_start, active_end + 1)
        }
        logger.info(
            "CycleFSMManager: %d FSMs for Cabin[%d]~[%d] "
            "(trigger=%.0f°, points=%d, interval=%.0fms)",
            len(self.fsms), active_start, active_end,
            cycle_cfg.get("start_angle", 100.0),
            cycle_cfg.get("collection_points", 36),
            cycle_cfg.get("collection_interval_s", 0.1) * 1000,
        )

    def get_processing_cabins(self) -> List[int]:
        return [cid for cid, fsm in self.fsms.items() if fsm.state == CycleState.PROCESSING]

    def get_fault_cabins(self) -> List[int]:
        return [cid for cid, fsm in self.fsms.items() if fsm.state == CycleState.FAULT]

    def update_all(self, cabin_frames: Dict[int, CabinFrame]) -> None:
        for cid, frame in cabin_frames.items():
            if cid in self.fsms:
                self.fsms[cid].update(frame)
