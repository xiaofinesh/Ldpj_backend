"""Finite State Machine for test-cycle detection per cabin.

Each cabin has its own FSM instance that transitions through:
    IDLE -> COLLECTING -> PROCESSING -> IDLE
with a FAULT state for timeout / anomaly handling.
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
    """State machine for a single cabin.

    Parameters
    ----------
    cabin_id : int
        Zero-based cabin index.
    cfg : dict
        The ``cycle_detection`` section from ``runtime.yaml``.
    """

    def __init__(self, cabin_id: int, cfg: Dict[str, Any]):
        self.cabin_id = cabin_id
        self._state = CycleState.IDLE
        self._data = CycleData()
        self._last_pressure: Optional[float] = None

        # Configurable thresholds
        self._start_drop = cfg.get("start_pressure_drop", 50.0)
        self._end_rise = cfg.get("end_pressure_rise", 50.0)
        self._min_points = cfg.get("min_collection_points", 100)
        self._max_points = cfg.get("max_collection_points", 3000)
        self._max_duration = cfg.get("max_collection_duration_s", 45)
        self._timeout = cfg.get("collection_timeout_s", 60)
        self._idle_pressure_min = cfg.get("idle_pressure_min", 900.0)

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
        pressure = frame.rt_pressure
        ts = frame.timestamp

        if self._state == CycleState.IDLE:
            self._handle_idle(pressure, ts, frame)
        elif self._state == CycleState.COLLECTING:
            self._handle_collecting(pressure, ts, frame)
        # PROCESSING and FAULT are externally managed

        self._last_pressure = pressure
        return self._state

    def harvest(self) -> CycleData:
        """Return collected data and transition to PROCESSING.

        Should only be called when state is PROCESSING.
        """
        data = self._data
        return data

    def reset(self) -> None:
        """Reset to IDLE after processing is complete."""
        self._state = CycleState.IDLE
        self._data = CycleData()
        self._last_pressure = None
        logger.debug("Cabin %d FSM reset to IDLE", self.cabin_id)

    def force_fault(self, reason: str) -> None:
        """Externally force the FSM into FAULT state."""
        self._state = CycleState.FAULT
        logger.warning("Cabin %d forced to FAULT: %s", self.cabin_id, reason)

    def clear_fault(self) -> None:
        """Clear a FAULT and return to IDLE."""
        self._state = CycleState.IDLE
        self._data = CycleData()
        self._last_pressure = None
        logger.info("Cabin %d FAULT cleared, back to IDLE", self.cabin_id)

    # -- internal transitions ------------------------------------------------

    def _handle_idle(self, pressure: float, ts: float, frame: CabinFrame) -> None:
        """Detect the start of a test cycle by a significant pressure drop."""
        if self._last_pressure is not None:
            drop = self._last_pressure - pressure
            if drop >= self._start_drop:
                self._state = CycleState.COLLECTING
                self._data = CycleData(start_time=ts)
                self._append(frame)
                logger.info(
                    "Cabin %d: IDLE -> COLLECTING (drop=%.1f mbar)",
                    self.cabin_id, drop,
                )

    def _handle_collecting(self, pressure: float, ts: float, frame: CabinFrame) -> None:
        """Accumulate data and detect end-of-cycle conditions."""
        self._append(frame)
        elapsed = ts - self._data.start_time

        # End condition 1: pressure rises back (test finished)
        if (self._last_pressure is not None
                and len(self._data.pressures) >= self._min_points):
            rise = pressure - self._last_pressure
            if rise >= self._end_rise:
                self._transition_to_processing("pressure rise detected")
                return

        # End condition 2: maximum collection points reached
        if len(self._data.pressures) >= self._max_points:
            self._transition_to_processing("max points reached")
            return

        # End condition 3: maximum collection duration reached
        if elapsed >= self._max_duration:
            self._transition_to_processing("max duration reached")
            return

        # Fault condition: timeout
        if elapsed >= self._timeout:
            self._state = CycleState.FAULT
            logger.warning(
                "Cabin %d: COLLECTING -> FAULT (timeout %.1fs)",
                self.cabin_id, elapsed,
            )

    def _transition_to_processing(self, reason: str) -> None:
        self._state = CycleState.PROCESSING
        logger.info(
            "Cabin %d: COLLECTING -> PROCESSING (%s, %d points)",
            self.cabin_id, reason, len(self._data.pressures),
        )

    def _append(self, frame: CabinFrame) -> None:
        self._data.pressures.append(frame.rt_pressure)
        self._data.angles.append(frame.rt_angle)
        self._data.timestamps.append(frame.timestamp)
        self._data.ai_values.append(frame.rt_ai)
        self._data.positions.append(frame.rt_position)


class CycleFSMManager:
    """Manages FSM instances for all cabins."""

    def __init__(self, cabin_count: int, cycle_cfg: Dict[str, Any]):
        self.fsms: Dict[int, CabinFSM] = {
            i: CabinFSM(i, cycle_cfg) for i in range(cabin_count)
        }

    def get_processing_cabins(self) -> List[int]:
        """Return cabin IDs whose FSM is in PROCESSING state."""
        return [cid for cid, fsm in self.fsms.items() if fsm.state == CycleState.PROCESSING]

    def get_fault_cabins(self) -> List[int]:
        """Return cabin IDs whose FSM is in FAULT state."""
        return [cid for cid, fsm in self.fsms.items() if fsm.state == CycleState.FAULT]

    def update_all(self, cabin_frames: Dict[int, CabinFrame]) -> None:
        """Update all FSMs with the latest cabin frames."""
        for cid, frame in cabin_frames.items():
            if cid in self.fsms:
                self.fsms[cid].update(frame)
