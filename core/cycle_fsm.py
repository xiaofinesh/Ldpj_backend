"""Finite State Machine for full-cycle data collection (v2.6.1).

Design highlights:
- Driven by CycleProfile (typed object) rather than a flat dict.
- Trigger at angle crossing trigger_angle; trigger_angle=0° is handled
  via wrap-around detection (e.g. last=358° → current=2°).
- Sampling uses ABSOLUTE target timestamps from a monotonic clock
  (frame.monotonic, not frame.timestamp). next_target_ts advances by
  += interval, NOT = ts + interval. This means per-frame jitter does
  not accumulate across the 70-point window, AND wall-clock NTP jumps
  cannot disrupt an in-flight collection.
- Backup end condition: angle wrap-back fires only when n_collected >=
  ``_wrap_back_floor(target)`` (max(target-3, target*0.95)) — a true
  safety net rather than the v2.5 70% rule that could end one sample
  short of full coverage.
- CycleData carries cycle_profile_id and CycleData.start_time uses
  monotonic seconds (FSM-internal); DB callers persist wall-clock from
  frame.timestamp instead.
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from core.cycle_profile import CycleProfile
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
    leak_valve_status: bool = False        # Sampled once at start of collection
    end_angle: float = 0.0                 # Angle when collection finished
    start_time: float = 0.0
    cycle_profile_id: str = ""             # v2.6: which profile drove this cycle


class CabinFSM:
    """State machine for one cabin's full-cycle collection.

    Parameters
    ----------
    cabin_id : int
        Cabin index (1-based; Cabin[0] is reserved).
    profile : CycleProfile
        Profile defining trigger angle, sample count, interval, timeout.
    """

    # Wrap-around tolerance: when last_angle > WRAP_FROM and current_angle <
    # WRAP_BACK we consider 0° has been crossed (one full revolution).
    WRAP_FROM_THRESHOLD = 330.0
    WRAP_BACK_THRESHOLD = 30.0

    @staticmethod
    def _wrap_back_floor(target_points: int) -> int:
        """Minimum n_collected for the wrap-back safety net to fire.

        Defined as max(target_points − 3, target_points × 0.95) so:
          - For typical target=70: floor = 67 (allow at most 3 missing pts)
          - For tiny test fixtures (target=20): the 95% rule dominates → 19
          - For target=100: the 3-point rule dominates → 97

        Replaces the v2.5 ``WRAP_BACK_MIN_FRACTION = 0.7`` which fired at
        ≥ 49 points for target=70. With ~1° angle jitter near the trigger,
        that 70% rule could end the cycle one sample short of full
        coverage and produce incomplete features. The new floor is strict
        enough that wrap-back is a true safety net — only fires when the
        cycle is essentially complete.
        """
        return int(max(target_points - 3, target_points * 0.95))

    def __init__(self, cabin_id: int, profile: CycleProfile):
        self.cabin_id = cabin_id
        self._profile = profile
        self._state = CycleState.IDLE
        self._data = CycleData()
        self._last_angle: Optional[float] = None
        # Absolute timestamp at which the *next* sample should be taken.
        # Updated by += interval (NOT = ts + interval), so per-frame jitter
        # does NOT accumulate across the 70-point window. This is the
        # canonical fix for the v2.5 sampling-drift symptom (cumulative
        # delay of several hundred ms over a long collection).
        self._next_target_ts: float = 0.0

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
        """Feed a polling frame and possibly transition state.

        ``ts`` is the monotonic timestamp (NTP-jump safe), used for
        absolute-target sample scheduling. The wall-clock timestamp on
        the frame is preserved for downstream DB persistence.
        """
        angle = frame.rt_angle
        # Prefer monotonic when available; fall back to wall-clock for
        # tests that construct CabinFrame without setting monotonic.
        ts = frame.monotonic if frame.monotonic else frame.timestamp

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
        self._state = CycleState.IDLE
        self._data = CycleData()
        self._last_angle = None
        self._next_target_ts = 0.0
        logger.debug("Cabin %d FSM reset to IDLE", self.cabin_id)

    def force_fault(self, reason: str = "") -> None:
        self._state = CycleState.FAULT
        logger.warning("Cabin %d: forced FAULT (%s)", self.cabin_id, reason)

    def clear_fault(self) -> None:
        self.reset()
        logger.info("Cabin %d: FAULT cleared -> IDLE", self.cabin_id)

    # -- internal handlers ---------------------------------------------------

    def _handle_idle(self, angle: float, ts: float, frame: CabinFrame) -> None:
        """Detect collection trigger.

        - For trigger_angle > 0: rising-edge crossing (last < trigger <= angle)
        - For trigger_angle == 0: wrap-around detection (last in [330°, 360°),
          current in (0°, 30°]), since a literal "<0 < 0" comparison is empty.
        """
        if self._last_angle is None:
            return

        trigger = self._profile.trigger_angle

        crossed = False
        if trigger > 0:
            if self._last_angle < trigger <= angle:
                crossed = True
        else:
            # Trigger at 0°: detect angle wrapping past 360° -> 0°
            if (self._last_angle >= self.WRAP_FROM_THRESHOLD
                    and angle <= self.WRAP_BACK_THRESHOLD):
                crossed = True

        if not crossed:
            return

        self._state = CycleState.COLLECTING
        self._data = CycleData(
            start_time=ts,
            leak_valve_status=frame.leak_valve_status,
            cycle_profile_id=self._profile.profile_id,
        )
        self._append(frame)
        # Sample 0 lands at the trigger; sample 1 targets trigger + interval.
        self._next_target_ts = ts + self._profile.collection_interval_s
        logger.info(
            "Cabin %d: IDLE -> COLLECTING (trigger=%.1f°, %.1f° -> %.1f°)",
            self.cabin_id, trigger, self._last_angle, angle,
        )

    def _handle_collecting(self, angle: float, ts: float, frame: CabinFrame) -> None:
        """Collect samples on absolute target ticks until full or wrap-back.

        ``_next_target_ts`` is advanced by += interval (not = ts + interval),
        so per-frame jitter does NOT accumulate across the 70-point window.
        """
        elapsed = ts - self._data.start_time

        if ts >= self._next_target_ts:
            self._append(frame)
            self._next_target_ts += self._profile.collection_interval_s

        target_points = self._profile.collection_points
        n_collected = len(self._data.pressures)

        # ── End condition 1: reached target point count ───────────
        if n_collected >= target_points:
            self._data.end_angle = angle
            self._transition_to_processing(
                f"collected {target_points} points, end_angle={angle:.1f}°"
            )
            return

        # ── End condition 2: angle wrap-back (full revolution) ────
        # Safety net: only fire when the cycle is essentially complete
        # (≤ 3 samples short AND ≥ 95% collected). See _wrap_back_floor.
        if (self._last_angle is not None
                and self._last_angle >= self.WRAP_FROM_THRESHOLD
                and angle <= self.WRAP_BACK_THRESHOLD
                and n_collected >= self._wrap_back_floor(target_points)):
            self._data.end_angle = angle
            self._transition_to_processing(
                f"wrap-back ({n_collected}/{target_points} points)"
            )
            return

        # ── End condition 3: timeout → FAULT ──────────────────────
        if elapsed >= self._profile.collection_timeout_s:
            self._data.end_angle = angle
            self._state = CycleState.FAULT
            logger.warning(
                "Cabin %d: COLLECTING -> FAULT (timeout %.1fs, %d/%d points)",
                self.cabin_id, elapsed, n_collected, target_points,
            )

    def _transition_to_processing(self, reason: str) -> None:
        self._state = CycleState.PROCESSING
        # start_time is now monotonic (see update()); use the same clock here.
        elapsed = (time.monotonic() - self._data.start_time
                   if self._data.start_time else 0)
        logger.info(
            "Cabin %d: COLLECTING -> PROCESSING (%s, %.3fs)",
            self.cabin_id, reason, elapsed,
        )

    def _append(self, frame: CabinFrame) -> None:
        self._data.pressures.append(frame.rt_pressure)
        self._data.angles.append(frame.rt_angle)
        self._data.timestamps.append(frame.timestamp)
        self._data.ai_values.append(frame.rt_ai)
        self._data.positions.append(frame.rt_position)


class CycleFSMManager:
    """Manages FSM instances for all active cabins (v2.6: profile-driven)."""

    def __init__(
        self,
        cabin_count: int,
        profile: CycleProfile,
        active_start: int = 1,
        active_end: Optional[int] = None,
    ):
        if active_end is None:
            active_end = cabin_count - 1
        self._profile = profile
        self.fsms: Dict[int, CabinFSM] = {
            i: CabinFSM(i, profile)
            for i in range(active_start, active_end + 1)
        }
        logger.info(
            "CycleFSMManager: %d FSMs for Cabin[%d]~[%d] "
            "(profile=%s, trigger=%.0f°, points=%d, interval=%.0fms)",
            len(self.fsms), active_start, active_end,
            profile.profile_id, profile.trigger_angle,
            profile.collection_points, profile.collection_interval_s * 1000,
        )

    def get_processing_cabins(self) -> List[int]:
        return [cid for cid, fsm in self.fsms.items() if fsm.state == CycleState.PROCESSING]

    def get_fault_cabins(self) -> List[int]:
        return [cid for cid, fsm in self.fsms.items() if fsm.state == CycleState.FAULT]

    def update_all(self, cabin_frames: Dict[int, CabinFrame]) -> None:
        for cid, frame in cabin_frames.items():
            if cid in self.fsms:
                self.fsms[cid].update(frame)
