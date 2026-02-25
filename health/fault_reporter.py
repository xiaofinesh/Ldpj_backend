"""Fault reporter – centralised fault event handling.

Collects active faults, logs them, and delegates to the alarm pusher
and PLC fault-code writer as configured.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from health.fault_codes import FaultCode, FaultLevel, get_fault

logger = logging.getLogger(__name__)


@dataclass
class FaultEvent:
    fault: FaultCode
    message: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False


class FaultReporter:
    """Collects and manages active fault events."""

    def __init__(self):
        self._active_faults: Dict[str, FaultEvent] = {}
        self._history: List[FaultEvent] = []
        self._on_fault_callbacks: List[Callable[[FaultEvent], None]] = []

    def register_callback(self, cb: Callable[[FaultEvent], None]) -> None:
        """Register a callback invoked whenever a new fault is raised."""
        self._on_fault_callbacks.append(cb)

    def raise_fault(self, code: str, message: str = "") -> FaultEvent:
        """Raise or update a fault.

        If the same code is already active, the event is updated (not duplicated).
        """
        fc = get_fault(code)
        msg = message or fc.description
        event = FaultEvent(fault=fc, message=msg)

        if code not in self._active_faults:
            logger.log(
                _level_to_int(fc.level),
                "FAULT RAISED [%s] %s: %s", code, fc.level.value, msg,
            )
            self._active_faults[code] = event
            self._history.append(event)
            for cb in self._on_fault_callbacks:
                try:
                    cb(event)
                except Exception as exc:
                    logger.error("Fault callback error: %s", exc)
        else:
            self._active_faults[code].timestamp = time.time()

        return event

    def resolve_fault(self, code: str) -> None:
        """Mark a fault as resolved and remove from active set."""
        if code in self._active_faults:
            self._active_faults[code].resolved = True
            logger.info("FAULT RESOLVED [%s]", code)
            del self._active_faults[code]

    @property
    def active_faults(self) -> Dict[str, FaultEvent]:
        return dict(self._active_faults)

    @property
    def has_critical(self) -> bool:
        return any(
            e.fault.level == FaultLevel.CRITICAL
            for e in self._active_faults.values()
        )

    def get_highest_plc_value(self) -> int:
        """Return the PLC value of the most severe active fault, or 0."""
        if not self._active_faults:
            return 0
        worst = max(
            self._active_faults.values(),
            key=lambda e: _level_priority(e.fault.level),
        )
        return worst.fault.plc_value

    def summary(self) -> Dict[str, Any]:
        return {
            "active_count": len(self._active_faults),
            "has_critical": self.has_critical,
            "faults": [
                {
                    "code": e.fault.code,
                    "level": e.fault.level.value,
                    "message": e.message,
                    "since": e.timestamp,
                }
                for e in self._active_faults.values()
            ],
        }


def _level_to_int(level: FaultLevel) -> int:
    return {
        FaultLevel.INFO: logging.INFO,
        FaultLevel.WARNING: logging.WARNING,
        FaultLevel.ERROR: logging.ERROR,
        FaultLevel.CRITICAL: logging.CRITICAL,
    }.get(level, logging.ERROR)


def _level_priority(level: FaultLevel) -> int:
    return {
        FaultLevel.INFO: 0,
        FaultLevel.WARNING: 1,
        FaultLevel.ERROR: 2,
        FaultLevel.CRITICAL: 3,
    }.get(level, 2)
