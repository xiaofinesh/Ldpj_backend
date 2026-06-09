"""Fault reporter – centralized fault tracking and notification."""

from __future__ import annotations
import enum, logging, threading, time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from health.fault_codes import FAULT_CODES

logger = logging.getLogger(__name__)


class FaultLevel(enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class FaultDef:
    code: str
    description: str
    level: FaultLevel
    plc_value: int


@dataclass
class FaultEvent:
    fault: FaultDef
    message: str
    timestamp: float
    resolved: bool = False


class FaultReporter:
    def __init__(self):
        # _lock guards _active_faults against concurrent raise/resolve (health
        # thread, proc-loop, result-sender callback) vs iteration (summary,
        # has_critical) — previously a "dict changed size during iteration".
        self._lock = threading.Lock()
        self._active_faults: Dict[str, FaultEvent] = {}
        self._callbacks: List[Callable] = []
        self._fault_defs: Dict[str, FaultDef] = {}
        for code, info in FAULT_CODES.items():
            self._fault_defs[code] = FaultDef(
                code=code, description=info["description"],
                level=FaultLevel(info["level"]), plc_value=info["plc_value"])

    def register_callback(self, cb: Callable) -> None:
        self._callbacks.append(cb)

    def raise_fault(self, code: str, message: str = "") -> None:
        fault_def = self._fault_defs.get(code)
        if not fault_def:
            fault_def = FaultDef(code=code, description="Unknown", level=FaultLevel.ERROR, plc_value=99)
        event = FaultEvent(fault=fault_def, message=message or fault_def.description, timestamp=time.time())
        with self._lock:
            if code in self._active_faults:
                return  # already active (dedup) — no re-log, no re-callback
            self._active_faults[code] = event
        logger.log(logging.ERROR if fault_def.level in (FaultLevel.ERROR, FaultLevel.CRITICAL) else logging.WARNING,
                    "FAULT [%s] %s: %s", code, fault_def.level.value, event.message)
        # Callbacks run OUTSIDE the lock (a callback may itself raise_fault).
        for cb in list(self._callbacks):
            try: cb(event)
            except Exception: pass

    def resolve_fault(self, code: str) -> None:
        with self._lock:
            ev = self._active_faults.pop(code, None)
        if ev is not None:
            logger.info("FAULT RESOLVED [%s]", code)

    @property
    def active_faults(self) -> Dict[str, FaultEvent]:
        with self._lock:
            return dict(self._active_faults)

    @property
    def has_critical(self) -> bool:
        with self._lock:
            return any(e.fault.level == FaultLevel.CRITICAL for e in self._active_faults.values())

    # Severity ranking for fault levels (higher = more severe)
    _SEVERITY = {FaultLevel.INFO: 0, FaultLevel.WARNING: 1,
                 FaultLevel.ERROR: 2, FaultLevel.CRITICAL: 3}

    def get_highest_plc_value(self) -> int:
        """Return plc_value of the most severe active fault (CRITICAL > ERROR > WARNING > INFO).

        Lower numeric plc_value does NOT mean lower severity — codes are arbitrary IDs.
        """
        with self._lock:
            if not self._active_faults: return 0
            most_severe = max(self._active_faults.values(),
                              key=lambda e: self._SEVERITY.get(e.fault.level, 0))
            return most_severe.fault.plc_value

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            items = list(self._active_faults.values())
        has_crit = any(e.fault.level == FaultLevel.CRITICAL for e in items)
        return {"active_count": len(items), "has_critical": has_crit,
                "faults": [{"code": e.fault.code, "level": e.fault.level.value,
                            "message": e.message, "since": e.timestamp}
                           for e in items]}
