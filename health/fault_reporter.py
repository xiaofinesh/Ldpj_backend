"""Fault reporter – centralized fault tracking and notification."""

from __future__ import annotations
import enum, logging, time
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
        if code in self._active_faults: return
        fault_def = self._fault_defs.get(code)
        if not fault_def:
            fault_def = FaultDef(code=code, description="Unknown", level=FaultLevel.ERROR, plc_value=99)
        event = FaultEvent(fault=fault_def, message=message or fault_def.description, timestamp=time.time())
        self._active_faults[code] = event
        logger.log(logging.ERROR if fault_def.level in (FaultLevel.ERROR, FaultLevel.CRITICAL) else logging.WARNING,
                    "FAULT [%s] %s: %s", code, fault_def.level.value, event.message)
        for cb in self._callbacks:
            try: cb(event)
            except Exception: pass

    def resolve_fault(self, code: str) -> None:
        if code in self._active_faults:
            self._active_faults[code].resolved = True
            del self._active_faults[code]
            logger.info("FAULT RESOLVED [%s]", code)

    @property
    def active_faults(self) -> Dict[str, FaultEvent]: return dict(self._active_faults)

    @property
    def has_critical(self) -> bool:
        return any(e.fault.level == FaultLevel.CRITICAL for e in self._active_faults.values())

    def get_highest_plc_value(self) -> int:
        if not self._active_faults: return 0
        return max(e.fault.plc_value for e in self._active_faults.values())

    def summary(self) -> Dict[str, Any]:
        return {"active_count": len(self._active_faults), "has_critical": self.has_critical,
                "faults": [{"code": e.fault.code, "level": e.fault.level.value,
                            "message": e.message, "since": e.timestamp}
                           for e in self._active_faults.values()]}
