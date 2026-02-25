"""Fault code registry for the health monitoring subsystem.

Each fault code has a severity level, a short mnemonic, and a human-readable
description.  The numeric code is also written to the PLC for HMI display.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class FaultLevel(enum.Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class FaultCode:
    code: str
    plc_value: int
    level: FaultLevel
    description: str


# Fault code registry
FAULT_CODES = {
    "F001": FaultCode("F001", 1, FaultLevel.CRITICAL, "PLC连接丢失"),
    "F002": FaultCode("F002", 2, FaultLevel.CRITICAL, "AI模型加载失败"),
    "F003": FaultCode("F003", 3, FaultLevel.ERROR,    "传感器数据异常 (超出合理范围)"),
    "F004": FaultCode("F004", 4, FaultLevel.WARNING,  "推理延迟超限"),
    "F005": FaultCode("F005", 5, FaultLevel.ERROR,    "磁盘空间不足"),
    "F006": FaultCode("F006", 6, FaultLevel.ERROR,    "数据库写入失败"),
    "F007": FaultCode("F007", 7, FaultLevel.WARNING,  "数据库容量接近上限"),
    "F008": FaultCode("F008", 8, FaultLevel.ERROR,    "采集线程异常终止"),
    "F009": FaultCode("F009", 9, FaultLevel.WARNING,  "状态机卡死 (COLLECTING 超时)"),
    "F010": FaultCode("F010", 10, FaultLevel.WARNING, "告警推送失败"),
}


def get_fault(code: str) -> FaultCode:
    """Look up a fault code; returns a generic UNKNOWN fault if not found."""
    return FAULT_CODES.get(code, FaultCode(code, 99, FaultLevel.ERROR, f"未知故障 {code}"))
