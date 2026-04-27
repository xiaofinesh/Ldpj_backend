"""Fault code definitions for the health monitoring system."""

FAULT_CODES = {
    "F001": {"description": "PLC连接丢失", "level": "CRITICAL", "plc_value": 1},
    "F002": {"description": "AI模型未加载", "level": "ERROR", "plc_value": 2},
    "F003": {"description": "传感器数据异常", "level": "WARNING", "plc_value": 3},
    "F004": {"description": "推理延迟过高", "level": "WARNING", "plc_value": 4},
    "F005": {"description": "磁盘空间不足", "level": "ERROR", "plc_value": 5},
    "F006": {"description": "数据库写入失败", "level": "ERROR", "plc_value": 6},
    "F007": {"description": "数据库容量告警", "level": "WARNING", "plc_value": 7},
    "F008": {"description": "采集线程异常终止", "level": "CRITICAL", "plc_value": 8},
    "F009": {"description": "舱室状态机故障", "level": "WARNING", "plc_value": 9},
    # v2.6
    "F010": {"description": "M1/M2 漏率估计差异过大", "level": "WARNING", "plc_value": 10},
    "F011": {"description": "M1 模型未标定该舱", "level": "WARNING", "plc_value": 11},
    "F012": {"description": "Q 估计低于系统分辨率", "level": "INFO", "plc_value": 12},
}
