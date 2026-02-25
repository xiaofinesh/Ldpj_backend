"""Result sender – writes inference results and fault codes back to PLC."""

from __future__ import annotations

import logging
import struct
from typing import Any, Dict

from core.exceptions import PLCWriteError

logger = logging.getLogger(__name__)


class ResultSender:
    """Writes inference results and fault codes to PLC data blocks.

    Parameters
    ----------
    plc_cfg : dict
        The full content of ``plc.yaml``.
    polling_engine : Any
        The PollingEngine instance (used to access the S7 connection).
    """

    def __init__(self, plc_cfg: Dict[str, Any], polling_engine: Any):
        self._polling_engine = polling_engine
        wb = plc_cfg.get("write_back", {})
        self._wb_db = wb.get("db_number", 9)
        self._wb_offset = wb.get("byte_offset", 200)
        self._wb_scale = wb.get("scale", 10)
        self._wb_base = wb.get("base", 0)

        fw = plc_cfg.get("fault_write", {})
        self._fw_db = fw.get("db_number", 9)
        self._fw_offset = fw.get("byte_offset", 202)

    def write_result(self, label: int, probability: float) -> None:
        """Write the inference result to PLC.

        The value written is: ``base + int(probability * scale)`` for label=1,
        or ``base`` for label=0 (leak detected).
        """
        try:
            if label == 1:
                value = self._wb_base + int(probability * self._wb_scale)
            else:
                value = self._wb_base
            data = struct.pack(">h", value)
            self._polling_engine._conn.db_write(self._wb_db, self._wb_offset, bytearray(data))
            logger.debug("Result written to PLC: label=%d value=%d", label, value)
        except Exception as exc:
            logger.error("Failed to write result to PLC: %s", exc)
            raise PLCWriteError(f"write_result failed: {exc}") from exc

    def write_fault_code(self, plc_value: int) -> None:
        """Write a fault code integer to PLC for HMI display."""
        try:
            data = struct.pack(">h", plc_value)
            self._polling_engine._conn.db_write(self._fw_db, self._fw_offset, bytearray(data))
            logger.debug("Fault code written to PLC: %d", plc_value)
        except Exception as exc:
            logger.error("Failed to write fault code to PLC: %s", exc)
