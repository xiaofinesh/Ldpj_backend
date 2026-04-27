"""Result sender – writes inference results and fault codes back to PLC.

Each cabin's result is written into its own CabinParam UDT within DB9:
  - leakTestResult_AI  (Bool at cabin_base + 12, bit 0): AI leak detection result
  - cabinHealthStatus  (REAL at cabin_base + 14):        model probability / confidence

v2.5: Per-cabin write-back with single-write optimization.
      - Merges Bool + REAL into one 8-byte db_write to avoid 'Job pending'
      - S7Connection._io_lock serializes all PLC I/O (polling + write-back)
      - Cabin[0] is reserved and skipped
"""

from __future__ import annotations

import logging
import struct
import threading
from typing import Any, Dict

from core.exceptions import PLCWriteError

logger = logging.getLogger(__name__)


class ResultSender:
    """Writes inference results and fault codes to PLC data blocks.

    Each cabin's result is written back into its own CabinParam structure
    within the cabin array, using a single 8-byte write per cabin to
    avoid snap7 'Job pending' errors from rapid successive writes.

    Parameters
    ----------
    plc_cfg : dict
        The full content of ``plc.yaml``.
    polling_engine : Any
        The PollingEngine instance (used to access the S7 connection).
    """

    def __init__(self, plc_cfg: Dict[str, Any], polling_engine: Any):
        self._polling_engine = polling_engine
        self._write_lock = threading.Lock()

        cabin_cfg = plc_cfg.get("cabin_array", {})
        self._db_number = cabin_cfg.get("db_number", 9)
        self._start_offset = cabin_cfg.get("start_offset", 0)
        self._cabin_size = cabin_cfg.get("cabin_size_bytes", 20)
        self._active_start = cabin_cfg.get("active_start", 1)
        self._active_end = cabin_cfg.get("active_end", 25)

        wb = plc_cfg.get("write_back", {})
        self._result_ai_offset = wb.get("leak_result_ai_offset", 12)
        self._health_offset = wb.get("cabin_health_offset", 14)

        fw = plc_cfg.get("fault_write", {})
        self._fw_db = fw.get("db_number", 9)
        self._fw_offset = fw.get("byte_offset", 520)

    def _cabin_base(self, cabin_id: int) -> int:
        """Calculate the absolute byte offset for a given cabin index."""
        return self._start_offset + cabin_id * self._cabin_size

    def write_result(self, cabin_id: int, label: int, probability: float) -> None:
        """Write the inference result into the cabin's own CabinParam.

        Uses a single 8-byte write covering bytes +12 to +19 of the cabin:
          +12: Bool byte (bit 0 = leakTestResult_AI, bit 1 preserved)
          +13: padding (0x00)
          +14..+17: cabinHealthStatus (REAL, big-endian)
          +18: leakValveStatus byte (preserved)
          +19: padding (preserved)

        v2.6 semantics change
        ---------------------
        The third parameter is no longer a probability ∈ [0, 1] but the
        Q_est leak rate in Pa·m³/s (typical range 1e-7..1e-2). The
        ``cabinHealthStatus`` REAL field now carries Q_est. **HMI display
        logic must be updated in lockstep with the backend rollout**,
        otherwise the operator will see "health" plummet from 95% to 0.001
        at the cutover. Coordinate with the automation engineer before
        deploying v2.6 to production.

        Parameters
        ----------
        cabin_id : int
            Zero-based cabin index. Cabin[0] is reserved and will be skipped.
        label : int
            Inference label: 0 = leak, 1 = OK, -1 = unknown.
        probability : float
            v2.5: probability of OK (∈ [0, 1]).
            v2.6: Q_est leak rate (Pa·m³/s).
            The bytes-on-wire format is identical (REAL); only the
            interpretation changes.
        """
        # Guard: skip reserved Cabin[0] and out-of-range cabins
        if cabin_id < self._active_start or cabin_id > self._active_end:
            logger.debug("Cabin %d: outside active range [%d..%d], skip write-back",
                         cabin_id, self._active_start, self._active_end)
            return

        if label not in (0, 1):
            logger.debug("Cabin %d: label=%d (unknown), skip write-back", cabin_id, label)
            return

        base = self._cabin_base(cabin_id)
        write_addr = base + self._result_ai_offset  # byte +12

        with self._write_lock:
            conn = self._polling_engine._conn
            try:
                # Read current 8 bytes (+12 to +19) to preserve bits we don't own
                current = conn.db_read(self._db_number, write_addr, 8)
                buf = bytearray(current)

                # Byte 0 (+12): set/clear bit 0 (leakTestResult_AI)
                # snap7 bit ordering: DBX N.0 = LSB (0x01) of byte N
                if label == 0:
                    # Leak detected → set bit 0 to true
                    buf[0] = buf[0] | 0x01
                else:
                    # OK → clear bit 0 to false
                    buf[0] = buf[0] & ~0x01

                # Bytes 2..5 (+14..+17): cabinHealthStatus (REAL)
                struct.pack_into(">f", buf, 2, probability)

                # Single write: 8 bytes at once
                conn.db_write(self._db_number, write_addr, buf)

                logger.debug(
                    "Cabin %d: write-back OK (addr=%d, label=%d, prob=%.4f)",
                    cabin_id, write_addr, label, probability,
                )
            except Exception as exc:
                logger.error("Cabin %d: PLC write-back failed: %s", cabin_id, exc)
                raise PLCWriteError(
                    f"write_result failed for cabin {cabin_id}: {exc}"
                ) from exc

    def write_fault_code(self, plc_value: int) -> None:
        """Write a fault code integer to PLC for HMI display."""
        with self._write_lock:
            try:
                data = struct.pack(">h", plc_value)
                self._polling_engine._conn.db_write(self._fw_db, self._fw_offset, bytearray(data))
                logger.debug("Fault code written to PLC: %d", plc_value)
            except Exception as exc:
                logger.error("Failed to write fault code to PLC: %s", exc)
                raise PLCWriteError(f"write_fault_code failed: {exc}") from exc
