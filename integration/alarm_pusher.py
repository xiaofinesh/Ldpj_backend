"""Alarm pusher – sends HTTP POST alerts to external IPC systems.

Runs asynchronously to avoid blocking the main processing loop.
Supports multiple targets with configurable retries.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AlarmPusher:
    """Push alarm notifications to external systems via HTTP POST.

    Parameters
    ----------
    ipc_cfg : dict
        The ``alarm_pusher`` section of ``ipc.yaml``.
    """

    def __init__(self, ipc_cfg: Dict[str, Any]):
        cfg = ipc_cfg.get("alarm_pusher", {})
        self._enabled = cfg.get("enabled", False)
        self._targets = cfg.get("targets", [])
        self._push_on_leak = cfg.get("push_on_leak", False)
        self._min_level = cfg.get("min_fault_level_to_push", "ERROR")
        self._level_order = {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def should_push(self, level: str) -> bool:
        """Check if the given level meets the minimum push threshold."""
        return self._level_order.get(level, 0) >= self._level_order.get(self._min_level, 2)

    def push_alarm(self, fault_code: str, message: str, level: str = "ERROR") -> None:
        """Send an alarm to all configured targets (non-blocking).

        Parameters
        ----------
        fault_code : str
            The fault code (e.g. ``"F001"``).
        message : str
            Human-readable description.
        level : str
            Severity level.
        """
        if not self._enabled:
            return
        if not self.should_push(level):
            return

        payload = {
            "source": "ldpj_backend",
            "fault_code": fault_code,
            "message": message,
            "level": level,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # Fire-and-forget in a background thread
        t = threading.Thread(
            target=self._send_to_all,
            args=(payload,),
            daemon=True,
            name="alarm-push",
        )
        t.start()

    def push_leak_alarm(self, cavity_id: int, probability: float) -> None:
        """Push a leak-detection alarm if configured."""
        if not self._push_on_leak:
            return
        self.push_alarm(
            fault_code="LEAK",
            message=f"舱室 {cavity_id} 检测到漏液 (概率={probability:.4f})",
            level="ERROR",
        )

    def _send_to_all(self, payload: Dict[str, Any]) -> None:
        for target in self._targets:
            url = target.get("url", "")
            timeout = target.get("timeout_s", 5)
            retries = target.get("retries", 3)
            self._send_with_retry(url, payload, timeout, retries)

    def _send_with_retry(
        self, url: str, payload: Dict[str, Any], timeout: float, retries: int
    ) -> None:
        import httpx

        for attempt in range(1, retries + 1):
            try:
                resp = httpx.post(
                    url,
                    json=payload,
                    timeout=timeout,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code < 300:
                    logger.info("Alarm pushed to %s (attempt %d)", url, attempt)
                    return
                else:
                    logger.warning(
                        "Alarm push to %s returned %d (attempt %d)",
                        url, resp.status_code, attempt,
                    )
            except Exception as exc:
                logger.warning("Alarm push to %s failed (attempt %d): %s", url, attempt, exc)

            if attempt < retries:
                time.sleep(1)

        logger.error("Alarm push to %s exhausted all %d retries", url, retries)
