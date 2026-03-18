"""Alarm pusher – sends HTTP POST alerts to external IPC systems."""

from __future__ import annotations
import json, logging, threading, time
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class AlarmPusher:
    def __init__(self, ipc_cfg: Dict[str, Any]):
        cfg = ipc_cfg.get("alarm_pusher", {})
        self._enabled = cfg.get("enabled", False)
        self._targets = cfg.get("targets", [])
        self._push_on_leak = cfg.get("push_on_leak", False)
        self._min_level = cfg.get("min_fault_level_to_push", "ERROR")
        self._level_order = {"INFO": 0, "WARNING": 1, "ERROR": 2, "CRITICAL": 3}

    @property
    def enabled(self) -> bool: return self._enabled

    def should_push(self, level: str) -> bool:
        return self._level_order.get(level, 0) >= self._level_order.get(self._min_level, 2)

    def push_alarm(self, fault_code: str, message: str, level: str = "ERROR") -> None:
        if not self._enabled or not self.should_push(level): return
        payload = {"source": "ldpj_backend", "fault_code": fault_code,
                   "message": message, "level": level, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        t = threading.Thread(target=self._send_to_all, args=(payload,), daemon=True)
        t.start()

    def push_leak_alarm(self, cavity_id: int, probability: float) -> None:
        if not self._push_on_leak: return
        self.push_alarm("LEAK", f"舱室 {cavity_id} 检测到漏液 (概率={probability:.4f})", "ERROR")

    def _send_to_all(self, payload):
        for target in self._targets:
            self._send_with_retry(target.get("url",""), payload,
                                   target.get("timeout_s",5), target.get("retries",3))

    def _send_with_retry(self, url, payload, timeout, retries):
        import httpx
        for attempt in range(1, retries + 1):
            try:
                resp = httpx.post(url, json=payload, timeout=timeout)
                if resp.status_code < 300:
                    logger.info("Alarm pushed to %s (attempt %d)", url, attempt); return
            except Exception as exc:
                logger.warning("Alarm push to %s failed (attempt %d): %s", url, attempt, exc)
            if attempt < retries: time.sleep(1)
        logger.error("Alarm push to %s exhausted all %d retries", url, retries)
