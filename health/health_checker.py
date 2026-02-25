"""Health checker – periodic self-diagnosis of the entire system.

Runs as a background thread and checks PLC connectivity, model status,
disk space, inference latency, polling thread liveness, and FSM states.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time
from typing import Any, Dict, Optional

from health.fault_reporter import FaultReporter

logger = logging.getLogger(__name__)


class HealthChecker:
    """Periodic health-check engine.

    Parameters
    ----------
    health_cfg : dict
        The full content of ``health.yaml``.
    fault_reporter : FaultReporter
        Central fault reporter instance.
    """

    def __init__(self, health_cfg: Dict[str, Any], fault_reporter: FaultReporter):
        self._cfg = health_cfg
        self._reporter = fault_reporter
        self._interval = health_cfg.get("check_interval_s", 60)
        self._checks_cfg = health_cfg.get("checks", {})
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # External references (set by main after construction)
        self._polling_engine: Any = None
        self._model: Any = None
        self._db_logger: Any = None
        self._fsm_manager: Any = None
        self._last_inference_ms: float = 0.0

    def set_references(
        self,
        polling_engine: Any = None,
        model: Any = None,
        db_logger: Any = None,
        fsm_manager: Any = None,
    ) -> None:
        """Inject references to other subsystems for checking."""
        if polling_engine is not None:
            self._polling_engine = polling_engine
        if model is not None:
            self._model = model
        if db_logger is not None:
            self._db_logger = db_logger
        if fsm_manager is not None:
            self._fsm_manager = fsm_manager

    def report_inference_latency(self, ms: float) -> None:
        """Called by the processing loop after each inference."""
        self._last_inference_ms = ms

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if not self._cfg.get("enabled", True):
            logger.info("HealthChecker disabled by config")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="health-checker")
        self._thread.start()
        logger.info("HealthChecker started (interval=%ds)", self._interval)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

    # -- manual trigger ------------------------------------------------------

    def run_all_checks(self) -> Dict[str, Any]:
        """Run all checks once and return a structured report."""
        report: Dict[str, Any] = {"timestamp": time.time(), "checks": {}}

        report["checks"]["plc_connection"] = self._check_plc()
        report["checks"]["model_loaded"] = self._check_model()
        report["checks"]["disk_space"] = self._check_disk()
        report["checks"]["inference_latency"] = self._check_latency()
        report["checks"]["polling_thread"] = self._check_polling()
        report["checks"]["fsm_stuck"] = self._check_fsm()
        report["checks"]["database"] = self._check_database()

        report["overall"] = "HEALTHY" if not self._reporter.has_critical else "DEGRADED"
        report["active_faults"] = self._reporter.summary()
        return report

    # -- internal loop -------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            try:
                self.run_all_checks()
            except Exception as exc:
                logger.error("Health check loop error: %s", exc, exc_info=True)
            time.sleep(self._interval)

    # -- individual checks ---------------------------------------------------

    def _check_plc(self) -> Dict[str, Any]:
        cfg = self._checks_cfg.get("plc_connection", {})
        if not cfg.get("enabled", True):
            return {"status": "SKIP"}
        connected = self._polling_engine.plc_connected if self._polling_engine else False
        if not connected:
            self._reporter.raise_fault("F001", "PLC连接丢失")
        else:
            self._reporter.resolve_fault("F001")
        return {"status": "OK" if connected else "FAIL", "connected": connected}

    def _check_model(self) -> Dict[str, Any]:
        cfg = self._checks_cfg.get("model_loaded", {})
        if not cfg.get("enabled", True):
            return {"status": "SKIP"}
        loaded = self._model.loaded if self._model else False
        if not loaded:
            self._reporter.raise_fault("F002", "AI模型未加载")
        else:
            self._reporter.resolve_fault("F002")
        return {"status": "OK" if loaded else "FAIL", "loaded": loaded}

    def _check_disk(self) -> Dict[str, Any]:
        cfg = self._checks_cfg.get("disk_space", {})
        if not cfg.get("enabled", True):
            return {"status": "SKIP"}
        min_free = cfg.get("min_free_mb", 100)
        try:
            usage = shutil.disk_usage("/")
            free_mb = usage.free / (1024 * 1024)
            ok = free_mb >= min_free
            if not ok:
                self._reporter.raise_fault("F005", f"磁盘剩余空间 {free_mb:.0f}MB < {min_free}MB")
            else:
                self._reporter.resolve_fault("F005")
            return {"status": "OK" if ok else "FAIL", "free_mb": round(free_mb, 1)}
        except Exception as exc:
            return {"status": "ERROR", "error": str(exc)}

    def _check_latency(self) -> Dict[str, Any]:
        cfg = self._checks_cfg.get("inference_latency", {})
        if not cfg.get("enabled", True):
            return {"status": "SKIP"}
        max_ms = cfg.get("max_ms", 500)
        ok = self._last_inference_ms <= max_ms
        if not ok:
            self._reporter.raise_fault("F004", f"推理延迟 {self._last_inference_ms:.0f}ms > {max_ms}ms")
        else:
            self._reporter.resolve_fault("F004")
        return {"status": "OK" if ok else "WARN", "last_ms": round(self._last_inference_ms, 1)}

    def _check_polling(self) -> Dict[str, Any]:
        cfg = self._checks_cfg.get("polling_thread", {})
        if not cfg.get("enabled", True):
            return {"status": "SKIP"}
        alive = self._polling_engine.is_running if self._polling_engine else False
        if not alive:
            self._reporter.raise_fault("F008", "采集线程异常终止")
        else:
            self._reporter.resolve_fault("F008")
        return {"status": "OK" if alive else "FAIL", "alive": alive}

    def _check_fsm(self) -> Dict[str, Any]:
        cfg = self._checks_cfg.get("fsm_stuck", {})
        if not cfg.get("enabled", True) or self._fsm_manager is None:
            return {"status": "SKIP"}
        max_stuck = cfg.get("max_stuck_duration_s", 120)
        stuck_cabins = []
        from core.cycle_fsm import CycleState
        for cid, fsm in self._fsm_manager.fsms.items():
            if fsm.state == CycleState.COLLECTING:
                elapsed = time.time() - fsm.data.start_time if fsm.data.start_time else 0
                if elapsed > max_stuck:
                    stuck_cabins.append(cid)
        if stuck_cabins:
            self._reporter.raise_fault("F009", f"舱室 {stuck_cabins} 状态机卡死")
        else:
            self._reporter.resolve_fault("F009")
        return {"status": "OK" if not stuck_cabins else "WARN", "stuck_cabins": stuck_cabins}

    def _check_database(self) -> Dict[str, Any]:
        if self._db_logger is None:
            return {"status": "SKIP"}
        size_mb = self._db_logger.get_db_size_mb()
        count = self._db_logger.count_records()
        if size_mb > 450:
            self._reporter.raise_fault("F007", f"数据库大小 {size_mb:.0f}MB 接近上限")
        else:
            self._reporter.resolve_fault("F007")
        return {"status": "OK", "size_mb": round(size_mb, 1), "record_count": count}
