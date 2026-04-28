"""Health checker – periodic self-diagnosis of the entire system."""

from __future__ import annotations
import logging, os, shutil, threading, time
from pathlib import Path
from typing import Any, Dict, Optional
from health.fault_reporter import FaultReporter

logger = logging.getLogger(__name__)


class HealthChecker:
    def __init__(self, health_cfg: Dict[str, Any], fault_reporter: FaultReporter):
        self._cfg = health_cfg
        self._reporter = fault_reporter
        self._interval = health_cfg.get("check_interval_s", 60)
        self._checks_cfg = health_cfg.get("checks", {})
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._polling_engine: Any = None
        self._model: Any = None
        self._db_logger: Any = None
        self._fsm_manager: Any = None
        self._last_inference_ms: float = 0.0

    def set_references(self, polling_engine=None, model=None, db_logger=None, fsm_manager=None):
        if polling_engine: self._polling_engine = polling_engine
        if model: self._model = model
        if db_logger: self._db_logger = db_logger
        if fsm_manager: self._fsm_manager = fsm_manager

    def report_inference_latency(self, ms: float):
        self._last_inference_ms = ms

    def start(self):
        if not self._cfg.get("enabled", True): return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="health-checker")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread: self._thread.join(timeout=10)

    def run_all_checks(self) -> Dict[str, Any]:
        report = {"timestamp": time.time(), "checks": {}}
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

    def _loop(self):
        while self._running:
            try: self.run_all_checks()
            except Exception as exc: logger.error("Health check error: %s", exc)
            time.sleep(self._interval)

    def _check_plc(self):
        cfg = self._checks_cfg.get("plc_connection", {})
        if not cfg.get("enabled", True): return {"status": "SKIP"}
        ok = self._polling_engine.plc_connected if self._polling_engine else False
        if not ok: self._reporter.raise_fault("F001", "PLC连接丢失")
        else: self._reporter.resolve_fault("F001")
        return {"status": "OK" if ok else "FAIL", "connected": ok}

    def _check_model(self):
        cfg = self._checks_cfg.get("model_loaded", {})
        if not cfg.get("enabled", True): return {"status": "SKIP"}
        ok = self._model.loaded if self._model else False
        if not ok: self._reporter.raise_fault("F002", "AI模型未加载")
        else: self._reporter.resolve_fault("F002")
        return {"status": "OK" if ok else "FAIL", "loaded": ok}

    def _check_disk(self):
        cfg = self._checks_cfg.get("disk_space", {})
        if not cfg.get("enabled", True): return {"status": "SKIP"}
        min_free = cfg.get("min_free_mb", 100)
        # Check the partition that holds the database, not the root partition.
        # On systems where /var or /opt is on a separate mount this is what
        # actually fills up first; the root partition free space is irrelevant.
        target = Path(self._db_logger._db_path).resolve().parent if self._db_logger else Path("/")
        if not target.exists():
            target = Path("/")
        try:
            free_mb = shutil.disk_usage(target).free / (1024 * 1024)
            ok = free_mb >= min_free
            if not ok: self._reporter.raise_fault("F005", f"磁盘剩余{free_mb:.0f}MB ({target})")
            else: self._reporter.resolve_fault("F005")
            return {"status": "OK" if ok else "FAIL",
                    "free_mb": round(free_mb, 1), "path": str(target)}
        except Exception as exc:
            return {"status": "ERROR", "error": str(exc)}

    def _check_latency(self):
        cfg = self._checks_cfg.get("inference_latency", {})
        if not cfg.get("enabled", True): return {"status": "SKIP"}
        max_ms = cfg.get("max_ms", 500)
        ok = self._last_inference_ms <= max_ms
        return {"status": "OK" if ok else "WARN", "last_ms": round(self._last_inference_ms, 1)}

    def _check_polling(self):
        cfg = self._checks_cfg.get("polling_thread", {})
        if not cfg.get("enabled", True): return {"status": "SKIP"}
        ok = self._polling_engine.is_running if self._polling_engine else False
        if not ok: self._reporter.raise_fault("F008", "采集线程异常终止")
        else: self._reporter.resolve_fault("F008")
        return {"status": "OK" if ok else "FAIL", "alive": ok}

    def _check_fsm(self):
        cfg = self._checks_cfg.get("fsm_stuck", {})
        if not cfg.get("enabled", True) or not self._fsm_manager: return {"status": "SKIP"}
        max_stuck = cfg.get("max_stuck_duration_s", 120)
        stuck = []
        from core.cycle_fsm import CycleState
        # FSM start_time is in monotonic seconds (see cycle_fsm docstring),
        # so the elapsed comparison must use time.monotonic() — not
        # time.time() (which differs by ~1.7e9 since they're different epochs).
        now_mono = time.monotonic()
        for cid, fsm in self._fsm_manager.fsms.items():
            if fsm.state == CycleState.COLLECTING:
                elapsed = now_mono - fsm.data.start_time if fsm.data.start_time else 0
                if elapsed > max_stuck: stuck.append(cid)
        if stuck: self._reporter.raise_fault("F009", f"舱室{stuck}卡死")
        else: self._reporter.resolve_fault("F009")
        return {"status": "OK" if not stuck else "WARN", "stuck_cabins": stuck}

    def _check_database(self):
        cfg = self._checks_cfg.get("database", {})
        if not self._db_logger: return {"status": "SKIP"}
        size_mb = self._db_logger.get_db_size_mb()
        count = self._db_logger.count_records()
        # Threshold: warn at 90% of configured max_size_mb (default 1000 MB).
        # Reading the limit from health.yaml lets ops tune without a code
        # change; previously this was hardcoded to 450 MB and ignored
        # runtime.yaml::database.max_size_mb.
        max_mb = float(cfg.get("max_size_mb", 1000))
        warn_at_mb = max_mb * float(cfg.get("warn_fraction", 0.9))
        if size_mb > warn_at_mb:
            self._reporter.raise_fault(
                "F007", f"数据库 {size_mb:.0f}MB 超过 {warn_at_mb:.0f}MB 警戒线",
            )
        else:
            self._reporter.resolve_fault("F007")
        return {"status": "OK", "size_mb": round(size_mb, 1),
                "warn_at_mb": round(warn_at_mb, 1), "record_count": count}
