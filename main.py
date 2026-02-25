#!/usr/bin/env python3
"""Ldpj_backend – Edge AI leak-detection backend system.

Entry point that wires all subsystems together and runs the main loop.

Usage
-----
    python main.py              # default: mock PLC, model optional
    python main.py --mode s7    # real PLC via snap7
    python main.py --mode mock  # synthetic data (development)
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.loaders import (
    load_health_config,
    load_ipc_config,
    load_models_config,
    load_plc_config,
    load_runtime_config,
)
from core.cycle_fsm import CycleFSMManager
from core.logging_setup import setup_logging
from core.polling_engine import PollingEngine
from health.fault_reporter import FaultReporter
from health.health_checker import HealthChecker
from integration.alarm_pusher import AlarmPusher
from integration.api_server import APIServer
from integration.result_sender import ResultSender
from models.supervised_xgb import SupervisedXGB
from pipeline.control import CommandController
from pipeline.processing_loop import ProcessingLoop
from storage.database_logger import DatabaseLogger

logger: logging.Logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ldpj_backend edge AI system")
    parser.add_argument(
        "--mode",
        choices=["s7", "mock"],
        default="mock",
        help="PLC connection mode: 's7' for real PLC, 'mock' for synthetic data",
    )
    return parser.parse_args()


def main() -> None:
    global logger

    args = parse_args()

    # ── 1. Load configurations ──────────────────────────────────────────
    plc_cfg = load_plc_config()
    runtime_cfg = load_runtime_config()
    models_cfg = load_models_config()
    health_cfg = load_health_config()
    ipc_cfg = load_ipc_config()

    # ── 2. Logging ──────────────────────────────────────────────────────
    logger = setup_logging(runtime_cfg.get("logging", {}))
    logger.info("=" * 60)
    logger.info("Ldpj_backend v2.1 starting (mode=%s)", args.mode)
    logger.info("=" * 60)

    # ── 3. Fault reporter ───────────────────────────────────────────────
    fault_reporter = FaultReporter()

    # ── 4. Alarm pusher ─────────────────────────────────────────────────
    alarm_pusher = AlarmPusher(ipc_cfg)
    # Register alarm pusher as a fault callback
    if alarm_pusher.enabled:
        fault_reporter.register_callback(
            lambda event: alarm_pusher.push_alarm(
                event.fault.code, event.message, event.fault.level.value
            )
        )

    # ── 5. Polling engine ───────────────────────────────────────────────
    polling_engine = PollingEngine(plc_cfg, mode=args.mode)
    polling_engine.start()

    # ── 6. Cycle FSM manager ────────────────────────────────────────────
    cabin_count = plc_cfg.get("cabin_array", {}).get("cabin_count", 25)
    cycle_cfg = runtime_cfg.get("cycle_detection", {})
    fsm_manager = CycleFSMManager(cabin_count, cycle_cfg)

    # ── 7. Model ────────────────────────────────────────────────────────
    model = SupervisedXGB(models_cfg, base_dir=PROJECT_ROOT)
    try:
        model.load()
    except Exception as exc:
        logger.warning("Model load failed (system will run without inference): %s", exc)
        fault_reporter.raise_fault("F002", str(exc))

    # ── 8. Database ─────────────────────────────────────────────────────
    db_path = runtime_cfg.get("database", {}).get("path", "ldpj_data.db")
    db_logger = DatabaseLogger(db_path)

    # ── 9. Result sender ────────────────────────────────────────────────
    result_sender = ResultSender(plc_cfg, polling_engine)

    # ── 10. Health checker ──────────────────────────────────────────────
    health_checker = HealthChecker(health_cfg, fault_reporter)
    health_checker.set_references(
        polling_engine=polling_engine,
        model=model,
        db_logger=db_logger,
        fsm_manager=fsm_manager,
    )
    health_checker.start()

    # ── 11. API server ──────────────────────────────────────────────────
    api_server = APIServer(ipc_cfg)
    api_server.set_references(
        db_logger=db_logger,
        health_checker=health_checker,
        polling_engine=polling_engine,
        model=model,
        fault_reporter=fault_reporter,
    )
    api_server.start()

    # ── 12. Processing loop ─────────────────────────────────────────────
    proc_loop = ProcessingLoop(
        runtime_cfg=runtime_cfg,
        polling_engine=polling_engine,
        fsm_manager=fsm_manager,
        model=model,
        db_logger=db_logger,
        result_sender=result_sender,
        alarm_pusher=alarm_pusher,
        health_checker=health_checker,
        fault_reporter=fault_reporter,
    )
    proc_loop.start()

    # ── 13. Command controller ──────────────────────────────────────────
    ctrl = CommandController()
    ctrl.register("s", proc_loop.resume)
    ctrl.register("e", proc_loop.pause)
    ctrl.register("w", lambda: print(f"Watchdog: {'ON' if proc_loop.toggle_watchdog() else 'OFF'}"))
    ctrl.register("h", lambda: print(json.dumps(health_checker.run_all_checks(), indent=2, default=str)))
    ctrl.register("d", lambda: print(json.dumps(proc_loop.get_diagnostics(), indent=2, default=str)))
    ctrl.register("q", lambda: _shutdown(polling_engine, health_checker, api_server, db_logger, proc_loop))
    ctrl.start()

    # ── 14. Graceful shutdown on signals ────────────────────────────────
    def signal_handler(sig, frame):
        _shutdown(polling_engine, health_checker, api_server, db_logger, proc_loop)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # ── 15. Main loop ───────────────────────────────────────────────────
    logger.info("System ready. Entering main loop...")
    try:
        while proc_loop.is_running:
            proc_loop.run_once()
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown(polling_engine, health_checker, api_server, db_logger, proc_loop)


def _shutdown(polling_engine, health_checker, api_server, db_logger, proc_loop):
    logger.info("Shutting down...")
    proc_loop.stop()
    health_checker.stop()
    api_server.stop()
    polling_engine.stop()
    db_logger.close()
    logger.info("Shutdown complete.")
    sys.exit(0)


if __name__ == "__main__":
    main()
