#!/usr/bin/env python3
"""Ldpj_backend v2.5 – Edge AI leak-detection backend system.

Usage
-----
    python main.py --mode mock   # synthetic data (development)
    python main.py --mode s7     # real PLC via snap7
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.loaders import (
    load_health_config, load_ipc_config, load_models_config,
    load_plc_config, load_runtime_config,
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
from storage.data_exporter import interactive_export

logger: logging.Logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ldpj_backend edge AI system")
    p.add_argument("--mode", choices=["s7", "mock"], default="mock")
    return p.parse_args()


def main() -> None:
    global logger
    args = parse_args()

    plc_cfg = load_plc_config()
    runtime_cfg = load_runtime_config()
    models_cfg = load_models_config()
    health_cfg = load_health_config()
    ipc_cfg = load_ipc_config()

    logger = setup_logging(runtime_cfg.get("logging", {}))
    logger.info("=" * 60)
    logger.info("Ldpj_backend v2.5 starting (mode=%s)", args.mode)
    logger.info("=" * 60)

    fault_reporter = FaultReporter()

    alarm_pusher = AlarmPusher(ipc_cfg)
    if alarm_pusher.enabled:
        fault_reporter.register_callback(
            lambda event: alarm_pusher.push_alarm(
                event.fault.code, event.message, event.fault.level.value))

    polling_engine = PollingEngine(plc_cfg, mode=args.mode)
    polling_engine.start()

    cabin_cfg = plc_cfg.get("cabin_array", {})
    cycle_cfg = runtime_cfg.get("cycle_detection", {})
    fsm_manager = CycleFSMManager(
        cabin_cfg.get("cabin_count", 26), cycle_cfg,
        active_start=cabin_cfg.get("active_start", 1),
        active_end=cabin_cfg.get("active_end", 25),
    )

    model = SupervisedXGB(models_cfg, base_dir=PROJECT_ROOT)
    try:
        model.load()
    except Exception as exc:
        logger.warning("Model load failed: %s", exc)
        fault_reporter.raise_fault("F002", str(exc))

    db_path = runtime_cfg.get("database", {}).get("path", "ldpj_data.db")
    db_logger = DatabaseLogger(db_path)

    result_sender = ResultSender(plc_cfg, polling_engine)

    health_checker = HealthChecker(health_cfg, fault_reporter)
    health_checker.set_references(
        polling_engine=polling_engine, model=model,
        db_logger=db_logger, fsm_manager=fsm_manager)
    health_checker.start()

    api_server = APIServer(ipc_cfg)
    api_server.set_references(
        db_logger=db_logger, health_checker=health_checker,
        polling_engine=polling_engine, model=model, fault_reporter=fault_reporter)
    api_server.start()

    proc_loop = ProcessingLoop(
        runtime_cfg=runtime_cfg, polling_engine=polling_engine,
        fsm_manager=fsm_manager, model=model, db_logger=db_logger,
        result_sender=result_sender, alarm_pusher=alarm_pusher,
        health_checker=health_checker, fault_reporter=fault_reporter)
    proc_loop.start()

    ctrl = CommandController()
    ctrl.register("s", proc_loop.resume)
    ctrl.register("e", proc_loop.pause)
    ctrl.register("w", lambda: print(f"Watchdog: {'ON' if proc_loop.toggle_watchdog() else 'OFF'}"))
    ctrl.register("h", lambda: print(json.dumps(health_checker.run_all_checks(), indent=2, default=str)))
    ctrl.register("d", lambda: print(json.dumps(proc_loop.get_diagnostics(), indent=2, default=str)))
    def _do_export():
        proc_loop.pause()
        try:
            interactive_export(db_logger, base_dir=PROJECT_ROOT)
        finally:
            proc_loop.resume()

    ctrl.register("x", _do_export)
    ctrl.register("q", lambda: _shutdown(polling_engine, health_checker, api_server, db_logger, proc_loop))
    ctrl.start()

    def signal_handler(sig, frame):
        _shutdown(polling_engine, health_checker, api_server, db_logger, proc_loop)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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
