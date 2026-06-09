#!/usr/bin/env python3
"""Ldpj_backend v2.6.2 – Edge AI leak-detection backend system.

Startup flow:
  1. Load configs, init subsystems, auto-start processing
  2. Show system status banner
  3. User selects display mode (normal/debug)
  4. Show command menu (processing already running)
  5. Enter command loop
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.loaders import (
    assert_kts_consistency,
    load_active_cycle_profile,
    load_cabins_config,
    load_health_config, load_ipc_config, load_models_config,
    load_plc_config, load_products_config, load_runtime_config,
    validate_operating_point,
    validate_product_resolution,
)
from core.cycle_fsm import CycleFSMManager
from core.logging_setup import setup_logging, set_console_mode
from core.polling_engine import PollingEngine
from health.fault_reporter import FaultReporter
from health.health_checker import HealthChecker
from integration.alarm_pusher import AlarmPusher
from integration.api_server import APIServer
from integration.result_sender import ResultSender
from models.linear_regression_m1 import LinearRegressionM1
from models.xgb_regressor_m2 import XGBRegressorM2
from pipeline.processing_loop import ProcessingLoop
from storage.database_logger import DatabaseLogger
from storage.data_exporter import interactive_export

logger: logging.Logger


# ── Status reporter (normal mode) ─────────────────────────────────────

class StatusReporter:
    def __init__(self, proc_loop, db_logger, polling_engine, model,
                 fault_reporter, interval=30.0):
        self._proc = proc_loop
        self._db = db_logger
        self._poller = polling_engine
        self._model = model
        self._faults = fault_reporter
        self._interval = interval
        self._thread = None
        self._running = False
        self._stop_event = threading.Event()

    def start(self):
        if self._running:
            return
        self._stop_event.clear()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="status-rpt")
        self._thread.start()

    def stop(self):
        # Interruptible + joined: a plain flag left the old thread sleeping up to
        # `interval` (30s) seconds, so rapid mode-toggling stacked live threads
        # that double-printed heartbeats. Wake it and wait for it to exit.
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    def _loop(self):
        while self._running:
            if self._stop_event.wait(self._interval):
                break
            if not self._running:
                break
            self._report()

    def _report(self):
        if self._proc.is_running and not self._proc.is_paused:
            state = "运行中"
        elif self._proc.is_paused:
            state = "已暂停"
        else:
            state = "已停止"
        plc = "已连接" if self._poller.plc_connected else "断开"
        mdl = self._model.version if self._model.loaded else "未加载"
        records = self._db.count_records()
        polls = self._poller.stats.get("total_polls", 0)
        errs = self._poller.stats.get("errors", 0)
        faults = len(self._faults.active_faults)
        # The console line bypasses the logging filter on purpose so the
        # operator sees a periodic heartbeat in normal mode (where the
        # console handler suppresses INFO). The mirror to the file logger
        # captures the same heartbeat in ldpj_backend.log for postmortems.
        msg = (f"{state} | PLC:{plc} | 模型:{mdl} | "
               f"轮询:{polls} 错误:{errs} | 记录:{records} | 故障:{faults}")
        print(f"\n[{time.strftime('%H:%M:%S')}] {msg}")
        logger.info("status: %s", msg)


# ── UI ─────────────────────────────────────────────────────────────────

def _print_banner(mode, plc_connected, model_loaded, model_version,
                  active_start, active_end):
    mode_str = "S7 (生产)" if mode == "s7" else "Mock (模拟)"
    plc_str = "已连接" if plc_connected else "未连接"
    mdl_str = f"已加载 ({model_version})" if model_loaded else "未加载"
    cabin_n = active_end - active_start + 1
    print()
    print("=" * 52)
    print("    Ldpj_backend v2.6.2 — 漏液检测系统")
    print("=" * 52)
    print(f"  运行模式:  {mode_str}")
    print(f"  PLC状态:   {plc_str}")
    print(f"  AI模型:    {mdl_str}")
    print(f"  活跃舱室:  Cabin[{active_start}]~Cabin[{active_end}] ({cabin_n} 个)")
    print("=" * 52)


def _select_mode() -> str:
    """Prompt user to select display mode, then show menu."""
    print()
    print("  请选择显示模式:")
    print("    [1] 正常模式 — 仅显示警告和错误, 每30秒报告状态")
    print("    [2] 调试模式 — 显示所有日志 (INFO/DEBUG)")
    try:
        choice = input("  请选择 (1/2, 默认1): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"
    mode = "debug" if choice == "2" else "normal"
    label = "调试模式" if mode == "debug" else "正常模式"
    print(f"  → {label}")
    return mode


def _print_menu():
    print()
    print("────────── 操作命令 ──────────")
    print("  s  恢复 采集与推理")
    print("  e  暂停 采集与推理")
    print("  x  导出数据到 CSV")
    print("  w  切换看门狗")
    print("  m  切换显示模式 (正常/调试)")
    print("  h  执行健康检查")
    print("  d  打印诊断信息")
    print("  q  退出程序")
    print("──────────────────────────────")
    print("  采集与推理已自动启动, 等待命令...")
    print()


# ── Main ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Ldpj_backend edge AI system")
    p.add_argument("--mode", choices=["s7", "mock"], default="mock")
    return p.parse_args()


def main():
    global logger
    args = parse_args()

    plc_cfg = load_plc_config()
    runtime_cfg = load_runtime_config()
    models_cfg = load_models_config()
    health_cfg = load_health_config()
    ipc_cfg = load_ipc_config()

    # Logging: file=INFO always, console controlled by mode
    logger = setup_logging(runtime_cfg.get("logging", {}))

    # ── Load v2.6 configs (profile / cabins / products) ────────────
    # Profile is required; cabins/products fall back to empty dicts so a
    # development environment without those files can still boot, with
    # M1's uncalibrated-cabin path triggering F011.
    try:
        cycle_profile = load_active_cycle_profile()
    except Exception as exc:
        logger.error("Failed to load active cycle profile: %s", exc)
        print(f"  错误: 周期配方加载失败 — {exc}")
        sys.exit(1)

    try:
        cabins_cfg = load_cabins_config()
    except FileNotFoundError as exc:
        logger.warning("cabins.yaml missing: %s. Using defaults.", exc)
        cabins_cfg = {}

    try:
        products_cfg = load_products_config()
    except FileNotFoundError as exc:
        logger.warning("products.yaml missing: %s. Using defaults.", exc)
        products_cfg = {}

    # ── 强制校验产品阈值 vs 系统分辨率 (README §5 / 部署说明 §3) ──────
    # q_threshold 必须 > A(估计分辨率); 建议 ≥ A_det(检出分辨率)。
    mi_cfg = runtime_cfg.get("model_inference", {}) or {}
    a_estimate = float(mi_cfg.get("a_estimate", 9.0e-3))
    a_det = float(mi_cfg.get("a_det", 1.56e-2))
    thr_errors, thr_warnings = validate_product_resolution(
        products_cfg, a_estimate, a_det)
    for w in thr_warnings:
        logger.warning("产品阈值建议: %s", w)
    if thr_errors:
        for e in thr_errors:
            logger.error("产品阈值非法: %s", e)
        print("  错误: 产品 Q_threshold 低于估计分辨率 A, 拒绝启动:")
        for e in thr_errors:
            print(f"    - {e}")
        print("  请修正 configs/products.yaml 后重试。")
        sys.exit(1)

    # ── Init subsystems ────────────────────────────────────────────
    fault_reporter = FaultReporter()

    alarm_pusher = AlarmPusher(ipc_cfg)
    if alarm_pusher.enabled:
        fault_reporter.register_callback(
            lambda ev: alarm_pusher.push_alarm(
                ev.fault.code, ev.message, ev.fault.level.value))

    polling_engine = PollingEngine(plc_cfg, mode=args.mode)
    polling_engine.start()

    cabin_cfg = plc_cfg.get("cabin_array", {})
    active_start = cabin_cfg.get("active_start", 1)
    active_end = cabin_cfg.get("active_end", 25)
    fsm_manager = CycleFSMManager(
        cabin_cfg.get("cabin_count", 26),
        cycle_profile,
        active_start=active_start,
        active_end=active_end,
    )

    # ── Load v2.6 regression models ────────────────────────────────
    m1_model = LinearRegressionM1(models_cfg, base_dir=PROJECT_ROOT)
    try:
        m1_model.load()
    except Exception as exc:
        logger.warning("M1 not loaded: %s. System will run but produce no Q_est.", exc)

    m2_model = XGBRegressorM2(models_cfg, base_dir=PROJECT_ROOT)
    try:
        m2_model.load()
    except Exception as exc:
        logger.warning("M2 not loaded: %s. Cross-check disabled.", exc)

    # ── v2.6.3 工况门: 运行工况 ↔ 模型标定一致性 (README/部署说明 R01) ──
    # 关闭"M1/M2 同向漂移、F010 无法察觉"的静默失配。仅在 M1 已加载时校验。
    if m1_model.loaded:
        active_op = cycle_profile.operating_point()
        op_gate = validate_operating_point(
            active_op, m1_model.operating_point, m2_model.operating_point)
        for w in op_gate["warnings"]:
            logger.warning("工况校验: %s", w)
        if op_gate["errors"]:
            for e in op_gate["errors"]:
                logger.error("工况校验失败: %s", e)
            print("  错误: 运行工况与模型标定不一致, 拒绝启动:")
            for e in op_gate["errors"]:
                print(f"    - {e}")
            print("  请修正 runtime.yaml 工况, 或部署对应工况的模型后重试。")
            sys.exit(1)
        # 执行动作: M1 确定性重缩放(间隔变化, 采样数保持) / M2 禁用(不可重缩放)
        if op_gate["m1_rescale_to"] is not None:
            m1_model.rescale_to_interval(op_gate["m1_rescale_to"])
        if op_gate["m2_disable_reason"]:
            m2_model.disable(op_gate["m2_disable_reason"])
        # k_ts 运行期不变量(重缩放后再校验 |β|/V_cabin 物理自洽)
        kts_errors = assert_kts_consistency(m1_model, cabins_cfg, active_op)
        if kts_errors:
            for e in kts_errors:
                logger.error("k_ts 自洽校验失败: %s", e)
            print("  错误: M1 系数与 V_cabin 物理自洽性(k_ts)校验失败, 拒绝启动:")
            for e in kts_errors:
                print(f"    - {e}")
            sys.exit(1)
        # 抛出工况告警故障(F013 间隔重缩放/M2禁用; F014 真空失配)
        for code, msg in op_gate["faults"]:
            fault_reporter.raise_fault(code, msg)

    db_logger = DatabaseLogger(
        runtime_cfg.get("database", {}).get("path", "ldpj_data.db"))

    result_sender = ResultSender(plc_cfg, polling_engine)
    # v2.6: async writeback so 25 simultaneous PROCESSING cabins don't
    # block the polling thread on snap7 RMW.
    result_sender.enable_async()

    health_checker = HealthChecker(health_cfg, fault_reporter)
    # HealthChecker still expects a single `model` reference for F002 — give
    # it M1, since M1 is the primary inference path.
    health_checker.set_references(
        polling_engine=polling_engine, model=m1_model,
        db_logger=db_logger, fsm_manager=fsm_manager)
    health_checker.start()

    api_server = APIServer(ipc_cfg)
    api_server.set_references(
        db_logger=db_logger, health_checker=health_checker,
        polling_engine=polling_engine, model=m1_model,
        fault_reporter=fault_reporter)
    api_server.start()

    proc_loop = ProcessingLoop(
        runtime_cfg=runtime_cfg,
        profile=cycle_profile,
        cabins_cfg=cabins_cfg,
        products_cfg=products_cfg,
        polling_engine=polling_engine,
        fsm_manager=fsm_manager,
        m1_model=m1_model,
        m2_model=m2_model,
        db_logger=db_logger,
        result_sender=result_sender,
        alarm_pusher=alarm_pusher,
        health_checker=health_checker,
        fault_reporter=fault_reporter,
    )

    # Auto-start processing
    proc_loop.start()

    status_reporter = StatusReporter(
        proc_loop, db_logger, polling_engine, m1_model, fault_reporter)

    # ── Suppress console during init, show banner ──────────────────
    set_console_mode("silent")
    time.sleep(0.3)

    _print_banner(
        mode=args.mode,
        plc_connected=polling_engine.plc_connected,
        model_loaded=m1_model.loaded,
        model_version=m1_model.version if m1_model.loaded else "",
        active_start=active_start,
        active_end=active_end)

    # ── Show menu first, then select mode ─────────────────────────
    _print_menu()
    display_mode = _select_mode()

    if display_mode == "debug":
        set_console_mode("debug")
    else:
        set_console_mode("normal")
        status_reporter.start()

    # ── Processing thread (created before shutdown so shutdown can join it) ──
    def _run_loop():
        try:
            while proc_loop.is_running:
                proc_loop.run_once()
        except Exception as exc:
            logger.error("Processing loop crashed: %s", exc, exc_info=True)

    loop_thread = threading.Thread(target=_run_loop, daemon=True, name="proc-loop")

    # ── Shutdown helper ────────────────────────────────────────────
    _shutting_down = threading.Event()

    def shutdown():
        # Guard against re-entrancy (SIGINT during shutdown, or 'q' then Ctrl-C).
        if _shutting_down.is_set():
            return
        _shutting_down.set()
        print("\n  正在关闭系统...")
        status_reporter.stop()
        proc_loop.stop()
        # Join the processing thread BEFORE closing the DB, so an in-flight
        # log_record() can't run against a closed sqlite connection.
        if loop_thread.is_alive():
            loop_thread.join(timeout=2.0)
        # Flush pending writebacks BEFORE polling stops so the writer thread
        # can still grab _io_lock from a live S7 connection.
        result_sender.shutdown()
        health_checker.stop()
        api_server.stop()
        polling_engine.stop()
        db_logger.close()
        print("  系统已关闭。")
        sys.exit(0)

    signal.signal(signal.SIGINT, lambda s, f: shutdown())
    signal.signal(signal.SIGTERM, lambda s, f: shutdown())

    loop_thread.start()

    # ── Command loop (main thread) ─────────────────────────────────
    while True:
        try:
            raw = sys.stdin.readline()
            if raw == "":
                # EOF: stdin is closed (systemd/headless service — stdin is
                # /dev/null). Do NOT busy-spin on readline() (100% CPU). Idle
                # until a signal triggers shutdown; processing keeps running in
                # the background thread.
                logger.info("stdin EOF; idling (processing continues; "
                            "send SIGTERM/SIGINT to stop)")
                while not _shutting_down.is_set():
                    time.sleep(3600)
                break
            line = raw.strip().lower()
            if not line:
                continue
            cmd = line[0]

            if cmd == "s":
                proc_loop.resume()
                print("  采集与推理已恢复")

            elif cmd == "e":
                proc_loop.pause()
                print("  采集与推理已暂停 (可按 x 导出数据)")

            elif cmd == "x":
                # Pause → export → auto-resume
                was_running = not proc_loop.is_paused
                if was_running:
                    proc_loop.pause()
                    time.sleep(0.2)  # Let current cycle finish
                try:
                    interactive_export(db_logger, base_dir=PROJECT_ROOT)
                finally:
                    if was_running:
                        proc_loop.resume()
                        print("  采集与推理已自动恢复")

            elif cmd == "m":
                if display_mode == "normal":
                    display_mode = "debug"
                    set_console_mode("debug")
                    status_reporter.stop()
                    print("  已切换到 [调试模式]")
                else:
                    display_mode = "normal"
                    set_console_mode("normal")
                    status_reporter.start()
                    print("  已切换到 [正常模式]")

            elif cmd == "w":
                on = proc_loop.toggle_watchdog()
                print(f"  看门狗: {'开启' if on else '关闭'}")

            elif cmd == "h":
                rpt = health_checker.run_all_checks()
                print(json.dumps(rpt, indent=2, ensure_ascii=False, default=str))

            elif cmd == "d":
                diag = proc_loop.get_diagnostics()
                print(json.dumps(diag, indent=2, ensure_ascii=False, default=str))

            elif cmd == "q":
                shutdown()

            else:
                print(f"  未知命令: '{cmd}'")
                _print_menu()

        except EOFError:
            break
        except KeyboardInterrupt:
            shutdown()
        except Exception as exc:
            print(f"  命令执行出错: {exc}")


if __name__ == "__main__":
    main()
