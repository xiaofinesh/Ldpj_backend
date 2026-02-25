"""Main processing loop – orchestrates data flow from polling to inference.

This module ties together the polling engine, cycle FSM, feature computation,
model inference, database logging, result writing, and alarm pushing.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from core.cycle_fsm import CycleFSMManager, CycleState
from core.features import compute_features, features_to_vector
from core.polling_engine import PollingEngine
from health.fault_reporter import FaultReporter
from health.health_checker import HealthChecker
from integration.alarm_pusher import AlarmPusher
from integration.result_sender import ResultSender
from models.supervised_xgb import SupervisedXGB
from storage.database_logger import DatabaseLogger

logger = logging.getLogger(__name__)


class ProcessingLoop:
    """Main processing loop.

    Parameters
    ----------
    runtime_cfg : dict
        The full content of ``runtime.yaml``.
    polling_engine : PollingEngine
    fsm_manager : CycleFSMManager
    model : SupervisedXGB
    db_logger : DatabaseLogger
    result_sender : ResultSender
    alarm_pusher : AlarmPusher
    health_checker : HealthChecker
    fault_reporter : FaultReporter
    """

    def __init__(
        self,
        runtime_cfg: Dict[str, Any],
        polling_engine: PollingEngine,
        fsm_manager: CycleFSMManager,
        model: SupervisedXGB,
        db_logger: DatabaseLogger,
        result_sender: ResultSender,
        alarm_pusher: AlarmPusher,
        health_checker: HealthChecker,
        fault_reporter: FaultReporter,
    ):
        self._cfg = runtime_cfg
        self._poller = polling_engine
        self._fsm = fsm_manager
        self._model = model
        self._db = db_logger
        self._sender = result_sender
        self._alarm = alarm_pusher
        self._health = health_checker
        self._reporter = fault_reporter

        self._threshold = runtime_cfg.get("threshold", 0.3)
        self._feature_mode = runtime_cfg.get("feature_mode", "7d")
        self._loop_interval = runtime_cfg.get("loop_interval", 0.05)
        self._running = False
        self._paused = False
        self._watchdog = True
        self._last_poll_ts = 0.0

    # -- lifecycle -----------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    def start(self) -> None:
        self._running = True
        self._paused = False
        logger.info("ProcessingLoop started")

    def stop(self) -> None:
        self._running = False
        logger.info("ProcessingLoop stopped")

    def pause(self) -> None:
        self._paused = True
        logger.info("ProcessingLoop paused")

    def resume(self) -> None:
        self._paused = False
        logger.info("ProcessingLoop resumed")

    def toggle_watchdog(self) -> bool:
        self._watchdog = not self._watchdog
        logger.info("Watchdog %s", "ON" if self._watchdog else "OFF")
        return self._watchdog

    # -- main loop -----------------------------------------------------------

    def run_once(self) -> None:
        """Execute one iteration of the processing loop.

        Called by main.py in a while-loop.
        """
        if not self._running or self._paused:
            time.sleep(self._loop_interval)
            return

        try:
            # 1. Feed latest polling data to FSMs
            self._feed_fsm()

            # 2. Check for cabins ready for processing
            ready = self._fsm.get_processing_cabins()
            for cabin_id in ready:
                self._process_cabin(cabin_id)

            # 3. Handle faulted cabins
            faulted = self._fsm.get_fault_cabins()
            for cabin_id in faulted:
                self._handle_fault(cabin_id)

        except Exception as exc:
            logger.error("Processing loop error: %s", exc, exc_info=True)

        time.sleep(self._loop_interval)

    # -- internal ------------------------------------------------------------

    def _feed_fsm(self) -> None:
        """Get new frames from the poller and update all FSMs."""
        frames = self._poller.drain_frames_since(self._last_poll_ts)
        if not frames:
            return

        for frame in frames:
            cabin_map = {c.cabin_index: c for c in frame.cabins}
            self._fsm.update_all(cabin_map)

        self._last_poll_ts = frames[-1].timestamp

    def _process_cabin(self, cabin_id: int) -> None:
        """Run feature extraction, inference, logging, and write-back."""
        fsm = self._fsm.fsms[cabin_id]
        data = fsm.harvest()

        if len(data.pressures) < 2:
            logger.warning("Cabin %d: insufficient data (%d points), skipping", cabin_id, len(data.pressures))
            fsm.reset()
            return

        t0 = time.perf_counter()

        # Feature extraction
        feats = compute_features(data.pressures, cabin_id)
        vec = features_to_vector(feats, mode=self._feature_mode)

        # Inference
        if self._model.loaded:
            result = self._model.predict(vec, threshold=self._threshold)
        else:
            result = {"label": -1, "probability": 0.0, "confidence": 0.0}
            logger.warning("Cabin %d: model not loaded, skipping inference", cabin_id)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._health.report_inference_latency(elapsed_ms)

        duration_s = (data.timestamps[-1] - data.timestamps[0]) if len(data.timestamps) > 1 else 0.0

        # Database logging
        try:
            self._db.log_record(
                cavity_id=cabin_id,
                pressures=data.pressures,
                angles=data.angles,
                ai_values=data.ai_values,
                positions=data.positions,
                features=feats,
                label=result["label"],
                probability=result["probability"],
                confidence=result["confidence"],
                model_version=self._model.version,
                duration_s=duration_s,
            )
        except Exception as exc:
            logger.error("DB logging failed for cabin %d: %s", cabin_id, exc)
            self._reporter.raise_fault("F006", f"数据库写入失败: {exc}")

        # PLC write-back
        try:
            self._sender.write_result(result["label"], result["probability"])
        except Exception as exc:
            logger.error("PLC write-back failed for cabin %d: %s", cabin_id, exc)

        # Alarm push (if leak detected)
        if result["label"] == 0:
            self._alarm.push_leak_alarm(cabin_id, result["probability"])

        # Log summary
        label_str = "OK" if result["label"] == 1 else ("LEAK" if result["label"] == 0 else "N/A")
        logger.info(
            "Cabin %d: %s (prob=%.4f, conf=%.4f, points=%d, %.1fms)",
            cabin_id, label_str, result["probability"], result["confidence"],
            len(data.pressures), elapsed_ms,
        )

        # Reset FSM
        fsm.reset()

    def _handle_fault(self, cabin_id: int) -> None:
        """Handle a cabin whose FSM entered FAULT state."""
        logger.warning("Cabin %d in FAULT state, resetting", cabin_id)
        self._reporter.raise_fault("F009", f"舱室 {cabin_id} 状态机故障")
        fsm = self._fsm.fsms[cabin_id]
        fsm.clear_fault()

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return a snapshot of the processing loop's internal state."""
        cabin_states = {}
        for cid, fsm in self._fsm.fsms.items():
            cabin_states[cid] = {
                "state": fsm.state.value,
                "points": fsm.point_count,
            }
        return {
            "running": self._running,
            "paused": self._paused,
            "watchdog": self._watchdog,
            "threshold": self._threshold,
            "feature_mode": self._feature_mode,
            "last_poll_ts": self._last_poll_ts,
            "poller_buffer": self._poller.buffer_length,
            "poller_stats": self._poller.stats,
            "cabin_states": cabin_states,
            "model_loaded": self._model.loaded,
            "model_version": self._model.version,
        }
