"""Main processing loop — orchestrates data flow from polling to inference (v2.6).

Pipeline per completed cycle:
    1. Drain new poll frames into the per-cabin FSMs.
    2. For each cabin in PROCESSING state:
       a. Compute v2.6.1 36-dim features (segment-by-angle, 5 sections).
       b. NO_BOTTLE early-out if hold_max < no_bottle_threshold.
       c. M1 (per-cabin linear regression) → q_est primary output.
       d. M2 (global XGBoost) when loaded → cross-check; raise F010 on
          disagreement above ``m_disagreement_threshold``.
       e. Compare q_est against the active product's q_threshold to label
          LEAK / OK; raise F012 when q_est is below A_resolution.
       f. Persist to DB (curves auto-compressed by storage layer).
       g. PLC write-back: cabinHealthStatus carries q_est (Pa·m³/s).
       h. Push alarm on LEAK.
    3. Reset any FSMs left in FAULT.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional

from configs.loaders import is_cabin_calibrated
from core.cycle_fsm import CycleFSMManager
from core.cycle_profile import CycleProfile
from core.feature_spec import FEATURE_ORDER_36D
from core.features import compute_features_v26, features_to_vector
from core.polling_engine import PollingEngine
from core.quality_flags import QF_SHORT_HOLD, compute_quality_flags
from core.rate_limit import warn_throttled
from health.fault_reporter import FaultReporter
from health.health_checker import HealthChecker
from integration.alarm_pusher import AlarmPusher
from integration.result_sender import ResultSender
from models.linear_regression_m1 import LinearRegressionM1
from models.xgb_regressor_m2 import XGBRegressorM2
from storage.database_logger import DatabaseLogger

logger = logging.getLogger(__name__)


# Label constants (PLC-visible, match v2.5 semantics for the bool field)
LABEL_LEAK = 0
LABEL_OK = 1
LABEL_NA = -1
LABEL_NO_BOTTLE = -2


class ProcessingLoop:
    """Main processing loop (v2.6 — Q regression).

    Parameters
    ----------
    runtime_cfg : dict
        Full runtime.yaml content. ``model_inference.a_resolution`` and
        ``model_inference.m_disagreement_threshold`` configure the v2.6
        decision logic; ``loop_interval`` and ``no_bottle_threshold``
        are unchanged from v2.5.
    profile : CycleProfile
        Active cycle profile (defines section boundaries for features).
    cabins_cfg : dict
        Loaded ``cabins.yaml``; used only to flag uncalibrated cabins.
    products_cfg : dict
        Loaded ``products.yaml``; the active product's ``q_threshold``
        drives LEAK/OK judgment.
    polling_engine, fsm_manager : as v2.5.
    m1_model, m2_model : v2.6 regression models. Either may be unloaded
        at startup; the loop degrades to LABEL_NA in that case rather
        than crashing.
    db_logger, result_sender, alarm_pusher, health_checker, fault_reporter:
        as v2.5.
    """

    def __init__(
        self,
        runtime_cfg: Dict[str, Any],
        profile: CycleProfile,
        cabins_cfg: Dict[str, Any],
        products_cfg: Dict[str, Any],
        polling_engine: PollingEngine,
        fsm_manager: CycleFSMManager,
        m1_model: LinearRegressionM1,
        m2_model: XGBRegressorM2,
        db_logger: DatabaseLogger,
        result_sender: ResultSender,
        alarm_pusher: AlarmPusher,
        health_checker: HealthChecker,
        fault_reporter: FaultReporter,
    ):
        self._cfg = runtime_cfg
        self._profile = profile
        self._cabins_cfg = cabins_cfg or {}
        self._products_cfg = products_cfg or {}
        self._poller = polling_engine
        self._fsm = fsm_manager
        self._m1 = m1_model
        self._m2 = m2_model
        self._db = db_logger
        self._sender = result_sender
        self._alarm = alarm_pusher
        self._health = health_checker
        self._reporter = fault_reporter

        # ── v2.6 inference parameters ─────────────────────────
        mi_cfg = runtime_cfg.get("model_inference", {}) or {}
        self._a_resolution = float(mi_cfg.get("a_resolution", 1.0e-5))
        self._m_disagreement_threshold = float(mi_cfg.get("m_disagreement_threshold", 0.20))

        # ── Active product ────────────────────────────────────
        self._current_product_id = self._products_cfg.get("default_product_id", "default")

        # ── Thin-wall micro-hole diameter (Yoshida choked-flow theory) ──
        # The second of the dual outputs: Q_est (leak rate) + d_est (equivalent
        # thin-wall pinhole diameter). Uses the operating point's vacuum plateau
        # as the downstream pressure so it tracks any vacuum change.
        self._p_chamber_pa = float(getattr(profile, "p_chamber_pa", 35000.0))
        self._p_atm_pa = float(getattr(profile, "p_atm_pa", 101325.0))

        # ── Misc ──────────────────────────────────────────────
        self._loop_interval = float(runtime_cfg.get("loop_interval", 0.05))
        self._no_bottle_threshold = float(runtime_cfg.get("no_bottle_threshold", 50.0))
        self._running = False
        self._paused = False
        self._watchdog = True
        # Seq cursor into the polling engine's frame buffer. -1 means
        # "no frames seen yet" (the engine's first frame has seq=0).
        self._last_poll_seq = -1

        # Surface model availability at construction time (one-shot fault).
        # HealthChecker.run_all_checks() also raises/resolves F002 dynamically.
        if not self._m1.loaded:
            logger.warning("M1 not loaded; system will run but produce no Q_est.")
            self._reporter.raise_fault("F002", "M1 模型未加载")

    # ── Lifecycle ──────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    def start(self) -> None:
        self._running = True
        self._paused = False
        logger.info("ProcessingLoop started (product=%s, A=%.2e, "
                    "M-disagree=%.0f%%)",
                    self._current_product_id, self._a_resolution,
                    self._m_disagreement_threshold * 100)

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

    def set_active_product(self, product_id: str) -> bool:
        """Switch the active product. Returns True if found, False otherwise."""
        if product_id in (self._products_cfg.get("products", {}) or {}):
            self._current_product_id = product_id
            logger.info("Active product set to %s", product_id)
            return True
        logger.warning("Unknown product_id: %s", product_id)
        return False

    # ── Main loop ──────────────────────────────────────────────

    def run_once(self) -> None:
        if not self._running or self._paused:
            time.sleep(self._loop_interval)
            return

        try:
            self._feed_fsm()
        except Exception as exc:
            logger.error("feed_fsm error: %s", exc, exc_info=True)

        # Per-cabin isolation: an exception while processing one cabin must NOT
        # abort the loop and leave that cabin stuck in PROCESSING (which would
        # re-raise every tick and starve all other cabins). Log, reset, move on.
        for cabin_id in self._fsm.get_processing_cabins():
            try:
                self._process_cabin(cabin_id)
            except Exception as exc:
                logger.error("Cabin %d processing error: %s; resetting FSM",
                             cabin_id, exc, exc_info=True)
                self._reporter.raise_fault("F006", f"舱 {cabin_id} 处理异常: {exc}")
                try:
                    self._fsm.fsms[cabin_id].reset()
                except Exception:
                    pass

        for cabin_id in self._fsm.get_fault_cabins():
            try:
                self._handle_fault(cabin_id)
            except Exception as exc:
                logger.error("Cabin %d fault-handling error: %s", cabin_id, exc)

        time.sleep(self._loop_interval)

    # ── Internals ─────────────────────────────────────────────

    def _feed_fsm(self) -> None:
        frames = self._poller.drain_frames_since_seq(self._last_poll_seq)
        if not frames:
            return
        for frame in frames:
            cabin_map = {c.cabin_index: c for c in frame.cabins}
            self._fsm.update_all(cabin_map)
        self._last_poll_seq = frames[-1].seq

    def _q_threshold_for(self, product_id: str) -> Optional[float]:
        """Look up Q_threshold for the active product, or None if absent."""
        product = (self._products_cfg.get("products", {}) or {}).get(product_id)
        if not product:
            return None
        thr = product.get("q_threshold")
        return float(thr) if thr is not None else None

    def _predict_q(
        self,
        feats: Dict[str, float],
        feature_vector_36d: List[float],
        cabin_id: int,
    ) -> Dict[str, Any]:
        """Run M1 + M2 and fuse. Returns a dict with q_est, m1_q, m2_q,
        m_disagreement, q_uncertainty, cabin_calibrated, valid, below_resolution."""
        if not self._m1.loaded:
            return {
                "q_est": 0.0, "valid": False,
                "m1_q": 0.0, "m2_q": None,
                "m_disagreement": 0.0,
                "q_uncertainty": float("inf"),
                "cabin_calibrated": False,
                "below_resolution": True,
            }

        primary_section = self._m1.primary_section
        primary_slope = float(feats.get(f"{primary_section}_trend_slope", 0.0))

        m1_result = self._m1.predict(primary_slope, cabin_id)
        m1_q = float(m1_result["q_est"])
        m1_calibrated = bool(m1_result["cabin_calibrated"])

        # Safety net: a non-finite Q (NaN/inf) — e.g. a corrupt NaN pressure on
        # the S7 wire propagating through the features — must NEVER fall through
        # to an OK verdict (NaN > threshold is False → would silently pass a
        # leaker). Force N/A and raise the sensor-data fault instead.
        if not math.isfinite(m1_q):
            warn_throttled(
                "processing_loop.nonfinite_q",
                "Cabin %d: non-finite Q (NaN/inf) from M1 (slope=%.3e); "
                "likely corrupt pressure data. Forcing N/A.",
                cabin_id, primary_slope,
            )
            self._reporter.raise_fault("F003", f"舱 {cabin_id}: 压力/特征出现 NaN/inf")
            return {
                "q_est": 0.0, "valid": False,
                "m1_q": 0.0, "m2_q": None, "m_disagreement": 0.0,
                "q_uncertainty": float("inf"),
                "cabin_calibrated": m1_calibrated, "below_resolution": True,
            }

        # Negative q_est is physically impossible (would mean pressure rising
        # during hold) but can happen due to fit noise around zero. Below
        # the resolution check will treat |m1_q| anyway; we just log it.
        if m1_q < 0 and abs(m1_q) > self._a_resolution:
            logger.debug(
                "Cabin %d: M1 returned negative Q=%.3e (slope=%.3e). "
                "Likely a noisy fit near zero; treated as no-leak.",
                cabin_id, m1_q, primary_slope,
            )

        # F011 (uncalibrated cabin): only raise if cabins.yaml also says so,
        # to avoid double-reporting when M1 just hasn't been retrained yet.
        if not m1_calibrated and not is_cabin_calibrated(self._cabins_cfg, cabin_id):
            self._reporter.raise_fault("F011", f"舱 {cabin_id} 未标定")

        # M2 cross-check
        m2_q: Optional[float] = None
        if self._m2.loaded:
            try:
                m2_result = self._m2.predict(feature_vector_36d)
                if m2_result["valid"]:
                    m2_q = float(m2_result["q_est"])
            except Exception as exc:
                logger.warning("M2 predict failed: %s", exc)

        if m2_q is not None and abs(m1_q) > 1e-12:
            disagreement = abs(m2_q - m1_q) / abs(m1_q)
        else:
            disagreement = 0.0

        if m2_q is not None and disagreement > self._m_disagreement_threshold:
            logger.warning(
                "Cabin %d: M1/M2 disagree %.1f%% (M1=%.3e, M2=%.3e)",
                cabin_id, disagreement * 100, m1_q, m2_q,
            )
            self._reporter.raise_fault(
                "F010",
                f"舱 {cabin_id}: M1/M2 漏率估计差异 {disagreement * 100:.1f}%",
            )

        below_resolution = abs(m1_q) < self._a_resolution
        return {
            "q_est": m1_q,         # M1 is the primary estimator
            "valid": True,
            "m1_q": m1_q,
            "m2_q": m2_q,
            "m_disagreement": disagreement,
            "q_uncertainty": float(m1_result["uncertainty"]),
            "cabin_calibrated": m1_calibrated,
            "below_resolution": below_resolution,
        }

    def _process_cabin(self, cabin_id: int) -> None:
        """v2.6: full-cycle features + dual-track Q regression + Q-threshold judgment."""
        fsm = self._fsm.fsms[cabin_id]
        data = fsm.harvest()

        if len(data.pressures) < 2:
            logger.warning("Cabin %d: insufficient data (%d points), skipping",
                           cabin_id, len(data.pressures))
            fsm.reset()
            return

        t0 = time.perf_counter()

        # ── Feature extraction (36-dim) ───────────────────────
        feats = compute_features_v26(
            data.pressures, data.angles, cabin_id, self._profile,
        )
        feature_vector = features_to_vector(feats, mode="36d")
        duration_s = (data.timestamps[-1] - data.timestamps[0]) if len(data.timestamps) > 1 else 0.0

        # ── Per-cycle data-quality bitmask (for DB column) ────
        quality_flags = compute_quality_flags(feats)
        if quality_flags & QF_SHORT_HOLD:
            # hold_trend_slope is M1's only signal; warn (rate-limited) so
            # operators see when the primary-section data is degraded.
            warn_throttled(
                "processing_loop.short_hold",
                "Cabin %d: hold section had < 2 points; M1 slope unreliable",
                cabin_id,
            )

        # ── NO_BOTTLE detection: hold-section max < threshold ─
        hold_max = float(feats.get("hold_max", 0.0))
        if hold_max < self._no_bottle_threshold:
            self._handle_no_bottle(cabin_id, fsm, data, feats, duration_s,
                                   quality_flags=quality_flags)
            return

        # ── Q inference ───────────────────────────────────────
        q_result = self._predict_q(feats, feature_vector, cabin_id)
        q_threshold = self._q_threshold_for(self._current_product_id)

        if not q_result["valid"]:
            label = LABEL_NA
            label_str = "N/A"
        elif q_result["below_resolution"]:
            label = LABEL_NA
            label_str = "BELOW_A"
            self._reporter.raise_fault(
                "F012",
                f"舱 {cabin_id}: Q={q_result['q_est']:.2e} 低于分辨率 A={self._a_resolution:.2e}",
            )
        elif q_threshold is None:
            label = LABEL_NA
            label_str = "NO_THRESHOLD"
            logger.warning("Active product %s has no q_threshold; cannot judge",
                           self._current_product_id)
        elif q_result["q_est"] > q_threshold:
            label = LABEL_LEAK
            label_str = "LEAK"
        else:
            label = LABEL_OK
            label_str = "OK"

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._health.report_inference_latency(elapsed_ms)

        # ── Second output: thin-wall pinhole equivalent diameter ──
        # d (µm) via Yoshida choked-flow theory. Only meaningful when a verdict
        # was actually formed (OK/LEAK) with positive Q; else 0.
        q_est_val = float(q_result.get("q_est", 0.0) or 0.0)
        d_est = (self._q_to_d_thinwall(q_est_val)
                 if label in (LABEL_OK, LABEL_LEAK) and q_est_val > 0 else 0.0)

        # ── DB logging ────────────────────────────────────────
        try:
            self._db.log_record(
                cavity_id=cabin_id,
                pressures=data.pressures,
                angles=data.angles,
                ai_values=data.ai_values,
                positions=data.positions,
                features=feats,
                label=label,
                # `probability` column kept for back-compat; v2.6 stores Q_est
                probability=q_result.get("q_est", 0.0),
                # `confidence` repurposed to (1 - relative uncertainty), clipped
                confidence=self._confidence_from_q(q_result),
                model_version=self._m1.version if self._m1.loaded else "none",
                duration_s=duration_s,
                leak_valve_status=data.leak_valve_status,
                end_angle=data.end_angle,
                cycle_profile_id=data.cycle_profile_id,
                q_est=q_result.get("q_est"),
                q_threshold=q_threshold,
                q_uncertainty=q_result.get("q_uncertainty"),
                m1_q=q_result.get("m1_q"),
                m2_q=q_result.get("m2_q"),
                m_disagreement=q_result.get("m_disagreement"),
                product_id=self._current_product_id,
                quality_flags=quality_flags,
                d_est=d_est,
            )
        except Exception as exc:
            logger.error("DB logging failed for cabin %d: %s", cabin_id, exc)
            self._reporter.raise_fault("F006", f"数据库写入失败: {exc}")

        # ── PLC write-back: leakRate = Q_est + leakHoleDiameter = d_est ──
        # Dual result; the PLC forms its own verdict (leakTestResult_edge) from
        # Q+d. Health status is not sent.
        if label in (LABEL_OK, LABEL_LEAK):
            try:
                self._sender.write_result(cabin_id, q_result["q_est"], d_est)
            except Exception as exc:
                logger.error("PLC write-back failed for cabin %d: %s", cabin_id, exc)

        # ── Alarm push (leak only) ────────────────────────────
        if label == LABEL_LEAK:
            self._alarm.push_leak_alarm(cabin_id, q_result["q_est"])

        logger.info(
            "Cabin %d: %s (Q=%.3e, d=%.1fum, threshold=%.3e, M1=%.3e, M2=%s, %.1fms)",
            cabin_id, label_str,
            q_result.get("q_est", 0.0), d_est, q_threshold or 0.0,
            q_result.get("m1_q", 0.0),
            f"{q_result['m2_q']:.3e}" if q_result.get("m2_q") is not None else "n/a",
            elapsed_ms,
        )

        fsm.reset()

    def _handle_no_bottle(self, cabin_id, fsm, data, feats, duration_s,
                          quality_flags: int = 0) -> None:
        logger.info("Cabin %d: NO_BOTTLE (hold_max=%.1f < %.1f, points=%d)",
                    cabin_id, feats.get("hold_max", 0.0),
                    self._no_bottle_threshold, len(data.pressures))
        try:
            self._db.log_record(
                cavity_id=cabin_id,
                pressures=data.pressures, angles=data.angles,
                ai_values=data.ai_values, positions=data.positions,
                features=feats,
                label=LABEL_NO_BOTTLE,
                probability=0.0, confidence=0.0,
                model_version=self._m1.version if self._m1.loaded else "none",
                duration_s=duration_s,
                leak_valve_status=data.leak_valve_status,
                end_angle=data.end_angle,
                cycle_profile_id=data.cycle_profile_id,
                q_est=0.0, q_threshold=None, q_uncertainty=None,
                m1_q=0.0, m2_q=None, m_disagreement=0.0,
                product_id=self._current_product_id,
                quality_flags=quality_flags,
                d_est=0.0,
            )
        except Exception as exc:
            logger.error("DB logging failed for cabin %d: %s", cabin_id, exc)
            self._reporter.raise_fault("F006", f"数据库写入失败: {exc}")
        fsm.reset()

    def _q_to_d_thinwall(self, q_est: float) -> float:
        """Thin-wall pinhole equivalent diameter (µm) from Q_est via Yoshida
        choked-flow theory. Upstream = atmosphere, downstream = the operating
        point's vacuum plateau (so d tracks any vacuum change). 0.0 if Q ≤ 0.
        """
        if q_est <= 0:
            return 0.0
        from core.q_d_conversion import q_to_d_choked
        try:
            return round(
                q_to_d_choked(q_est, p_u=self._p_atm_pa, p_d=self._p_chamber_pa), 2)
        except Exception as exc:
            logger.debug("q_to_d_choked failed for Q=%.3e: %s", q_est, exc)
            return 0.0

    @staticmethod
    def _confidence_from_q(q_result: Dict[str, Any]) -> float:
        """Map (1-σ absolute uncertainty / |q_est|) into a confidence score [0,1].

        High relative uncertainty → low confidence, and vice versa. The
        ``q_uncertainty`` field carries the *absolute* 1-σ value (Pa·m³/s);
        we divide by |q_est| here to get the relative form.
        """
        unc_abs = q_result.get("q_uncertainty", float("inf"))
        q = q_result.get("q_est", 0.0) or 1.0
        if unc_abs == float("inf") or abs(q) < 1e-12:
            return 0.0
        rel_unc = abs(unc_abs) / abs(q)
        return float(max(0.0, min(1.0, 1.0 - rel_unc)))

    def _handle_fault(self, cabin_id: int) -> None:
        # Per-cycle FAULT (e.g. collection timeout) is recoverable and common
        # (empty position, transient comms hiccup). Log + reset; persistent
        # stuck cabins are tracked separately by HealthChecker._check_fsm via F009.
        logger.warning("Cabin %d in FAULT state, resetting", cabin_id)
        self._fsm.fsms[cabin_id].clear_fault()

    # ── Diagnostics ─────────────────────────────────────────

    def get_diagnostics(self) -> Dict[str, Any]:
        cabin_states = {}
        for cid, fsm in self._fsm.fsms.items():
            cabin_states[cid] = {"state": fsm.state.value, "points": fsm.point_count}
        return {
            "running": self._running,
            "paused": self._paused,
            "watchdog": self._watchdog,
            "no_bottle_threshold": self._no_bottle_threshold,
            "a_resolution": self._a_resolution,
            "m_disagreement_threshold": self._m_disagreement_threshold,
            "current_product_id": self._current_product_id,
            "last_poll_seq": self._last_poll_seq,
            "poller_buffer": self._poller.buffer_length,
            "poller_stats": self._poller.stats,
            "cabin_states": cabin_states,
            "m1_loaded": self._m1.loaded,
            "m1_version": self._m1.version if self._m1.loaded else "none",
            "m2_loaded": self._m2.loaded,
            "m2_version": self._m2.version if self._m2.loaded else "none",
            "profile_id": self._profile.profile_id if self._profile else "none",
        }
