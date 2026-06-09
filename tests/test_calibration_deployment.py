"""Lock-in tests for the v2.6.2-cal20260605 calibration deployment.

These guard the four coupled P0 facts (README §2 / 部署说明.md):
  1. runtime.yaml hold window is [93,283) with boundaries [0/73/93/283/300/360].
  2. The deployed M1 table loads: 25 cabins, β ∈ [−0.2352,−0.2092].
  3. The deployed M2 bundle loads + predicts (model+scaler+metadata coupled).
  4. m1_coefficients.json β and cabins.yaml V_cabin are physically self-
     consistent: k_ts = |β| / V_cabin ≈ 1014 across all 25 cabins (catches
     risks R01 / R05 — coefficient-table vs V_cabin drift).

Plus the README §5 product-threshold-vs-resolution validation rule.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

import dataclasses

from configs.loaders import (
    assert_kts_consistency,
    load_active_cycle_profile,
    load_cabins_config,
    load_models_config,
    load_products_config,
    load_runtime_config,
    get_v_cabin,
    validate_operating_point,
    validate_product_resolution,
)
from models.linear_regression_m1 import LinearRegressionM1

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── 1. Segmentation window ──────────────────────────────────────────────

class TestHoldWindow:
    def test_boundaries_match_calibration(self):
        p = load_active_cycle_profile()
        assert p.sections["baseline_pre"] == (0.0, 73.0)
        assert p.sections["evac"] == (73.0, 93.0)
        assert p.sections["hold"] == (93.0, 283.0)        # ★ 标定窗
        assert p.sections["release"] == (283.0, 300.0)
        assert p.sections["baseline_post"] == (300.0, 360.0)
        assert p.primary_section == "hold"

    def test_hold_end_below_release_onset(self):
        """Hold must end before the break-vacuum onset (median 287.5°)."""
        p = load_active_cycle_profile()
        assert p.sections["hold"][1] <= 287.5


# ── 2. Deployed M1 ──────────────────────────────────────────────────────

class TestDeployedM1:
    def test_loads_25_cabins(self):
        m1 = LinearRegressionM1(load_models_config(), base_dir=_PROJECT_ROOT)
        m1.load()
        assert m1.loaded
        assert m1.version == "v2.6.2-cal20260605"
        assert len(m1.calibrated_cabins) == 25
        assert m1.primary_section == "hold"

    def test_beta_range(self):
        """β ∈ [−0.2352,−0.2092]; recover |β| via predict on slope=-1."""
        m1 = LinearRegressionM1(load_models_config(), base_dir=_PROJECT_ROOT)
        m1.load()
        for cid in range(1, 26):
            r = m1.predict(-1.0, cid)         # q = β·(-1)+α ≈ |β| (|α| ≪ |β|)
            assert 0.205 < r["q_est"] < 0.240, f"cabin {cid}: |β|≈{r['q_est']}"
        # A representative ~100 μm hold slope reproduces Q ≈ 0.0328
        r = m1.predict(-0.143, 1)
        assert r["cabin_calibrated"] is True
        assert 0.025 < r["q_est"] < 0.04

    def test_predict_is_positive_for_leak_slope(self):
        m1 = LinearRegressionM1(load_models_config(), base_dir=_PROJECT_ROOT)
        m1.load()
        # Negative hold slope (vacuum decaying) → positive Q for every cabin
        for cid in range(1, 26):
            assert m1.predict(-0.2, cid)["q_est"] > 0


# ── 3. Deployed M2 (needs xgboost + joblib) ─────────────────────────────

class TestDeployedM2:
    def test_loads_and_predicts(self):
        pytest.importorskip("xgboost")
        pytest.importorskip("joblib")
        from models.xgb_regressor_m2 import XGBRegressorM2
        from core.feature_spec import FEATURE_ORDER_36D

        m2 = XGBRegressorM2(load_models_config(), base_dir=_PROJECT_ROOT)
        m2.load()
        assert m2.loaded
        assert m2.version == "v2.6.2-cal20260605"
        assert len(m2.feature_subset) == 20
        assert m2.log_space is True

        vec = [0.0] * 36
        vec[FEATURE_ORDER_36D.index("hold_trend_slope")] = -0.143
        vec[FEATURE_ORDER_36D.index("hold_variance")] = 2.0
        vec[FEATURE_ORDER_36D.index("hold_difference")] = 8.0
        out = m2.predict(vec)
        assert out["valid"] is True
        assert out["q_est"] > 0


# ── 4. β ↔ V_cabin physical self-consistency (R01 / R05) ────────────────

class TestKtsConsistency:
    def test_kts_across_all_cabins(self):
        """k_ts = |β| / V_cabin must be ≈ 1014 (CV tiny) for all 25 cabins."""
        m1 = LinearRegressionM1(load_models_config(), base_dir=_PROJECT_ROOT)
        m1.load()
        cabins = load_cabins_config()
        kts = []
        for cid in range(1, 26):
            # Recover |β| from the table via predict on slope=-1 (q=beta*-1+alpha)
            r = m1.predict(-1.0, cid)
            beta_abs = abs(r["q_est"])  # ≈ |β| since |α| ≪ |β|
            v_cabin, _ = get_v_cabin(cabins, cid)
            kts.append(beta_abs / v_cabin)
        median = sorted(kts)[len(kts) // 2]
        assert 1000.0 < median < 1030.0, f"median k_ts={median}"
        # Every cabin within a few % of the theoretical 1000 Pa/s·(mbar/sample)
        for cid, k in enumerate(kts, start=1):
            assert 990.0 < k < 1040.0, f"cabin {cid}: k_ts={k}"


# ── 5. Product threshold vs resolution (README §5) ──────────────────────

class TestProductResolutionValidation:
    def _ad(self):
        mi = load_runtime_config()["model_inference"]
        return float(mi["a_estimate"]), float(mi["a_det"])

    def test_shipped_products_pass(self):
        a_est, a_det = self._ad()
        errs, _ = validate_product_resolution(load_products_config(), a_est, a_det)
        assert errs == []

    def test_threshold_below_A_is_error(self):
        a_est, a_det = self._ad()
        cfg = {"products": {"BAD": {"q_threshold": a_est / 2}}}
        errs, warns = validate_product_resolution(cfg, a_est, a_det)
        assert len(errs) == 1 and "BAD" in errs[0]

    def test_threshold_between_A_and_Adet_is_warning(self):
        a_est, a_det = self._ad()
        mid = (a_est + a_det) / 2.0
        cfg = {"products": {"MID": {"q_threshold": mid}}}
        errs, warns = validate_product_resolution(cfg, a_est, a_det)
        assert errs == [] and len(warns) == 1 and "MID" in warns[0]

    def test_threshold_above_Adet_is_clean(self):
        a_est, a_det = self._ad()
        cfg = {"products": {"GOOD": {"q_threshold": a_det * 2}}}
        errs, warns = validate_product_resolution(cfg, a_est, a_det)
        assert errs == [] and warns == []

    def test_missing_threshold_skipped(self):
        a_est, a_det = self._ad()
        cfg = {"products": {"NOTHR": {"name": "x"}}}
        errs, warns = validate_product_resolution(cfg, a_est, a_det)
        assert errs == [] and warns == []


# ── 6. Operating-point gate (v2.6.3) ────────────────────────────────────

def _deployed_m1():
    m1 = LinearRegressionM1(load_models_config(), base_dir=_PROJECT_ROOT)
    m1.load()
    return m1


class TestOperatingPointGate:
    """The startup gate: deployed point passes; drift is rescaled or refused."""

    def _active(self, **overrides):
        base = load_active_cycle_profile().operating_point()
        return dataclasses.replace(base, **overrides)

    def test_a_deployed_point_passes_clean(self):
        m1 = _deployed_m1()
        assert m1.operating_point is not None  # artifact carries the fingerprint
        r = validate_operating_point(m1.operating_point, m1.operating_point, m1.operating_point)
        assert r["errors"] == []
        assert r["m1_rescale_to"] is None
        assert r["m2_disable_reason"] is None
        assert r["faults"] == []

    def test_b_interval_change_count_preserved_rescales(self):
        m1 = _deployed_m1()
        # density-preserving: halve interval AND cycle → in-hold count fixed (37)
        active = self._active(interval_s=0.05, cycle_total_ms=3450)
        r = validate_operating_point(active, m1.operating_point, m1.operating_point)
        assert r["errors"] == []
        assert r["m1_rescale_to"] == pytest.approx(0.05)
        assert r["m2_disable_reason"] is not None
        assert any(code == "F013" for code, _ in r["faults"])

    def test_c_interval_change_count_drops_refuses(self):
        m1 = _deployed_m1()
        # fixed cycle_total_ms → 70 samples span a shorter arc, in-hold 37→34
        active = self._active(interval_s=0.05)
        r = validate_operating_point(active, m1.operating_point, m1.operating_point)
        assert len(r["errors"]) == 1
        assert r["m1_rescale_to"] is None  # the ripple: refuse, don't rescale

    def test_c2_rate_change_cycle_total_ms_refuses(self):
        """调整产量(转速): cycle_total_ms changes while interval_s stays — this
        moves the in-hold count (37→35) and MUST refuse, even though interval
        and sections match. (Regression guard for the gate's independent
        in-hold-count branch.)"""
        m1 = _deployed_m1()
        active = self._active(cycle_total_ms=6700)  # interval unchanged
        assert active.in_hold_sample_count != m1.operating_point.in_hold_sample_count
        r = validate_operating_point(active, m1.operating_point, m1.operating_point)
        assert len(r["errors"]) >= 1
        assert r["m1_rescale_to"] is None
        assert r["faults"] == []

    def test_d_hold_window_mismatch_refuses(self):
        m1 = _deployed_m1()
        active = self._active(
            hold_window_deg=(90.0, 290.0),
            sections={
                "baseline_pre": (0.0, 75.0), "evac": (75.0, 90.0),
                "hold": (90.0, 290.0), "release": (290.0, 304.0),
                "baseline_post": (304.0, 360.0),
            },
        )
        r = validate_operating_point(active, m1.operating_point, m1.operating_point)
        assert len(r["errors"]) >= 1
        assert r["m1_rescale_to"] is None

    def test_e_vacuum_only_warns_not_refuses(self):
        m1 = _deployed_m1()
        active = self._active(p_chamber_pa=45000.0)
        r = validate_operating_point(active, m1.operating_point, m1.operating_point)
        assert r["errors"] == []          # M1 is slope-invariant → no refuse
        assert any(code == "F014" for code, _ in r["faults"])

    def test_missing_m1_operating_point_is_fatal(self):
        active = load_active_cycle_profile().operating_point()
        r = validate_operating_point(active, None, None)
        assert len(r["errors"]) == 1


class TestM1Rescale:
    """The deterministic, no-retrain β rescale (M1 only)."""

    def test_beta_scales_alpha_fixed(self):
        m1 = _deployed_m1()
        before = m1.predict(-1.0, 1)["q_est"]      # ≈ -β1 + α1
        # capture raw beta/alpha
        b0 = float(m1._coefs[1]["beta"]); a0 = float(m1._coefs[1]["alpha"])
        ub0 = float(m1._coefs[1]["u_beta"])
        factor = m1.rescale_to_interval(0.05)       # interval_cal 0.1 → 0.05
        assert factor == pytest.approx(2.0)
        assert m1._coefs[1]["beta"] == pytest.approx(b0 * 2.0)
        assert m1._coefs[1]["u_beta"] == pytest.approx(ub0 * 2.0)
        assert m1._coefs[1]["alpha"] == pytest.approx(a0)   # α unchanged

    def test_rescale_is_idempotent(self):
        m1 = _deployed_m1()
        m1.rescale_to_interval(0.05)
        b1 = float(m1._coefs[1]["beta"])
        m1.rescale_to_interval(0.05)               # second call → no-op
        assert m1._coefs[1]["beta"] == pytest.approx(b1)

    def test_kts_holds_after_rescale(self):
        """Post-rescale k_ts must track the NEW operating point's 100/interval."""
        m1 = _deployed_m1()
        cabins = load_cabins_config()
        active = dataclasses.replace(
            load_active_cycle_profile().operating_point(),
            interval_s=0.05, cycle_total_ms=3450)
        m1.rescale_to_interval(0.05)
        errs = assert_kts_consistency(m1, cabins, active)   # theory now 2000
        assert errs == []

    def test_kts_at_deployed_point(self):
        m1 = _deployed_m1()
        cabins = load_cabins_config()
        active = load_active_cycle_profile().operating_point()
        assert assert_kts_consistency(m1, cabins, active) == []

    def test_kts_skips_uncalibrated_cabins(self):
        """Model loaded but cabins.yaml absent (dev/first-boot) must NOT hard-exit:
        uncalibrated cabins (fallback V_cabin) are skipped, not flagged."""
        m1 = _deployed_m1()
        active = load_active_cycle_profile().operating_point()
        assert assert_kts_consistency(m1, {}, active) == []


class TestM2Disable:
    """M2.disable() makes loaded False so the cross-check (and F010) is skipped."""

    def test_disable_makes_unloaded(self):
        pytest.importorskip("xgboost")
        pytest.importorskip("joblib")
        from models.xgb_regressor_m2 import XGBRegressorM2
        m2 = XGBRegressorM2(load_models_config(), base_dir=_PROJECT_ROOT)
        m2.load()
        assert m2.loaded is True
        m2.disable("operating-point interval mismatch")
        assert m2.loaded is False  # processing_loop's `if m2.loaded` now skips it
