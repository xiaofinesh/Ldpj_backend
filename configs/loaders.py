"""YAML configuration loaders for Ldpj_backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from core.cycle_profile import (
    CycleProfile,
    load_active_cycle_profile as _load_active_cycle_profile_from_dict,
)
from core.operating_point import OperatingPoint

_BASE_DIR = Path(__file__).resolve().parent


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a yaml config file as a dict.

    Returns ``{}`` only when the file is missing — that's a graceful
    degradation case (system can boot with defaults). YAML syntax errors
    or other exceptions ARE re-raised so a typo doesn't cause the system
    to silently boot with completely empty config (which used to happen
    in v2.6.1: a single misplaced colon → empty dict → all defaults →
    confusing downstream behavior). Callers should treat raised
    exceptions as a startup blocker.
    """
    p = Path(path)
    if not p.is_absolute():
        p = _BASE_DIR / p
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_plc_config() -> Dict[str, Any]:
    return load_yaml("plc.yaml")

def load_runtime_config() -> Dict[str, Any]:
    return load_yaml("runtime.yaml")

def load_models_config() -> Dict[str, Any]:
    return load_yaml("models.yaml")

def load_health_config() -> Dict[str, Any]:
    return load_yaml("health.yaml")

def load_ipc_config() -> Dict[str, Any]:
    return load_yaml("ipc.yaml")


def load_active_cycle_profile() -> CycleProfile:
    """Load runtime.yaml and extract the active CycleProfile (v2.6).

    Convenience wrapper that reads runtime.yaml then dispatches to
    core.cycle_profile.load_active_cycle_profile().
    """
    return _load_active_cycle_profile_from_dict(load_runtime_config())


# ── Cabin V_cabin calibration (v2.6) ──────────────────────────────────

def load_cabins_config(path: str | Path = "cabins.yaml") -> Dict[str, Any]:
    """Load V_cabin calibration values for all cabins.

    Path is resolved relative to the configs/ directory if not absolute.

    Returns
    -------
    dict with keys: calibration_date, calibrator, cabins, default

    Raises
    ------
    FileNotFoundError if the file is missing.
    """
    p = Path(path)
    if not p.is_absolute():
        p = _BASE_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"cabins config not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_v_cabin(cabins_cfg: Dict[str, Any], cabin_id: int) -> Tuple[float, float]:
    """Look up (v_cabin, u_v_cabin) in m³ for a specific cabin.

    Falls back to the ``default`` block when the cabin has no calibrated
    entry. Returns numeric defaults (3.5e-4, 1e-5) if even ``default`` is
    missing, so that downstream physics never sees None.
    """
    entry = cabins_cfg.get("cabins", {}).get(cabin_id)
    if entry and "v_cabin" in entry:
        return float(entry["v_cabin"]), float(entry.get("u_v_cabin", 0.0))
    default = cabins_cfg.get("default", {}) or {}
    return float(default.get("v_cabin", 3.5e-4)), float(default.get("u_v_cabin", 1.0e-5))


def is_cabin_calibrated(cabins_cfg: Dict[str, Any], cabin_id: int) -> bool:
    """Check whether ``cabin_id`` has been measured (vs using a placeholder).

    Heuristic: an entry whose ``notes`` contains '占位' is treated as
    uncalibrated, even if a numeric ``v_cabin`` is present.
    """
    entry = cabins_cfg.get("cabins", {}).get(cabin_id, {})
    if not entry or entry.get("v_cabin") is None:
        return False
    notes = entry.get("notes", "") or ""
    return "占位" not in notes


# ── Product configuration (v2.6) ──────────────────────────────────────

def load_products_config(path: str | Path = "products.yaml") -> Dict[str, Any]:
    """Load product configuration (Q_threshold per product).

    Path is resolved relative to the configs/ directory if not absolute.

    Returns
    -------
    dict with keys: default_product_id, products

    Raises
    ------
    FileNotFoundError if the file is missing.
    """
    p = Path(path)
    if not p.is_absolute():
        p = _BASE_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"products config not found: {p}")
    with open(p, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def get_product(products_cfg: Dict[str, Any], product_id: str) -> Dict[str, Any]:
    """Look up a product by id, falling back to the configured default.

    Returns an empty dict if neither the requested product nor the default
    can be resolved (so callers always get a Mapping back).
    """
    products = products_cfg.get("products", {}) or {}
    if product_id in products:
        return products[product_id]
    default_id = products_cfg.get("default_product_id", "")
    if default_id and default_id in products:
        return products[default_id]
    return {}


def validate_product_resolution(
    products_cfg: Dict[str, Any],
    a_estimate: float,
    a_det: float,
) -> Tuple[list, list]:
    """Check every product's ``q_threshold`` against the system resolution.

    Implements the README §5 / 部署说明 §3 rule
    "强制校验 Q_threshold > A, 建议 ≥ A_det":

    - **强制 (error)**: ``q_threshold`` MUST be > ``a_estimate`` (A, the
      estimate resolution). Below A the Q estimate is noise-dominated, so a
      threshold placed there cannot reliably separate LEAK from OK.
    - **建议 (warning)**: ``q_threshold`` SHOULD be ≥ ``a_det`` (the
      detection resolution). Between A and A_det a true leak may fall below
      POD ≥ 99.87%.

    Products without a ``q_threshold`` are skipped (they are judged N/A at
    runtime). Returns ``(errors, warnings)`` as lists of human-readable
    strings; callers decide whether errors are fatal (main.py treats them
    as a startup blocker, mirroring the "强制" requirement).
    """
    errors: list = []
    warnings: list = []
    products = products_cfg.get("products", {}) or {}
    for pid, prod in products.items():
        thr = (prod or {}).get("q_threshold")
        if thr is None:
            continue
        try:
            thr = float(thr)
        except (TypeError, ValueError):
            errors.append(f"产品 '{pid}': q_threshold 不是数值 ({thr!r})")
            continue
        if thr <= a_estimate:
            errors.append(
                f"产品 '{pid}': q_threshold={thr:.2e} ≤ A={a_estimate:.2e} "
                f"(估计分辨率); 阈值必须高于 A 才能可靠判废"
            )
        elif thr < a_det:
            warnings.append(
                f"产品 '{pid}': q_threshold={thr:.2e} < A_det={a_det:.2e} "
                f"(检出分辨率); 建议 ≥ A_det, 否则漏率可能低于 POD≥99.87%"
            )
    return errors, warnings


# ── Operating-point gate (v2.6.3) ─────────────────────────────────────

def validate_operating_point(
    active_op: OperatingPoint,
    m1_op: Optional[OperatingPoint],
    m2_op: Optional[OperatingPoint],
) -> Dict[str, Any]:
    """Gate the active operating point against the deployed models' calibration.

    Closes the silent-drift hole: ``hold_trend_slope`` feeds BOTH M1 and M2, so
    on operating-point drift they shift together and the F010 disagreement check
    is blind. This compares the active CycleProfile's fingerprint to each model's
    stored ``operating_point`` and returns a decision the caller (main.py)
    executes.

    Decision rules (active vs M1 calibration):
      - M1 artifact has NO operating_point   → ERROR (legacy bundle; re-deploy).
      - profile_id / hold-window / sections / primary_section mismatch
                                              → ERROR (segmentation drift corrupts
                                                both models; refuse, never rescale).
      - interval mismatch, in-hold count CHANGED → ERROR (β linear rescale no
                                                longer exact; the ripple — refuse).
      - interval mismatch, in-hold count PRESERVED → RESCALE M1 (β·interval_cal/
                                                interval_active) + DISABLE M2 + F013.
      - vacuum (p_chamber) mismatch          → WARNING + F014 (M1 slope-invariant;
                                                M2 absolute-pressure features suspect).
      - M2 artifact has NO operating_point   → WARNING (cross-check stays on).

    Returns
    -------
    dict with keys: ``errors`` (list[str], fatal → sys.exit), ``warnings``
    (list[str]), ``faults`` (list[(code, msg)]), ``m1_rescale_to`` (float|None,
    the active interval_s to rescale M1 to), ``m2_disable_reason`` (str|None).
    """
    errors: list = []
    warnings: list = []
    faults: list = []
    m1_rescale_to: Optional[float] = None
    m2_disable_reason: Optional[str] = None

    if m1_op is None:
        errors.append(
            "M1 工件缺少 operating_point 工况指纹 (旧版 bundle); "
            "请用 v2.6.3+ 重新训练/部署 (scripts/deploy_model.sh)"
        )
        return {
            "errors": errors, "warnings": warnings, "faults": faults,
            "m1_rescale_to": None, "m2_disable_reason": None,
        }

    c = m1_op.compare(active_op)  # self = calibration, other = active

    # 1) Identity / geometry — fatal (corrupts M1 AND M2 together)
    if not c["profile_id_match"]:
        errors.append(
            f"运行 profile_id='{active_op.profile_id}' 与 M1 标定 "
            f"'{m1_op.profile_id}' 不一致 (工况身份变更); 若为有意调整产量/真空, "
            f"请按新工况重标定并经 deploy_model.sh 部署对应模型"
        )
    geom_mismatch = (
        not c["sections_match"] or not c["hold_window_match"]
        or not c["primary_section_match"]
    )
    if geom_mismatch:
        errors.append(
            f"分段/保压窗与 M1 标定不一致 (标定 hold={m1_op.hold_window_deg} → "
            f"运行 {active_op.hold_window_deg}); 会同时污染 M1 和 M2 "
            f"(F010 一致性检查无法察觉), 必须按新工况重标定"
        )
    if errors:
        # Identity/geometry is fatal; do not offer interval/vacuum actions.
        return {
            "errors": errors, "warnings": warnings, "faults": faults,
            "m1_rescale_to": None, "m2_disable_reason": None,
        }

    # 2) In-hold sample count — INDEPENDENT hard gate. ANY time-base change
    # (rotation speed / cycle_total_ms, points, trigger_angle, OR interval) that
    # moves how many samples land in the hold window changes the M1 slope-fit
    # distribution AND M2's hold features together — and F010 is blind to it.
    # The β linear rescale is exact only when this count is preserved, so a
    # change here is fatal regardless of whether interval_s also changed. This
    # is the "调整产量(转速)" case: cycle_total_ms changes while interval stays.
    if not c["in_hold_count_preserved"]:
        errors.append(
            f"保压窗内采样数与标定不一致 ({c['in_hold_count_self']} → {c['in_hold_count_other']}); "
            f"采样分布改变 (转速/cycle_total_ms、采样点数或间隔变化所致), "
            f"M1 斜率拟合分布与标定不符, 线性重缩放不再精确 — 拒绝启动, 需按新工况重标定 M1/M2"
        )
        return {
            "errors": errors, "warnings": warnings, "faults": faults,
            "m1_rescale_to": None, "m2_disable_reason": None,
        }

    # 3) Time-base: interval mismatch with in-hold count PRESERVED (e.g. rate +
    # interval scaled together so angular density is unchanged) → exact M1
    # β rescale; M2 (XGBoost) cannot follow → disable + F013.
    if not c["interval_match"]:
        m1_rescale_to = active_op.interval_s
        m2_disable_reason = (
            f"运行间隔 {active_op.interval_s}s ≠ 标定 {m1_op.interval_s}s; "
            f"M2(XGBoost) 不可线性重缩放, 交叉校验已禁用 (非故障)"
        )
        faults.append((
            "F013",
            f"采样间隔 {active_op.interval_s}s≠标定 {m1_op.interval_s}s; "
            f"M1 β 已×{c['interval_ratio']:.4f} 重缩放, M2 交叉校验关闭 (非故障)",
        ))

    # 4) Vacuum: warning only (M1 slope-invariant; M2 absolute features suspect)
    if not c["vacuum_match"]:
        warnings.append(
            f"运行真空 {active_op.p_chamber_pa:.0f}Pa ≠ 标定 {m1_op.p_chamber_pa:.0f}Pa; "
            f"M1 不受影响 (仅用斜率), M2 绝压特征可能失准"
        )
        faults.append((
            "F014",
            f"运行点真空 {active_op.p_chamber_pa:.0f}Pa≠标定 "
            f"{m1_op.p_chamber_pa:.0f}Pa, M2 绝压特征失准",
        ))

    # 5) M2 provenance (non-fatal)
    if m2_op is None:
        warnings.append(
            "M2 工件缺少 operating_point 工况指纹 (交叉校验仍启用, 但无法校验工况一致性)"
        )

    return {
        "errors": errors, "warnings": warnings, "faults": faults,
        "m1_rescale_to": m1_rescale_to, "m2_disable_reason": m2_disable_reason,
    }


def assert_kts_consistency(
    m1_model: Any,
    cabins_cfg: Dict[str, Any],
    active_op: OperatingPoint,
    tol: float = 0.04,
) -> list:
    """Runtime promotion of the k_ts≈1014 lock-in test.

    After any M1 β rescale (and at the matched deployed point), re-derive
    ``k_ts = |β| / V_cabin`` per calibrated cabin and check it is within ``tol``
    of the theoretical ``active_op.k_ts_per_sample`` (= 100/interval_active).
    Catches a β that is not purely the time-base coupling (e.g. a corrupted
    rescale or a V_cabin/β table that drifted out of physical self-consistency).

    Returns a list of human-readable error strings (empty = OK).
    """
    errs: list = []
    if not getattr(m1_model, "loaded", False):
        return errs
    theory = active_op.k_ts_per_sample
    if theory <= 0:
        return errs
    for cid in m1_model.calibrated_cabins:
        # Skip cabins without a real measured V_cabin (fallback value would give
        # a meaningless k_ts and spuriously hard-exit a dev/first-boot box that
        # has models but no cabins.yaml). k_ts self-consistency only applies to
        # genuinely calibrated cabins.
        if not is_cabin_calibrated(cabins_cfg, cid):
            continue
        # predict(-1.0) = β·(-1)+α ≈ |β| since |α| ≪ |β|
        beta_abs = abs(float(m1_model.predict(-1.0, cid)["q_est"]))
        v_cabin, _ = get_v_cabin(cabins_cfg, cid)
        if v_cabin <= 0:
            continue
        kts = beta_abs / v_cabin
        if abs(kts / theory - 1.0) > tol:
            errs.append(
                f"舱 {cid}: k_ts={kts:.0f} 偏离理论 {theory:.0f} 超过 {tol * 100:.0f}% "
                f"(|β|={beta_abs:.4f}, V_cabin={v_cabin:.2e})"
            )
    return errs
