"""Q ↔ d (equivalent defect diameter) conversion (v2.6).

Two flow regimes are supported:

1. Laminar flow (l/d > 100): Hagen-Poiseuille with Sampson correction.
       Q = π · d⁴ / [128 · η · (l + 0.41 · d)] · p̄ · Δp
   where Δp = p_atm − p_chamber and p̄ = (p_atm + p_chamber) / 2.

2. Choked flow (l/d < 3): Yoshida critical-flow equation.
       Q = C_d · p_u · (π · d² / 4) · √[γRT/M · (2/(γ+1))^((γ+1)/(γ−1))]
   where C_d = 0.8623 − 0.2541 · (p_d / p_u).

References:
    * GB/T 40336-2021 (China national standard for leak detection)
    * Yoshida (2021) Packag. Technol. Sci.
"""

from __future__ import annotations

import math


# Physical constants (air at 20 °C unless overridden)
ETA_AIR = 1.83e-5      # dynamic viscosity, Pa·s
GAMMA = 1.4            # heat capacity ratio (cp/cv) for diatomic air
M_AIR = 0.029          # molar mass, kg/mol
R = 8.314              # universal gas constant, J/(mol·K)
T_REF = 293.15         # reference temperature, K (= 20 °C)
P_ATM = 101325.0       # atmospheric pressure, Pa
P_VACUUM = 35000.0     # default chamber absolute pressure (≈ 350 mbar abs), Pa


# ── Laminar (Hagen-Poiseuille with Sampson correction) ──────────────

def d_to_q_laminar(
    d_um: float,
    l_ref_mm: float = 0.5,
    p_atm: float = P_ATM,
    p_chamber: float = P_VACUUM,
    eta: float = ETA_AIR,
) -> float:
    """Forward Hagen-Poiseuille: equivalent diameter (μm) → leak rate Q (Pa·m³/s)."""
    if d_um <= 0:
        return 0.0
    d = d_um * 1e-6        # μm → m
    l = l_ref_mm * 1e-3    # mm → m
    p_bar = (p_atm + p_chamber) / 2.0
    delta_p = p_atm - p_chamber
    return math.pi * d ** 4 / (128.0 * eta * (l + 0.41 * d)) * p_bar * delta_p


def q_to_d_laminar(
    q: float,
    l_ref_mm: float = 0.5,
    p_atm: float = P_ATM,
    p_chamber: float = P_VACUUM,
    eta: float = ETA_AIR,
) -> float:
    """Reverse Hagen-Poiseuille: Q (Pa·m³/s) → equivalent diameter (μm).

    The Sampson term ``0.41·d`` in the denominator depends on the answer,
    so we iterate: start by ignoring it, refine until the diameter
    converges to within 1e-7 (relative).
    """
    if q <= 0:
        return 0.0

    l = l_ref_mm * 1e-3
    p_bar = (p_atm + p_chamber) / 2.0
    delta_p = p_atm - p_chamber

    # Initial guess ignoring Sampson: Q ≈ π·d⁴ / (128·η·l) · p̄·Δp
    d4 = q * 128.0 * eta * l / (math.pi * p_bar * delta_p)
    d = d4 ** 0.25

    # Fixed-point iteration with the Sampson correction
    for _ in range(20):
        eff_l = l + 0.41 * d
        d4 = q * 128.0 * eta * eff_l / (math.pi * p_bar * delta_p)
        d_new = d4 ** 0.25
        if d > 0 and abs(d_new - d) / d < 1e-7:
            d = d_new
            break
        d = d_new

    return d * 1e6  # m → μm


# ── Choked (Yoshida critical flow) ──────────────────────────────────

def _choked_sqrt_factor(temperature: float = T_REF) -> float:
    """√[γRT/M · (2/(γ+1))^((γ+1)/(γ−1))] — depends only on T."""
    return math.sqrt(
        GAMMA * R * temperature / M_AIR
        * (2.0 / (GAMMA + 1.0)) ** ((GAMMA + 1.0) / (GAMMA - 1.0))
    )


def _discharge_coefficient(p_u: float, p_d: float) -> float:
    """C_d = 0.8623 − 0.2541·(p_d/p_u). Clamped to [0.5, 0.95] for safety.

    When the raw value falls outside [0.5, 0.95] (i.e. unusually extreme
    pressure ratio), emits a rate-limited warning so the operator can
    notice that the choked-flow conversion is in degraded territory.
    """
    p_ratio = p_d / p_u if p_u > 0 else 0.0
    raw_c_d = 0.8623 - 0.2541 * p_ratio
    c_d = max(0.5, min(0.95, raw_c_d))
    if c_d != raw_c_d:
        from core.rate_limit import warn_throttled
        warn_throttled(
            "q_d_choked_cd_clamped",
            "Choked-flow C_d clamped from %.3f to %.3f (p_u=%g, p_d=%g)",
            raw_c_d, c_d, p_u, p_d,
        )
    return c_d


def d_to_q_choked(
    d_um: float,
    p_u: float = P_ATM,
    p_d: float = P_VACUUM,
    T: float = T_REF,
) -> float:
    """Forward Yoshida critical flow: d (μm) → Q (Pa·m³/s)."""
    if d_um <= 0:
        return 0.0
    d = d_um * 1e-6
    area = math.pi * d ** 2 / 4.0
    c_d = _discharge_coefficient(p_u, p_d)
    return c_d * p_u * area * _choked_sqrt_factor(T)


def q_to_d_choked(
    q: float,
    p_u: float = P_ATM,
    p_d: float = P_VACUUM,
    T: float = T_REF,
) -> float:
    """Reverse Yoshida critical flow: Q (Pa·m³/s) → d (μm)."""
    if q <= 0:
        return 0.0
    c_d = _discharge_coefficient(p_u, p_d)
    sqrt_factor = _choked_sqrt_factor(T)
    area = q / (c_d * p_u * sqrt_factor)
    if area <= 0:
        return 0.0
    d_squared = area * 4.0 / math.pi
    return d_squared ** 0.5 * 1e6


# ── Dispatch by regime ──────────────────────────────────────────────

def q_to_d(q: float, regime: str = "laminar", **kwargs) -> float:
    """Dispatch ``q_to_d_*`` by ``regime`` ('laminar' or 'choked')."""
    if regime == "laminar":
        return q_to_d_laminar(q, **kwargs)
    if regime == "choked":
        return q_to_d_choked(q, **kwargs)
    raise ValueError(f"Unknown flow regime: {regime!r} (expected 'laminar' or 'choked')")


def d_to_q(d_um: float, regime: str = "laminar", **kwargs) -> float:
    """Dispatch ``d_to_q_*`` by ``regime`` ('laminar' or 'choked')."""
    if regime == "laminar":
        return d_to_q_laminar(d_um, **kwargs)
    if regime == "choked":
        return d_to_q_choked(d_um, **kwargs)
    raise ValueError(f"Unknown flow regime: {regime!r} (expected 'laminar' or 'choked')")
