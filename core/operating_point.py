"""Operating-point contract (v2.6.3).

An *operating point* is the set of process setpoints the deployed models were
calibrated at — split into two orthogonal axes:

  1. THROUGHPUT / rotation speed (the cycle time-base): bph, cycle_total_ms,
     interval_s, points, trigger_angle, and the angle section boundaries.
  2. VACUUM / evacuation level: the chamber plateau pressure (p_chamber_pa)
     and atmosphere (p_atm_pa).

Why this module exists
----------------------
``hold_trend_slope`` (mbar/sample) feeds BOTH M1 (Q = β·slope + α) AND M2's
top-2 features (hold_variance, hold_trend_slope). If the operating point drifts,
M1 and M2 shift *together*, so the F010 M1/M2-disagreement check stays silent —
a silent failure mode. This module turns the operating point into a machine-
checkable fingerprint so a mismatch is REFUSED or FLAGGED at startup, never
silent. See ``configs.loaders.validate_operating_point`` for the gate.

Key physics (the rate coupling)
-------------------------------
M1 is exactly linear in the per-sample slope, and the sampling interval enters
β as a pure scale:

    |β| = V_cabin · 100 / interval_s        (Pa/mbar = 100; per-sample → per-second)
    k_ts = |β| / V_cabin = 100 / interval_s ≈ 1000  at interval_s = 0.1

So when only ``interval_s`` changes, β rescales deterministically (no retrain):

    β_active = β_cal · (interval_cal / interval_active)

M2 (XGBoost) splits on the *raw* slope/variance values, so it has NO closed-form
rescale — it must be retrained or disabled on a mismatch.

The in-hold-count caveat
------------------------
The β rescale is exact ONLY when the number of samples that fall inside the hold
window is preserved. Because angle boundaries are geometric but sampling is in
time, changing ``interval_s`` at fixed ``points``/``cycle_total_ms`` changes how
many of the ``points`` samples land in [hold_start, hold_end) — which changes the
slope-fit distribution itself. ``in_hold_sample_count`` makes this checkable; the
gate refuses an "interval-only" change when the count moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

# Full 5-section order (kept local to avoid importing cycle_profile and creating
# a cycle; cycle_profile.SECTION_NAMES is the same list).
_SECTION_ORDER = ["baseline_pre", "evac", "hold", "release", "baseline_post"]


@dataclass(frozen=True)
class OperatingPoint:
    """Immutable description of the calibration/active operating point.

    Throughput axis: ``bph``, ``cycle_total_ms``, ``interval_s``, ``points``,
    ``trigger_angle``, ``hold_window_deg``, ``sections``, ``primary_section``.
    Vacuum axis (secondary, recorded-but-inert today): ``p_chamber_pa``,
    ``p_atm_pa``. ``slope_unit`` pins the β math contract.
    """

    profile_id: str
    bph: int
    cycle_total_ms: int
    interval_s: float
    points: int
    trigger_angle: float
    hold_window_deg: Tuple[float, float]
    sections: Dict[str, Tuple[float, float]]
    primary_section: str = "hold"
    p_chamber_pa: float = 35000.0
    p_atm_pa: float = 101325.0
    slope_unit: str = "mbar_per_sample"

    # ── derived (not stored) ────────────────────────────────────────────

    @property
    def k_ts_per_sample(self) -> float:
        """Pa/s per (mbar/sample) implied by the time-base = 100 / interval_s.

        At interval_s=0.1 this is 1000 (deployed median |β|/V_cabin ≈ 1014).
        """
        return 100.0 / self.interval_s if self.interval_s > 0 else float("inf")

    @property
    def deg_per_sample(self) -> float:
        """Angular advance between consecutive samples (degrees).

        ω = 360 / (cycle_total_ms/1000) deg/s; deg_per_sample = ω · interval_s.
        """
        if self.cycle_total_ms <= 0:
            return 0.0
        omega_deg_per_s = 360.0 / (self.cycle_total_ms / 1000.0)
        return omega_deg_per_s * self.interval_s

    @property
    def in_hold_sample_count(self) -> int:
        """How many of the ``points`` samples fall inside the hold window.

        Sample k (k=0..points-1) lands at angle (trigger + k·deg_per_sample) mod
        360. Counts those whose angle ∈ [hold_start, hold_end). This is the
        quantity the β-rescale exactness depends on (37 at interval 0.1 for the
        deployed point; 34 at 0.05).
        """
        dps = self.deg_per_sample
        if dps <= 0:
            return 0
        start, end = self.hold_window_deg
        n = 0
        for k in range(self.points):
            ang = (self.trigger_angle + k * dps) % 360.0
            if start <= ang < end:
                n += 1
        return n

    # ── serialization ───────────────────────────────────────────────────

    def fingerprint(self) -> Dict[str, Any]:
        """Stable, JSON-serializable dict for artifacts and comparison.

        Tuples become lists; section order is canonical. Two operating points
        with equal fingerprints are interchangeable for inference.
        """
        return {
            "profile_id": self.profile_id,
            "bph": int(self.bph),
            "cycle_total_ms": int(self.cycle_total_ms),
            "interval_s": float(self.interval_s),
            "points": int(self.points),
            "trigger_angle": float(self.trigger_angle),
            "hold_window_deg": [float(self.hold_window_deg[0]), float(self.hold_window_deg[1])],
            "sections": {
                name: [float(self.sections[name][0]), float(self.sections[name][1])]
                for name in _SECTION_ORDER if name in self.sections
            },
            "primary_section": self.primary_section,
            "p_chamber_pa": float(self.p_chamber_pa),
            "p_atm_pa": float(self.p_atm_pa),
            "slope_unit": self.slope_unit,
        }

    @classmethod
    def from_fingerprint(cls, d: Dict[str, Any]) -> "OperatingPoint":
        """Rebuild from a fingerprint dict (artifact ``operating_point`` block)."""
        sections = {
            name: (float(bounds[0]), float(bounds[1]))
            for name, bounds in (d.get("sections", {}) or {}).items()
        }
        hw = d.get("hold_window_deg")
        if hw is None and "hold" in sections:
            hw = sections["hold"]
        return cls(
            profile_id=str(d.get("profile_id", "")),
            bph=int(d.get("bph", 0)),
            cycle_total_ms=int(d.get("cycle_total_ms", 0)),
            interval_s=float(d.get("interval_s", 0.1)),
            points=int(d.get("points", 70)),
            trigger_angle=float(d.get("trigger_angle", 0.0)),
            hold_window_deg=(float(hw[0]), float(hw[1])) if hw else (0.0, 0.0),
            sections=sections,
            primary_section=str(d.get("primary_section", "hold")),
            p_chamber_pa=float(d.get("p_chamber_pa", 35000.0)),
            p_atm_pa=float(d.get("p_atm_pa", 101325.0)),
            slope_unit=str(d.get("slope_unit", "mbar_per_sample")),
        )

    # ── comparison ──────────────────────────────────────────────────────

    def _sections_equal(self, other: "OperatingPoint", tol: float = 0.5) -> bool:
        if set(self.sections) != set(other.sections):
            return False
        for name, (s, e) in self.sections.items():
            os_, oe = other.sections[name]
            if abs(s - os_) > tol or abs(e - oe) > tol:
                return False
        return True

    def compare(self, other: "OperatingPoint") -> Dict[str, Any]:
        """Per-field comparison of two operating points (self = calibration,
        other = active). Returns a dict the gate consumes.

        ``interval_ratio`` is interval_cal / interval_active — the exact M1
        β-rescale factor. ``in_hold_count_preserved`` decides whether the
        rescale is valid or the change must be refused.
        """
        return {
            "profile_id_match": self.profile_id == other.profile_id,
            "sections_match": self._sections_equal(other),
            "hold_window_match": (
                abs(self.hold_window_deg[0] - other.hold_window_deg[0]) <= 0.5
                and abs(self.hold_window_deg[1] - other.hold_window_deg[1]) <= 0.5
            ),
            "primary_section_match": self.primary_section == other.primary_section,
            "interval_match": abs(self.interval_s - other.interval_s) <= 1e-9,
            "interval_ratio": (
                self.interval_s / other.interval_s if other.interval_s > 0 else float("inf")
            ),
            "in_hold_count_self": self.in_hold_sample_count,
            "in_hold_count_other": other.in_hold_sample_count,
            "in_hold_count_preserved": self.in_hold_sample_count == other.in_hold_sample_count,
            "vacuum_match": abs(self.p_chamber_pa - other.p_chamber_pa) <= 1.0,
        }


def fingerprint_from_profile(profile: Any) -> Dict[str, Any]:
    """Single shared serializer: project a CycleProfile into a fingerprint dict.

    Used by both train_m1 and train_m2 so they emit byte-identical
    ``operating_point`` blocks, and by the runtime to build the active point.
    Reads the optional vacuum attributes if the profile exposes them
    (``p_chamber_pa`` / ``p_atm_pa``), else uses the standard defaults.
    """
    return profile.operating_point().fingerprint()
