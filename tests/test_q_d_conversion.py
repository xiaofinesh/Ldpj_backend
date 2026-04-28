"""Tests for Q ↔ d conversion (v2.6)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.q_d_conversion import (
    d_to_q,
    d_to_q_choked,
    d_to_q_laminar,
    q_to_d,
    q_to_d_choked,
    q_to_d_laminar,
)


class TestLaminar:
    @pytest.mark.parametrize("d_in", [10, 50, 100, 150, 200, 300])
    def test_round_trip(self, d_in):
        q = d_to_q_laminar(d_in)
        d_out = q_to_d_laminar(q)
        assert abs(d_out - d_in) / d_in < 1e-3  # < 0.1%

    def test_q_grows_roughly_as_d4(self):
        """Hagen-Poiseuille: Q ∝ d⁴ (with mild correction from Sampson term)."""
        q1 = d_to_q_laminar(100)
        q2 = d_to_q_laminar(200)
        ratio = q2 / q1
        # Pure d⁴ would give 16; Sampson term softens it slightly
        assert 13.0 < ratio < 17.0

    def test_zero_input(self):
        assert q_to_d_laminar(0.0) == 0.0
        assert d_to_q_laminar(0.0) == 0.0

    def test_negative_input(self):
        assert q_to_d_laminar(-1.0) == 0.0
        assert d_to_q_laminar(-1.0) == 0.0

    def test_l_ref_affects_q(self):
        """Larger reference channel length reduces Q for same diameter."""
        q_thin = d_to_q_laminar(100, l_ref_mm=0.3)
        q_thick = d_to_q_laminar(100, l_ref_mm=1.0)
        assert q_thin > q_thick

    def test_typical_microdefect_in_expected_range(self):
        """30 μm pin-hole through 0.5 mm wall at standard pressures should be
        in the range 1e-3 ~ 1e-1 Pa·m³/s (sanity check the numerics).

        Note: 100 μm+ holes already exceed laminar's l/d>100 validity domain
        (l/d = 5 here), so we use a smaller hole that the formula handles well.
        """
        q = d_to_q_laminar(30)
        assert 1e-3 < q < 1e-1


class TestChoked:
    @pytest.mark.parametrize("d_in", [10, 50, 100, 150, 200])
    def test_round_trip(self, d_in):
        q = d_to_q_choked(d_in)
        d_out = q_to_d_choked(q)
        assert abs(d_out - d_in) / d_in < 1e-3

    def test_q_grows_as_d2(self):
        """Choked flow: Q ∝ area = π·d²/4, so doubling d gives 4× Q."""
        q1 = d_to_q_choked(100)
        q2 = d_to_q_choked(200)
        assert 3.5 < q2 / q1 < 4.5

    def test_zero_input(self):
        assert q_to_d_choked(0.0) == 0.0
        assert d_to_q_choked(0.0) == 0.0

    def test_pressure_ratio_effects_cd(self):
        """Higher p_d/p_u ratio → lower discharge coefficient."""
        q_low_ratio = d_to_q_choked(100, p_u=101325, p_d=10000)
        q_high_ratio = d_to_q_choked(100, p_u=101325, p_d=80000)
        assert q_low_ratio > q_high_ratio


class TestDispatch:
    def test_dispatch_laminar(self):
        assert q_to_d(1e-3, regime="laminar") == q_to_d_laminar(1e-3)
        assert d_to_q(100, regime="laminar") == d_to_q_laminar(100)

    def test_dispatch_choked(self):
        assert q_to_d(1e-3, regime="choked") == q_to_d_choked(1e-3)
        assert d_to_q(100, regime="choked") == d_to_q_choked(100)

    def test_unknown_regime_raises(self):
        with pytest.raises(ValueError, match="Unknown flow regime"):
            q_to_d(1e-3, regime="turbulent")
        with pytest.raises(ValueError, match="Unknown flow regime"):
            d_to_q(100, regime="transition")

    def test_kwargs_pass_through(self):
        """Dispatch must forward regime-specific kwargs to the underlying impl."""
        # l_ref_mm is laminar-specific
        a = d_to_q(100, regime="laminar", l_ref_mm=0.3)
        b = d_to_q(100, regime="laminar", l_ref_mm=1.0)
        assert a > b


class TestExtremePressureClampWarning:
    """The C_d formula clamps to [0.5, 0.95] when the upstream/downstream
    pressure ratio is unrealistic. The clamp now emits a rate-limited
    warning so the conversion's degraded state is visible in the log."""

    def test_high_pressure_ratio_logs_clamp_warning(self, caplog):
        """p_d ≫ p_u → C_d = 0.8623 - 0.2541·(very large) → strongly negative,
        clamped UP to 0.5. The first occurrence always logs."""
        from core import q_d_conversion
        from core.rate_limit import reset_state
        reset_state()  # clear inter-test rate-limit state
        with caplog.at_level("WARNING"):
            cd = q_d_conversion._discharge_coefficient(p_u=1, p_d=10000)
        assert cd == pytest.approx(0.5)
        assert any("C_d clamped" in rec.message or "q_d_choked_cd_clamped" in rec.message
                   for rec in caplog.records)

    def test_normal_pressure_ratio_no_warning(self, caplog):
        """Within the validity domain → no clamp → no warning."""
        from core import q_d_conversion
        from core.rate_limit import reset_state
        reset_state()
        with caplog.at_level("WARNING"):
            cd = q_d_conversion._discharge_coefficient(p_u=101325, p_d=35000)
        # Raw formula yields ~0.78 — well inside [0.5, 0.95]
        assert 0.5 < cd < 0.95
        assert not any("clamped" in rec.message for rec in caplog.records)


class TestRelativeMagnitudes:
    """Cross-regime sanity checks. The two formulas live in different regimes
    (l/d > 100 vs l/d < 3) so we don't assert one bounds the other for all d.
    What we DO assert: both give finite positive values, and the d-dependence
    is laminar∝d⁴ vs choked∝d² (so they cross over at some d)."""

    def test_both_regimes_yield_finite_positive(self):
        for d_um in [10, 30, 50, 100]:
            q_lam = d_to_q_laminar(d_um)
            q_chk = d_to_q_choked(d_um)
            assert q_lam > 0 and q_lam < 1e3
            assert q_chk > 0 and q_chk < 1e3

    def test_d4_vs_d2_growth_creates_crossover(self):
        """Laminar grows faster (d⁴), so for large enough d laminar > choked."""
        # At 10 μm choked dominates; at 100 μm laminar dominates.
        assert d_to_q_choked(10) > d_to_q_laminar(10)
        assert d_to_q_laminar(100) > d_to_q_choked(100)
