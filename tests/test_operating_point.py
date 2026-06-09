"""Unit tests for core.operating_point (v2.6.3 operating-point contract)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.operating_point import OperatingPoint


def _deployed_op(**overrides):
    """The deployed operating point (v2.6.2-cal20260605), with optional overrides."""
    base = dict(
        profile_id="bph_13000",
        bph=13000,
        cycle_total_ms=6900,
        interval_s=0.1,
        points=70,
        trigger_angle=0.0,
        hold_window_deg=(93.0, 283.0),
        sections={
            "baseline_pre": (0.0, 73.0),
            "evac": (73.0, 93.0),
            "hold": (93.0, 283.0),
            "release": (283.0, 300.0),
            "baseline_post": (300.0, 360.0),
        },
        primary_section="hold",
        p_chamber_pa=35000.0,
        p_atm_pa=101325.0,
    )
    base.update(overrides)
    return OperatingPoint(**base)


class TestDerived:
    def test_k_ts_per_sample(self):
        assert _deployed_op().k_ts_per_sample == pytest.approx(1000.0)
        # interval halved → resolution doubles → k_ts doubles
        assert _deployed_op(interval_s=0.05).k_ts_per_sample == pytest.approx(2000.0)

    def test_in_hold_count_deployed_is_37(self):
        """Lock the ripple number: 37 samples land in hold [93,283) at interval 0.1."""
        assert _deployed_op().in_hold_sample_count == 37

    def test_in_hold_count_drops_at_finer_interval(self):
        """At interval 0.05 (fixed cycle/points) only 34 samples reach hold —
        the 70 samples span ~182° and don't even cover the full window."""
        assert _deployed_op(interval_s=0.05).in_hold_sample_count == 34

    def test_in_hold_count_preserved_when_density_constant(self):
        """Halving interval AND cycle_total_ms keeps deg_per_sample → count fixed."""
        op = _deployed_op(interval_s=0.05, cycle_total_ms=3450)
        assert op.in_hold_sample_count == 37
        assert op.deg_per_sample == pytest.approx(_deployed_op().deg_per_sample)


class TestFingerprint:
    def test_round_trip(self):
        op = _deployed_op()
        fp = op.fingerprint()
        back = OperatingPoint.from_fingerprint(fp)
        assert back.fingerprint() == fp
        assert back.profile_id == "bph_13000"
        assert back.hold_window_deg == (93.0, 283.0)
        assert back.sections["hold"] == (93.0, 283.0)

    def test_fingerprint_is_json_serializable(self):
        import json
        fp = _deployed_op().fingerprint()
        s = json.dumps(fp)  # must not raise
        assert "bph_13000" in s
        # tuples became lists
        assert fp["hold_window_deg"] == [93.0, 283.0]


class TestCompare:
    def test_identical(self):
        c = _deployed_op().compare(_deployed_op())
        assert c["profile_id_match"] and c["sections_match"] and c["interval_match"]
        assert c["in_hold_count_preserved"] and c["vacuum_match"]
        assert c["interval_ratio"] == pytest.approx(1.0)

    def test_interval_change_breaks_count(self):
        """Pure interval change at fixed cycle/points: ratio set, count NOT preserved."""
        cal = _deployed_op()
        active = _deployed_op(interval_s=0.05)
        c = cal.compare(active)
        assert c["interval_match"] is False
        assert c["interval_ratio"] == pytest.approx(2.0)   # β scale = interval_cal/interval_active
        assert c["in_hold_count_preserved"] is False        # 37 vs 34 → must refuse

    def test_interval_change_count_preserved(self):
        """Density-preserving change → rescale is exact (count preserved)."""
        cal = _deployed_op()
        active = _deployed_op(interval_s=0.05, cycle_total_ms=3450)
        c = cal.compare(active)
        assert c["interval_match"] is False
        assert c["interval_ratio"] == pytest.approx(2.0)
        assert c["in_hold_count_preserved"] is True

    def test_vacuum_mismatch(self):
        c = _deployed_op().compare(_deployed_op(p_chamber_pa=45000.0))
        assert c["vacuum_match"] is False
        assert c["interval_match"] is True  # vacuum is orthogonal to rate

    def test_section_mismatch(self):
        active = _deployed_op(
            hold_window_deg=(90.0, 290.0),
            sections={
                "baseline_pre": (0.0, 75.0), "evac": (75.0, 90.0),
                "hold": (90.0, 290.0), "release": (290.0, 304.0),
                "baseline_post": (304.0, 360.0),
            },
        )
        c = _deployed_op().compare(active)
        assert c["sections_match"] is False
        assert c["hold_window_match"] is False
