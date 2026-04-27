"""Unit tests for cycle_profile module (v2.6)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.cycle_profile import CycleProfile, load_active_cycle_profile, SECTION_NAMES


def _valid_profile_dict():
    return {
        "description": "test",
        "bph": 13000,
        "cycle_total_ms": 6900,
        "sections": {
            "baseline_pre":  [0.0,   57.6],
            "evac":          [57.6,  93.0],
            "stable":        [93.0,  115.0],
            "hold":          [115.0, 273.6],
            "release":       [273.6, 302.4],
            "baseline_post": [302.4, 360.0],
        },
        "collection": {
            "trigger_angle": 0.0,
            "points": 70,
            "interval_s": 0.1,
            "timeout_s": 10.0,
        },
        "primary_section": "hold",
    }


class TestCycleProfile:
    def test_from_dict_basic(self):
        p = CycleProfile.from_dict("test", _valid_profile_dict())
        assert p.profile_id == "test"
        assert p.bph == 13000
        assert p.collection_points == 70
        assert p.collection_interval_s == pytest.approx(0.1)
        assert p.primary_section == "hold"
        assert p.sections["hold"] == (115.0, 273.6)

    def test_validate_passes(self):
        p = CycleProfile.from_dict("test", _valid_profile_dict())
        p.validate()  # should not raise

    def test_validate_missing_section(self):
        d = _valid_profile_dict()
        del d["sections"]["evac"]
        p = CycleProfile.from_dict("test", d)
        with pytest.raises(ValueError, match="missing sections"):
            p.validate()

    def test_validate_overlap(self):
        d = _valid_profile_dict()
        # evac now ends at 100, but stable starts at 93 -> stable starts before evac ended
        d["sections"]["evac"] = [57.6, 100.0]
        p = CycleProfile.from_dict("test", d)
        with pytest.raises(ValueError, match="before previous section"):
            p.validate()

    def test_validate_inverted_range(self):
        d = _valid_profile_dict()
        d["sections"]["hold"] = [273.6, 115.0]  # end <= start
        p = CycleProfile.from_dict("test", d)
        with pytest.raises(ValueError, match="invalid range"):
            p.validate()

    def test_validate_exceeds_360(self):
        d = _valid_profile_dict()
        d["sections"]["baseline_post"] = [302.4, 400.0]
        p = CycleProfile.from_dict("test", d)
        with pytest.raises(ValueError, match="exceeds 360"):
            p.validate()

    def test_validate_bad_primary_section(self):
        d = _valid_profile_dict()
        d["primary_section"] = "nonexistent"
        p = CycleProfile.from_dict("test", d)
        with pytest.raises(ValueError, match="primary_section"):
            p.validate()

    def test_validate_bad_sampling(self):
        d = _valid_profile_dict()
        d["collection"]["points"] = 0
        p = CycleProfile.from_dict("test", d)
        with pytest.raises(ValueError, match="invalid sampling"):
            p.validate()

    def test_validate_timeout_too_short(self):
        d = _valid_profile_dict()
        d["collection"]["timeout_s"] = 1.0  # 70 * 0.1 = 7.0 > 1.0
        p = CycleProfile.from_dict("test", d)
        with pytest.raises(ValueError, match="timeout"):
            p.validate()

    def test_section_names_constant(self):
        assert SECTION_NAMES == [
            "baseline_pre", "evac", "stable", "hold", "release", "baseline_post"
        ]


class TestLoadActiveProfile:
    def test_load_valid(self):
        runtime_cfg = {
            "active_profile": "test",
            "cycle_profiles": {"test": _valid_profile_dict()},
        }
        p = load_active_cycle_profile(runtime_cfg)
        assert p.profile_id == "test"
        assert p.collection_points == 70

    def test_load_missing_active(self):
        runtime_cfg = {"cycle_profiles": {"test": _valid_profile_dict()}}
        with pytest.raises(ValueError, match="active_profile"):
            load_active_cycle_profile(runtime_cfg)

    def test_load_unknown_profile(self):
        runtime_cfg = {
            "active_profile": "missing",
            "cycle_profiles": {"test": _valid_profile_dict()},
        }
        with pytest.raises(ValueError, match="not found"):
            load_active_cycle_profile(runtime_cfg)


class TestRealRuntimeYaml:
    """Verify the actual configs/runtime.yaml shipped with the repo loads cleanly."""

    def test_active_profile_loads(self):
        from configs.loaders import load_active_cycle_profile as real_loader
        p = real_loader()
        assert p.profile_id == "bph_13000"
        assert p.bph == 13000
        assert p.cycle_total_ms == 6900
        assert p.collection_points == 70
        assert p.primary_section == "hold"
        # Section ordering sanity
        assert p.sections["baseline_pre"][0] == 0.0
        assert p.sections["baseline_post"][1] == pytest.approx(360.0)
