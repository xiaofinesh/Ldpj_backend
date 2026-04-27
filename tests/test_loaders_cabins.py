"""Tests for cabins config loader (v2.6 — V_cabin calibration)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import yaml

from configs.loaders import (
    get_v_cabin,
    is_cabin_calibrated,
    load_cabins_config,
)


@pytest.fixture
def cabins_yaml(tmp_path):
    f = tmp_path / "cabins.yaml"
    data = {
        "calibration_date": "2026-05-06",
        "calibrator": "tester",
        "cabins": {
            1: {"v_cabin": 3.45e-4, "u_v_cabin": 6e-6, "notes": "calibrated"},
            2: {"v_cabin": 3.50e-4, "u_v_cabin": 7e-6, "notes": "初始占位值"},
            3: {"v_cabin": 3.20e-4, "u_v_cabin": 5e-6, "notes": "calibrated 2026-05-07"},
        },
        "default": {"v_cabin": 3.50e-4, "u_v_cabin": 1e-5, "notes": "fallback"},
    }
    f.write_text(yaml.dump(data, allow_unicode=True), encoding="utf-8")
    return f


class TestLoadCabinsConfig:
    def test_load_basic(self, cabins_yaml):
        cfg = load_cabins_config(cabins_yaml)
        assert cfg["calibration_date"] == "2026-05-06"
        assert cfg["calibrator"] == "tester"
        assert 1 in cfg["cabins"]
        assert "default" in cfg

    def test_load_missing_file(self, tmp_path):
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            load_cabins_config(missing)

    def test_load_real_repo_config(self):
        """Production configs/cabins.yaml ships with 25 placeholder entries."""
        cfg = load_cabins_config()  # default = configs/cabins.yaml
        assert "cabins" in cfg
        assert "default" in cfg
        # 25 cabins must be present
        assert set(cfg["cabins"].keys()) == set(range(1, 26))


class TestGetVCabin:
    def test_calibrated_returns_actual_value(self, cabins_yaml):
        cfg = load_cabins_config(cabins_yaml)
        v, u = get_v_cabin(cfg, 1)
        assert v == pytest.approx(3.45e-4)
        assert u == pytest.approx(6e-6)

    def test_unknown_cabin_falls_back_to_default(self, cabins_yaml):
        cfg = load_cabins_config(cabins_yaml)
        v, u = get_v_cabin(cfg, 99)
        assert v == pytest.approx(3.50e-4)
        assert u == pytest.approx(1e-5)

    def test_no_default_block_uses_hardcoded(self, tmp_path):
        f = tmp_path / "minimal.yaml"
        f.write_text(yaml.dump({"cabins": {}}), encoding="utf-8")
        cfg = load_cabins_config(f)
        v, u = get_v_cabin(cfg, 5)
        # Hardcoded fallback: 3.5e-4, 1e-5
        assert v == pytest.approx(3.5e-4)
        assert u == pytest.approx(1.0e-5)

    def test_returns_floats_not_strings(self, cabins_yaml):
        cfg = load_cabins_config(cabins_yaml)
        v, u = get_v_cabin(cfg, 1)
        assert isinstance(v, float)
        assert isinstance(u, float)


class TestIsCabinCalibrated:
    def test_calibrated(self, cabins_yaml):
        cfg = load_cabins_config(cabins_yaml)
        assert is_cabin_calibrated(cfg, 1) is True
        assert is_cabin_calibrated(cfg, 3) is True

    def test_placeholder_not_calibrated(self, cabins_yaml):
        cfg = load_cabins_config(cabins_yaml)
        assert is_cabin_calibrated(cfg, 2) is False  # notes contains "占位"

    def test_missing_cabin_not_calibrated(self, cabins_yaml):
        cfg = load_cabins_config(cabins_yaml)
        assert is_cabin_calibrated(cfg, 99) is False

    def test_real_repo_all_placeholders(self):
        """Initial repo state: all 25 cabins are placeholders (notes contains '占位')."""
        cfg = load_cabins_config()
        for cid in range(1, 26):
            assert is_cabin_calibrated(cfg, cid) is False
