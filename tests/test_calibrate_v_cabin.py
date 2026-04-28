"""Tests for scripts/calibrate_v_cabin.py CLI tool."""

import csv
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _import_calibrate_module(monkeypatch, tmp_path):
    """Load scripts/calibrate_v_cabin.py with CABINS_YAML / LOG_CSV repointed
    at tmp_path so tests do not touch the real repo files."""
    mod_path = PROJECT_ROOT / "scripts" / "calibrate_v_cabin.py"
    spec = importlib.util.spec_from_file_location("calibrate_v_cabin", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "CABINS_YAML", tmp_path / "cabins.yaml")
    monkeypatch.setattr(mod, "LOG_CSV", tmp_path / "log" / "v_cabin_log.csv")
    # Seed an empty cabins.yaml the script will mutate
    (tmp_path / "cabins.yaml").write_text(
        yaml.dump({"calibration_date": "", "calibrator": "",
                   "cabins": {}, "default": {"v_cabin": 3.5e-4, "u_v_cabin": 1e-5}},
                  allow_unicode=True),
        encoding="utf-8",
    )
    return mod


class TestStats:
    def test_unit_conversion_350g_to_3_5e_minus_4_m3(self, monkeypatch, tmp_path):
        mod = _import_calibrate_module(monkeypatch, tmp_path)
        rc = mod.main([
            "--cabin", "5",
            "--weights-grams", "350.0,350.0,350.0",
            "--calibrator", "alice",
        ])
        assert rc == 0
        cfg = yaml.safe_load((tmp_path / "cabins.yaml").read_text(encoding="utf-8"))
        v_cabin = float(cfg["cabins"][5]["v_cabin"])
        # 350 g water → 3.5e-4 m³
        assert v_cabin == pytest.approx(3.5e-4, rel=1e-4)


class TestAcceptedRun:
    def test_writes_yaml_and_log(self, monkeypatch, tmp_path):
        mod = _import_calibrate_module(monkeypatch, tmp_path)
        rc = mod.main([
            "--cabin", "7",
            "--weights-grams", "348.2,348.5,348.0",
            "--calibrator", "alice",
            "--notes", "first batch",
        ])
        assert rc == 0
        # YAML updated
        cfg = yaml.safe_load((tmp_path / "cabins.yaml").read_text(encoding="utf-8"))
        entry = cfg["cabins"][7]
        assert entry["v_cabin"] == pytest.approx(348.23 * 1e-6, rel=1e-3)
        assert "first batch" in entry["notes"]
        # Log appended with accepted=yes
        rows = list(csv.reader((tmp_path / "log" / "v_cabin_log.csv").open(encoding="utf-8")))
        assert rows[0][-1] == "accepted"   # header
        assert rows[1][-1] == "yes"         # data row
        assert rows[1][1] == "7"            # cabin_id


class TestRejectedRun:
    def test_high_cv_rejects_yaml_but_logs_history(self, monkeypatch, tmp_path):
        mod = _import_calibrate_module(monkeypatch, tmp_path)
        rc = mod.main([
            "--cabin", "9",
            "--weights-grams", "300,400,350",  # CV ~14%, way above 2%
            "--calibrator", "alice",
        ])
        assert rc == 1
        # YAML untouched (no cabin 9 entry)
        cfg = yaml.safe_load((tmp_path / "cabins.yaml").read_text(encoding="utf-8"))
        assert 9 not in cfg["cabins"]
        # Log still got the rejected attempt
        rows = list(csv.reader((tmp_path / "log" / "v_cabin_log.csv").open(encoding="utf-8")))
        assert rows[1][-1] == "no"           # accepted=no
        assert "REJECTED" in rows[1][-2]     # notes column


class TestDryRun:
    def test_dry_run_neither_writes_nor_logs(self, monkeypatch, tmp_path):
        mod = _import_calibrate_module(monkeypatch, tmp_path)
        original = (tmp_path / "cabins.yaml").read_text(encoding="utf-8")
        rc = mod.main([
            "--cabin", "3",
            "--weights-grams", "348.2,348.5,348.0",
            "--dry-run",
        ])
        assert rc == 0
        assert (tmp_path / "cabins.yaml").read_text(encoding="utf-8") == original
        assert not (tmp_path / "log" / "v_cabin_log.csv").exists()


class TestInputValidation:
    def test_too_few_repeats(self, monkeypatch, tmp_path, capsys):
        mod = _import_calibrate_module(monkeypatch, tmp_path)
        rc = mod.main(["--cabin", "1", "--weights-grams", "350.0,350.0"])
        assert rc == 2
        assert "need >= 3 repeats" in capsys.readouterr().err

    def test_cabin_out_of_range(self, monkeypatch, tmp_path, capsys):
        mod = _import_calibrate_module(monkeypatch, tmp_path)
        rc = mod.main(["--cabin", "0", "--weights-grams", "350,350,350"])
        assert rc == 2
        assert "cabin must be in" in capsys.readouterr().err

    def test_negative_weight(self, monkeypatch, tmp_path, capsys):
        mod = _import_calibrate_module(monkeypatch, tmp_path)
        rc = mod.main(["--cabin", "1", "--weights-grams", "350,-1,350"])
        assert rc == 2
        assert "must be positive" in capsys.readouterr().err
