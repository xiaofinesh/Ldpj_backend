"""Tests for train.train_m2 (v2.6 Task 10)."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.xgb_regressor_m2 import XGBRegressorM2
from tests.fixtures.generate_mock_q_data import generate
from train.train_m2 import main as train_main


@pytest.fixture
def labeled_csv(tmp_path):
    """Larger dataset so XGB has something to learn from."""
    df = generate(n_cabins=10, n_rounds=30, seed=42)
    csv = tmp_path / "labeled.csv"
    df.to_csv(csv, index=False)
    return csv


class TestTrainM2:
    def test_writes_three_artifacts(self, tmp_path, labeled_csv):
        out_dir = tmp_path / "m2_out"
        rc = train_main([
            "--data", str(labeled_csv),
            "--output", str(out_dir),
            "--version", "test_v1",
            "--top-k-features", "8",
            "--n-estimators", "30",       # cheap for tests
            "--max-depth", "3",
        ])
        assert rc == 0
        assert (out_dir / "m2_xgb_model.json").exists()
        assert (out_dir / "m2_xgb_scaler.joblib").exists()
        assert (out_dir / "m2_metadata.json").exists()

    def test_metadata_describes_subset(self, tmp_path, labeled_csv):
        out_dir = tmp_path / "m2_out"
        top_k = 5
        rc = train_main([
            "--data", str(labeled_csv),
            "--output", str(out_dir),
            "--top-k-features", str(top_k),
            "--n-estimators", "30",
        ])
        assert rc == 0
        meta = json.loads((out_dir / "m2_metadata.json").read_text(encoding="utf-8"))
        assert meta["log_space"] is True
        # XGBoost may rank fewer than top_k features when the signal is
        # concentrated; subset is "top-K or however many were used", whichever
        # is smaller. Always > 0 and never above top_k.
        assert 0 < len(meta["feature_subset"]) <= top_k
        # Importance map keyed by the same names as feature_subset
        for name in meta["feature_subset"]:
            assert name in meta["feature_importance"]
        # Evaluation block exists with the expected keys
        for key in ("train_r2", "test_r2", "test_mae_pa_m3_s"):
            assert key in meta["evaluation"]

    def test_model_round_trip_via_xgbregressor_m2(self, tmp_path, labeled_csv):
        """Train M2, reload through XGBRegressorM2, predict on real input."""
        out_dir = tmp_path / "m2_out"
        top_k = 6
        rc = train_main([
            "--data", str(labeled_csv),
            "--output", str(out_dir),
            "--version", "test_v1",
            "--top-k-features", str(top_k),
            "--n-estimators", "30",
        ])
        assert rc == 0

        # Build the dict that matches models.yaml::m2 paths
        cfg = {"m2": {
            "model_path": str((out_dir / "m2_xgb_model.json").relative_to(tmp_path)),
            "scaler_path": str((out_dir / "m2_xgb_scaler.joblib").relative_to(tmp_path)),
            "metadata_path": str((out_dir / "m2_metadata.json").relative_to(tmp_path)),
        }}
        m2 = XGBRegressorM2(cfg, base_dir=tmp_path)
        m2.load()
        assert m2.loaded
        # XGBoost may rank fewer than top_k useful features (see note above)
        assert 0 < len(m2.feature_subset) <= top_k
        assert m2.log_space is True

        # Predict on a real row's full 36-dim feature vector
        from core.feature_spec import FEATURE_ORDER_36D
        df = pd.read_csv(labeled_csv)
        feats = json.loads(df.iloc[0]["features"])
        full_vec = [float(feats.get(k, 0.0)) for k in FEATURE_ORDER_36D]
        result = m2.predict(full_vec)
        assert result["valid"] is True
        # Q is bounded by log10 clamp
        assert 0 < result["q_est"] <= 1.0
