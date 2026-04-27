"""Tests for M2 XGBoost regressor (v2.6)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from core.exceptions import ModelLoadError, ModelPredictError
from core.feature_spec import FEATURE_ORDER_43D
from models.xgb_regressor_m2 import XGBRegressorM2


def _train_synthetic_m2(tmp_path: Path, *,
                        feature_subset=None,
                        log_space: bool = True,
                        version: str = "test_v1",
                        n_samples: int = 200):
    """Train a tiny XGB regressor on synthetic data and dump artifacts.

    Returns (model_path, scaler_path, meta_path), all under tmp_path.
    """
    import xgboost as xgb
    import joblib
    from sklearn.preprocessing import StandardScaler

    if feature_subset is None:
        feature_subset = list(FEATURE_ORDER_43D)

    n_feat = len(feature_subset)
    rng = np.random.default_rng(42)
    X = rng.random((n_samples, n_feat)).astype(np.float32)

    # Synthetic target: depends primarily on the first selected feature
    if log_space:
        # log10(Q) = -3 + 2 * x[:, 0]  → Q in [1e-3, 1e-1]
        y = -3.0 + 2.0 * X[:, 0]
    else:
        y = 1e-3 * X[:, 0]

    scaler = StandardScaler().fit(X)
    dtrain = xgb.DMatrix(scaler.transform(X), label=y)
    booster = xgb.train(
        params={"max_depth": 3, "eta": 0.1, "objective": "reg:squarederror"},
        dtrain=dtrain,
        num_boost_round=30,
    )

    model_path = tmp_path / "m2_xgb_model.json"
    scaler_path = tmp_path / "m2_xgb_scaler.joblib"
    meta_path = tmp_path / "m2_metadata.json"

    booster.save_model(str(model_path))
    joblib.dump(scaler, scaler_path)
    meta_path.write_text(json.dumps({
        "version": version,
        "feature_subset": feature_subset,
        "log_space": log_space,
        "feature_importance": {},
        "evaluation": {"r_squared": 0.9},
    }), encoding="utf-8")

    return model_path, scaler_path, meta_path


def _cfg(model_path, scaler_path, meta_path, base_dir, version="test_v1"):
    return {"m2": {
        "model_path": str(model_path.relative_to(base_dir)),
        "scaler_path": str(scaler_path.relative_to(base_dir)),
        "metadata_path": str(meta_path.relative_to(base_dir)),
        "version": version,
    }}


# ── Load paths ────────────────────────────────────────────────────────

class TestLoad:
    def test_load_full_43d(self, tmp_path):
        mp, sp, mt = _train_synthetic_m2(tmp_path)
        m2 = XGBRegressorM2(_cfg(mp, sp, mt, tmp_path), base_dir=tmp_path)
        m2.load()
        assert m2.loaded is True
        assert m2.version == "test_v1"
        assert len(m2.feature_subset) == 43
        assert m2.log_space is True

    def test_load_missing_model_raises_modelloaderror(self, tmp_path):
        cfg = {"m2": {
            "model_path": "nonexistent.json",
            "scaler_path": "nonexistent.joblib",
            "metadata_path": "nonexistent.json",
        }}
        m2 = XGBRegressorM2(cfg, base_dir=tmp_path)
        with pytest.raises(ModelLoadError):
            m2.load()
        assert m2.loaded is False

    def test_load_subset_metadata(self, tmp_path):
        """When metadata's feature_subset is shorter, only those features are used."""
        subset = [
            "hold_max", "hold_min", "hold_trend_slope",
            "evac_trend_slope", "cavity_id",
        ]
        mp, sp, mt = _train_synthetic_m2(tmp_path, feature_subset=subset)
        m2 = XGBRegressorM2(_cfg(mp, sp, mt, tmp_path), base_dir=tmp_path)
        m2.load()
        assert m2.feature_subset == subset

    def test_load_unknown_feature_in_subset_raises(self, tmp_path):
        """Metadata mentioning a non-existent feature name should fail loudly."""
        # Train with a valid subset first, then corrupt metadata
        mp, sp, mt = _train_synthetic_m2(tmp_path, feature_subset=["hold_max"])
        meta = json.loads(mt.read_text(encoding="utf-8"))
        meta["feature_subset"] = ["does_not_exist_feature"]
        mt.write_text(json.dumps(meta), encoding="utf-8")

        m2 = XGBRegressorM2(_cfg(mp, sp, mt, tmp_path), base_dir=tmp_path)
        with pytest.raises(ModelLoadError, match="feature_subset"):
            m2.load()
        assert m2.loaded is False

    def test_load_without_metadata_falls_back_to_full(self, tmp_path):
        """If metadata is absent, M2 assumes full 43-dim + log_space."""
        mp, sp, mt = _train_synthetic_m2(tmp_path)
        mt.unlink()  # delete metadata
        m2 = XGBRegressorM2(_cfg(mp, sp, mt, tmp_path), base_dir=tmp_path)
        m2.load()
        assert m2.loaded
        assert len(m2.feature_subset) == 43
        assert m2.log_space is True


# ── Predict paths ────────────────────────────────────────────────────

class TestPredict:
    def test_predict_returns_positive_q_in_log_space(self, tmp_path):
        mp, sp, mt = _train_synthetic_m2(tmp_path)
        m2 = XGBRegressorM2(_cfg(mp, sp, mt, tmp_path), base_dir=tmp_path)
        m2.load()
        result = m2.predict([0.5] * 43)
        assert result["valid"] is True
        # Synthetic relation: log10(Q) ≈ -3 + 2*0.5 = -2 → Q ≈ 1e-2
        assert result["q_est"] > 0.0
        assert result["q_est"] < 1.0

    def test_predict_wrong_dim_raises_modelpredicterror(self, tmp_path):
        mp, sp, mt = _train_synthetic_m2(tmp_path)
        m2 = XGBRegressorM2(_cfg(mp, sp, mt, tmp_path), base_dir=tmp_path)
        m2.load()
        with pytest.raises(ModelPredictError):
            m2.predict([0.5] * 10)

    def test_predict_when_not_loaded(self, tmp_path):
        """Calling predict before load returns valid=False, doesn't crash."""
        m2 = XGBRegressorM2({}, base_dir=tmp_path)
        result = m2.predict([0.5] * 43)
        assert result == {"q_est": 0.0, "valid": False}

    def test_subset_prediction_uses_internal_indices(self, tmp_path):
        """Caller always passes 43-dim; M2 selects subset internally."""
        subset = ["hold_max", "hold_trend_slope", "cavity_id"]
        mp, sp, mt = _train_synthetic_m2(tmp_path, feature_subset=subset)
        m2 = XGBRegressorM2(_cfg(mp, sp, mt, tmp_path), base_dir=tmp_path)
        m2.load()
        # 43-dim input — model only consumes 3 of them
        result = m2.predict([0.5] * 43)
        assert result["valid"] is True

    def test_log_space_disabled_returns_raw_value(self, tmp_path):
        """When log_space=False, output is the booster's prediction directly."""
        mp, sp, mt = _train_synthetic_m2(tmp_path, log_space=False)
        m2 = XGBRegressorM2(_cfg(mp, sp, mt, tmp_path), base_dir=tmp_path)
        m2.load()
        assert m2.log_space is False
        result = m2.predict([0.5] * 43)
        # Synthetic linear-space target: Q ≈ 1e-3 * 0.5 = 5e-4 (small, positive)
        assert result["valid"] is True
        # Should be a small positive number (no 10** transform)
        assert 0 <= result["q_est"] < 0.1

    def test_log_space_clamps_extreme_predictions(self, tmp_path):
        """log10(Q) > 0 (Q > 1) is clamped before exponentiation."""
        mp, sp, mt = _train_synthetic_m2(tmp_path)
        m2 = XGBRegressorM2(_cfg(mp, sp, mt, tmp_path), base_dir=tmp_path)
        m2.load()
        # Even with absurd inputs, q_est must be finite and bounded
        result = m2.predict([1e6] * 43)
        assert result["valid"] is True
        assert result["q_est"] <= 1.0  # log10(Q) clamped at 0 → Q <= 1
        assert result["q_est"] > 0.0
