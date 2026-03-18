"""XGBoost supervised model wrapper for inference."""

from __future__ import annotations
import json, logging
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from core.exceptions import ModelLoadError, ModelPredictError

logger = logging.getLogger(__name__)


class SupervisedXGB:
    def __init__(self, models_cfg: Dict[str, Any], base_dir: Path | str = "."):
        self._base = Path(base_dir)
        current = models_cfg.get("current", {})
        self._model_path = self._base / current.get("model_path", "models/artifacts/current/xgb_model.json")
        self._scaler_path = self._base / current.get("scaler_path", "models/artifacts/current/xgb_scaler.joblib")
        self._version = current.get("version", "unknown")
        self._model: Any = None
        self._scaler: Any = None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def version(self) -> str:
        return self._version

    def load(self) -> None:
        try:
            import xgboost as xgb
            import joblib
            if not self._model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self._model_path}")
            if not self._scaler_path.exists():
                raise FileNotFoundError(f"Scaler file not found: {self._scaler_path}")
            booster = xgb.Booster()
            booster.load_model(str(self._model_path))
            self._model = booster
            self._scaler = joblib.load(str(self._scaler_path))
            self._loaded = True
            logger.info("Model loaded: version=%s, path=%s", self._version, self._model_path)
        except Exception as exc:
            self._loaded = False
            raise ModelLoadError(f"Failed to load model: {exc}") from exc

    def predict(self, features: List[float], threshold: float = 0.25) -> Dict[str, Any]:
        """Probability >= threshold => label 1 (OK), else label 0 (Leak)."""
        if not self._loaded:
            raise ModelPredictError("Model not loaded")
        try:
            import xgboost as xgb
            arr = np.array(features, dtype=np.float64).reshape(1, -1)
            arr_scaled = self._scaler.transform(arr)
            dmat = xgb.DMatrix(arr_scaled)
            prob = float(self._model.predict(dmat)[0])
            label = 1 if prob >= threshold else 0
            confidence = prob if label == 1 else (1.0 - prob)
            return {"label": label, "probability": round(prob, 6), "confidence": round(confidence, 6)}
        except Exception as exc:
            raise ModelPredictError(f"Inference failed: {exc}") from exc

    def get_metadata(self) -> Dict[str, Any]:
        meta_path = self._model_path.parent / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass
        return {"version": self._version, "loaded": self._loaded}
