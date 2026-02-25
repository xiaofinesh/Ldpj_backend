"""XGBoost supervised model wrapper for inference.

Loads a trained XGBoost model and StandardScaler from disk, and provides
a simple ``predict`` interface that accepts raw feature vectors.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.exceptions import ModelLoadError, ModelPredictError

logger = logging.getLogger(__name__)


class SupervisedXGB:
    """Wrapper around a trained XGBoost classifier + StandardScaler.

    Parameters
    ----------
    models_cfg : dict
        The full content of ``models.yaml``.
    base_dir : Path | str
        Project root directory used to resolve relative model paths.
    """

    def __init__(self, models_cfg: Dict[str, Any], base_dir: Path | str = "."):
        self._base = Path(base_dir)
        current = models_cfg.get("current", {})
        self._model_path = self._base / current.get("model_path", "models/artifacts/current/xgb_model.json")
        self._scaler_path = self._base / current.get("scaler_path", "models/artifacts/current/xgb_scaler.joblib")
        self._version = current.get("version", "unknown")

        self._model: Any = None
        self._scaler: Any = None
        self._loaded = False

    # -- public interface ----------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def version(self) -> str:
        return self._version

    def load(self) -> None:
        """Load model and scaler from disk.

        Raises
        ------
        ModelLoadError
            If any required file is missing or corrupted.
        """
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

    def predict(self, features: List[float], threshold: float = 0.3) -> Dict[str, Any]:
        """Run inference on a single feature vector.

        Parameters
        ----------
        features : list[float]
            Ordered feature vector matching the feature contract.
        threshold : float
            Classification threshold. Probability >= threshold => label 1 (OK).

        Returns
        -------
        dict
            ``{"label": int, "probability": float, "confidence": float}``

        Raises
        ------
        ModelPredictError
            If inference fails.
        """
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

            return {
                "label": label,
                "probability": round(prob, 6),
                "confidence": round(confidence, 6),
            }

        except Exception as exc:
            raise ModelPredictError(f"Inference failed: {exc}") from exc

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the currently loaded model."""
        meta_path = self._model_path.parent / "metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                pass
        return {"version": self._version, "loaded": self._loaded}
