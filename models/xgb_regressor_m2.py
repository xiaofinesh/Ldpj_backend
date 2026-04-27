"""M2 model — global XGBoost regressor for Q estimation (v2.6).

Input: 43-dim feature vector (or a subset selected during training).
Output: Q_est in Pa·m³/s.

Training is done in log10(Q) space to handle the wide dynamic range
(~1 order of magnitude across leak severities) and keep relative rather
than absolute error in focus. Predictions are returned in linear Q space.

Metadata contract (m2_metadata.json):
    {
        "version": "v2.6.0",
        "feature_subset": ["hold_trend_slope", "hold_max", ..., "cavity_id"],
        "log_space": true,
        "feature_importance": {...},
        "evaluation": {"r_squared": 0.97, "mae": 0.12}
    }

If ``feature_subset`` is missing, the full 43-dim vector is assumed.
``log_space=true`` (default) means the booster's output y is interpreted
as log10(Q); we apply 10**y for the final answer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from core.exceptions import ModelLoadError, ModelPredictError

logger = logging.getLogger(__name__)


# Cap log10(Q) into a sane range before exponentiating; predictions outside
# this range are clamped (we'd rather report a saturated value than an
# overflow). Q range = [1e-12, 1e0] = atto-scale leak to wide-open vacuum.
_LOG10Q_MIN = -12.0
_LOG10Q_MAX = 0.0


class XGBRegressorM2:
    """Global XGBoost regressor for Q.

    Coefficients live in ``models/artifacts/current/`` by default; paths
    are read from ``models.yaml::m2``.
    """

    def __init__(self, models_cfg: Dict[str, Any], base_dir: str | Path = "."):
        self._base = Path(base_dir)
        m2_cfg = (models_cfg or {}).get("m2", {}) or {}
        self._model_path = self._base / m2_cfg.get(
            "model_path", "models/artifacts/current/m2_xgb_model.json"
        )
        self._scaler_path = self._base / m2_cfg.get(
            "scaler_path", "models/artifacts/current/m2_xgb_scaler.joblib"
        )
        self._metadata_path = self._base / m2_cfg.get(
            "metadata_path", "models/artifacts/current/m2_metadata.json"
        )
        self._version: str = m2_cfg.get("version", "unknown")
        self._model: Any = None
        self._scaler: Any = None
        self._loaded = False
        self._log_space = True
        # Subset of FEATURE_ORDER_43D actually used by this model
        self._feature_subset: List[str] = []
        # Indices into the full 43-dim vector (computed at load time)
        self._feature_indices: List[int] = []

    # ── public properties ─────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def version(self) -> str:
        return self._version

    @property
    def feature_subset(self) -> List[str]:
        return list(self._feature_subset)

    @property
    def log_space(self) -> bool:
        return self._log_space

    # ── lifecycle ─────────────────────────────────────────────────

    def load(self) -> None:
        """Load the booster, optional scaler, and metadata.

        Raises
        ------
        ModelLoadError if the model file cannot be opened or metadata is
        inconsistent (e.g. feature_subset references unknown names).
        """
        from core.feature_spec import FEATURE_ORDER_43D

        try:
            import xgboost as xgb  # local import: training-only env friendly
            import joblib

            if not self._model_path.exists():
                raise FileNotFoundError(f"M2 model not found: {self._model_path}")

            booster = xgb.Booster()
            booster.load_model(str(self._model_path))
            self._model = booster

            self._scaler = None
            if self._scaler_path.exists():
                self._scaler = joblib.load(str(self._scaler_path))

            if self._metadata_path.exists():
                with open(self._metadata_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                self._version = meta.get("version", self._version)
                self._feature_subset = list(meta.get("feature_subset") or FEATURE_ORDER_43D)
                self._log_space = bool(meta.get("log_space", True))
            else:
                logger.warning(
                    "M2 metadata not found; assuming full 43-dim and log-space"
                )
                self._feature_subset = list(FEATURE_ORDER_43D)
                self._log_space = True

            # Pre-compute indices into the full 43-dim vector
            try:
                self._feature_indices = [
                    FEATURE_ORDER_43D.index(name) for name in self._feature_subset
                ]
            except ValueError as exc:
                raise ModelLoadError(
                    f"M2 metadata feature_subset references unknown feature: {exc}"
                ) from exc

            self._loaded = True
            logger.info(
                "M2 loaded: version=%s, %d features, log_space=%s",
                self._version, len(self._feature_subset), self._log_space,
            )
        except ModelLoadError:
            self._loaded = False
            raise
        except Exception as exc:
            self._loaded = False
            raise ModelLoadError(f"Failed to load M2: {exc}") from exc

    # ── inference ────────────────────────────────────────────────

    def predict(self, full_features: List[float]) -> Dict[str, Any]:
        """Predict Q_est from the full 43-dim feature vector.

        The 43-dim contract is enforced; the model internally selects the
        ``feature_subset`` dictated by metadata. This keeps callers from
        having to know which features the model uses.

        Parameters
        ----------
        full_features : list of float, length 43

        Returns
        -------
        dict with q_est (Pa·m³/s) and valid (bool).
        """
        if not self._loaded:
            return {"q_est": 0.0, "valid": False}

        try:
            import xgboost as xgb

            if len(full_features) != 43:
                raise ValueError(
                    f"M2 expects 43-dim input, got {len(full_features)}"
                )

            x_subset = np.asarray(
                [full_features[i] for i in self._feature_indices],
                dtype=np.float32,
            ).reshape(1, -1)

            if self._scaler is not None:
                x_subset = self._scaler.transform(x_subset)

            # Pass feature_names so a booster saved with them (training side
            # uses feature_names for importance ranking) doesn't error here.
            dmat = xgb.DMatrix(x_subset, feature_names=self._feature_subset)
            y_pred = float(self._model.predict(dmat)[0])

            if self._log_space:
                # Clamp to a sane range before exponentiating; un-clamped
                # predictions on degenerate inputs can otherwise blow up.
                y_pred = max(_LOG10Q_MIN, min(_LOG10Q_MAX, y_pred))
                q_est = 10.0 ** y_pred
            else:
                q_est = y_pred

            return {"q_est": q_est, "valid": True}
        except Exception as exc:
            raise ModelPredictError(f"M2 predict failed: {exc}") from exc
