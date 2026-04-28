"""M1 model — per-cabin linear regression for Q estimation (v2.6).

Each cabin has independent (β, α) coefficients trained from rotation
calibration data. Inference is plain arithmetic; no ML library needed.

Coefficient table format (JSON):

    {
        "version": "v2.6.0",
        "trained_at": "2026-05-22T14:00:00",
        "feature": "hold_trend_slope",
        "target": "Q (Pa·m³/s)",
        "primary_section": "hold",
        "cabins": {
            "1": {
                "beta": 12.3, "alpha": 0.001,
                "r_squared": 0.998, "n_samples": 150,
                "u_beta": 0.4, "u_alpha": 0.0002
            },
            ...
        }
    }

Inference:
    Q = beta * primary_trend_slope + alpha
    u_Q² = (u_beta * slope)² + u_alpha²       (uncorrelated 1-sigma combine)

When a cabin is not in the table, we fall back to the mean of all
calibrated cabins and flag ``cabin_calibrated=False`` so callers can
emit F011 / down-rate confidence.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class LinearRegressionM1:
    """Per-cabin linear regression model.

    The coefficients live in ``models/artifacts/current/m1_coefficients.json``
    by default; the path is read from ``models.yaml::m1.coefficients_path``.
    """

    def __init__(self, models_cfg: Dict[str, Any], base_dir: str | Path = "."):
        self._base = Path(base_dir)
        m1_cfg = (models_cfg or {}).get("m1", {}) or {}
        self._coef_path = self._base / m1_cfg.get(
            "coefficients_path", "models/artifacts/current/m1_coefficients.json"
        )
        self._version: str = m1_cfg.get("version", "unknown")
        self._coefs: Dict[int, Dict[str, float]] = {}
        self._loaded = False
        self._primary_section: str = "hold"

    # ── public properties ─────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def version(self) -> str:
        return self._version

    @property
    def primary_section(self) -> str:
        return self._primary_section

    @property
    def calibrated_cabins(self) -> List[int]:
        return sorted(self._coefs.keys())

    # ── lifecycle ─────────────────────────────────────────────────

    def load(self) -> None:
        """Load coefficient table from JSON file.

        Raises
        ------
        FileNotFoundError if the file is missing. ``loaded`` stays False.
        """
        if not self._coef_path.exists():
            self._loaded = False
            raise FileNotFoundError(f"M1 coefficients not found: {self._coef_path}")

        with open(self._coef_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        self._version = data.get("version", self._version)
        self._primary_section = data.get("primary_section", "hold")
        # Cabin IDs in JSON are strings; coerce back to int.
        raw = data.get("cabins", {}) or {}
        self._coefs = {int(k): v for k, v in raw.items()}
        self._loaded = True

        logger.info(
            "M1 loaded: version=%s, %d cabins, primary_section=%s",
            self._version, len(self._coefs), self._primary_section,
        )

    # ── inference ────────────────────────────────────────────────

    def predict(self, primary_trend_slope: float, cabin_id: int) -> Dict[str, Any]:
        """Predict Q_est for one cabin.

        Parameters
        ----------
        primary_trend_slope : float
            The trend_slope of the primary_section (typically ``hold``).
        cabin_id : int

        Returns
        -------
        dict
            q_est : float (Pa·m³/s)
            uncertainty : float (1-sigma absolute)
            relative_uncertainty : float (uncertainty / |q_est|)
            cabin_calibrated : bool
        """
        coef = self._coefs.get(cabin_id)
        if coef is None:
            return self._predict_fallback(primary_trend_slope)

        beta = float(coef["beta"])
        alpha = float(coef.get("alpha", 0.0))
        u_beta = float(coef.get("u_beta", 0.0))
        u_alpha = float(coef.get("u_alpha", 0.0))

        q = beta * primary_trend_slope + alpha
        # u² = (u_beta * slope)² + u_alpha²  (assume β, α uncorrelated)
        uncertainty = ((u_beta * primary_trend_slope) ** 2 + u_alpha ** 2) ** 0.5
        rel_unc = uncertainty / abs(q) if abs(q) > 1e-12 else 1.0

        return {
            "q_est": q,
            "uncertainty": uncertainty,
            "relative_uncertainty": rel_unc,
            "cabin_calibrated": True,
        }

    # ── private ──────────────────────────────────────────────────

    def _predict_fallback(self, primary_trend_slope: float) -> Dict[str, Any]:
        """Cabin not in table: use mean of calibrated cabins, flag uncalibrated.

        If no cabins are calibrated at all (M1 effectively empty), return
        a zero estimate with infinite uncertainty so callers can refuse.
        """
        if not self._coefs:
            return {
                "q_est": 0.0,
                "uncertainty": float("inf"),
                "relative_uncertainty": 1.0,
                "cabin_calibrated": False,
            }
        avg_beta = sum(c["beta"] for c in self._coefs.values()) / len(self._coefs)
        avg_alpha = sum(c.get("alpha", 0.0) for c in self._coefs.values()) / len(self._coefs)
        q = avg_beta * primary_trend_slope + avg_alpha
        # Conservative 30% relative uncertainty for the fallback path
        uncertainty = abs(q) * 0.30 if q != 0 else 1e-5
        return {
            "q_est": q,
            "uncertainty": uncertainty,
            "relative_uncertainty": 0.30,
            "cabin_calibrated": False,
        }
