"""Train M1 — per-cabin linear regression for Q estimation (Task 10).

Reads the CSV produced by ``prepare_q_data`` (must contain ``q_measured``
and either a feature column named ``<primary>_trend_slope`` directly or
the v2.5/v2.6 ``features`` JSON column). For each cabin, fits

    q_measured = β · slope + α

via ordinary least squares and bootstraps (default 1000 resamples) for
1-sigma uncertainties on β and α. Writes a JSON file consumable by
``models.linear_regression_m1.LinearRegressionM1``.

Usage
-----
    python -m train.train_m1 \\
        --data train_data_S2.csv \\
        --output models/artifacts/v2.6.0/m1_coefficients.json \\
        --version v2.6.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.loaders import load_runtime_config
from core.cycle_profile import load_active_cycle_profile

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train M1 per-cabin linear regression")
    p.add_argument("--data", required=True, help="Q-labeled CSV from prepare_q_data")
    p.add_argument("--output", required=True, help="Output coefficients JSON")
    p.add_argument("--version", default="v2.6.0")
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    p.add_argument("--min-r2", type=float, default=0.99,
                   help="Acceptance threshold; cabins below are flagged but still saved")
    p.add_argument("--min-samples-per-cabin", type=int, default=20)
    p.add_argument("--feature-name", default=None,
                   help="Feature column name (default: <primary>_trend_slope)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def fit_one_cabin(slopes: np.ndarray, q_values: np.ndarray,
                  bootstrap_samples: int, seed: int) -> Optional[Dict[str, Any]]:
    """Fit Q = β · slope + α with least squares; bootstrap for u_β / u_α.

    Returns None if there are too few samples for a meaningful fit.
    """
    n = len(slopes)
    if n < 5:
        return None

    slopes = np.asarray(slopes, dtype=np.float64)
    q_values = np.asarray(q_values, dtype=np.float64)

    coef = np.polyfit(slopes, q_values, 1)
    beta, alpha = float(coef[0]), float(coef[1])

    q_pred = beta * slopes + alpha
    ss_res = float(np.sum((q_values - q_pred) ** 2))
    ss_tot = float(np.sum((q_values - np.mean(q_values)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    rng = np.random.default_rng(seed)
    betas, alphas = [], []
    for _ in range(bootstrap_samples):
        idx = rng.choice(n, size=n, replace=True)
        try:
            c = np.polyfit(slopes[idx], q_values[idx], 1)
            betas.append(c[0])
            alphas.append(c[1])
        except Exception:
            pass

    u_beta = float(np.std(betas)) if betas else 0.0
    u_alpha = float(np.std(alphas)) if alphas else 0.0

    return {
        "beta": beta, "alpha": alpha,
        "r_squared": float(r_squared),
        "u_beta": u_beta, "u_alpha": u_alpha,
        "n_samples": int(n),
    }


def _resolve_slope_column(df: pd.DataFrame, feature_name: str) -> pd.Series:
    """Return a Series of slope values, drawn either from a direct column or
    from the ``features`` JSON column of the row."""
    if feature_name in df.columns:
        return df[feature_name]
    if "features" in df.columns:
        def _extract(s):
            if not isinstance(s, str):
                return None
            try:
                return float(json.loads(s).get(feature_name, 0.0))
            except (json.JSONDecodeError, TypeError, ValueError):
                return None
        return df["features"].apply(_extract)
    raise ValueError(f"Cannot find feature '{feature_name}' (no column, no 'features' JSON)")


def main(argv=None) -> int:
    args = parse_args(argv)

    runtime_cfg = load_runtime_config()
    profile = load_active_cycle_profile(runtime_cfg)
    feature_name = args.feature_name or f"{profile.primary_section}_trend_slope"
    logger.info("Using feature: %s", feature_name)

    df = pd.read_csv(args.data)
    logger.info("Loaded %d rows", len(df))

    if "q_measured" not in df.columns:
        sys.exit("ERROR: input CSV must contain 'q_measured'")

    df["_slope"] = _resolve_slope_column(df, feature_name)
    df_valid = df.dropna(subset=["q_measured", "_slope"]).copy()
    logger.info("After cleaning: %d valid rows", len(df_valid))

    cabin_coefs: Dict[int, Dict[str, Any]] = {}
    failed = []
    for cabin_id in sorted(df_valid["cavity_id"].unique()):
        sub = df_valid[df_valid["cavity_id"] == cabin_id]
        if len(sub) < args.min_samples_per_cabin:
            logger.warning("Cabin %d: only %d samples (< %d), skipping",
                           cabin_id, len(sub), args.min_samples_per_cabin)
            continue
        result = fit_one_cabin(
            sub["_slope"].to_numpy(),
            sub["q_measured"].to_numpy(),
            args.bootstrap_samples,
            args.seed + int(cabin_id),
        )
        if result is None:
            continue

        passed = result["r_squared"] >= args.min_r2
        if not passed:
            # Strict quality gate: do NOT write rejected cabins to the
            # output JSON. At inference time, LinearRegressionM1 will
            # treat the cabin as uncalibrated → fall back to the
            # global mean and the processing loop raises F011, which is
            # the intended behavior for sensors that haven't been
            # successfully calibrated yet.
            logger.warning(
                "Cabin %d: R²=%.4f < %.2f — REJECTED, "
                "will fall back to global mean at inference (F011)",
                cabin_id, result["r_squared"], args.min_r2,
            )
            failed.append(int(cabin_id))
            continue

        result["passes_acceptance"] = True
        cabin_coefs[int(cabin_id)] = result
        logger.info("Cabin %d: β=%.3e α=%.3e R²=%.4f n=%d",
                    cabin_id, result["beta"], result["alpha"],
                    result["r_squared"], result["n_samples"])

    out = {
        "version": args.version,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "feature": feature_name,
        "target": "Q (Pa·m³/s)",
        "primary_section": profile.primary_section,
        "n_cabins_calibrated": len(cabin_coefs),
        "n_cabins_failed_acceptance": len(failed),
        "acceptance": {"min_r2": args.min_r2},
        "cabins": cabin_coefs,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    logger.info("Wrote M1 coefficients: %s", out_path)
    if failed:
        logger.warning("Cabins failing acceptance: %s", failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
