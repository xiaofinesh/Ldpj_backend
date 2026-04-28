"""Train M2 — global XGBoost regressor for Q estimation (Task 10).

Two-pass training with feature selection:

  Pass 1: train on all 36 features → use gain importance to rank.
  Pass 2: keep top-K, refit a fresh StandardScaler on the reduced
          input, retrain with the same hyperparameters.

Targets are in log10(Q) space (Q is clipped to [1e-7, 1.0] to keep log
finite). The per-feature scaler shipped with the model is the
*reduced* one, so deployment-time inference matches training-time exactly.

Train/test split: by ``round_id`` to avoid leakage between consecutive
shots of the same cycle. If ``round_id`` is absent, falls back to a
random 80/20 split (with a warning).

Usage
-----
    python -m train.train_m2 \\
        --data train_data_S2.csv \\
        --output models/artifacts/v2.6.0/ \\
        --version v2.6.0 \\
        --top-k-features 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.feature_spec import FEATURE_ORDER_36D

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train M2 XGBoost regressor")
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--version", default="v2.6.0")
    p.add_argument("--top-k-features", type=int, default=20,
                   help="Keep top-K features by gain importance after pass 1")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--reg-lambda", type=float, default=1.0)
    p.add_argument("--holdout-round", type=int, default=5,
                   help="round_id used as holdout (default 5)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def extract_features(df: pd.DataFrame, names: List[str]) -> np.ndarray:
    """Pull a numeric matrix out of the CSV's ``features`` JSON column."""
    rows = []
    for s in df["features"]:
        feats = json.loads(s) if isinstance(s, str) else (s or {})
        rows.append([float(feats.get(name, 0.0)) for name in names])
    return np.asarray(rows, dtype=np.float32)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Local imports so module imports without xgboost installed (rare on edge)
    import joblib
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(args.data)
    df = df.dropna(subset=["q_measured", "features"]).reset_index(drop=True)
    logger.info("Loaded %d valid rows", len(df))

    if "round_id" not in df.columns:
        logger.warning("No round_id column; using random 80/20 split")
        rng = np.random.default_rng(args.seed)
        df["round_id"] = rng.integers(1, 6, len(df))

    # Train/test split by round
    df_train = df[df["round_id"] != args.holdout_round].reset_index(drop=True)
    df_test = df[df["round_id"] == args.holdout_round].reset_index(drop=True)
    logger.info("Train: %d rows (round != %d), Test: %d rows",
                len(df_train), args.holdout_round, len(df_test))

    if len(df_train) == 0 or len(df_test) == 0:
        sys.exit("ERROR: empty train or test split (check holdout-round)")

    # Extract full 36-dim features
    X_train = extract_features(df_train, FEATURE_ORDER_36D)
    X_test = extract_features(df_test, FEATURE_ORDER_36D)

    # Target: log10(Q), with Q clipped to a finite, monotonically meaningful range
    y_train_q = np.clip(df_train["q_measured"].to_numpy(), 1e-7, 1.0)
    y_test_q = np.clip(df_test["q_measured"].to_numpy(), 1e-7, 1.0)
    y_train = np.log10(y_train_q)
    y_test = np.log10(y_test_q)

    # ── Pass 1: full 36d for importance ranking ───────────────────
    scaler_full = StandardScaler().fit(X_train)
    X_train_s = scaler_full.transform(X_train)

    logger.info("Pass 1: training on all 36 features for gain ranking")
    booster_full = xgb.train(
        params={
            "max_depth": args.max_depth,
            "eta": args.learning_rate,
            "reg_lambda": args.reg_lambda,
            "objective": "reg:squarederror",
            "verbosity": 0,
            "seed": args.seed,
        },
        dtrain=xgb.DMatrix(X_train_s, label=y_train, feature_names=FEATURE_ORDER_36D),
        num_boost_round=args.n_estimators,
    )

    importance = booster_full.get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    if not sorted_imp:
        sys.exit("ERROR: no features survived pass 1 — input may be degenerate")

    top_k_names = [name for name, _ in sorted_imp[:args.top_k_features]]
    logger.info("Top-%d features (head): %s", args.top_k_features, top_k_names[:5])

    # ── Pass 2: refit scaler on selected raw features, retrain ────
    feature_indices = [FEATURE_ORDER_36D.index(name) for name in top_k_names]
    X_train_sel_raw = X_train[:, feature_indices]
    X_test_sel_raw = X_test[:, feature_indices]
    scaler_sel = StandardScaler().fit(X_train_sel_raw)
    X_train_sel = scaler_sel.transform(X_train_sel_raw)
    X_test_sel = scaler_sel.transform(X_test_sel_raw)

    logger.info("Pass 2: training with top-%d features", len(top_k_names))
    booster = xgb.train(
        params={
            "max_depth": args.max_depth,
            "eta": args.learning_rate,
            "reg_lambda": args.reg_lambda,
            "objective": "reg:squarederror",
            "verbosity": 0,
            "seed": args.seed,
        },
        dtrain=xgb.DMatrix(X_train_sel, label=y_train, feature_names=top_k_names),
        num_boost_round=args.n_estimators,
    )

    # ── Evaluation in linear Q space ─────────────────────────────
    dmat_train = xgb.DMatrix(X_train_sel, feature_names=top_k_names)
    dmat_test = xgb.DMatrix(X_test_sel, feature_names=top_k_names)
    y_train_pred = 10.0 ** booster.predict(dmat_train)
    y_test_pred = 10.0 ** booster.predict(dmat_test)

    train_r2 = float(r2_score(y_train_q, y_train_pred))
    test_r2 = float(r2_score(y_test_q, y_test_pred))
    test_mae = float(mean_absolute_error(y_test_q, y_test_pred))

    logger.info("Train R²=%.4f, Test R²=%.4f, Test MAE=%.3e",
                train_r2, test_r2, test_mae)
    if abs(train_r2 - test_r2) > 0.05:
        logger.warning("Possible overfitting: train R² %.4f vs test R² %.4f",
                       train_r2, test_r2)

    # ── Save artifacts ────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out_dir / "m2_xgb_model.json"))
    joblib.dump(scaler_sel, out_dir / "m2_xgb_scaler.joblib")

    metadata = {
        "version": args.version,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": args.data,
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "log_space": True,
        "feature_subset": top_k_names,
        "feature_importance": {n: float(v) for n, v in sorted_imp[:args.top_k_features]},
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "reg_lambda": args.reg_lambda,
        },
        "evaluation": {
            "train_r2": train_r2,
            "test_r2": test_r2,
            "test_mae_pa_m3_s": test_mae,
        },
    }
    with open(out_dir / "m2_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    logger.info("Wrote M2 artifacts to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
