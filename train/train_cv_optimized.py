#!/usr/bin/env python3
"""Train XGBoost model with K-fold CV and threshold optimization.

Supports both new system data (labels from leak_valve_status, pre-computed
features) and old system data (labels from column, re-computed features).

Label convention (consistent with core/label_spec.py):
    label = 0: LEAK (漏液)
    label = 1: OK   (密封正常)

Usage
-----
    # New system data — labels from leak_valve_status, features pre-computed:
    python -m train.train_cv_optimized --data cleaned.csv --output models/artifacts/optimized \\
        --use-precomputed

    # New system data — re-compute features from pressure_data:
    python -m train.train_cv_optimized --data cleaned.csv --output models/artifacts/optimized

    # Old system data — labels from column, flipped:
    python -m train.train_cv_optimized --data old.csv --output models/artifacts/optimized \\
        --label-source column --label-column prediction --flip-labels
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FEATURE_ORDER_7D = ["max", "min", "difference", "average", "variance", "trend_slope", "cavity_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost with CV optimization")
    parser.add_argument("--data", required=True, help="Path to CSV data file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-folds", type=int, default=10, help="Number of CV folds")
    parser.add_argument("--version", default="optimized_v1", help="Model version string")

    # ── Label source selection ──────────────────────────────────────────
    parser.add_argument(
        "--label-source", choices=["valve", "column"], default="valve",
        help="How to derive labels. "
             "'valve' (default): use leak_valve_status column (1=LEAK, 0=OK). "
             "'column': use --label-column with optional --flip-labels.",
    )
    parser.add_argument(
        "--flip-labels", action="store_true",
        help="Flip labels from old system convention (0=OK,1=LEAK) to new (0=LEAK,1=OK).",
    )
    parser.add_argument(
        "--label-column", default="label",
        help="Name of the label column (only used when --label-source=column).",
    )

    # ── Feature source selection ────────────────────────────────────────
    parser.add_argument(
        "--use-precomputed", action="store_true",
        help="Use pre-computed 'features' JSON column instead of re-computing.",
    )

    # ── Filtering ───────────────────────────────────────────────────────
    parser.add_argument(
        "--min-feature-max", type=float, default=None,
        help="Filter: only include samples where features.max >= this value.",
    )
    return parser.parse_args()


# ── Label extraction helpers ────────────────────────────────────────────

def _label_from_valve(row) -> int | None:
    """Derive label from leak_valve_status: 1 (valve open) → LEAK(0), 0 → OK(1)."""
    try:
        status = int(row["leak_valve_status"])
        return 0 if status == 1 else 1
    except (ValueError, TypeError, KeyError):
        return None


def _label_from_column(row, label_column: str, flip: bool) -> int | None:
    """Derive label from an explicit column, with optional flip."""
    try:
        raw = int(row[label_column])
        if raw < 0:
            return None
        return (1 - raw) if flip else raw
    except (ValueError, TypeError, KeyError):
        return None


# ── Feature extraction helpers ──────────────────────────────────────────

def _features_from_precomputed(row) -> list[float] | None:
    """Parse the pre-computed 'features' JSON column into a 7D vector."""
    try:
        feat_str = row["features"]
        if isinstance(feat_str, str):
            feats = json.loads(feat_str)
        elif isinstance(feat_str, dict):
            feats = feat_str
        else:
            return None
        return [float(feats.get(k, 0.0)) for k in FEATURE_ORDER_7D]
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None


def _features_from_pressure(row) -> list[float] | None:
    """Re-compute 7D features from pressure_data column."""
    from core.features import compute_features, features_to_vector
    try:
        pressures = json.loads(row["pressure_data"])
        if len(pressures) < 2 or all(p == 0.0 for p in pressures):
            return None
        cid = int(row.get("cavity_id", 0))
        feats = compute_features(pressures, cid)
        return features_to_vector(feats, mode="7d")
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None


# ── Data loader ─────────────────────────────────────────────────────────

def load_data(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load data, parse features/labels, and return arrays."""
    logger.info(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)

    features_list = []
    labels = []
    valid_indices = []

    for idx, row in df.iterrows():
        # ── Extract label ──
        if args.label_source == "valve":
            label = _label_from_valve(row)
        else:
            label = _label_from_column(row, args.label_column, args.flip_labels)
        if label is None:
            continue

        # ── Extract features ──
        if args.use_precomputed:
            vec = _features_from_precomputed(row)
        else:
            vec = _features_from_pressure(row)
        if vec is None:
            continue

        # ── Feature-level filter ──
        if args.min_feature_max is not None:
            if vec[0] < args.min_feature_max:
                continue

        features_list.append(vec)
        labels.append(label)
        valid_indices.append(idx)

    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)

    logger.info(f"Loaded {len(X)} samples.")
    logger.info(f"Class distribution: Label 0 (LEAK): {np.sum(y == 0)}, Label 1 (OK): {np.sum(y == 1)}")

    return X, y, df.iloc[valid_indices].reset_index(drop=True)


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Find the best threshold maximizing F1 score using vectorized precision-recall curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores = np.nan_to_num(f1_scores)

    valid_f1 = f1_scores[:-1]

    if len(valid_f1) == 0:
        return 0.5, 0.0

    best_idx = np.argmax(valid_f1)
    best_f1 = valid_f1[best_idx]
    best_th = thresholds[best_idx]

    return float(best_th), float(best_f1)


def train_and_evaluate(args: argparse.Namespace):
    X, y, df = load_data(args)

    if len(np.unique(y)) < 2:
        logger.error("Data must contain at least 2 classes.")
        sys.exit(1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Fold Stratified CV
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    # Store predictions for global optimization
    y_prob_cv = np.full(len(y), -1.0, dtype=np.float64)
    fold_metrics = []

    logger.info(f"Starting {args.n_folds}-fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42
        )
        clf.fit(X_train, y_train, verbose=False)

        probs = clf.predict_proba(X_val)[:, 1]
        y_prob_cv[val_idx] = probs

        preds = (probs >= 0.5).astype(int)
        acc = accuracy_score(y_val, preds)
        fold_metrics.append(acc)

    avg_acc = np.mean(fold_metrics)
    logger.info(f"Average CV Accuracy (th=0.5): {avg_acc:.4f}")

    # Optimize Threshold on aggregated predictions
    best_th, best_f1 = optimize_threshold(y, y_prob_cv)
    logger.info(f"Optimal Threshold: {best_th:.6f} (Max F1: {best_f1:.4f})")

    # Final Evaluation with Best Threshold
    y_pred_final = (y_prob_cv >= best_th).astype(int)

    cm = confusion_matrix(y, y_pred_final)
    acc = accuracy_score(y, y_pred_final)
    prec = precision_score(y, y_pred_final)
    rec = recall_score(y, y_pred_final)
    auc = roc_auc_score(y, y_prob_cv)

    logger.info("=== Final CV Results (Optimized Threshold) ===")
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"AUC-ROC:   {auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    # Acceptance check per spec
    leak_recall = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    ok_precision = cm[1][1] / (cm[0][1] + cm[1][1]) if (cm[0][1] + cm[1][1]) > 0 else 0
    logger.info("=== Acceptance Check ===")
    logger.info(f"  F1-Score >= 0.985:      {best_f1:.4f}  {'PASS' if best_f1 >= 0.985 else 'FAIL'}")
    logger.info(f"  Recall(LEAK) >= 0.990:  {leak_recall:.4f}  {'PASS' if leak_recall >= 0.990 else 'FAIL'}")
    logger.info(f"  Precision(OK) >= 0.980: {ok_precision:.4f}  {'PASS' if ok_precision >= 0.980 else 'FAIL'}")

    # Train Final Model on ALL Data
    logger.info("Training final model on full dataset...")
    final_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
    final_clf.fit(X_scaled, y)

    # Save Artifacts
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "xgb_model.json"
    final_clf.save_model(model_path)

    scaler_path = out_dir / "xgb_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    # Feature Importance
    importance = dict(zip(FEATURE_ORDER_7D, final_clf.feature_importances_.tolist()))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    # Save Metadata
    metadata = {
        "version": args.version,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": args.data,
        "dataset_size": int(X.shape[0]),
        "threshold": float(best_th),
        "label_source": args.label_source,
        "use_precomputed_features": args.use_precomputed,
        "metrics": {
            "accuracy": float(acc),
            "f1": float(best_f1),
            "precision": float(prec),
            "recall": float(rec),
            "auc": float(auc),
            "leak_recall": float(leak_recall),
            "ok_precision": float(ok_precision),
        },
        "feature_importance": importance,
        "n_folds": args.n_folds
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Evaluation report
    eval_report = (
        f"Model Version: {args.version}\n"
        f"Trained At: {metadata['trained_at']}\n"
        f"Dataset: {args.data} ({X.shape[0]} samples)\n"
        f"Label source: {args.label_source}\n"
        f"Features: {'pre-computed' if args.use_precomputed else 're-computed'}\n"
        f"Threshold: {best_th:.6f}\n"
        f"CV Folds: {args.n_folds}\n\n"
        f"=== Metrics ===\n"
        f"Accuracy:  {acc:.4f}\n"
        f"F1-Score:  {best_f1:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall:    {rec:.4f}\n"
        f"AUC-ROC:   {auc:.4f}\n\n"
        f"=== Acceptance Check ===\n"
        f"F1-Score >= 0.985:      {best_f1:.4f}  {'PASS' if best_f1 >= 0.985 else 'FAIL'}\n"
        f"Recall(LEAK) >= 0.990:  {leak_recall:.4f}  {'PASS' if leak_recall >= 0.990 else 'FAIL'}\n"
        f"Precision(OK) >= 0.980: {ok_precision:.4f}  {'PASS' if ok_precision >= 0.980 else 'FAIL'}\n\n"
        f"=== Confusion Matrix ===\n{cm}\n\n"
        f"=== Feature Importance ===\n"
    )
    for k, v in sorted_imp:
        eval_report += f"  {k:12s}: {v:.4f}\n"

    with open(out_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(eval_report)

    logger.info(f"Model and results saved to {out_dir}")
    print(f"\nBest Threshold: {best_th:.6f}")
    print("Feature Importance:")
    for k, v in sorted_imp:
        print(f"  {k:12s}: {v:.4f}")


if __name__ == "__main__":
    train_and_evaluate(parse_args())
