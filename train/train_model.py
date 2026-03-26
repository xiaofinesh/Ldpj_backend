#!/usr/bin/env python3
"""Standard model training script for Ldpj_backend.

Reads a CSV dataset exported from the Ldpj_backend system, trains an XGBoost
classifier with 10-fold stratified cross-validation, optimizes the decision
threshold, then trains a final model on all data and saves artifacts.

Label convention (consistent with core/label_spec.py):
    label = 0: LEAK (漏液)
    label = 1: OK   (密封正常)

Labelling strategy (priority order):
    1. --label-source=valve  (DEFAULT for new system data)
       Uses `leak_valve_status` column:  1 → LEAK(0),  0 → OK(1)
    2. --label-source=column
       Uses the column specified by --label-column (default: 'label').
       Supports --flip-labels for old system CSVs where 0=OK, 1=LEAK.

Feature source (priority order):
    1. Pre-computed `features` JSON column (if --use-precomputed)
    2. Re-compute from `pressure_data` using core.features (default)

Usage
-----
    # New system data — labels from leak_valve_status, features pre-computed:
    python -m train.train_model --data export.csv --output models/artifacts/v1.0 \\
        --use-precomputed

    # New system data — labels from leak_valve_status, re-compute features:
    python -m train.train_model --data export.csv --output models/artifacts/v1.0

    # Old system data — labels from 'prediction' column, flipped:
    python -m train.train_model --data old_data.csv --output models/artifacts/v1.0 \\
        --label-source column --label-column prediction --flip-labels

    # Old system data with pressure filter:
    python -m train.train_model --data old_data.csv --output models/artifacts/v1.0 \\
        --label-source column --flip-labels --min-pressure -5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEATURE_ORDER_7D = ["max", "min", "difference", "average", "variance", "trend_slope", "cavity_id"]


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost model for leak detection")
    p.add_argument("--data", required=True, help="Path to CSV dataset")
    p.add_argument("--output", required=True, help="Output directory for model artifacts")
    p.add_argument("--version", default="v1.0", help="Model version string")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-folds", type=int, default=10, help="Number of CV folds (default: 10)")
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.1)

    # ── Label source selection ──────────────────────────────────────────
    p.add_argument(
        "--label-source", choices=["valve", "column"], default="valve",
        help="How to derive labels. "
             "'valve' (default): use leak_valve_status column (1=LEAK, 0=OK). "
             "'column': use --label-column with optional --flip-labels.",
    )
    p.add_argument(
        "--flip-labels", action="store_true",
        help="Flip labels from old system convention (0=OK,1=LEAK) to new (0=LEAK,1=OK). "
             "Only used when --label-source=column.",
    )
    p.add_argument(
        "--label-column", default="label",
        help="Name of the label column in CSV (default: 'label'; use 'prediction' for old CSVs). "
             "Only used when --label-source=column.",
    )

    # ── Feature source selection ────────────────────────────────────────
    p.add_argument(
        "--use-precomputed", action="store_true",
        help="Use pre-computed 'features' JSON column instead of re-computing from pressure_data. "
             "Faster and ensures exact feature parity with inference pipeline.",
    )

    # ── Filtering ───────────────────────────────────────────────────────
    p.add_argument(
        "--min-pressure", type=float, default=None,
        help="Filter: only include samples where min(pressure) < this value. "
             "Useful for excluding all-zero rows from old data. E.g. --min-pressure -5",
    )
    p.add_argument(
        "--min-feature-max", type=float, default=None,
        help="Filter: only include samples where features.max >= this value. "
             "Useful for excluding no-bottle / zero-pressure rows. E.g. --min-feature-max 500",
    )
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  Label helpers
# ═══════════════════════════════════════════════════════════════════════

def _label_from_valve(row) -> int | None:
    """leak_valve_status: 1 (valve open) → LEAK(0), 0 → OK(1)."""
    try:
        status = int(row["leak_valve_status"])
        return 0 if status == 1 else 1
    except (ValueError, TypeError, KeyError):
        return None


def _label_from_column(row, label_column: str, flip: bool) -> int | None:
    """Explicit column label, with optional flip for old system."""
    try:
        raw = int(row[label_column])
        if raw < 0:  # -1 / -2 means no inference
            return None
        return (1 - raw) if flip else raw
    except (ValueError, TypeError, KeyError):
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Feature helpers
# ═══════════════════════════════════════════════════════════════════════

def _features_from_precomputed(row) -> list[float] | None:
    """Parse pre-computed 'features' JSON → 7D vector."""
    try:
        feat_str = row["features"]
        feats = json.loads(feat_str) if isinstance(feat_str, str) else feat_str
        return [float(feats.get(k, 0.0)) for k in FEATURE_ORDER_7D]
    except (json.JSONDecodeError, TypeError, ValueError, KeyError, AttributeError):
        return None


def _features_from_pressure(row) -> list[float] | None:
    """Re-compute 7D features from pressure_data."""
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


# ═══════════════════════════════════════════════════════════════════════
#  Data loader
# ═══════════════════════════════════════════════════════════════════════

def load_and_prepare(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    """Load CSV and build (X, y) arrays.

    Returns
    -------
    tuple of (X: np.ndarray shape (n,7), y: np.ndarray shape (n,))
    """
    df = pd.read_csv(args.data)

    # Validate required columns
    if args.label_source == "valve" and "leak_valve_status" not in df.columns:
        raise ValueError("CSV must contain 'leak_valve_status' column for --label-source=valve")
    if args.label_source == "column" and args.label_column not in df.columns:
        raise ValueError(f"CSV must contain '{args.label_column}' column for --label-source=column")
    if args.use_precomputed and "features" not in df.columns:
        raise ValueError("CSV must contain 'features' column for --use-precomputed")
    if not args.use_precomputed and "pressure_data" not in df.columns:
        raise ValueError("CSV must contain 'pressure_data' column when not using --use-precomputed")

    features_list: list[list[float]] = []
    labels: list[int] = []
    skipped_label = 0
    skipped_feature = 0
    skipped_filter = 0

    for _, row in df.iterrows():
        # ── Step 1: Extract label ──
        if args.label_source == "valve":
            label = _label_from_valve(row)
        else:
            label = _label_from_column(row, args.label_column, args.flip_labels)
        if label is None:
            skipped_label += 1
            continue

        # ── Step 2: Extract features ──
        if args.use_precomputed:
            vec = _features_from_precomputed(row)
        else:
            # Apply pressure filter before computing features
            if args.min_pressure is not None:
                try:
                    pressures = json.loads(row["pressure_data"])
                    if min(pressures) > args.min_pressure:
                        skipped_filter += 1
                        continue
                except (json.JSONDecodeError, TypeError):
                    pass
            vec = _features_from_pressure(row)
        if vec is None:
            skipped_feature += 1
            continue

        # ── Step 3: Feature-level filter ──
        if args.min_feature_max is not None:
            if vec[0] < args.min_feature_max:  # vec[0] = max
                skipped_filter += 1
                continue

        features_list.append(vec)
        labels.append(label)

    if skipped_label > 0:
        logger.info(f"  Skipped {skipped_label} rows (invalid/missing label)")
    if skipped_feature > 0:
        logger.info(f"  Skipped {skipped_feature} rows (invalid/zero features)")
    if skipped_filter > 0:
        logger.info(f"  Skipped {skipped_filter} rows (filtered by pressure/feature threshold)")

    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    return X, y


# ═══════════════════════════════════════════════════════════════════════
#  Threshold optimiser
# ═══════════════════════════════════════════════════════════════════════

def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """Find the best threshold maximising F1-score via precision-recall curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores = np.nan_to_num(f1_scores)

    valid_f1 = f1_scores[:-1]
    if len(valid_f1) == 0:
        return 0.5, 0.0

    best_idx = np.argmax(valid_f1)
    return float(thresholds[best_idx]), float(valid_f1[best_idx])


# ═══════════════════════════════════════════════════════════════════════
#  Main training pipeline
# ═══════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    import xgboost as xgb

    # ── 1. Load data ──────────────────────────────────────────────────
    logger.info(f"Loading data from {args.data} ...")
    logger.info(f"  Label source: {args.label_source}")
    if args.label_source == "valve":
        logger.info("  Using leak_valve_status: 1→LEAK(0), 0→OK(1)")
    else:
        logger.info(f"  Label column: '{args.label_column}', flip={args.flip_labels}")
    logger.info(f"  Feature source: {'pre-computed JSON' if args.use_precomputed else 're-compute from pressure_data'}")

    X, y = load_and_prepare(args)

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    class_dist = dict(zip(*np.unique(y, return_counts=True)))
    logger.info(f"  label=0 (LEAK): {class_dist.get(0, 0)}")
    logger.info(f"  label=1 (OK):   {class_dist.get(1, 0)}")

    if len(class_dist) < 2:
        logger.error("Need at least 2 classes for training. Check your data and labels.")
        sys.exit(1)

    # ── 2. StandardScaler ─────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 3. K-Fold Stratified Cross-Validation ─────────────────────────
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.random_state)
    y_prob_cv = np.full(len(y), -1.0, dtype=np.float64)
    fold_metrics: list[float] = []

    logger.info(f"Starting {args.n_folds}-fold Stratified Cross-Validation ...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = xgb.XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            learning_rate=args.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=args.random_state,
        )
        clf.fit(X_train, y_train, verbose=False)

        probs = clf.predict_proba(X_val)[:, 1]
        y_prob_cv[val_idx] = probs

        preds = (probs >= 0.5).astype(int)
        fold_f1 = f1_score(y_val, preds, average="binary")
        fold_metrics.append(fold_f1)
        logger.info(f"  Fold {fold + 1}/{args.n_folds}: F1={fold_f1:.4f}")

    avg_f1 = np.mean(fold_metrics)
    std_f1 = np.std(fold_metrics)
    logger.info(f"{args.n_folds}-Fold CV F1: {avg_f1:.4f} (+/- {std_f1:.4f})")

    # ── 4. Optimize threshold ─────────────────────────────────────────
    best_th, best_f1 = optimize_threshold(y, y_prob_cv)
    logger.info(f"Optimal threshold: {best_th:.6f} (F1={best_f1:.4f})")

    # ── 5. Evaluate with optimised threshold ──────────────────────────
    y_pred_final = (y_prob_cv >= best_th).astype(int)

    acc = accuracy_score(y, y_pred_final)
    prec = precision_score(y, y_pred_final, average="binary")
    rec = recall_score(y, y_pred_final, average="binary")
    auc = roc_auc_score(y, y_prob_cv)
    cm = confusion_matrix(y, y_pred_final)
    report_text = classification_report(y, y_pred_final, target_names=["LEAK(0)", "OK(1)"])

    leak_recall = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    ok_precision = cm[1][1] / (cm[0][1] + cm[1][1]) if (cm[0][1] + cm[1][1]) > 0 else 0

    print("\n" + "=" * 55)
    print("  Evaluation Results (optimised threshold)")
    print("=" * 55)
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Score:  {best_f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"Threshold: {best_th:.6f}")
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"             pred=LEAK  pred=OK")
    print(f"  act=LEAK   {cm[0][0]:>8d}  {cm[0][1]:>7d}")
    print(f"  act=OK     {cm[1][0]:>8d}  {cm[1][1]:>7d}")
    print(f"\n{report_text}")

    # Acceptance check per spec: F1>=0.985, Recall(LEAK)>=0.990, Precision(OK)>=0.980
    print("=== Acceptance Check ===")
    print(f"  F1-Score >= 0.985:      {best_f1:.4f}  {'PASS' if best_f1 >= 0.985 else 'FAIL'}")
    print(f"  Recall(LEAK) >= 0.990:  {leak_recall:.4f}  {'PASS' if leak_recall >= 0.990 else 'FAIL'}")
    print(f"  Precision(OK) >= 0.980: {ok_precision:.4f}  {'PASS' if ok_precision >= 0.980 else 'FAIL'}")
    all_pass = best_f1 >= 0.985 and leak_recall >= 0.990 and ok_precision >= 0.980
    print(f"\n  Overall: {'ALL PASSED - Model is deployable!' if all_pass else 'NOT ALL PASSED'}")

    # ── 6. Train final model on ALL data ──────────────────────────────
    logger.info("Training final model on full dataset ...")
    final_clf = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=args.random_state,
    )
    final_clf.fit(X_scaled, y)

    # Feature importance
    importance = dict(zip(FEATURE_ORDER_7D, final_clf.feature_importances_.tolist()))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nFeature Importance:")
    for name, imp in sorted_imp:
        print(f"  {name:20s}: {imp:.4f}")

    # ── 7. Save artifacts ─────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "xgb_model.json"
    final_clf.get_booster().save_model(str(model_path))

    scaler_path = out_dir / "xgb_scaler.joblib"
    joblib.dump(scaler, str(scaler_path))

    metadata = {
        "version": args.version,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": args.data,
        "dataset_size": int(X.shape[0]),
        "feature_mode": "7d",
        "feature_order": FEATURE_ORDER_7D,
        "label_convention": {
            "0": "LEAK",
            "1": "OK",
            "description": "0=leak detected, 1=good seal / no leak",
        },
        "label_source": args.label_source,
        "label_flipped": args.flip_labels if args.label_source == "column" else False,
        "label_column": args.label_column if args.label_source == "column" else "leak_valve_status",
        "use_precomputed_features": args.use_precomputed,
        "threshold": round(best_th, 6),
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "n_folds": args.n_folds,
        },
        "evaluation": {
            "accuracy": round(acc, 6),
            "f1_score": round(best_f1, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "auc_roc": round(auc, 6),
            "leak_recall": round(leak_recall, 6),
            "ok_precision": round(ok_precision, 6),
            "cv_f1_mean": round(float(avg_f1), 6),
            "cv_f1_std": round(float(std_f1), 6),
        },
        "feature_importance": {k: round(v, 6) for k, v in sorted_imp},
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    eval_report = (
        f"Model Version: {args.version}\n"
        f"Trained At: {metadata['trained_at']}\n"
        f"Dataset: {args.data} ({X.shape[0]} samples)\n"
        f"Label source: {args.label_source}\n"
        f"Labels: 0=LEAK, 1=OK\n"
        f"Features: {'pre-computed' if args.use_precomputed else 're-computed from pressure_data'}\n"
        f"Threshold: {best_th:.6f}\n"
        f"CV Folds: {args.n_folds}\n\n"
        f"=== Metrics ===\n"
        f"Accuracy:  {acc:.4f}\n"
        f"F1-Score:  {best_f1:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall:    {rec:.4f}\n"
        f"AUC-ROC:   {auc:.4f}\n"
        f"{args.n_folds}-Fold CV F1: {avg_f1:.4f} (+/- {std_f1:.4f})\n\n"
        f"=== Acceptance Check ===\n"
        f"F1-Score >= 0.985:      {best_f1:.4f}  {'PASS' if best_f1 >= 0.985 else 'FAIL'}\n"
        f"Recall(LEAK) >= 0.990:  {leak_recall:.4f}  {'PASS' if leak_recall >= 0.990 else 'FAIL'}\n"
        f"Precision(OK) >= 0.980: {ok_precision:.4f}  {'PASS' if ok_precision >= 0.980 else 'FAIL'}\n\n"
        f"=== Confusion Matrix ===\n"
        f"             pred=LEAK  pred=OK\n"
        f"  act=LEAK   {cm[0][0]:>8d}  {cm[0][1]:>7d}\n"
        f"  act=OK     {cm[1][0]:>8d}  {cm[1][1]:>7d}\n\n"
        f"=== Classification Report ===\n{report_text}\n\n"
        f"=== Feature Importance ===\n"
    )
    for name, imp in sorted_imp:
        eval_report += f"  {name:20s}: {imp:.4f}\n"

    with open(out_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(eval_report)

    print(f"\nArtifacts saved to: {out_dir}")
    print(f"  - {model_path.name}")
    print(f"  - {scaler_path.name}")
    print(f"  - metadata.json")
    print(f"  - evaluation_report.txt")


if __name__ == "__main__":
    train(parse_args())
