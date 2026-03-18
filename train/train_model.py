#!/usr/bin/env python3
"""Standard model training script for Ldpj_backend.

Reads a labelled CSV dataset, computes features, trains an XGBoost classifier
with StandardScaler, evaluates on a held-out test set, and saves the model
artifacts to a versioned directory.

Label convention (consistent with core/label_spec.py):
    label = 0: LEAK (漏液)
    label = 1: OK   (密封正常)

IMPORTANT: Old system CSVs use INVERTED labels:
    old prediction=0 → good seal → new label=1 (OK)
    old prediction=1 → leak      → new label=0 (LEAK)
    Use --flip-labels when training on old data.

Usage
-----
    # Train on new system data (labels already correct):
    python -m train.train_model --data new_data.csv --output models/artifacts/v1.0

    # Train on OLD system exported CSV (labels inverted):
    python -m train.train_model --data old_data.csv --output models/artifacts/v1.0 --flip-labels

    # Use 'prediction' column from old CSV as label source:
    python -m train.train_model --data old_data.csv --output models/artifacts/v1.0 \\
        --flip-labels --label-column prediction
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.features import compute_features, features_to_vector

FEATURE_ORDER_7D = ["max", "min", "difference", "average", "variance", "trend_slope", "cavity_id"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train XGBoost model for leak detection")
    p.add_argument("--data", required=True, help="Path to labelled CSV dataset")
    p.add_argument("--output", required=True, help="Output directory for model artifacts")
    p.add_argument("--version", default="v1.0", help="Model version string")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument(
        "--flip-labels", action="store_true",
        help="Flip labels from old system convention (0=OK,1=LEAK) to new (0=LEAK,1=OK)",
    )
    p.add_argument(
        "--label-column", default="label",
        help="Name of the label column in CSV (default: 'label'; use 'prediction' for old CSVs)",
    )
    p.add_argument(
        "--min-pressure", type=float, default=None,
        help="Filter: only include samples where min(pressure) < this value. "
             "Useful for excluding all-zero rows from old data. E.g. --min-pressure -5",
    )
    return p.parse_args()


def load_and_prepare(csv_path: str, label_column: str = "label",
                     flip_labels: bool = False,
                     min_pressure_filter: float = None) -> tuple:
    """Load CSV and compute feature matrix.

    Parameters
    ----------
    csv_path : str
        Path to CSV file.
    label_column : str
        Column name for the label. 'label' for new data, 'prediction' for old.
    flip_labels : bool
        If True, flip labels: new_label = 1 - old_label.
        Required when using old system data where 0=OK, 1=LEAK.
    min_pressure_filter : float or None
        If set, exclude rows where min(pressure_data) > this value.
        Useful for filtering out all-zero pressure rows.

    Returns
    -------
    tuple of (X, y)
    """
    df = pd.read_csv(csv_path)

    if "pressure_data" not in df.columns:
        raise ValueError("CSV must contain 'pressure_data' column")
    if label_column not in df.columns:
        raise ValueError(f"CSV must contain '{label_column}' column")

    features_list = []
    labels = []
    skipped_zero = 0
    skipped_invalid = 0

    for _, row in df.iterrows():
        try:
            pressures = json.loads(row["pressure_data"])
        except (json.JSONDecodeError, TypeError):
            skipped_invalid += 1
            continue

        # Filter out all-zero or near-zero pressure samples
        if min_pressure_filter is not None:
            if min(pressures) > min_pressure_filter:
                skipped_zero += 1
                continue

        # Skip samples with too few valid points
        if len(pressures) < 2 or all(p == 0.0 for p in pressures):
            skipped_zero += 1
            continue

        cid = int(row.get("cavity_id", 0))
        feats = compute_features(pressures, cid)
        vec = features_to_vector(feats, mode="7d")

        raw_label = int(row[label_column])
        if flip_labels:
            # Old convention: 0=OK, 1=LEAK → New convention: 0=LEAK, 1=OK
            label = 1 - raw_label
        else:
            label = raw_label

        features_list.append(vec)
        labels.append(label)

    if skipped_zero > 0:
        print(f"  Skipped {skipped_zero} rows (zero/insufficient pressure data)")
    if skipped_invalid > 0:
        print(f"  Skipped {skipped_invalid} rows (invalid pressure_data JSON)")

    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    return X, y


def train(args: argparse.Namespace) -> None:
    import xgboost as xgb

    print(f"Loading data from {args.data} ...")
    if args.flip_labels:
        print(f"  *** Label flipping enabled (old→new convention) ***")
    print(f"  Label column: '{args.label_column}'")

    X, y = load_and_prepare(
        args.data,
        label_column=args.label_column,
        flip_labels=args.flip_labels,
        min_pressure_filter=args.min_pressure,
    )
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    class_dist = dict(zip(*np.unique(y, return_counts=True)))
    print(f"Class distribution: {class_dist}")
    print(f"  label=0 (LEAK): {class_dist.get(0, 0)}")
    print(f"  label=1 (OK):   {class_dist.get(1, 0)}")

    if len(class_dist) < 2:
        print("\nERROR: Need at least 2 classes for training. Check your data and labels.")
        sys.exit(1)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # XGBoost
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
    clf.fit(X_train_s, y_train, eval_set=[(X_test_s, y_test)], verbose=False)

    # Evaluate
    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")
    prec = precision_score(y_test, y_pred, average="binary")
    rec = recall_score(y_test, y_pred, average="binary")
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0

    # NOTE: target_names aligned with label convention: 0=LEAK, 1=OK
    report_text = classification_report(y_test, y_pred, target_names=["LEAK(0)", "OK(1)"])
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"             pred=LEAK  pred=OK")
    print(f"  act=LEAK   {cm[0][0]:>8d}  {cm[0][1]:>7d}")
    print(f"  act=OK     {cm[1][0]:>8d}  {cm[1][1]:>7d}")
    print(f"\n{report_text}")

    # Feature importance
    importance = dict(zip(FEATURE_ORDER_7D, clf.feature_importances_.tolist()))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("Feature Importance:")
    for name, imp in sorted_imp:
        print(f"  {name:20s}: {imp:.4f}")

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    cv_scores = cross_val_score(clf, scaler.transform(X), y, cv=skf, scoring="f1")
    print(f"\n5-Fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Save artifacts
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "xgb_model.json"
    clf.get_booster().save_model(str(model_path))

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
        "label_flipped": args.flip_labels,
        "label_column": args.label_column,
        "hyperparameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
        },
        "evaluation": {
            "accuracy": round(acc, 6),
            "f1_score": round(f1, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "auc_roc": round(auc, 6),
            "cv_f1_mean": round(float(cv_scores.mean()), 6),
            "cv_f1_std": round(float(cv_scores.std()), 6),
        },
        "feature_importance": {k: round(v, 6) for k, v in sorted_imp},
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    eval_report = (
        f"Model Version: {args.version}\n"
        f"Trained At: {metadata['trained_at']}\n"
        f"Dataset: {args.data} ({X.shape[0]} samples)\n"
        f"Labels: 0=LEAK, 1=OK (flipped={args.flip_labels})\n\n"
        f"=== Metrics ===\n"
        f"Accuracy:  {acc:.4f}\n"
        f"F1-Score:  {f1:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall:    {rec:.4f}\n"
        f"AUC-ROC:   {auc:.4f}\n"
        f"5-Fold CV: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n\n"
        f"=== Confusion Matrix ===\n{cm}\n\n"
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
