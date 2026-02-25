#!/usr/bin/env python3
"""Standard model training script for Ldpj_backend.

Reads a labelled CSV dataset, computes features, trains an XGBoost classifier
with StandardScaler, evaluates on a held-out test set, and saves the model
artifacts to a versioned directory.

Usage
-----
    python -m train.train_model \\
        --data  train_data.csv \\
        --output models/artifacts/v1.0_20260225 \\
        --version v1.0
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
    return p.parse_args()


def load_and_prepare(csv_path: str) -> tuple:
    """Load CSV and compute feature matrix.

    Expected CSV columns: pressure_data (JSON array), cavity_id, label.
    """
    df = pd.read_csv(csv_path)
    required = {"pressure_data", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}")

    features_list = []
    labels = []
    for _, row in df.iterrows():
        pressures = json.loads(row["pressure_data"])
        cid = int(row.get("cavity_id", 0))
        feats = compute_features(pressures, cid)
        vec = features_to_vector(feats, mode="7d")
        features_list.append(vec)
        labels.append(int(row["label"]))

    X = np.array(features_list, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)
    return X, y


def train(args: argparse.Namespace) -> None:
    import xgboost as xgb

    print(f"Loading data from {args.data} ...")
    X, y = load_and_prepare(args.data)
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

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

    report_text = classification_report(y_test, y_pred, target_names=["Leak(0)", "OK(1)"])
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"\n{report_text}")

    # Feature importance
    importance = dict(zip(FEATURE_ORDER_7D, clf.feature_importances_.tolist()))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nFeature Importance:")
    for name, imp in sorted_imp:
        print(f"  {name:20s}: {imp:.4f}")

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    cv_scores = cross_val_score(clf, scaler.transform(X), y, cv=skf, scoring="f1")
    print(f"\n5-Fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Save artifacts
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Model (JSON format for Booster)
    model_path = out_dir / "xgb_model.json"
    clf.get_booster().save_model(str(model_path))

    # 2. Scaler
    scaler_path = out_dir / "xgb_scaler.joblib"
    joblib.dump(scaler, str(scaler_path))

    # 3. Metadata
    metadata = {
        "version": args.version,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": args.data,
        "dataset_size": int(X.shape[0]),
        "feature_mode": "7d",
        "feature_order": FEATURE_ORDER_7D,
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

    # 4. Evaluation report
    eval_report = (
        f"Model Version: {args.version}\n"
        f"Trained At: {metadata['trained_at']}\n"
        f"Dataset: {args.data} ({X.shape[0]} samples)\n\n"
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
