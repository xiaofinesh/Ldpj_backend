#!/usr/bin/env python3
"""Train XGBoost model with 10-fold CV and threshold optimization.

Usage:
    python -m train.train_cv_optimized --data cleaned_data.csv --output models/artifacts/optimized
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

FEATURE_ORDER_7D = ["max", "min", "difference", "average", "variance", "trend_slope", "cavity_id"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost with CV optimization")
    parser.add_argument("--data", required=True, help="Path to cleaned_data.csv")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--n-folds", type=int, default=10, help="Number of CV folds")
    return parser.parse_args()


def load_data(csv_path: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load data, parse features/labels, and return arrays."""
    logger.info(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    features_list = []
    labels = []
    valid_indices = []

    for idx, row in df.iterrows():
        # 1. Parse Label from leak_valve_statuses
        # If any valve status is True => LEAK (0)
        # If all False => OK (1)
        try:
            valve_statuses = str(row.get("leak_valve_statuses", "[]")).lower()
            if "true" in valve_statuses:
                label = 0  # LEAK
            else:
                label = 1  # OK
        except Exception:
            continue

        # 2. Parse Features
        try:
            feat_str = row.get("features", "{}")
            if isinstance(feat_str, str):
                feats = json.loads(feat_str)
            else:
                feats = feat_str  # Already dict?
            
            vec = [float(feats.get(k, 0.0)) for k in FEATURE_ORDER_7D]
        except (json.JSONDecodeError, TypeError, ValueError):
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
    
    # Calculate F1 for each threshold
    # Note: division by zero possible
    with np.errstate(divide='ignore', invalid='ignore'):
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores = np.nan_to_num(f1_scores)
        
    # thresholds array is shorter than precisions/recalls by 1
    # We ignore the last precision/recall value (which is for threshold=1.0 usually)
    valid_f1 = f1_scores[:-1]
    
    if len(valid_f1) == 0:
        return 0.5, 0.0
        
    best_idx = np.argmax(valid_f1)
    best_f1 = valid_f1[best_idx]
    best_th = thresholds[best_idx]
    
    return float(best_th), float(best_f1)


def train_and_evaluate(args: argparse.Namespace):
    X, y, df = load_data(args.data)
    
    if len(np.unique(y)) < 2:
        logger.error("Data must contain at least 2 classes.")
        sys.exit(1)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 10-Fold Stratified CV
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    # Store predictions for global optimization
    # Initialize with -1 to detect missing
    y_prob_cv = np.full(len(y), -1.0, dtype=np.float64)
    y_true_cv = y.copy() # Just to be explicit, though y is already aligned
    
    fold_metrics = []

    logger.info(f"Starting {args.n_folds}-fold Cross-Validation...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train XGBoost
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
        
        # Predict probability of class 1 (OK)
        probs = clf.predict_proba(X_val)[:, 1]
        y_prob_cv[val_idx] = probs
        
        # Fold metrics (using default 0.5 threshold for logging)
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
        "version": "optimized_v1",
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "threshold": float(best_th),
        "metrics": {
            "accuracy": float(acc),
            "f1": float(best_f1),
            "precision": float(prec),
            "recall": float(rec),
            "auc": float(auc),
        },
        "feature_importance": importance,
        "n_folds": args.n_folds
    }
    
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Model and results saved to {out_dir}")
    print(f"\nBest Threshold: {best_th:.6f}")
    print("Feature Importance:")
    for k, v in sorted_imp:
        print(f"  {k:12s}: {v:.4f}")


if __name__ == "__main__":
    train_and_evaluate(parse_args())
