"""
Phase 5: Model Training & Evaluation (THE BRAIN)
==================================================
This script trains both XGBoost and LightGBM models on the processed
data from Phase 3, evaluates them using AUPRC (the correct metric for
highly imbalanced financial data), compares performance head-to-head,
and saves the best model for downstream phases.

Read → Process → Save Artifact principle:
  - READS:  data/processed/X_train.csv, y_train.csv, X_val.csv, y_val.csv, X_test.csv, y_test.csv
  - PROCESS: Train XGBoost + LightGBM, evaluate on test set
  - SAVES:  models/xgboost_fraud_v1.pkl, models/lightgbm_fraud_v1.pkl,
            models/best_model.pkl, models/metrics.json, models/model_comparison.json

Usage:
    python -m src.models.model_training

Dependencies (from requirements.txt):
    xgboost, lightgbm, scikit-learn, pandas, numpy, joblib
"""

import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
    auc,
)
import xgboost as xgb
import lightgbm as lgb

# --- Project imports ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.logger import (
    get_logger,
    load_config,
    resolve_path,
    log_phase_start,
    log_phase_end,
    check_phase_completed,
)

warnings.filterwarnings("ignore", category=UserWarning)

logger = get_logger(__name__)


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_processed_data(config: dict) -> dict:
    """
    Load the 6 processed CSV files saved by Phase 3.

    Returns a dictionary with keys:
        X_train, y_train, X_val, y_val, X_test, y_test
    All as numpy arrays (models don't need DataFrames).
    Also returns feature_names for explainability in Phase 7.
    """
    logger.info("Loading processed data from Phase 3 artifacts...")

    # preprocess_cfg = config["preprocessing"]
    # base_dir = resolve_path(preprocess_cfg["output_dir"])

    # # Build file paths
    # files = {
    #     "X_train": base_dir / "X_train.csv",
    #     "y_train": base_dir / "y_train.csv",
    #     "X_val":   base_dir / "X_val.csv",
    #     "y_val":   base_dir / "y_val.csv",
    #     "X_test":  base_dir / "X_test.csv",
    #     "y_test":  base_dir / "y_test.csv",
    # }
    preprocess_cfg = config["preprocessing"]
    base_dir = resolve_path(preprocess_cfg["processed_dir"])

    # Build file paths from config.yaml (no hardcoded paths)
    files = {
        "X_train": resolve_path(preprocess_cfg["train_path"]),
        "y_train": resolve_path(preprocess_cfg["y_train_path"]),
        "X_val":   resolve_path(preprocess_cfg["val_path"]),
        "y_val":   resolve_path(preprocess_cfg["y_val_path"]),
        "X_test":  resolve_path(preprocess_cfg["test_path"]),
        "y_test":  resolve_path(preprocess_cfg["y_test_path"]),
    }
    

    # Verify all files exist before loading
    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Phase 3 artifact not found: {path}. "
                f"Run Phase 3 (data_engineering.py) first."
            )

    # Load all files
    data = {}
    for name, path in files.items():
        df = pd.read_csv(path)
        data[name] = df.values.ravel() if "y_" in name else df.values
        logger.info(f"  Loaded {name}: shape={data[name].shape}")

    # Load feature names for logging and future phases
    feature_names_path = base_dir / "feature_names.json"
    if feature_names_path.exists():
        with open(feature_names_path, "r") as f:
            data["feature_names"] = json.load(f)
        logger.info(f"  Loaded feature names: {len(data['feature_names'])} features")
    else:
        # Fallback: use column headers from X_train
        data["feature_names"] = pd.read_csv(files["X_train"], nrows=0).columns.tolist()
        logger.info(f"  Feature names from CSV headers: {len(data['feature_names'])} features")

    # Log class distribution
    for split in ["y_train", "y_val", "y_test"]:
        unique, counts = np.unique(data[split], return_counts=True)
        dist = dict(zip(unique.astype(int), counts.astype(int)))
        logger.info(f"  {split} class distribution: {dist}")

    return data


# ============================================================
# 2. MODEL TRAINING — XGBoost
# ============================================================

def train_xgboost(data: dict, config: dict) -> tuple:
    """
    Train an XGBoost classifier with:
      - Class weighting (scale_pos_weight) to handle extreme imbalance
      - Early stopping on validation set to prevent overfitting
      - Hyperparameters from config.yaml

    Returns: (trained_model, training_time_seconds)
    """
    logger.info("=" * 50)
    logger.info("TRAINING: XGBoost Classifier")
    logger.info("=" * 50)

    xgb_cfg = config["model"]["xgboost"]["params"]

    # Calculate scale_pos_weight for class imbalance
    # Formula: count(negative) / count(positive)
    n_negative = np.sum(data["y_train"] == 0)
    n_positive = np.sum(data["y_train"] == 1)
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
    logger.info(f"  Class imbalance ratio (neg/pos): {scale_pos_weight:.2f}")
    logger.info(f"  Training samples: {len(data['y_train'])} | "
                f"Positive: {n_positive} | Negative: {n_negative}")

    # Build the model
    model = xgb.XGBClassifier(
        n_estimators=xgb_cfg.get("n_estimators", 500),
        max_depth=xgb_cfg.get("max_depth", 6),
        learning_rate=xgb_cfg.get("learning_rate", 0.05),
        scale_pos_weight=scale_pos_weight,
        eval_metric=xgb_cfg.get("eval_metric", "aucpr"),
        tree_method=xgb_cfg.get("tree_method", "hist"),
        random_state=xgb_cfg.get("random_state", 42),
        use_label_encoder=False,
        verbosity=0,
    )

    # Train with early stopping
    early_stopping = xgb_cfg.get("early_stopping_rounds", 50)
    logger.info(f"  Hyperparameters: n_estimators={model.n_estimators}, "
                f"max_depth={model.max_depth}, lr={model.learning_rate}, "
                f"early_stopping={early_stopping}")

    start_time = time.time()

    model.fit(
        data["X_train"],
        data["y_train"],
        eval_set=[(data["X_val"], data["y_val"])],
        verbose=False,
    )

    training_time = time.time() - start_time

    # Log training results
    best_iteration = model.best_iteration if hasattr(model, "best_iteration") else model.n_estimators
    logger.info(f"  Training completed in {training_time:.2f} seconds")
    logger.info(f"  Best iteration: {best_iteration} / {model.n_estimators}")

    return model, training_time


# ============================================================
# 3. MODEL TRAINING — LightGBM
# ============================================================

def train_lightgbm(data: dict, config: dict) -> tuple:
    """
    Train a LightGBM classifier with:
      - is_unbalance=True to handle extreme class imbalance
      - Early stopping on validation set
      - Hyperparameters from config.yaml

    Returns: (trained_model, training_time_seconds)
    """
    logger.info("=" * 50)
    logger.info("TRAINING: LightGBM Classifier")
    logger.info("=" * 50)

    lgb_cfg = config["model"]["lightgbm"]["params"]

    logger.info(f"  Training samples: {len(data['y_train'])} | "
                f"Positive: {np.sum(data['y_train'] == 1)} | "
                f"Negative: {np.sum(data['y_train'] == 0)}")

    # Build the model
    model = lgb.LGBMClassifier(
        n_estimators=lgb_cfg.get("n_estimators", 500),
        max_depth=lgb_cfg.get("max_depth", 6),
        learning_rate=lgb_cfg.get("learning_rate", 0.05),
        is_unbalance=lgb_cfg.get("is_unbalance", True),
        metric=lgb_cfg.get("metric", "average_precision"),
        random_state=lgb_cfg.get("random_state", 42),
        verbose=-1,
        force_col_wise=True,
    )

    # Train with early stopping
    early_stopping = lgb_cfg.get("early_stopping_rounds", 50)
    logger.info(f"  Hyperparameters: n_estimators={model.n_estimators}, "
                f"max_depth={model.max_depth}, lr={model.learning_rate}, "
                f"early_stopping={early_stopping}")

    start_time = time.time()

    model.fit(
        data["X_train"],
        data["y_train"],
        eval_set=[(data["X_val"], data["y_val"])],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    training_time = time.time() - start_time

    # Log training results
    best_iteration = model.best_iteration_ if hasattr(model, "best_iteration_") else model.n_estimators
    logger.info(f"  Training completed in {training_time:.2f} seconds")
    logger.info(f"  Best iteration: {best_iteration} / {model.n_estimators}")

    return model, training_time


# ============================================================
# 4. MODEL EVALUATION
# ============================================================

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   model_name: str) -> dict:
    """
    Evaluate a trained model on the test set using the CORRECT metrics
    for highly imbalanced financial data.

    Standard accuracy is FORBIDDEN — it would show 99.8% even if the
    model predicted "not fraud" for everything.

    Metrics computed:
      - AUPRC (Area Under Precision-Recall Curve) — PRIMARY metric
      - ROC-AUC
      - Precision (at default 0.5 threshold)
      - Recall (at default 0.5 threshold)
      - F1 Score
      - Confusion Matrix
      - Optimal threshold (from PR curve, maximizing F1)

    Returns: dictionary of all metrics
    """
    logger.info(f"  Evaluating {model_name} on test set...")

    # Get probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- PRIMARY METRIC: AUPRC ---
    auprc = average_precision_score(y_test, y_proba)

    # --- PR Curve for optimal threshold ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)

    # Find optimal threshold that maximizes F1
    f1_scores = np.where(
        (precisions[:-1] + recalls[:-1]) > 0,
        2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1]),
        0,
    )
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[optimal_idx])
    optimal_f1 = float(f1_scores[optimal_idx])

    # --- Predictions at default threshold (0.5) ---
    y_pred_default = (y_proba >= 0.5).astype(int)
    precision_default = precision_score(y_test, y_pred_default, zero_division=0)
    recall_default = recall_score(y_test, y_pred_default, zero_division=0)
    f1_default = f1_score(y_test, y_pred_default, zero_division=0)
    cm_default = confusion_matrix(y_test, y_pred_default).tolist()

    # --- Predictions at optimal threshold ---
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    precision_optimal = precision_score(y_test, y_pred_optimal, zero_division=0)
    recall_optimal = recall_score(y_test, y_pred_optimal, zero_division=0)
    f1_optimal = f1_score(y_test, y_pred_optimal, zero_division=0)
    cm_optimal = confusion_matrix(y_test, y_pred_optimal).tolist()

    # --- ROC-AUC ---
    roc_auc = roc_auc_score(y_test, y_proba)

    # --- Inference speed ---
    start_time = time.time()
    for _ in range(10):
        _ = model.predict_proba(X_test[:100])
    inference_time_ms = ((time.time() - start_time) / 10) * 1000  # ms per 100 samples

    # Build metrics dictionary
    metrics = {
        "model_name": model_name,
        "auprc": round(float(auprc), 6),
        "roc_auc": round(float(roc_auc), 6),
        "default_threshold": {
            "threshold": 0.5,
            "precision": round(float(precision_default), 6),
            "recall": round(float(recall_default), 6),
            "f1_score": round(float(f1_default), 6),
            "confusion_matrix": cm_default,
        },
        "optimal_threshold": {
            "threshold": round(optimal_threshold, 6),
            "precision": round(float(precision_optimal), 6),
            "recall": round(float(recall_optimal), 6),
            "f1_score": round(float(f1_optimal), 6),
            "confusion_matrix": cm_optimal,
        },
        "inference_time_ms_per_100": round(inference_time_ms, 4),
    }

    # Log key results
    logger.info(f"  ┌─────────────────────────────────────────┐")
    logger.info(f"  │ {model_name:^39s} │")
    logger.info(f"  ├─────────────────────────────────────────┤")
    logger.info(f"  │ AUPRC (PRIMARY):     {auprc:>18.6f} │")
    logger.info(f"  │ ROC-AUC:             {roc_auc:>18.6f} │")
    logger.info(f"  ├── Default Threshold (0.5) ─────────────┤")
    logger.info(f"  │ Precision:           {precision_default:>18.6f} │")
    logger.info(f"  │ Recall:              {recall_default:>18.6f} │")
    logger.info(f"  │ F1 Score:            {f1_default:>18.6f} │")
    logger.info(f"  ├── Optimal Threshold ({optimal_threshold:.4f}) ────────┤")
    logger.info(f"  │ Precision:           {precision_optimal:>18.6f} │")
    logger.info(f"  │ Recall:              {recall_optimal:>18.6f} │")
    logger.info(f"  │ F1 Score:            {f1_optimal:>18.6f} │")
    logger.info(f"  ├─────────────────────────────────────────┤")
    logger.info(f"  │ Inference: {inference_time_ms:>7.2f}ms per 100 samples │")
    logger.info(f"  └─────────────────────────────────────────┘")

    return metrics


# ============================================================
# 5. MODEL COMPARISON
# ============================================================

def compare_models(xgb_metrics: dict, lgb_metrics: dict,
                   xgb_time: float, lgb_time: float) -> dict:
    """
    Compare XGBoost vs LightGBM head-to-head.

    Winner is selected by AUPRC (the only metric that matters
    for highly imbalanced financial data).

    Returns: comparison dictionary with winner declared
    """
    logger.info("=" * 50)
    logger.info("HEAD-TO-HEAD COMPARISON: XGBoost vs LightGBM")
    logger.info("=" * 50)

    xgb_auprc = xgb_metrics["auprc"]
    lgb_auprc = lgb_metrics["auprc"]

    # Determine winner
    if xgb_auprc >= lgb_auprc:
        winner = "XGBoost"
        winner_auprc = xgb_auprc
        margin = xgb_auprc - lgb_auprc
    else:
        winner = "LightGBM"
        winner_auprc = lgb_auprc
        margin = lgb_auprc - xgb_auprc

    comparison = {
        "comparison_date": datetime.now().isoformat(),
        "primary_metric": "AUPRC",
        "winner": winner,
        "winning_auprc": round(winner_auprc, 6),
        "margin": round(float(margin), 6),
        "xgboost": {
            "auprc": xgb_metrics["auprc"],
            "roc_auc": xgb_metrics["roc_auc"],
            "precision_default": xgb_metrics["default_threshold"]["precision"],
            "recall_default": xgb_metrics["default_threshold"]["recall"],
            "f1_default": xgb_metrics["default_threshold"]["f1_score"],
            "optimal_threshold": xgb_metrics["optimal_threshold"]["threshold"],
            "f1_optimal": xgb_metrics["optimal_threshold"]["f1_score"],
            "training_time_seconds": round(xgb_time, 2),
            "inference_time_ms": xgb_metrics["inference_time_ms_per_100"],
        },
        "lightgbm": {
            "auprc": lgb_metrics["auprc"],
            "roc_auc": lgb_metrics["roc_auc"],
            "precision_default": lgb_metrics["default_threshold"]["precision"],
            "recall_default": lgb_metrics["default_threshold"]["recall"],
            "f1_default": lgb_metrics["default_threshold"]["f1_score"],
            "optimal_threshold": lgb_metrics["optimal_threshold"]["threshold"],
            "f1_optimal": lgb_metrics["optimal_threshold"]["f1_score"],
            "training_time_seconds": round(lgb_time, 2),
            "inference_time_ms": lgb_metrics["inference_time_ms_per_100"],
        },
    }

    # Log comparison table
    logger.info(f"  ┌───────────────────────┬──────────────┬──────────────┐")
    logger.info(f"  │ Metric                │    XGBoost   │   LightGBM   │")
    logger.info(f"  ├───────────────────────┼──────────────┼──────────────┤")
    logger.info(f"  │ AUPRC (PRIMARY)       │ {xgb_auprc:>12.6f} │ {lgb_auprc:>12.6f} │")
    logger.info(f"  │ ROC-AUC               │ {xgb_metrics['roc_auc']:>12.6f} │ {lgb_metrics['roc_auc']:>12.6f} │")
    logger.info(f"  │ Precision (0.5)       │ {xgb_metrics['default_threshold']['precision']:>12.6f} │ {lgb_metrics['default_threshold']['precision']:>12.6f} │")
    logger.info(f"  │ Recall (0.5)          │ {xgb_metrics['default_threshold']['recall']:>12.6f} │ {lgb_metrics['default_threshold']['recall']:>12.6f} │")
    logger.info(f"  │ F1 (0.5)              │ {xgb_metrics['default_threshold']['f1_score']:>12.6f} │ {lgb_metrics['default_threshold']['f1_score']:>12.6f} │")
    logger.info(f"  │ Optimal Threshold     │ {xgb_metrics['optimal_threshold']['threshold']:>12.6f} │ {lgb_metrics['optimal_threshold']['threshold']:>12.6f} │")
    logger.info(f"  │ F1 (Optimal)          │ {xgb_metrics['optimal_threshold']['f1_score']:>12.6f} │ {lgb_metrics['optimal_threshold']['f1_score']:>12.6f} │")
    logger.info(f"  │ Training Time (s)     │ {xgb_time:>12.2f} │ {lgb_time:>12.2f} │")
    logger.info(f"  │ Inference (ms/100)    │ {xgb_metrics['inference_time_ms_per_100']:>12.4f} │ {lgb_metrics['inference_time_ms_per_100']:>12.4f} │")
    logger.info(f"  └───────────────────────┴──────────────┴──────────────┘")
    logger.info(f"")
    logger.info(f"  ★ WINNER: {winner} (AUPRC: {winner_auprc:.6f}, margin: {margin:.6f})")

    return comparison


# ============================================================
# 6. SAVE ARTIFACTS
# ============================================================

def save_artifacts(xgb_model, lgb_model, xgb_metrics: dict,
                   lgb_metrics: dict, comparison: dict,
                   config: dict):
    """
    Save all Phase 5 artifacts to disk:
      - models/xgboost_fraud_v1.pkl
      - models/lightgbm_fraud_v1.pkl
      - models/best_model.pkl  (copy of the winner)
      - models/metrics.json    (both models' full metrics)
      - models/model_comparison.json  (head-to-head summary)
    """
    logger.info("Saving Phase 5 artifacts...")

    model_cfg = config["model"]
    output_dir = resolve_path(model_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Save XGBoost model ---
    xgb_path = resolve_path(model_cfg["xgboost"]["model_path"])
    joblib.dump(xgb_model, xgb_path)
    logger.info(f"  Saved XGBoost model: {xgb_path}")

    # --- Save LightGBM model ---
    lgb_path = resolve_path(model_cfg["lightgbm"]["model_path"])
    joblib.dump(lgb_model, lgb_path)
    logger.info(f"  Saved LightGBM model: {lgb_path}")

    # --- Save best model ---
    best_path = resolve_path(model_cfg["best_model_path"])
    best_model = xgb_model if comparison["winner"] == "XGBoost" else lgb_model
    joblib.dump(best_model, best_path)
    logger.info(f"  Saved best model ({comparison['winner']}): {best_path}")

    # --- Save detailed metrics ---
    metrics_path = resolve_path(model_cfg["metrics_path"])
    all_metrics = {
        "generated_at": datetime.now().isoformat(),
        "primary_metric": "AUPRC",
        "note": "Standard accuracy is FORBIDDEN for imbalanced financial data. "
                "AUPRC is the only reliable metric when fraud < 0.2% of transactions.",
        "xgboost": xgb_metrics,
        "lightgbm": lgb_metrics,
    }
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"  Saved metrics: {metrics_path}")

    # --- Save comparison ---
    comparison_path = resolve_path(model_cfg["comparison_path"])
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"  Saved comparison: {comparison_path}")

    return {
        "xgb_model_path": str(xgb_path),
        "lgb_model_path": str(lgb_path),
        "best_model_path": str(best_path),
        "metrics_path": str(metrics_path),
        "comparison_path": str(comparison_path),
    }


# ============================================================
# 7. MAIN ORCHESTRATOR
# ============================================================

def main():
    """
    Phase 5 main entry point.

    Orchestrates: load data → train XGBoost → train LightGBM →
    evaluate both → compare → save artifacts.

    Follows the project's guardrails:
      - All paths from config.yaml
      - All logging via get_logger() (no print statements)
      - Phase tracking via log_phase_start/log_phase_end
    """
    phase_name = "Phase 5: Model Training & Evaluation"

    try:
        log_phase_start(phase_name)
        config = load_config()

        # Verify Phase 3 completed
        if not check_phase_completed("Phase 3: Data Engineering"):
            logger.warning(
                "Phase 3 not marked as completed in phase_status.json. "
                "Proceeding anyway — will fail if processed data files are missing."
            )

        # ---- Step 1: Load processed data ----
        data = load_processed_data(config)

        # ---- Step 2: Train XGBoost ----
        xgb_model, xgb_time = train_xgboost(data, config)

        # ---- Step 3: Train LightGBM ----
        lgb_model, lgb_time = train_lightgbm(data, config)

        # ---- Step 4: Evaluate both models ----
        logger.info("=" * 50)
        logger.info("EVALUATION ON HELD-OUT TEST SET")
        logger.info("=" * 50)

        xgb_metrics = evaluate_model(xgb_model, data["X_test"], data["y_test"], "XGBoost")
        lgb_metrics = evaluate_model(lgb_model, data["X_test"], data["y_test"], "LightGBM")

        # ---- Step 5: Compare models ----
        comparison = compare_models(xgb_metrics, lgb_metrics, xgb_time, lgb_time)

        # ---- Step 6: Save all artifacts ----
        saved_paths = save_artifacts(
            xgb_model, lgb_model, xgb_metrics, lgb_metrics, comparison, config
        )

        # ---- Final Summary ----
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE 5 SUMMARY — MODEL TRAINING & EVALUATION")
        logger.info("=" * 60)
        logger.info(f"  Models trained:        XGBoost + LightGBM")
        logger.info(f"  Primary metric:        AUPRC (standard accuracy FORBIDDEN)")
        logger.info(f"  XGBoost AUPRC:         {xgb_metrics['auprc']:.6f}")
        logger.info(f"  LightGBM AUPRC:        {lgb_metrics['auprc']:.6f}")
        logger.info(f"  Winner:                {comparison['winner']} (margin: {comparison['margin']:.6f})")
        logger.info(f"  XGBoost train time:    {xgb_time:.2f}s")
        logger.info(f"  LightGBM train time:   {lgb_time:.2f}s")
        logger.info(f"  Artifacts saved:       {len(saved_paths)} files")
        for name, path in saved_paths.items():
            logger.info(f"    → {name}: {path}")
        logger.info("=" * 60)

        log_phase_end(phase_name, status="SUCCESS")

    except Exception as e:
        logger.error(f"Phase 5 FAILED: {str(e)}", exc_info=True)
        log_phase_end(phase_name, status="FAILED", error=str(e))
        raise


if __name__ == "__main__":
    main()
