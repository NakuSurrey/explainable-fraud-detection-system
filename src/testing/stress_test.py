"""
Phase 6: Adversarial Stress Testing (THE QA CHECK)
====================================================
Proves the model's robustness by feeding it manipulated data
that simulates real-world adversarial fraud tactics.

Reads:
  - models/best_model.pkl        (from Phase 5)
  - data/processed/X_test.csv    (from Phase 3)
  - data/processed/y_test.csv    (from Phase 3)
  - models/metrics.json          (from Phase 5 -- baseline metrics)

Saves:
  - reports/stress_test_results.json   (machine-readable results)
  - reports/stress_test_report.txt     (human-readable report)

Usage:
    python -m src.testing.stress_test
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Add project root to sys.path so imports work when run as module
# ---------------------------------------------------------------------------
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.logger import (
    get_logger,
    load_config,
    resolve_path,
    log_phase_start,
    log_phase_end,
)

logger = get_logger(__name__)


# ============================================================
# 1. LOAD MODEL AND DATA
# ============================================================

def load_model(config: dict):
    """Load the best model saved by Phase 5."""
    model_path = resolve_path(config["model"]["best_model_path"])
    if not model_path.exists():
        raise FileNotFoundError(
            f"Best model not found at {model_path}. "
            "Run Phase 5 (model training) first."
        )
    model = joblib.load(model_path)
    logger.info(f"Loaded best model from {model_path}")
    return model


def load_test_data(config: dict):
    """Load test features and labels from Phase 3 artifacts."""
    preprocess_cfg = config["preprocessing"]
    X_test_path = resolve_path(preprocess_cfg["test_path"])
    y_test_path = resolve_path(preprocess_cfg["y_test_path"])

    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            f"Test data not found. Expected:\n"
            f"  {X_test_path}\n  {y_test_path}\n"
            "Run Phase 3 (data engineering) first."
        )

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    logger.info(
        f"Loaded test data: {X_test.shape[0]} samples, "
        f"{X_test.shape[1]} features"
    )
    return X_test, y_test


def load_baseline_metrics(config: dict) -> dict:
    """Load baseline metrics from Phase 5 for comparison."""
    metrics_path = resolve_path(config["model"]["metrics_path"])
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Baseline metrics not found at {metrics_path}. "
            "Run Phase 5 first."
        )
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    logger.info(f"Loaded baseline metrics from {metrics_path}")
    return metrics


# ============================================================
# 2. ADVERSARIAL PERTURBATION STRATEGIES
# ============================================================

def get_fraud_samples(X_test: pd.DataFrame, y_test: np.ndarray,
                      n_samples: int) -> tuple:
    """
    Extract fraud samples for adversarial manipulation.
    These are the transactions the model SHOULD catch.
    We will manipulate them to see if the model still flags them.
    """
    fraud_mask = y_test == 1
    fraud_indices = np.where(fraud_mask)[0]

    if len(fraud_indices) == 0:
        raise ValueError("No fraud samples found in test set.")

    # If we have fewer fraud samples than requested, use all of them
    if len(fraud_indices) <= n_samples:
        selected = fraud_indices
        logger.info(
            f"Using all {len(selected)} fraud samples "
            f"(requested {n_samples})"
        )
    else:
        rng = np.random.RandomState(42)
        selected = rng.choice(fraud_indices, size=n_samples, replace=False)
        logger.info(f"Selected {n_samples} fraud samples for testing")

    return X_test.iloc[selected].copy(), y_test[selected].copy(), selected


def perturb_amount_reduction(X_fraud: pd.DataFrame,
                             reduction_pct: float) -> pd.DataFrame:
    """
    Simulate a fraudster lowering transaction amounts to avoid detection.

    In real-world fraud, attackers learn detection thresholds and
    deliberately keep amounts just below the radar. This test checks
    if the model relies too heavily on the Amount feature.

    Args:
        X_fraud: DataFrame of fraud transactions
        reduction_pct: Fraction to reduce Amount by (e.g., 0.3 = 30% lower)

    Returns:
        Perturbed copy of X_fraud
    """
    X_perturbed = X_fraud.copy()
    if "Amount" in X_perturbed.columns:
        original_mean = X_perturbed["Amount"].mean()
        X_perturbed["Amount"] = X_perturbed["Amount"] * (1 - reduction_pct)
        new_mean = X_perturbed["Amount"].mean()
        logger.info(
            f"  Amount reduced by {reduction_pct*100:.0f}%: "
            f"mean {original_mean:.2f} -> {new_mean:.2f}"
        )
    else:
        logger.warning("  'Amount' column not found -- skipping amount perturbation")
    return X_perturbed


def perturb_time_shift(X_fraud: pd.DataFrame,
                       shift_hours: float) -> pd.DataFrame:
    """
    Simulate fraudsters shifting attack times to business hours.

    Many fraud models learn that late-night transactions are risky.
    Sophisticated attackers shift their activity to normal business
    hours to blend in. This tests time-dependency.

    Args:
        X_fraud: DataFrame of fraud transactions
        shift_hours: Hours to shift the Time feature by

    Returns:
        Perturbed copy of X_fraud
    """
    X_perturbed = X_fraud.copy()
    if "Time" in X_perturbed.columns:
        shift_seconds = shift_hours * 3600
        original_mean = X_perturbed["Time"].mean()
        X_perturbed["Time"] = X_perturbed["Time"] + shift_seconds
        new_mean = X_perturbed["Time"].mean()
        logger.info(
            f"  Time shifted by +{shift_hours}h: "
            f"mean {original_mean:.0f}s -> {new_mean:.0f}s"
        )
    else:
        logger.warning("  'Time' column not found -- skipping time perturbation")
    return X_perturbed


def perturb_feature_noise(X_fraud: pd.DataFrame,
                          noise_std: float) -> pd.DataFrame:
    """
    Add Gaussian noise to ALL PCA features (V1-V28).

    This simulates data drift or slight behavioral changes that
    occur naturally as fraud patterns evolve over time. It tests
    whether the model generalizes or overfits to exact feature values.

    Args:
        X_fraud: DataFrame of fraud transactions
        noise_std: Standard deviation of Gaussian noise to add

    Returns:
        Perturbed copy of X_fraud
    """
    X_perturbed = X_fraud.copy()
    pca_cols = [c for c in X_perturbed.columns if c.startswith("V")]

    if len(pca_cols) == 0:
        logger.warning("  No PCA columns (V*) found -- skipping noise perturbation")
        return X_perturbed

    rng = np.random.RandomState(42)
    noise = rng.normal(0, noise_std, size=(len(X_perturbed), len(pca_cols)))
    X_perturbed[pca_cols] = X_perturbed[pca_cols].values + noise
    logger.info(
        f"  Gaussian noise (std={noise_std}) added to "
        f"{len(pca_cols)} PCA features"
    )
    return X_perturbed


def perturb_combined_attack(X_fraud: pd.DataFrame,
                            amount_reduction: float,
                            time_shift: float,
                            noise_std: float) -> pd.DataFrame:
    """
    Simulate a sophisticated attacker using ALL evasion tactics at once.

    This is the worst-case scenario: the attacker lowers amounts,
    shifts to business hours, AND slightly alters behavioral patterns.
    If the model survives this, it is production-ready.
    """
    X_perturbed = X_fraud.copy()
    X_perturbed = perturb_amount_reduction(X_perturbed, amount_reduction)
    X_perturbed = perturb_time_shift(X_perturbed, time_shift)
    X_perturbed = perturb_feature_noise(X_perturbed, noise_std)
    return X_perturbed


# ============================================================
# 3. EVALUATION UNDER STRESS
# ============================================================

def evaluate_under_stress(model, X_perturbed: pd.DataFrame,
                          y_true: np.ndarray,
                          test_name: str,
                          optimal_threshold: float = 0.5) -> dict:
    """
    Evaluate model performance on perturbed data.

    Uses both default (0.5) and optimal thresholds from Phase 5.

    Returns:
        Dictionary of metrics for this stress scenario
    """
    y_proba = model.predict_proba(X_perturbed)[:, 1]

    # Metrics at default threshold
    y_pred_default = (y_proba >= 0.5).astype(int)
    # Metrics at optimal threshold
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

    auprc = average_precision_score(y_true, y_proba)

    results = {
        "test_name": test_name,
        "n_samples": int(len(y_true)),
        "n_fraud_actual": int(y_true.sum()),
        "auprc": round(float(auprc), 6),
        "default_threshold": {
            "threshold": 0.5,
            "fraud_detected": int(y_pred_default.sum()),
            "fraud_missed": int(y_true.sum() - (y_pred_default & y_true.astype(bool)).sum()),
            "detection_rate": round(
                float(recall_score(y_true, y_pred_default, zero_division=0)), 6
            ),
            "precision": round(
                float(precision_score(y_true, y_pred_default, zero_division=0)), 6
            ),
            "f1_score": round(
                float(f1_score(y_true, y_pred_default, zero_division=0)), 6
            ),
        },
        "optimal_threshold": {
            "threshold": round(optimal_threshold, 6),
            "fraud_detected": int(y_pred_optimal.sum()),
            "fraud_missed": int(y_true.sum() - (y_pred_optimal & y_true.astype(bool)).sum()),
            "detection_rate": round(
                float(recall_score(y_true, y_pred_optimal, zero_division=0)), 6
            ),
            "precision": round(
                float(precision_score(y_true, y_pred_optimal, zero_division=0)), 6
            ),
            "f1_score": round(
                float(f1_score(y_true, y_pred_optimal, zero_division=0)), 6
            ),
        },
        "mean_fraud_probability": round(float(y_proba.mean()), 6),
        "median_fraud_probability": round(float(np.median(y_proba)), 6),
        "min_fraud_probability": round(float(y_proba.min()), 6),
        "max_fraud_probability": round(float(y_proba.max()), 6),
    }

    # Log summary
    det_rate = results["optimal_threshold"]["detection_rate"]
    missed = results["optimal_threshold"]["fraud_missed"]
    status = "PASS" if det_rate >= 0.50 else "WARN" if det_rate >= 0.30 else "FAIL"
    logger.info(
        f"  [{status}] {test_name}: "
        f"AUPRC={auprc:.4f}, "
        f"Detection={det_rate*100:.1f}%, "
        f"Missed={missed}/{results['n_fraud_actual']}"
    )

    return results


# ============================================================
# 4. GENERATE HUMAN-READABLE REPORT
# ============================================================

def generate_report(all_results: dict, config: dict) -> str:
    """
    Generate a detailed, human-readable stress test report.

    This report is designed for:
    - Auditors who need to verify model robustness
    - Risk managers who need to understand failure modes
    - Future developers who inherit this codebase
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ADVERSARIAL STRESS TEST REPORT")
    lines.append("Explainable Fraud Detection System")
    lines.append(f"Generated: {all_results['metadata']['timestamp']}")
    lines.append("=" * 70)
    lines.append("")

    # --- Executive Summary ---
    lines.append("-" * 70)
    lines.append("EXECUTIVE SUMMARY")
    lines.append("-" * 70)
    summary = all_results["summary"]
    lines.append(f"Total tests run:         {summary['total_tests']}")
    lines.append(f"Tests PASSED:            {summary['passed']}")
    lines.append(f"Tests WARNING:           {summary['warnings']}")
    lines.append(f"Tests FAILED:            {summary['failed']}")
    lines.append(f"Overall verdict:         {summary['verdict']}")
    lines.append(f"Baseline AUPRC:          {summary['baseline_auprc']:.6f}")
    lines.append(f"Worst-case AUPRC:        {summary['worst_case_auprc']:.6f}")
    lines.append(
        f"Max AUPRC degradation:   "
        f"{summary['max_auprc_degradation']:.6f} "
        f"({summary['max_degradation_pct']:.1f}%)"
    )
    lines.append("")

    # --- Baseline Performance ---
    lines.append("-" * 70)
    lines.append("BASELINE PERFORMANCE (unperturbed test data)")
    lines.append("-" * 70)
    bl = all_results["baseline"]
    lines.append(f"AUPRC:           {bl['auprc']:.6f}")
    lines.append(f"Detection Rate:  {bl['optimal_threshold']['detection_rate']*100:.1f}%")
    lines.append(f"Precision:       {bl['optimal_threshold']['precision']*100:.1f}%")
    lines.append(f"F1 Score:        {bl['optimal_threshold']['f1_score']:.4f}")
    lines.append(f"Threshold used:  {bl['optimal_threshold']['threshold']:.6f}")
    lines.append("")

    # --- Individual Test Results ---
    lines.append("-" * 70)
    lines.append("DETAILED TEST RESULTS")
    lines.append("-" * 70)

    for category_name, tests in all_results["tests"].items():
        lines.append("")
        lines.append(f"  Category: {category_name.upper().replace('_', ' ')}")
        lines.append(f"  {'.'*60}")

        for test in tests:
            auprc = test["auprc"]
            det_rate = test["optimal_threshold"]["detection_rate"]
            missed = test["optimal_threshold"]["fraud_missed"]
            total = test["n_fraud_actual"]
            baseline_auprc = bl["auprc"]
            degradation = baseline_auprc - auprc
            deg_pct = (degradation / baseline_auprc * 100) if baseline_auprc > 0 else 0

            status = "PASS" if det_rate >= 0.50 else "WARN" if det_rate >= 0.30 else "FAIL"

            lines.append(f"")
            lines.append(f"    Test: {test['test_name']}")
            lines.append(f"    Status:           [{status}]")
            lines.append(f"    AUPRC:            {auprc:.6f} (degradation: {degradation:+.6f} / {deg_pct:+.1f}%)")
            lines.append(f"    Detection Rate:   {det_rate*100:.1f}% ({total - missed}/{total} caught)")
            lines.append(f"    Precision:        {test['optimal_threshold']['precision']*100:.1f}%")
            lines.append(f"    Avg Fraud Prob:   {test['mean_fraud_probability']:.4f}")

    lines.append("")

    # --- Robustness Analysis ---
    lines.append("-" * 70)
    lines.append("ROBUSTNESS ANALYSIS")
    lines.append("-" * 70)
    lines.append("")

    # Amount sensitivity
    amount_tests = all_results["tests"].get("amount_reduction", [])
    if amount_tests:
        lines.append("  Amount Sensitivity:")
        lines.append("  The model's resilience when transaction amounts are reduced.")
        for t in amount_tests:
            deg = bl["auprc"] - t["auprc"]
            lines.append(
                f"    {t['test_name']:45s} -> AUPRC drop: {deg:+.6f}"
            )
        lines.append("")

    # Time sensitivity
    time_tests = all_results["tests"].get("time_shift", [])
    if time_tests:
        lines.append("  Temporal Sensitivity:")
        lines.append("  The model's resilience when transaction times are shifted.")
        for t in time_tests:
            deg = bl["auprc"] - t["auprc"]
            lines.append(
                f"    {t['test_name']:45s} -> AUPRC drop: {deg:+.6f}"
            )
        lines.append("")

    # Noise sensitivity
    noise_tests = all_results["tests"].get("feature_noise", [])
    if noise_tests:
        lines.append("  Feature Noise Sensitivity:")
        lines.append("  The model's resilience to Gaussian noise on PCA features.")
        for t in noise_tests:
            deg = bl["auprc"] - t["auprc"]
            lines.append(
                f"    {t['test_name']:45s} -> AUPRC drop: {deg:+.6f}"
            )
        lines.append("")

    # Combined attack
    combined_tests = all_results["tests"].get("combined_attack", [])
    if combined_tests:
        lines.append("  Combined Attack (worst-case):")
        lines.append("  Simultaneous amount reduction + time shift + feature noise.")
        for t in combined_tests:
            deg = bl["auprc"] - t["auprc"]
            lines.append(
                f"    {t['test_name']:45s} -> AUPRC drop: {deg:+.6f}"
            )
        lines.append("")

    # --- Conclusion ---
    lines.append("-" * 70)
    lines.append("CONCLUSION & DEPLOYMENT RECOMMENDATION")
    lines.append("-" * 70)
    lines.append("")

    verdict = summary["verdict"]
    if verdict == "PRODUCTION READY":
        lines.append(
            "  The model has passed all adversarial stress tests. "
            "It demonstrates strong\n"
            "  robustness against amount manipulation, temporal shifts, "
            "and feature noise.\n"
            "  RECOMMENDATION: Approved for deployment to production."
        )
    elif verdict == "CONDITIONALLY READY":
        lines.append(
            "  The model passed most tests but showed weakness in some "
            "scenarios.\n"
            "  RECOMMENDATION: Deploy with enhanced monitoring on the "
            "flagged scenarios.\n"
            "  Consider retraining with augmented adversarial examples."
        )
    else:
        lines.append(
            "  The model failed critical stress tests. "
            "It is NOT ready for production.\n"
            "  RECOMMENDATION: Do NOT deploy. Retrain with additional "
            "data or adjust\n"
            "  feature engineering to reduce sensitivity to the "
            "failing perturbations."
        )

    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def main():
    """Run the complete adversarial stress testing pipeline."""
    phase_name = "Phase 6: Adversarial Stress Testing"
    log_phase_start(phase_name)

    try:
        config = load_config()
        stress_cfg = config["stress_test"]

        # ---------------------------------------------------
        # Step 1: Load model and data
        # ---------------------------------------------------
        logger.info("=" * 50)
        logger.info("STEP 1/5: Loading model and test data")
        logger.info("=" * 50)

        model = load_model(config)
        X_test, y_test = load_test_data(config)

        # Load baseline metrics to get optimal threshold
        baseline_metrics = load_baseline_metrics(config)

        # Extract the winning model's optimal threshold
        # The metrics.json stores both models -- find the winner
        winner_key = None
        for key in ["xgboost", "lightgbm"]:
            if key in baseline_metrics:
                if winner_key is None:
                    winner_key = key
                elif baseline_metrics[key]["auprc"] > baseline_metrics[winner_key]["auprc"]:
                    winner_key = key

        if winner_key and "optimal_threshold" in baseline_metrics[winner_key]:
            optimal_threshold = baseline_metrics[winner_key]["optimal_threshold"]["threshold"]
        else:
            optimal_threshold = 0.5
            logger.warning(
                "Could not find optimal threshold in metrics.json. "
                "Using default 0.5"
            )

        logger.info(f"Using optimal threshold: {optimal_threshold:.6f}")

        # ---------------------------------------------------
        # Step 2: Establish baseline on unperturbed data
        # ---------------------------------------------------
        logger.info("")
        logger.info("=" * 50)
        logger.info("STEP 2/5: Establishing baseline on fraud samples")
        logger.info("=" * 50)

        n_samples = stress_cfg.get("n_adversarial_samples", 500)
        X_fraud, y_fraud, fraud_indices = get_fraud_samples(
            X_test, y_test, n_samples
        )

        baseline = evaluate_under_stress(
            model, X_fraud, y_fraud,
            "BASELINE (no perturbation)",
            optimal_threshold
        )

        # ---------------------------------------------------
        # Step 3: Run amount reduction tests
        # ---------------------------------------------------
        logger.info("")
        logger.info("=" * 50)
        logger.info("STEP 3/5: Amount reduction attacks")
        logger.info("=" * 50)

        perturbations = stress_cfg.get("perturbations", {})
        amount_reductions = perturbations.get(
            "amount_reduction_pct", [0.1, 0.2, 0.3, 0.5]
        )

        amount_results = []
        for pct in amount_reductions:
            logger.info(f"  Testing: Reduce Amount by {pct*100:.0f}%")
            X_pert = perturb_amount_reduction(X_fraud, pct)
            result = evaluate_under_stress(
                model, X_pert, y_fraud,
                f"Amount reduced by {pct*100:.0f}%",
                optimal_threshold
            )
            result["perturbation_params"] = {"type": "amount_reduction", "reduction_pct": pct}
            amount_results.append(result)

        # ---------------------------------------------------
        # Step 4: Run time shift tests
        # ---------------------------------------------------
        logger.info("")
        logger.info("=" * 50)
        logger.info("STEP 4/5: Time shift attacks")
        logger.info("=" * 50)

        time_shifts = perturbations.get(
            "time_shift_hours", [1, 3, 6, 12]
        )

        time_results = []
        for hours in time_shifts:
            logger.info(f"  Testing: Shift Time by +{hours}h")
            X_pert = perturb_time_shift(X_fraud, hours)
            result = evaluate_under_stress(
                model, X_pert, y_fraud,
                f"Time shifted by +{hours}h",
                optimal_threshold
            )
            result["perturbation_params"] = {"type": "time_shift", "shift_hours": hours}
            time_results.append(result)

        # ---------------------------------------------------
        # Step 5: Run feature noise tests
        # ---------------------------------------------------
        logger.info("")
        logger.info("=" * 50)
        logger.info("STEP 5/5: Feature noise + combined attacks")
        logger.info("=" * 50)

        noise_stds = perturbations.get(
            "feature_noise_std", [0.01, 0.05, 0.1]
        )

        noise_results = []
        for std in noise_stds:
            logger.info(f"  Testing: Gaussian noise (std={std})")
            X_pert = perturb_feature_noise(X_fraud, std)
            result = evaluate_under_stress(
                model, X_pert, y_fraud,
                f"Feature noise (std={std})",
                optimal_threshold
            )
            result["perturbation_params"] = {"type": "feature_noise", "noise_std": std}
            noise_results.append(result)

        # Combined attack: worst-case scenario
        logger.info("")
        logger.info("  Testing: COMBINED ATTACK (worst-case scenario)")
        combined_results = []

        # Use the most aggressive perturbation from each category
        worst_amount = max(amount_reductions)
        worst_time = max(time_shifts)
        worst_noise = max(noise_stds)

        X_combined = perturb_combined_attack(
            X_fraud, worst_amount, worst_time, worst_noise
        )
        combined_result = evaluate_under_stress(
            model, X_combined, y_fraud,
            f"COMBINED: Amount-{worst_amount*100:.0f}% + "
            f"Time+{worst_time}h + Noise(std={worst_noise})",
            optimal_threshold
        )
        combined_result["perturbation_params"] = {
            "type": "combined",
            "amount_reduction_pct": worst_amount,
            "time_shift_hours": worst_time,
            "noise_std": worst_noise,
        }
        combined_results.append(combined_result)

        # Also test a moderate combined attack
        mid_amount = amount_reductions[len(amount_reductions) // 2]
        mid_time = time_shifts[len(time_shifts) // 2]
        mid_noise = noise_stds[len(noise_stds) // 2]

        X_moderate = perturb_combined_attack(
            X_fraud, mid_amount, mid_time, mid_noise
        )
        moderate_result = evaluate_under_stress(
            model, X_moderate, y_fraud,
            f"COMBINED (moderate): Amount-{mid_amount*100:.0f}% + "
            f"Time+{mid_time}h + Noise(std={mid_noise})",
            optimal_threshold
        )
        moderate_result["perturbation_params"] = {
            "type": "combined_moderate",
            "amount_reduction_pct": mid_amount,
            "time_shift_hours": mid_time,
            "noise_std": mid_noise,
        }
        combined_results.append(moderate_result)

        # ---------------------------------------------------
        # Compile all results
        # ---------------------------------------------------
        all_tests = {
            "amount_reduction": amount_results,
            "time_shift": time_results,
            "feature_noise": noise_results,
            "combined_attack": combined_results,
        }

        # Count pass/warn/fail
        total_tests = 0
        passed = 0
        warnings = 0
        failed = 0

        all_auprcs = [baseline["auprc"]]

        for category_tests in all_tests.values():
            for t in category_tests:
                total_tests += 1
                det_rate = t["optimal_threshold"]["detection_rate"]
                if det_rate >= 0.50:
                    passed += 1
                elif det_rate >= 0.30:
                    warnings += 1
                else:
                    failed += 1
                all_auprcs.append(t["auprc"])

        worst_auprc = min(all_auprcs)
        max_degradation = baseline["auprc"] - worst_auprc
        max_deg_pct = (max_degradation / baseline["auprc"] * 100) if baseline["auprc"] > 0 else 0

        # Determine verdict
        if failed == 0 and warnings == 0:
            verdict = "PRODUCTION READY"
        elif failed == 0:
            verdict = "CONDITIONALLY READY"
        else:
            verdict = "NEEDS IMPROVEMENT"

        all_results = {
            "metadata": {
                "phase": "Phase 6: Adversarial Stress Testing",
                "timestamp": datetime.now().isoformat(),
                "model_used": "best_model.pkl (XGBoost)",
                "optimal_threshold": optimal_threshold,
                "n_fraud_samples_tested": int(len(y_fraud)),
                "perturbation_config": {
                    "amount_reduction_pct": amount_reductions,
                    "time_shift_hours": time_shifts,
                    "feature_noise_std": noise_stds,
                },
            },
            "summary": {
                "total_tests": total_tests,
                "passed": passed,
                "warnings": warnings,
                "failed": failed,
                "verdict": verdict,
                "baseline_auprc": round(baseline["auprc"], 6),
                "worst_case_auprc": round(worst_auprc, 6),
                "max_auprc_degradation": round(max_degradation, 6),
                "max_degradation_pct": round(max_deg_pct, 1),
            },
            "baseline": baseline,
            "tests": all_tests,
        }

        # ---------------------------------------------------
        # Save artifacts
        # ---------------------------------------------------
        logger.info("")
        logger.info("=" * 50)
        logger.info("SAVING ARTIFACTS")
        logger.info("=" * 50)

        # Save JSON results
        results_path = resolve_path(stress_cfg["results_path"])
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Saved: {results_path}")

        # Save human-readable report
        report_path = resolve_path(stress_cfg["report_path"])
        report_text = generate_report(all_results, config)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info(f"Saved: {report_path}")

        # ---------------------------------------------------
        # Final summary
        # ---------------------------------------------------
        logger.info("")
        logger.info("=" * 50)
        logger.info("PHASE 6 COMPLETE -- STRESS TEST SUMMARY")
        logger.info("=" * 50)
        logger.info(f"  Verdict:            {verdict}")
        logger.info(f"  Tests Passed:       {passed}/{total_tests}")
        logger.info(f"  Tests Warning:      {warnings}/{total_tests}")
        logger.info(f"  Tests Failed:       {failed}/{total_tests}")
        logger.info(f"  Baseline AUPRC:     {baseline['auprc']:.6f}")
        logger.info(f"  Worst-case AUPRC:   {worst_auprc:.6f}")
        logger.info(f"  Max degradation:    {max_degradation:.6f} ({max_deg_pct:.1f}%)")
        logger.info("=" * 50)

        log_phase_end(phase_name, "SUCCESS")

    except Exception as e:
        logger.error(f"Phase 6 FAILED: {e}", exc_info=True)
        log_phase_end(phase_name, "FAILED", str(e))
        raise


if __name__ == "__main__":
    main()
