"""
Phase 11: Model Monitoring (THE WATCHTOWER)
=============================================
Tracks prediction drift over time by comparing production prediction
distributions against a training baseline using PSI (Population
Stability Index).

Read -> Process -> Save Artifact principle:
  - READS:  models/prediction_baseline.json (training distribution)
  - READS:  data/monitoring/predictions.db (logged production predictions)
  - PROCESS: Bins distributions, calculates PSI, determines drift verdict
  - SAVES:  data/monitoring/predictions.db (prediction logs)
  - SAVES:  models/prediction_baseline.json (generated once from training data)

What PSI is:
  PSI measures how much a probability distribution has shifted compared
  to a reference distribution. It splits both distributions into buckets
  (bins), then calculates how different each bucket is.

  PSI < 0.1  = no meaningful drift (model is stable)
  PSI 0.1-0.2 = moderate drift (keep watching)
  PSI >= 0.2 = significant drift (consider retraining)

Usage:
    # generate baseline (run once after training):
    python -m src.monitoring.prediction_monitor --generate-baseline

    # check drift (run periodically or on demand):
    python -m src.monitoring.prediction_monitor --check-drift

Dependencies (all already in requirements.txt):
    numpy, pandas, sqlalchemy
"""

import sys
import json
import sqlite3
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

# --- project imports ---
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.logger import get_logger, load_config, resolve_path

logger = get_logger("phase11.monitoring")


# ============================================================
# 1. CONSTANTS
# ============================================================

# 10 equal-width bins from 0.0 to 1.0 — standard for probability distributions
PSI_BIN_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# small value to avoid division by zero or log(0) in PSI formula
PSI_EPSILON = 1e-6

# default number of recent predictions to use when checking drift
DEFAULT_PREDICTION_WINDOW = 500


# ============================================================
# 2. DATABASE INITIALIZATION
# ============================================================

def get_monitoring_db_path() -> Path:
    """
    Resolve the path to the monitoring SQLite database.
    Uses config.yaml if a monitoring.db_path key exists,
    otherwise falls back to data/monitoring/predictions.db.
    """
    config = load_config()
    monitoring_cfg = config.get("monitoring", {})
    db_path_str = monitoring_cfg.get("db_path", "data/monitoring/predictions.db")
    db_path = resolve_path(db_path_str)
    # make sure the parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def init_monitoring_db() -> Path:
    """
    Create the predictions table if it does not exist yet.
    Returns the path to the database file.

    Table schema:
      id              — auto-incrementing primary key
      timestamp       — ISO 8601 string when the prediction was made
      probability     — fraud probability the model returned (0.0 to 1.0)
      risk_level      — text label (LOW, MODERATE, HIGH, CRITICAL)
      is_flagged      — 1 if probability >= threshold, 0 otherwise
      threshold_used  — the threshold value that was applied
    """
    db_path = get_monitoring_db_path()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            probability     REAL NOT NULL,
            risk_level      TEXT NOT NULL,
            is_flagged      INTEGER NOT NULL,
            threshold_used  REAL NOT NULL
        )
    """)

    # index on timestamp — needed for querying recent predictions quickly
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
        ON predictions (timestamp)
    """)

    conn.commit()
    conn.close()

    logger.info(f"Monitoring DB initialized at: {db_path}")
    return db_path


# ============================================================
# 3. PREDICTION LOGGING
# ============================================================

def log_prediction(
    probability: float,
    risk_level: str,
    is_flagged: bool,
    threshold_used: float
) -> int:
    """
    Write one prediction record to the monitoring database.
    Called silently by the API after every /predict response.

    Returns the row ID of the inserted record.

    This function is designed to never crash the API.
    If logging fails for any reason, it catches the error,
    logs a warning, and returns -1 so the API can continue.
    """
    try:
        db_path = get_monitoring_db_path()
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO predictions (timestamp, probability, risk_level, is_flagged, threshold_used)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.utcnow().isoformat(),
                round(float(probability), 6),
                str(risk_level),
                1 if is_flagged else 0,
                round(float(threshold_used), 6),
            )
        )

        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return row_id

    except Exception as e:
        # never let monitoring crash the prediction pipeline
        logger.warning(f"Failed to log prediction to monitoring DB: {e}")
        return -1


def get_recent_predictions(
    limit: int = DEFAULT_PREDICTION_WINDOW,
    days: Optional[int] = None
) -> list:
    """
    Fetch recent predictions from the monitoring database.

    Two modes:
      - limit only:  returns the last N predictions (most recent first)
      - days + limit: returns up to N predictions from the last X days

    Returns a list of dicts, each with:
      id, timestamp, probability, risk_level, is_flagged, threshold_used
    """
    db_path = get_monitoring_db_path()

    if not db_path.exists():
        logger.warning("Monitoring DB does not exist yet — no predictions logged")
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if days is not None:
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        cursor.execute(
            """
            SELECT id, timestamp, probability, risk_level, is_flagged, threshold_used
            FROM predictions
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (cutoff, limit)
        )
    else:
        cursor.execute(
            """
            SELECT id, timestamp, probability, risk_level, is_flagged, threshold_used
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,)
        )

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_prediction_count() -> int:
    """
    Return the total number of predictions logged so far.
    Quick check to see if enough data exists for drift analysis.
    """
    db_path = get_monitoring_db_path()

    if not db_path.exists():
        return 0

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    count = cursor.fetchone()[0]
    conn.close()
    return count


# ============================================================
# 4. BASELINE GENERATION
# ============================================================

def generate_baseline(probabilities: np.ndarray, output_path: Optional[str] = None) -> dict:
    """
    Create a baseline distribution from training data probabilities.

    This should be run ONCE after model training. It takes the
    fraud probabilities the model predicted on the test set,
    bins them into 10 buckets, and saves the proportions.

    How it works step by step:
      Step 1: Take the array of probabilities (one per test sample)
      Step 2: Count how many fall into each of 10 bins (0.0-0.1, 0.1-0.2, ... 0.9-1.0)
      Step 3: Convert counts to proportions (each bin / total)
      Step 4: Save as JSON with metadata

    Args:
        probabilities: numpy array of fraud probabilities from test set predictions
        output_path: where to save the JSON file (defaults to models/prediction_baseline.json)

    Returns:
        The baseline dictionary that was saved
    """
    config = load_config()

    if output_path is None:
        monitoring_cfg = config.get("monitoring", {})
        output_path = monitoring_cfg.get(
            "baseline_path", "models/prediction_baseline.json"
        )

    save_path = resolve_path(output_path)

    # bin the probabilities into 10 buckets
    counts, _ = np.histogram(probabilities, bins=PSI_BIN_EDGES)

    # convert to proportions (each bucket's share of the total)
    total = len(probabilities)
    proportions = (counts / total).tolist()

    baseline = {
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "n_samples": total,
            "source": "test_set_predictions",
            "bin_edges": PSI_BIN_EDGES,
            "n_bins": len(PSI_BIN_EDGES) - 1,
        },
        "distribution": proportions,
        "statistics": {
            "mean_probability": round(float(np.mean(probabilities)), 6),
            "median_probability": round(float(np.median(probabilities)), 6),
            "std_probability": round(float(np.std(probabilities)), 6),
            "min_probability": round(float(np.min(probabilities)), 6),
            "max_probability": round(float(np.max(probabilities)), 6),
            "fraud_rate_at_0.5": round(
                float(np.mean(probabilities >= 0.5)), 6
            ),
        },
    }

    # save to disk
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(baseline, f, indent=2)

    logger.info(f"Prediction baseline saved to: {save_path}")
    logger.info(
        f"  Samples: {total}, "
        f"Mean prob: {baseline['statistics']['mean_probability']:.6f}, "
        f"Fraud rate (>0.5): {baseline['statistics']['fraud_rate_at_0.5']:.4f}"
    )

    return baseline


def load_baseline(baseline_path: Optional[str] = None) -> dict:
    """
    Load the previously saved baseline distribution from JSON.

    Returns the baseline dict, or raises FileNotFoundError
    if no baseline has been generated yet.
    """
    config = load_config()

    if baseline_path is None:
        monitoring_cfg = config.get("monitoring", {})
        baseline_path = monitoring_cfg.get(
            "baseline_path", "models/prediction_baseline.json"
        )

    full_path = resolve_path(baseline_path)

    if not full_path.exists():
        raise FileNotFoundError(
            f"No baseline found at {full_path}. "
            f"Run 'python -m src.monitoring.prediction_monitor --generate-baseline' first."
        )

    with open(full_path, "r") as f:
        baseline = json.load(f)

    return baseline


# ============================================================
# 5. PSI CALCULATION
# ============================================================

def calculate_psi(expected: list, actual: list) -> dict:
    """
    Calculate Population Stability Index between two distributions.

    How PSI works — step by step:
      Step 1: Take two lists of proportions (expected and actual)
              Each list has 10 values — one per bin
              Each value = what fraction of total predictions fell in that bin

      Step 2: For each bin, apply the PSI formula:
              (actual_i - expected_i) * ln(actual_i / expected_i)

      Step 3: Sum all 10 bin values = total PSI score

      Step 4: Interpret:
              PSI < 0.1   -> NO_DRIFT (stable)
              PSI 0.1-0.2 -> MODERATE_DRIFT (watch)
              PSI >= 0.2  -> SIGNIFICANT_DRIFT (retrain)

    Why we add epsilon (0.000001):
      If a bin has zero predictions, the proportion would be 0.
      log(0) is undefined and 1/0 is infinity.
      Adding a tiny number prevents this math error without
      changing the result in any meaningful way.

    Args:
        expected: list of 10 proportions from baseline
        actual: list of 10 proportions from recent predictions

    Returns:
        dict with psi_score, per_bin_psi, verdict, recommendation
    """
    expected_arr = np.array(expected, dtype=float)
    actual_arr = np.array(actual, dtype=float)

    # add epsilon to both — prevents division by zero and log(0)
    expected_safe = expected_arr + PSI_EPSILON
    actual_safe = actual_arr + PSI_EPSILON

    # re-normalize after adding epsilon so they still sum to ~1.0
    expected_safe = expected_safe / expected_safe.sum()
    actual_safe = actual_safe / actual_safe.sum()

    # PSI formula for each bin
    per_bin_psi = (actual_safe - expected_safe) * np.log(actual_safe / expected_safe)
    total_psi = float(np.sum(per_bin_psi))

    # determine verdict using config thresholds
    config = load_config()
    monitoring_cfg = config.get("monitoring", {})
    psi_threshold = monitoring_cfg.get("psi_threshold", 0.2)

    # three levels: stable, moderate, significant
    if total_psi < 0.1:
        verdict = "NO_DRIFT"
        recommendation = "Model is stable. No action needed."
    elif total_psi < psi_threshold:
        verdict = "MODERATE_DRIFT"
        recommendation = "Distribution has shifted slightly. Monitor closely over the next cycle."
    else:
        verdict = "SIGNIFICANT_DRIFT"
        recommendation = (
            f"PSI ({total_psi:.4f}) exceeds threshold ({psi_threshold}). "
            f"Review recent data for distribution changes and consider retraining."
        )

    return {
        "psi_score": round(total_psi, 6),
        "verdict": verdict,
        "recommendation": recommendation,
        "threshold_used": psi_threshold,
        "per_bin_psi": [round(float(v), 6) for v in per_bin_psi],
    }


# ============================================================
# 6. DRIFT REPORT GENERATION
# ============================================================

def generate_drift_report(
    prediction_window: int = DEFAULT_PREDICTION_WINDOW,
    days: Optional[int] = None
) -> dict:
    """
    Full drift analysis — the main function that produces the complete report.

    How it works step by step:
      Step 1: Load the baseline distribution (from generate_baseline)
      Step 2: Fetch recent predictions from the monitoring DB
      Step 3: Bin the recent predictions into the same 10 buckets
      Step 4: Calculate PSI between baseline and recent
      Step 5: Build a full report with metadata, comparison, and verdict

    Args:
        prediction_window: max number of recent predictions to analyze
        days: optional — only look at predictions from the last N days

    Returns:
        Complete drift report dict with all details
    """
    # Step 1 — load baseline
    try:
        baseline = load_baseline()
    except FileNotFoundError as e:
        return {
            "status": "ERROR",
            "message": str(e),
            "recommendation": "Generate a baseline first using the --generate-baseline command.",
        }

    baseline_distribution = baseline["distribution"]

    # Step 2 — fetch recent predictions
    recent = get_recent_predictions(limit=prediction_window, days=days)

    if len(recent) < 30:
        return {
            "status": "INSUFFICIENT_DATA",
            "message": f"Only {len(recent)} predictions logged. Need at least 30 for meaningful drift analysis.",
            "predictions_logged": len(recent),
            "recommendation": "Continue using the system. Drift analysis will be available after 30+ predictions.",
        }

    # extract probabilities from recent predictions
    recent_probabilities = np.array([r["probability"] for r in recent])

    # Step 3 — bin recent predictions into same 10 buckets
    recent_counts, _ = np.histogram(recent_probabilities, bins=PSI_BIN_EDGES)
    recent_distribution = (recent_counts / len(recent_probabilities)).tolist()

    # Step 4 — calculate PSI
    psi_result = calculate_psi(baseline_distribution, recent_distribution)

    # Step 5 — build the full report
    config = load_config()
    monitoring_cfg = config.get("monitoring", {})

    report = {
        "status": "OK",
        "generated_at": datetime.utcnow().isoformat(),
        "drift_analysis": {
            "psi_score": psi_result["psi_score"],
            "verdict": psi_result["verdict"],
            "recommendation": psi_result["recommendation"],
            "threshold": psi_result["threshold_used"],
            "per_bin_psi": psi_result["per_bin_psi"],
        },
        "baseline": {
            "n_samples": baseline["metadata"]["n_samples"],
            "generated_at": baseline["metadata"]["generated_at"],
            "distribution": baseline_distribution,
            "mean_probability": baseline["statistics"]["mean_probability"],
        },
        "recent": {
            "n_samples": len(recent_probabilities),
            "window_days": days,
            "distribution": recent_distribution,
            "mean_probability": round(float(np.mean(recent_probabilities)), 6),
            "median_probability": round(float(np.median(recent_probabilities)), 6),
            "std_probability": round(float(np.std(recent_probabilities)), 6),
            "fraud_rate_at_0.5": round(
                float(np.mean(recent_probabilities >= 0.5)), 6
            ),
        },
        "comparison": {
            "mean_shift": round(
                float(np.mean(recent_probabilities)) - baseline["statistics"]["mean_probability"],
                6,
            ),
            "bin_labels": [
                f"{PSI_BIN_EDGES[i]:.1f}-{PSI_BIN_EDGES[i+1]:.1f}"
                for i in range(len(PSI_BIN_EDGES) - 1)
            ],
            "baseline_proportions": baseline_distribution,
            "recent_proportions": recent_distribution,
        },
        "config": {
            "psi_threshold": monitoring_cfg.get("psi_threshold", 0.2),
            "auprc_decay_threshold": monitoring_cfg.get("auprc_decay_threshold", 0.05),
            "drift_check_interval_days": monitoring_cfg.get("drift_check_interval_days", 7),
        },
    }

    # log the result
    logger.info(
        f"Drift report: PSI={psi_result['psi_score']:.4f}, "
        f"Verdict={psi_result['verdict']}, "
        f"Samples={len(recent_probabilities)}"
    )

    return report


# ============================================================
# 7. MONITORING SUMMARY (quick status check)
# ============================================================

def get_monitoring_summary() -> dict:
    """
    Quick summary of the monitoring system status.
    Used by the /health endpoint or dashboard for a fast overview
    without running the full drift analysis.
    """
    total_predictions = get_prediction_count()

    # check if baseline exists
    try:
        baseline = load_baseline()
        baseline_exists = True
        baseline_date = baseline["metadata"]["generated_at"]
        baseline_samples = baseline["metadata"]["n_samples"]
    except FileNotFoundError:
        baseline_exists = False
        baseline_date = None
        baseline_samples = 0

    # get the most recent prediction timestamp
    recent = get_recent_predictions(limit=1)
    last_prediction_at = recent[0]["timestamp"] if recent else None

    return {
        "total_predictions_logged": total_predictions,
        "baseline_exists": baseline_exists,
        "baseline_generated_at": baseline_date,
        "baseline_samples": baseline_samples,
        "last_prediction_at": last_prediction_at,
        "minimum_for_drift_check": 30,
        "ready_for_drift_check": total_predictions >= 30 and baseline_exists,
    }


# ============================================================
# 8. CLI ENTRY POINT
# ============================================================

def _generate_baseline_from_training():
    """
    Generate baseline by loading the test set and running predictions.

    This loads the saved model and test data from Phase 5,
    gets probabilities for every test sample, and saves
    the distribution as the baseline.
    """
    import joblib
    import pandas as pd

    config = load_config()

    logger.info("=" * 60)
    logger.info("GENERATING PREDICTION BASELINE FROM TRAINING DATA")
    logger.info("=" * 60)

    # load the best model
    model_path = resolve_path(config["api"]["model_path"])
    logger.info(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # load test set features
    preprocess_cfg = config["preprocessing"]
    test_path = resolve_path(preprocess_cfg["test_path"])
    logger.info(f"Loading test data from: {test_path}")
    X_test = pd.read_csv(test_path).values

    # get probabilities for every test sample
    logger.info(f"Running predictions on {len(X_test)} test samples...")
    probabilities = model.predict_proba(X_test)[:, 1]

    # generate and save baseline
    baseline = generate_baseline(probabilities)

    logger.info("=" * 60)
    logger.info("BASELINE GENERATION COMPLETE")
    logger.info(f"  Samples: {baseline['metadata']['n_samples']}")
    logger.info(f"  Mean probability: {baseline['statistics']['mean_probability']:.6f}")
    logger.info(f"  Saved to: models/prediction_baseline.json")
    logger.info("=" * 60)

    return baseline


def _run_drift_check():
    """
    Run a drift check from the command line.
    Loads baseline, fetches recent predictions, calculates PSI.
    """
    config = load_config()
    monitoring_cfg = config.get("monitoring", {})
    days = monitoring_cfg.get("drift_check_interval_days", 7)

    logger.info("=" * 60)
    logger.info("RUNNING PREDICTION DRIFT CHECK")
    logger.info("=" * 60)

    report = generate_drift_report(days=days)

    if report["status"] == "ERROR":
        logger.error(f"Drift check failed: {report['message']}")
        print(f"\nERROR: {report['message']}")
        print(f"Recommendation: {report['recommendation']}")
        return report

    if report["status"] == "INSUFFICIENT_DATA":
        logger.warning(f"Not enough data: {report['message']}")
        print(f"\nINSUFFICIENT DATA: {report['message']}")
        print(f"Recommendation: {report['recommendation']}")
        return report

    # print results to console
    drift = report["drift_analysis"]
    print(f"\n{'=' * 50}")
    print(f"  PREDICTION DRIFT REPORT")
    print(f"{'=' * 50}")
    print(f"  PSI Score:      {drift['psi_score']:.4f}")
    print(f"  Verdict:        {drift['verdict']}")
    print(f"  Threshold:      {drift['threshold']}")
    print(f"  Recommendation: {drift['recommendation']}")
    print(f"{'=' * 50}")
    print(f"  Baseline samples: {report['baseline']['n_samples']}")
    print(f"  Recent samples:   {report['recent']['n_samples']}")
    print(f"  Mean shift:       {report['comparison']['mean_shift']:+.6f}")
    print(f"{'=' * 50}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 11: Prediction Monitoring — generate baseline or check drift"
    )
    parser.add_argument(
        "--generate-baseline",
        action="store_true",
        help="Generate baseline distribution from training data test set"
    )
    parser.add_argument(
        "--check-drift",
        action="store_true",
        help="Run drift check against the saved baseline"
    )

    args = parser.parse_args()

    if args.generate_baseline:
        _generate_baseline_from_training()
    elif args.check_drift:
        _run_drift_check()
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python -m src.monitoring.prediction_monitor --generate-baseline")
        print("  python -m src.monitoring.prediction_monitor --check-drift")
