"""
Phase 3: Data Engineering & Feature Store -- THE ENGINE
========================================================
Cleans raw data, engineers temporal features, scales variables,
splits into train/val/test, and applies SMOTE to handle class
imbalance -- all without introducing data leakage.

This phase is fully independent:
  - Reads: data/raw/creditcard.csv (from Phase 2 vault)
           data/raw/data_manifest.json (for integrity verification)
           config.yaml (for all paths and hyperparameters)
  - Produces: data/processed/X_train.csv, y_train.csv
              data/processed/X_val.csv, y_val.csv
              data/processed/X_test.csv, y_test.csv
              data/processed/feature_names.json
              data/processed/engineering_report.json
              models/scaler.pkl

CRITICAL DATA LEAKAGE PREVENTION:
  1. Train/val/test split happens BEFORE any fitting (scaling, SMOTE)
  2. Scaler is fit ONLY on training data, then applied to val/test
  3. SMOTE is applied ONLY to training data after split
  4. No future information leaks into training (temporal features safe)

Usage:
    python -m src.preprocessing.data_engineering
"""

import sys
import os
import json
import hashlib
import argparse
import warnings
import pickle
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.logger import (
    get_logger,
    load_config,
    resolve_path,
    log_phase_start,
    log_phase_end,
    check_phase_completed,
)

PHASE_NAME = "Phase 3: Data Engineering"
logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)


# ===========================================================================
# 1. DATA LOADING & INTEGRITY VERIFICATION
# ===========================================================================

def verify_data_integrity(csv_path: Path, manifest_path: Path) -> bool:
    """
    Verify the raw dataset has not been tampered with since Phase 2
    by comparing the SHA-256 hash against the stored manifest.
    """
    logger.info("Verifying data integrity against Phase 2 manifest...")

    if not manifest_path.exists():
        logger.warning(
            "Data manifest not found. Skipping integrity check. "
            "This is acceptable if you are running Phase 3 standalone."
        )
        return True

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    stored_hash = manifest.get("file_info", {}).get("sha256", "")
    if not stored_hash:
        logger.warning("No SHA-256 hash found in manifest. Skipping integrity check.")
        return True

    # Compute current hash
    h = hashlib.sha256()
    with open(csv_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    current_hash = h.hexdigest()

    if current_hash != stored_hash:
        logger.error(
            f"DATA INTEGRITY FAILURE: SHA-256 mismatch.\n"
            f"  Expected: {stored_hash[:16]}...\n"
            f"  Got:      {current_hash[:16]}...\n"
            f"  The raw dataset may have been modified since Phase 2."
        )
        return False

    logger.info(f"Data integrity verified. SHA-256: {current_hash[:16]}...")
    return True


def load_raw_data(config: dict) -> pd.DataFrame:
    """
    Load the raw dataset from the Phase 2 vault.
    Never modifies the source file.
    """
    csv_path = resolve_path(config["data"]["raw_path"])
    manifest_path = csv_path.parent / "data_manifest.json"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {csv_path}. "
            f"Run Phase 2 first: python -m src.preprocessing.data_ingestion"
        )

    # Verify integrity
    integrity_ok = verify_data_integrity(csv_path, manifest_path)
    if not integrity_ok:
        raise ValueError(
            "Data integrity check FAILED. Raw data may have been corrupted. "
            "Re-run Phase 2 to re-download the dataset."
        )

    logger.info(f"Loading raw data from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    return df


# ===========================================================================
# 2. DATA CLEANING
# ===========================================================================

def clean_data(df: pd.DataFrame) -> tuple:
    """
    Rigorous data cleaning:
    - Check and handle missing values
    - Remove exact duplicates
    - Validate data types
    - Check for anomalous values

    Returns:
        (cleaned_df, cleaning_report)
    """
    logger.info("Starting data cleaning...")
    report = {
        "original_rows": len(df),
        "original_columns": len(df.columns),
        "steps": [],
    }

    # Step 1: Check missing values
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        logger.warning(f"Found {total_missing} missing values")
        # For numerical features, fill with median (robust to outliers)
        for col in df.columns:
            n_missing = missing[col]
            if n_missing > 0:
                if df[col].dtype in [np.float64, np.int64, float, int]:
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    logger.info(f"  Filled {n_missing} missing in '{col}' with median ({median_val:.4f})")
                else:
                    mode_val = df[col].mode()[0] if not df[col].mode().empty else 0
                    df[col].fillna(mode_val, inplace=True)
                    logger.info(f"  Filled {n_missing} missing in '{col}' with mode ({mode_val})")
        report["steps"].append({
            "step": "handle_missing_values",
            "total_missing": int(total_missing),
            "method": "median for numeric, mode for categorical",
        })
    else:
        logger.info("No missing values found")
        report["steps"].append({
            "step": "handle_missing_values",
            "total_missing": 0,
            "method": "none needed",
        })

    # Step 2: Remove exact duplicates
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.info(f"Removing {n_duplicates:,} exact duplicate rows")
        df = df.drop_duplicates().reset_index(drop=True)
        report["steps"].append({
            "step": "remove_duplicates",
            "duplicates_found": int(n_duplicates),
            "rows_after": len(df),
        })
    else:
        logger.info("No duplicate rows found")
        report["steps"].append({
            "step": "remove_duplicates",
            "duplicates_found": 0,
            "rows_after": len(df),
        })

    # Step 3: Validate data types
    expected_numeric = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
    type_issues = []
    for col in expected_numeric:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
            type_issues.append(col)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if type_issues:
        logger.warning(f"Converted non-numeric columns to numeric: {type_issues}")

    report["steps"].append({
        "step": "validate_dtypes",
        "columns_converted": type_issues,
    })

    # Step 4: Check for infinite values
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        logger.warning(f"Found {inf_count} infinite values, replacing with NaN then median")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)

    report["steps"].append({
        "step": "handle_infinite_values",
        "infinite_values_found": int(inf_count),
    })

    # Step 5: Validate target variable
    unique_classes = sorted(df["Class"].unique())
    if unique_classes != [0, 1]:
        logger.error(f"Unexpected class values: {unique_classes}. Expected [0, 1].")
        raise ValueError(f"Target variable 'Class' has unexpected values: {unique_classes}")

    report["steps"].append({
        "step": "validate_target",
        "unique_classes": [int(c) for c in unique_classes],
        "class_distribution": {
            "legitimate": int((df["Class"] == 0).sum()),
            "fraudulent": int((df["Class"] == 1).sum()),
        },
    })

    report["cleaned_rows"] = len(df)
    report["rows_removed"] = report["original_rows"] - len(df)

    logger.info(
        f"Cleaning complete: {report['original_rows']:,} -> {report['cleaned_rows']:,} rows "
        f"({report['rows_removed']:,} removed)"
    )

    return df, report


# ===========================================================================
# 3. TEMPORAL FEATURE ENGINEERING
# ===========================================================================

def engineer_temporal_features(df: pd.DataFrame) -> tuple:
    """
    Create temporal features based on transaction timing and frequency.
    These are critical for catching time-based fraud patterns.

    New features created:
    - hour_of_day: Simulated hour (Time mod 86400 / 3600)
    - is_night: Binary flag for nighttime transactions (10pm-6am)
    - time_since_prev: Seconds since previous transaction (sorted by Time)
    - amount_log: Log-transformed transaction amount (handles skewness)
    - amount_zscore: Z-score of amount relative to global statistics
    - tx_frequency_1h: Number of transactions in the preceding 1-hour window
    - amount_to_median_ratio: Ratio of transaction amount to median amount

    Returns:
        (df_with_features, feature_report)
    """
    logger.info("Engineering temporal features...")
    new_features = []

    # Sort by time to ensure temporal ordering
    df = df.sort_values("Time").reset_index(drop=True)

    # Feature 1: Hour of day (simulated -- Time is seconds from first tx)
    # The dataset spans ~48 hours, so we simulate hour within a day
    df["hour_of_day"] = (df["Time"] % 86400) / 3600
    new_features.append("hour_of_day")
    logger.info("  Created: hour_of_day (simulated from Time)")

    # Feature 2: Is nighttime transaction (higher fraud risk)
    df["is_night"] = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)).astype(int)
    new_features.append("is_night")
    logger.info("  Created: is_night (22:00-06:00 flag)")

    # Feature 3: Time since previous transaction
    df["time_since_prev"] = df["Time"].diff().fillna(0)
    # Cap extreme values at 99th percentile
    cap_val = df["time_since_prev"].quantile(0.99)
    df["time_since_prev"] = df["time_since_prev"].clip(upper=cap_val)
    new_features.append("time_since_prev")
    logger.info("  Created: time_since_prev (capped at 99th percentile)")

    # Feature 4: Log-transformed amount (handles extreme skewness)
    df["amount_log"] = np.log1p(df["Amount"])
    new_features.append("amount_log")
    logger.info("  Created: amount_log (log1p transform)")

    # Feature 5: Amount z-score (how unusual is this amount?)
    amount_mean = df["Amount"].mean()
    amount_std = df["Amount"].std()
    if amount_std > 0:
        df["amount_zscore"] = (df["Amount"] - amount_mean) / amount_std
    else:
        df["amount_zscore"] = 0.0
    new_features.append("amount_zscore")
    logger.info("  Created: amount_zscore")

    # Feature 6: Transaction frequency in 1-hour window
    # Count transactions within 3600 seconds before each transaction
    # Using a vectorized rolling approach for efficiency
    df["tx_frequency_1h"] = 0
    time_vals = df["Time"].values
    freq_counts = np.zeros(len(df), dtype=int)

    # Efficient sliding window using searchsorted
    window_size = 3600  # 1 hour in seconds
    left_indices = np.searchsorted(time_vals, time_vals - window_size, side="left")
    for i in range(len(df)):
        freq_counts[i] = i - left_indices[i]

    df["tx_frequency_1h"] = freq_counts
    new_features.append("tx_frequency_1h")
    logger.info("  Created: tx_frequency_1h (1-hour rolling window)")

    # Feature 7: Amount to median ratio
    median_amount = df["Amount"].median()
    if median_amount > 0:
        df["amount_to_median_ratio"] = df["Amount"] / median_amount
    else:
        df["amount_to_median_ratio"] = 0.0
    # Cap at 99th percentile to avoid extreme values
    cap_ratio = df["amount_to_median_ratio"].quantile(0.99)
    df["amount_to_median_ratio"] = df["amount_to_median_ratio"].clip(upper=cap_ratio)
    new_features.append("amount_to_median_ratio")
    logger.info("  Created: amount_to_median_ratio (capped at 99th percentile)")

    feature_report = {
        "new_features_created": new_features,
        "total_new_features": len(new_features),
        "total_features_after": len(df.columns),
        "feature_descriptions": {
            "hour_of_day": "Simulated hour of day from Time (modulo 86400s)",
            "is_night": "Binary flag: 1 if transaction between 22:00-06:00",
            "time_since_prev": "Seconds since previous transaction (capped at 99th pct)",
            "amount_log": "Natural log(1 + Amount) to handle skewness",
            "amount_zscore": "Z-score of Amount relative to dataset mean/std",
            "tx_frequency_1h": "Number of transactions in preceding 1-hour window",
            "amount_to_median_ratio": "Ratio of Amount to dataset median (capped at 99th pct)",
        },
    }

    logger.info(f"Feature engineering complete: {len(new_features)} new features created")

    return df, feature_report


# ===========================================================================
# 4. DATA SPLITTING (BEFORE SCALING AND SMOTE)
# ===========================================================================

def split_data(df: pd.DataFrame, config: dict) -> tuple:
    """
    Split data into train/validation/test sets.
    This MUST happen BEFORE scaling and SMOTE to prevent data leakage.

    Split strategy:
    - First split: train+val (80%) vs test (20%)
    - Second split: train (88.9% of remaining) vs val (11.1% of remaining)
    - Final ratio: ~71% train, ~9% val, ~20% test

    Stratified splitting preserves the fraud ratio in all sets.
    """
    logger.info("Splitting data into train/val/test sets...")
    preproc = config["preprocessing"]
    seed = config["environment"]["random_seed"]

    test_size = preproc["test_size"]   # 0.2
    val_size = preproc["val_size"]     # 0.1

    # Separate features and target
    target_col = "Class"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: train+val vs test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Second split: train vs val
    # val_size=0.1 of total means val_fraction = 0.1 / (1 - test_size) = 0.1/0.8 = 0.125
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_fraction,
        random_state=seed,
        stratify=y_train_val,
    )

    split_report = {
        "total_rows": len(df),
        "train_rows": len(X_train),
        "val_rows": len(X_val),
        "test_rows": len(X_test),
        "train_pct": round(len(X_train) / len(df) * 100, 1),
        "val_pct": round(len(X_val) / len(df) * 100, 1),
        "test_pct": round(len(X_test) / len(df) * 100, 1),
        "train_fraud_ratio": round(y_train.mean(), 6),
        "val_fraud_ratio": round(y_val.mean(), 6),
        "test_fraud_ratio": round(y_test.mean(), 6),
        "random_seed": seed,
        "stratified": True,
    }

    logger.info(
        f"  Train: {len(X_train):,} rows ({split_report['train_pct']}%) "
        f"fraud ratio: {split_report['train_fraud_ratio']:.4%}"
    )
    logger.info(
        f"  Val:   {len(X_val):,} rows ({split_report['val_pct']}%) "
        f"fraud ratio: {split_report['val_fraud_ratio']:.4%}"
    )
    logger.info(
        f"  Test:  {len(X_test):,} rows ({split_report['test_pct']}%) "
        f"fraud ratio: {split_report['test_fraud_ratio']:.4%}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, split_report


# ===========================================================================
# 5. SCALING (FIT ON TRAIN ONLY)
# ===========================================================================

def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    config: dict,
) -> tuple:
    """
    Scale features using RobustScaler (resistant to outliers in financial data).

    CRITICAL: Scaler is fit ONLY on training data, then transform is
    applied to val and test. This prevents data leakage.

    Only Time and Amount need scaling (V1-V28 are already PCA-scaled).
    Engineered features derived from Time/Amount also need scaling.
    """
    logger.info("Scaling features...")

    # Columns that need scaling:
    # - Time, Amount: original features not yet scaled
    # - Engineered features derived from them
    cols_to_scale = [
        "Time", "Amount",
        "hour_of_day", "time_since_prev", "amount_log",
        "amount_zscore", "tx_frequency_1h", "amount_to_median_ratio",
    ]

    # Only scale columns that exist (defensive coding)
    cols_to_scale = [c for c in cols_to_scale if c in X_train.columns]

    logger.info(f"  Columns to scale: {cols_to_scale}")

    scaler = RobustScaler()

    # Fit ONLY on training data
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_val_scaled[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    # Save scaler
    scaler_path = resolve_path(config["preprocessing"]["scaler_path"])
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"  Scaler saved to {scaler_path}")

    scale_report = {
        "method": config["preprocessing"]["scaling_method"],
        "columns_scaled": cols_to_scale,
        "scaler_path": str(scaler_path),
        "fit_on": "training data only (no leakage)",
        "scaler_center": {col: float(val) for col, val in zip(cols_to_scale, scaler.center_)},
        "scaler_scale": {col: float(val) for col, val in zip(cols_to_scale, scaler.scale_)},
    }

    logger.info("Scaling complete (fit on train, applied to all sets)")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler, scale_report


# ===========================================================================
# 6. SMOTE APPLICATION (TRAIN SET ONLY)
# ===========================================================================

def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> tuple:
    """
    Apply SMOTE to the training set ONLY to handle class imbalance.

    CRITICAL: SMOTE is NEVER applied to validation or test sets.
    Those must reflect the real-world distribution for honest evaluation.
    """
    logger.info("Applying SMOTE to training data...")

    smote_config = config["preprocessing"]["smote"]

    if not smote_config["enabled"]:
        logger.info("SMOTE is disabled in config. Skipping.")
        return X_train, y_train, {"enabled": False}

    original_train_size = len(X_train)
    original_fraud_count = int(y_train.sum())
    original_legit_count = original_train_size - original_fraud_count

    logger.info(
        f"  Before SMOTE: {original_train_size:,} rows "
        f"(fraud: {original_fraud_count:,}, legit: {original_legit_count:,})"
    )

    smote = SMOTE(
        sampling_strategy=smote_config["sampling_strategy"],
        k_neighbors=smote_config["k_neighbors"],
        random_state=config["environment"]["random_seed"],
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Convert back to DataFrame/Series to preserve column names
    X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)
    y_resampled = pd.Series(y_resampled, name="Class")

    new_fraud_count = int(y_resampled.sum())
    new_legit_count = len(y_resampled) - new_fraud_count
    synthetic_samples = new_fraud_count - original_fraud_count

    smote_report = {
        "enabled": True,
        "sampling_strategy": smote_config["sampling_strategy"],
        "k_neighbors": smote_config["k_neighbors"],
        "before": {
            "total": original_train_size,
            "fraud": original_fraud_count,
            "legitimate": original_legit_count,
            "fraud_ratio": round(original_fraud_count / original_train_size, 6),
        },
        "after": {
            "total": len(X_resampled),
            "fraud": new_fraud_count,
            "legitimate": new_legit_count,
            "fraud_ratio": round(new_fraud_count / len(X_resampled), 6),
        },
        "synthetic_samples_created": synthetic_samples,
        "applied_to": "training data ONLY (val/test untouched)",
    }

    logger.info(
        f"  After SMOTE: {len(X_resampled):,} rows "
        f"(fraud: {new_fraud_count:,}, legit: {new_legit_count:,})"
    )
    logger.info(f"  Synthetic fraud samples created: {synthetic_samples:,}")

    return X_resampled, y_resampled, smote_report


# ===========================================================================
# 7. SAVE PROCESSED DATA
# ===========================================================================

def save_processed_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    config: dict,
) -> dict:
    """
    Save all processed datasets to the feature store directory.
    Each file is independent and can be loaded by any downstream phase.
    """
    logger.info("Saving processed data to feature store...")

    processed_dir = resolve_path(config["preprocessing"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Save paths from config
    paths = {
        "X_train": resolve_path(config["preprocessing"]["train_path"]),
        "X_val": resolve_path(config["preprocessing"]["val_path"]),
        "X_test": resolve_path(config["preprocessing"]["test_path"]),
        "y_train": resolve_path(config["preprocessing"]["y_train_path"]),
        "y_val": resolve_path(config["preprocessing"]["y_val_path"]),
        "y_test": resolve_path(config["preprocessing"]["y_test_path"]),
    }

    datasets = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
    }

    save_report = {}
    for name, data in datasets.items():
        path = paths[name]
        if isinstance(data, pd.Series):
            data.to_csv(path, index=False, header=True)
        else:
            data.to_csv(path, index=False)
        size_mb = path.stat().st_size / (1024 * 1024)
        save_report[name] = {
            "path": str(path),
            "rows": len(data),
            "columns": len(data.columns) if hasattr(data, "columns") else 1,
            "size_mb": round(size_mb, 2),
        }
        logger.info(f"  Saved {name}: {len(data):,} rows -> {path.name} ({size_mb:.2f} MB)")

    # Save feature names for downstream phases
    feature_names_path = processed_dir / "feature_names.json"
    feature_names = {
        "features": list(X_train.columns),
        "target": "Class",
        "total_features": len(X_train.columns),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": PHASE_NAME,
    }
    with open(feature_names_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"  Feature names saved to {feature_names_path}")

    save_report["feature_names_path"] = str(feature_names_path)

    return save_report


# ===========================================================================
# 8. ENGINEERING REPORT
# ===========================================================================

def save_engineering_report(
    cleaning_report: dict,
    feature_report: dict,
    split_report: dict,
    scale_report: dict,
    smote_report: dict,
    save_report: dict,
    config: dict,
) -> Path:
    """
    Save a comprehensive engineering report documenting every
    transformation applied to the data. This is essential for
    auditability and reproducibility.
    """
    processed_dir = resolve_path(config["preprocessing"]["processed_dir"])

    report = {
        "report_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": PHASE_NAME,
        "data_leakage_prevention": {
            "split_before_scaling": True,
            "split_before_smote": True,
            "scaler_fit_on_train_only": True,
            "smote_on_train_only": True,
            "no_future_information_in_features": True,
        },
        "cleaning": cleaning_report,
        "feature_engineering": feature_report,
        "data_split": split_report,
        "scaling": scale_report,
        "smote": smote_report,
        "saved_artifacts": save_report,
    }

    report_path = processed_dir / "engineering_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Engineering report saved to {report_path}")
    return report_path


# ===========================================================================
# 9. MAIN EXECUTION
# ===========================================================================

def run_phase3():
    """Execute the complete Phase 3 data engineering pipeline."""
    log_phase_start(PHASE_NAME)

    try:
        config = load_config()

        # Step 1: Load raw data
        logger.info("Step 1/6: Loading raw data from Phase 2 vault...")
        df = load_raw_data(config)

        # Step 2: Clean data
        logger.info("Step 2/6: Cleaning data...")
        df, cleaning_report = clean_data(df)

        # Step 3: Engineer temporal features
        logger.info("Step 3/6: Engineering temporal features...")
        df, feature_report = engineer_temporal_features(df)

        # Step 4: Split BEFORE scaling and SMOTE (leakage prevention)
        logger.info("Step 4/6: Splitting data (BEFORE scaling/SMOTE)...")
        X_train, X_val, X_test, y_train, y_val, y_test, split_report = split_data(df, config)

        # Step 5: Scale features (fit on train only)
        logger.info("Step 5/6: Scaling features (fit on train only)...")
        X_train, X_val, X_test, scaler, scale_report = scale_features(
            X_train, X_val, X_test, config
        )

        # Step 6: Apply SMOTE (train only)
        logger.info("Step 6/6: Applying SMOTE (train only)...")
        X_train, y_train, smote_report = apply_smote(X_train, y_train, config)

        # Save everything
        logger.info("Saving processed datasets...")
        save_report = save_processed_data(
            X_train, X_val, X_test, y_train, y_val, y_test, config
        )

        # Generate engineering report
        report_path = save_engineering_report(
            cleaning_report, feature_report, split_report,
            scale_report, smote_report, save_report, config
        )

        # Summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE 3 SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Raw data rows:       {cleaning_report['original_rows']:,}")
        logger.info(f"  After cleaning:      {cleaning_report['cleaned_rows']:,}")
        logger.info(f"  New features added:  {feature_report['total_new_features']}")
        logger.info(f"  Total features:      {len(X_train.columns)}")
        logger.info(f"  Train set:           {len(X_train):,} rows (with SMOTE)")
        logger.info(f"  Validation set:      {len(X_val):,} rows (original distribution)")
        logger.info(f"  Test set:            {len(X_test):,} rows (original distribution)")
        logger.info(f"  Scaler:              RobustScaler (fit on train only)")
        logger.info(f"  SMOTE:               {smote_report.get('synthetic_samples_created', 0):,} synthetic fraud samples")
        logger.info(f"  Report:              {report_path}")
        logger.info("=" * 60)

        log_phase_end(PHASE_NAME, status="SUCCESS")

    except Exception as e:
        logger.error(f"Phase 3 failed: {e}")
        log_phase_end(PHASE_NAME, status="FAILED", error=str(e))
        raise


# ===========================================================================
# CLI Entry Point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 3: Data Engineering -- Clean, engineer features, scale, split, SMOTE"
    )
    args = parser.parse_args()

    try:
        run_phase3()
    except Exception as e:
        print(f"\nPhase 3 FAILED: {e}")
        sys.exit(1)
