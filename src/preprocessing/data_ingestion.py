"""
Phase 2: Data Ingestion — THE VAULT
====================================
Downloads the Kaggle Credit Card Fraud Detection dataset and stores it
safely in the raw data vault. Simulates GDPR-compliant handling by
generating a data manifest and privacy audit log.

This phase is fully independent:
  - Reads: config.yaml (for paths and dataset info)
  - Produces: data/raw/creditcard.csv (NEVER overwritten by later phases)
              data/raw/data_manifest.json (schema, stats, hash for integrity)
              data/raw/gdpr_privacy_log.json (simulated GDPR audit trail)

Usage:
    python -m src.preprocessing.data_ingestion

Requires:
    - kaggle API credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME + KAGGLE_KEY env vars)
    - OR manual download: place creditcard.csv into data/raw/ and re-run with --skip-download
"""

import sys
import os
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime, timezone

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
)

PHASE_NAME = "Phase 2: Data Ingestion"
logger = get_logger(__name__)


# ===========================================================================
# 1. DOWNLOAD FUNCTIONS
# ===========================================================================

def download_from_kaggle(dataset_name: str, output_dir: Path) -> Path:
    """
    Download dataset using the Kaggle API.
    Falls back to manual instructions if kaggle is not installed or
    credentials are missing.

    Args:
        dataset_name: Kaggle dataset identifier (e.g. 'mlg-ulb/creditcardfraud')
        output_dir: Directory to save the downloaded file

    Returns:
        Path to the downloaded CSV file
    """
    logger.info(f"Attempting Kaggle API download: {dataset_name}")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        logger.info("Kaggle API authenticated successfully")

        # Download and unzip
        api.dataset_download_files(
            dataset_name,
            path=str(output_dir),
            unzip=True,
        )
        logger.info(f"Dataset downloaded and extracted to {output_dir}")

        # Find the CSV (the dataset contains creditcard.csv)
        csv_path = output_dir / "creditcard.csv"
        if csv_path.exists():
            return csv_path

        # If exact name not found, look for any CSV
        csv_files = list(output_dir.glob("*.csv"))
        if csv_files:
            logger.info(f"Found CSV: {csv_files[0].name}")
            return csv_files[0]

        raise FileNotFoundError("No CSV file found after extraction")

    except ImportError:
        logger.warning("kaggle package not installed. Install with: pip install kaggle")
        raise
    except Exception as e:
        logger.warning(f"Kaggle API download failed: {e}")
        raise


def download_from_url(output_dir: Path) -> Path:
    """
    Fallback: download directly via HTTP if Kaggle API is unavailable.
    Uses the openml mirror of the dataset.

    Args:
        output_dir: Directory to save the downloaded file

    Returns:
        Path to the downloaded CSV file
    """
    import urllib.request
    import zipfile
    import io

    url = "https://www.openml.org/data/get_csv/1673544/phpKo8OWT"
    csv_path = output_dir / "creditcard.csv"

    logger.info(f"Attempting direct download from OpenML mirror...")

    try:
        urllib.request.urlretrieve(url, str(csv_path))
        logger.info(f"Downloaded to {csv_path}")
        return csv_path
    except Exception as e:
        logger.warning(f"Direct download failed: {e}")
        raise


def ensure_dataset(config: dict, skip_download: bool = False) -> Path:
    """
    Ensure the raw dataset exists. Tries multiple download methods.
    If skip_download is True, only checks if the file already exists.

    Args:
        config: Loaded config.yaml dictionary
        skip_download: If True, skip download and just verify file exists

    Returns:
        Path to the raw CSV file

    Raises:
        FileNotFoundError: If file cannot be obtained
    """
    raw_path = resolve_path(config["data"]["raw_path"])
    raw_dir = raw_path.parent
    raw_dir.mkdir(parents=True, exist_ok=True)

    # If file already exists, use it
    if raw_path.exists():
        file_size_mb = raw_path.stat().st_size / (1024 * 1024)
        logger.info(f"Raw dataset already exists: {raw_path} ({file_size_mb:.1f} MB)")
        return raw_path

    if skip_download:
        raise FileNotFoundError(
            f"Dataset not found at {raw_path}. "
            f"Please download creditcard.csv from "
            f"https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            f"and place it in {raw_dir}/"
        )

    # Try Kaggle API first
    dataset_name = config["data"].get("dataset_name", "mlg-ulb/creditcardfraud")
    try:
        return download_from_kaggle(dataset_name, raw_dir)
    except Exception as e:
        logger.warning(f"Kaggle method failed: {e}")

    # Try direct URL fallback
    try:
        return download_from_url(raw_dir)
    except Exception as e:
        logger.warning(f"URL method failed: {e}")

    # All methods failed — give clear manual instructions
    raise FileNotFoundError(
        "\n" + "=" * 60 + "\n"
        "MANUAL DOWNLOAD REQUIRED\n"
        "=" * 60 + "\n"
        "Could not download automatically. Please:\n\n"
        "1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
        "2. Click 'Download' (you may need a free Kaggle account)\n"
        "3. Extract the ZIP file\n"
        f"4. Place 'creditcard.csv' into: {raw_dir}/\n"
        "5. Re-run this script with: python -m src.preprocessing.data_ingestion --skip-download\n"
        "=" * 60
    )


# ===========================================================================
# 2. DATA VALIDATION
# ===========================================================================

def compute_file_hash(filepath: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file for integrity verification."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(csv_path: Path) -> dict:
    """
    Validate the downloaded dataset meets expected characteristics.
    Does NOT use pandas — intentionally lightweight to avoid heavy
    dependencies at the ingestion stage.

    Args:
        csv_path: Path to the raw CSV file

    Returns:
        Dictionary with validation results and basic statistics

    Raises:
        ValueError: If the dataset fails critical validation checks
    """
    logger.info("Validating dataset integrity...")

    # --- Basic file checks ---
    file_size_bytes = csv_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    if file_size_mb < 50:
        raise ValueError(
            f"Dataset too small ({file_size_mb:.1f} MB). "
            f"Expected ~150 MB for the Credit Card Fraud dataset. "
            f"The file may be corrupted or incomplete."
        )

    # --- Read header and count rows ---
    with open(csv_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
        columns = [col.strip('"').strip() for col in header_line.split(",")]
        row_count = sum(1 for _ in f)  # Count remaining lines

    logger.info(f"  Columns: {len(columns)}")
    logger.info(f"  Rows: {row_count:,}")
    logger.info(f"  File size: {file_size_mb:.1f} MB")

    # --- Validate expected schema ---
    expected_columns_subset = {"Time", "Amount", "Class"}
    actual_columns_set = set(columns)

    missing_critical = expected_columns_subset - actual_columns_set
    if missing_critical:
        raise ValueError(
            f"Dataset is missing critical columns: {missing_critical}. "
            f"Found columns: {columns[:5]}... "
            f"Ensure this is the Kaggle Credit Card Fraud dataset."
        )

    # Check for PCA features (V1 through V28)
    pca_features = [c for c in columns if c.startswith("V") and c[1:].isdigit()]
    if len(pca_features) < 20:
        logger.warning(
            f"Expected 28 PCA features (V1-V28), found {len(pca_features)}. "
            f"Dataset may be a different version."
        )

    # --- Row count sanity check ---
    if row_count < 200000:
        logger.warning(
            f"Expected ~284,807 rows, found {row_count:,}. "
            f"Dataset may be truncated."
        )

    # --- Compute integrity hash ---
    file_hash = compute_file_hash(csv_path)
    logger.info(f"  SHA-256: {file_hash[:16]}...")

    # --- Quick fraud ratio check (read Class column) ---
    fraud_count = 0
    class_col_idx = columns.index("Class")
    with open(csv_path, "r", encoding="utf-8") as f:
        f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) > class_col_idx:
                val = parts[class_col_idx].strip().strip('"')
                if val == "1":
                    fraud_count += 1

    fraud_ratio = fraud_count / row_count if row_count > 0 else 0
    logger.info(f"  Fraud transactions: {fraud_count:,} ({fraud_ratio:.4%})")

    if fraud_ratio > 0.05:
        logger.warning(
            f"Fraud ratio ({fraud_ratio:.2%}) is unusually high. "
            f"Expected ~0.17% for this dataset."
        )

    validation_result = {
        "file_path": str(csv_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "sha256_hash": file_hash,
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "pca_features_found": len(pca_features),
        "fraud_count": fraud_count,
        "legitimate_count": row_count - fraud_count,
        "fraud_ratio": round(fraud_ratio, 6),
        "validation_passed": True,
        "validated_at": datetime.now(timezone.utc).isoformat(),
    }

    logger.info("Dataset validation PASSED")
    return validation_result


# ===========================================================================
# 3. GDPR COMPLIANCE SIMULATION
# ===========================================================================

def generate_gdpr_privacy_log(csv_path: Path, validation: dict) -> dict:
    """
    Simulate GDPR-compliant data handling by generating a privacy audit log.
    In a real system, this would track data access, consent, and retention.

    This demonstrates awareness of:
      - Article 5(1)(b): Purpose limitation
      - Article 5(1)(c): Data minimization
      - Article 5(1)(e): Storage limitation
      - Article 5(1)(f): Integrity and confidentiality
      - Article 30: Records of processing activities

    Args:
        csv_path: Path to the raw dataset
        validation: Validation results dictionary

    Returns:
        GDPR privacy log dictionary
    """
    logger.info("Generating GDPR privacy audit log...")

    privacy_log = {
        "processing_activity": "Fraud Detection Model Training",
        "legal_basis": "Legitimate interest (fraud prevention) — Article 6(1)(f)",
        "data_controller": "YOUR_ORGANIZATION_NAME",
        "data_processor": "ML Pipeline (automated)",
        "record_created_at": datetime.now(timezone.utc).isoformat(),

        "data_source": {
            "origin": "Kaggle Public Dataset (mlg-ulb/creditcardfraud)",
            "description": "Credit card transactions from European cardholders, September 2013",
            "original_collectors": "ULB Machine Learning Group",
            "anonymization_method": "PCA transformation applied to original features by data provider",
            "personal_data_present": False,
            "pca_note": (
                "All features V1-V28 are PCA-transformed. Original feature names and values "
                "are not recoverable. Only 'Time' (seconds from first transaction) and "
                "'Amount' (transaction amount) are in original form. No names, card numbers, "
                "addresses, or other PII are present in this dataset."
            ),
        },

        "data_minimization_assessment": {
            "article_reference": "Article 5(1)(c)",
            "status": "COMPLIANT",
            "justification": (
                "Dataset contains only transaction-level features necessary for fraud "
                "pattern detection. PCA transformation by the data provider ensures no "
                "excess personal data is retained. No demographic features (age, gender, "
                "ethnicity, location) are present, eliminating direct discrimination risk."
            ),
            "columns_retained": validation["column_count"],
            "columns_description": "Time, V1-V28 (PCA), Amount, Class (label)",
        },

        "purpose_limitation": {
            "article_reference": "Article 5(1)(b)",
            "stated_purpose": "Training and evaluating a fraud detection model",
            "secondary_use_prohibited": True,
            "note": "Data shall not be used for customer profiling, marketing, or credit scoring.",
        },

        "storage_limitation": {
            "article_reference": "Article 5(1)(e)",
            "raw_data_location": str(csv_path),
            "retention_policy": "Raw data retained for model reproducibility. In production, "
                                "raw data would be subject to retention schedules.",
            "deletion_procedure": "Delete data/raw/ directory contents when no longer needed.",
        },

        "integrity_and_confidentiality": {
            "article_reference": "Article 5(1)(f)",
            "file_hash_algorithm": "SHA-256",
            "file_hash": validation["sha256_hash"],
            "access_control": ".gitignore prevents raw data from being committed to version control",
            "encryption_at_rest": "Not applied (simulation environment). "
                                  "In production: AES-256 encryption required.",
        },

        "data_protection_impact_assessment": {
            "risk_level": "LOW",
            "justification": (
                "Dataset is pre-anonymized via PCA by the original data provider. "
                "No re-identification risk. No special category data (Article 9). "
                "No cross-border transfer concerns (public dataset)."
            ),
        },

        "automated_decision_making": {
            "article_reference": "Article 22",
            "applies": True,
            "safeguards": [
                "SHAP values provide per-transaction explanations (Phase 7)",
                "Human-in-the-loop override available in dashboard (Phase 9-10)",
                "Model decisions are advisory, not final — human investigator reviews",
                "Stress testing validates model robustness (Phase 6)",
            ],
        },

        "data_subject_rights_simulation": {
            "right_to_explanation": "Provided via SHAP waterfall plots and plain-English summaries",
            "right_to_human_review": "Dashboard includes human override capability",
            "right_to_object": "Feedback mechanism allows flagging incorrect decisions",
            "note": "In production, formal DSAR (Data Subject Access Request) process required.",
        },
    }

    logger.info("GDPR privacy audit log generated")
    return privacy_log


# ===========================================================================
# 4. DATA MANIFEST GENERATION
# ===========================================================================

def generate_data_manifest(csv_path: Path, validation: dict) -> dict:
    """
    Create a data manifest that serves as the 'contract' between Phase 2
    and all downstream phases. Any phase can read this manifest to understand
    the raw data without loading the full CSV.

    Args:
        csv_path: Path to the raw dataset
        validation: Validation results dictionary

    Returns:
        Data manifest dictionary
    """
    logger.info("Generating data manifest...")

    manifest = {
        "manifest_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "Phase 2: Data Ingestion",

        "dataset": {
            "name": "Credit Card Fraud Detection",
            "source": "Kaggle (mlg-ulb/creditcardfraud)",
            "source_url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
            "description": (
                "Transactions made by European credit cardholders in September 2013. "
                "Contains 284,807 transactions over two days, with 492 frauds (0.172%). "
                "Features V1-V28 are PCA-transformed; Time and Amount are original."
            ),
        },

        "file_info": {
            "path": str(csv_path),
            "format": "CSV",
            "encoding": "utf-8",
            "delimiter": ",",
            "has_header": True,
            "size_bytes": validation["file_size_bytes"],
            "size_mb": validation["file_size_mb"],
            "sha256": validation["sha256_hash"],
        },

        "schema": {
            "total_columns": validation["column_count"],
            "total_rows": validation["row_count"],
            "columns": validation["columns"],
            "feature_types": {
                "Time": "float64 — seconds elapsed from first transaction in dataset",
                "V1_to_V28": "float64 — PCA-transformed features (original names confidential)",
                "Amount": "float64 — transaction amount in original currency",
                "Class": "int (0 = legitimate, 1 = fraud) — target variable",
            },
        },

        "class_distribution": {
            "total_transactions": validation["row_count"],
            "legitimate": validation["legitimate_count"],
            "fraudulent": validation["fraud_count"],
            "fraud_ratio": validation["fraud_ratio"],
            "imbalance_note": (
                "Extreme class imbalance (~0.17% fraud). Standard accuracy is meaningless. "
                "Model evaluation MUST use AUPRC, Precision, and Recall."
            ),
        },

        "data_quality_notes": {
            "missing_values": "None expected in this dataset (pre-cleaned by provider)",
            "duplicates": "To be checked in Phase 3 (Data Engineering)",
            "outliers": "Amount column has extreme outliers; RobustScaler recommended",
            "pca_note": "V1-V28 are already scaled via PCA; only Time and Amount need scaling",
        },

        "downstream_contract": {
            "Phase 3 reads": "data/raw/creditcard.csv",
            "Phase 3 must NOT": "Modify or overwrite raw data",
            "Phase 3 outputs to": "data/processed/",
            "integrity_check": "Phase 3 should verify SHA-256 hash before processing",
        },
    }

    logger.info("Data manifest generated")
    return manifest


# ===========================================================================
# 5. MAIN EXECUTION
# ===========================================================================

def run_phase2(skip_download: bool = False):
    """
    Execute the complete Phase 2 pipeline.

    Args:
        skip_download: If True, skip download attempts and require
                       the file to already exist in data/raw/
    """
    log_phase_start(PHASE_NAME)

    try:
        config = load_config()

        # --- Step 1: Ensure dataset exists ---
        logger.info("Step 1/4: Ensuring dataset is available...")
        csv_path = ensure_dataset(config, skip_download=skip_download)

        # --- Step 2: Validate dataset ---
        logger.info("Step 2/4: Validating dataset integrity...")
        validation = validate_dataset(csv_path)

        # --- Step 3: Generate data manifest ---
        logger.info("Step 3/4: Generating data manifest...")
        manifest = generate_data_manifest(csv_path, validation)
        manifest_path = csv_path.parent / "data_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Data manifest saved to {manifest_path}")

        # --- Step 4: Generate GDPR privacy log ---
        logger.info("Step 4/4: Generating GDPR privacy audit log...")
        privacy_log = generate_gdpr_privacy_log(csv_path, validation)
        privacy_path = csv_path.parent / "gdpr_privacy_log.json"
        with open(privacy_path, "w", encoding="utf-8") as f:
            json.dump(privacy_log, f, indent=2)
        logger.info(f"GDPR privacy log saved to {privacy_path}")

        # --- Summary ---
        logger.info("")
        logger.info("=" * 50)
        logger.info("PHASE 2 SUMMARY")
        logger.info("=" * 50)
        logger.info(f"  Dataset: {csv_path.name}")
        logger.info(f"  Rows: {validation['row_count']:,}")
        logger.info(f"  Columns: {validation['column_count']}")
        logger.info(f"  Fraud ratio: {validation['fraud_ratio']:.4%}")
        logger.info(f"  File size: {validation['file_size_mb']:.1f} MB")
        logger.info(f"  SHA-256: {validation['sha256_hash'][:16]}...")
        logger.info(f"  Manifest: {manifest_path}")
        logger.info(f"  GDPR Log: {privacy_path}")
        logger.info("=" * 50)

        log_phase_end(PHASE_NAME, status="SUCCESS")
        return validation

    except Exception as e:
        logger.error(f"Phase 2 failed: {e}")
        log_phase_end(PHASE_NAME, status="FAILED", error=str(e))
        raise


# ===========================================================================
# CLI Entry Point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Data Ingestion — Download and validate the fraud dataset"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download; only validate an existing file in data/raw/",
    )
    args = parser.parse_args()

    try:
        run_phase2(skip_download=args.skip_download)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"\nPhase 2 FAILED: {e}")
        sys.exit(1)
