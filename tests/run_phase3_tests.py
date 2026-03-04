"""
Phase 3 Verification Runner -- Run: python tests/run_phase3_tests.py
Works without pytest (for offline environments).

Tests are split into three groups:
  Group A: Always runnable (script existence, config, imports)
  Group B: Require Phase 3 execution (processed data artifacts)
  Group C: Data leakage verification (critical integrity checks)
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import PROJECT_ROOT, load_config, resolve_path

PASS = 0
FAIL = 0
SKIP = 0


def run_test(name, func):
    global PASS, FAIL, SKIP
    try:
        result = func()
        if result == "SKIP":
            print(f"  SKIP  {name}")
            SKIP += 1
        else:
            print(f"  PASS  {name}")
            PASS += 1
    except Exception as e:
        print(f"  FAIL  {name}")
        print(f"        -> {e}")
        FAIL += 1


def _processed_exists() -> bool:
    config = load_config()
    return resolve_path(config["preprocessing"]["train_path"]).exists()


def _get_processed_dir() -> Path:
    config = load_config()
    return resolve_path(config["preprocessing"]["processed_dir"])


# ===========================================================================
# Group A: Script & Config (always runnable)
# ===========================================================================

def test_data_engineering_script_exists():
    script = PROJECT_ROOT / "src" / "preprocessing" / "data_engineering.py"
    assert script.exists(), f"data_engineering.py not found at {script}"
    assert len(script.read_text(encoding="utf-8")) > 2000, "Script too short"

def test_data_engineering_is_importable():
    from src.preprocessing.data_engineering import (
        run_phase3, load_raw_data, clean_data,
        engineer_temporal_features, split_data,
        scale_features, apply_smote, save_processed_data,
        verify_data_integrity,
    )
    assert callable(run_phase3)

def test_config_has_preprocessing_section():
    config = load_config()
    assert "preprocessing" in config, "Missing 'preprocessing' section"
    preproc = config["preprocessing"]
    for key in ["processed_dir", "train_path", "test_path", "val_path",
                 "y_train_path", "y_test_path", "y_val_path",
                 "scaler_path", "test_size", "val_size", "scaling_method", "smote"]:
        assert key in preproc, f"Missing config key: preprocessing.{key}"

def test_config_smote_settings():
    config = load_config()
    smote = config["preprocessing"]["smote"]
    for key in ["enabled", "sampling_strategy", "k_neighbors"]:
        assert key in smote, f"SMOTE config missing '{key}'"

def test_script_documents_leakage_prevention():
    content = (PROJECT_ROOT / "src" / "preprocessing" / "data_engineering.py").read_text(encoding="utf-8").upper()
    assert "LEAKAGE" in content, "Must document data leakage prevention"
    assert "BEFORE" in content, "Must mention splitting BEFORE scaling/SMOTE"

def test_phase_tracking_integration():
    content = (PROJECT_ROOT / "src" / "preprocessing" / "data_engineering.py").read_text(encoding="utf-8")
    assert "log_phase_start" in content, "Must call log_phase_start()"
    assert "log_phase_end" in content, "Must call log_phase_end()"

def test_script_uses_robust_scaler():
    content = (PROJECT_ROOT / "src" / "preprocessing" / "data_engineering.py").read_text(encoding="utf-8")
    assert "RobustScaler" in content, "Must use RobustScaler"

def test_script_creates_temporal_features():
    content = (PROJECT_ROOT / "src" / "preprocessing" / "data_engineering.py").read_text(encoding="utf-8")
    for feature in ["hour_of_day", "is_night", "time_since_prev", "amount_log", "tx_frequency_1h"]:
        assert feature in content, f"Must create temporal feature: {feature}"


# ===========================================================================
# Group B: Processed Data Artifacts (require Phase 3 execution)
# ===========================================================================

def test_processed_train_exists():
    if not _processed_exists():
        return "SKIP"
    import pandas as pd
    config = load_config()
    X_train = pd.read_csv(resolve_path(config["preprocessing"]["train_path"]))
    assert len(X_train) > 0, "X_train is empty"
    assert len(X_train.columns) > 30, f"Too few features: {len(X_train.columns)}"

def test_processed_val_exists():
    if not _processed_exists():
        return "SKIP"
    import pandas as pd
    config = load_config()
    X_val = pd.read_csv(resolve_path(config["preprocessing"]["val_path"]))
    assert len(X_val) > 0, "X_val is empty"

def test_processed_test_exists():
    if not _processed_exists():
        return "SKIP"
    import pandas as pd
    config = load_config()
    X_test = pd.read_csv(resolve_path(config["preprocessing"]["test_path"]))
    assert len(X_test) > 0, "X_test is empty"

def test_scaler_exists():
    if not _processed_exists():
        return "SKIP"
    import pickle
    config = load_config()
    scaler_path = resolve_path(config["preprocessing"]["scaler_path"])
    assert scaler_path.exists(), f"Scaler not found at {scaler_path}"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    assert hasattr(scaler, "transform"), "Not a valid scaler"

def test_feature_names_json_exists():
    if not _processed_exists():
        return "SKIP"
    processed_dir = _get_processed_dir()
    feature_path = processed_dir / "feature_names.json"
    assert feature_path.exists(), "feature_names.json not found"
    content = json.loads(feature_path.read_text(encoding="utf-8"))
    assert "features" in content, "Missing 'features' key"
    assert len(content["features"]) > 30, "Too few features listed"

def test_engineering_report_exists():
    if not _processed_exists():
        return "SKIP"
    processed_dir = _get_processed_dir()
    report_path = processed_dir / "engineering_report.json"
    assert report_path.exists(), "engineering_report.json not found"
    content = json.loads(report_path.read_text(encoding="utf-8"))
    for section in ["data_leakage_prevention", "cleaning", "feature_engineering",
                     "data_split", "scaling", "smote", "saved_artifacts"]:
        assert section in content, f"Report missing section: '{section}'"

def test_temporal_features_in_data():
    if not _processed_exists():
        return "SKIP"
    import pandas as pd
    config = load_config()
    X_train = pd.read_csv(resolve_path(config["preprocessing"]["train_path"]), nrows=5)
    for feat in ["hour_of_day", "is_night", "time_since_prev", "amount_log", "tx_frequency_1h"]:
        assert feat in X_train.columns, f"Missing temporal feature: {feat}"


# ===========================================================================
# Group C: Data Leakage Verification (critical)
# ===========================================================================

def test_no_leakage_val_test_not_smoted():
    if not _processed_exists():
        return "SKIP"
    import pandas as pd
    config = load_config()
    y_val = pd.read_csv(resolve_path(config["preprocessing"]["y_val_path"]))
    y_test = pd.read_csv(resolve_path(config["preprocessing"]["y_test_path"]))
    val_fraud_ratio = y_val.iloc[:, 0].mean()
    test_fraud_ratio = y_test.iloc[:, 0].mean()
    assert val_fraud_ratio < 0.01, f"Val fraud ratio ({val_fraud_ratio:.4%}) too high -- SMOTE leakage?"
    assert test_fraud_ratio < 0.01, f"Test fraud ratio ({test_fraud_ratio:.4%}) too high -- SMOTE leakage?"

def test_smote_applied_to_train():
    if not _processed_exists():
        return "SKIP"
    import pandas as pd
    config = load_config()
    y_train = pd.read_csv(resolve_path(config["preprocessing"]["y_train_path"]))
    train_fraud_ratio = y_train.iloc[:, 0].mean()
    assert train_fraud_ratio > 0.05, f"Train fraud ratio ({train_fraud_ratio:.4%}) too low -- SMOTE not applied?"

def test_feature_consistency_across_splits():
    if not _processed_exists():
        return "SKIP"
    import pandas as pd
    config = load_config()
    X_train = pd.read_csv(resolve_path(config["preprocessing"]["train_path"]), nrows=1)
    X_val = pd.read_csv(resolve_path(config["preprocessing"]["val_path"]), nrows=1)
    X_test = pd.read_csv(resolve_path(config["preprocessing"]["test_path"]), nrows=1)
    train_cols = set(X_train.columns)
    val_cols = set(X_val.columns)
    test_cols = set(X_test.columns)
    assert train_cols == val_cols, f"Train/val column mismatch: {train_cols.symmetric_difference(val_cols)}"
    assert train_cols == test_cols, f"Train/test column mismatch: {train_cols.symmetric_difference(test_cols)}"


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 3 VERIFICATION: Data Engineering & Feature Store")
    print("=" * 60 + "\n")

    if not _processed_exists():
        print("  NOTE: Processed data not found yet.")
        print("        Group B & C tests will be SKIPPED.")
        print("        Run Phase 3 first, then re-run these tests.\n")

    group_a = [
        ("Script exists and is substantial", test_data_engineering_script_exists),
        ("Module is importable", test_data_engineering_is_importable),
        ("Config has preprocessing section", test_config_has_preprocessing_section),
        ("Config has SMOTE settings", test_config_smote_settings),
        ("Script documents leakage prevention", test_script_documents_leakage_prevention),
        ("Phase tracking integration", test_phase_tracking_integration),
        ("Uses RobustScaler", test_script_uses_robust_scaler),
        ("Creates temporal features", test_script_creates_temporal_features),
    ]

    group_b = [
        ("X_train exists and has features", test_processed_train_exists),
        ("X_val exists", test_processed_val_exists),
        ("X_test exists", test_processed_test_exists),
        ("Scaler pickle exists", test_scaler_exists),
        ("Feature names JSON exists", test_feature_names_json_exists),
        ("Engineering report exists", test_engineering_report_exists),
        ("Temporal features in data", test_temporal_features_in_data),
    ]

    group_c = [
        ("NO LEAKAGE: Val/test not SMOTEd", test_no_leakage_val_test_not_smoted),
        ("SMOTE applied to train", test_smote_applied_to_train),
        ("Feature consistency across splits", test_feature_consistency_across_splits),
    ]

    print("--- Group A: Script & Config (always runnable) ---\n")
    for name, func in group_a:
        run_test(name, func)

    print("\n--- Group B: Processed Data Artifacts (require Phase 3) ---\n")
    for name, func in group_b:
        run_test(name, func)

    print("\n--- Group C: Data Leakage Verification (critical) ---\n")
    for name, func in group_c:
        run_test(name, func)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {SKIP} skipped out of {PASS + FAIL + SKIP}")
    print(f"{'=' * 60}\n")

    if FAIL > 0:
        sys.exit(1)

    if SKIP > 0:
        print("Some tests were skipped because processed data hasn't been generated yet.")
        print("After running Phase 3, re-run this script to verify all tests pass.")
    else:
        print("Phase 3 VERIFIED -- All tests passed.")
