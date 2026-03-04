"""
Phase 3 Verification -- Run: python -m pytest tests/test_phase3.py -v

Tests verify that Phase 3 data engineering artifacts exist and contain
the correct structure, feature engineering, scaling, splitting, and
SMOTE application -- all without data leakage.

NOTE: Group A tests are always runnable (script, config, imports).
Group B and C tests require Phase 3 to have been executed with the dataset.
"""
import sys
import json
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _processed_exists() -> bool:
    from src.utils.logger import resolve_path, load_config
    config = load_config()
    return resolve_path(config["preprocessing"]["train_path"]).exists()


def _get_processed_dir() -> Path:
    from src.utils.logger import resolve_path, load_config
    config = load_config()
    return resolve_path(config["preprocessing"]["processed_dir"])


# ===========================================================================
# Group A: Script & Config (always runnable)
# ===========================================================================

def test_data_engineering_script_exists():
    """The data engineering script must exist at the expected location."""
    from src.utils.logger import PROJECT_ROOT
    script = PROJECT_ROOT / "src" / "preprocessing" / "data_engineering.py"
    assert script.exists(), f"data_engineering.py not found at {script}"
    content = script.read_text(encoding="utf-8")
    assert len(content) > 2000, "data_engineering.py is too short to be functional"


def test_data_engineering_is_importable():
    """The data engineering module must be importable without errors."""
    from src.preprocessing.data_engineering import (
        run_phase3,
        load_raw_data,
        clean_data,
        engineer_temporal_features,
        split_data,
        scale_features,
        apply_smote,
        save_processed_data,
        verify_data_integrity,
    )
    assert callable(run_phase3)
    assert callable(load_raw_data)
    assert callable(clean_data)
    assert callable(engineer_temporal_features)
    assert callable(split_data)
    assert callable(scale_features)
    assert callable(apply_smote)
    assert callable(save_processed_data)
    assert callable(verify_data_integrity)


def test_config_has_preprocessing_section():
    """config.yaml must have preprocessing section with all required keys."""
    from src.utils.logger import load_config
    config = load_config()
    assert "preprocessing" in config, "config.yaml missing 'preprocessing' section"
    preproc = config["preprocessing"]
    required_keys = [
        "processed_dir", "train_path", "test_path", "val_path",
        "y_train_path", "y_test_path", "y_val_path",
        "scaler_path", "test_size", "val_size", "scaling_method", "smote",
    ]
    for key in required_keys:
        assert key in preproc, f"config preprocessing missing key: '{key}'"


def test_config_smote_settings():
    """SMOTE configuration must have required parameters."""
    from src.utils.logger import load_config
    config = load_config()
    smote = config["preprocessing"]["smote"]
    assert "enabled" in smote, "SMOTE config missing 'enabled'"
    assert "sampling_strategy" in smote, "SMOTE config missing 'sampling_strategy'"
    assert "k_neighbors" in smote, "SMOTE config missing 'k_neighbors'"


def test_script_documents_leakage_prevention():
    """Script must document data leakage prevention strategies."""
    from src.utils.logger import PROJECT_ROOT
    content = (PROJECT_ROOT / "src" / "preprocessing" / "data_engineering.py").read_text(encoding="utf-8").upper()
    assert "LEAKAGE" in content, "Script must document data leakage prevention"
    assert "BEFORE" in content, "Script must mention splitting BEFORE scaling/SMOTE"


def test_phase_tracking_integration():
    """Script must use the centralized phase tracker."""
    from src.utils.logger import PROJECT_ROOT
    content = (PROJECT_ROOT / "src" / "preprocessing" / "data_engineering.py").read_text(encoding="utf-8")
    assert "log_phase_start" in content, "Script must call log_phase_start()"
    assert "log_phase_end" in content, "Script must call log_phase_end()"
    assert "Phase 3" in content, "Script must identify itself as Phase 3"


def test_script_uses_robust_scaler():
    """Script must use RobustScaler (resistant to outliers in financial data)."""
    from src.utils.logger import PROJECT_ROOT
    content = (PROJECT_ROOT / "src" / "preprocessing" / "data_engineering.py").read_text(encoding="utf-8")
    assert "RobustScaler" in content, "Script must use RobustScaler"


def test_script_creates_temporal_features():
    """Script must engineer temporal features."""
    from src.utils.logger import PROJECT_ROOT
    content = (PROJECT_ROOT / "src" / "preprocessing" / "data_engineering.py").read_text(encoding="utf-8")
    for feature in ["hour_of_day", "is_night", "time_since_prev", "amount_log", "tx_frequency_1h"]:
        assert feature in content, f"Script must create temporal feature: {feature}"


# ===========================================================================
# Group B: Processed Data Artifacts (require Phase 3 execution)
# ===========================================================================

def test_processed_train_exists():
    """Training data must exist after Phase 3."""
    if not _processed_exists():
        import pytest
        pytest.skip("Processed data not found -- run Phase 3 first")
    from src.utils.logger import resolve_path, load_config
    config = load_config()
    import pandas as pd
    X_train = pd.read_csv(resolve_path(config["preprocessing"]["train_path"]))
    assert len(X_train) > 0, "X_train is empty"
    assert len(X_train.columns) > 30, f"X_train has too few features: {len(X_train.columns)}"


def test_processed_val_exists():
    """Validation data must exist after Phase 3."""
    if not _processed_exists():
        import pytest
        pytest.skip("Processed data not found -- run Phase 3 first")
    from src.utils.logger import resolve_path, load_config
    config = load_config()
    import pandas as pd
    X_val = pd.read_csv(resolve_path(config["preprocessing"]["val_path"]))
    assert len(X_val) > 0, "X_val is empty"


def test_processed_test_exists():
    """Test data must exist after Phase 3."""
    if not _processed_exists():
        import pytest
        pytest.skip("Processed data not found -- run Phase 3 first")
    from src.utils.logger import resolve_path, load_config
    config = load_config()
    import pandas as pd
    X_test = pd.read_csv(resolve_path(config["preprocessing"]["test_path"]))
    assert len(X_test) > 0, "X_test is empty"


def test_scaler_exists():
    """Scaler pickle must exist after Phase 3."""
    if not _processed_exists():
        import pytest
        pytest.skip("Processed data not found -- run Phase 3 first")
    from src.utils.logger import resolve_path, load_config
    config = load_config()
    scaler_path = resolve_path(config["preprocessing"]["scaler_path"])
    assert scaler_path.exists(), f"Scaler not found at {scaler_path}"
    import pickle
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    assert hasattr(scaler, "transform"), "Saved object is not a valid scaler"


def test_feature_names_json_exists():
    """Feature names JSON must exist for downstream phases."""
    if not _processed_exists():
        import pytest
        pytest.skip("Processed data not found -- run Phase 3 first")
    processed_dir = _get_processed_dir()
    feature_path = processed_dir / "feature_names.json"
    assert feature_path.exists(), "feature_names.json not found"
    content = json.loads(feature_path.read_text(encoding="utf-8"))
    assert "features" in content, "feature_names.json missing 'features' key"
    assert len(content["features"]) > 30, "Too few features listed"


def test_engineering_report_exists():
    """Engineering report must exist and contain all sections."""
    if not _processed_exists():
        import pytest
        pytest.skip("Processed data not found -- run Phase 3 first")
    processed_dir = _get_processed_dir()
    report_path = processed_dir / "engineering_report.json"
    assert report_path.exists(), "engineering_report.json not found"
    content = json.loads(report_path.read_text(encoding="utf-8"))
    required_sections = [
        "data_leakage_prevention", "cleaning", "feature_engineering",
        "data_split", "scaling", "smote", "saved_artifacts",
    ]
    for section in required_sections:
        assert section in content, f"Engineering report missing section: '{section}'"


def test_temporal_features_in_data():
    """Processed data must contain the engineered temporal features."""
    if not _processed_exists():
        import pytest
        pytest.skip("Processed data not found -- run Phase 3 first")
    from src.utils.logger import resolve_path, load_config
    config = load_config()
    import pandas as pd
    X_train = pd.read_csv(resolve_path(config["preprocessing"]["train_path"]))
    expected_features = ["hour_of_day", "is_night", "time_since_prev", "amount_log", "tx_frequency_1h"]
    for feat in expected_features:
        assert feat in X_train.columns, f"Temporal feature missing from training data: {feat}"


# ===========================================================================
# Group C: Data Leakage Verification (require Phase 3 execution)
# ===========================================================================

def test_no_leakage_val_test_not_smoted():
    """Validation and test sets must NOT be affected by SMOTE."""
    if not _processed_exists():
        import pytest
        pytest.skip("Processed data not found -- run Phase 3 first")
    from src.utils.logger import resolve_path, load_config
    config = load_config()
    import pandas as pd

    y_val = pd.read_csv(resolve_path(config["preprocessing"]["y_val_path"]))
    y_test = pd.read_csv(resolve_path(config["preprocessing"]["y_test_path"]))

    # Original fraud ratio is ~0.17%. SMOTE would push it much higher.
    # Val and test fraud ratios should remain close to 0.17%
    val_fraud_ratio = y_val.iloc[:, 0].mean()
    test_fraud_ratio = y_test.iloc[:, 0].mean()

    assert val_fraud_ratio < 0.01, (
        f"Val fraud ratio ({val_fraud_ratio:.4%}) is too high -- "
        f"SMOTE may have leaked into validation set!"
    )
    assert test_fraud_ratio < 0.01, (
        f"Test fraud ratio ({test_fraud_ratio:.4%}) is too high -- "
        f"SMOTE may have leaked into test set!"
    )


def test_smote_applied_to_train():
    """Training set should have more fraud samples than original ratio."""
    if not _processed_exists():
        import pytest
        pytest.skip("Processed data not found -- run Phase 3 first")
    from src.utils.logger import resolve_path, load_config
    config = load_config()
    import pandas as pd

    y_train = pd.read_csv(resolve_path(config["preprocessing"]["y_train_path"]))
    train_fraud_ratio = y_train.iloc[:, 0].mean()

    # With SMOTE sampling_strategy=0.3, fraud should be ~23% of total
    # At minimum it should be much higher than the original 0.17%
    assert train_fraud_ratio > 0.05, (
        f"Train fraud ratio ({train_fraud_ratio:.4%}) is too low -- "
        f"SMOTE may not have been applied correctly."
    )


def test_feature_consistency_across_splits():
    """All three sets must have the same features (columns)."""
    if not _processed_exists():
        import pytest
        pytest.skip("Processed data not found -- run Phase 3 first")
    from src.utils.logger import resolve_path, load_config
    config = load_config()
    import pandas as pd

    X_train = pd.read_csv(resolve_path(config["preprocessing"]["train_path"]), nrows=1)
    X_val = pd.read_csv(resolve_path(config["preprocessing"]["val_path"]), nrows=1)
    X_test = pd.read_csv(resolve_path(config["preprocessing"]["test_path"]), nrows=1)

    train_cols = set(X_train.columns)
    val_cols = set(X_val.columns)
    test_cols = set(X_test.columns)

    assert train_cols == val_cols, (
        f"Train and val have different features.\n"
        f"  Only in train: {train_cols - val_cols}\n"
        f"  Only in val: {val_cols - train_cols}"
    )
    assert train_cols == test_cols, (
        f"Train and test have different features.\n"
        f"  Only in train: {train_cols - test_cols}\n"
        f"  Only in test: {test_cols - train_cols}"
    )
