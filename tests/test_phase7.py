"""
Phase 7 Verification Tests -- Explainable AI (XAI) Generation
==============================================================
Run: python -m pytest tests/test_phase7.py -v
  or: python tests/run_phase7_tests.py

Group A (8 tests): Always runnable -- test code structure and imports
Group B (10 tests): Require Phase 7 artifacts -- test saved outputs
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import load_config, resolve_path


# ===================================================================
# GROUP A: Always Runnable (code structure & imports)
# ===================================================================

def test_xai_engine_imports():
    """Test that the xai_engine module can be imported."""
    from src.explainability import xai_engine
    assert hasattr(xai_engine, "main")
    assert hasattr(xai_engine, "fit_shap_explainer")
    assert hasattr(xai_engine, "fit_lime_explainer")
    assert hasattr(xai_engine, "compute_shap_values")
    assert hasattr(xai_engine, "compute_global_feature_importance")
    assert hasattr(xai_engine, "generate_example_explanations")


def test_shap_library_available():
    """Test that SHAP library is installed and importable."""
    import shap
    assert hasattr(shap, "TreeExplainer")
    assert hasattr(shap, "Explanation")


def test_lime_library_available():
    """Test that LIME library is installed and importable."""
    import lime
    import lime.lime_tabular
    assert hasattr(lime.lime_tabular, "LimeTabularExplainer")


def test_config_has_explainability_section():
    """Test that config.yaml has the explainability section."""
    config = load_config()
    assert "explainability" in config, "Missing 'explainability' in config.yaml"
    xai_cfg = config["explainability"]
    assert "shap_explainer_path" in xai_cfg
    assert "shap_values_path" in xai_cfg
    assert "lime_explainer_path" in xai_cfg
    assert "background_samples" in xai_cfg
    assert "max_display_features" in xai_cfg


def test_global_feature_importance_function():
    """Test compute_global_feature_importance with dummy data."""
    from src.explainability.xai_engine import compute_global_feature_importance

    # Create dummy SHAP values: 10 samples, 5 features
    np.random.seed(42)
    shap_vals = np.random.randn(10, 5)
    names = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]

    result = compute_global_feature_importance(shap_vals, names)

    assert isinstance(result, list)
    assert len(result) == 5
    assert all("feature" in r and "importance" in r for r in result)
    # Should be sorted descending
    importances = [r["importance"] for r in result]
    assert importances == sorted(importances, reverse=True)


def test_global_feature_importance_no_names():
    """Test compute_global_feature_importance with no feature names."""
    from src.explainability.xai_engine import compute_global_feature_importance

    np.random.seed(42)
    shap_vals = np.random.randn(10, 3)

    result = compute_global_feature_importance(shap_vals, None)

    assert len(result) == 3
    assert result[0]["feature"].startswith("Feature_")


def test_load_optimal_threshold():
    """Test load_optimal_threshold returns a float."""
    from src.explainability.xai_engine import load_optimal_threshold

    config = load_config()
    threshold = load_optimal_threshold(config)

    assert isinstance(threshold, float)
    assert 0.0 <= threshold <= 1.0


def test_load_feature_names():
    """Test load_feature_names returns a list or None."""
    from src.explainability.xai_engine import load_feature_names

    config = load_config()
    names = load_feature_names(config)

    if names is not None:
        assert isinstance(names, list)
        assert len(names) > 0
        assert all(isinstance(n, str) for n in names)


# ===================================================================
# GROUP B: Require Phase 7 Artifacts
# ===================================================================

def test_shap_explainer_exists():
    """Test that SHAP explainer pickle was saved."""
    config = load_config()
    path = resolve_path(config["explainability"]["shap_explainer_path"])
    assert path.exists(), f"SHAP explainer not found at {path}"
    assert path.stat().st_size > 0, "SHAP explainer file is empty"


def test_shap_values_exist():
    """Test that SHAP values pickle was saved."""
    config = load_config()
    path = resolve_path(config["explainability"]["shap_values_path"])
    assert path.exists(), f"SHAP values not found at {path}"
    assert path.stat().st_size > 0, "SHAP values file is empty"


def test_lime_explainer_exists():
    """Test that LIME explainer pickle was saved."""
    config = load_config()
    path = resolve_path(config["explainability"]["lime_explainer_path"])
    assert path.exists(), f"LIME explainer not found at {path}"
    assert path.stat().st_size > 0, "LIME explainer file is empty"


def test_xai_report_json_exists():
    """Test that the XAI report JSON was generated."""
    path = resolve_path("reports/xai_report.json")
    assert path.exists(), f"XAI report JSON not found at {path}"

    with open(path, "r") as f:
        report = json.load(f)

    assert "phase" in report
    assert "global_feature_importance" in report
    assert "example_shap_explanations" in report
    assert "timing" in report
    assert "configuration" in report


def test_xai_report_txt_exists():
    """Test that the human-readable XAI report was generated."""
    path = resolve_path("reports/xai_report.txt")
    assert path.exists(), f"XAI report TXT not found at {path}"

    content = path.read_text(encoding="utf-8")
    assert "EXPLAINABLE AI" in content
    assert "GLOBAL FEATURE IMPORTANCE" in content
    assert "COMPLIANCE NOTE" in content


def test_shap_explainer_is_loadable():
    """Test that the saved SHAP explainer can be unpickled."""
    config = load_config()
    path = resolve_path(config["explainability"]["shap_explainer_path"])

    with open(path, "rb") as f:
        explainer = pickle.load(f)

    assert explainer is not None
    assert hasattr(explainer, "shap_values") or hasattr(explainer, "__call__")


def test_shap_values_shape():
    """Test that SHAP values have correct shape matching test data."""
    config = load_config()

    shap_path = resolve_path(config["explainability"]["shap_values_path"])
    with open(shap_path, "rb") as f:
        shap_values = pickle.load(f)

    X_test_path = resolve_path(config["preprocessing"]["test_path"])
    X_test = pd.read_csv(X_test_path)

    # SHAP values should match test set dimensions
    assert shap_values.shape[0] == X_test.shape[0], (
        f"SHAP values rows ({shap_values.shape[0]}) != "
        f"X_test rows ({X_test.shape[0]})"
    )
    assert shap_values.shape[1] == X_test.shape[1], (
        f"SHAP values cols ({shap_values.shape[1]}) != "
        f"X_test cols ({X_test.shape[1]})"
    )


def test_lime_explainer_is_functional():
    """Test that the saved LIME explainer can generate an explanation."""
    config = load_config()

    # lime_path = resolve_path(config["explainability"]["lime_explainer_path"])
    # with open(lime_path, "rb") as f:
    #     lime_explainer = pickle.load(f)
    import dill
    lime_path = resolve_path(config["explainability"]["lime_explainer_path"])
    with open(lime_path, "rb") as f:
        lime_explainer = dill.load(f)

    model_path = resolve_path(config["model"]["best_model_path"])
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test_path = resolve_path(config["preprocessing"]["test_path"])
    X_test = pd.read_csv(X_test_path)

    # Explain the first test instance
    instance = X_test.iloc[0].values
    explanation = lime_explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=5,
        num_samples=100,
    )

    features = explanation.as_list()
    assert len(features) > 0, "LIME returned no feature explanations"
    assert all(len(f) == 2 for f in features), "LIME features should be (name, weight) tuples"


def test_global_importance_has_entries():
    """Test that global feature importance was computed with entries."""
    path = resolve_path("reports/xai_report.json")
    with open(path, "r") as f:
        report = json.load(f)

    importance = report["global_feature_importance"]
    assert len(importance) > 0, "Global feature importance is empty"
    assert importance[0]["importance"] >= importance[-1]["importance"], (
        "Feature importance should be sorted descending"
    )


def test_example_explanations_have_plain_english():
    """Test that example explanations include plain-English text."""
    path = resolve_path("reports/xai_report.json")
    with open(path, "r") as f:
        report = json.load(f)

    examples = report["example_shap_explanations"]
    assert len(examples) > 0, "No example explanations found"

    for ex in examples:
        assert "plain_english_explanation" in ex, "Missing plain English explanation"
        assert len(ex["plain_english_explanation"]) > 20, "Explanation too short"
        assert "top_features" in ex, "Missing top features"
        assert len(ex["top_features"]) > 0, "No top features listed"
        assert "predicted_probability" in ex
        assert "actual_label_text" in ex
