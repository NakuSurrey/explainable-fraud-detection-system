"""
Phase 5 Standalone Test Runner
================================
Run: python tests/run_phase5_tests.py

Group A: Script & Config (always runnable) — 8 tests
Group B: Model Artifacts (require Phase 5 execution) — 8 tests
"""

import sys
import json
import pickle
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import load_config, resolve_path, PROJECT_ROOT

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


def _models_exist() -> bool:
    config = load_config()
    return resolve_path(config["model"]["xgboost"]["model_path"]).exists()


# ===========================================================================
# GROUP A: Script & Config (always runnable) — 8 tests
# ===========================================================================

def test_model_training_script_exists():
    script = PROJECT_ROOT / "src" / "models" / "model_training.py"
    assert script.exists(), f"Script not found: {script}"
    content = script.read_text(encoding="utf-8")
    assert len(content) > 3000, "Script is too small — likely incomplete"


def test_model_training_is_importable():
    from src.models.model_training import (
        load_processed_data,
        train_xgboost,
        train_lightgbm,
        evaluate_model,
        compare_models,
        save_artifacts,
        main,
    )
    assert callable(load_processed_data)
    assert callable(train_xgboost)
    assert callable(train_lightgbm)
    assert callable(evaluate_model)
    assert callable(compare_models)
    assert callable(save_artifacts)
    assert callable(main)


def test_model_training_has_main_block():
    script = PROJECT_ROOT / "src" / "models" / "model_training.py"
    content = script.read_text(encoding="utf-8")
    assert '__name__' in content and '__main__' in content, \
        "Script must have __main__ block"


def test_config_has_model_section():
    config = load_config()
    assert "model" in config, "config.yaml missing 'model' section"
    m = config["model"]
    for key in ["output_dir", "xgboost", "lightgbm",
                "metrics_path", "comparison_path", "best_model_path"]:
        assert key in m, f"Missing '{key}'"


def test_config_xgboost_params_valid():
    config = load_config()
    xgb = config["model"]["xgboost"]
    assert "model_path" in xgb
    assert "params" in xgb
    p = xgb["params"]
    assert p["n_estimators"] > 0
    assert p["max_depth"] > 0
    assert 0 < p["learning_rate"] <= 1.0
    assert p["eval_metric"] == "aucpr"
    assert p["random_state"] == 42


def test_config_lightgbm_params_valid():
    config = load_config()
    lgb = config["model"]["lightgbm"]
    assert "model_path" in lgb
    assert "params" in lgb
    p = lgb["params"]
    assert p["n_estimators"] > 0
    assert p["max_depth"] > 0
    assert 0 < p["learning_rate"] <= 1.0
    assert p["is_unbalance"] is True
    assert p["random_state"] == 42


def test_models_directory_exists():
    config = load_config()
    models_dir = resolve_path(config["model"]["output_dir"])
    assert models_dir.exists(), f"Models directory not found: {models_dir}"
    assert models_dir.is_dir()


def test_phase_tracking_integration():
    script = PROJECT_ROOT / "src" / "models" / "model_training.py"
    content = script.read_text(encoding="utf-8")
    assert "log_phase_start" in content
    assert "log_phase_end" in content
    assert "get_logger" in content
    assert "load_config" in content


# ===========================================================================
# GROUP B: Model Artifacts (require Phase 5 execution) — 8 tests
# ===========================================================================

def test_xgboost_model_exists():
    if not _models_exist():
        return "SKIP"
    config = load_config()
    path = resolve_path(config["model"]["xgboost"]["model_path"])
    assert path.exists(), f"XGBoost model not found: {path}"
    assert path.stat().st_size > 0, "XGBoost model file is empty"


def test_lightgbm_model_exists():
    if not _models_exist():
        return "SKIP"
    config = load_config()
    path = resolve_path(config["model"]["lightgbm"]["model_path"])
    assert path.exists(), f"LightGBM model not found: {path}"
    assert path.stat().st_size > 0, "LightGBM model file is empty"


def test_best_model_exists():
    if not _models_exist():
        return "SKIP"
    config = load_config()
    path = resolve_path(config["model"]["best_model_path"])
    assert path.exists(), f"Best model not found: {path}"
    assert path.stat().st_size > 0, "Best model file is empty"


def test_models_are_loadable_and_predict():
    if not _models_exist():
        return "SKIP"
    import joblib
    config = load_config()

    xgb_model = joblib.load(resolve_path(config["model"]["xgboost"]["model_path"]))
    lgb_model = joblib.load(resolve_path(config["model"]["lightgbm"]["model_path"]))

    # dummy_input = np.random.randn(5, 30)
    dummy_input = np.random.randn(5, 37)

    xgb_proba = xgb_model.predict_proba(dummy_input)
    lgb_proba = lgb_model.predict_proba(dummy_input)

    assert xgb_proba.shape == (5, 2), f"XGBoost shape wrong: {xgb_proba.shape}"
    assert lgb_proba.shape == (5, 2), f"LightGBM shape wrong: {lgb_proba.shape}"
    assert np.all((xgb_proba >= 0) & (xgb_proba <= 1)), "XGBoost probabilities out of range"
    assert np.all((lgb_proba >= 0) & (lgb_proba <= 1)), "LightGBM probabilities out of range"


def test_metrics_json_exists_and_valid():
    if not _models_exist():
        return "SKIP"
    config = load_config()
    path = resolve_path(config["model"]["metrics_path"])
    assert path.exists(), f"Metrics file not found: {path}"

    with open(path, "r") as f:
        metrics = json.load(f)

    assert "xgboost" in metrics, "Metrics missing 'xgboost'"
    assert "lightgbm" in metrics, "Metrics missing 'lightgbm'"
    assert metrics["primary_metric"] == "AUPRC"

    for model_key in ["xgboost", "lightgbm"]:
        m = metrics[model_key]
        assert "auprc" in m, f"{model_key} missing 'auprc'"
        assert 0 < m["auprc"] <= 1.0, f"{model_key} AUPRC out of range: {m['auprc']}"
        assert "roc_auc" in m
        assert "default_threshold" in m
        assert "optimal_threshold" in m


def test_metrics_auprc_above_minimum():
    if not _models_exist():
        return "SKIP"
    config = load_config()
    path = resolve_path(config["model"]["metrics_path"])

    with open(path, "r") as f:
        metrics = json.load(f)

    min_auprc = 0.50
    for model_key in ["xgboost", "lightgbm"]:
        auprc = metrics[model_key]["auprc"]
        assert auprc >= min_auprc, \
            f"{model_key} AUPRC {auprc:.4f} is below minimum threshold {min_auprc}"


def test_comparison_json_exists_and_valid():
    if not _models_exist():
        return "SKIP"
    config = load_config()
    path = resolve_path(config["model"]["comparison_path"])
    assert path.exists(), f"Comparison file not found: {path}"

    with open(path, "r") as f:
        comparison = json.load(f)

    assert "winner" in comparison
    assert comparison["winner"] in ("XGBoost", "LightGBM"), \
        f"Invalid winner: {comparison['winner']}"
    assert comparison["primary_metric"] == "AUPRC"
    assert "winning_auprc" in comparison
    assert "margin" in comparison
    assert "xgboost" in comparison
    assert "lightgbm" in comparison


def test_best_model_matches_winner():
    if not _models_exist():
        return "SKIP"
    import joblib
    config = load_config()

    comp_path = resolve_path(config["model"]["comparison_path"])
    with open(comp_path, "r") as f:
        comparison = json.load(f)

    winner = comparison["winner"]

    best_model = joblib.load(resolve_path(config["model"]["best_model_path"]))
    if winner == "XGBoost":
        winner_model = joblib.load(resolve_path(config["model"]["xgboost"]["model_path"]))
    else:
        winner_model = joblib.load(resolve_path(config["model"]["lightgbm"]["model_path"]))

    # dummy_input = np.random.RandomState(42).randn(10, 30)
    dummy_input = np.random.RandomState(42).randn(10, 37)
    best_preds = best_model.predict_proba(dummy_input)
    winner_preds = winner_model.predict_proba(dummy_input)
    np.testing.assert_array_almost_equal(
        best_preds, winner_preds, decimal=6,
        err_msg=f"best_model.pkl does not match {winner} model"
    )


# ===========================================================================
# RUNNER
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 5 VERIFICATION: Model Training & Evaluation")
    print("=" * 60 + "\n")

    print("GROUP A: Script & Config (always runnable)")
    print("-" * 45)
    run_test("test_model_training_script_exists", test_model_training_script_exists)
    run_test("test_model_training_is_importable", test_model_training_is_importable)
    run_test("test_model_training_has_main_block", test_model_training_has_main_block)
    run_test("test_config_has_model_section", test_config_has_model_section)
    run_test("test_config_xgboost_params_valid", test_config_xgboost_params_valid)
    run_test("test_config_lightgbm_params_valid", test_config_lightgbm_params_valid)
    run_test("test_models_directory_exists", test_models_directory_exists)
    run_test("test_phase_tracking_integration", test_phase_tracking_integration)

    print()
    print("GROUP B: Model Artifacts (require Phase 5 execution)")
    print("-" * 45)
    run_test("test_xgboost_model_exists", test_xgboost_model_exists)
    run_test("test_lightgbm_model_exists", test_lightgbm_model_exists)
    run_test("test_best_model_exists", test_best_model_exists)
    run_test("test_models_are_loadable_and_predict", test_models_are_loadable_and_predict)
    run_test("test_metrics_json_exists_and_valid", test_metrics_json_exists_and_valid)
    run_test("test_metrics_auprc_above_minimum", test_metrics_auprc_above_minimum)
    run_test("test_comparison_json_exists_and_valid", test_comparison_json_exists_and_valid)
    run_test("test_best_model_matches_winner", test_best_model_matches_winner)

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print("=" * 60)

    if FAIL > 0:
        print("\nSome tests FAILED. Check the errors above.")
        if SKIP > 0:
            print("After running Phase 5, re-run this script to verify all tests pass.")
    else:
        if SKIP > 0:
            print(f"\n{SKIP} tests were skipped (Phase 5 not yet executed).")
            print("Run Phase 5 first, then re-run this script.")
        else:
            print("\nPhase 5 VERIFIED — All tests passed.")
