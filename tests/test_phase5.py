"""
Phase 5 Verification — Run: python -m pytest tests/test_phase5.py -v

GROUP A: Script & Config (always runnable, no artifacts needed) — 8 tests
GROUP B: Model Artifacts (require Phase 5 execution) — 8 tests
"""

import sys
import json
import pickle
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import load_config, resolve_path, PROJECT_ROOT


# ===========================================================================
# Helpers
# ===========================================================================

def _models_exist() -> bool:
    """Check if Phase 5 artifacts exist."""
    config = load_config()
    return resolve_path(config["model"]["xgboost"]["model_path"]).exists()


def _get_model_dir() -> Path:
    config = load_config()
    return resolve_path(config["model"]["output_dir"])


# ===========================================================================
# GROUP A: Script & Config (always runnable) — 8 tests
# ===========================================================================

class TestGroupA_ScriptAndConfig:
    """These tests run without Phase 5 execution."""

    def test_model_training_script_exists(self):
        """Model training script must exist and be substantial."""
        script = PROJECT_ROOT / "src" / "models" / "model_training.py"
        assert script.exists(), f"Script not found: {script}"
        content = script.read_text(encoding="utf-8")
        assert len(content) > 3000, "Script is too small — likely incomplete"

    def test_model_training_is_importable(self):
        """All public functions must be importable."""
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

    def test_model_training_has_main_block(self):
        """Script must support CLI execution via __main__."""
        script = PROJECT_ROOT / "src" / "models" / "model_training.py"
        content = script.read_text(encoding="utf-8")
        assert '__name__' in content and '__main__' in content, \
            "Script must have __main__ block"

    def test_config_has_model_section(self):
        """config.yaml must have complete model section."""
        config = load_config()
        assert "model" in config, "config.yaml missing 'model' section"
        m = config["model"]
        required_keys = [
            "output_dir", "xgboost", "lightgbm",
            "metrics_path", "comparison_path", "best_model_path",
        ]
        for key in required_keys:
            assert key in m, f"config.yaml model section missing '{key}'"

    def test_config_xgboost_params_valid(self):
        """XGBoost hyperparameters must have valid values."""
        config = load_config()
        xgb = config["model"]["xgboost"]
        assert "model_path" in xgb, "Missing XGBoost model_path"
        assert "params" in xgb, "Missing XGBoost params"
        p = xgb["params"]
        assert p["n_estimators"] > 0, "n_estimators must be positive"
        assert p["max_depth"] > 0, "max_depth must be positive"
        assert 0 < p["learning_rate"] <= 1.0, "learning_rate must be (0, 1]"
        assert p["eval_metric"] == "aucpr", "eval_metric must be aucpr"
        assert p["random_state"] == 42, "random_state must be 42 for reproducibility"

    def test_config_lightgbm_params_valid(self):
        """LightGBM hyperparameters must have valid values."""
        config = load_config()
        lgb = config["model"]["lightgbm"]
        assert "model_path" in lgb, "Missing LightGBM model_path"
        assert "params" in lgb, "Missing LightGBM params"
        p = lgb["params"]
        assert p["n_estimators"] > 0, "n_estimators must be positive"
        assert p["max_depth"] > 0, "max_depth must be positive"
        assert 0 < p["learning_rate"] <= 1.0, "learning_rate must be (0, 1]"
        assert p["is_unbalance"] is True, "is_unbalance must be True for imbalanced data"
        assert p["random_state"] == 42, "random_state must be 42 for reproducibility"

    def test_models_directory_exists(self):
        """The models/ output directory must exist."""
        models_dir = _get_model_dir()
        assert models_dir.exists(), f"Models directory not found: {models_dir}"
        assert models_dir.is_dir(), f"Models path is not a directory: {models_dir}"

    def test_phase_tracking_integration(self):
        """Script must use centralized phase tracking."""
        script = PROJECT_ROOT / "src" / "models" / "model_training.py"
        content = script.read_text(encoding="utf-8")
        assert "log_phase_start" in content, "Must use log_phase_start()"
        assert "log_phase_end" in content, "Must use log_phase_end()"
        assert "get_logger" in content, "Must use get_logger()"
        assert "load_config" in content, "Must use load_config()"


# ===========================================================================
# GROUP B: Model Artifacts (require Phase 5 execution) — 8 tests
# ===========================================================================

class TestGroupB_ModelArtifacts:
    """These tests require Phase 5 to have been executed."""

    def test_xgboost_model_exists(self):
        """XGBoost model file must exist after Phase 5."""
        if not _models_exist():
            pytest.skip("Phase 5 not yet executed — run model_training.py first")
        config = load_config()
        path = resolve_path(config["model"]["xgboost"]["model_path"])
        assert path.exists(), f"XGBoost model not found: {path}"
        assert path.stat().st_size > 0, "XGBoost model file is empty"

    def test_lightgbm_model_exists(self):
        """LightGBM model file must exist after Phase 5."""
        if not _models_exist():
            pytest.skip("Phase 5 not yet executed")
        config = load_config()
        path = resolve_path(config["model"]["lightgbm"]["model_path"])
        assert path.exists(), f"LightGBM model not found: {path}"
        assert path.stat().st_size > 0, "LightGBM model file is empty"

    def test_best_model_exists(self):
        """Best model (winner) must be saved separately."""
        if not _models_exist():
            pytest.skip("Phase 5 not yet executed")
        config = load_config()
        path = resolve_path(config["model"]["best_model_path"])
        assert path.exists(), f"Best model not found: {path}"
        assert path.stat().st_size > 0, "Best model file is empty"

    def test_models_are_loadable_and_predict(self):
        """Both models must load and produce predictions."""
        if not _models_exist():
            pytest.skip("Phase 5 not yet executed")
        import joblib
        config = load_config()

        # Load both models
        xgb_model = joblib.load(resolve_path(config["model"]["xgboost"]["model_path"]))
        lgb_model = joblib.load(resolve_path(config["model"]["lightgbm"]["model_path"]))

        # Create dummy input (30 features based on the dataset)
        # dummy_input = np.random.randn(5, 30)
        dummy_input = np.random.randn(5, 37)

        # Both must produce probability predictions
        xgb_proba = xgb_model.predict_proba(dummy_input)
        lgb_proba = lgb_model.predict_proba(dummy_input)

        assert xgb_proba.shape == (5, 2), f"XGBoost output shape wrong: {xgb_proba.shape}"
        assert lgb_proba.shape == (5, 2), f"LightGBM output shape wrong: {lgb_proba.shape}"
        assert np.all((xgb_proba >= 0) & (xgb_proba <= 1)), "XGBoost probabilities out of range"
        assert np.all((lgb_proba >= 0) & (lgb_proba <= 1)), "LightGBM probabilities out of range"

    def test_metrics_json_exists_and_valid(self):
        """metrics.json must exist with correct structure and AUPRC values."""
        if not _models_exist():
            pytest.skip("Phase 5 not yet executed")
        config = load_config()
        path = resolve_path(config["model"]["metrics_path"])
        assert path.exists(), f"Metrics file not found: {path}"

        with open(path, "r") as f:
            metrics = json.load(f)

        # Must have both models
        assert "xgboost" in metrics, "Metrics missing 'xgboost'"
        assert "lightgbm" in metrics, "Metrics missing 'lightgbm'"
        assert metrics["primary_metric"] == "AUPRC", "Primary metric must be AUPRC"

        # AUPRC must be present and reasonable
        for model_key in ["xgboost", "lightgbm"]:
            m = metrics[model_key]
            assert "auprc" in m, f"{model_key} missing 'auprc'"
            assert 0 < m["auprc"] <= 1.0, f"{model_key} AUPRC out of range: {m['auprc']}"
            assert "roc_auc" in m, f"{model_key} missing 'roc_auc'"
            assert "default_threshold" in m, f"{model_key} missing 'default_threshold'"
            assert "optimal_threshold" in m, f"{model_key} missing 'optimal_threshold'"

    def test_metrics_auprc_above_minimum(self):
        """Both models must achieve at least 0.50 AUPRC (sanity check)."""
        if not _models_exist():
            pytest.skip("Phase 5 not yet executed")
        config = load_config()
        path = resolve_path(config["model"]["metrics_path"])

        with open(path, "r") as f:
            metrics = json.load(f)

        min_auprc = 0.50  # Reasonable floor for credit card fraud detection
        for model_key in ["xgboost", "lightgbm"]:
            auprc = metrics[model_key]["auprc"]
            assert auprc >= min_auprc, \
                f"{model_key} AUPRC {auprc:.4f} is below minimum threshold {min_auprc}"

    def test_comparison_json_exists_and_valid(self):
        """model_comparison.json must exist with winner declared."""
        if not _models_exist():
            pytest.skip("Phase 5 not yet executed")
        config = load_config()
        path = resolve_path(config["model"]["comparison_path"])
        assert path.exists(), f"Comparison file not found: {path}"

        with open(path, "r") as f:
            comparison = json.load(f)

        assert "winner" in comparison, "Comparison missing 'winner'"
        assert comparison["winner"] in ("XGBoost", "LightGBM"), \
            f"Invalid winner: {comparison['winner']}"
        assert "primary_metric" in comparison, "Comparison missing 'primary_metric'"
        assert comparison["primary_metric"] == "AUPRC", "Primary metric must be AUPRC"
        assert "winning_auprc" in comparison, "Comparison missing 'winning_auprc'"
        assert "margin" in comparison, "Comparison missing 'margin'"
        assert "xgboost" in comparison, "Comparison missing 'xgboost' details"
        assert "lightgbm" in comparison, "Comparison missing 'lightgbm' details"

    def test_best_model_matches_winner(self):
        """best_model.pkl must be the same as the declared winner."""
        if not _models_exist():
            pytest.skip("Phase 5 not yet executed")
        import joblib
        config = load_config()

        # Load comparison to find winner
        comp_path = resolve_path(config["model"]["comparison_path"])
        with open(comp_path, "r") as f:
            comparison = json.load(f)

        winner = comparison["winner"]

        # Load best model and the winner model
        best_model = joblib.load(resolve_path(config["model"]["best_model_path"]))

        if winner == "XGBoost":
            winner_model = joblib.load(resolve_path(config["model"]["xgboost"]["model_path"]))
        else:
            winner_model = joblib.load(resolve_path(config["model"]["lightgbm"]["model_path"]))

        # Both should produce identical predictions on same input
        # dummy_input = np.random.RandomState(42).randn(10, 30)
        dummy_input = np.random.RandomState(42).randn(10, 37)
        
        best_preds = best_model.predict_proba(dummy_input)
        winner_preds = winner_model.predict_proba(dummy_input)
        np.testing.assert_array_almost_equal(
            best_preds, winner_preds, decimal=6,
            err_msg=f"best_model.pkl does not match {winner} model"
        )
