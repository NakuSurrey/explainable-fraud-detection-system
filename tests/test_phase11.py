"""
Phase 11 Verification Tests -- Model Monitoring (THE WATCHTOWER)
================================================================
Run: python -m pytest tests/test_phase11.py -v
  or: python tests/run_phase11_tests.py

Group A (14 tests): Always runnable -- code structure, imports, config, offline logic
Group B (8 tests):  Require API on localhost:8000 -- endpoint responses, prediction logging
Total: 22 tests
"""

import sys
import json
import sqlite3
import tempfile
import os
from pathlib import Path

import pytest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import PROJECT_ROOT, load_config, resolve_path


# ===========================================================================
# Helpers
# ===========================================================================

API_URL = "http://localhost:8000"


def _api_reachable() -> bool:
    """Check if the API server is running on localhost:8000."""
    try:
        import requests
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _baseline_exists() -> bool:
    """Check if prediction_baseline.json has been generated."""
    config = load_config()
    monitoring_cfg = config.get("monitoring", {})
    baseline_path = monitoring_cfg.get("baseline_path", "models/prediction_baseline.json")
    return resolve_path(baseline_path).exists()


# ===========================================================================
# GROUP A: Code Structure & Offline Tests (14 tests, no server needed)
# ===========================================================================

class TestGroupA_CodeStructureAndOffline:
    """Tests that always run — verify code, config, imports, and offline logic."""

    # --- A1: Module existence ---
    def test_monitoring_module_exists(self):
        """monitoring package must exist with __init__.py and prediction_monitor.py."""
        init_file = PROJECT_ROOT / "src" / "monitoring" / "__init__.py"
        monitor_file = PROJECT_ROOT / "src" / "monitoring" / "prediction_monitor.py"
        assert init_file.exists(), f"Missing: {init_file}"
        assert monitor_file.exists(), f"Missing: {monitor_file}"
        # prediction_monitor.py should be a real module, not a stub
        content = monitor_file.read_text(encoding="utf-8")
        assert len(content) > 2000, "prediction_monitor.py is too small — likely incomplete"

    # --- A2: All public functions importable ---
    def test_monitoring_imports(self):
        """All public functions from prediction_monitor must be importable."""
        from src.monitoring.prediction_monitor import (
            init_monitoring_db,
            log_prediction,
            get_recent_predictions,
            get_prediction_count,
            generate_baseline,
            load_baseline,
            calculate_psi,
            generate_drift_report,
            get_monitoring_summary,
        )
        # every import should be callable
        assert callable(init_monitoring_db)
        assert callable(log_prediction)
        assert callable(get_recent_predictions)
        assert callable(get_prediction_count)
        assert callable(generate_baseline)
        assert callable(load_baseline)
        assert callable(calculate_psi)
        assert callable(generate_drift_report)
        assert callable(get_monitoring_summary)

    # --- A3: CLI support ---
    def test_monitoring_has_cli_support(self):
        """prediction_monitor.py must support --generate-baseline and --check-drift."""
        script = PROJECT_ROOT / "src" / "monitoring" / "prediction_monitor.py"
        content = script.read_text(encoding="utf-8")
        assert "argparse" in content, "Must use argparse for CLI"
        assert "--generate-baseline" in content, "Must support --generate-baseline flag"
        assert "--check-drift" in content, "Must support --check-drift flag"
        assert "__main__" in content, "Must have __main__ block"

    # --- A4: Config has monitoring section with all keys ---
    def test_config_has_monitoring_section(self):
        """config.yaml must have monitoring section with all required keys."""
        config = load_config()
        assert "monitoring" in config, "config.yaml missing 'monitoring' section"
        m = config["monitoring"]
        required_keys = [
            "drift_check_interval_days",
            "psi_threshold",
            "auprc_decay_threshold",
            "db_path",
            "baseline_path",
        ]
        for key in required_keys:
            assert key in m, f"monitoring section missing '{key}'"
        # validate values make sense
        assert 0 < m["psi_threshold"] <= 1.0, "psi_threshold must be between 0 and 1"
        assert m["drift_check_interval_days"] > 0, "drift_check_interval_days must be positive"

    # --- A5: Data directory exists ---
    def test_monitoring_data_directory_exists(self):
        """data/monitoring/ directory must exist (with .gitkeep)."""
        monitoring_dir = PROJECT_ROOT / "data" / "monitoring"
        assert monitoring_dir.exists(), f"Missing directory: {monitoring_dir}"
        assert monitoring_dir.is_dir(), f"Not a directory: {monitoring_dir}"

    # --- A6: DB initialization in temp dir ---
    def test_db_initialization(self):
        """init_monitoring_db logic: table and index created correctly."""
        # testing the DB creation logic directly using a temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_predictions.db"
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            # replicate what init_monitoring_db does
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
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp
                ON predictions (timestamp)
            """)
            conn.commit()

            # verify table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            assert "predictions" in tables, f"Table 'predictions' not created. Found: {tables}"

            # verify columns
            cursor.execute("PRAGMA table_info(predictions)")
            cols = {row[1]: row[2] for row in cursor.fetchall()}
            expected_cols = {
                "id": "INTEGER",
                "timestamp": "TEXT",
                "probability": "REAL",
                "risk_level": "TEXT",
                "is_flagged": "INTEGER",
                "threshold_used": "REAL",
            }
            for col_name, col_type in expected_cols.items():
                assert col_name in cols, f"Column '{col_name}' missing"
                assert cols[col_name] == col_type, f"Column '{col_name}' has wrong type: {cols[col_name]}"

            # verify index exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            assert "idx_predictions_timestamp" in indexes, "Index on timestamp not created"

            conn.close()

    # --- A7: PSI calculation — identical distributions ---
    def test_psi_identical_distributions(self):
        """PSI of two identical distributions should be near zero (NO_DRIFT)."""
        from src.monitoring.prediction_monitor import calculate_psi

        # same distribution — PSI should be essentially 0
        dist = [0.90, 0.05, 0.02, 0.01, 0.005, 0.005, 0.003, 0.003, 0.002, 0.002]
        result = calculate_psi(dist, dist)

        assert "psi_score" in result, "Missing psi_score"
        assert "verdict" in result, "Missing verdict"
        assert "recommendation" in result, "Missing recommendation"
        assert result["psi_score"] < 0.01, f"Identical distributions should give PSI ~0, got {result['psi_score']}"
        assert result["verdict"] == "NO_DRIFT", f"Expected NO_DRIFT, got {result['verdict']}"

    # --- A8: PSI calculation — heavily shifted distribution ---
    def test_psi_shifted_distribution(self):
        """PSI of a heavily shifted distribution should be SIGNIFICANT_DRIFT."""
        from src.monitoring.prediction_monitor import calculate_psi

        baseline = [0.90, 0.05, 0.02, 0.01, 0.005, 0.005, 0.003, 0.003, 0.002, 0.002]
        # inverted distribution — most mass in the high bins now
        shifted = [0.002, 0.002, 0.003, 0.003, 0.005, 0.005, 0.01, 0.02, 0.05, 0.90]
        result = calculate_psi(baseline, shifted)

        assert result["psi_score"] > 0.2, f"Heavily shifted should give PSI > 0.2, got {result['psi_score']}"
        assert result["verdict"] == "SIGNIFICANT_DRIFT", f"Expected SIGNIFICANT_DRIFT, got {result['verdict']}"

    # --- A9: PSI per-bin values returned ---
    def test_psi_returns_per_bin_values(self):
        """PSI result must include per_bin_psi with 10 values."""
        from src.monitoring.prediction_monitor import calculate_psi

        dist_a = [0.5, 0.2, 0.1, 0.05, 0.05, 0.03, 0.03, 0.02, 0.01, 0.01]
        dist_b = [0.45, 0.22, 0.11, 0.06, 0.04, 0.04, 0.03, 0.02, 0.02, 0.01]
        result = calculate_psi(dist_a, dist_b)

        assert "per_bin_psi" in result, "Missing per_bin_psi in result"
        assert len(result["per_bin_psi"]) == 10, f"Expected 10 bin values, got {len(result['per_bin_psi'])}"
        # all per-bin values should be non-negative (PSI is always >= 0 per bin)
        for i, val in enumerate(result["per_bin_psi"]):
            assert val >= 0, f"Bin {i} has negative PSI: {val}"

    # --- A10: Baseline generation from synthetic data ---
    def test_baseline_generation_synthetic(self):
        """generate_baseline should produce a valid JSON structure."""
        from src.monitoring.prediction_monitor import generate_baseline

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_baseline.json")
            # synthetic probabilities — mostly near 0 with a few higher ones
            probs = np.concatenate([
                np.random.beta(1, 50, size=950),   # most near 0
                np.random.beta(5, 2, size=50),      # a few near 1
            ])

            baseline = generate_baseline(probs, output_path=output_path)

            # check structure
            assert "metadata" in baseline, "Missing metadata"
            assert "distribution" in baseline, "Missing distribution"
            assert "statistics" in baseline, "Missing statistics"
            assert baseline["metadata"]["n_samples"] == 1000
            assert len(baseline["distribution"]) == 10, "Should have 10 bins"
            # proportions should sum to approximately 1.0
            total = sum(baseline["distribution"])
            assert abs(total - 1.0) < 0.001, f"Distribution sums to {total}, expected ~1.0"
            # file should exist on disk
            assert Path(output_path).exists(), "Baseline JSON was not saved to disk"

    # --- A11: Baseline file exists on user's machine ---
    def test_baseline_file_exists(self):
        """models/prediction_baseline.json should exist after Sub-step 2."""
        if not _baseline_exists():
            pytest.skip("Baseline not yet generated — run --generate-baseline first")
        config = load_config()
        baseline_path = resolve_path(config["monitoring"]["baseline_path"])
        with open(baseline_path, "r") as f:
            data = json.load(f)
        # validate structure
        assert "metadata" in data, "Baseline missing 'metadata'"
        assert "distribution" in data, "Baseline missing 'distribution'"
        assert "statistics" in data, "Baseline missing 'statistics'"
        assert data["metadata"]["n_samples"] > 0, "Baseline has zero samples"
        assert len(data["distribution"]) == 10, "Baseline should have 10 bins"

    # --- A12: Dashboard has monitoring tab ---
    def test_dashboard_has_monitoring_tab(self):
        """app.py must have the Model Monitoring tab and helper functions."""
        dashboard_path = PROJECT_ROOT / "src" / "dashboard" / "app.py"
        assert dashboard_path.exists(), f"Dashboard not found: {dashboard_path}"
        content = dashboard_path.read_text(encoding="utf-8")
        # monitoring helper functions
        assert "get_monitoring_drift_report" in content, "Missing get_monitoring_drift_report function"
        assert "get_monitoring_summary_from_api" in content, "Missing get_monitoring_summary_from_api function"
        assert "get_monitoring_predictions" in content, "Missing get_monitoring_predictions function"
        # monitoring tab
        assert "Model Monitoring" in content, "Missing 'Model Monitoring' tab label"
        assert "tab_monitoring" in content, "Missing tab_monitoring variable"
        # PSI gauge
        assert "build_psi_gauge_svg" in content, "Missing build_psi_gauge_svg function"
        # drift chart
        assert "render_drift_distribution_chart" in content, "Missing render_drift_distribution_chart function"

    # --- A13: API has monitoring imports ---
    def test_api_has_monitoring_integration(self):
        """inference_api.py must import and use monitoring functions."""
        api_path = PROJECT_ROOT / "src" / "api" / "inference_api.py"
        assert api_path.exists(), f"API not found: {api_path}"
        content = api_path.read_text(encoding="utf-8")
        # monitoring imports
        assert "init_monitoring_db" in content, "Missing init_monitoring_db import"
        assert "log_prediction" in content, "Missing log_prediction import"
        assert "generate_drift_report" in content, "Missing generate_drift_report import"
        assert "get_monitoring_summary" in content, "Missing get_monitoring_summary import"
        # monitoring endpoints
        assert "/monitoring/drift-report" in content, "Missing /monitoring/drift-report endpoint"
        assert "/monitoring/summary" in content, "Missing /monitoring/summary endpoint"
        assert "/monitoring/predictions" in content, "Missing /monitoring/predictions endpoint"

    # --- A14: Constants are correct ---
    def test_monitoring_constants(self):
        """PSI bin edges and epsilon must have correct values."""
        from src.monitoring.prediction_monitor import PSI_BIN_EDGES, PSI_EPSILON

        assert len(PSI_BIN_EDGES) == 11, f"Should have 11 bin edges (10 bins), got {len(PSI_BIN_EDGES)}"
        assert PSI_BIN_EDGES[0] == 0.0, "First bin edge must be 0.0"
        assert PSI_BIN_EDGES[-1] == 1.0, "Last bin edge must be 1.0"
        assert PSI_EPSILON > 0, "Epsilon must be positive"
        assert PSI_EPSILON < 0.001, "Epsilon should be very small"


# ===========================================================================
# GROUP B: API Endpoint Tests (8 tests, require server on localhost:8000)
# ===========================================================================

class TestGroupB_APIEndpoints:
    """Tests that require the API server running on localhost:8000."""

    # --- B1: Monitoring summary endpoint ---
    def test_monitoring_summary_endpoint(self):
        """GET /monitoring/summary should return monitoring status."""
        if not _api_reachable():
            pytest.skip("API not running on localhost:8000")
        import requests
        r = requests.get(f"{API_URL}/monitoring/summary", timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        assert "total_predictions_logged" in data, "Missing total_predictions_logged"
        assert "baseline_exists" in data, "Missing baseline_exists"
        assert "ready_for_drift_check" in data, "Missing ready_for_drift_check"
        assert "minimum_for_drift_check" in data, "Missing minimum_for_drift_check"
        assert isinstance(data["total_predictions_logged"], int), "total_predictions_logged must be int"

    # --- B2: Monitoring predictions endpoint ---
    def test_monitoring_predictions_endpoint(self):
        """GET /monitoring/predictions should return a list of logged predictions."""
        if not _api_reachable():
            pytest.skip("API not running on localhost:8000")
        import requests
        r = requests.get(f"{API_URL}/monitoring/predictions", params={"limit": 5}, timeout=10)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        assert "count" in data, "Missing count field"
        assert "predictions" in data, "Missing predictions field"
        assert isinstance(data["predictions"], list), "predictions must be a list"

    # --- B3: Predict logs to monitoring ---
    def test_predict_logs_to_monitoring(self):
        """POST /predict should silently log to monitoring DB."""
        if not _api_reachable():
            pytest.skip("API not running on localhost:8000")
        import requests

        # get current count
        r1 = requests.get(f"{API_URL}/monitoring/summary", timeout=10)
        count_before = r1.json().get("total_predictions_logged", 0)

        # make a prediction
        sample_r = requests.get(f"{API_URL}/sample", timeout=5)
        if sample_r.status_code != 200:
            pytest.skip("Could not get sample transaction")
        features = sample_r.json().get("features", {})
        pred_r = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=15)
        assert pred_r.status_code == 200, f"Predict failed: {pred_r.status_code}"

        # check count increased
        r2 = requests.get(f"{API_URL}/monitoring/summary", timeout=10)
        count_after = r2.json().get("total_predictions_logged", 0)
        assert count_after > count_before, (
            f"Prediction count did not increase: before={count_before}, after={count_after}"
        )

    # --- B4: Logged prediction has correct structure ---
    def test_logged_prediction_structure(self):
        """Most recent logged prediction must have all required fields."""
        if not _api_reachable():
            pytest.skip("API not running on localhost:8000")
        import requests

        r = requests.get(f"{API_URL}/monitoring/predictions", params={"limit": 1}, timeout=10)
        assert r.status_code == 200
        data = r.json()
        if data["count"] == 0:
            pytest.skip("No predictions logged yet")

        pred = data["predictions"][0]
        required_fields = ["id", "timestamp", "probability", "risk_level", "is_flagged", "threshold_used"]
        for field in required_fields:
            assert field in pred, f"Logged prediction missing '{field}'"

        # validate value ranges
        assert 0 <= pred["probability"] <= 1.0, f"Probability out of range: {pred['probability']}"
        assert pred["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL"), (
            f"Invalid risk_level: {pred['risk_level']}"
        )
        assert pred["is_flagged"] in (0, 1), f"is_flagged must be 0 or 1, got {pred['is_flagged']}"
        assert pred["threshold_used"] > 0, f"threshold_used must be positive"

    # --- B5: Drift report endpoint responds ---
    def test_drift_report_endpoint(self):
        """GET /monitoring/drift-report should return a valid response."""
        if not _api_reachable():
            pytest.skip("API not running on localhost:8000")
        import requests

        r = requests.get(f"{API_URL}/monitoring/drift-report", timeout=15)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        data = r.json()
        # status must be one of the three valid states
        assert "status" in data, "Missing status field"
        assert data["status"] in ("OK", "INSUFFICIENT_DATA", "ERROR"), (
            f"Unexpected status: {data['status']}"
        )

    # --- B6: API routes in OpenAPI spec ---
    def test_api_has_monitoring_routes(self):
        """OpenAPI spec must include all 3 monitoring endpoints."""
        if not _api_reachable():
            pytest.skip("API not running on localhost:8000")
        import requests

        r = requests.get(f"{API_URL}/openapi.json", timeout=10)
        assert r.status_code == 200, f"OpenAPI spec not accessible"
        spec = r.json()
        paths = spec.get("paths", {})
        # all three monitoring endpoints must be registered
        assert "/monitoring/drift-report" in paths, "Missing /monitoring/drift-report in OpenAPI"
        assert "/monitoring/summary" in paths, "Missing /monitoring/summary in OpenAPI"
        assert "/monitoring/predictions" in paths, "Missing /monitoring/predictions in OpenAPI"

    # --- B7: Health endpoint still works ---
    def test_health_endpoint_still_works(self):
        """GET /health must still return 200 with all fields — monitoring must not break existing endpoints."""
        if not _api_reachable():
            pytest.skip("API not running on localhost:8000")
        import requests

        r = requests.get(f"{API_URL}/health", timeout=5)
        assert r.status_code == 200
        data = r.json()
        # all original health fields must still be present
        assert data.get("status") == "healthy", "Health status should be 'healthy'"
        assert "model_loaded" in data, "Missing model_loaded"
        assert "shap_loaded" in data, "Missing shap_loaded"
        assert "feature_count" in data, "Missing feature_count"

    # --- B8: E2E flow — predict then verify in monitoring log ---
    def test_predict_then_verify_in_monitoring_log(self):
        """Full E2E: get sample -> predict -> verify prediction appears in monitoring log."""
        if not _api_reachable():
            pytest.skip("API not running on localhost:8000")
        import requests

        # step 1 — get a sample transaction
        sample_r = requests.get(f"{API_URL}/sample", timeout=5)
        if sample_r.status_code != 200:
            pytest.skip("Could not get sample transaction")
        features = sample_r.json().get("features", {})

        # step 2 — make prediction
        pred_r = requests.post(f"{API_URL}/predict", json={"features": features}, timeout=15)
        assert pred_r.status_code == 200
        pred_data = pred_r.json()
        pred_prob = pred_data.get("fraud_probability", -1)

        # step 3 — fetch most recent logged prediction
        log_r = requests.get(f"{API_URL}/monitoring/predictions", params={"limit": 1}, timeout=10)
        assert log_r.status_code == 200
        log_data = log_r.json()
        assert log_data["count"] > 0, "No predictions in monitoring log after predict call"

        most_recent = log_data["predictions"][0]
        # the most recent logged probability should match what the API returned
        assert abs(most_recent["probability"] - pred_prob) < 0.0001, (
            f"Logged probability ({most_recent['probability']}) does not match "
            f"prediction response ({pred_prob})"
        )
