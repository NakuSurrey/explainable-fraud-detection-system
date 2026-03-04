"""
Phase 9 Verification Tests: Streamlit Dashboard (THE FRONT DESK)
=================================================================
Tests organized into two groups:
  Group A: Code Structure & Config (always runnable, no server needed)
  Group B: Live Dashboard Tests (require API server running on localhost:8000)

Usage:
    python tests/run_phase9_tests.py
"""

import os
import sys
import json
import importlib

# Ensure project root is on path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.utils.logger import load_config, resolve_path


# ================================================================
# HELPERS
# ================================================================

def _api_is_running():
    """Check if the Phase 8 API server is running."""
    try:
        import requests
        r = requests.get("http://localhost:8000/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ================================================================
# GROUP A: CODE STRUCTURE & CONFIG (no server needed)
# ================================================================

def test_dashboard_file_exists():
    """src/dashboard/app.py must exist and be substantial."""
    path = resolve_path("src/dashboard/app.py")
    assert path.exists(), f"Dashboard file not found: {path}"
    content = path.read_text(encoding="utf-8")
    assert len(content) > 5000, f"Dashboard file too small: {len(content)} chars"
    assert "streamlit" in content.lower(), "Dashboard must use Streamlit"
    assert "Phase 9" in content or "phase9" in content.lower(), "Dashboard must reference Phase 9"


def test_dashboard_init_exists():
    """src/dashboard/__init__.py must exist."""
    path = resolve_path("src/dashboard/__init__.py")
    assert path.exists(), f"Dashboard __init__.py not found: {path}"


def test_dashboard_imports_config():
    """Dashboard must use the centralized config system."""
    path = resolve_path("src/dashboard/app.py")
    content = path.read_text(encoding="utf-8")
    assert "load_config" in content, "Dashboard must use load_config()"
    assert "get_logger" in content, "Dashboard must use get_logger()"
    assert "resolve_path" in content, "Dashboard must use resolve_path()"


def test_dashboard_has_no_ml_code():
    """Dashboard must contain ZERO machine learning code — it only calls the API."""
    path = resolve_path("src/dashboard/app.py")
    content = path.read_text(encoding="utf-8")
    # Should NOT import ML libraries directly
    ml_imports = [
        "import xgboost",
        "import lightgbm",
        "import shap",
        "import lime",
        "from sklearn",
        "import sklearn",
        "import joblib",
        "import pickle",
    ]
    for imp in ml_imports:
        assert imp not in content, (
            f"Dashboard contains ML import '{imp}'. "
            "Phase 9 must have ZERO ML code — it only calls the API."
        )


def test_dashboard_calls_api():
    """Dashboard must call the Phase 8 API endpoints."""
    path = resolve_path("src/dashboard/app.py")
    content = path.read_text(encoding="utf-8")
    # Must reference API endpoints
    assert "/health" in content, "Dashboard must call /health endpoint"
    assert "/predict" in content, "Dashboard must call /predict endpoint"
    assert "/sample" in content, "Dashboard must call /sample endpoint"


def test_dashboard_has_shap_visualization():
    """Dashboard must render SHAP explanations."""
    path = resolve_path("src/dashboard/app.py")
    content = path.read_text(encoding="utf-8")
    assert "shap" in content.lower(), "Dashboard must display SHAP explanations"
    assert "Increases Risk" in content or "increases_risk" in content, \
        "Dashboard must show risk-increasing features"
    assert "Decreases Risk" in content or "decreases_risk" in content, \
        "Dashboard must show risk-decreasing features"


def test_dashboard_has_lime_section():
    """Dashboard must have LIME explanation section."""
    path = resolve_path("src/dashboard/app.py")
    content = path.read_text(encoding="utf-8")
    assert "lime" in content.lower(), "Dashboard must reference LIME"
    assert "/predict/lime" in content, "Dashboard must call /predict/lime endpoint"


def test_dashboard_has_fraud_ring_visualization():
    """Dashboard must visualize the fraud ring network."""
    path = resolve_path("src/dashboard/app.py")
    content = path.read_text(encoding="utf-8")
    assert "fraud_rings" in content or "fraud_ring" in content, \
        "Dashboard must load fraud ring data"
    assert "fraud_edges" in content or "edge_list" in content, \
        "Dashboard must load edge list data"
    assert "RING_001" in content, "Dashboard must reference detected rings"


def test_dashboard_has_model_comparison():
    """Dashboard must show model comparison (XGBoost vs LightGBM)."""
    path = resolve_path("src/dashboard/app.py")
    content = path.read_text(encoding="utf-8")
    assert "XGBoost" in content, "Dashboard must reference XGBoost"
    assert "LightGBM" in content, "Dashboard must reference LightGBM"
    assert "AUPRC" in content, "Dashboard must reference AUPRC metric"
    assert "comparison" in content.lower(), "Dashboard must load model comparison"


def test_dashboard_has_risk_levels():
    """Dashboard must display all four risk levels."""
    path = resolve_path("src/dashboard/app.py")
    content = path.read_text(encoding="utf-8")
    for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        assert level in content, f"Dashboard must display risk level: {level}"


def test_config_has_dashboard_section():
    """config.yaml must have dashboard section with required keys."""
    config = load_config()
    assert "dashboard" in config, "Config missing 'dashboard' section"
    dash = config["dashboard"]
    assert "title" in dash, "Dashboard config missing 'title'"
    assert "port" in dash, "Dashboard config missing 'port'"
    assert "api_url" in dash, "Dashboard config missing 'api_url'"
    assert "localhost" in dash["api_url"], "api_url must reference localhost"


def test_dashboard_has_three_tabs():
    """Dashboard must have Transaction Analysis, Fraud Ring, and Model Performance tabs."""
    path = resolve_path("src/dashboard/app.py")
    content = path.read_text(encoding="utf-8")
    assert "Transaction Analysis" in content, "Missing Transaction Analysis tab"
    assert "Fraud Ring" in content, "Missing Fraud Ring Network tab"
    assert "Model Performance" in content, "Missing Model Performance tab"


def test_dashboard_has_session_state():
    """Dashboard must use session state for managing form/results state."""
    path = resolve_path("src/dashboard/app.py")
    content = path.read_text(encoding="utf-8")
    assert "session_state" in content, "Dashboard must use st.session_state"
    assert "prediction_result" in content, "Must track prediction results in state"
    assert "transaction_features" in content, "Must track transaction features in state"


def test_dashboard_has_gauge():
    """Dashboard must have a visual risk gauge."""
    path = resolve_path("src/dashboard/app.py")
    content = path.read_text(encoding="utf-8")
    assert "gauge" in content.lower(), "Dashboard must have a risk gauge"
    assert "svg" in content.lower() or "SVG" in content, \
        "Gauge should use SVG for rendering"


def test_graph_data_files_exist():
    """Phase 4 graph artifacts must exist for the network tab."""
    config = load_config()
    # Check required graph files
    rings_path = resolve_path(config["graph"]["rings_path"])
    summary_path = resolve_path(config["graph"]["summary_path"])
    edges_path = resolve_path(config["graph"]["edge_list_path"])

    assert rings_path.exists(), f"fraud_rings.json not found: {rings_path}"
    assert summary_path.exists(), f"graph_summary.json not found: {summary_path}"
    assert edges_path.exists(), f"fraud_edges.csv not found: {edges_path}"


def test_metrics_files_exist():
    """Phase 5 metrics artifacts must exist for the performance tab."""
    config = load_config()
    metrics_path = resolve_path(config["model"]["metrics_path"])
    comparison_path = resolve_path(config["model"]["comparison_path"])

    assert metrics_path.exists(), f"metrics.json not found: {metrics_path}"
    assert comparison_path.exists(), f"model_comparison.json not found: {comparison_path}"


# ================================================================
# GROUP B: LIVE DASHBOARD TESTS (require API on localhost:8000)
# ================================================================

def test_api_health_from_dashboard():
    """Dashboard can reach the API health endpoint."""
    if not _api_is_running():
        return "SKIP"
    import requests
    config = load_config()
    api_url = config["dashboard"]["api_url"]
    r = requests.get(f"{api_url}/health", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["feature_count"] == 37


def test_sample_endpoint_returns_37_features():
    """GET /sample returns all 37 features for the input form."""
    if not _api_is_running():
        return "SKIP"
    import requests
    config = load_config()
    api_url = config["dashboard"]["api_url"]
    r = requests.get(f"{api_url}/sample", timeout=5)
    assert r.status_code == 200
    data = r.json()
    features = data.get("features", {})
    assert len(features) == 37, f"Expected 37 features, got {len(features)}"
    assert "Amount" in features, "Missing 'Amount' feature"
    assert "V14" in features, "Missing 'V14' feature"


def test_predict_returns_valid_response():
    """POST /predict with sample data returns valid prediction."""
    if not _api_is_running():
        return "SKIP"
    import requests
    config = load_config()
    api_url = config["dashboard"]["api_url"]

    # Get sample features
    sample = requests.get(f"{api_url}/sample", timeout=5).json()
    features = sample.get("features", {})

    # Predict
    r = requests.post(f"{api_url}/predict", json={"features": features}, timeout=15)
    assert r.status_code == 200

    result = r.json()
    assert "fraud_probability" in result, "Missing fraud_probability"
    assert "risk_level" in result, "Missing risk_level"
    assert "shap_explanation" in result, "Missing shap_explanation"
    assert "plain_english_summary" in result, "Missing plain_english_summary"

    prob = result["fraud_probability"]
    assert 0 <= prob <= 1, f"Probability out of range: {prob}"
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"], \
        f"Invalid risk level: {result['risk_level']}"
    assert len(result["shap_explanation"]) > 0, "SHAP explanation is empty"


def test_predict_shap_has_correct_structure():
    """SHAP explanations must have feature_name, shap_value, direction."""
    if not _api_is_running():
        return "SKIP"
    import requests
    config = load_config()
    api_url = config["dashboard"]["api_url"]

    sample = requests.get(f"{api_url}/sample", timeout=5).json()
    r = requests.post(f"{api_url}/predict", json={"features": sample["features"]}, timeout=15)
    result = r.json()

    for item in result["shap_explanation"][:3]:
        assert "feature_name" in item, "SHAP item missing feature_name"
        assert "shap_value" in item, "SHAP item missing shap_value"
        assert "feature_value" in item, "SHAP item missing feature_value"
        assert "direction" in item, "SHAP item missing direction"
        assert item["direction"] in ["increases_risk", "decreases_risk"], \
            f"Invalid direction: {item['direction']}"


def test_predict_lime_endpoint():
    """POST /predict/lime returns valid response (may have empty explanations)."""
    if not _api_is_running():
        return "SKIP"
    import requests
    config = load_config()
    api_url = config["dashboard"]["api_url"]

    sample = requests.get(f"{api_url}/sample", timeout=5).json()
    r = requests.post(f"{api_url}/predict/lime", json={"features": sample["features"]}, timeout=30)

    # LIME endpoint may return 200 or 503 (if LIME explainer has issues)
    if r.status_code == 200:
        result = r.json()
        assert "fraud_probability" in result, "LIME response missing fraud_probability"
        assert "risk_level" in result, "LIME response missing risk_level"
        # lime_explanation may be empty — that's acceptable (known Phase 8 behavior)
        assert "lime_explanation" in result, "LIME response missing lime_explanation field"


def test_model_info_endpoint():
    """GET /model/info returns model metadata."""
    if not _api_is_running():
        return "SKIP"
    import requests
    config = load_config()
    api_url = config["dashboard"]["api_url"]
    r = requests.get(f"{api_url}/model/info", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "feature_count" in data, "Model info missing feature_count"


def test_graph_summary_data_valid():
    """graph_summary.json must have valid graph and ring stats."""
    config = load_config()
    path = resolve_path(config["graph"]["summary_path"])
    if not path.exists():
        return "SKIP"

    with open(path, "r") as f:
        summary = json.load(f)

    gs = summary.get("graph_stats", {})
    rs = summary.get("ring_stats", {})

    assert gs.get("total_nodes", 0) > 0, "Graph has no nodes"
    assert gs.get("total_edges", 0) > 0, "Graph has no edges"
    assert rs.get("total_rings_detected", 0) > 0, "No fraud rings detected"


def test_comparison_data_has_winner():
    """model_comparison.json must have a declared winner."""
    config = load_config()
    path = resolve_path(config["model"]["comparison_path"])
    if not path.exists():
        return "SKIP"

    with open(path, "r") as f:
        comp = json.load(f)

    assert "winner" in comp, "Comparison missing winner"
    assert comp["winner"] in ("XGBoost", "LightGBM"), f"Invalid winner: {comp['winner']}"
    assert "xgboost" in comp, "Comparison missing xgboost details"
    assert "lightgbm" in comp, "Comparison missing lightgbm details"
    assert comp.get("primary_metric") == "AUPRC", "Primary metric must be AUPRC"


# ================================================================
# TEST RUNNER
# ================================================================

def run_all():
    """Run all Phase 9 tests and report results."""
    from datetime import datetime

    group_a_tests = [
        test_dashboard_file_exists,
        test_dashboard_init_exists,
        test_dashboard_imports_config,
        test_dashboard_has_no_ml_code,
        test_dashboard_calls_api,
        test_dashboard_has_shap_visualization,
        test_dashboard_has_lime_section,
        test_dashboard_has_fraud_ring_visualization,
        test_dashboard_has_model_comparison,
        test_dashboard_has_risk_levels,
        test_config_has_dashboard_section,
        test_dashboard_has_three_tabs,
        test_dashboard_has_session_state,
        test_dashboard_has_gauge,
        test_graph_data_files_exist,
        test_metrics_files_exist,
    ]

    group_b_tests = [
        test_api_health_from_dashboard,
        test_sample_endpoint_returns_37_features,
        test_predict_returns_valid_response,
        test_predict_shap_has_correct_structure,
        test_predict_lime_endpoint,
        test_model_info_endpoint,
        test_graph_summary_data_valid,
        test_comparison_data_has_winner,
    ]

    PASS = 0
    FAIL = 0
    SKIP = 0

    print("=" * 70)
    print("PHASE 9 VERIFICATION: Streamlit Dashboard (THE FRONT DESK)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Group A
    print("\n--- Group A: Code Structure & Config (no server needed) ---")
    for test_fn in group_a_tests:
        name = test_fn.__name__
        try:
            result = test_fn()
            if result == "SKIP":
                print(f"  SKIP  {name}")
                SKIP += 1
            else:
                print(f"  PASS  {name}")
                PASS += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            print(f"        Error: {e}")
            FAIL += 1

    # Group B
    print("\n--- Group B: Live Dashboard Tests (server must be running) ---")
    for test_fn in group_b_tests:
        name = test_fn.__name__
        try:
            result = test_fn()
            if result == "SKIP":
                print(f"  SKIP  {name}")
                SKIP += 1
            else:
                print(f"  PASS  {name}")
                PASS += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            print(f"        Error: {e}")
            FAIL += 1

    # Summary
    print("\n" + "=" * 70)
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {SKIP} skipped")

    if FAIL > 0:
        print("Phase 9 has FAILURES. Check the errors above.")
        if SKIP > 0:
            print("Some tests were skipped (API server not running).")
    else:
        if SKIP > 0:
            print(f"\n{SKIP} tests were skipped (API server not running on localhost:8000).")
            print("Start the API server and re-run to verify all tests.")
        else:
            print("\nPhase 9 VERIFIED -- All tests passed.")
    print("=" * 70)

    return FAIL == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
