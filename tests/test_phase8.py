"""
Phase 8 Verification Tests — Inference API (THE MESSENGER)
============================================================
Run: python tests/run_phase8_tests.py

Test Groups:
    Group A (8 tests): Code structure, imports, schema validation
                       — run BEFORE starting the API server
    Group B (10 tests): Live API endpoint testing
                       — run AFTER starting the API server (python -m src.api.inference_api)

Total: 18 tests
"""

import sys
import json
import os
import importlib
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


# ============================================================
# GROUP A: Code Structure & Offline Tests (no API server needed)
# ============================================================

def test_inference_api_file_exists():
    """Verify src/api/inference_api.py exists."""
    api_path = _project_root / "src" / "api" / "inference_api.py"
    assert api_path.exists(), f"inference_api.py not found at {api_path}"
    content = api_path.read_text(encoding="utf-8")
    assert len(content) > 1000, "inference_api.py is too small — likely empty or placeholder"
    assert "FastAPI" in content, "inference_api.py does not reference FastAPI"
    assert "predict" in content, "inference_api.py does not contain predict function"


def test_api_module_imports():
    """Verify the API module can be imported without errors."""
    from src.api import inference_api
    assert hasattr(inference_api, "app"), "FastAPI 'app' not found in module"
    assert hasattr(inference_api, "model_service"), "model_service not found in module"
    assert hasattr(inference_api, "ModelService"), "ModelService class not found"
    assert hasattr(inference_api, "predict_single"), "predict_single function not found"
    assert hasattr(inference_api, "predict_lime"), "predict_lime function not found"


def test_pydantic_schemas():
    """Verify all Pydantic request/response schemas are defined."""
    from src.api.inference_api import (
        TransactionInput,
        BatchTransactionInput,
        ShapExplanation,
        PredictionResponse,
        BatchPredictionResponse,
        LimeExplanation,
        LimePredictionResponse,
        HealthResponse,
        ModelInfoResponse,
    )
    # Verify TransactionInput accepts features dict
    txn = TransactionInput(features={"V1": 1.0, "V2": 2.0})
    assert txn.features["V1"] == 1.0

    # Verify BatchTransactionInput accepts list
    batch = BatchTransactionInput(transactions=[txn, txn])
    assert len(batch.transactions) == 2

    # Verify ShapExplanation
    shap_exp = ShapExplanation(
        feature_name="V14", shap_value=-0.5, feature_value=-5.2, direction="increases_risk"
    )
    assert shap_exp.feature_name == "V14"


def test_risk_classification():
    """Verify risk classification logic is correct."""
    from src.api.inference_api import classify_risk

    # CRITICAL: >= 0.80
    level, score, flagged = classify_risk(0.95)
    assert level == "CRITICAL"
    assert score == 95
    assert flagged is True

    # HIGH: >= 0.50
    level, score, flagged = classify_risk(0.65)
    assert level == "HIGH"
    assert flagged is True

    # MEDIUM: >= 0.20
    level, score, flagged = classify_risk(0.35)
    assert level == "MEDIUM"
    assert flagged is True

    # LOW: < 0.20
    level, score, flagged = classify_risk(0.05)
    assert level == "LOW"
    assert score == 5
    assert flagged is False

    # Edge cases
    level, _, _ = classify_risk(0.0)
    assert level == "LOW"
    level, _, _ = classify_risk(1.0)
    assert level == "CRITICAL"
    level, _, _ = classify_risk(0.80)
    assert level == "CRITICAL"
    level, _, _ = classify_risk(0.50)
    assert level == "HIGH"
    level, _, _ = classify_risk(0.20)
    assert level == "MEDIUM"


def test_plain_english_generator():
    """Verify plain-English explanation generation."""
    from src.api.inference_api import generate_plain_english

    # High risk with SHAP explanations
    shap_data = [
        {"feature_name": "V14", "shap_value": 0.8, "feature_value": -5.2, "direction": "increases_risk"},
        {"feature_name": "V4", "shap_value": 0.5, "feature_value": 2.1, "direction": "increases_risk"},
        {"feature_name": "V12", "shap_value": -0.3, "feature_value": 0.1, "direction": "decreases_risk"},
    ]
    text = generate_plain_english(shap_data, "CRITICAL", 0.92)
    assert "ALERT" in text
    assert "92.0%" in text
    assert "V14" in text
    assert "risk contribution" in text

    # Low risk
    text = generate_plain_english(shap_data, "LOW", 0.05)
    assert "legitimate" in text.lower()
    assert "5.0%" in text

    # Empty SHAP
    text = generate_plain_english([], "LOW", 0.03)
    assert "3.0%" in text


def test_model_service_init():
    """Verify ModelService initializes with correct defaults."""
    from src.api.inference_api import ModelService

    svc = ModelService()
    assert svc.model is None
    assert svc.shap_explainer is None
    assert svc.lime_explainer is None
    assert svc.scaler is None
    assert svc.feature_names is None
    assert svc.is_loaded is False
    assert svc.prediction_count == 0
    assert svc.model_version == "v1"


def test_config_has_api_section():
    """Verify config.yaml has the api section with required keys."""
    from src.utils.logger import load_config

    config = load_config()
    assert "api" in config, "config.yaml missing 'api' section"

    api = config["api"]
    assert "host" in api, "api config missing 'host'"
    assert "port" in api, "api config missing 'port'"
    assert "model_path" in api, "api config missing 'model_path'"
    assert "explainer_path" in api, "api config missing 'explainer_path'"
    assert "scaler_path" in api, "api config missing 'scaler_path'"
    assert "cors_origins" in api, "api config missing 'cors_origins'"

    assert api["port"] == 8000, f"Expected port 8000, got {api['port']}"
    assert isinstance(api["cors_origins"], list), "cors_origins must be a list"


def test_fastapi_routes_registered():
    """Verify all expected routes are registered on the FastAPI app."""
    from src.api.inference_api import app

    route_paths = [route.path for route in app.routes]

    expected_routes = ["/health", "/model/info", "/predict", "/predict/batch",
                       "/predict/lime", "/features", "/sample"]

    for route in expected_routes:
        assert route in route_paths, f"Route '{route}' not registered. Found: {route_paths}"


# ============================================================
# GROUP B: Live API Tests (require API server running)
# ============================================================
# These tests use httpx to call the running API.
# They are SKIPPED if the API is not reachable.

def _api_is_running():
    """Check if the API server is running on localhost:8000."""
    try:
        import httpx
        resp = httpx.get("http://localhost:8000/health", timeout=3.0)
        return resp.status_code == 200
    except Exception:
        return False


def _skip_if_api_not_running():
    """Return True if tests should be skipped."""
    if not _api_is_running():
        return True
    return False


def test_health_endpoint():
    """GET /health returns healthy status with all components."""
    if _skip_if_api_not_running():
        return "SKIP"

    import httpx
    resp = httpx.get("http://localhost:8000/health", timeout=10.0)
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["shap_loaded"] is True
    assert data["scaler_loaded"] is True
    assert data["feature_count"] == 37
    assert "timestamp" in data


def test_model_info_endpoint():
    """GET /model/info returns model metadata."""
    if _skip_if_api_not_running():
        return "SKIP"

    import httpx
    resp = httpx.get("http://localhost:8000/model/info", timeout=10.0)
    assert resp.status_code == 200

    data = resp.json()
    assert data["feature_count"] == 37
    assert len(data["feature_names"]) == 37
    assert "XGB" in data["model_type"] or "LGBM" in data["model_type"] or \
           "xgb" in data["model_type"].lower() or "lgb" in data["model_type"].lower() or \
           "Classifier" in data["model_type"], \
           f"Unexpected model type: {data['model_type']}"
    assert isinstance(data["metrics"], dict)


def test_features_endpoint():
    """GET /features returns the 37 feature names."""
    if _skip_if_api_not_running():
        return "SKIP"

    import httpx
    resp = httpx.get("http://localhost:8000/features", timeout=10.0)
    assert resp.status_code == 200

    data = resp.json()
    assert data["feature_count"] == 37
    assert "V1" in data["feature_names"]
    assert "V14" in data["feature_names"]
    assert "Amount" in data["feature_names"]


def test_sample_endpoint():
    """GET /sample returns a template with all 37 features."""
    if _skip_if_api_not_running():
        return "SKIP"

    import httpx
    resp = httpx.get("http://localhost:8000/sample", timeout=10.0)
    assert resp.status_code == 200

    data = resp.json()
    assert "features" in data
    assert len(data["features"]) == 37
    # All values should be 0.0
    assert all(v == 0.0 for v in data["features"].values())


def test_predict_endpoint():
    """POST /predict returns a valid prediction with SHAP explanation."""
    if _skip_if_api_not_running():
        return "SKIP"

    import httpx

    # Get sample transaction first
    sample_resp = httpx.get("http://localhost:8000/sample", timeout=10.0)
    sample = sample_resp.json()

    # Make prediction
    resp = httpx.post(
        "http://localhost:8000/predict",
        json={"features": sample["features"]},
        timeout=30.0,
    )
    assert resp.status_code == 200

    data = resp.json()
    assert "fraud_probability" in data
    assert 0.0 <= data["fraud_probability"] <= 1.0
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert 0 <= data["risk_score"] <= 100
    assert isinstance(data["is_flagged"], bool)
    assert data["prediction_label"] in ("LEGITIMATE", "FRAUDULENT")
    assert isinstance(data["shap_explanation"], list)
    assert len(data["plain_english_summary"]) > 10
    assert data["inference_time_ms"] >= 0
    assert "transaction_id" in data
    assert "timestamp" in data


def test_predict_shap_explanation_structure():
    """Verify SHAP explanations have correct structure and are sorted."""
    if _skip_if_api_not_running():
        return "SKIP"

    import httpx

    sample_resp = httpx.get("http://localhost:8000/sample", timeout=10.0)
    sample = sample_resp.json()

    resp = httpx.post(
        "http://localhost:8000/predict",
        json={"features": sample["features"]},
        timeout=30.0,
    )
    data = resp.json()

    if len(data["shap_explanation"]) > 0:
        first = data["shap_explanation"][0]
        assert "feature_name" in first
        assert "shap_value" in first
        assert "feature_value" in first
        assert first["direction"] in ("increases_risk", "decreases_risk")

        # Verify sorted by absolute SHAP value (descending)
        shap_abs = [abs(e["shap_value"]) for e in data["shap_explanation"]]
        assert shap_abs == sorted(shap_abs, reverse=True), \
            "SHAP explanations not sorted by absolute value"


def test_predict_missing_features():
    """POST /predict with missing features returns 422."""
    if _skip_if_api_not_running():
        return "SKIP"

    import httpx
    resp = httpx.post(
        "http://localhost:8000/predict",
        json={"features": {"V1": 1.0}},  # Only 1 of 37 features
        timeout=10.0,
    )
    assert resp.status_code == 422, f"Expected 422, got {resp.status_code}"


def test_batch_predict_endpoint():
    """POST /predict/batch returns predictions for multiple transactions."""
    if _skip_if_api_not_running():
        return "SKIP"

    import httpx

    sample_resp = httpx.get("http://localhost:8000/sample", timeout=10.0)
    sample = sample_resp.json()

    # Send 3 identical transactions
    batch_payload = {
        "transactions": [
            {"features": sample["features"]},
            {"features": sample["features"]},
            {"features": sample["features"]},
        ]
    }

    resp = httpx.post(
        "http://localhost:8000/predict/batch",
        json=batch_payload,
        timeout=60.0,
    )
    assert resp.status_code == 200

    data = resp.json()
    assert data["total_transactions"] == 3
    assert len(data["predictions"]) == 3
    assert "batch_inference_time_ms" in data
    assert isinstance(data["flagged_count"], int)


def test_predict_lime_endpoint():
    """POST /predict/lime returns a LIME-explained prediction."""
    if _skip_if_api_not_running():
        return "SKIP"

    import httpx

    sample_resp = httpx.get("http://localhost:8000/sample", timeout=10.0)
    sample = sample_resp.json()

    resp = httpx.post(
        "http://localhost:8000/predict/lime",
        json={"features": sample["features"]},
        timeout=60.0,  # LIME is slow
    )
    # Accept 200 (success) or 503 (LIME not loaded)
    assert resp.status_code in (200, 503), f"Unexpected status: {resp.status_code}"

    if resp.status_code == 200:
        data = resp.json()
        assert "fraud_probability" in data
        assert "risk_level" in data
        assert "lime_explanation" in data
        assert isinstance(data["lime_explanation"], list)
        assert "plain_english_summary" in data


def test_openapi_docs_available():
    """Verify the auto-generated API docs are accessible."""
    if _skip_if_api_not_running():
        return "SKIP"

    import httpx
    resp = httpx.get("http://localhost:8000/docs", timeout=10.0)
    assert resp.status_code == 200
    assert "swagger" in resp.text.lower() or "openapi" in resp.text.lower() or \
           "Fraud Detection" in resp.text


# ============================================================
# TEST RUNNER
# ============================================================

ALL_TESTS = {
    "Group A: Code Structure (no server needed)": [
        ("test_inference_api_file_exists", test_inference_api_file_exists),
        ("test_api_module_imports", test_api_module_imports),
        ("test_pydantic_schemas", test_pydantic_schemas),
        ("test_risk_classification", test_risk_classification),
        ("test_plain_english_generator", test_plain_english_generator),
        ("test_model_service_init", test_model_service_init),
        ("test_config_has_api_section", test_config_has_api_section),
        ("test_fastapi_routes_registered", test_fastapi_routes_registered),
    ],
    "Group B: Live API Tests (server must be running)": [
        ("test_health_endpoint", test_health_endpoint),
        ("test_model_info_endpoint", test_model_info_endpoint),
        ("test_features_endpoint", test_features_endpoint),
        ("test_sample_endpoint", test_sample_endpoint),
        ("test_predict_endpoint", test_predict_endpoint),
        ("test_predict_shap_explanation_structure", test_predict_shap_explanation_structure),
        ("test_predict_missing_features", test_predict_missing_features),
        ("test_batch_predict_endpoint", test_batch_predict_endpoint),
        ("test_predict_lime_endpoint", test_predict_lime_endpoint),
        ("test_openapi_docs_available", test_openapi_docs_available),
    ],
}


def run_all():
    total = 0
    passed = 0
    failed = 0
    skipped = 0
    failures = []

    print("=" * 70)
    print("PHASE 8 VERIFICATION: Inference API (THE MESSENGER)")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    for group_name, tests in ALL_TESTS.items():
        print(f"\n--- {group_name} ---")
        for test_name, test_fn in tests:
            total += 1
            try:
                result = test_fn()
                if result == "SKIP":
                    skipped += 1
                    print(f"  SKIP  {test_name}")
                else:
                    passed += 1
                    print(f"  PASS  {test_name}")
            except Exception as e:
                failed += 1
                print(f"  FAIL  {test_name}")
                print(f"        -> {type(e).__name__}: {e}")
                failures.append((test_name, e))

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    if skipped > 0:
        print(f"  (Skipped tests require the API server to be running on localhost:8000)")
        print(f"  Start with: python -m src.api.inference_api")
        print(f"  Then re-run: python tests/run_phase8_tests.py")
    if failed == 0:
        print(f"Phase 8 VERIFIED -- All tests passed.")
    else:
        print(f"Phase 8 has {failed} failure(s):")
        for name, err in failures:
            print(f"  - {name}: {err}")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
