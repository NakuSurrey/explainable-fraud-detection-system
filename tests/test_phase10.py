"""
Phase 10 Test Suite: Human-in-the-Loop & CI/CD (THE SENTINEL)
==============================================================
Tests for the feedback manager, API feedback endpoints, dashboard
feedback integration, and CI/CD pipeline configuration.

Groups:
    Group A (always runnable, no server/artifacts needed) — 14 tests
    Group B (require running API server on localhost:8000) — 8 tests

Usage:
    python tests/run_phase10_tests.py
    python -m pytest tests/test_phase10.py -v
"""

import os
import sys
import json
import sqlite3
import hashlib
import tempfile
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ============================================================
# GROUP A: CODE STRUCTURE & OFFLINE TESTS (no server needed)
# ============================================================

# --- A1: Feedback module files exist ---
def test_feedback_module_exists():
    """Verify src/feedback/__init__.py and feedback_manager.py exist."""
    init_path = _PROJECT_ROOT / "src" / "feedback" / "__init__.py"
    manager_path = _PROJECT_ROOT / "src" / "feedback" / "feedback_manager.py"
    assert init_path.exists(), f"Missing: {init_path}"
    assert manager_path.exists(), f"Missing: {manager_path}"
    # Check file has substantial content
    content = manager_path.read_text(encoding="utf-8")
    assert len(content) > 1000, f"feedback_manager.py is too small ({len(content)} chars)"


# --- A2: Feedback manager imports ---
def test_feedback_manager_imports():
    """Verify all functions are importable from feedback_manager."""
    from src.feedback.feedback_manager import (
        generate_transaction_id,
        init_db,
        save_feedback,
        get_feedback_history,
        get_feedback_by_id,
        get_feedback_by_transaction,
        update_feedback,
        delete_feedback,
        export_corrections,
        check_retrain_threshold,
        get_feedback_stats,
    )
    # All imported successfully
    assert callable(generate_transaction_id)
    assert callable(init_db)
    assert callable(save_feedback)
    assert callable(get_feedback_history)
    assert callable(get_feedback_by_id)
    assert callable(get_feedback_by_transaction)
    assert callable(update_feedback)
    assert callable(delete_feedback)
    assert callable(export_corrections)
    assert callable(check_retrain_threshold)
    assert callable(get_feedback_stats)


# --- A3: Transaction ID generation ---
def test_transaction_id_generation():
    """Verify transaction ID is deterministic and correct format."""
    from src.feedback.feedback_manager import generate_transaction_id

    features = {"V1": -1.35, "V2": 1.19, "Amount": 149.62}

    # Same input produces same ID
    id1 = generate_transaction_id(features)
    id2 = generate_transaction_id(features)
    assert id1 == id2, "Same features must produce same transaction ID"

    # ID is 16 hex characters
    assert len(id1) == 16, f"Transaction ID should be 16 chars, got {len(id1)}"
    assert all(c in "0123456789abcdef" for c in id1), "Transaction ID must be hex"

    # Different input produces different ID
    features2 = {"V1": -1.35, "V2": 1.19, "Amount": 200.00}
    id3 = generate_transaction_id(features2)
    assert id1 != id3, "Different features must produce different IDs"

    # Order doesn't matter (sorted internally)
    features_reordered = {"Amount": 149.62, "V2": 1.19, "V1": -1.35}
    id4 = generate_transaction_id(features_reordered)
    assert id1 == id4, "Dict ordering should not affect transaction ID"


# --- A4: Database initialization ---
def test_db_initialization():
    """Verify init_db creates the database and table."""
    from src.feedback.feedback_manager import init_db, _get_connection

    # Use a temporary config override
    with tempfile.TemporaryDirectory() as tmpdir:
        test_config = {
            "feedback": {
                "db_path": os.path.join(tmpdir, "test_feedback.db"),
                "retrain_threshold": 100,
                "export_path": os.path.join(tmpdir, "test_export.csv"),
            }
        }

        db_path = init_db(config=test_config)
        assert os.path.exists(db_path), "Database file should be created"

        # Verify table exists
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback'")
        tables = cursor.fetchall()
        conn.close()
        assert len(tables) == 1, "Feedback table should exist"


# --- A5: Save and retrieve feedback ---
def test_save_and_retrieve_feedback():
    """Verify saving and querying feedback records."""
    from src.feedback.feedback_manager import init_db, save_feedback, get_feedback_by_id

    with tempfile.TemporaryDirectory() as tmpdir:
        test_config = {
            "feedback": {
                "db_path": os.path.join(tmpdir, "test_feedback.db"),
                "retrain_threshold": 100,
                "export_path": os.path.join(tmpdir, "test_export.csv"),
            }
        }

        init_db(config=test_config)

        # Save a record
        record = save_feedback(
            transaction_id="abc123def456",
            original_probability=0.85,
            original_risk_level="CRITICAL",
            original_is_flagged=True,
            correction_type="false_positive",
            investigator_notes="Customer confirmed purchase",
            config=test_config,
        )

        assert record["id"] == 1, "First record should have id=1"
        assert record["transaction_id"] == "abc123def456"
        assert record["correction_type"] == "false_positive"

        # Retrieve it
        fetched = get_feedback_by_id(1, config=test_config)
        assert fetched is not None, "Record should be retrievable"
        assert fetched["original_probability"] == 0.85
        assert fetched["original_risk_level"] == "CRITICAL"
        assert fetched["original_is_flagged"] == True
        assert fetched["investigator_notes"] == "Customer confirmed purchase"


# --- A6: Invalid correction type ---
def test_invalid_correction_type():
    """Verify ValueError for invalid correction_type."""
    from src.feedback.feedback_manager import init_db, save_feedback

    with tempfile.TemporaryDirectory() as tmpdir:
        test_config = {
            "feedback": {
                "db_path": os.path.join(tmpdir, "test_feedback.db"),
                "retrain_threshold": 100,
                "export_path": os.path.join(tmpdir, "test_export.csv"),
            }
        }
        init_db(config=test_config)

        try:
            save_feedback(
                transaction_id="test123",
                original_probability=0.5,
                original_risk_level="MEDIUM",
                original_is_flagged=True,
                correction_type="invalid_type",
                config=test_config,
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "invalid_type" in str(e)


# --- A7: Update feedback ---
def test_update_feedback():
    """Verify updating a feedback record."""
    from src.feedback.feedback_manager import init_db, save_feedback, update_feedback, get_feedback_by_id

    with tempfile.TemporaryDirectory() as tmpdir:
        test_config = {
            "feedback": {
                "db_path": os.path.join(tmpdir, "test_feedback.db"),
                "retrain_threshold": 100,
                "export_path": os.path.join(tmpdir, "test_export.csv"),
            }
        }
        init_db(config=test_config)

        save_feedback(
            transaction_id="update_test",
            original_probability=0.3,
            original_risk_level="MEDIUM",
            original_is_flagged=True,
            correction_type="confirmed_fraud",
            investigator_notes="Original note",
            config=test_config,
        )

        # Update correction type and notes
        updated = update_feedback(
            feedback_id=1,
            correction_type="false_positive",
            investigator_notes="Changed my mind",
            config=test_config,
        )

        assert updated is not None
        assert updated["correction_type"] == "false_positive"
        assert updated["investigator_notes"] == "Changed my mind"
        assert updated["updated_at"] is not None


# --- A8: Delete feedback ---
def test_delete_feedback():
    """Verify deleting a feedback record."""
    from src.feedback.feedback_manager import init_db, save_feedback, delete_feedback, get_feedback_by_id

    with tempfile.TemporaryDirectory() as tmpdir:
        test_config = {
            "feedback": {
                "db_path": os.path.join(tmpdir, "test_feedback.db"),
                "retrain_threshold": 100,
                "export_path": os.path.join(tmpdir, "test_export.csv"),
            }
        }
        init_db(config=test_config)

        save_feedback(
            transaction_id="delete_test",
            original_probability=0.9,
            original_risk_level="CRITICAL",
            original_is_flagged=True,
            correction_type="confirmed_fraud",
            config=test_config,
        )

        # Delete it
        deleted = delete_feedback(1, config=test_config)
        assert deleted == True, "Should return True for successful delete"

        # Verify it's gone
        fetched = get_feedback_by_id(1, config=test_config)
        assert fetched is None, "Deleted record should not be retrievable"

        # Delete non-existent record
        deleted2 = delete_feedback(999, config=test_config)
        assert deleted2 == False, "Should return False for non-existent record"


# --- A9: Feedback history with pagination ---
def test_feedback_history():
    """Verify paginated feedback history query."""
    from src.feedback.feedback_manager import init_db, save_feedback, get_feedback_history

    with tempfile.TemporaryDirectory() as tmpdir:
        test_config = {
            "feedback": {
                "db_path": os.path.join(tmpdir, "test_feedback.db"),
                "retrain_threshold": 100,
                "export_path": os.path.join(tmpdir, "test_export.csv"),
            }
        }
        init_db(config=test_config)

        # Insert 5 records
        for i in range(5):
            save_feedback(
                transaction_id=f"hist_test_{i}",
                original_probability=0.1 * (i + 1),
                original_risk_level="LOW",
                original_is_flagged=False,
                correction_type="confirmed_fraud" if i % 2 == 0 else "false_positive",
                config=test_config,
            )

        # Query all
        result = get_feedback_history(limit=50, config=test_config)
        assert result["total"] == 5
        assert len(result["records"]) == 5

        # Query with pagination
        page1 = get_feedback_history(limit=2, offset=0, config=test_config)
        assert len(page1["records"]) == 2
        assert page1["total"] == 5

        # Query with filter
        confirmed = get_feedback_history(correction_type="confirmed_fraud", config=test_config)
        assert confirmed["total"] == 3  # indices 0, 2, 4


# --- A10: Export corrections to CSV ---
def test_export_corrections():
    """Verify CSV export functionality."""
    from src.feedback.feedback_manager import init_db, save_feedback, export_corrections

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_export.csv")
        test_config = {
            "feedback": {
                "db_path": os.path.join(tmpdir, "test_feedback.db"),
                "retrain_threshold": 100,
                "export_path": csv_path,
            }
        }
        init_db(config=test_config)

        # Insert records
        save_feedback(
            transaction_id="export_1",
            original_probability=0.7,
            original_risk_level="HIGH",
            original_is_flagged=True,
            correction_type="false_positive",
            investigator_notes="Test export",
            config=test_config,
        )

        # Export
        result = export_corrections(config=test_config)
        assert result["total_records"] == 1
        assert os.path.exists(result["path"])

        # Verify CSV content
        with open(result["path"], "r") as f:
            content = f.read()
        assert "export_1" in content
        assert "false_positive" in content
        assert "Test export" in content


# --- A11: Retrain threshold check ---
def test_retrain_threshold():
    """Verify retrain threshold logic."""
    from src.feedback.feedback_manager import init_db, save_feedback, check_retrain_threshold

    with tempfile.TemporaryDirectory() as tmpdir:
        test_config = {
            "feedback": {
                "db_path": os.path.join(tmpdir, "test_feedback.db"),
                "retrain_threshold": 3,  # Low threshold for testing
                "export_path": os.path.join(tmpdir, "test_export.csv"),
            }
        }
        init_db(config=test_config)

        # Before threshold
        status = check_retrain_threshold(config=test_config)
        assert status["total_corrections"] == 0
        assert status["threshold"] == 3
        assert status["retrain_recommended"] == False
        assert status["remaining"] == 3

        # Add 3 records to hit threshold
        for i in range(3):
            save_feedback(
                transaction_id=f"retrain_{i}",
                original_probability=0.5,
                original_risk_level="HIGH",
                original_is_flagged=True,
                correction_type="confirmed_fraud",
                config=test_config,
            )

        # After threshold
        status2 = check_retrain_threshold(config=test_config)
        assert status2["total_corrections"] == 3
        assert status2["retrain_recommended"] == True
        assert status2["remaining"] == 0


# --- A12: Feedback statistics ---
def test_feedback_stats():
    """Verify feedback statistics calculation."""
    from src.feedback.feedback_manager import init_db, save_feedback, get_feedback_stats

    with tempfile.TemporaryDirectory() as tmpdir:
        test_config = {
            "feedback": {
                "db_path": os.path.join(tmpdir, "test_feedback.db"),
                "retrain_threshold": 100,
                "export_path": os.path.join(tmpdir, "test_export.csv"),
            }
        }
        init_db(config=test_config)

        # Empty stats
        stats = get_feedback_stats(config=test_config)
        assert stats["total"] == 0

        # Add records
        save_feedback("s1", 0.9, "CRITICAL", True, "confirmed_fraud", config=test_config)
        save_feedback("s2", 0.1, "LOW", False, "false_positive", config=test_config)
        save_feedback("s3", 0.8, "HIGH", True, "confirmed_fraud", config=test_config)

        stats2 = get_feedback_stats(config=test_config)
        assert stats2["total"] == 3
        assert stats2["confirmed_fraud"] == 2
        assert stats2["false_positive"] == 1
        assert stats2["avg_probability_confirmed_fraud"] is not None
        assert stats2["oldest_feedback"] is not None
        assert stats2["newest_feedback"] is not None


# --- A13: Config has feedback section ---
def test_config_has_feedback_section():
    """Verify config.yaml has all Phase 10 feedback settings."""
    from src.utils.logger import load_config

    config = load_config()
    assert "feedback" in config, "config.yaml missing 'feedback' section"
    fb = config["feedback"]
    assert "db_path" in fb, "Missing feedback.db_path"
    assert "retrain_threshold" in fb, "Missing feedback.retrain_threshold"
    assert "export_path" in fb, "Missing feedback.export_path"
    assert fb["retrain_threshold"] == 100

    assert "cicd" in config, "config.yaml missing 'cicd' section"
    ci = config["cicd"]
    assert "test_command" in ci, "Missing cicd.test_command"
    assert "min_auprc_threshold" in ci, "Missing cicd.min_auprc_threshold"


# --- A14: CI/CD workflow file exists ---
def test_cicd_workflow_exists():
    """Verify .github/workflows/test.yml exists and has correct structure."""
    workflow_path = _PROJECT_ROOT / ".github" / "workflows" / "test.yml"
    assert workflow_path.exists(), f"CI/CD workflow missing: {workflow_path}"

    content = workflow_path.read_text(encoding="utf-8")
    assert len(content) > 100, "Workflow file is too small"

    # Check key elements
    assert "pytest" in content, "Workflow should run pytest"
    assert "python" in content.lower(), "Workflow should set up Python"
    assert "ubuntu" in content.lower(), "Workflow should run on Ubuntu"
    assert "push" in content, "Workflow should trigger on push"
    assert "requirements.txt" in content, "Workflow should install requirements"


# ============================================================
# GROUP B: LIVE API TESTS (require server on localhost:8000)
# ============================================================

def _api_is_running():
    """Check if the API server is running."""
    try:
        import requests
        r = requests.get("http://localhost:8000/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# --- B1: Feedback submission endpoint ---
def test_feedback_submit_endpoint():
    """POST /feedback should accept and store feedback."""
    if not _api_is_running():
        return "SKIP"

    import requests

    # First get a sample transaction
    sample = requests.get("http://localhost:8000/sample", timeout=5).json()
    features = sample.get("features", {})

    # Submit feedback
    payload = {
        "features": features,
        "original_probability": 0.75,
        "original_risk_level": "HIGH",
        "original_is_flagged": True,
        "correction_type": "false_positive",
        "investigator_notes": "Test from Phase 10 test suite",
    }
    r = requests.post("http://localhost:8000/feedback", json=payload, timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"

    data = r.json()
    assert data["success"] == True
    assert data["feedback_id"] > 0
    assert len(data["transaction_id"]) == 16
    assert "retrain_status" in data


# --- B2: Feedback history endpoint ---
def test_feedback_history_endpoint():
    """GET /feedback/history should return paginated results."""
    if not _api_is_running():
        return "SKIP"

    import requests
    r = requests.get("http://localhost:8000/feedback/history", params={"limit": 10}, timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    data = r.json()
    assert "records" in data
    assert "total" in data
    assert isinstance(data["records"], list)


# --- B3: Feedback stats endpoint ---
def test_feedback_stats_endpoint():
    """GET /feedback/stats should return statistics."""
    if not _api_is_running():
        return "SKIP"

    import requests
    r = requests.get("http://localhost:8000/feedback/stats", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    data = r.json()
    assert "stats" in data
    assert "retrain_status" in data
    assert "total" in data["stats"]


# --- B4: Feedback export endpoint ---
def test_feedback_export_endpoint():
    """GET /feedback/export should export to CSV."""
    if not _api_is_running():
        return "SKIP"

    import requests
    r = requests.get("http://localhost:8000/feedback/export", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"

    data = r.json()
    assert data["success"] == True
    assert "total_records" in data
    assert "path" in data


# --- B5: Feedback with invalid type returns 422 ---
def test_feedback_invalid_type_endpoint():
    """POST /feedback with invalid correction_type should return 422."""
    if not _api_is_running():
        return "SKIP"

    import requests
    sample = requests.get("http://localhost:8000/sample", timeout=5).json()
    features = sample.get("features", {})

    payload = {
        "features": features,
        "original_probability": 0.5,
        "original_risk_level": "MEDIUM",
        "original_is_flagged": True,
        "correction_type": "totally_wrong",
        "investigator_notes": "",
    }
    r = requests.post("http://localhost:8000/feedback", json=payload, timeout=10)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"


# --- B6: Dashboard file has feedback integration ---
def test_dashboard_has_feedback_ui():
    """Verify app.py has Phase 10 feedback elements."""
    dashboard_path = _PROJECT_ROOT / "src" / "dashboard" / "app.py"
    assert dashboard_path.exists(), "Dashboard app.py not found"

    content = dashboard_path.read_text(encoding="utf-8")

    assert "submit_feedback" in content, "Dashboard should have submit_feedback function"
    assert "feedback_submitted" in content, "Dashboard should have feedback_submitted session state"
    assert "Confirmed Fraud" in content, "Dashboard should have Confirmed Fraud button"
    assert "False Positive" in content, "Dashboard should have False Positive button"
    assert "/feedback" in content, "Dashboard should call /feedback endpoint"
    assert "Investigator" in content, "Dashboard should reference investigator"
    assert "feedback_history" in content.lower() or "feedback/history" in content, "Dashboard should show feedback history"


# --- B7: API has feedback endpoints registered ---
def test_api_has_feedback_routes():
    """Verify the API has all feedback routes registered."""
    if not _api_is_running():
        return "SKIP"

    import requests
    # Check OpenAPI spec for feedback routes
    r = requests.get("http://localhost:8000/openapi.json", timeout=5)
    assert r.status_code == 200

    spec = r.json()
    paths = spec.get("paths", {})

    assert "/feedback" in paths, "API missing /feedback endpoint"
    assert "/feedback/history" in paths, "API missing /feedback/history endpoint"
    assert "/feedback/stats" in paths, "API missing /feedback/stats endpoint"
    assert "/feedback/export" in paths, "API missing /feedback/export endpoint"


# --- B8: End-to-end predict then feedback ---
def test_predict_then_feedback_flow():
    """Full flow: predict a transaction, then submit feedback on it."""
    if not _api_is_running():
        return "SKIP"

    import requests

    # Step 1: Get sample
    sample = requests.get("http://localhost:8000/sample", timeout=5).json()
    features = sample.get("features", {})

    # Step 2: Predict
    pred_r = requests.post("http://localhost:8000/predict", json={"features": features}, timeout=15)
    assert pred_r.status_code == 200
    prediction = pred_r.json()

    # Step 3: Submit feedback
    feedback_payload = {
        "features": features,
        "original_probability": prediction["fraud_probability"],
        "original_risk_level": prediction["risk_level"],
        "original_is_flagged": prediction["is_flagged"],
        "correction_type": "confirmed_fraud",
        "investigator_notes": "E2E test: predicted then corrected",
    }
    fb_r = requests.post("http://localhost:8000/feedback", json=feedback_payload, timeout=10)
    assert fb_r.status_code == 200
    fb_data = fb_r.json()
    assert fb_data["success"] == True

    # Step 4: Verify it appears in history
    hist_r = requests.get("http://localhost:8000/feedback/history", params={"limit": 1}, timeout=10)
    assert hist_r.status_code == 200
    hist = hist_r.json()
    assert hist["total"] > 0
    assert len(hist["records"]) > 0


# ============================================================
# TEST RUNNER
# ============================================================

def run_all():
    """Run all Phase 10 tests and report results."""
    import time

    print("=" * 70)
    print("PHASE 10 VERIFICATION: Human-in-the-Loop & CI/CD (THE SENTINEL)")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    group_a_tests = [
        ("test_feedback_module_exists", test_feedback_module_exists),
        ("test_feedback_manager_imports", test_feedback_manager_imports),
        ("test_transaction_id_generation", test_transaction_id_generation),
        ("test_db_initialization", test_db_initialization),
        ("test_save_and_retrieve_feedback", test_save_and_retrieve_feedback),
        ("test_invalid_correction_type", test_invalid_correction_type),
        ("test_update_feedback", test_update_feedback),
        ("test_delete_feedback", test_delete_feedback),
        ("test_feedback_history", test_feedback_history),
        ("test_export_corrections", test_export_corrections),
        ("test_retrain_threshold", test_retrain_threshold),
        ("test_feedback_stats", test_feedback_stats),
        ("test_config_has_feedback_section", test_config_has_feedback_section),
        ("test_cicd_workflow_exists", test_cicd_workflow_exists),
    ]

    group_b_tests = [
        ("test_feedback_submit_endpoint", test_feedback_submit_endpoint),
        ("test_feedback_history_endpoint", test_feedback_history_endpoint),
        ("test_feedback_stats_endpoint", test_feedback_stats_endpoint),
        ("test_feedback_export_endpoint", test_feedback_export_endpoint),
        ("test_feedback_invalid_type_endpoint", test_feedback_invalid_type_endpoint),
        ("test_dashboard_has_feedback_ui", test_dashboard_has_feedback_ui),
        ("test_api_has_feedback_routes", test_api_has_feedback_routes),
        ("test_predict_then_feedback_flow", test_predict_then_feedback_flow),
    ]

    passed = 0
    failed = 0
    skipped = 0

    print("\n--- Group A: Code Structure & Offline Tests (no server needed) ---")
    for name, func in group_a_tests:
        try:
            result = func()
            if result == "SKIP":
                print(f"  SKIP  {name}")
                skipped += 1
            else:
                print(f"  PASS  {name}")
                passed += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            print(f"        Error: {e}")
            failed += 1

    print("\n--- Group B: Live API Tests (server must be running) ---")
    for name, func in group_b_tests:
        try:
            result = func()
            if result == "SKIP":
                print(f"  SKIP  {name}")
                skipped += 1
            else:
                print(f"  PASS  {name}")
                passed += 1
        except Exception as e:
            print(f"  FAIL  {name}")
            print(f"        Error: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    if failed == 0 and skipped == 0:
        print("Phase 10 VERIFIED -- All tests passed.")
    elif failed == 0:
        print("Phase 10 PARTIAL -- Group A passed, Group B skipped (start API server).")
    else:
        print("Phase 10 ISSUES -- Some tests failed. Review errors above.")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
