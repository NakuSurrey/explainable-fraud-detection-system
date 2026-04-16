"""
Phase 11 Verification Runner -- Run: python tests/run_phase11_tests.py
Works without pytest (for offline environments).

Tests are split into two groups:
  Group A: Always runnable (code structure, config, imports, offline logic)
  Group B: Require API running on localhost:8000 (endpoints, prediction logging)
"""
import sys
import traceback
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = 0
FAIL = 0
SKIP = 0


def run_test(name, func, skip_condition=False, skip_reason=""):
    global PASS, FAIL, SKIP
    if skip_condition:
        print(f"  SKIP  {name}")
        print(f"        -> {skip_reason}")
        SKIP += 1
        return
    try:
        func()
        print(f"  PASS  {name}")
        PASS += 1
    except Exception as e:
        print(f"  FAIL  {name}")
        print(f"        -> {e}")
        FAIL += 1


def api_reachable() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:8000/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def baseline_exists() -> bool:
    try:
        from src.utils.logger import load_config, resolve_path
        config = load_config()
        path = resolve_path(config["monitoring"]["baseline_path"])
        return path.exists()
    except Exception:
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 11 VERIFICATION: Model Monitoring (THE WATCHTOWER)")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # import all test functions from the pytest file
    from tests.test_phase11 import (
        TestGroupA_CodeStructureAndOffline,
        TestGroupB_APIEndpoints,
    )

    # create test instances
    group_a = TestGroupA_CodeStructureAndOffline()
    group_b = TestGroupB_APIEndpoints()

    has_baseline = baseline_exists()
    has_api = api_reachable()

    print("\n--- Group A: Code Structure & Offline Tests (no server needed) ---\n")

    run_test("test_monitoring_module_exists", group_a.test_monitoring_module_exists)
    run_test("test_monitoring_imports", group_a.test_monitoring_imports)
    run_test("test_monitoring_has_cli_support", group_a.test_monitoring_has_cli_support)
    run_test("test_config_has_monitoring_section", group_a.test_config_has_monitoring_section)
    run_test("test_monitoring_data_directory_exists", group_a.test_monitoring_data_directory_exists)
    run_test("test_db_initialization", group_a.test_db_initialization)
    run_test("test_psi_identical_distributions", group_a.test_psi_identical_distributions)
    run_test("test_psi_shifted_distribution", group_a.test_psi_shifted_distribution)
    run_test("test_psi_returns_per_bin_values", group_a.test_psi_returns_per_bin_values)
    run_test("test_baseline_generation_synthetic", group_a.test_baseline_generation_synthetic)
    run_test(
        "test_baseline_file_exists",
        group_a.test_baseline_file_exists,
        not has_baseline,
        "Baseline not yet generated — run --generate-baseline first",
    )
    run_test("test_dashboard_has_monitoring_tab", group_a.test_dashboard_has_monitoring_tab)
    run_test("test_api_has_monitoring_integration", group_a.test_api_has_monitoring_integration)
    run_test("test_monitoring_constants", group_a.test_monitoring_constants)

    print("\n--- Group B: API Endpoint Tests (require server on localhost:8000) ---\n")

    skip_api = not has_api
    skip_reason = "API not running on localhost:8000 — start with: python -m src.api.inference_api"

    run_test("test_monitoring_summary_endpoint", group_b.test_monitoring_summary_endpoint, skip_api, skip_reason)
    run_test("test_monitoring_predictions_endpoint", group_b.test_monitoring_predictions_endpoint, skip_api, skip_reason)
    run_test("test_predict_logs_to_monitoring", group_b.test_predict_logs_to_monitoring, skip_api, skip_reason)
    run_test("test_logged_prediction_structure", group_b.test_logged_prediction_structure, skip_api, skip_reason)
    run_test("test_drift_report_endpoint", group_b.test_drift_report_endpoint, skip_api, skip_reason)
    run_test("test_api_has_monitoring_routes", group_b.test_api_has_monitoring_routes, skip_api, skip_reason)
    run_test("test_health_endpoint_still_works", group_b.test_health_endpoint_still_works, skip_api, skip_reason)
    run_test("test_predict_then_verify_in_monitoring_log", group_b.test_predict_then_verify_in_monitoring_log, skip_api, skip_reason)

    # final summary
    print("\n" + "=" * 60)
    total = PASS + FAIL + SKIP
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    if FAIL == 0:
        print("Phase 11 VERIFIED -- All tests passed.")
    else:
        print(f"Phase 11 HAS FAILURES -- {FAIL} test(s) need attention.")
    print("=" * 60 + "\n")
