"""
Phase 7 Standalone Test Runner
================================
Run: python tests/run_phase7_tests.py

Runs all Phase 7 tests without requiring pytest.
Group A tests always run. Group B tests are skipped if artifacts are missing.
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PASS = 0
FAIL = 0
SKIP = 0


def run_test(name, func, skip_condition=False, skip_reason=""):
    global PASS, FAIL, SKIP
    if skip_condition:
        SKIP += 1
        print(f"  SKIP  {name} -- {skip_reason}")
        return
    try:
        func()
        PASS += 1
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL += 1
        print(f"  FAIL  {name}")
        print(f"        -> {e}")
        traceback.print_exc()


def artifacts_exist():
    """Check if Phase 7 artifacts exist."""
    try:
        from src.utils.logger import load_config, resolve_path
        config = load_config()
        xai_cfg = config["explainability"]
        paths = [
            resolve_path(xai_cfg["shap_explainer_path"]),
            resolve_path(xai_cfg["shap_values_path"]),
            resolve_path(xai_cfg["lime_explainer_path"]),
            resolve_path("reports/xai_report.json"),
        ]
        return all(p.exists() for p in paths)
    except Exception:
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Phase 7: Explainable AI (XAI) -- Test Suite")
    print("=" * 60)

    from tests.test_phase7 import (
        # Group A
        test_xai_engine_imports,
        test_shap_library_available,
        test_lime_library_available,
        test_config_has_explainability_section,
        test_global_feature_importance_function,
        test_global_feature_importance_no_names,
        test_load_optimal_threshold,
        test_load_feature_names,
        # Group B
        test_shap_explainer_exists,
        test_shap_values_exist,
        test_lime_explainer_exists,
        test_xai_report_json_exists,
        test_xai_report_txt_exists,
        test_shap_explainer_is_loadable,
        test_shap_values_shape,
        test_lime_explainer_is_functional,
        test_global_importance_has_entries,
        test_example_explanations_have_plain_english,
    )

    has_artifacts = artifacts_exist()
    skip_reason = "Phase 7 not yet executed -- run xai_engine.py first"

    print("\n--- Group A: Code Structure & Imports (always run) ---\n")

    run_test("test_xai_engine_imports", test_xai_engine_imports)
    run_test("test_shap_library_available", test_shap_library_available)
    run_test("test_lime_library_available", test_lime_library_available)
    run_test("test_config_has_explainability_section", test_config_has_explainability_section)
    run_test("test_global_feature_importance_function", test_global_feature_importance_function)
    run_test("test_global_feature_importance_no_names", test_global_feature_importance_no_names)
    run_test("test_load_optimal_threshold", test_load_optimal_threshold)
    run_test("test_load_feature_names", test_load_feature_names)

    print("\n--- Group B: Artifact Verification (require Phase 7 run) ---\n")

    run_test("test_shap_explainer_exists", test_shap_explainer_exists,
             not has_artifacts, skip_reason)
    run_test("test_shap_values_exist", test_shap_values_exist,
             not has_artifacts, skip_reason)
    run_test("test_lime_explainer_exists", test_lime_explainer_exists,
             not has_artifacts, skip_reason)
    run_test("test_xai_report_json_exists", test_xai_report_json_exists,
             not has_artifacts, skip_reason)
    run_test("test_xai_report_txt_exists", test_xai_report_txt_exists,
             not has_artifacts, skip_reason)
    run_test("test_shap_explainer_is_loadable", test_shap_explainer_is_loadable,
             not has_artifacts, skip_reason)
    run_test("test_shap_values_shape", test_shap_values_shape,
             not has_artifacts, skip_reason)
    run_test("test_lime_explainer_is_functional", test_lime_explainer_is_functional,
             not has_artifacts, skip_reason)
    run_test("test_global_importance_has_entries", test_global_importance_has_entries,
             not has_artifacts, skip_reason)
    run_test("test_example_explanations_have_plain_english", test_example_explanations_have_plain_english,
             not has_artifacts, skip_reason)

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print("=" * 60)

    if FAIL > 0:
        print("\nSome tests FAILED. Check the errors above.")
        if SKIP > 0:
            print("After running Phase 7, re-run this script to verify all tests pass.")
    else:
        if SKIP > 0:
            print(f"\n{SKIP} tests were skipped (Phase 7 not yet executed).")
            print("Run Phase 7 first, then re-run this script.")
        else:
            print("\nPhase 7 VERIFIED -- All tests passed.")
