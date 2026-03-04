"""
Phase 6 Standalone Test Runner
================================
Runs all Phase 6 verification tests without requiring pytest.

Usage:
    python tests/run_phase6_tests.py

Expected: All tests pass after running:
    python -m src.testing.stress_test
"""

import os
import sys
import json
import traceback
from pathlib import Path

# Ensure project root is in path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.logger import load_config, resolve_path


# ============================================================
# Test Runner Framework
# ============================================================

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results = []

    def run_test(self, test_name, test_func):
        try:
            test_func()
            self.passed += 1
            self.results.append(("PASS", test_name, None))
            print(f"  PASS  {test_name}")
        except AssertionError as e:
            self.failed += 1
            self.results.append(("FAIL", test_name, str(e)))
            print(f"  FAIL  {test_name}")
            print(f"        -> {e}")
        except FileNotFoundError as e:
            self.skipped += 1
            self.results.append(("SKIP", test_name, str(e)))
            print(f"  SKIP  {test_name}")
            print(f"        -> {e}")
        except Exception as e:
            self.failed += 1
            self.results.append(("FAIL", test_name, str(e)))
            print(f"  FAIL  {test_name}")
            print(f"        -> {e}")
            traceback.print_exc()

    def summary(self):
        total = self.passed + self.failed + self.skipped
        print("")
        print("=" * 60)
        print(f"RESULTS: {self.passed} passed, {self.failed} failed, {self.skipped} skipped")
        print("=" * 60)
        if self.failed == 0 and self.skipped == 0:
            print("Phase 6 VERIFIED -- All tests passed.")
        elif self.failed == 0:
            print("Phase 6 PARTIAL -- Some tests skipped (run Phase 6 first).")
        else:
            print("Phase 6 HAS ISSUES -- See failed tests above.")
        print("")


def main():
    print("")
    print("=" * 60)
    print("PHASE 6 VERIFICATION: Adversarial Stress Testing")
    print("=" * 60)
    print("")

    runner = TestRunner()
    config = load_config()
    stress_cfg = config["stress_test"]

    # Load results for structural tests
    results_path = resolve_path(stress_cfg["results_path"])
    report_path = resolve_path(stress_cfg["report_path"])

    stress_results = None
    if results_path.exists():
        with open(results_path, "r") as f:
            stress_results = json.load(f)

    # ============================================================
    # GROUP A: Artifact Existence (4 tests)
    # ============================================================
    print("GROUP A: Artifact Existence")
    print("-" * 40)

    def test_results_json_exists():
        assert results_path.exists(), f"Missing: {results_path}"
    runner.run_test("test_stress_results_json_exists", test_results_json_exists)

    def test_report_txt_exists():
        assert report_path.exists(), f"Missing: {report_path}"
    runner.run_test("test_stress_report_txt_exists", test_report_txt_exists)

    def test_results_valid_json():
        assert results_path.exists(), "Results file not found"
        with open(results_path, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict), "Results must be a JSON object"
    runner.run_test("test_results_valid_json", test_results_valid_json)

    def test_report_not_empty():
        assert report_path.exists(), "Report file not found"
        content = report_path.read_text(encoding="utf-8")
        assert len(content) > 100, f"Report too short: {len(content)} chars"
    runner.run_test("test_report_not_empty", test_report_not_empty)

    print("")

    # ============================================================
    # GROUP B: Results Structure (5 tests)
    # ============================================================
    print("GROUP B: Results Structure")
    print("-" * 40)

    def test_has_metadata():
        assert stress_results is not None, "Results not loaded"
        assert "metadata" in stress_results, "Missing 'metadata'"
        meta = stress_results["metadata"]
        assert "timestamp" in meta, "Missing timestamp"
        assert "optimal_threshold" in meta, "Missing optimal_threshold"
        assert "n_fraud_samples_tested" in meta, "Missing n_fraud_samples_tested"
    runner.run_test("test_has_metadata", test_has_metadata)

    def test_has_summary():
        assert stress_results is not None, "Results not loaded"
        assert "summary" in stress_results, "Missing 'summary'"
        summary = stress_results["summary"]
        for key in ["total_tests", "passed", "warnings", "failed",
                     "verdict", "baseline_auprc", "worst_case_auprc"]:
            assert key in summary, f"Summary missing key: {key}"
    runner.run_test("test_has_summary", test_has_summary)

    def test_has_baseline():
        assert stress_results is not None, "Results not loaded"
        assert "baseline" in stress_results, "Missing 'baseline'"
        bl = stress_results["baseline"]
        assert "auprc" in bl, "Baseline missing AUPRC"
        assert "optimal_threshold" in bl, "Baseline missing optimal_threshold"
    runner.run_test("test_has_baseline", test_has_baseline)

    def test_has_all_categories():
        assert stress_results is not None, "Results not loaded"
        assert "tests" in stress_results, "Missing 'tests'"
        tests = stress_results["tests"]
        for cat in ["amount_reduction", "time_shift", "feature_noise", "combined_attack"]:
            assert cat in tests, f"Missing test category: {cat}"
    runner.run_test("test_has_all_test_categories", test_has_all_categories)

    def test_counts_add_up():
        assert stress_results is not None, "Results not loaded"
        summary = stress_results["summary"]
        total = summary["total_tests"]
        parts = summary["passed"] + summary["warnings"] + summary["failed"]
        assert total == parts, (
            f"Counts mismatch: {total} != {summary['passed']}+"
            f"{summary['warnings']}+{summary['failed']}"
        )
    runner.run_test("test_counts_add_up", test_counts_add_up)

    print("")

    # ============================================================
    # GROUP C: Metric Validity (5 tests)
    # ============================================================
    print("GROUP C: Metric Validity")
    print("-" * 40)

    def test_baseline_auprc_valid():
        assert stress_results is not None, "Results not loaded"
        auprc = stress_results["baseline"]["auprc"]
        assert 0 <= auprc <= 1, f"Baseline AUPRC out of range: {auprc}"
    runner.run_test("test_baseline_auprc_valid", test_baseline_auprc_valid)

    def test_all_auprcs_valid():
        assert stress_results is not None, "Results not loaded"
        for cat, tests in stress_results["tests"].items():
            for t in tests:
                assert 0 <= t["auprc"] <= 1, (
                    f"AUPRC out of range in {t['test_name']}: {t['auprc']}"
                )
    runner.run_test("test_all_auprcs_valid", test_all_auprcs_valid)

    def test_detection_rates_valid():
        assert stress_results is not None, "Results not loaded"
        for cat, tests in stress_results["tests"].items():
            for t in tests:
                rate = t["optimal_threshold"]["detection_rate"]
                assert 0 <= rate <= 1, (
                    f"Detection rate out of range in {t['test_name']}: {rate}"
                )
    runner.run_test("test_detection_rates_valid", test_detection_rates_valid)

    def test_precision_values_valid():
        assert stress_results is not None, "Results not loaded"
        for cat, tests in stress_results["tests"].items():
            for t in tests:
                prec = t["optimal_threshold"]["precision"]
                assert 0 <= prec <= 1, (
                    f"Precision out of range in {t['test_name']}: {prec}"
                )
    runner.run_test("test_precision_values_valid", test_precision_values_valid)

    def test_fraud_probability_ranges():
        assert stress_results is not None, "Results not loaded"
        for cat, tests in stress_results["tests"].items():
            for t in tests:
                assert 0 <= t["mean_fraud_probability"] <= 1, (
                    f"Mean prob out of range in {t['test_name']}"
                )
                assert t["min_fraud_probability"] <= t["max_fraud_probability"], (
                    f"Min > Max in {t['test_name']}"
                )
    runner.run_test("test_fraud_probability_ranges", test_fraud_probability_ranges)

    print("")

    # ============================================================
    # GROUP D: Verdict Logic (3 tests)
    # ============================================================
    print("GROUP D: Verdict Logic")
    print("-" * 40)

    def test_verdict_valid_string():
        assert stress_results is not None, "Results not loaded"
        valid = ["PRODUCTION READY", "CONDITIONALLY READY", "NEEDS IMPROVEMENT"]
        verdict = stress_results["summary"]["verdict"]
        assert verdict in valid, f"Unknown verdict: {verdict}"
    runner.run_test("test_verdict_is_valid_string", test_verdict_valid_string)

    def test_verdict_matches_counts():
        assert stress_results is not None, "Results not loaded"
        summary = stress_results["summary"]
        if summary["failed"] == 0 and summary["warnings"] == 0:
            expected = "PRODUCTION READY"
        elif summary["failed"] == 0:
            expected = "CONDITIONALLY READY"
        else:
            expected = "NEEDS IMPROVEMENT"
        assert summary["verdict"] == expected, (
            f"Verdict mismatch: got '{summary['verdict']}', "
            f"expected '{expected}'"
        )
    runner.run_test("test_verdict_matches_counts", test_verdict_matches_counts)

    def test_worst_case_not_better_than_baseline():
        assert stress_results is not None, "Results not loaded"
        summary = stress_results["summary"]
        assert summary["worst_case_auprc"] <= summary["baseline_auprc"] + 0.01, (
            "Worst case should not significantly exceed baseline"
        )
    runner.run_test("test_worst_case_reasonable", test_worst_case_not_better_than_baseline)

    print("")

    # ============================================================
    # Summary
    # ============================================================
    runner.summary()


if __name__ == "__main__":
    main()
