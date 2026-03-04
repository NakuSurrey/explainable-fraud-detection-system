"""
Phase 6 Tests (pytest style)
==============================
Validates all stress test artifacts and functionality.

Usage:
    python -m pytest tests/test_phase6.py -v
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is in path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.utils.logger import load_config, resolve_path


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def stress_results(config):
    results_path = resolve_path(config["stress_test"]["results_path"])
    if not results_path.exists():
        pytest.skip("Stress test results not found. Run Phase 6 first.")
    with open(results_path, "r") as f:
        return json.load(f)


# ============================================================
# GROUP A: Artifact Existence Tests
# ============================================================

class TestArtifactsExist:
    """Verify all Phase 6 artifacts were created."""

    def test_stress_results_json_exists(self, config):
        path = resolve_path(config["stress_test"]["results_path"])
        assert path.exists(), f"Missing: {path}"

    def test_stress_report_txt_exists(self, config):
        path = resolve_path(config["stress_test"]["report_path"])
        assert path.exists(), f"Missing: {path}"

    def test_results_json_valid(self, config):
        path = resolve_path(config["stress_test"]["results_path"])
        with open(path, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict), "Results must be a JSON object"

    def test_report_not_empty(self, config):
        path = resolve_path(config["stress_test"]["report_path"])
        content = path.read_text(encoding="utf-8")
        assert len(content) > 100, "Report is too short to be valid"


# ============================================================
# GROUP B: Results Structure Tests
# ============================================================

class TestResultsStructure:
    """Verify the stress test results have correct structure."""

    def test_has_metadata(self, stress_results):
        assert "metadata" in stress_results
        meta = stress_results["metadata"]
        assert "timestamp" in meta
        assert "optimal_threshold" in meta
        assert "n_fraud_samples_tested" in meta

    def test_has_summary(self, stress_results):
        assert "summary" in stress_results
        summary = stress_results["summary"]
        required_keys = [
            "total_tests", "passed", "warnings", "failed",
            "verdict", "baseline_auprc", "worst_case_auprc",
        ]
        for key in required_keys:
            assert key in summary, f"Summary missing key: {key}"

    def test_has_baseline(self, stress_results):
        assert "baseline" in stress_results
        bl = stress_results["baseline"]
        assert "auprc" in bl
        assert "optimal_threshold" in bl

    def test_has_all_test_categories(self, stress_results):
        assert "tests" in stress_results
        tests = stress_results["tests"]
        expected = ["amount_reduction", "time_shift", "feature_noise", "combined_attack"]
        for cat in expected:
            assert cat in tests, f"Missing test category: {cat}"

    def test_test_counts_add_up(self, stress_results):
        summary = stress_results["summary"]
        total = summary["total_tests"]
        parts = summary["passed"] + summary["warnings"] + summary["failed"]
        assert total == parts, (
            f"Test counts don't add up: {total} != "
            f"{summary['passed']}+{summary['warnings']}+{summary['failed']}"
        )


# ============================================================
# GROUP C: Metric Validity Tests
# ============================================================

class TestMetricValidity:
    """Verify all metrics are within valid ranges."""

    def test_baseline_auprc_valid(self, stress_results):
        auprc = stress_results["baseline"]["auprc"]
        assert 0 <= auprc <= 1, f"Baseline AUPRC out of range: {auprc}"

    def test_all_auprcs_valid(self, stress_results):
        for category, tests in stress_results["tests"].items():
            for t in tests:
                assert 0 <= t["auprc"] <= 1, (
                    f"AUPRC out of range in {t['test_name']}: {t['auprc']}"
                )

    def test_detection_rates_valid(self, stress_results):
        for category, tests in stress_results["tests"].items():
            for t in tests:
                rate = t["optimal_threshold"]["detection_rate"]
                assert 0 <= rate <= 1, (
                    f"Detection rate out of range in {t['test_name']}: {rate}"
                )

    def test_precision_values_valid(self, stress_results):
        for category, tests in stress_results["tests"].items():
            for t in tests:
                prec = t["optimal_threshold"]["precision"]
                assert 0 <= prec <= 1, (
                    f"Precision out of range in {t['test_name']}: {prec}"
                )

    def test_fraud_probability_ranges(self, stress_results):
        for category, tests in stress_results["tests"].items():
            for t in tests:
                assert 0 <= t["mean_fraud_probability"] <= 1, (
                    f"Mean prob out of range in {t['test_name']}"
                )
                assert t["min_fraud_probability"] <= t["max_fraud_probability"], (
                    f"Min > Max in {t['test_name']}"
                )


# ============================================================
# GROUP D: Verdict Logic Tests
# ============================================================

class TestVerdictLogic:
    """Verify the verdict was determined correctly."""

    def test_verdict_is_valid_string(self, stress_results):
        valid = ["PRODUCTION READY", "CONDITIONALLY READY", "NEEDS IMPROVEMENT"]
        verdict = stress_results["summary"]["verdict"]
        assert verdict in valid, f"Unknown verdict: {verdict}"

    def test_verdict_matches_counts(self, stress_results):
        summary = stress_results["summary"]
        if summary["failed"] == 0 and summary["warnings"] == 0:
            assert summary["verdict"] == "PRODUCTION READY"
        elif summary["failed"] == 0:
            assert summary["verdict"] == "CONDITIONALLY READY"
        else:
            assert summary["verdict"] == "NEEDS IMPROVEMENT"

    def test_worst_case_not_better_than_baseline(self, stress_results):
        summary = stress_results["summary"]
        # Worst case should typically not exceed baseline
        # (small numerical fluctuations are okay)
        assert summary["worst_case_auprc"] <= summary["baseline_auprc"] + 0.01, (
            "Worst case AUPRC should not significantly exceed baseline"
        )
