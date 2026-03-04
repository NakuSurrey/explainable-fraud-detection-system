"""
Phase 0 Verification Test
==========================
Run: python -m pytest tests/test_phase0.py -v
Confirms the foundation is solid before any other phase begins.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_config_loads():
    """config.yaml loads without errors."""
    from src.utils.logger import load_config
    config = load_config()
    assert config is not None
    assert "project" in config
    assert config["project"]["name"] == "Explainable Fraud Detection System"


def test_all_config_sections_exist():
    """Every phase has its config section."""
    from src.utils.logger import load_config
    config = load_config()
    required = [
        "environment", "data", "preprocessing", "graph",
        "model", "stress_test", "explainability", "api",
        "dashboard", "feedback", "cicd", "deployment", "monitoring",
    ]
    for section in required:
        assert section in config, f"Missing config section: {section}"


def test_logger_creates():
    """Logger initializes and writes without crash."""
    from src.utils.logger import get_logger
    logger = get_logger("test")
    logger.info("Phase 0 verification test passed.")
    assert True


def test_directory_structure():
    """All required directories exist."""
    from src.utils.logger import PROJECT_ROOT
    dirs = [
        "data/raw", "data/processed", "data/feedback",
        "models", "logs", "tests", "graphs", "reports", "docs",
        "src/utils", "src/preprocessing", "src/features",
        "src/models", "src/explainability", "src/api",
        "src/dashboard", "src/graph_analytics", "src/stress_testing",
        ".github/workflows",
    ]
    for d in dirs:
        assert (PROJECT_ROOT / d).is_dir(), f"Missing directory: {d}"


def test_resolve_path():
    """resolve_path correctly builds absolute paths from config values."""
    from src.utils.logger import resolve_path, PROJECT_ROOT
    p = resolve_path("data/raw/creditcard.csv")
    assert p == PROJECT_ROOT / "data" / "raw" / "creditcard.csv"


def test_phase_tracking():
    """Phase start/end tracking works."""
    from src.utils.logger import log_phase_start, log_phase_end, check_phase_completed
    log_phase_start("Phase 0: Test")
    log_phase_end("Phase 0: Test", status="SUCCESS")
    assert check_phase_completed("Phase 0: Test")
