"""
Phase 8 Standalone Test Runner
================================
Runs all Phase 8 tests without requiring pytest.

Usage:
    python tests/run_phase8_tests.py
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_phase8 import run_all

if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
