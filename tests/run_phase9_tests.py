"""
Phase 9 Test Runner — Standalone (no pytest needed)
====================================================
Usage:
    python tests/run_phase9_tests.py
"""

import sys
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tests.test_phase9 import run_all

success = run_all()
sys.exit(0 if success else 1)
