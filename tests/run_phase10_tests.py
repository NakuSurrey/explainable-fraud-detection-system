"""
Phase 10 Standalone Test Runner
================================
Run: python tests/run_phase10_tests.py

Calls the run_all() function from test_phase10.py.
"""

import sys
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tests.test_phase10 import run_all

success = run_all()
sys.exit(0 if success else 1)
