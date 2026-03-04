"""
Phase 1 Verification Runner — Run: python tests/run_phase1_tests.py
Works without pytest (for offline environments).
"""
import sys, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import PROJECT_ROOT

PASS = 0
FAIL = 0

def run_test(name, func):
    global PASS, FAIL
    try:
        func()
        print(f"  PASS  {name}")
        PASS += 1
    except Exception as e:
        print(f"  FAIL  {name}")
        print(f"        -> {e}")
        FAIL += 1

def test_readme_exists():
    readme = PROJECT_ROOT / "README.md"
    assert readme.exists(), "README.md not found"
    assert len(readme.read_text(encoding="utf-8")) > 1000, "README.md too short"

def test_readme_starts_with_business_problem():
    content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    lines = content.split("\n")
    headings = [l.strip() for l in lines if l.strip().startswith("## ")]
    assert len(headings) >= 2, "Need at least 2 section headings"
    assert "business" in headings[0].lower() or "problem" in headings[0].lower(), \
        f"First section should be Business Problem, got: {headings[0]}"

def test_readme_contains_required_sections():
    content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()
    for term in ["business problem", "roi", "tech stack", "architecture", "auprc",
                  "shap", "lime", "xgboost", "lightgbm", "smote", "fca", "gdpr",
                  "data drift", "retraining", "getting started", "repository structure"]:
        assert term in content, f"Missing: '{term}'"

def test_readme_forbids_accuracy():
    content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "accuracy" in content and ("not used" in content or "forbidden" in content or "meaningless" in content)

def test_readme_mentions_latency():
    content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "millisecond" in content, "Must mention millisecond inference latency"

def test_compliance_notes_exist():
    f = PROJECT_ROOT / "docs" / "compliance_notes.md"
    assert f.exists(), "docs/compliance_notes.md not found"
    assert len(f.read_text(encoding="utf-8")) > 500

def test_compliance_covers_fca():
    content = (PROJECT_ROOT / "docs" / "compliance_notes.md").read_text(encoding="utf-8").lower()
    for term in ["consumer duty", "fca", "bias", "discrimination", "shap"]:
        assert term in content, f"Missing FCA content: '{term}'"

def test_compliance_covers_gdpr():
    content = (PROJECT_ROOT / "docs" / "compliance_notes.md").read_text(encoding="utf-8").lower()
    for term in ["gdpr", "article 22", "data minimization", "personal data"]:
        assert term in content, f"Missing GDPR content: '{term}'"

def test_compliance_covers_audit():
    content = (PROJECT_ROOT / "docs" / "compliance_notes.md").read_text(encoding="utf-8").lower()
    assert "audit" in content, "Must describe audit trail"

def test_architecture_diagram_exists():
    docs = PROJECT_ROOT / "docs"
    arch_files = list(docs.glob("architecture_diagram*"))
    assert len(arch_files) > 0, "No architecture diagram found in docs/"

def test_all_phase1_artifacts():
    for f in ["README.md", "docs/compliance_notes.md"]:
        assert (PROJECT_ROOT / f).exists(), f"Missing: {f}"

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 1 VERIFICATION: Business Logic & Compliance")
    print("=" * 60 + "\n")

    tests = [
        ("README exists and is comprehensive", test_readme_exists),
        ("README starts with Business Problem", test_readme_starts_with_business_problem),
        ("README contains all required sections", test_readme_contains_required_sections),
        ("README forbids standard accuracy", test_readme_forbids_accuracy),
        ("README mentions millisecond latency", test_readme_mentions_latency),
        ("Compliance notes exist", test_compliance_notes_exist),
        ("Compliance covers FCA Consumer Duty", test_compliance_covers_fca),
        ("Compliance covers GDPR", test_compliance_covers_gdpr),
        ("Compliance covers audit trail", test_compliance_covers_audit),
        ("Architecture diagram exists", test_architecture_diagram_exists),
        ("All Phase 1 artifacts present", test_all_phase1_artifacts),
    ]

    for name, func in tests:
        run_test(name, func)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    print(f"{'=' * 60}\n")

    if FAIL > 0:
        sys.exit(1)
    print("Phase 1 VERIFIED — All tests passed.")
