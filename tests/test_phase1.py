"""
Phase 1 Verification — Run: python -m pytest tests/test_phase1.py -v

Tests verify that all Phase 1 documentation artifacts exist and
contain the required business-context and compliance content.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_readme_exists():
    """README.md must exist at project root."""
    from src.utils.logger import PROJECT_ROOT
    readme = PROJECT_ROOT / "README.md"
    assert readme.exists(), "README.md not found at project root"
    content = readme.read_text(encoding="utf-8")
    assert len(content) > 1000, "README.md is too short to be comprehensive"


def test_readme_starts_with_business_problem():
    """README must lead with the business problem, not the tech stack."""
    from src.utils.logger import PROJECT_ROOT
    content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    # Find first major heading after the title
    lines = content.split("\n")
    headings = [l.strip() for l in lines if l.strip().startswith("## ")]
    assert len(headings) >= 2, "README needs at least 2 section headings"
    assert "business" in headings[0].lower() or "problem" in headings[0].lower(), \
        f"First section should be Business Problem, got: {headings[0]}"


def test_readme_contains_required_sections():
    """README must contain all critical sections."""
    from src.utils.logger import PROJECT_ROOT
    content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()
    required = [
        "business problem",
        "roi",
        "tech stack",
        "architecture",
        "auprc",
        "shap",
        "lime",
        "xgboost",
        "lightgbm",
        "smote",
        "fca",
        "gdpr",
        "data drift",
        "retraining",
        "getting started",
        "repository structure",
    ]
    for term in required:
        assert term in content, f"README.md missing required content: '{term}'"


def test_readme_forbids_standard_accuracy():
    """README must explicitly state that standard accuracy is not used."""
    from src.utils.logger import PROJECT_ROOT
    content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "accuracy is" in content and ("not used" in content or "forbidden" in content or "meaningless" in content), \
        "README must explicitly state that standard accuracy is not the evaluation metric"


def test_readme_mentions_inference_latency():
    """README must document why XGBoost/LightGBM were chosen (millisecond inference)."""
    from src.utils.logger import PROJECT_ROOT
    content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8").lower()
    assert "millisecond" in content, \
        "README must mention millisecond inference latency as a model selection reason"


def test_compliance_notes_exist():
    """Compliance documentation must exist in docs/."""
    from src.utils.logger import PROJECT_ROOT
    compliance = PROJECT_ROOT / "docs" / "compliance_notes.md"
    assert compliance.exists(), "docs/compliance_notes.md not found"
    content = compliance.read_text(encoding="utf-8")
    assert len(content) > 500, "compliance_notes.md is too short"


def test_compliance_covers_fca():
    """Compliance notes must cover FCA Consumer Duty."""
    from src.utils.logger import PROJECT_ROOT
    content = (PROJECT_ROOT / "docs" / "compliance_notes.md").read_text(encoding="utf-8").lower()
    required = ["consumer duty", "fca", "bias", "discrimination", "shap"]
    for term in required:
        assert term in content, f"compliance_notes.md missing FCA content: '{term}'"


def test_compliance_covers_gdpr():
    """Compliance notes must cover GDPR alignment."""
    from src.utils.logger import PROJECT_ROOT
    content = (PROJECT_ROOT / "docs" / "compliance_notes.md").read_text(encoding="utf-8").lower()
    required = ["gdpr", "article 22", "data minimization", "personal data"]
    for term in required:
        assert term in content, f"compliance_notes.md missing GDPR content: '{term}'"


def test_compliance_covers_audit_trail():
    """Compliance notes must describe the audit trail."""
    from src.utils.logger import PROJECT_ROOT
    content = (PROJECT_ROOT / "docs" / "compliance_notes.md").read_text(encoding="utf-8").lower()
    assert "audit trail" in content or "audit" in content, \
        "compliance_notes.md must describe the audit trail"


def test_architecture_diagram_exists():
    """Architecture diagram must exist in docs/."""
    from src.utils.logger import PROJECT_ROOT
    # Check for any architecture diagram file
    docs = PROJECT_ROOT / "docs"
    arch_files = list(docs.glob("architecture_diagram*"))
    assert len(arch_files) > 0, "No architecture diagram found in docs/"


def test_phase1_artifacts_complete():
    """All Phase 1 artifacts must be present."""
    from src.utils.logger import PROJECT_ROOT
    required_files = [
        "README.md",
        "docs/compliance_notes.md",
    ]
    for f in required_files:
        path = PROJECT_ROOT / f
        assert path.exists(), f"Phase 1 artifact missing: {f}"
