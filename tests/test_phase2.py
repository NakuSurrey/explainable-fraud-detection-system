"""
Phase 2 Verification — Run: python -m pytest tests/test_phase2.py -v

Tests verify that Phase 2 data ingestion artifacts exist and contain
the correct structure, validation data, and GDPR compliance simulation.

NOTE: These tests assume Phase 2 has been run successfully (i.e., the
dataset has been downloaded and validated). If the dataset is not present,
download-dependent tests will be skipped with a clear message.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helper: check if dataset exists (some tests require it)
# ---------------------------------------------------------------------------
def _dataset_exists() -> bool:
    from src.utils.logger import resolve_path, load_config
    config = load_config()
    return resolve_path(config["data"]["raw_path"]).exists()


def _get_raw_dir() -> Path:
    from src.utils.logger import resolve_path, load_config
    config = load_config()
    return resolve_path(config["data"]["raw_path"]).parent


# ===========================================================================
# Test Group 1: Script & Module Existence
# ===========================================================================

def test_data_ingestion_script_exists():
    """The data ingestion script must exist at the expected location."""
    from src.utils.logger import PROJECT_ROOT
    script = PROJECT_ROOT / "src" / "preprocessing" / "data_ingestion.py"
    assert script.exists(), f"data_ingestion.py not found at {script}"
    content = script.read_text(encoding="utf-8")
    assert len(content) > 1000, "data_ingestion.py is too short to be functional"


def test_data_ingestion_is_importable():
    """The data ingestion module must be importable without errors."""
    from src.preprocessing.data_ingestion import (
        run_phase2,
        ensure_dataset,
        validate_dataset,
        generate_data_manifest,
        generate_gdpr_privacy_log,
        compute_file_hash,
    )
    assert callable(run_phase2)
    assert callable(ensure_dataset)
    assert callable(validate_dataset)
    assert callable(generate_data_manifest)
    assert callable(generate_gdpr_privacy_log)
    assert callable(compute_file_hash)


def test_data_ingestion_has_cli_support():
    """Script must support --skip-download flag for manual workflow."""
    from src.utils.logger import PROJECT_ROOT
    content = (PROJECT_ROOT / "src" / "preprocessing" / "data_ingestion.py").read_text(encoding="utf-8")
    assert "--skip-download" in content, "Script must support --skip-download CLI flag"
    assert "argparse" in content, "Script must use argparse for CLI arguments"


# ===========================================================================
# Test Group 2: Config Integration
# ===========================================================================

def test_config_has_data_section():
    """config.yaml must have the data section with required keys."""
    from src.utils.logger import load_config
    config = load_config()
    assert "data" in config, "config.yaml missing 'data' section"
    data = config["data"]
    assert "raw_path" in data, "config.yaml data section missing 'raw_path'"
    assert "dataset_source" in data, "config.yaml data section missing 'dataset_source'"
    assert "dataset_name" in data, "config.yaml data section missing 'dataset_name'"


def test_raw_directory_exists():
    """The raw data directory must exist (created in Phase 0)."""
    raw_dir = _get_raw_dir()
    assert raw_dir.exists(), f"Raw data directory not found: {raw_dir}"
    assert raw_dir.is_dir(), f"Raw data path is not a directory: {raw_dir}"


# ===========================================================================
# Test Group 3: Dataset Validation (requires download)
# ===========================================================================

def test_raw_dataset_exists():
    """Raw dataset CSV must exist after Phase 2 execution."""
    if not _dataset_exists():
        import pytest
        pytest.skip("Dataset not downloaded yet — run Phase 2 first")
    from src.utils.logger import resolve_path, load_config
    config = load_config()
    csv_path = resolve_path(config["data"]["raw_path"])
    assert csv_path.exists(), f"Dataset not found at {csv_path}"
    size_mb = csv_path.stat().st_size / (1024 * 1024)
    assert size_mb > 50, f"Dataset too small ({size_mb:.1f} MB), expected ~150 MB"


def test_dataset_has_correct_schema():
    """Dataset must have expected columns (Time, V1-V28, Amount, Class)."""
    if not _dataset_exists():
        import pytest
        pytest.skip("Dataset not downloaded yet — run Phase 2 first")
    from src.utils.logger import resolve_path, load_config
    config = load_config()
    csv_path = resolve_path(config["data"]["raw_path"])

    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    columns = [c.strip('"').strip() for c in header.split(",")]

    assert "Time" in columns, "Missing 'Time' column"
    assert "Amount" in columns, "Missing 'Amount' column"
    assert "Class" in columns, "Missing 'Class' column"
    pca = [c for c in columns if c.startswith("V") and c[1:].isdigit()]
    assert len(pca) == 28, f"Expected 28 PCA features (V1-V28), found {len(pca)}"
    assert len(columns) == 31, f"Expected 31 columns, found {len(columns)}"


def test_dataset_has_expected_row_count():
    """Dataset should have approximately 284,807 rows."""
    if not _dataset_exists():
        import pytest
        pytest.skip("Dataset not downloaded yet — run Phase 2 first")
    from src.utils.logger import resolve_path, load_config
    config = load_config()
    csv_path = resolve_path(config["data"]["raw_path"])

    with open(csv_path, "r", encoding="utf-8") as f:
        f.readline()  # Skip header
        row_count = sum(1 for _ in f)

    assert row_count > 200000, f"Too few rows ({row_count:,}), expected ~284,807"
    assert row_count < 400000, f"Too many rows ({row_count:,}), dataset may be wrong"


# ===========================================================================
# Test Group 4: Manifest & GDPR Artifacts (requires download + Phase 2 run)
# ===========================================================================

def test_data_manifest_exists():
    """Data manifest JSON must be generated by Phase 2."""
    manifest_path = _get_raw_dir() / "data_manifest.json"
    if not manifest_path.exists():
        import pytest
        pytest.skip("Data manifest not found — run Phase 2 first")
    content = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "dataset" in content, "Manifest missing 'dataset' section"
    assert "schema" in content, "Manifest missing 'schema' section"
    assert "class_distribution" in content, "Manifest missing 'class_distribution'"
    assert "file_info" in content, "Manifest missing 'file_info'"


def test_data_manifest_has_integrity_hash():
    """Manifest must include a SHA-256 hash for data integrity verification."""
    manifest_path = _get_raw_dir() / "data_manifest.json"
    if not manifest_path.exists():
        import pytest
        pytest.skip("Data manifest not found — run Phase 2 first")
    content = json.loads(manifest_path.read_text(encoding="utf-8"))
    sha = content.get("file_info", {}).get("sha256", "")
    assert len(sha) == 64, f"SHA-256 hash should be 64 chars, got {len(sha)}"


def test_data_manifest_has_downstream_contract():
    """Manifest must include the downstream contract for Phase 3."""
    manifest_path = _get_raw_dir() / "data_manifest.json"
    if not manifest_path.exists():
        import pytest
        pytest.skip("Data manifest not found — run Phase 2 first")
    content = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "downstream_contract" in content, "Manifest missing downstream contract"
    contract = content["downstream_contract"]
    assert "Phase 3 reads" in contract, "Contract missing Phase 3 read instructions"
    assert "Phase 3 must NOT" in contract, "Contract missing Phase 3 restrictions"


def test_gdpr_privacy_log_exists():
    """GDPR privacy audit log must be generated by Phase 2."""
    gdpr_path = _get_raw_dir() / "gdpr_privacy_log.json"
    if not gdpr_path.exists():
        import pytest
        pytest.skip("GDPR log not found — run Phase 2 first")
    content = json.loads(gdpr_path.read_text(encoding="utf-8"))
    assert "legal_basis" in content, "GDPR log missing legal basis"
    assert "data_minimization_assessment" in content, "GDPR log missing data minimization"
    assert "automated_decision_making" in content, "GDPR log missing Article 22"


def test_gdpr_log_references_correct_articles():
    """GDPR log must reference specific GDPR articles."""
    gdpr_path = _get_raw_dir() / "gdpr_privacy_log.json"
    if not gdpr_path.exists():
        import pytest
        pytest.skip("GDPR log not found — run Phase 2 first")
    content_str = gdpr_path.read_text(encoding="utf-8").lower()
    assert "article 22" in content_str, "GDPR log must reference Article 22 (automated decisions)"
    assert "article 5" in content_str, "GDPR log must reference Article 5 (data principles)"
    assert "article 6" in content_str, "GDPR log must reference Article 6 (lawful processing)"


def test_gdpr_log_has_dpia():
    """GDPR log must include a Data Protection Impact Assessment."""
    gdpr_path = _get_raw_dir() / "gdpr_privacy_log.json"
    if not gdpr_path.exists():
        import pytest
        pytest.skip("GDPR log not found — run Phase 2 first")
    content = json.loads(gdpr_path.read_text(encoding="utf-8"))
    assert "data_protection_impact_assessment" in content, "GDPR log missing DPIA"
    dpia = content["data_protection_impact_assessment"]
    assert "risk_level" in dpia, "DPIA missing risk level"
    assert "justification" in dpia, "DPIA missing justification"


# ===========================================================================
# Test Group 5: Phase Independence Verification
# ===========================================================================

def test_raw_data_is_read_only_by_convention():
    """Verify the ingestion script documents that raw data must not be modified."""
    from src.utils.logger import PROJECT_ROOT
    content = (PROJECT_ROOT / "src" / "preprocessing" / "data_ingestion.py").read_text(encoding="utf-8")
    assert "NEVER" in content.upper() or "never" in content.lower(), \
        "Script should document that raw data is never overwritten"


def test_phase_tracking_integration():
    """Ingestion script must use the centralized phase tracker."""
    from src.utils.logger import PROJECT_ROOT
    content = (PROJECT_ROOT / "src" / "preprocessing" / "data_ingestion.py").read_text(encoding="utf-8")
    assert "log_phase_start" in content, "Script must call log_phase_start()"
    assert "log_phase_end" in content, "Script must call log_phase_end()"
    assert "PHASE_NAME" in content or "Phase 2" in content, "Script must identify itself as Phase 2"
