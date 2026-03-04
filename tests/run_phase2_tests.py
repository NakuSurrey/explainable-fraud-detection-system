"""
Phase 2 Verification Runner — Run: python tests/run_phase2_tests.py
Works without pytest (for offline environments).

Tests are split into two groups:
  Group A: Always runnable (script existence, config, imports)
  Group B: Require dataset download (skipped with message if CSV missing)
"""
import sys
import json
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import PROJECT_ROOT, load_config, resolve_path

PASS = 0
FAIL = 0
SKIP = 0


def run_test(name, func):
    global PASS, FAIL, SKIP
    try:
        result = func()
        if result == "SKIP":
            print(f"  SKIP  {name}")
            SKIP += 1
        else:
            print(f"  PASS  {name}")
            PASS += 1
    except Exception as e:
        print(f"  FAIL  {name}")
        print(f"        -> {e}")
        FAIL += 1


def _dataset_exists() -> bool:
    config = load_config()
    return resolve_path(config["data"]["raw_path"]).exists()


def _get_raw_dir() -> Path:
    config = load_config()
    return resolve_path(config["data"]["raw_path"]).parent


# ===========================================================================
# Group A: Always runnable
# ===========================================================================

def test_data_ingestion_script_exists():
    script = PROJECT_ROOT / "src" / "preprocessing" / "data_ingestion.py"
    assert script.exists(), f"data_ingestion.py not found at {script}"
    assert len(script.read_text(encoding="utf-8")) > 1000, "Script too short"

def test_data_ingestion_is_importable():
    from src.preprocessing.data_ingestion import (
        run_phase2, ensure_dataset, validate_dataset,
        generate_data_manifest, generate_gdpr_privacy_log, compute_file_hash,
    )
    assert callable(run_phase2)

def test_data_ingestion_has_cli_support():
    content = (PROJECT_ROOT / "src" / "preprocessing" / "data_ingestion.py").read_text(encoding="utf-8")
    assert "--skip-download" in content, "Missing --skip-download flag"
    assert "argparse" in content, "Must use argparse"

def test_config_has_data_section():
    config = load_config()
    assert "data" in config, "Missing 'data' section"
    for key in ["raw_path", "dataset_source", "dataset_name"]:
        assert key in config["data"], f"Missing config key: data.{key}"

def test_raw_directory_exists():
    raw_dir = _get_raw_dir()
    assert raw_dir.exists() and raw_dir.is_dir(), f"Raw dir missing: {raw_dir}"

def test_raw_data_is_read_only_by_convention():
    content = (PROJECT_ROOT / "src" / "preprocessing" / "data_ingestion.py").read_text(encoding="utf-8")
    assert "never" in content.lower(), "Script should document raw data is never overwritten"

def test_phase_tracking_integration():
    content = (PROJECT_ROOT / "src" / "preprocessing" / "data_ingestion.py").read_text(encoding="utf-8")
    assert "log_phase_start" in content, "Must call log_phase_start()"
    assert "log_phase_end" in content, "Must call log_phase_end()"


# ===========================================================================
# Group B: Require dataset download
# ===========================================================================

def test_raw_dataset_exists():
    if not _dataset_exists():
        return "SKIP"
    config = load_config()
    csv_path = resolve_path(config["data"]["raw_path"])
    size_mb = csv_path.stat().st_size / (1024 * 1024)
    assert size_mb > 50, f"Dataset too small ({size_mb:.1f} MB)"

def test_dataset_has_correct_schema():
    if not _dataset_exists():
        return "SKIP"
    config = load_config()
    csv_path = resolve_path(config["data"]["raw_path"])
    with open(csv_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
    columns = [c.strip('"').strip() for c in header.split(",")]
    assert "Time" in columns, "Missing 'Time'"
    assert "Amount" in columns, "Missing 'Amount'"
    assert "Class" in columns, "Missing 'Class'"
    pca = [c for c in columns if c.startswith("V") and c[1:].isdigit()]
    assert len(pca) == 28, f"Expected 28 PCA features, found {len(pca)}"

def test_dataset_has_expected_row_count():
    if not _dataset_exists():
        return "SKIP"
    config = load_config()
    csv_path = resolve_path(config["data"]["raw_path"])
    with open(csv_path, "r", encoding="utf-8") as f:
        f.readline()
        row_count = sum(1 for _ in f)
    assert 200000 < row_count < 400000, f"Unexpected row count: {row_count:,}"

def test_data_manifest_exists():
    manifest_path = _get_raw_dir() / "data_manifest.json"
    if not manifest_path.exists():
        return "SKIP"
    content = json.loads(manifest_path.read_text(encoding="utf-8"))
    for section in ["dataset", "schema", "class_distribution", "file_info"]:
        assert section in content, f"Manifest missing '{section}'"

def test_data_manifest_has_integrity_hash():
    manifest_path = _get_raw_dir() / "data_manifest.json"
    if not manifest_path.exists():
        return "SKIP"
    content = json.loads(manifest_path.read_text(encoding="utf-8"))
    sha = content.get("file_info", {}).get("sha256", "")
    assert len(sha) == 64, f"SHA-256 hash should be 64 chars, got {len(sha)}"

def test_data_manifest_has_downstream_contract():
    manifest_path = _get_raw_dir() / "data_manifest.json"
    if not manifest_path.exists():
        return "SKIP"
    content = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "downstream_contract" in content, "Missing downstream contract"

def test_gdpr_privacy_log_exists():
    gdpr_path = _get_raw_dir() / "gdpr_privacy_log.json"
    if not gdpr_path.exists():
        return "SKIP"
    content = json.loads(gdpr_path.read_text(encoding="utf-8"))
    for key in ["legal_basis", "data_minimization_assessment", "automated_decision_making"]:
        assert key in content, f"GDPR log missing '{key}'"

def test_gdpr_log_references_correct_articles():
    gdpr_path = _get_raw_dir() / "gdpr_privacy_log.json"
    if not gdpr_path.exists():
        return "SKIP"
    text = gdpr_path.read_text(encoding="utf-8").lower()
    for article in ["article 22", "article 5", "article 6"]:
        assert article in text, f"GDPR log must reference {article}"

def test_gdpr_log_has_dpia():
    gdpr_path = _get_raw_dir() / "gdpr_privacy_log.json"
    if not gdpr_path.exists():
        return "SKIP"
    content = json.loads(gdpr_path.read_text(encoding="utf-8"))
    assert "data_protection_impact_assessment" in content, "Missing DPIA"
    dpia = content["data_protection_impact_assessment"]
    assert "risk_level" in dpia and "justification" in dpia


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 2 VERIFICATION: Data Ingestion (The Vault)")
    print("=" * 60 + "\n")

    if not _dataset_exists():
        print("  NOTE: Dataset not downloaded yet.")
        print("        Group B tests (dataset-dependent) will be SKIPPED.")
        print("        Run Phase 2 first, then re-run these tests.\n")

    group_a = [
        ("Ingestion script exists", test_data_ingestion_script_exists),
        ("Ingestion module is importable", test_data_ingestion_is_importable),
        ("CLI --skip-download support", test_data_ingestion_has_cli_support),
        ("Config has data section", test_config_has_data_section),
        ("Raw data directory exists", test_raw_directory_exists),
        ("Raw data read-only convention", test_raw_data_is_read_only_by_convention),
        ("Phase tracking integration", test_phase_tracking_integration),
    ]

    group_b = [
        ("Raw dataset CSV exists", test_raw_dataset_exists),
        ("Dataset has correct schema", test_dataset_has_correct_schema),
        ("Dataset has expected row count", test_dataset_has_expected_row_count),
        ("Data manifest exists & valid", test_data_manifest_exists),
        ("Manifest has SHA-256 hash", test_data_manifest_has_integrity_hash),
        ("Manifest has downstream contract", test_data_manifest_has_downstream_contract),
        ("GDPR privacy log exists", test_gdpr_privacy_log_exists),
        ("GDPR log references articles", test_gdpr_log_references_correct_articles),
        ("GDPR log has DPIA", test_gdpr_log_has_dpia),
    ]

    print("--- Group A: Script & Config (always runnable) ---\n")
    for name, func in group_a:
        run_test(name, func)

    print("\n--- Group B: Dataset & Artifacts (require download) ---\n")
    for name, func in group_b:
        run_test(name, func)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {SKIP} skipped out of {PASS + FAIL + SKIP}")
    print(f"{'=' * 60}\n")

    if FAIL > 0:
        sys.exit(1)

    if SKIP > 0:
        print("Some tests were skipped because the dataset hasn't been downloaded yet.")
        print("After running Phase 2, re-run this script to verify all tests pass.")
    else:
        print("Phase 2 VERIFIED — All tests passed.")
