"""
Phase 4 Verification Runner -- Run: python tests/run_phase4_tests.py
Works without pytest (for offline environments).

Tests are split into two groups:
  Group A: Always runnable (script existence, config, imports)
  Group B: Require Phase 4 execution (graph artifacts)
"""
import sys
import json
import pickle
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


def _graph_exists() -> bool:
    config = load_config()
    return resolve_path(config["graph"]["graph_path"]).exists()


# ===========================================================================
# GROUP A: Script & Config (always runnable) -- 8 tests
# ===========================================================================

def test_graph_builder_script_exists():
    script = PROJECT_ROOT / "src" / "graph_analytics" / "graph_builder.py"
    assert script.exists(), f"Script not found: {script}"
    content = script.read_text(encoding="utf-8")
    assert len(content) > 1000, "Script is too small"


def test_graph_builder_is_importable():
    from src.graph_analytics.graph_builder import (
        run_phase4,
        load_fraud_transactions,
        compute_similarity_matrix,
        build_fraud_graph,
        detect_fraud_rings,
        save_graph,
        save_edge_list,
        save_rings,
        save_graph_summary,
    )
    assert callable(run_phase4)
    assert callable(load_fraud_transactions)


def test_graph_builder_has_cli_support():
    script = PROJECT_ROOT / "src" / "graph_analytics" / "graph_builder.py"
    content = script.read_text(encoding="utf-8")
    assert "argparse" in content
    assert "__main__" in content


def test_config_has_graph_section():
    config = load_config()
    assert "graph" in config
    g = config["graph"]
    for key in ["graph_path", "edge_list_path", "rings_path", "summary_path",
                "strategy", "similarity_metric", "similarity_threshold",
                "time_window_seconds", "min_ring_size", "pca_features"]:
        assert key in g, f"Missing '{key}'"


def test_config_graph_parameters_valid():
    config = load_config()
    g = config["graph"]
    assert 0 < g["similarity_threshold"] <= 1.0
    assert g["time_window_seconds"] > 0
    assert g["min_ring_size"] >= 2
    assert len(g["pca_features"]) == 28
    assert g["strategy"] == "similarity_temporal"


def test_config_graph_strategy_is_combined():
    config = load_config()
    assert config["graph"]["similarity_metric"] in ("cosine", "euclidean")


def test_graphs_directory_exists():
    config = load_config()
    graphs_dir = resolve_path(config["graph"]["output_dir"])
    assert graphs_dir.exists(), f"Graphs directory not found: {graphs_dir}"
    assert graphs_dir.is_dir()


def test_phase_tracking_integration():
    script = PROJECT_ROOT / "src" / "graph_analytics" / "graph_builder.py"
    content = script.read_text(encoding="utf-8")
    assert "log_phase_start" in content
    assert "log_phase_end" in content


# ===========================================================================
# GROUP B: Graph Artifacts (require Phase 4 execution) -- 8 tests
# ===========================================================================

def test_graph_file_exists():
    if not _graph_exists():
        return "SKIP"
    config = load_config()
    path = resolve_path(config["graph"]["graph_path"])
    assert path.exists()
    assert path.stat().st_size > 0


def test_graph_is_valid_networkx():
    if not _graph_exists():
        return "SKIP"
    import networkx as nx
    config = load_config()
    path = resolve_path(config["graph"]["graph_path"])
    with open(path, "rb") as f:
        G = pickle.load(f)
    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() > 0


def test_graph_nodes_have_metadata():
    if not _graph_exists():
        return "SKIP"
    config = load_config()
    path = resolve_path(config["graph"]["graph_path"])
    with open(path, "rb") as f:
        G = pickle.load(f)
    for node in list(G.nodes())[:10]:
        attrs = G.nodes[node]
        assert "time" in attrs, f"Node {node} missing 'time'"
        assert "amount" in attrs, f"Node {node} missing 'amount'"
        assert "original_index" in attrs, f"Node {node} missing 'original_index'"


def test_edge_list_exists():
    if not _graph_exists():
        return "SKIP"
    import pandas as pd
    config = load_config()
    path = resolve_path(config["graph"]["edge_list_path"])
    assert path.exists()
    df = pd.read_csv(path)
    expected = {"source", "target", "similarity", "time_diff_seconds"}
    assert expected.issubset(set(df.columns))


def test_edges_meet_thresholds():
    if not _graph_exists():
        return "SKIP"
    import pandas as pd
    config = load_config()
    path = resolve_path(config["graph"]["edge_list_path"])
    df = pd.read_csv(path)
    if len(df) == 0:
        return "SKIP"
    sim_thresh = config["graph"]["similarity_threshold"]
    time_window = config["graph"]["time_window_seconds"]
    assert (df["similarity"] >= sim_thresh - 0.001).all(), "Edges below similarity threshold"
    assert (df["time_diff_seconds"] <= time_window + 0.01).all(), "Edges exceed time window"


def test_fraud_rings_file_exists():
    if not _graph_exists():
        return "SKIP"
    config = load_config()
    path = resolve_path(config["graph"]["rings_path"])
    assert path.exists()
    with open(path, "r") as f:
        data = json.load(f)
    assert "total_rings" in data
    assert "rings" in data
    assert "generated_by" in data


def test_rings_meet_min_size():
    if not _graph_exists():
        return "SKIP"
    config = load_config()
    path = resolve_path(config["graph"]["rings_path"])
    with open(path, "r") as f:
        data = json.load(f)
    min_size = config["graph"]["min_ring_size"]
    for ring in data["rings"]:
        assert ring["size"] >= min_size, \
            f"Ring {ring['ring_id']} size {ring['size']} < {min_size}"


def test_graph_summary_exists():
    if not _graph_exists():
        return "SKIP"
    config = load_config()
    path = resolve_path(config["graph"]["summary_path"])
    assert path.exists()
    with open(path, "r") as f:
        data = json.load(f)
    assert "strategy" in data
    assert "parameters" in data
    assert "graph_stats" in data
    assert "ring_stats" in data


# ===========================================================================
# RUNNER
# ===========================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PHASE 4 VERIFICATION: Network Graph Generation")
    print("=" * 60 + "\n")

    print("GROUP A: Script & Config (always runnable)")
    print("-" * 45)
    run_test("test_graph_builder_script_exists", test_graph_builder_script_exists)
    run_test("test_graph_builder_is_importable", test_graph_builder_is_importable)
    run_test("test_graph_builder_has_cli_support", test_graph_builder_has_cli_support)
    run_test("test_config_has_graph_section", test_config_has_graph_section)
    run_test("test_config_graph_parameters_valid", test_config_graph_parameters_valid)
    run_test("test_config_graph_strategy_is_combined", test_config_graph_strategy_is_combined)
    run_test("test_graphs_directory_exists", test_graphs_directory_exists)
    run_test("test_phase_tracking_integration", test_phase_tracking_integration)

    print()
    print("GROUP B: Graph Artifacts (require Phase 4 execution)")
    print("-" * 45)
    run_test("test_graph_file_exists", test_graph_file_exists)
    run_test("test_graph_is_valid_networkx", test_graph_is_valid_networkx)
    run_test("test_graph_nodes_have_metadata", test_graph_nodes_have_metadata)
    run_test("test_edge_list_exists", test_edge_list_exists)
    run_test("test_edges_meet_thresholds", test_edges_meet_thresholds)
    run_test("test_fraud_rings_file_exists", test_fraud_rings_file_exists)
    run_test("test_rings_meet_min_size", test_rings_meet_min_size)
    run_test("test_graph_summary_exists", test_graph_summary_exists)

    print("\n" + "=" * 60)
    print(f"RESULTS: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print("=" * 60)

    if FAIL > 0:
        print("\nSome tests FAILED. Check the errors above.")
        if SKIP > 0:
            print("After running Phase 4, re-run this script to verify all tests pass.")
    else:
        if SKIP > 0:
            print(f"\n{SKIP} tests were skipped (Phase 4 not yet executed).")
            print("Run Phase 4 first, then re-run this script.")
        else:
            print("\nPhase 4 VERIFIED -- All tests passed.")
