"""
Phase 4 Tests: Network Graph Generation
=========================================
Group A (8 tests): Always runnable -- script existence, config, imports
Group B (8 tests): Require Phase 4 execution -- graph artifacts
Total: 16 tests
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import PROJECT_ROOT, load_config, resolve_path


# ===========================================================================
# Helpers
# ===========================================================================

def _graph_exists() -> bool:
    config = load_config()
    return resolve_path(config["graph"]["graph_path"]).exists()


def _get_graph_dir() -> Path:
    config = load_config()
    return resolve_path(config["graph"]["output_dir"])


# ===========================================================================
# GROUP A: Script & Config (always runnable) -- 8 tests
# ===========================================================================

class TestGroupA_ScriptAndConfig:
    """These tests run without Phase 4 execution."""

    def test_graph_builder_script_exists(self):
        """Graph builder script must exist and be substantial."""
        script = PROJECT_ROOT / "src" / "graph_analytics" / "graph_builder.py"
        assert script.exists(), f"Script not found: {script}"
        content = script.read_text(encoding="utf-8")
        assert len(content) > 1000, "Script is too small -- likely incomplete"

    def test_graph_builder_is_importable(self):
        """All public functions must be importable."""
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
        assert callable(compute_similarity_matrix)
        assert callable(build_fraud_graph)
        assert callable(detect_fraud_rings)
        assert callable(save_graph)
        assert callable(save_edge_list)
        assert callable(save_rings)
        assert callable(save_graph_summary)

    def test_graph_builder_has_cli_support(self):
        """Script must support CLI execution."""
        script = PROJECT_ROOT / "src" / "graph_analytics" / "graph_builder.py"
        content = script.read_text(encoding="utf-8")
        assert "argparse" in content, "Script must use argparse for CLI"
        assert '__name__' in content and '__main__' in content, "Script must have __main__ block"

    def test_config_has_graph_section(self):
        """config.yaml must have expanded graph section."""
        config = load_config()
        assert "graph" in config, "config.yaml missing 'graph' section"
        g = config["graph"]
        required_keys = [
            "graph_path", "edge_list_path", "rings_path", "summary_path",
            "strategy", "similarity_metric", "similarity_threshold",
            "time_window_seconds", "min_ring_size", "pca_features",
        ]
        for key in required_keys:
            assert key in g, f"config.yaml graph section missing '{key}'"

    def test_config_graph_parameters_valid(self):
        """Graph parameters must have valid values."""
        config = load_config()
        g = config["graph"]
        assert 0 < g["similarity_threshold"] <= 1.0, "Threshold must be (0, 1]"
        assert g["time_window_seconds"] > 0, "Time window must be positive"
        assert g["min_ring_size"] >= 2, "Min ring size must be at least 2"
        assert len(g["pca_features"]) == 28, "Should have V1-V28 (28 PCA features)"
        assert g["strategy"] == "similarity_temporal", "Strategy must be similarity_temporal"

    def test_config_graph_strategy_is_combined(self):
        """Strategy must be the combined similarity + temporal approach."""
        config = load_config()
        assert config["graph"]["similarity_metric"] in ("cosine", "euclidean"), \
            "Similarity metric must be cosine or euclidean"

    def test_graphs_directory_exists(self):
        """The graphs/ output directory must exist."""
        graphs_dir = _get_graph_dir()
        assert graphs_dir.exists(), f"Graphs directory not found: {graphs_dir}"
        assert graphs_dir.is_dir(), f"Graphs path is not a directory: {graphs_dir}"

    def test_phase_tracking_integration(self):
        """Script must use centralized phase tracking."""
        script = PROJECT_ROOT / "src" / "graph_analytics" / "graph_builder.py"
        content = script.read_text(encoding="utf-8")
        assert "log_phase_start" in content, "Must use log_phase_start()"
        assert "log_phase_end" in content, "Must use log_phase_end()"


# ===========================================================================
# GROUP B: Graph Artifacts (require Phase 4 execution) -- 8 tests
# ===========================================================================

class TestGroupB_GraphArtifacts:
    """These tests require Phase 4 to have been executed."""

    def test_graph_file_exists(self):
        """Graph pickle file must exist after Phase 4."""
        if not _graph_exists():
            pytest.skip("Phase 4 not yet executed -- run Phase 4 first")
        config = load_config()
        path = resolve_path(config["graph"]["graph_path"])
        assert path.exists(), f"Graph file not found: {path}"
        assert path.stat().st_size > 0, "Graph file is empty"

    def test_graph_is_valid_networkx(self):
        """Saved graph must be a valid NetworkX Graph object."""
        if not _graph_exists():
            pytest.skip("Phase 4 not yet executed")
        import pickle
        import networkx as nx
        config = load_config()
        path = resolve_path(config["graph"]["graph_path"])
        with open(path, "rb") as f:
            G = pickle.load(f)
        assert isinstance(G, nx.Graph), "Saved object is not a NetworkX Graph"
        assert G.number_of_nodes() > 0, "Graph has no nodes"

    def test_graph_nodes_have_metadata(self):
        """Every node must have time and amount metadata."""
        if not _graph_exists():
            pytest.skip("Phase 4 not yet executed")
        import pickle
        config = load_config()
        path = resolve_path(config["graph"]["graph_path"])
        with open(path, "rb") as f:
            G = pickle.load(f)
        for node in list(G.nodes())[:10]:  # Check first 10
            attrs = G.nodes[node]
            assert "time" in attrs, f"Node {node} missing 'time' attribute"
            assert "amount" in attrs, f"Node {node} missing 'amount' attribute"
            assert "original_index" in attrs, f"Node {node} missing 'original_index'"

    def test_edge_list_exists(self):
        """Edge list CSV must exist."""
        if not _graph_exists():
            pytest.skip("Phase 4 not yet executed")
        config = load_config()
        path = resolve_path(config["graph"]["edge_list_path"])
        assert path.exists(), f"Edge list not found: {path}"
        import pandas as pd
        df = pd.read_csv(path)
        expected_cols = {"source", "target", "similarity", "time_diff_seconds"}
        assert expected_cols.issubset(set(df.columns)), \
            f"Edge list missing columns. Found: {list(df.columns)}"

    def test_edges_meet_thresholds(self):
        """All edges must satisfy both similarity and temporal thresholds."""
        if not _graph_exists():
            pytest.skip("Phase 4 not yet executed")
        config = load_config()
        import pandas as pd
        path = resolve_path(config["graph"]["edge_list_path"])
        df = pd.read_csv(path)
        if len(df) == 0:
            pytest.skip("No edges in graph -- thresholds may be too strict")
        sim_thresh = config["graph"]["similarity_threshold"]
        time_window = config["graph"]["time_window_seconds"]
        assert (df["similarity"] >= sim_thresh - 0.001).all(), \
            "Some edges have similarity below threshold"
        assert (df["time_diff_seconds"] <= time_window + 0.01).all(), \
            "Some edges exceed time window"

    def test_fraud_rings_file_exists(self):
        """Fraud rings JSON must exist with correct structure."""
        if not _graph_exists():
            pytest.skip("Phase 4 not yet executed")
        config = load_config()
        path = resolve_path(config["graph"]["rings_path"])
        assert path.exists(), f"Rings file not found: {path}"
        with open(path, "r") as f:
            data = json.load(f)
        assert "total_rings" in data, "Rings file missing 'total_rings'"
        assert "rings" in data, "Rings file missing 'rings' list"
        assert "generated_by" in data, "Rings file missing 'generated_by'"

    def test_rings_meet_min_size(self):
        """All detected rings must have >= min_ring_size nodes."""
        if not _graph_exists():
            pytest.skip("Phase 4 not yet executed")
        config = load_config()
        path = resolve_path(config["graph"]["rings_path"])
        with open(path, "r") as f:
            data = json.load(f)
        min_size = config["graph"]["min_ring_size"]
        for ring in data["rings"]:
            assert ring["size"] >= min_size, \
                f"Ring {ring['ring_id']} has size {ring['size']} < min_ring_size {min_size}"

    def test_graph_summary_exists(self):
        """Graph summary JSON must exist with required sections."""
        if not _graph_exists():
            pytest.skip("Phase 4 not yet executed")
        config = load_config()
        path = resolve_path(config["graph"]["summary_path"])
        assert path.exists(), f"Summary not found: {path}"
        with open(path, "r") as f:
            data = json.load(f)
        assert "strategy" in data, "Summary missing 'strategy'"
        assert "parameters" in data, "Summary missing 'parameters'"
        assert "graph_stats" in data, "Summary missing 'graph_stats'"
        assert "ring_stats" in data, "Summary missing 'ring_stats'"


# Import json at module level for Group B tests
import json
