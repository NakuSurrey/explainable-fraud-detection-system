"""
Phase 4: Network Graph Generation -- THE FRAUD RING MAPPER
============================================================
Builds a NetworkX graph of fraud transaction connections using a combined
similarity + temporal proximity strategy.

Strategy:
  Two fraudulent transactions are connected (edge) if BOTH conditions hold:
    1. Cosine similarity of their PCA features (V1-V28) >= threshold (0.85)
    2. Time difference <= time_window_seconds (7200s = 2 hours)

Connected components with >= min_ring_size nodes are labeled as fraud rings.

Reads from:  Phase 3 processed data (data/processed/X_test.csv, y_test.csv)
Saves to:    graphs/ directory (graph, edges, rings, summary)

Run:
    python -m src.graph_analytics.graph_builder
"""

import argparse
import json
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timezone
from pathlib import Path
from itertools import combinations

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.logger import (
    get_logger,
    load_config,
    resolve_path,
    log_phase_start,
    log_phase_end,
)

logger = get_logger(__name__)
PHASE_NAME = "Phase 4: Network Graph Generation"


# ===========================================================================
# 1. LOAD DATA
# ===========================================================================

def load_fraud_transactions(config):
    """Load processed data and filter to fraud-only transactions.

    Uses the TEST set so the graph represents unseen data (not SMOTE-augmented
    training data). This keeps the graph honest and avoids synthetic nodes.
    """
    logger.info("Loading test set from Phase 3 artifacts...")

    x_test_path = resolve_path(config["preprocessing"]["test_path"])
    y_test_path = resolve_path(config["preprocessing"]["y_test_path"])

    if not x_test_path.exists():
        raise FileNotFoundError(
            f"Phase 3 artifact not found: {x_test_path}. Run Phase 3 first."
        )
    if not y_test_path.exists():
        raise FileNotFoundError(
            f"Phase 3 artifact not found: {y_test_path}. Run Phase 3 first."
        )

    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()

    logger.info(f"  Test set loaded: {len(X_test):,} transactions")
    logger.info(f"  Fraud in test set: {int(y_test.sum()):,} transactions")

    # Filter to fraud only
    fraud_mask = y_test == 1
    X_fraud = X_test[fraud_mask].reset_index(drop=True)

    # Keep original test-set indices as transaction IDs
    fraud_indices = y_test[fraud_mask].index.tolist()

    logger.info(f"  Fraud transactions for graph: {len(X_fraud):,}")

    return X_fraud, fraud_indices


# ===========================================================================
# 2. COMPUTE SIMILARITY MATRIX
# ===========================================================================

def compute_similarity_matrix(X_fraud, config):
    """Compute pairwise cosine similarity on PCA features (V1-V28).

    Returns an (N x N) similarity matrix where N = number of fraud transactions.
    """
    graph_config = config["graph"]
    pca_cols = graph_config["pca_features"]

    # Filter to only PCA columns that exist in the data
    available_pca = [c for c in pca_cols if c in X_fraud.columns]
    if len(available_pca) == 0:
        raise ValueError("No PCA features found in data. Check config graph.pca_features.")

    logger.info(f"  Computing cosine similarity on {len(available_pca)} PCA features...")

    pca_data = X_fraud[available_pca].values

    # Normalize rows to unit vectors for cosine similarity
    norms = np.linalg.norm(pca_data, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    pca_normalized = pca_data / norms

    # Cosine similarity = dot product of normalized vectors
    similarity_matrix = np.dot(pca_normalized, pca_normalized.T)

    # Clip to [-1, 1] to handle floating point errors
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

    logger.info(f"  Similarity matrix shape: {similarity_matrix.shape}")

    return similarity_matrix


# ===========================================================================
# 3. BUILD GRAPH
# ===========================================================================

def build_fraud_graph(X_fraud, fraud_indices, similarity_matrix, config):
    """Build a NetworkX graph connecting fraud transactions.

    Edges are drawn when BOTH conditions are met:
      1. Cosine similarity >= similarity_threshold
      2. Time difference <= time_window_seconds
    """
    graph_config = config["graph"]
    sim_threshold = graph_config["similarity_threshold"]
    time_window = graph_config["time_window_seconds"]
    time_col = graph_config["time_column"]
    amount_col = graph_config["amount_column"]

    n_fraud = len(X_fraud)
    logger.info(f"  Building graph with {n_fraud} fraud nodes...")
    logger.info(f"  Similarity threshold: {sim_threshold}")
    logger.info(f"  Time window: {time_window}s ({time_window / 3600:.1f} hours)")

    G = nx.Graph()

    # Add nodes with metadata
    for i in range(n_fraud):
        node_id = f"TX_{fraud_indices[i]}"
        attrs = {
            "original_index": fraud_indices[i],
            "time": float(X_fraud.iloc[i].get(time_col, 0)) if time_col in X_fraud.columns else 0.0,
            "amount": float(X_fraud.iloc[i].get(amount_col, 0)) if amount_col in X_fraud.columns else 0.0,
        }
        G.add_node(node_id, **attrs)

    # Extract time values for temporal check
    if time_col in X_fraud.columns:
        times = X_fraud[time_col].values
    else:
        logger.warning(f"  Time column '{time_col}' not found. Using index as proxy.")
        times = np.arange(n_fraud, dtype=float)

    # Build edges using both similarity AND temporal criteria
    edges_added = 0
    pairs_checked = 0
    total_pairs = n_fraud * (n_fraud - 1) // 2

    logger.info(f"  Checking {total_pairs:,} pairs...")

    for i in range(n_fraud):
        for j in range(i + 1, n_fraud):
            pairs_checked += 1

            # Check similarity
            sim = similarity_matrix[i, j]
            if sim < sim_threshold:
                continue

            # Check temporal proximity
            time_diff = abs(float(times[i]) - float(times[j]))
            if time_diff > time_window:
                continue

            # Both conditions met -- add edge
            node_i = f"TX_{fraud_indices[i]}"
            node_j = f"TX_{fraud_indices[j]}"
            G.add_edge(
                node_i,
                node_j,
                similarity=round(float(sim), 4),
                time_diff_seconds=round(time_diff, 2),
            )
            edges_added += 1

    logger.info(f"  Pairs checked: {pairs_checked:,}")
    logger.info(f"  Edges added: {edges_added:,}")
    logger.info(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


# ===========================================================================
# 4. DETECT FRAUD RINGS
# ===========================================================================

def detect_fraud_rings(G, config):
    """Identify connected components that qualify as fraud rings.

    A fraud ring is a connected component with >= min_ring_size nodes.
    """
    graph_config = config["graph"]
    min_size = graph_config["min_ring_size"]

    components = list(nx.connected_components(G))
    logger.info(f"  Total connected components: {len(components)}")

    rings = []
    isolated_count = 0
    small_cluster_count = 0

    for idx, component in enumerate(components):
        size = len(component)
        if size < min_size:
            if size == 1:
                isolated_count += 1
            else:
                small_cluster_count += 1
            continue

        # Extract ring details
        subgraph = G.subgraph(component)
        node_list = sorted(list(component))

        # Compute ring statistics
        amounts = [G.nodes[n].get("amount", 0) for n in component]
        times = [G.nodes[n].get("time", 0) for n in component]
        edge_sims = [d["similarity"] for _, _, d in subgraph.edges(data=True)]

        ring_info = {
            "ring_id": f"RING_{len(rings) + 1:03d}",
            "size": size,
            "nodes": node_list,
            "num_edges": subgraph.number_of_edges(),
            "density": round(nx.density(subgraph), 4),
            "avg_amount": round(float(np.mean(amounts)), 2) if amounts else 0,
            "total_amount": round(float(np.sum(amounts)), 2) if amounts else 0,
            "min_amount": round(float(np.min(amounts)), 2) if amounts else 0,
            "max_amount": round(float(np.max(amounts)), 2) if amounts else 0,
            "time_span_seconds": round(float(max(times) - min(times)), 2) if times else 0,
            "avg_similarity": round(float(np.mean(edge_sims)), 4) if edge_sims else 0,
            "min_similarity": round(float(np.min(edge_sims)), 4) if edge_sims else 0,
        }
        rings.append(ring_info)

    # Sort rings by size (largest first)
    rings.sort(key=lambda r: r["size"], reverse=True)

    logger.info(f"  Fraud rings detected (size >= {min_size}): {len(rings)}")
    logger.info(f"  Isolated fraud nodes (no connections): {isolated_count}")
    logger.info(f"  Small clusters (size < {min_size}): {small_cluster_count}")

    for ring in rings[:5]:
        logger.info(f"    {ring['ring_id']}: {ring['size']} nodes, "
                     f"{ring['num_edges']} edges, "
                     f"avg amount ${ring['avg_amount']:.2f}")

    return rings, isolated_count, small_cluster_count


# ===========================================================================
# 5. SAVE ARTIFACTS
# ===========================================================================

def save_graph(G, config):
    """Save the NetworkX graph object to disk."""
    graph_path = resolve_path(config["graph"]["graph_path"])
    graph_path.parent.mkdir(parents=True, exist_ok=True)

    with open(graph_path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = graph_path.stat().st_size / 1024
    logger.info(f"  Graph saved: {graph_path} ({size_kb:.1f} KB)")
    return str(graph_path)


def save_edge_list(G, config):
    """Save the edge list as a CSV for portability."""
    edge_path = resolve_path(config["graph"]["edge_list_path"])
    edge_path.parent.mkdir(parents=True, exist_ok=True)

    edges = []
    for u, v, data in G.edges(data=True):
        edges.append({
            "source": u,
            "target": v,
            "similarity": data.get("similarity", 0),
            "time_diff_seconds": data.get("time_diff_seconds", 0),
        })

    if edges:
        df_edges = pd.DataFrame(edges)
        df_edges.to_csv(edge_path, index=False)
    else:
        # Save empty CSV with headers
        pd.DataFrame(columns=["source", "target", "similarity", "time_diff_seconds"]).to_csv(
            edge_path, index=False
        )

    logger.info(f"  Edge list saved: {edge_path} ({len(edges)} edges)")
    return str(edge_path)


def save_rings(rings, config):
    """Save detected fraud rings as JSON."""
    rings_path = resolve_path(config["graph"]["rings_path"])
    rings_path.parent.mkdir(parents=True, exist_ok=True)

    rings_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": PHASE_NAME,
        "total_rings": len(rings),
        "rings": rings,
    }

    with open(rings_path, "w", encoding="utf-8") as f:
        json.dump(rings_data, f, indent=2, default=str)

    logger.info(f"  Fraud rings saved: {rings_path} ({len(rings)} rings)")
    return str(rings_path)


def save_graph_summary(G, rings, isolated_count, small_cluster_count, config):
    """Save overall graph statistics as JSON."""
    summary_path = resolve_path(config["graph"]["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute graph-level metrics
    total_ring_nodes = sum(r["size"] for r in rings)
    largest_ring = rings[0]["size"] if rings else 0

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": PHASE_NAME,
        "strategy": config["graph"]["strategy"],
        "parameters": {
            "similarity_metric": config["graph"]["similarity_metric"],
            "similarity_threshold": config["graph"]["similarity_threshold"],
            "time_window_seconds": config["graph"]["time_window_seconds"],
            "min_ring_size": config["graph"]["min_ring_size"],
        },
        "graph_stats": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "density": round(nx.density(G), 6) if G.number_of_nodes() > 1 else 0,
            "connected_components": nx.number_connected_components(G),
        },
        "ring_stats": {
            "total_rings_detected": len(rings),
            "total_nodes_in_rings": total_ring_nodes,
            "largest_ring_size": largest_ring,
            "smallest_ring_size": rings[-1]["size"] if rings else 0,
            "avg_ring_size": round(np.mean([r["size"] for r in rings]), 2) if rings else 0,
            "isolated_nodes": isolated_count,
            "small_clusters": small_cluster_count,
        },
        "ring_summary": [
            {
                "ring_id": r["ring_id"],
                "size": r["size"],
                "edges": r["num_edges"],
                "density": r["density"],
                "total_amount": r["total_amount"],
                "time_span_seconds": r["time_span_seconds"],
            }
            for r in rings
        ],
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"  Graph summary saved: {summary_path}")
    return str(summary_path)


# ===========================================================================
# 6. MAIN EXECUTION
# ===========================================================================

def run_phase4():
    """Execute the complete Phase 4 graph generation pipeline."""
    log_phase_start(PHASE_NAME)

    try:
        config = load_config()

        logger.info("Step 1/5: Loading fraud transactions from Phase 3...")
        X_fraud, fraud_indices = load_fraud_transactions(config)

        if len(X_fraud) == 0:
            logger.warning("No fraud transactions found in test set. Saving empty graph.")
            G = nx.Graph()
            rings = []
            isolated_count = 0
            small_cluster_count = 0
        else:
            logger.info("Step 2/5: Computing pairwise similarity matrix...")
            similarity_matrix = compute_similarity_matrix(X_fraud, config)

            logger.info("Step 3/5: Building fraud graph (similarity + temporal)...")
            G = build_fraud_graph(X_fraud, fraud_indices, similarity_matrix, config)

            logger.info("Step 4/5: Detecting fraud rings...")
            rings, isolated_count, small_cluster_count = detect_fraud_rings(G, config)

        logger.info("Step 5/5: Saving artifacts...")
        graph_file = save_graph(G, config)
        edge_file = save_edge_list(G, config)
        rings_file = save_rings(rings, config)
        summary_file = save_graph_summary(G, rings, isolated_count, small_cluster_count, config)

        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("PHASE 4 SUMMARY")
        logger.info("=" * 60)
        logger.info(f"  Strategy:            {config['graph']['strategy']}")
        logger.info(f"  Fraud transactions:  {len(X_fraud):,}")
        logger.info(f"  Graph nodes:         {G.number_of_nodes():,}")
        logger.info(f"  Graph edges:         {G.number_of_edges():,}")
        logger.info(f"  Fraud rings found:   {len(rings)}")
        if rings:
            logger.info(f"  Largest ring:        {rings[0]['size']} nodes")
            total_in_rings = sum(r['size'] for r in rings)
            logger.info(f"  Nodes in rings:      {total_in_rings} / {G.number_of_nodes()}")
        logger.info(f"  Isolated nodes:      {isolated_count}")
        logger.info(f"  Artifacts saved to:  graphs/")
        logger.info("=" * 60)

        log_phase_end(PHASE_NAME, status="SUCCESS")

    except Exception as e:
        logger.error(f"Phase 4 failed: {e}")
        log_phase_end(PHASE_NAME, status="FAILED", error=str(e))
        raise


# ===========================================================================
# CLI Entry Point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4: Network Graph Generation -- Build fraud ring graph"
    )
    args = parser.parse_args()

    try:
        run_phase4()
    except Exception as e:
        print(f"\nPhase 4 FAILED: {e}")
        sys.exit(1)
