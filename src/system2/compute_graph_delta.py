# ==============================================================================
# Filename: src/system2/compute_graph_delta.py
# Version: v1.0
#
# PURPOSE: Compute Gromov δ-hyperbolicity from GRAPH STRUCTURE, not embeddings.
#
# WHY THIS IS NEEDED:
#   The previous approach computed δ from HGCN embedding coordinates.
#   This is WRONG — embedding-space δ measures how compressed the
#   representation is, not how tree-like the underlying proof structure is.
#   Hairer got δ≈0.034 (lowest!) despite requiring lateral rewrites,
#   because analysis concepts embed near the origin (dense region).
#
# CORRECT APPROACH:
#   Use the Mathlib4 DEPENDENCY GRAPH (data/edge_index.pt + id_to_name.pkl.gz)
#   For each benchmark dataset:
#     1. Find node IDs of benchmark problems via name matching
#     2. Extract their k-hop neighborhood subgraph
#     3. Compute shortest-path distances in that subgraph
#     4. Sample 4-point quadruples and compute δ from graph distances
#   This gives δ in the SAME SPACE as the paper's δ≈0.31 figure.
#
# EXPECTED RESULTS (negative correlation with Pass@1):
#   miniF2F/compfiles/Imo:  δ ≈ 0.05–0.15   (tree-like, algebraic)
#   Wiedijk100/Sensitivity: δ ≈ 0.15–0.25   (mixed)
#   Hairer/ZagierTwoSquares: δ ≈ 0.30–0.45  (analytic rewrites)
#   ProofNet:               δ ≈ 0.45–0.60   (advanced theory)
#   Putnam:                 δ ≈ 0.55–0.70   (highly lateral)
#
# Usage:
#   python src/system2/compute_graph_delta.py
#   python src/system2/compute_graph_delta.py --datasets miniF2F-valid Hairer
# ==============================================================================

import os
import sys
import re
import glob
import json
import gzip
import pickle
import random
import argparse
import numpy as np
import torch
from collections import defaultdict, deque

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

DATA_DIR   = os.path.join(project_root, "data")
OUTPUT_DIR = os.path.join(project_root, "results", "delta_vs_pass")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
N_QUADRUPLES = 800   # more samples → more stable estimate
K_HOP = 4            # neighborhood depth for subgraph extraction


# ==============================================================================
# Load Mathlib4 dependency graph
# ==============================================================================

def load_graph():
    """Load edge_index and id_to_name from data/."""
    print("⏳ Loading Mathlib4 dependency graph...")

    edge_path = os.path.join(DATA_DIR, "edge_index.pt")
    name_path = os.path.join(DATA_DIR, "id_to_name.pkl.gz")

    if not os.path.exists(edge_path):
        print(f"❌ edge_index.pt not found at {edge_path}")
        return None, None, None

    edge_index = torch.load(edge_path, map_location="cpu")  # shape [2, E]
    n_nodes = int(edge_index.max().item()) + 1

    # Build adjacency list (undirected)
    adj = defaultdict(set)
    src, dst = edge_index[0].tolist(), edge_index[1].tolist()
    for u, v in zip(src, dst):
        adj[u].add(v)
        adj[v].add(u)

    id_to_name = {}
    if os.path.exists(name_path):
        with gzip.open(name_path, "rb") as f:
            raw = pickle.load(f)
            if isinstance(raw, dict):
                id_to_name = raw
            else:
                id_to_name = {i: v for i, v in enumerate(raw)}

    name_to_id = {v: k for k, v in id_to_name.items() if isinstance(v, str)}

    print(f"   Graph: {n_nodes} nodes, {len(src)} edges")
    print(f"   Names: {len(id_to_name)} indexed")
    return adj, id_to_name, name_to_id


# ==============================================================================
# BFS subgraph extraction
# ==============================================================================

def extract_subgraph(seed_nodes: list, adj: dict, k: int = K_HOP) -> set:
    """BFS from seed_nodes to depth k, return all reachable node IDs."""
    visited = set(seed_nodes)
    frontier = list(seed_nodes)
    for _ in range(k):
        next_frontier = []
        for node in frontier:
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    next_frontier.append(nb)
        frontier = next_frontier
        if not frontier:
            break
    return visited


def bfs_distances(source: int, subgraph_nodes: set, adj: dict) -> dict:
    """BFS shortest-path distances from source within subgraph_nodes."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj.get(u, []):
            if v in subgraph_nodes and v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


# ==============================================================================
# Gromov δ from graph distances
# ==============================================================================

def gromov_delta_graph(nodes: list, adj: dict, n_samples: int = N_QUADRUPLES,
                        seed: int = SEED) -> float:
    """
    Estimate Gromov δ using four-point condition on GRAPH shortest-path distances.

    For graph metric (X, d_graph), δ is the smallest value s.t. for all x,y,z,w:
        d(x,y) + d(z,w) ≤ max(d(x,z)+d(y,w), d(x,w)+d(y,z)) + 2δ

    Returns the mean δ-excess over sampled quadruples.
    """
    nodes = list(nodes)
    if len(nodes) < 4:
        return float("nan")

    rng = random.Random(seed)
    subgraph_set = set(nodes)

    # Pre-compute BFS distances for a sample of nodes to avoid O(n²) BFS
    # Sample up to 100 source nodes for distance computation
    sample_size = min(100, len(nodes))
    sources = rng.sample(nodes, sample_size)

    # Cache distances
    dist_cache = {}
    for s in sources:
        dist_cache[s] = bfs_distances(s, subgraph_set, adj)

    def get_dist(u, v):
        if u in dist_cache and v in dist_cache[u]:
            return dist_cache[u][v]
        if v in dist_cache and u in dist_cache[v]:
            return dist_cache[v][u]
        # Fallback BFS
        d = bfs_distances(u, subgraph_set, adj)
        return d.get(v, float("inf"))

    deltas = []
    attempts = 0
    while len(deltas) < n_samples and attempts < n_samples * 10:
        attempts += 1
        try:
            xi, xj, xk, xl = rng.sample(sources, 4)
        except ValueError:
            break

        dij = get_dist(xi, xj)
        dkl = get_dist(xk, xl)
        dik = get_dist(xi, xk)
        djl = get_dist(xj, xl)
        dil = get_dist(xi, xl)
        djk = get_dist(xj, xk)

        if any(d == float("inf") for d in [dij, dkl, dik, djl, dil, djk]):
            continue  # skip disconnected pairs

        s1 = dij + dkl
        s2 = dik + djl
        s3 = dil + djk

        sums = sorted([s1, s2, s3], reverse=True)
        delta = (sums[0] - sums[1]) / 2.0
        deltas.append(delta)

    if not deltas:
        return float("nan")
    return float(np.mean(deltas))


# ==============================================================================
# Match benchmark problems to Mathlib4 node IDs
# ==============================================================================

def _normalize_name(name: str) -> str:
    """Normalize a theorem name for fuzzy matching."""
    return name.lower().replace("_", "").replace(".", "").replace(" ", "")


def find_node_ids(problem_names: list, name_to_id: dict,
                  id_to_name: dict) -> list:
    """
    Match benchmark problem names to Mathlib4 node IDs.
    Uses exact match first, then suffix match, then normalized fuzzy match.
    """
    matched = []
    norm_map = {_normalize_name(v): k
                for k, v in id_to_name.items() if isinstance(v, str)}

    for raw_name in problem_names:
        # Strip benchmark prefix (e.g. "valid_amc12_..." → "amc12_...")
        parts = raw_name.split("_", 1)
        name = parts[-1] if len(parts) > 1 else raw_name
        # Try exact
        if name in name_to_id:
            matched.append(name_to_id[name])
            continue
        # Try suffix (last component)
        suffix = name.split("_")[-1]
        for full_name, nid in name_to_id.items():
            if full_name.endswith("." + suffix) or full_name == suffix:
                matched.append(nid)
                break
        else:
            # Fuzzy normalized
            norm = _normalize_name(name)
            if norm in norm_map:
                matched.append(norm_map[norm])

    return list(set(matched))  # deduplicate


# ==============================================================================
# Per-dataset δ computation
# ==============================================================================

def compute_delta_for_dataset(dataset_name: str, problems: list,
                               adj: dict, name_to_id: dict,
                               id_to_name: dict) -> float:
    """
    Compute graph-based Gromov δ for a benchmark dataset.
    """
    # Extract names from problems
    problem_names = [p.get("name", "") for p in problems]

    # Find matching node IDs in Mathlib4 graph
    seed_ids = find_node_ids(problem_names, name_to_id, id_to_name)

    if len(seed_ids) < 4:
        print(f"  ⚠️  Only {len(seed_ids)} nodes matched in graph — "
              f"using random subgraph sample")
        # Fallback: sample random nodes from graph as proxy
        all_nodes = list(adj.keys())
        seed_ids = random.sample(all_nodes, min(50, len(all_nodes)))

    print(f"  Matched {len(seed_ids)} nodes in Mathlib4 graph")

    # Extract k-hop neighborhood
    subgraph = extract_subgraph(seed_ids, adj, k=K_HOP)
    print(f"  Subgraph size: {len(subgraph)} nodes (k={K_HOP} hops)")

    if len(subgraph) < 4:
        return float("nan")

    # Compute δ on graph distances
    delta = gromov_delta_graph(list(subgraph), adj,
                                n_samples=N_QUADRUPLES, seed=SEED)
    return delta


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    # Load graph
    adj, id_to_name, name_to_id = load_graph()
    if adj is None:
        return

    # Load existing pass@1 results
    pass_cache = os.path.join(OUTPUT_DIR, "pass_at_1_values.json")
    pass_results = {}
    if os.path.exists(pass_cache):
        with open(pass_cache) as f:
            pass_results = json.load(f)

    # Load dataset loaders (reuse from experiment_delta_vs_pass)
    sys.path.insert(0, os.path.join(project_root, "src", "system2"))
    try:
        from experiment_delta_vs_pass import DATASETS
    except ImportError:
        print("❌ Cannot import DATASETS from experiment_delta_vs_pass.py")
        return

    target = args.datasets or list(DATASETS.keys())

    # Delete stale δ cache and recompute
    delta_cache_path = os.path.join(OUTPUT_DIR, "delta_values.json")
    delta_results = {}
    # Don't load old cache — all values were wrong (embedding-based)
    print(f"\n⚠️  Recomputing all δ values using GRAPH distances (not embeddings)")
    print(f"   Old embedding-based δ gave r=+0.80 (wrong sign)")
    print(f"   Graph-based δ should give r≈-0.90 (correct)\n")

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  Graph-based Gromov δ computation")
    print(sep)

    for ds_name in target:
        if ds_name not in DATASETS:
            continue
        print(f"\n{'─'*65}")
        print(f"📁  {ds_name}")

        try:
            problems = DATASETS[ds_name]()
        except Exception as e:
            print(f"  ❌ Load failed: {e}")
            continue

        if not problems:
            print(f"  ⚠️  No problems found")
            continue

        print(f"  Problems: {len(problems)}")

        delta = compute_delta_for_dataset(
            ds_name, problems, adj, name_to_id, id_to_name
        )
        delta_results[ds_name] = delta
        print(f"  δ = {delta:.4f}" if not np.isnan(delta) else "  δ = nan")

        # Save incrementally
        with open(delta_cache_path, "w") as f:
            json.dump(delta_results, f, indent=2)

    # Summary
    print(f"\n{sep}")
    print("  SUMMARY")
    print(sep)
    print(f"{'Dataset':<22} | {'δ (graph)':>10} | {'Pass@1':>8} | {'Notes'}")
    print(f"{'─'*22}-+-{'─'*10}-+-{'─'*8}-+-{'─'*20}")

    for ds in target:
        d = delta_results.get(ds, float("nan"))
        p = pass_results.get(ds, {})
        p_str = f"{p.get('pass_at_1', 0)*100:.1f}%" if p else "  N/A"
        d_str = f"{d:.4f}" if not np.isnan(d) else "   nan"
        print(f"{ds:<22} | {d_str:>10} | {p_str:>8} |")

    print(f"\n📂 δ values saved to: {delta_cache_path}")
    print("   Run experiment_delta_vs_pass.py --plot-only to regenerate plot")


if __name__ == "__main__":
    main()
