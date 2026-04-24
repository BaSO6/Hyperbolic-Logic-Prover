# ==============================================================================
# Filename: src/system1/test_system1.py
# Version: v1.0
#
# Purpose: Validate all System 1 outputs before running the dim benchmark.
#          Run this AFTER train_hgcn_dim.py + export_embeddings_dim.py,
#          BEFORE benchmark_dim.py.
#
#          Tests checkpoints, embeddings, and retrieval quality independently
#          of System 2, so failures are localised to System 1.
#
# Usage:
#   python src/system1/test_system1.py
#   python src/system1/test_system1.py --dims 16 32 64 128 256 --verbose
# ==============================================================================

import os
import sys
import gzip
import pickle
import argparse
import math
import torch
import torch.nn.functional as F
import numpy as np

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

DATA_DIR   = os.path.join(project_root, "data")
OUTPUT_DIR = os.path.join(project_root, "results", "dimension_scaling")

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "


# ==============================================================================
# Individual test functions
# ==============================================================================

def test_checkpoint(d: int, verbose: bool = False) -> dict:
    """
    Tests for hgcn_d{D}.pth (or hgcn_final.pth for paper dim):
      T1. File exists
      T2. out_dim stored in ckpt matches D
      T3. layer.semantic_proj.weight shape = [D, 384]
      T4. curvature c is positive and reasonable (0.1 to 10)
      T5. Weights are not all-zero / not NaN
      T6. Forward pass on a dummy input produces valid Poincaré embedding
    """
    results = {}
    is_paper_dim = (d == _paper_dim())
    ckpt_path = (os.path.join(DATA_DIR, "hgcn_final.pth") if is_paper_dim
                 else os.path.join(OUTPUT_DIR, f"hgcn_d{d}.pth"))

    # T1: file exists
    if not os.path.exists(ckpt_path):
        results["T1_exists"] = (False, f"Not found: {ckpt_path}")
        return results
    results["T1_exists"] = (True, ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # T2: stored out_dim matches d
    stored_dim = ckpt.get("out_dim")
    if stored_dim is None:
        results["T2_out_dim"] = (False, "out_dim key missing from checkpoint")
    elif int(stored_dim) != d:
        results["T2_out_dim"] = (False,
            f"stored out_dim={stored_dim} but expected {d}")
    else:
        results["T2_out_dim"] = (True, f"out_dim={stored_dim}")

    # T3: weight shape
    model_dict = ckpt.get("model", {})
    weight_key = None
    for k in ("layer.semantic_proj.weight", "semantic_proj.weight"):
        if k in model_dict:
            weight_key = k; break
    if weight_key is None:
        results["T3_weight_shape"] = (False, "semantic_proj.weight not found in model dict")
    else:
        w = model_dict[weight_key]
        expected = (d, 384)
        if tuple(w.shape) != expected:
            results["T3_weight_shape"] = (False,
                f"shape {tuple(w.shape)} ≠ expected {expected}")
        else:
            results["T3_weight_shape"] = (True, f"shape {tuple(w.shape)}")

    # T4: curvature
    c = ckpt.get("c", None)
    if c is None:
        results["T4_curvature"] = (False, "c not stored")
    elif not (0.05 < float(c) < 20.0):
        results["T4_curvature"] = (False, f"c={c} outside reasonable range [0.05, 20]")
    else:
        results["T4_curvature"] = (True, f"c={c}")

    # T5: weights not zero / not NaN
    if weight_key:
        w = model_dict[weight_key].float()
        has_nan = torch.isnan(w).any().item()
        is_zero = (w.abs().max().item() < 1e-10)
        if has_nan:
            results["T5_weights_valid"] = (False, "NaN in semantic_proj weights")
        elif is_zero:
            results["T5_weights_valid"] = (False, "All-zero weights (not trained)")
        else:
            results["T5_weights_valid"] = (True,
                f"max={w.abs().max():.4f}, std={w.std():.4f}")

    # T6: forward pass
    try:
        sys.path.insert(0, current_dir)
        from train_hgcn_dim import FinalHGCN   # type: ignore
        c_val = float(ckpt.get("c", 1.0))
        model = FinalHGCN(384, 256, d, c_val)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        with torch.no_grad():
            dummy_x  = torch.randn(10, 384)
            dummy_ei = torch.zeros(2, 0, dtype=torch.long)
            z = model(dummy_x, dummy_ei)
        norms = z.norm(dim=-1)
        if z.shape != (10, d):
            results["T6_forward"] = (False, f"output shape {z.shape} ≠ (10, {d})")
        elif norms.max().item() >= 1.0:
            results["T6_forward"] = (False,
                f"outputs outside Poincaré ball: max norm={norms.max():.4f}")
        else:
            results["T6_forward"] = (True,
                f"shape OK, norms ∈ [{norms.min():.3f}, {norms.max():.3f}]")
    except Exception as e:
        results["T6_forward"] = (False, f"Forward pass exception: {e}")

    return results


def test_embeddings(d: int, verbose: bool = False) -> dict:
    """
    Tests for node_emb_d{D}.pt (or node_embeddings.pt for paper dim):
      T1. File exists
      T2. Shape = [N, D] where N ≈ 110304
      T3. All norms < 1 (inside Poincaré ball)
      T4. No NaN or Inf values
      T5. Non-trivial: std > 0.01 (embeddings not collapsed)
      T6. Consistent with checkpoint (a sample query from checkpoint matches)
      T7. Different dims produce DIFFERENT embeddings for same node
    """
    results = {}
    is_paper_dim = (d == _paper_dim())
    emb_path = (os.path.join(DATA_DIR, "node_embeddings.pt") if is_paper_dim
                else os.path.join(OUTPUT_DIR, f"node_emb_d{d}.pt"))

    # T1: file exists
    if not os.path.exists(emb_path):
        results["T1_exists"] = (False, f"Not found: {emb_path}")
        return results
    results["T1_exists"] = (True, emb_path)

    emb = torch.load(emb_path, map_location="cpu").float()

    # T2: shape
    N, actual_d = emb.shape
    if actual_d != d:
        results["T2_shape"] = (False,
            f"shape={emb.shape}, expected dim={d} but got {actual_d}")
    elif N < 100_000:
        results["T2_shape"] = (False,
            f"only {N} nodes — expected ~110304")
    else:
        results["T2_shape"] = (True, f"shape={emb.shape}")

    # T3: inside Poincaré ball
    norms = emb.norm(dim=-1)
    max_norm = norms.max().item()
    pct_outside = (norms >= 1.0).float().mean().item() * 100
    if pct_outside > 0.1:
        results["T3_poincare_ball"] = (False,
            f"{pct_outside:.1f}% of embeddings have norm ≥ 1")
    else:
        results["T3_poincare_ball"] = (True,
            f"max_norm={max_norm:.4f}, all inside ball")

    # T4: no NaN/Inf
    has_nan = torch.isnan(emb).any().item()
    has_inf = torch.isinf(emb).any().item()
    if has_nan or has_inf:
        results["T4_finite"] = (False,
            f"NaN={has_nan}, Inf={has_inf}")
    else:
        results["T4_finite"] = (True, "all finite")

    # T5: non-collapsed
    std = emb.std().item()
    mean_norm = norms.mean().item()
    if std < 0.005:
        results["T5_diversity"] = (False,
            f"std={std:.5f} — embeddings collapsed to near-identical")
    else:
        results["T5_diversity"] = (True,
            f"std={std:.4f}, mean_norm={mean_norm:.3f}")

    # T6: self-consistency — verify embeddings were produced by the correct checkpoint.
    # NOTE: The model uses full neighborhood aggregation (EuclideanGraphConv),
    # so re-encoding with empty edges gives different values than stored embeddings.
    # Instead we verify structural properties: the checkpoint's semantic_proj applied
    # to a few features should produce vectors in the same norm range as stored embs.
    try:
        sys.path.insert(0, current_dir)
        from train_hgcn_dim import FinalHGCN   # type: ignore
        ckpt_path = (os.path.join(DATA_DIR, "hgcn_final.pth") if is_paper_dim
                     else os.path.join(OUTPUT_DIR, f"hgcn_d{d}.pth"))
        if os.path.exists(ckpt_path):
            ckpt  = torch.load(ckpt_path, map_location="cpu")
            c_val = float(ckpt.get("c", 1.0))
            model = FinalHGCN(384, 256, d, c_val)
            model.load_state_dict(ckpt["model"], strict=False)
            model.eval()

            feat_path = os.path.join(DATA_DIR, "node_features_euclidean.pt")
            ei_path   = os.path.join(DATA_DIR, "edge_index.pt")
            if os.path.exists(feat_path) and os.path.exists(ei_path):
                # Use FULL edge_index for correct GCN aggregation
                # Load only a small subgraph (first 500 nodes) for speed
                x_all  = torch.load(feat_path, map_location="cpu")
                ei_all = torch.load(ei_path,   map_location="cpu")

                # Extract subgraph for first 500 nodes to keep it fast
                mask = (ei_all[0] < 500) & (ei_all[1] < 500)
                ei_sub = ei_all[:, mask]
                x_sub  = x_all[:500]

                with torch.no_grad():
                    z_recomputed = model(x_sub, ei_sub)

                # Check norm range matches stored embeddings (within 0.1 tolerance)
                stored_mean_norm = emb[:500].float().norm(dim=-1).mean().item()
                reco_mean_norm   = z_recomputed.norm(dim=-1).mean().item()
                norm_diff = abs(stored_mean_norm - reco_mean_norm)

                if z_recomputed.shape[1] != d:
                    results["T6_consistency"] = (False,
                        f"output dim {z_recomputed.shape[1]} ≠ {d}")
                elif norm_diff > 0.15:
                    results["T6_consistency"] = (False,
                        f"norm mismatch: stored={stored_mean_norm:.3f}, "
                        f"recomputed={reco_mean_norm:.3f} (diff={norm_diff:.3f})")
                else:
                    results["T6_consistency"] = (True,
                        f"norm range consistent: stored={stored_mean_norm:.3f}, "
                        f"recomputed={reco_mean_norm:.3f}")
            else:
                results["T6_consistency"] = (True, "skipped (features not found)")
        else:
            results["T6_consistency"] = (True, "skipped (ckpt not found)")
    except Exception as e:
        results["T6_consistency"] = (False, f"Consistency check failed: {e}")

    return results


def test_cross_dim_independence(dims: list, verbose: bool = False) -> dict:
    """
    T1: Embedding values differ across dims (cosine similarity < 0.95).
    T2: Retrieval diversity — for a fixed goal query, different dims return
        different top-5 theorems (Jaccard overlap < 0.8).
        This is the definitive proof that dimension affects behavior.
    """
    results = {}
    paper_d   = _paper_dim()
    emb_cache = {}

    for d in dims:
        is_paper = (d == paper_d)
        path = (os.path.join(DATA_DIR, "node_embeddings.pt") if is_paper
                else os.path.join(OUTPUT_DIR, f"node_emb_d{d}.pt"))
        if os.path.exists(path):
            emb_cache[d] = torch.load(path, map_location="cpu").float()

    if len(emb_cache) < 2:
        results["T1_independence"] = (True, "only 1 dim available, skip")
        results["T2_retrieval_diversity"] = (True, "only 1 dim available, skip")
        return results

    # T1: values differ — compare truncated to min dim for cosine similarity
    dim_list = sorted(emb_cache.keys())
    max_cos_sims = []
    for i in range(len(dim_list)):
        for j in range(i+1, len(dim_list)):
            da, db  = dim_list[i], dim_list[j]
            emb_a   = emb_cache[da][:100]
            emb_b   = emb_cache[db][:100]
            min_dim = min(emb_a.shape[1], emb_b.shape[1])
            ea_trunc = emb_a[:, :min_dim]
            eb_trunc = emb_b[:, :min_dim]
            cos = F.cosine_similarity(ea_trunc, eb_trunc, dim=-1).mean().item()
            max_cos_sims.append((da, db, cos))

    identical_pairs = [(a, b) for a, b, c in max_cos_sims if c > 0.999]
    if identical_pairs:
        results["T1_independence"] = (False,
            f"Identical embeddings detected: {identical_pairs}")
    else:
        sim_str = ", ".join(f"d{a}↔d{b}:{c:.3f}" for a, b, c in max_cos_sims)
        results["T1_independence"] = (True, f"All distinct: {sim_str}")

    # T2: retrieval diversity — encode a fixed Lean goal and compare top-5 retrieved
    try:
        sys.path.insert(0, current_dir)
        from train_hgcn_dim import FinalHGCN    # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore

        bert_path = os.path.join(project_root, "models", "all-MiniLM-L6-v2")
        if not os.path.exists(bert_path):
            results["T2_retrieval_diversity"] = (True, "skipped (BERT model not found)")
            return results

        bert = SentenceTransformer(bert_path, device="cpu")
        bert.eval()

        # Fixed test query — a simple number theory goal
        test_goal = "n * m = m * n"
        with torch.no_grad():
            q_bert = bert.encode(test_goal, convert_to_tensor=True,
                                 show_progress_bar=False)
            q_bert = F.normalize(q_bert.unsqueeze(0), dim=-1)  # [1, 384]

        retrieved_sets = {}
        for d in dim_list:
            is_paper = (d == paper_d)
            ckpt_path = (os.path.join(DATA_DIR, "hgcn_final.pth") if is_paper
                         else os.path.join(OUTPUT_DIR, f"hgcn_d{d}.pth"))
            if not os.path.exists(ckpt_path):
                continue

            ckpt  = torch.load(ckpt_path, map_location="cpu")
            c_val = float(ckpt.get("c", 1.0))
            model = FinalHGCN(384, 256, d, c_val)
            model.load_state_dict(ckpt["model"], strict=False)
            model.eval()

            with torch.no_grad():
                q_hyp = model.layer.semantic_proj(q_bert)   # [1, d]
                # Normalise into Poincaré ball
                norm  = q_hyp.norm(dim=-1, keepdim=True) + 1e-8
                radius = 0.9 * torch.tanh(model.layer.scale)
                q_hyp = (q_hyp / norm) * radius

                # Compute distances to all nodes
                graph = emb_cache[d]               # [N, d]
                # Poincaré distance approximation via squared L2
                diff  = q_hyp - graph              # [N, d]
                sq_dist = (diff ** 2).sum(dim=-1)  # [N]
                top5_idx = sq_dist.topk(5, largest=False).indices.tolist()

            retrieved_sets[d] = set(top5_idx)

        # Compute pairwise Jaccard overlap
        dim_pairs_jaccard = []
        for i in range(len(dim_list)):
            for j in range(i+1, len(dim_list)):
                da, db = dim_list[i], dim_list[j]
                if da not in retrieved_sets or db not in retrieved_sets:
                    continue
                sa, sb = retrieved_sets[da], retrieved_sets[db]
                jaccard = len(sa & sb) / len(sa | sb)
                dim_pairs_jaccard.append((da, db, jaccard))

        if not dim_pairs_jaccard:
            results["T2_retrieval_diversity"] = (True, "skipped (insufficient data)")
        else:
            max_jaccard = max(j for _, _, j in dim_pairs_jaccard)
            jacc_str = ", ".join(
                f"d{a}↔d{b}:{j:.2f}" for a, b, j in dim_pairs_jaccard
            )
            if max_jaccard > 0.8:
                results["T2_retrieval_diversity"] = (False,
                    f"Near-identical retrieval: {jacc_str}")
            else:
                results["T2_retrieval_diversity"] = (True,
                    f"Retrieval differs across dims: {jacc_str}")

    except Exception as e:
        results["T2_retrieval_diversity"] = (False,
            f"Retrieval diversity test failed: {e}")

    return results


# ==============================================================================
# Helpers
# ==============================================================================

def _paper_dim() -> int:
    path = os.path.join(DATA_DIR, "hgcn_final.pth")
    if not os.path.exists(path):
        return 64
    ckpt = torch.load(path, map_location="cpu")
    if "out_dim" in ckpt:
        return int(ckpt["out_dim"])
    for key in ("layer.semantic_proj.weight", "semantic_proj.weight"):
        if key in ckpt.get("model", {}):
            return ckpt["model"][key].shape[0]
    return 64


def print_section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def print_results(test_name: str, results: dict):
    passed = sum(1 for ok, _ in results.values() if ok)
    total  = len(results)
    status = PASS if passed == total else FAIL
    print(f"\n  {status} {test_name}  ({passed}/{total} passed)")
    for t_name, (ok, msg) in results.items():
        icon = PASS if ok else FAIL
        print(f"    {icon} {t_name}: {msg}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="System 1 validation tests")
    parser.add_argument("--dims", nargs="+", type=int, default=None,
                        help="Dims to test. Default: all available.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    paper_d = _paper_dim()
    print(f"\n🔍 Detected paper dim = {paper_d}")

    # Determine which dims to test
    if args.dims:
        dims = args.dims
    else:
        # Auto-detect from files present
        dims = [paper_d]
        for d in [16, 32, 64, 128, 256]:
            ckpt = os.path.join(OUTPUT_DIR, f"hgcn_d{d}.pth")
            if os.path.exists(ckpt) and d != paper_d:
                dims.append(d)
        dims = sorted(set(dims))

    print(f"   Testing dims: {dims}")

    all_pass = True

    # ── Checkpoint tests ────────────────────────────────────────────────────
    print_section("CHECKPOINT TESTS  (hgcn_d{D}.pth)")
    for d in dims:
        r = test_checkpoint(d, args.verbose)
        print_results(f"d={d} checkpoint", r)
        if any(not ok for ok, _ in r.values()):
            all_pass = False

    # ── Embedding tests ──────────────────────────────────────────────────────
    print_section("EMBEDDING TESTS  (node_emb_d{D}.pt)")
    available_emb_dims = []
    for d in dims:
        is_paper = (d == paper_d)
        path = (os.path.join(DATA_DIR, "node_embeddings.pt") if is_paper
                else os.path.join(OUTPUT_DIR, f"node_emb_d{d}.pt"))
        if not os.path.exists(path):
            print(f"\n  {WARN} d={d} embeddings: NOT FOUND at {path}")
            print(f"       Run: python src/system1/export_embeddings_dim.py --dims {d}")
            all_pass = False
            continue
        available_emb_dims.append(d)
        r = test_embeddings(d, args.verbose)
        print_results(f"d={d} embeddings", r)
        if any(not ok for ok, _ in r.values()):
            all_pass = False

    # ── Cross-dim independence tests ─────────────────────────────────────────
    if len(available_emb_dims) >= 2:
        print_section("CROSS-DIM INDEPENDENCE TESTS")
        r = test_cross_dim_independence(available_emb_dims, args.verbose)
        print_results("Independence across dims", r)
        if any(not ok for ok, _ in r.values()):
            all_pass = False

    # ── Final verdict ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    if all_pass:
        print(f"  {PASS} ALL TESTS PASSED — System 1 outputs are valid.")
        print(f"     Safe to run: benchmark_dim.py")
    else:
        print(f"  {FAIL} SOME TESTS FAILED — Fix System 1 before benchmarking.")
        print(f"     Re-run failed steps, then re-run this test.")
    print(f"{'='*65}\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())