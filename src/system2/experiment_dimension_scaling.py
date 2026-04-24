# ==============================================================================
# Filename: src/system2/experiment_dimension_scaling.py
# Version: v3.0 — System 1 Retraining Approach
#
# WHY v1 AND v2 FAILED:
#   Both tried to monkey-patch a live agent at runtime. This breaks because
#   RiemannSearchAgent is deeply coupled: the HGCN checkpoint, the query
#   encoder, and node_embeddings.pt must ALL use the same OUT_DIM.
#   You cannot change d at inference time without re-running System 1.
#
# CORRECT APPROACH (inspired by the uploaded System 1 scripts):
#   The dimension lives in System 1 → train_final.py:OUT_DIM.
#   For each target d, this script:
#     Step 1. Retrain HGCN with that OUT_DIM          (train_final logic)
#     Step 2. Export embeddings at that d              (export_embeddings logic)
#     Step 3. Run benchmark_minif2f worker with the
#             d-specific checkpoint + embeddings       (subprocess, no monkey-patch)
#
#   Each dimension is fully self-consistent:
#     hgcn_d{D}.pth  → checkpoint with OUT_DIM=D
#     node_emb_d{D}.pt → [N, D] embeddings
#     Benchmark worker loads these files directly — zero patching.
#
# HOW TO RUN:
#   # First: check your actual production OUT_DIM
#   python3 -c "import torch; c=torch.load('data/hgcn_final.pth'); \
#     print('out_dim:', c.get('out_dim'), \
#     'weight:', c['model']['layer.semantic_proj.weight'].shape)"
#
#   # Then run (d=<actual> uses the paper result, no retraining):
#   python src/system2/experiment_dimension_scaling.py \
#       --dims 16 32 64 128 256 --n 50 --paper-dim 64
#
#   # Full run (replace 64 with your actual paper OUT_DIM):
#   python src/system2/experiment_dimension_scaling.py \
#       --dims 16 32 64 128 256 --n 240 --paper-dim 64
# ==============================================================================

import os
import sys
import re
import glob
import time
import random
import argparse
import json
import gc
import math
import multiprocessing as mp
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["HF_ENDPOINT"]             = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DATA_DIR   = os.path.join(project_root, "data")
OUTPUT_DIR = os.path.join(project_root, "results", "dimension_scaling")
MODEL_PATH = (
    os.path.join(project_root, "models/DeepSeek-Prover-V1.5-RL")
)
CKPT_ORIG  = os.path.join(DATA_DIR, "hgcn_final.pth")   # paper checkpoint
EMB_ORIG   = os.path.join(DATA_DIR, "node_embeddings.pt")

SEED                   = 42
MAX_SECONDS_PER_PROBLEM = 300
HEARTBEAT_TIMEOUT      = 90

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── detect paper OUT_DIM from the actual checkpoint on disk ──────────────────
def detect_paper_dim() -> int:
    if not os.path.exists(CKPT_ORIG):
        return 64
    ckpt = torch.load(CKPT_ORIG, map_location="cpu")
    # Prefer stored field; fall back to weight shape
    if "out_dim" in ckpt:
        return int(ckpt["out_dim"])
    model_dict = ckpt.get("model", {})
    for key in ("layer.semantic_proj.weight", "semantic_proj.weight"):
        if key in model_dict:
            return model_dict[key].shape[0]
    return 64


# ==============================================================================
# Step 1 — Retrain HGCN at target OUT_DIM
# ==============================================================================

# Reuse the architecture from train_final.py verbatim

from src.system1.manifold_math import PoincareBall   # type: ignore


class _EuclideanGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        x_trans = self.linear(x)
        if edge_index.size(1) == 0:
            return x_trans
        row, col = edge_index
        out = torch.zeros_like(x_trans)
        deg = torch.zeros(x.size(0), 1, device=x.device)
        deg.index_add_(0, row, torch.ones(row.size(0), 1, device=x.device))
        out.index_add_(0, row, x_trans[col])
        return F.relu(out / (deg + 1e-8))


class _HyperbolicResidualLayer(nn.Module):
    def __init__(self, in_dim, out_dim, c=1.0):
        super().__init__()
        self.manifold    = PoincareBall(c)
        self.semantic_proj = nn.Linear(in_dim, out_dim)
        self.structure_proj = nn.Linear(in_dim, out_dim)
        self.graph_conv  = _EuclideanGraphConv(out_dim, out_dim)
        self.gate        = nn.Linear(out_dim * 2, 1)
        self.scale       = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_index):
        z_sem    = self.semantic_proj(x)
        z_struct = F.relu(self.structure_proj(x))
        z_struct = self.graph_conv(z_struct, edge_index)
        alpha    = torch.sigmoid(
            self.gate(torch.cat([z_sem, z_struct], dim=-1))
        )
        z_tan    = alpha * z_sem + (1 - alpha) * z_struct
        x_norm   = z_tan.norm(dim=-1, keepdim=True) + 1e-8
        radius   = 0.9 * torch.tanh(self.scale)
        return self.manifold.expmap0(z_tan / x_norm * radius)


class _FinalHGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, c=1.0):
        super().__init__()
        self.layer    = _HyperbolicResidualLayer(in_dim, out_dim, c)
        self.manifold = self.layer.manifold
        self.hidden_dim = hidden_dim
        self.out_dim    = out_dim
        self.c          = c

    def forward(self, x, edge_index):
        return self.layer(x, edge_index)


def train_hgcn_for_dim(out_dim: int, ckpt_save_path: str,
                        device: str = "cuda", epochs: int = 201) -> bool:
    """
    Retrain the HGCN with the given OUT_DIM and save to ckpt_save_path.
    Returns True on success.
    Uses HIDDEN_DIM=256 (same as train_final.py v12.1).
    """
    if os.path.exists(ckpt_save_path):
        print(f"  ↩️  Checkpoint exists: {ckpt_save_path}")
        return True

    HIDDEN_DIM = 256
    CURVATURE  = 1.0
    dev = torch.device(device)

    x_path  = os.path.join(DATA_DIR, "node_features_euclidean.pt")
    ei_path = os.path.join(DATA_DIR, "edge_index.pt")
    if not os.path.exists(x_path) or not os.path.exists(ei_path):
        print(f"  ❌ Features/edges not found in {DATA_DIR}")
        return False

    print(f"  📥 Loading features for retraining at d={out_dim}...")
    x          = torch.load(x_path, map_location=dev)
    edge_index = torch.load(ei_path, map_location=dev)

    model     = _FinalHGCN(x.shape[1], HIDDEN_DIM, out_dim, CURVATURE).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    print(f"  🚀 Training HGCN d={out_dim}  ({epochs} epochs, {x.shape[0]} nodes)...")
    has_edges = edge_index.size(1) > 100

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model(x, edge_index)

        norms    = z.norm(dim=-1)
        loss_reg = ((norms - 0.8) ** 2).mean()
        loss_link = torch.tensor(0.0, device=dev)

        if has_edges:
            perm      = torch.randperm(edge_index.size(1), device=dev)[:10000]
            u, v      = edge_index[:, perm]
            pos_dist  = model.manifold.dist(z[u], z[v])
            neg_v     = torch.randint(0, z.size(0), (len(u),), device=dev)
            neg_dist  = model.manifold.dist(z[u], z[neg_v])
            loss_link = F.relu(pos_dist - neg_dist + 1.0).mean()

        loss = loss_link + 0.1 * loss_reg
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"    Ep {epoch:03d} | loss={loss.item():.4f} "
                  f"(link={loss_link:.4f}, reg={loss_reg:.4f})")

    torch.save({
        "model":      model.state_dict(),
        "c":          CURVATURE,
        "hidden_dim": HIDDEN_DIM,
        "out_dim":    out_dim,
        "note":       f"dim_scaling_d{out_dim}",
    }, ckpt_save_path)
    print(f"  ✅ Checkpoint saved: {ckpt_save_path}")
    return True


# ==============================================================================
# Step 2 — Export embeddings at target OUT_DIM
# ==============================================================================

def export_embeddings_for_dim(out_dim: int, ckpt_path: str,
                               emb_save_path: str,
                               device: str = "cuda") -> bool:
    """Forward-pass the trained HGCN and save node_embeddings at d=out_dim."""
    if os.path.exists(emb_save_path):
        print(f"  ↩️  Embeddings exist: {emb_save_path}")
        return True

    HIDDEN_DIM = 256
    dev = torch.device(device)

    x_path  = os.path.join(DATA_DIR, "node_features_euclidean.pt")
    ei_path = os.path.join(DATA_DIR, "edge_index.pt")
    if not (os.path.exists(x_path) and os.path.exists(ckpt_path)):
        print(f"  ❌ Missing features or checkpoint")
        return False

    x          = torch.load(x_path, map_location=dev)
    edge_index = torch.load(ei_path, map_location=dev)
    ckpt       = torch.load(ckpt_path, map_location=dev)

    model = _FinalHGCN(x.shape[1], HIDDEN_DIM, out_dim, 1.0).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"  🚀 Exporting embeddings d={out_dim} ({x.shape[0]} nodes)...")
    with torch.no_grad():
        z = model(x, edge_index)

    torch.save(z.cpu(), emb_save_path)
    print(f"  ✅ Embeddings saved {z.shape}: {emb_save_path}")
    return True


# ==============================================================================
# Step 3 — Benchmark with dimension-specific assets
# ==============================================================================

def load_minif2f_test(n: int, seed: int) -> list:
    _VALID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.']*$")
    root   = os.path.join(DATA_DIR, "miniF2F", "MiniF2F", "Test")
    if not os.path.exists(root):
        print(f"❌ Not found: {root}"); return []

    probs, seen = [], set()
    for fpath in glob.glob(os.path.join(root, "**", "*.lean"), recursive=True):
        if any(b in fpath for b in ("lake-packages", "_build", "_manual")):
            continue
        base = os.path.basename(fpath).replace(".lean", "")
        try:
            content = open(fpath, encoding="utf-8").read()
        except Exception:
            continue
        in_block = False
        lines    = content.splitlines()
        i        = 0
        while i < len(lines):
            line = lines[i]
            if "/-" in line:  in_block = True
            if "-/" in line:
                in_block = False; i += 1; continue
            if in_block or line.strip().startswith("--"):
                i += 1; continue
            m = re.match(
                r"^(?:@\[.*?\]\s*)?(?:theorem|lemma)\s+([^\s\{(:\[]+)",
                line.strip())
            if not m: i += 1; continue
            name = m.group(1).strip()
            if not _VALID.match(name): i += 1; continue
            decl_lines = [line]
            j = i + 1
            while j < len(lines) and ":=" not in "".join(decl_lines):
                decl_lines.append(lines[j]); j += 1
            decl_raw = " ".join(l.strip() for l in decl_lines)
            if ":=" in decl_raw:
                decl_raw = decl_raw[:decl_raw.index(":=")].strip()
            decl = f"theorem {name} {decl_raw.split(name,1)[-1].strip()}"
            uid  = f"test_{base}_{name.replace('.','_')}"
            if uid not in seen:
                seen.add(uid)
                probs.append({"name": uid, "decl": decl})
            i = j

    if n > 0:
        probs = random.Random(seed).sample(probs, min(n, len(probs)))
    print(f"📂 MiniF2F-test: {len(probs)} problems")
    return probs


_DONE    = "DONE"
_CRASHED = "CRASHED"


def _worker_benchmark(ckpt_path: str, emb_path: str,
                       problem_queue, result_queue, csv_path: str):
    """
    Subprocess: loads RiemannSearchAgent with dim-specific checkpoint + embeddings.
    No monkey-patching — everything is self-consistent.
    """
    import queue as _queue
    import threading

    def _send(kind):
        try: result_queue.put({"_sentinel": kind})
        except Exception: pass

    # Temporarily swap the standard data files for this dimension
    orig_ckpt = os.path.join(DATA_DIR, "hgcn_final.pth")
    orig_emb  = os.path.join(DATA_DIR, "node_embeddings.pt")
    tmp_ckpt  = orig_ckpt + ".bak_dim_exp"
    tmp_emb   = orig_emb  + ".bak_dim_exp"

    backed_up_ckpt = False
    backed_up_emb  = False
    try:
        # Back up originals and symlink/copy dimension-specific files
        if os.path.exists(orig_ckpt) and ckpt_path != orig_ckpt:
            shutil.copy2(orig_ckpt, tmp_ckpt)
            shutil.copy2(ckpt_path, orig_ckpt)
            backed_up_ckpt = True
        if os.path.exists(orig_emb) and emb_path != orig_emb:
            shutil.copy2(orig_emb, tmp_emb)
            shutil.copy2(emb_path, orig_emb)
            backed_up_emb = True

        from src.system2.lie_search import RiemannSearchAgent
        agent = RiemannSearchAgent(orig_ckpt, MODEL_PATH, device="cuda")
        print(f"  ✅ Worker loaded (ckpt={os.path.basename(ckpt_path)}, "
              f"emb={os.path.basename(emb_path)})", flush=True)

    except Exception as e:
        print(f"  ❌ Worker init failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        # Restore originals before exiting
        if backed_up_ckpt: shutil.copy2(tmp_ckpt, orig_ckpt)
        if backed_up_emb:  shutil.copy2(tmp_emb,  orig_emb)
        _send(_CRASHED); return

    records = []
    if os.path.exists(csv_path):
        try: records = pd.read_csv(csv_path).to_dict("records")
        except Exception: pass

    while True:
        try:
            task = problem_queue.get(timeout=10)
        except _queue.Empty:
            break
        if task is None:
            break

        holder = [None]
        def _run():
            try:    holder[0] = agent.search(task["decl"], max_steps=40)
            except Exception as ex:
                holder[0] = {"status": f"ScriptCrash:{ex}"}

        t0     = time.time()
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=MAX_SECONDS_PER_PROBLEM)

        raw    = (holder[0] or {}).get("status", "Unknown") \
                 if not thread.is_alive() else "Timeout"
        status = "ScriptCrash" if str(raw).startswith("ScriptCrash") else raw
        elapsed = time.time() - t0

        records.append({"name": task["name"], "status": status,
                        "time": round(elapsed, 2)})
        pd.DataFrame(records).to_csv(csv_path, index=False)
        result_queue.put({"name": task["name"], "status": status})
        gc.collect()
        torch.cuda.empty_cache()

    # Restore originals
    if backed_up_ckpt: shutil.copy2(tmp_ckpt, orig_ckpt)
    if backed_up_emb:  shutil.copy2(tmp_emb,  orig_emb)
    _send(_DONE)


def run_benchmark_for_dim(out_dim: int, ckpt_path: str, emb_path: str,
                           problems: list) -> dict:
    """Run Pass@1 for one dimension using dimension-specific assets."""
    import queue as _queue
    csv_path = os.path.join(OUTPUT_DIR, f"dim_{out_dim}.csv")

    done = set()
    if os.path.exists(csv_path):
        try:
            df    = pd.read_csv(csv_path)
            done  = set(df["name"].unique())
            print(f"  ↩️  Resuming d={out_dim} ({len(done)} done)")
        except Exception: pass

    todo = [p for p in problems if p["name"] not in done]
    if not todo:
        df     = pd.read_csv(csv_path)
        solved = int((df["status"] == "Success").sum())
        return {"dim": out_dim, "solved": solved, "total": len(df),
                "pass": solved / len(df) if len(df) else 0.0}

    try:    mp.set_start_method("spawn", force=True)
    except RuntimeError: pass

    manager = mp.Manager()
    p_queue = manager.Queue()
    r_queue = manager.Queue()
    for p in todo: p_queue.put(p)
    p_queue.put(None)

    worker = mp.Process(target=_worker_benchmark,
                        args=(ckpt_path, emb_path, p_queue, r_queue, csv_path))
    worker.start()

    pbar          = tqdm(total=len(todo), desc=f"  d={out_dim}", leave=False)
    finished      = 0
    last_activity = time.time()

    while True:
        try:
            res = r_queue.get(timeout=2.0)
        except _queue.Empty:
            silence = time.time() - last_activity
            if not worker.is_alive() and r_queue.empty():
                print(f"\n  ⚠️  Worker exited ({len(todo)-finished} unfinished)",
                      flush=True)
                break
            if silence > HEARTBEAT_TIMEOUT:
                print(f"\n  ⚠️  No activity for {HEARTBEAT_TIMEOUT}s — terminate",
                      flush=True)
                worker.terminate(); break
            continue

        last_activity = time.time()
        if isinstance(res, dict) and "_sentinel" in res:
            while True:
                try:
                    ex = r_queue.get_nowait()
                    if "_sentinel" not in ex:
                        finished += 1; pbar.update(1)
                except _queue.Empty:
                    break
            break

        finished += 1; pbar.update(1)
        if finished >= len(todo): break

    pbar.close()
    worker.join(timeout=30)
    if worker.is_alive(): worker.terminate(); worker.join()

    try:
        df     = pd.read_csv(csv_path)
        solved = int((df["status"] == "Success").sum())
        total  = len(df)
    except Exception:
        solved, total = 0, len(problems)

    return {"dim": out_dim, "solved": solved, "total": total,
            "pass": solved / total if total > 0 else 0.0}


# ==============================================================================
# Reporting
# ==============================================================================

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k/n; d = 1+z**2/n
    c = (p+z**2/(2*n))/d
    m = z*math.sqrt(p*(1-p)/n + z**2/(4*n**2))/d
    return c-m, c+m


def print_results(results: list, paper_dim: int):
    print(f"\n{'='*68}")
    print("  DIMENSION SCALING RESULTS")
    print(f"{'='*68}")
    print(f"{'d':>6}  {'Pass@1':>8}  {'Solved/Total':>14}  {'95% CI':<22}  note")
    print("-"*68)
    for r in results:
        lo, hi = wilson(r["solved"], r["total"])
        note   = "← paper" if r["dim"] == paper_dim else ""
        flag   = "⚠️  " if r["total"] < 240 else ""
        print(f"{r['dim']:>6}  {r['pass']*100:>7.2f}%"
              f"  {r['solved']:>6}/{r['total']:<6}"
              f"  [{lo*100:.1f}%,{hi*100:.1f}%]  {flag}{note}")

    print(f"\n📋 LaTeX table row:")
    sorted_r  = sorted(results, key=lambda r: r["dim"])
    dims_str  = " & ".join(str(r["dim"])             for r in sorted_r)
    pass_str  = " & ".join(f"{r['pass']*100:.2f}"    for r in sorted_r)
    print(f"\\textbf{{Dimension $d$}} & {dims_str} \\\\")
    print(f"\\textbf{{Pass@1 (\\%)}} & {pass_str} \\\\")


# ==============================================================================
# Main
# ==============================================================================

def main():
    paper_dim = detect_paper_dim()
    print(f"\n🔍 Detected paper OUT_DIM = {paper_dim}  (from {CKPT_ORIG})")

    parser = argparse.ArgumentParser(
        description="Dimension scaling via System 1 retraining (v3.0)")
    parser.add_argument("--dims", nargs="+", type=int,
                        default=[16, 32, paper_dim, paper_dim*2],
                        help=f"OUT_DIM values to test (paper={paper_dim})")
    parser.add_argument("--n",    type=int, default=50,
                        help="MiniF2F-test problems per dim (50=pilot, 240=full)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=201,
                        help="HGCN training epochs per dim (default 201)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip retraining — only run benchmarks "
                             "(use if checkpoints already exist)")
    parser.add_argument("--paper-dim", type=int, default=None,
                        help="Override the auto-detected paper dim")
    args = parser.parse_args()

    if args.paper_dim:
        paper_dim = args.paper_dim
        print(f"  (overridden to {paper_dim})")

    # Ensure paper_dim is always in the list
    dims = sorted(set(args.dims + [paper_dim]))

    print(f"\n{'='*68}")
    print(f"  Dimension Scaling v3.0 — System 1 Retraining")
    print(f"  dims={dims}  paper_dim={paper_dim}  n={args.n}  epochs={args.epochs}")
    print(f"{'='*68}\n")

    problems = load_minif2f_test(n=args.n, seed=args.seed)
    if not problems: return

    results      = []
    summary_path = os.path.join(OUTPUT_DIR, "dimension_scaling_summary.json")

    # Load any cached summary
    existing = {}
    if os.path.exists(summary_path):
        try:
            for r in json.load(open(summary_path)):
                existing[r["dim"]] = r
        except Exception: pass

    for d in dims:
        print(f"\n{'─'*68}")
        print(f"📐  d = {d}")

        ckpt_path = (CKPT_ORIG if d == paper_dim
                     else os.path.join(OUTPUT_DIR, f"hgcn_d{d}.pth"))
        emb_path  = (EMB_ORIG if d == paper_dim
                     else os.path.join(OUTPUT_DIR, f"node_emb_d{d}.pt"))

        # Check cache
        if d in existing and existing[d].get("total", 0) >= args.n:
            res = existing[d]
            lo, hi = wilson(res["solved"], res["total"])
            print(f"  ↩️  Cached: {res['pass']*100:.2f}% "
                  f"CI=[{lo*100:.1f}%,{hi*100:.1f}%]")
            results.append(res)
            continue

        # Step 1: Retrain (skip for paper_dim)
        if d == paper_dim:
            print(f"  ✅ Using paper checkpoint (no retraining needed)")
        elif not args.skip_train:
            ok = train_hgcn_for_dim(d, ckpt_path, epochs=args.epochs)
            if not ok:
                print(f"  ❌ Training failed for d={d}, skipping")
                continue
        else:
            if not os.path.exists(ckpt_path):
                print(f"  ⚠️  --skip-train but no checkpoint at {ckpt_path}")
                continue

        # Step 2: Export embeddings (skip for paper_dim)
        if d != paper_dim:
            ok = export_embeddings_for_dim(d, ckpt_path, emb_path)
            if not ok:
                print(f"  ❌ Export failed for d={d}, skipping")
                continue

        # Step 3: Benchmark
        res = run_benchmark_for_dim(d, ckpt_path, emb_path, problems)
        lo, hi = wilson(res["solved"], res["total"])
        print(f"  Pass@1 = {res['pass']*100:.2f}%  "
              f"({res['solved']}/{res['total']})  "
              f"CI=[{lo*100:.1f}%, {hi*100:.1f}%]")

        results.append(res)
        existing[d] = res

        with open(summary_path, "w") as f:
            json.dump(list(existing.values()), f, indent=2)

    print_results(results, paper_dim)
    print(f"\n📂 Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()