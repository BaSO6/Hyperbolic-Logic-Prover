# ==============================================================================
# Filename: src/system2/benchmark_curvature.py
# Version: v2.0
#
# Purpose: Benchmark Pass@1 for each HGCN curvature variant.
#          ONLY benchmarking — no training, no embedding export.
#          Run AFTER src/system1/train_hgcn_curvature.py.
#
# Critical fix vs v1.0 (all-in-one script in system2):
#   v1 had a curvature consistency bug: the agent's manifold, energy, and
#   Lie algebra all used c=1.0 even when the projector and embeddings used
#   c=0.5 or c=2.0. This made retrieval distances wrong for c≠1.0.
#
#   v2 fix: after patching the projector, also update agent.manifold,
#   agent.energy, and agent.c with the correct curvature value.
#   The Lie dynamics stay at c=1.0 (trained jointly) but ALL retrieval
#   components now use the correct curvature consistently.
#
# Usage:
#   # MUST run system1 first:
#   python src/system1/train_hgcn_curvature.py --curvatures 0.5 1.0 2.0
#
#   # Then benchmark:
#   python src/system2/benchmark_curvature.py --curvatures 0.5 1.0 2.0 --n 240
#   python src/system2/benchmark_curvature.py --curvatures 0.5 2.0 --n 50  # pilot
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

import torch
import torch.nn as nn
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
OUTPUT_DIR = os.path.join(project_root, "results", "curvature_sensitivity")
MODEL_PATH = (
    os.path.join(project_root, "models/DeepSeek-Prover-V1.5-RL")
)
SEED                    = 42
MAX_SECONDS_PER_PROBLEM = 300
HEARTBEAT_TIMEOUT       = 600

_SENTINEL_DONE    = "DONE"
_SENTINEL_CRASHED = "CRASHED"


# ==============================================================================
# Asset path helpers — must match train_hgcn_curvature.py
# ==============================================================================

def c_str(c: float) -> str:
    return f"{c:.1f}".replace(".", "p")


def asset_paths(c: float) -> tuple:
    """Return (ckpt_path, emb_path). Never touches production files."""
    if abs(c - 1.0) < 1e-6:
        return (
            os.path.join(DATA_DIR, "hgcn_final.pth"),
            os.path.join(DATA_DIR, "node_embeddings.pt"),
        )
    return (
        os.path.join(OUTPUT_DIR, f"hgcn_c{c_str(c)}.pth"),
        os.path.join(OUTPUT_DIR, f"node_emb_c{c_str(c)}.pt"),
    )


def validate_assets(c: float) -> bool:
    """Check that training artifacts exist and have correct curvature stored."""
    cp, ep = asset_paths(c)
    if not os.path.exists(cp):
        print(f"  ❌ Missing checkpoint: {cp}")
        print(f"     Run: python src/system1/train_hgcn_curvature.py --curvatures {c}")
        return False
    if not os.path.exists(ep):
        print(f"  ❌ Missing embeddings: {ep}")
        print(f"     Run: python src/system1/train_hgcn_curvature.py --curvatures {c}")
        return False

    # Verify stored curvature matches requested
    ckpt = torch.load(cp, map_location="cpu")
    stored_c = ckpt.get("c", None)
    if stored_c is None:
        print(f"  ⚠️  c={c}: checkpoint has no 'c' field stored")
    elif abs(float(stored_c) - c) > 0.01:
        print(f"  ❌ c={c}: checkpoint stores c={stored_c} — mismatch")
        return False

    # Verify embedding dimension
    emb = torch.load(ep, map_location="cpu")
    if emb.shape[0] < 100_000:
        print(f"  ❌ c={c}: embeddings only {emb.shape[0]} nodes (expected ~110k)")
        return False

    print(f"  ✅ c={c}: assets validated  ckpt_c={stored_c}  emb{emb.shape}")
    return True


# ==============================================================================
# Problem loader
# ==============================================================================

_VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.']*$")


def load_minif2f_valid(n: int, seed: int) -> list:
    root = os.path.join(DATA_DIR, "miniF2F", "MiniF2F", "Valid")
    if not os.path.exists(root):
        print(f"❌ MiniF2F Valid not found: {root}"); return []
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
            if "/-" in line:   in_block = True
            if "-/" in line:   in_block = False; i += 1; continue
            if in_block or line.strip().startswith("--"):
                i += 1; continue
            m = re.match(
                r"^(?:@\[.*?\]\s*)?(?:theorem|lemma)\s+([^\s\{(:\[]+)",
                line.strip())
            if not m: i += 1; continue
            name = m.group(1).strip()
            if not _VALID_NAME.match(name): i += 1; continue
            decl_lines = [line]
            j = i + 1
            while j < len(lines) and ":=" not in "".join(decl_lines):
                decl_lines.append(lines[j]); j += 1
            decl_raw = " ".join(l.strip() for l in decl_lines)
            if ":=" in decl_raw:
                decl_raw = decl_raw[:decl_raw.index(":=")].strip()
            decl = f"theorem {name} {decl_raw.split(name,1)[-1].strip()}"
            uid  = f"valid_{base}_{name.replace('.','_')}"
            if uid not in seen:
                seen.add(uid); probs.append({"name": uid, "decl": decl})
            i = j
    if n > 0:
        probs = random.Random(seed).sample(probs, min(n, len(probs)))
    print(f"📂 MiniF2F-valid: {len(probs)} problems")
    return probs


# ==============================================================================
# Worker: patch retrieval AND manifold curvature consistently
# ==============================================================================

def _worker(c: float, ckpt_path_c: str, emb_path_c: str,
            problem_queue, result_queue, csv_path: str):
    """
    Loads agent with paper checkpoint (trained Lie dynamics at c=1.0).
    Then patches ALL retrieval components to use curvature c:
      - Projector weights (semantic_proj, scale only — InferenceHGCN)
      - graph_emb (node embeddings trained at c)
      - agent.manifold → PoincareBall(c)   ← FIX: was missing in v1
      - agent.energy   → HyperbolicEnergy(c) ← FIX: was missing in v1
      - agent.c        → c                   ← FIX: was missing in v1
    Lie algebra stays at c=1.0 (jointly trained).
    """
    import queue as _queue
    import threading

    paper_ckpt = os.path.join(DATA_DIR, "hgcn_final.pth")

    def _send(kind):
        try: result_queue.put({"_sentinel": kind})
        except Exception: pass

    try:
        from src.system2.lie_search import RiemannSearchAgent  # type: ignore
        from src.system1.manifold_math import PoincareBall     # type: ignore

        # Load paper agent (trained Lie dynamics at c=1.0)
        agent = RiemannSearchAgent(paper_ckpt, MODEL_PATH, device="cuda")

        if abs(c - 1.0) > 1e-6:
            # ── Patch projector weights ─────────────────────────────────────
            # InferenceHGCN (lie_search.py) only has: semantic_proj, scale.
            # InferenceHGCN only has semantic_proj + scale (not gate/).
            # the training model (FinalHGCN). Only patch what actually exists.
            ckpt_c  = torch.load(ckpt_path_c, map_location="cuda")
            model_c = ckpt_c["model"]
            proj    = agent.goal_encoder.projector   # InferenceHGCN instance

            # Patch semantic_proj (always present in InferenceHGCN)
            for w_key, b_key in [
                ("layer.semantic_proj.weight", "layer.semantic_proj.bias"),
                ("semantic_proj.weight",        "semantic_proj.bias"),      # fallback
            ]:
                if w_key in model_c:
                    out_d, in_d = model_c[w_key].shape
                    new_l = nn.Linear(in_d, out_d, bias=(b_key in model_c)).to("cuda")
                    new_l.weight.data.copy_(model_c[w_key])
                    if b_key in model_c:
                        new_l.bias.data.copy_(model_c[b_key])
                    proj.semantic_proj = new_l
                    break   # use first matching key

            # Patch scale (always present in InferenceHGCN)
            for scale_key in ("layer.scale", "scale"):
                if scale_key in model_c:
                    proj.scale.data.copy_(model_c[scale_key])
                    break

            proj.eval()

            # ── Patch graph embeddings ──────────────────────────────────────
            agent.graph_emb = torch.load(emb_path_c, map_location="cuda")
            agent.retrieval_mode = "hyperbolic"

            # ── FIX: Update manifold curvature for distance computation ─────
            # Without this, agent.manifold.dist() uses c=1.0 for embeddings
            # that were produced with c=0.5 or c=2.0 → wrong distances.
            agent.manifold = PoincareBall(c)
            agent.c        = c

            # Update GoalEncoder's manifold too (used for query projection)
            if hasattr(agent.goal_encoder, "manifold"):
                agent.goal_encoder.manifold = PoincareBall(c)
            if hasattr(agent.goal_encoder, "c"):
                agent.goal_encoder.c = c

            # Update projector's manifold
            if hasattr(proj, "manifold"):
                proj.manifold = PoincareBall(c)

            # Update energy function if it exists
            if hasattr(agent, "energy") and hasattr(agent.energy, "c"):
                agent.energy.c = c

            # Verify shape consistency
            q_dim = proj.semantic_proj.weight.shape[0]
            g_dim = agent.graph_emb.shape[1]
            if q_dim != g_dim:
                raise RuntimeError(
                    f"Dim mismatch after patching: query={q_dim}, graph={g_dim}")

        print(f"  ✅ c={c}: agent ready  "
              f"manifold.c={agent.manifold.c if hasattr(agent.manifold,'c') else '?'}  "
              f"graph_emb{agent.graph_emb.shape}", flush=True)

    except Exception as e:
        print(f"  ❌ Worker init failed c={c}: {e}", flush=True)
        import traceback; traceback.print_exc()
        _send(_SENTINEL_CRASHED); return

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

        if thread.is_alive():
            status = "Timeout"
        else:
            raw    = (holder[0] or {}).get("status", "Unknown")
            status = "ScriptCrash" if str(raw).startswith("ScriptCrash") else raw

        records.append({"name": task["name"], "curvature": c,
                        "status": status, "time": round(time.time() - t0, 2)})
        pd.DataFrame(records).to_csv(csv_path, index=False)
        result_queue.put({"name": task["name"], "status": status})
        gc.collect()
        torch.cuda.empty_cache()

    _send(_SENTINEL_DONE)


# ==============================================================================
# Per-curvature runner
# ==============================================================================

def wilson(k, n, z=1.96):
    if n==0: return 0.0,0.0
    p=k/n; d=1+z**2/n
    c_=(p+z**2/(2*n))/d
    m=z*math.sqrt(p*(1-p)/n+z**2/(4*n**2))/d
    return c_-m, c_+m


def run_one_curvature(c: float, ckpt_path_c: str, emb_path_c: str,
                      problems: list) -> dict:
    import queue as _queue
    csv_path = os.path.join(OUTPUT_DIR, f"curvature_c{c_str(c)}.csv")

    done = set()
    if os.path.exists(csv_path):
        try:
            df         = pd.read_csv(csv_path)
            n_crash    = (df["status"] == "ScriptCrash").sum()
            crash_rate = n_crash / len(df) if len(df) else 0
            if crash_rate > 0.40:
                print(f"  🗑️  Stale CSV (crash={crash_rate*100:.0f}%) — deleting")
                os.remove(csv_path)
            else:
                done = set(df["name"].unique())
                print(f"  ↩️  Resuming c={c} ({len(done)} done, "
                      f"crash={crash_rate*100:.0f}%)")
        except Exception: pass

    todo = [p for p in problems if p["name"] not in done]
    if not todo:
        df     = pd.read_csv(csv_path)
        solved = int((df["status"] == "Success").sum())
        return {"c": c, "solved": solved, "total": len(df),
                "pass": solved/len(df) if len(df) else 0.0}

    try:    mp.set_start_method("spawn", force=True)
    except RuntimeError: pass

    manager = mp.Manager()
    p_queue = manager.Queue()
    r_queue = manager.Queue()
    for prob in todo: p_queue.put(prob)
    p_queue.put(None)

    worker = mp.Process(target=_worker,
                        args=(c, ckpt_path_c, emb_path_c,
                              p_queue, r_queue, csv_path))
    worker.start()

    pbar          = tqdm(total=len(todo), desc=f"  c={c}", leave=False)
    finished      = 0
    last_activity = time.time()

    while True:
        try:
            res = r_queue.get(timeout=2.0)
        except _queue.Empty:
            if not worker.is_alive() and r_queue.empty(): break
            if time.time() - last_activity > HEARTBEAT_TIMEOUT:
                print(f"\n  ⚠️  No activity {HEARTBEAT_TIMEOUT}s", flush=True)
                worker.terminate(); break
            continue
        last_activity = time.time()
        if isinstance(res, dict) and "_sentinel" in res: break
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

    return {"c": c, "solved": solved, "total": total,
            "pass": solved/total if total > 0 else 0.0}


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Curvature sensitivity benchmark (run after train_hgcn_curvature.py)")
    parser.add_argument("--curvatures", nargs="+", type=float,
                        default=[0.5, 1.0, 2.0])
    parser.add_argument("--n",    type=int, default=240)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  Curvature Sensitivity Benchmark  |  c∈{sorted(args.curvatures)}  n={args.n}")
    print(f"  NOTE: Run train_hgcn_curvature.py first if assets are missing")
    print(f"{'='*65}\n")

    # Validate all assets before starting any benchmark
    print("Validating assets...")
    for c in sorted(args.curvatures):
        if not validate_assets(c):
            print(f"\n❌ Asset validation failed for c={c}. Aborting.")
            print(f"   Run: python src/system1/train_hgcn_curvature.py "
                  f"--curvatures {' '.join(str(v) for v in args.curvatures)}")
            return
    print("All assets valid.\n")

    problems = load_minif2f_valid(args.n, args.seed)
    if not problems: return

    summary_path = os.path.join(OUTPUT_DIR, "curvature_summary.json")
    existing     = {}
    if os.path.exists(summary_path):
        try:
            for r in json.load(open(summary_path)):
                existing[r["c"]] = r
        except Exception: pass

    results = []
    for c in sorted(args.curvatures):
        print(f"\n{'─'*65}\n📐  c = {c}")

        # Check cache
        if c in existing and existing[c].get("total", 0) >= args.n:
            res    = existing[c]
            lo, hi = wilson(res["solved"], res["total"])
            print(f"  ↩️  Cached: {res['pass']*100:.2f}%  "
                  f"CI=[{lo*100:.1f}%, {hi*100:.1f}%]")
            results.append(res); continue

        cp, ep = asset_paths(c)
        res    = run_one_curvature(c, cp, ep, problems)
        lo, hi = wilson(res["solved"], res["total"])
        print(f"  Pass@1 = {res['pass']*100:.2f}%  "
              f"({res['solved']}/{res['total']})  "
              f"CI=[{lo*100:.1f}%, {hi*100:.1f}%]")

        results.append(res)
        existing[c] = res
        with open(summary_path, "w") as f:
            json.dump(list(existing.values()), f, indent=2)

    # Final table
    print(f"\n{'='*65}")
    print("  CURVATURE SENSITIVITY RESULTS")
    print(f"{'='*65}")
    print(f"{'c':>6}  {'Pass@1':>8}  {'n':>6}  {'95% CI':<22}  note")
    print("-"*55)
    for r in sorted(results, key=lambda x: x["c"]):
        lo, hi = wilson(r["solved"], r["total"])
        note   = "← paper" if abs(r["c"] - 1.0) < 1e-6 else ""
        print(f"{r['c']:>6.1f}  {r['pass']*100:>7.2f}%  {r['total']:>6}  "
              f"[{lo*100:.1f}%, {hi*100:.1f}%]  {note}")

    print(f"\n📋 LaTeX (3-row table for appendix):")
    print(r"\begin{tabular}{ccc}")
    print(r"\toprule")
    print(r"Curvature $c$ & Pass@1 (\%) & 95\% CI \\ \midrule")
    for r in sorted(results, key=lambda x: x["c"]):
        lo, hi = wilson(r["solved"], r["total"])
        note   = r" & \textbf{" if abs(r["c"]-1.0) < 1e-6 else " & "
        close  = r"} (paper)" if abs(r["c"]-1.0) < 1e-6 else ""
        print(f"{r['c']:.1f}{note}{r['pass']*100:.2f}{close}"
              f" & [{lo*100:.1f}, {hi*100:.1f}] \\\\")
    print(r"\bottomrule" + "\n" + r"\end{tabular}")

    print(f"\n📂 Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()