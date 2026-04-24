# ==============================================================================
# Filename: src/system2/benchmark_dim.py
# Version: v1.0
#
# Purpose: Run Pass@1 benchmark for each OUT_DIM using the dimension-specific
#          checkpoint and embeddings produced by train_hgcn_dim.py and
#          export_embeddings_dim.py.
#
# KEY DESIGN PRINCIPLE — No file swapping, no monkey-patching:
#   RiemannSearchAgent.__init__(hgcn_ckpt, ...) already takes the checkpoint
#   path as an argument. We pass hgcn_d{D}.pth directly.
#   After init, we replace agent.graph_emb with the dim-specific embeddings
#   (node_emb_d{D}.pt). This is safe because both files were produced by
#   the same HGCN with OUT_DIM=D — they are self-consistent by construction.
#   The production files (hgcn_final.pth, node_embeddings.pt) are NEVER read
#   or written.
#
# Run AFTER export_embeddings_dim.py.
#
# Usage:
#   # Pilot (50 problems per dim, ~30min per dim on A100)
#   python src/system2/benchmark_dim.py --dims 16 32 128 256 --n 50
#
#   # Full run (240 problems per dim, ~2–3h per dim)
#   python src/system2/benchmark_dim.py --dims 16 32 128 256 --n 240
#
#   # Single dim
#   python src/system2/benchmark_dim.py --dims 128 --n 240
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
SEED                    = 42
MAX_SECONDS_PER_PROBLEM = 300
HEARTBEAT_TIMEOUT       = 600  # must be > MAX_SECONDS_PER_PROBLEM=300s

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# Helpers: detect paper dim + find asset paths
# ==============================================================================

def detect_paper_dim() -> int:
    ckpt_path = os.path.join(DATA_DIR, "hgcn_final.pth")
    if not os.path.exists(ckpt_path):
        return 64
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "out_dim" in ckpt:
        return int(ckpt["out_dim"])
    for key in ("layer.semantic_proj.weight", "semantic_proj.weight"):
        if key in ckpt.get("model", {}):
            return ckpt["model"][key].shape[0]
    return 64


def asset_paths(d: int, paper_dim: int) -> tuple:
    """Return (ckpt_path, emb_path) for dimension d. Never touches originals."""
    if d == paper_dim:
        return (
            os.path.join(DATA_DIR, "hgcn_final.pth"),
            os.path.join(DATA_DIR, "node_embeddings.pt"),
        )
    return (
        os.path.join(OUTPUT_DIR, f"hgcn_d{d}.pth"),
        os.path.join(OUTPUT_DIR, f"node_emb_d{d}.pt"),
    )


# ==============================================================================
# MiniF2F loader
# ==============================================================================

_VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.']*$")


def load_minif2f_test(n: int, seed: int) -> list:
    root = os.path.join(DATA_DIR, "miniF2F", "MiniF2F", "Test")
    if not os.path.exists(root):
        print(f"❌ MiniF2F Test not found: {root}"); return []

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
            if not _VALID_NAME.match(name): i += 1; continue
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
                seen.add(uid); probs.append({"name": uid, "decl": decl})
            i = j

    if n > 0:
        probs = random.Random(seed).sample(probs, min(n, len(probs)))
    print(f"📂 MiniF2F-test: {len(probs)} problems")
    return probs


# ==============================================================================
# Subprocess worker — loads dim-specific assets directly
# ==============================================================================

_SENTINEL_DONE    = "DONE"
_SENTINEL_CRASHED = "CRASHED"


def _worker(dim: int, emb_path: str,
            problem_queue, result_queue, csv_path: str):
    """
    Correct approach for dimension ablation:

    The dimension d affects RETRIEVAL (goal_encoder + graph_emb) but NOT
    the Lie dynamics or LLM. We therefore:
      1. Load agent with the PAPER checkpoint (hgcn_final.pth, d=64, trained)
         → Lie dynamics, LLM, tac_to_coeff all use trained d=64 weights
      2. Patch ONLY the query encoder projector with hgcn_d{D}.pth weights
         → query encoding uses d-specific HGCN projection
      3. Replace agent.graph_emb with node_emb_d{D}.pt
         → retrieval database uses d-specific embeddings

    Both query and database use the same d-specific projection, so
    hyperbolic distances are valid. Lie dynamics use trained d=64 weights
    so there are no random/untrained parameters causing crashes.

    This cleanly measures: does d-specific retrieval quality affect Pass@1?
    """
    import queue as _queue
    import threading

    DATA_DIR_W = os.path.join(project_root, "data")
    paper_ckpt = os.path.join(DATA_DIR_W, "hgcn_final.pth")

    def _send(kind):
        try: result_queue.put({"_sentinel": kind})
        except Exception: pass

    try:
        from src.system2.lie_search import RiemannSearchAgent

        # Step 1: load agent with PAPER checkpoint (trained Lie dynamics)
        agent = RiemannSearchAgent(paper_ckpt, MODEL_PATH, device="cuda")
        paper_d = agent.graph_emb.shape[1] if agent.graph_emb is not None else 64

        if dim == paper_d:
            # Paper dim: no patching needed
            print(f"  ✅ d={dim} (paper): using agent as-is", flush=True)
        else:
            # Step 2: patch query encoder with d-specific HGCN projector
            dim_ckpt = os.path.join(OUTPUT_DIR, f"hgcn_d{dim}.pth")
            if not os.path.exists(dim_ckpt):
                print(f"  ❌ Missing checkpoint: {dim_ckpt}", flush=True)
                _send(_SENTINEL_CRASHED); return

            ckpt_d = torch.load(dim_ckpt, map_location="cuda")
            # Extract projector weights from the dim-specific checkpoint
            model_d = ckpt_d["model"]

            # Cannot use load_state_dict — the existing Linear has shape [64, 384]
            # and we need [dim, 384]. load_state_dict raises RuntimeError on size
            # mismatch even with strict=False.
            # Fix: REPLACE the Linear layers entirely, then copy weights directly.
            import torch.nn as nn

            proj = agent.goal_encoder.projector

            # Replace semantic_proj
            w_key = "layer.semantic_proj.weight"
            b_key = "layer.semantic_proj.bias"
            if w_key in model_d:
                new_linear = nn.Linear(384, dim, bias=(b_key in model_d)).to("cuda")
                new_linear.weight.data.copy_(model_d[w_key])
                if b_key in model_d:
                    new_linear.bias.data.copy_(model_d[b_key])
                proj.semantic_proj = new_linear

            # Replace structure_proj if present
            ws_key = "layer.structure_proj.weight"
            bs_key = "layer.structure_proj.bias"
            if ws_key in model_d and hasattr(proj, "structure_proj"):
                new_s = nn.Linear(384, dim, bias=(bs_key in model_d)).to("cuda")
                new_s.weight.data.copy_(model_d[ws_key])
                if bs_key in model_d:
                    new_s.bias.data.copy_(model_d[bs_key])
                proj.structure_proj = new_s

            # Replace gate (input: dim*2 → 1)
            wg_key = "layer.gate.weight"
            bg_key = "layer.gate.bias"
            if wg_key in model_d and hasattr(proj, "gate"):
                new_g = nn.Linear(dim * 2, 1, bias=(bg_key in model_d)).to("cuda")
                new_g.weight.data.copy_(model_d[wg_key])
                if bg_key in model_d:
                    new_g.bias.data.copy_(model_d[bg_key])
                proj.gate = new_g

            # Copy scale parameter
            if "layer.scale" in model_d and hasattr(proj, "scale"):
                proj.scale.data.copy_(model_d["layer.scale"])

            proj.eval()

            # Step 3: replace graph_emb with d-specific embeddings
            if not os.path.exists(emb_path):
                print(f"  ❌ Missing embeddings: {emb_path}", flush=True)
                print(f"     Run: python src/system1/export_embeddings_dim.py "
                      f"--dims {dim}", flush=True)
                _send(_SENTINEL_CRASHED); return

            agent.graph_emb = torch.load(emb_path, map_location="cuda")
            agent.retrieval_mode = "hyperbolic"

            # Verify shapes are consistent
            q_dim = agent.goal_encoder.projector.semantic_proj.weight.shape[0]
            g_dim = agent.graph_emb.shape[1]
            if q_dim != g_dim:
                print(f"  ❌ Dim mismatch: query={q_dim}, graph={g_dim}", flush=True)
                _send(_SENTINEL_CRASHED); return

            print(f"  ✅ d={dim}: query_encoder→{q_dim}d, "
                  f"graph_emb{agent.graph_emb.shape}  "
                  f"(Lie dynamics kept at d={paper_d})", flush=True)

    except Exception as e:
        print(f"  ❌ Worker init failed for d={dim}: {e}", flush=True)
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
            r   = holder[0] or {}
            raw = r.get("status", "Unknown")
            status = "ScriptCrash" if str(raw).startswith("ScriptCrash") else raw

        records.append({"name": task["name"], "dim": dim,
                        "status": status, "time": round(time.time() - t0, 2)})
        pd.DataFrame(records).to_csv(csv_path, index=False)
        result_queue.put({"name": task["name"], "status": status})
        gc.collect()
        torch.cuda.empty_cache()

    _send(_SENTINEL_DONE)


# ==============================================================================
# Per-dim benchmark runner
# ==============================================================================

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k/n; d = 1+z**2/n
    c = (p+z**2/(2*n))/d
    m = z*math.sqrt(p*(1-p)/n + z**2/(4*n**2))/d
    return c-m, c+m


def run_one_dim(d: int, ckpt_path: str, emb_path: str,
                problems: list) -> dict:
    import queue as _queue

    csv_path = os.path.join(OUTPUT_DIR, f"dim_{d}.csv")

    # Detect and delete stale CSVs from old broken experiments
    done = set()
    if os.path.exists(csv_path):
        try:
            df         = pd.read_csv(csv_path)
            n_crash    = (df["status"] == "ScriptCrash").sum()
            crash_rate = n_crash / len(df) if len(df) > 0 else 0
            if crash_rate > 0.40:
                print(f"  🗑️  Deleting stale CSV d={d} "
                      f"({crash_rate*100:.0f}% ScriptCrash — old broken run)")
                os.remove(csv_path)
            else:
                done = set(df["name"].unique())
                print(f"  ↩️  Resuming d={d} ({len(done)} done, "
                      f"{crash_rate*100:.0f}% crash rate)")
        except Exception:
            pass

    todo = [p for p in problems if p["name"] not in done]
    if not todo:
        df     = pd.read_csv(csv_path)
        solved = int((df["status"] == "Success").sum())
        return {"dim": d, "solved": solved, "total": len(df),
                "pass": solved / len(df) if len(df) else 0.0}

    try:    mp.set_start_method("spawn", force=True)
    except RuntimeError: pass

    manager = mp.Manager()
    p_queue = manager.Queue()
    r_queue = manager.Queue()
    for prob in todo: p_queue.put(prob)
    p_queue.put(None)

    # Pass dim and emb_path — worker loads paper checkpoint itself
    # and patches only the retrieval components
    worker = mp.Process(target=_worker,
                        args=(d, emb_path, p_queue, r_queue, csv_path))
    worker.start()

    pbar          = tqdm(total=len(todo), desc=f"  d={d}", leave=False)
    finished      = 0
    last_activity = time.time()

    while True:
        try:
            res = r_queue.get(timeout=2.0)
        except _queue.Empty:
            silence = time.time() - last_activity
            if not worker.is_alive() and r_queue.empty():
                print(f"\n  ⚠️  Worker exited with {len(todo)-finished} unfinished",
                      flush=True)
                break
            if silence > HEARTBEAT_TIMEOUT:
                print(f"\n  ⚠️  No activity {HEARTBEAT_TIMEOUT}s — terminating",
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
                except _queue.Empty: break
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

    return {"dim": d, "solved": solved, "total": total,
            "pass": solved / total if total > 0 else 0.0}


# ==============================================================================
# Main
# ==============================================================================

def main():
    paper_dim = detect_paper_dim()

    parser = argparse.ArgumentParser(
        description="Benchmark Pass@1 for each dim (run after export_embeddings_dim.py)")
    parser.add_argument("--dims", nargs="+", type=int, required=True,
                        help="OUT_DIM values to benchmark")
    parser.add_argument("--n",    type=int, default=50,
                        help="MiniF2F-test problems per dim (50=pilot, 240=full)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  Benchmark Dim  |  dims={sorted(args.dims)}  n={args.n}")
    print(f"  Paper dim={paper_dim}  (hgcn_final.pth — never overwritten)")
    print(f"{'='*65}\n")

    problems = load_minif2f_test(args.n, args.seed)
    if not problems: return

    # Load cached summary
    summary_path = os.path.join(OUTPUT_DIR, "dimension_scaling_summary.json")
    existing = {}
    if os.path.exists(summary_path):
        try:
            for r in json.load(open(summary_path)):
                existing[r["dim"]] = r
        except Exception: pass

    results = []
    for d in sorted(args.dims):
        print(f"\n{'─'*65}")
        print(f"📐  d = {d}")

        # Check cache — but reject stale CSVs with >40% ScriptCrash
        # (leftover from the v1/v2 monkey-patch experiments)
        if d in existing and existing[d].get("total", 0) >= args.n:
            csv_path_chk = os.path.join(OUTPUT_DIR, f"dim_{d}.csv")
            stale = False
            if os.path.exists(csv_path_chk):
                df_chk     = pd.read_csv(csv_path_chk)
                n_crash    = (df_chk["status"] == "ScriptCrash").sum()
                crash_rate = n_crash / len(df_chk) if len(df_chk) else 0
                if crash_rate > 0.40:
                    print(f"  ⚠️  Rejecting cache d={d}: "
                          f"{crash_rate*100:.0f}% ScriptCrash — stale experiment")
                    del existing[d]
                    stale = True
            if not stale:
                res = existing[d]
                lo, hi = wilson(res["solved"], res["total"])
                print(f"  ↩️  Cached: {res['pass']*100:.2f}%  "
                      f"CI=[{lo*100:.1f}%, {hi*100:.1f}%]")
                results.append(res); continue

        ckpt_path, emb_path = asset_paths(d, paper_dim)

        # Validate assets exist
        if not os.path.exists(ckpt_path):
            print(f"  ❌ Missing checkpoint: {ckpt_path}")
            print(f"     Run: python src/system1/train_hgcn_dim.py --dims {d}")
            continue
        if not os.path.exists(emb_path):
            print(f"  ❌ Missing embeddings: {emb_path}")
            print(f"     Run: python src/system1/export_embeddings_dim.py --dims {d}")
            continue

        res = run_one_dim(d, ckpt_path, emb_path, problems)
        lo, hi = wilson(res["solved"], res["total"])
        print(f"  Pass@1 = {res['pass']*100:.2f}%  "
              f"({res['solved']}/{res['total']})  "
              f"CI=[{lo*100:.1f}%, {hi*100:.1f}%]")

        results.append(res)
        existing[d] = res
        with open(summary_path, "w") as f:
            json.dump(list(existing.values()), f, indent=2)

    # Final summary table
    if results:
        sorted_r = sorted(results, key=lambda r: r["dim"])
        print(f"\n{'='*65}")
        print("  DIMENSION SCALING RESULTS")
        print(f"{'='*65}")
        print(f"{'d':>6}  {'Pass@1':>8}  {'n':>8}  {'95% CI':<22}  note")
        print("-"*65)
        for r in sorted_r:
            lo, hi = wilson(r["solved"], r["total"])
            note   = "← paper" if r["dim"] == paper_dim else ""
            nflag  = "⚠️ " if r["total"] < 200 else ""
            print(f"{r['dim']:>6}  {r['pass']*100:>7.2f}%  "
                  f"{r['total']:>8}  [{lo*100:.1f}%, {hi*100:.1f}%]  "
                  f"{nflag}{note}")

        print(f"\n📋 LaTeX:")
        print("\\textbf{{Dimension $d$}} & "
              + " & ".join(str(r["dim"]) for r in sorted_r) + " \\\\")
        print("\\textbf{{Pass@1 (\\%)}} & "
              + " & ".join(f"{r['pass']*100:.2f}" for r in sorted_r) + " \\\\")

    print(f"\n📂 Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()