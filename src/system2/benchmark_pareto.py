#!/usr/bin/env python3
# ==============================================================================
# Filename: src/system2/benchmark_pareto.py
# Version: v2.0
#
# Purpose: Compute-normalized comparison — Experiment D.
#   Fixes the one-shot generation parsing crash by utilizing the same REPL
#   interactive loop (RiemannSearchAgent with HLP disabled) as benchmark_standalone_v2.
#   Runs DS-Prover at N ∈ {1, 10, 50, 100} rollouts. Each rollout is a full 
#   40-step search. Records Pass@1 vs cumulative wall-clock time.
#
# Usage:
#   python src/system2/benchmark_pareto.py --rollouts 1 10 50 100 --n 100
#   python src/system2/benchmark_pareto.py --rollouts 1 10 --n 50  # pilot
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
OUTPUT_DIR = os.path.join(project_root, "results", "pareto")
MODEL_PATH = (
    os.path.join(project_root, "models/DeepSeek-Prover-V1.5-RL")
)
CKPT_PATH  = os.path.join(DATA_DIR, "hgcn_final.pth")   # needed for agent init

SEED              = 42
MAX_STEPS         = 40
HEARTBEAT_DEFAULT = 1200

os.makedirs(OUTPUT_DIR, exist_ok=True)

_SENTINEL_DONE    = "DONE"
_SENTINEL_CRASHED = "CRASHED"


def wilson(k, n, z=1.96):
    if n==0: return 0.0,0.0
    p=k/n; d=1+z**2/n
    c=(p+z**2/(2*n))/d
    m=z*math.sqrt(p*(1-p)/n+z**2/(4*n**2))/d
    return c-m, c+m


# ==============================================================================
# Problem loader
# ==============================================================================

_VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.']*$")

def load_problems(split: str, n: int, seed: int) -> list:
    folder = "Test" if split == "test" else "Valid"
    root   = os.path.join(DATA_DIR, "miniF2F", "MiniF2F", folder)
    if not os.path.exists(root):
        print(f"❌ {root} not found"); return []
    probs, seen = [], set()
    for fpath in glob.glob(os.path.join(root, "**", "*.lean"), recursive=True):
        if any(b in fpath for b in ("lake-packages", "_build", "_manual")):
            continue
        base = os.path.basename(fpath).replace(".lean", "")
        try: content = open(fpath, encoding="utf-8").read()
        except Exception: continue
        in_block = False
        lines = content.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]
            if "/-" in line: in_block = True
            if "-/" in line: in_block = False; i += 1; continue
            if in_block or line.strip().startswith("--"): i += 1; continue
            m = re.match(r"^(?:@\[.*?\]\s*)?(?:theorem|lemma)\s+([^\s\{(:\[]+)",
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
            uid  = f"{split}_{base}_{name.replace('.','_')}"
            if uid not in seen:
                seen.add(uid); probs.append({"name": uid, "decl": decl})
            i = j
    if n > 0:
        probs = random.Random(seed).sample(probs, min(n, len(probs)))
    print(f"📂 MiniF2F-{split}: {len(probs)} problems")
    return probs


# ==============================================================================
# Worker: Interactive REPL Search Agent (No HLP Baseline)
# ==============================================================================

def _worker(n_rollouts: int, temperature: float,
            problem_queue, result_queue, csv_path: str):
    import queue as _queue
    import threading

    def _send(kind):
        try: result_queue.put({"_sentinel": kind})
        except Exception: pass

    try:
        from src.system2.lie_search import RiemannSearchAgent

        agent = RiemannSearchAgent(CKPT_PATH, MODEL_PATH, device="cuda")

        # ── Disable ALL hyperbolic components (Align with No-HLP Baseline) ──
        agent.graph_emb      = None
        agent.retrieval_mode = "flat"
        agent.idx_to_name    = {}

        for attr in ("trust_threshold", "tau"):
            if hasattr(agent, attr):
                setattr(agent, attr, 0.0)

        if hasattr(agent, "lie") and hasattr(agent.lie, "matrices"):
            with torch.no_grad():
                for m in agent.lie.matrices:
                    m.zero_()
        # ────────────────────────────────────────────────────────────────────

        # Inject sampling temperature if N > 1
        if hasattr(agent, "temperature"):
            agent.temperature = temperature
        if hasattr(agent, "do_sample"):
            agent.do_sample = (temperature > 0.0 and n_rollouts > 1)

        print(f"  ✅ DS-Prover N={n_rollouts} T={temperature} ready (Iterative REPL loop)", flush=True)

    except Exception as e:
        print(f"  ❌ Init failed: {e}", flush=True)
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

        status = "Failed"
        t0     = time.time()
        
        # Limit total time for this problem based on requested rollouts
        # 300s max per search rollout
        max_time_for_task = 300 * n_rollouts

        for rollout in range(n_rollouts):
            holder = [None]
            def _run():
                try:
                    holder[0] = agent.search(task["decl"], max_steps=MAX_STEPS)
                except Exception as ex:
                    holder[0] = {"status": f"ScriptCrash:{ex}"}

            thread = threading.Thread(target=_run, daemon=True)
            thread.start()
            thread.join(timeout=300)

            if thread.is_alive():
                rollout_status = "Timeout"
            else:
                raw = (holder[0] or {}).get("status", "Unknown")
                rollout_status = "ScriptCrash" if str(raw).startswith("ScriptCrash") else raw

            # Short-circuit on success
            if rollout_status == "Success":
                status = "Success"
                break

            # Global timeout barrier for this problem
            if time.time() - t0 > max_time_for_task:
                status = "Timeout"
                break

        raw_status = "ScriptCrash" if str(status).startswith("ScriptCrash") else status
        elapsed    = time.time() - t0

        records.append({"name": task["name"], "n_rollouts": n_rollouts,
                        "status": raw_status, "time": round(elapsed, 2),
                        "cumulative_time": round(elapsed, 2)})
        pd.DataFrame(records).to_csv(csv_path, index=False)
        result_queue.put({"name": task["name"], "status": raw_status,
                          "time": elapsed})
        gc.collect()
        torch.cuda.empty_cache()

    _send(_SENTINEL_DONE)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pareto frontier benchmark (Iterative REPL)")
    parser.add_argument("--rollouts", nargs="+", type=int,
                        default=[1, 10, 50, 100])
    parser.add_argument("--split",   default="test", choices=["test","valid"])
    parser.add_argument("--n",       type=int, default=100)
    parser.add_argument("--temp",    type=float, default=0.6,
                        help="Sampling temperature for N>1 (N=1 uses greedy)")
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  Pareto Frontier (Iterative) | N∈{args.rollouts}  split={args.split}  n={args.n}")
    print(f"{'='*65}\n")

    problems = load_problems(args.split, args.n, SEED)
    if not problems: return

    all_results = []

    for n_roll in sorted(args.rollouts):
        print(f"\n{'─'*65}\n📐  N = {n_roll}")
        csv_path = os.path.join(OUTPUT_DIR,
                                f"pareto_N{n_roll}_{args.split}.csv")
        temp = 0.0 if n_roll == 1 else args.temp

        done = set()
        if os.path.exists(csv_path):
            try:
                df   = pd.read_csv(csv_path)
                # Drop severe crash files to enforce a clean run
                n_crash = (df["status"] == "ScriptCrash").sum()
                if n_crash / len(df) > 0.4 and len(df) > 5:
                    os.remove(csv_path)
                else:
                    done = set(df["name"].unique())
                    print(f"  ↩️  Resuming ({len(done)} done)")
            except Exception: pass

        todo = [p for p in problems if p["name"] not in done]

        if todo:
            try:    mp.set_start_method("spawn", force=True)
            except RuntimeError: pass

            manager = mp.Manager()
            p_queue = manager.Queue()
            r_queue = manager.Queue()
            for prob in todo: p_queue.put(prob)
            p_queue.put(None)

            worker = mp.Process(target=_worker,
                                args=(n_roll, temp, p_queue, r_queue, csv_path))
            worker.start()

            pbar          = tqdm(total=len(todo), desc=f"  N={n_roll}", leave=False)
            finished      = 0
            last_activity = time.time()
            total_time    = 0.0
            import queue as _queue

            # Dynamically scale heartbeat timeout based on N rollouts
            # Max time per task = 300s * n_roll. Buffer = 60s
            timeout_limit = max(HEARTBEAT_DEFAULT, 300 * n_roll + 60)

            while True:
                try:
                    res = r_queue.get(timeout=2.0)
                except _queue.Empty:
                    if not worker.is_alive() and r_queue.empty(): break
                    if time.time() - last_activity > timeout_limit:
                        print(f"\n  ⚠️  No activity for {timeout_limit}s, terminating worker.", flush=True)
                        worker.terminate(); break
                    continue
                last_activity = time.time()
                if isinstance(res, dict) and "_sentinel" in res: break
                finished += 1
                total_time += res.get("time", 0)
                pbar.update(1)
                if finished >= len(todo): break

            pbar.close()
            worker.join(timeout=60)
            if worker.is_alive(): worker.terminate(); worker.join()

        try:
            df       = pd.read_csv(csv_path)
            solved   = int((df["status"] == "Success").sum())
            total    = len(df)
            avg_time = df["time"].mean()
            cum_time = df["time"].sum()
        except Exception:
            solved, total, avg_time, cum_time = 0, args.n, 0.0, 0.0

        lo, hi = wilson(solved, total)
        print(f"  Pass@1 = {solved/total*100:.2f}%  "
              f"({solved}/{total})  "
              f"avg={avg_time:.0f}s/prob  total={cum_time/3600:.1f}h")

        all_results.append({
            "n_rollouts": n_roll, "solved": solved, "total": total,
            "pass": solved/total if total else 0,
            "lo": lo, "hi": hi,
            "avg_time_s": avg_time, "cum_time_s": cum_time,
        })

    # Add HLP reference point (from benchmark_standalone)
    hlp_ref = {"n_rollouts": 1, "method": "HLP (Ours)",
               "pass": 0.6575, "lo": 0.596, "hi": 0.715, "avg_time_s": 28}

    print(f"\n{'='*65}")
    print("  PARETO FRONTIER RESULTS (Iterative REPL)")
    print(f"{'='*65}")
    print(f"{'N':>6}  {'Pass@1':>8}  {'avg_time':>10}  {'95% CI':<22}")
    print("-"*55)
    for r in all_results:
        lo,hi = r['lo'],r['hi']
        print(f"{r['n_rollouts']:>6}  {r['pass']*100:>7.2f}%  "
              f"{r['avg_time_s']:>9.0f}s  [{lo*100:.1f}%, {hi*100:.1f}%]")
    print(f"  (HLP N=1: 65.75%  avg≈28s  [{hlp_ref['lo']*100:.1f}%, "
          f"{hlp_ref['hi']*100:.1f}%])")

    with open(os.path.join(OUTPUT_DIR, "pareto_summary.json"), "w") as f:
        json.dump(all_results + [hlp_ref], f, indent=2)

    print(f"\n📋 For Figure: Pass@1 vs cumulative_time points")
    print(f"   HLP:  ({hlp_ref['avg_time_s']}s,  {hlp_ref['pass']*100:.2f}%)")
    for r in all_results:
        print(f"   N={r['n_rollouts']:>3}: ({r['avg_time_s']:.0f}s, "
              f"{r['pass']*100:.2f}%)")

    print(f"\n📂 Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()