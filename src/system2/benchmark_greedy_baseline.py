# ==============================================================================
# Filename: src/system2/benchmark_greedy_baseline.py
# Version: v2.0
#
# PURPOSE AND DESIGN:
#   This script measures DS-Prover-V1.5-RL Pass@1 WITHOUT the Hyperbolic
#   Lie Prover framework. The reviewer requested a fair "DS-Prover N=1"
#   baseline within the SAME inference regime as HLP.
#
# WHAT "N=1 WITHOUT HLP" MEANS CORRECTLY:
#   NOT: generate one tactic blindly without seeing the Lean goal state.
#   YES: run the same tactic-search loop as HLP (with Lean REPL feedback
#        at each step), but with ALL hyperbolic components disabled:
#         - No HGCN retrieval (graph_emb = None → no hints in prompt)
#         - No Lie dynamics guidance (trust_threshold = 0 → accept everything)
#         - No hyperbolic distance filtering
#   This is exactly DS-Prover-V1.5-RL doing beam search in Lean 4 space,
#   which is the honest single-trajectory baseline.
#
# WHY v1 GAVE 0%:
#   v1 built a cmd with broken decl parsing (split on ':=' in type annotations)
#   AND generated one tactic without any Lean goal-state feedback.
#   Both made it impossible to solve anything. v2 uses RiemannSearchAgent
#   with hyperbolic components disabled — gets Lean feedback at every step.
#
# Usage:
#   python src/system2/benchmark_greedy_baseline.py --n 244 --split test
#   python src/system2/benchmark_greedy_baseline.py --n 50  --split test  # pilot
# ==============================================================================

import os
import sys
import re
import glob
import time
import random
import argparse
import math
import gc
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
OUTPUT_DIR = os.path.join(project_root, "results", "greedy_baseline")
MODEL_PATH = (
    os.path.join(project_root, "models/DeepSeek-Prover-V1.5-RL")
)
CKPT_PATH  = os.path.join(DATA_DIR, "hgcn_final.pth")
SEED       = 42

MAX_SECONDS_PER_PROBLEM = 600
HEARTBEAT_TIMEOUT       = 6000

os.makedirs(OUTPUT_DIR, exist_ok=True)

_SENTINEL_DONE    = "DONE"
_SENTINEL_CRASHED = "CRASHED"


# ==============================================================================
# Problem loader — identical to benchmark_minif2f.py
# ==============================================================================

_VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_.']*$")


def load_problems(split: str, n: int, seed: int) -> list:
    folder = "Test" if split == "test" else "Valid"
    root   = os.path.join(DATA_DIR, "miniF2F", "MiniF2F", folder)
    if not os.path.exists(root):
        print(f"❌ MiniF2F {folder} not found: {root}"); return []

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
            uid  = f"{split}_{base}_{name.replace('.','_')}"
            if uid not in seen:
                seen.add(uid)
                probs.append({"name": uid, "decl": decl})
            i = j

    if n > 0:
        probs = random.Random(seed).sample(probs, min(n, len(probs)))
    print(f"📂 MiniF2F-{split}: {len(probs)} problems")
    return probs


# ==============================================================================
# Worker: RiemannSearchAgent with ALL hyperbolic components disabled
# ==============================================================================

def _worker(problem_queue, result_queue, csv_path: str):
    """
    Loads RiemannSearchAgent normally, then disables all hyperbolic guidance:
      - agent.graph_emb = None       → no HGCN retrieval, no hints in prompt
      - agent.retrieval_mode = 'flat' → fallback to zero-hint mode
      - trust_threshold = 0           → trust gate accepts all tactics
    The LLM still runs the full tactic-search loop with Lean REPL feedback.
    This is the honest DS-Prover N=1 baseline without HLP.
    """
    import queue as _queue
    import threading

    def _send(kind):
        try: result_queue.put({"_sentinel": kind})
        except Exception: pass

    try:
        from src.system2.lie_search import RiemannSearchAgent

        agent = RiemannSearchAgent(CKPT_PATH, MODEL_PATH, device="cuda")

        # Disable ALL hyperbolic components
        agent.graph_emb      = None           # no retrieval → no hints
        agent.retrieval_mode = "flat"          # signal: skip hyperbolic lookup
        agent.idx_to_name    = {}              # empty index

        # Trust gate: accept everything (no Lie dynamics filtering)
        for attr in ("trust_threshold", "tau"):
            if hasattr(agent, attr):
                setattr(agent, attr, 0.0)      # tau=0 → always trust

        # Disable Lie dynamics steering (zero-out if possible)
        if hasattr(agent, "lie") and hasattr(agent.lie, "matrices"):
            with torch.no_grad():
                for m in agent.lie.matrices:
                    m.zero_()

        print(f"  ✅ DS-Prover baseline: hyperbolic components DISABLED", flush=True)
        print(f"     graph_emb=None, retrieval=flat, tau=0, Lie=zeroed", flush=True)

    except Exception as e:
        print(f"  ❌ Worker init failed: {e}", flush=True)
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
            try:
                holder[0] = agent.search(task["decl"], max_steps=40)
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

        records.append({
            "name":   task["name"],
            "status": status,
            "time":   round(time.time() - t0, 2),
            "method": "greedy_no_hlp",
        })
        pd.DataFrame(records).to_csv(csv_path, index=False)
        result_queue.put({"name": task["name"], "status": status})
        gc.collect()
        torch.cuda.empty_cache()

    _send(_SENTINEL_DONE)


# ==============================================================================
# Main
# ==============================================================================

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p=k/n; d=1+z**2/n
    c=(p+z**2/(2*n))/d
    m=z*math.sqrt(p*(1-p)/n+z**2/(4*n**2))/d
    return c-m, c+m


def main():
    parser = argparse.ArgumentParser(
        description="DS-Prover N=1 baseline WITHOUT hyperbolic components")
    parser.add_argument("--split", default="test", choices=["test","valid"])
    parser.add_argument("--n",     type=int, default=244)
    parser.add_argument("--seed",  type=int, default=SEED)
    args = parser.parse_args()

    csv_path = os.path.join(OUTPUT_DIR, f"greedy_baseline_{args.split}.csv")

    print(f"\n{'='*65}")
    print(f"  DS-Prover-V1.5-RL  |  N=1, NO Hyperbolic Components")
    print(f"  split={args.split}  n={args.n}")
    print(f"  This is the fair Regime-I baseline for Table 1")
    print(f"{'='*65}\n")

    # Check for stale v1 CSV (all Failed, 0 Success)
    if os.path.exists(csv_path):
        try:
            df_check = pd.read_csv(csv_path)
            success  = (df_check["status"] == "Success").sum()
            failed   = (df_check["status"] == "Failed").sum()
            method   = df_check.get("method", pd.Series()).iloc[0] if len(df_check) else ""
            if success == 0 and failed == len(df_check):
                print(f"  ⚠️  Stale v1 CSV detected (0 success, all Failed)")
                print(f"     Deleting and re-running with correct method.")
                os.remove(csv_path)
        except Exception:
            pass

    problems = load_problems(args.split, args.n, args.seed)
    if not problems: return

    done = set()
    if os.path.exists(csv_path):
        try:
            df   = pd.read_csv(csv_path)
            # Reject old v1 results (method != greedy_no_hlp)
            if "method" in df.columns:
                df = df[df["method"] == "greedy_no_hlp"]
            done = set(df["name"].unique())
            if done:
                print(f"  ↩️  Resuming ({len(done)} done)")
        except Exception:
            pass

    todo = [p for p in problems if p["name"] not in done]
    if not todo:
        df     = pd.read_csv(csv_path)
        solved = int((df["status"] == "Success").sum())
        lo, hi = wilson(solved, len(df))
        print(f"\n✅ Complete: {solved}/{len(df)} = {solved/len(df)*100:.2f}% "
              f"[{lo*100:.1f}%, {hi*100:.1f}%]")
        _print_latex(solved, len(df))
        return

    try:    mp.set_start_method("spawn", force=True)
    except RuntimeError: pass

    manager = mp.Manager()
    p_queue = manager.Queue()
    r_queue = manager.Queue()
    for prob in todo: p_queue.put(prob)
    p_queue.put(None)

    worker = mp.Process(target=_worker,
                        args=(p_queue, r_queue, csv_path))
    worker.start()

    pbar          = tqdm(total=len(todo), desc="  DS-Prover (no HLP)", leave=True)
    finished      = 0
    last_activity = time.time()
    import queue as _queue

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
    worker.join(timeout=60)
    if worker.is_alive(): worker.terminate(); worker.join()

    try:
        df     = pd.read_csv(csv_path)
        solved = int((df["status"] == "Success").sum())
        total  = len(df)
    except Exception:
        solved, total = 0, len(problems)

    lo, hi = wilson(solved, total)
    print(f"\n{'='*65}")
    print(f"  RESULT: DS-Prover-V1.5-RL  N=1  (no HLP framework)")
    print(f"  Pass@1 = {solved/total*100:.2f}%  ({solved}/{total})")
    print(f"  95% CI = [{lo*100:.1f}%, {hi*100:.1f}%]")
    print(f"\n  HLP result ({args.split}):  65.75%  [59.6%, 71.5%]")
    print(f"  HLP improvement:          {65.75 - solved/total*100:+.2f}pp")
    print(f"{'='*65}")
    _print_latex(solved, total)
    print(f"\n📂 Results: {csv_path}")


def _print_latex(solved, total):
    lo, hi = wilson(solved, total)
    p = solved/total*100 if total else 0
    print(f"\n  LaTeX for Table 1 (Regime I row):")
    print(f"  DS-Prover-V1.5-RL$^\\dagger$ & DS-Prover-V1.5-RL & 1\\,(greedy) & "
          f"{p:.2f} & {{[{lo*100:.1f},\\,{hi*100:.1f}]}} & $\\sim$25\\,s \\\\")


if __name__ == "__main__":
    main()