# ==============================================================================
# Filename: src/system2/benchmark_standalone_v2.py
# Version: v2.0
#
# WHY v1 GAVE 0%:
#   v1 generated one tactic blindly (no goal-state feedback from Lean).
#   Without seeing the actual proof state, every model outputs wrong/empty
#   tactics → 0% regardless of model capability.
#   Mathstral pass4 got 3/100 only because sampling occasionally guesses
#   a trivially-correct one-liner (norm_num, omega) for the simplest problems.
#
# CORRECT APPROACH — "No-HLP Baseline":
#   Use the SAME RiemannSearchAgent search loop as HLP (40 steps, Lean REPL
#   feedback at every step), but with ALL hyperbolic components DISABLED:
#     - graph_emb = None       → no HGCN hints in prompt
#     - retrieval_mode = 'flat' → skip hyperbolic lookup
#     - tau = 0                → trust gate accepts all tactics
#     - Lie matrices zeroed    → no dynamics steering
#
#   This is identical to benchmark_greedy_baseline.py v2, extended to ALL
#   available backbone models. The result for DeepSeek-Prover-V1.5-RL was
#   33.33% — a meaningful, honest baseline.
#
# COMPARISON TO HILBERT:
#   HILBERT "standalone" = DS-Prover-V2 pass@4 = 61.3% (HILBERT Table 1)
#   Our "no-HLP" baseline = DS-Prover-V1.5 = 33.33%
#   Our HLP result        = 65.75%
#   HLP improvement over no-HLP baseline: +32.4pp
#
# Usage:
#   # All models, no-HLP baseline (comparable to HILBERT's standalone col)
#   python src/system2/benchmark_standalone_v2.py --split test --n 100
#
#   # Single model
#   python src/system2/benchmark_standalone_v2.py \
#       --model InternLM2-StepProver --split test --n 100
#
#   # Quick pilot
#   python src/system2/benchmark_standalone_v2.py --split valid --n 50
# ==============================================================================

import os
import sys
import re
import glob
import time
import random
import argparse
import math
import json
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
MODELS_DIR = os.path.join(project_root, "models")
CKPT_PATH  = os.path.join(DATA_DIR, "hgcn_final.pth")   # needed for agent init
OUTPUT_DIR = os.path.join(project_root, "results", "no_hlp_baseline")
SEED       = 42

MAX_STEPS               = 40    # same as HLP main benchmark
MAX_SECONDS_PER_PROBLEM = 300
HEARTBEAT_TIMEOUT       = 1200

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SIZE_B = {
    "DeepSeek-Prover-V1.5-RL":       7.0,
    "InternLM2-StepProver":          7.0,
    "Qwen2.5-Math-7B-Instruct":      7.0,
    "Mathstral-7B-v0.1":             7.0,
    "NuminaMath-7B-TIR":             7.0,
    "Kimina-Prover-Distill-7B":      7.0,
    "DeepSeek-R1-Distill-Llama-8B":  8.0,
    "Llama-3.1-8B-Instruct":         8.0,
    "Llama-3.1-70B-Instruct":       70.0,
    "Qwen2.5-Math-72B-Instruct":    72.0,
    "NuminaMath-72B-TIR":           72.0,
}

# HLP results for the "vs HLP" column in the output table
HLP_RESULTS = {
    "DeepSeek-Prover-V1.5-RL":      65.75,
    "InternLM2-StepProver":         60.00,
    "Qwen2.5-Math-7B-Instruct":     59.58,
    "Mathstral-7B-v0.1":            60.00,
    "NuminaMath-7B-TIR":            59.58,
    "Kimina-Prover-Distill-7B":      0.00,
    "DeepSeek-R1-Distill-Llama-8B": 60.00,
    "Llama-3.1-8B-Instruct":        59.58,
    "Llama-3.1-70B-Instruct":       73.00,
    "Qwen2.5-Math-72B-Instruct":    71.00,
    "NuminaMath-72B-TIR":           70.00,
}

_SENTINEL_DONE    = "DONE"
_SENTINEL_CRASHED = "CRASHED"


# ==============================================================================
# Helpers
# ==============================================================================

def detect_models() -> dict:
    if not os.path.exists(MODELS_DIR):
        return {}
    valid = {}
    for entry in sorted(os.listdir(MODELS_DIR)):
        full = os.path.join(MODELS_DIR, entry)
        if not os.path.isdir(full):
            continue
        has_config  = os.path.exists(os.path.join(full, "config.json"))
        has_weights = any(
            f.endswith(".safetensors")
            for f in os.listdir(full)
            if os.path.isfile(os.path.join(full, f))
        )
        if has_config or has_weights:
            valid[entry] = full
    return valid


def model_size(name: str) -> float:
    return MODEL_SIZE_B.get(name, 7.0)


def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
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
        print(f"❌ MiniF2F {folder} not found"); return []
    probs, seen = [], set()
    for fpath in glob.glob(os.path.join(root, "**", "*.lean"), recursive=True):
        if any(b in fpath for b in ("lake-packages", "_build", "_manual")):
            continue
        base = os.path.basename(fpath).replace(".lean", "")
        try: content = open(fpath, encoding="utf-8").read()
        except: continue
        in_block = False; lines = content.splitlines(); i = 0
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
            decl_lines = [line]; j = i+1
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
# Worker: RiemannSearchAgent with ALL hyperbolic components disabled
# ==============================================================================

def _worker(model_name: str, model_path: str,
            problem_queue, result_queue, csv_path: str):
    """
    Loads RiemannSearchAgent with the given backbone LLM but disables
    ALL hyperbolic components:
      graph_emb = None       → no retrieval, no hints
      retrieval_mode = flat  → bypass HGCN lookup
      tau = 0                → trust gate accepts all tactics
      Lie matrices zeroed    → no dynamics steering

    The agent still uses the full 40-step Lean REPL feedback loop,
    just without any geometric guidance.
    """
    import queue as _queue
    import threading

    def _send(kind):
        try: result_queue.put({"_sentinel": kind})
        except Exception: pass

    try:
        from src.system2.lie_search import RiemannSearchAgent

        agent = RiemannSearchAgent(CKPT_PATH, model_path, device="cuda")

        # ── Disable ALL hyperbolic components ──────────────────────────────
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
        # ───────────────────────────────────────────────────────────────────

        sz = model_size(model_name)
        print(f"  ✅ {model_name} ({sz:.0f}B) ready — HLP DISABLED "
              f"(no hints, tau=0, Lie=0)", flush=True)

    except Exception as e:
        print(f"  ❌ {model_name} init failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        _send(_SENTINEL_CRASHED); return

    records = []
    if os.path.exists(csv_path):
        try: records = pd.read_csv(csv_path).to_dict("records")
        except: pass

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
                holder[0] = agent.search(task["decl"], max_steps=MAX_STEPS)
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
            "model":  model_name,
            "status": status,
            "time":   round(time.time() - t0, 2),
            "method": "no_hlp",
        })
        pd.DataFrame(records).to_csv(csv_path, index=False)
        result_queue.put({"name": task["name"], "status": status})
        gc.collect()
        torch.cuda.empty_cache()

    _send(_SENTINEL_DONE)


# ==============================================================================
# Per-model runner
# ==============================================================================

def run_one_model(model_name: str, model_path: str,
                  problems: list, split: str) -> dict:
    import queue as _queue

    run_id   = f"{model_name.replace('/','_').replace('.','_')}_{split}_no_hlp"
    csv_path = os.path.join(OUTPUT_DIR, f"{run_id}.csv")

    done = set()
    if os.path.exists(csv_path):
        try:
            df         = pd.read_csv(csv_path)
            n_crash    = (df["status"] == "ScriptCrash").sum()
            crash_rate = n_crash / len(df) if len(df) else 0
            if crash_rate > 0.40 and len(df) > 5:
                print(f"  ⚠️  Stale CSV ({crash_rate*100:.0f}% crash) — deleting")
                os.remove(csv_path)
            else:
                done = set(df["name"].unique())
                print(f"  ↩️  Resuming ({len(done)} done, "
                      f"crash={crash_rate*100:.0f}%)")
        except: pass

    todo = [p for p in problems if p["name"] not in done]
    if not todo:
        df     = pd.read_csv(csv_path)
        solved = int((df["status"] == "Success").sum())
        return {"model": model_name, "split": split,
                "solved": solved, "total": len(df),
                "pass": solved/len(df) if len(df) else 0.0}

    try:    mp.set_start_method("spawn", force=True)
    except RuntimeError: pass

    manager = mp.Manager()
    p_queue = manager.Queue()
    r_queue = manager.Queue()
    for prob in todo: p_queue.put(prob)
    p_queue.put(None)

    worker = mp.Process(target=_worker,
                        args=(model_name, model_path,
                              p_queue, r_queue, csv_path))
    worker.start()

    pbar          = tqdm(total=len(todo),
                         desc=f"  {model_name[:22]} (no HLP)",
                         leave=False)
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
    worker.join(timeout=60)
    if worker.is_alive(): worker.terminate(); worker.join()

    try:
        df     = pd.read_csv(csv_path)
        solved = int((df["status"] == "Success").sum())
        total  = len(df)
    except:
        solved, total = 0, len(problems)

    return {"model": model_name, "split": split,
            "solved": solved, "total": total,
            "pass": solved/total if total else 0.0}


# ==============================================================================
# Main
# ==============================================================================

def main():
    all_models = detect_models()

    parser = argparse.ArgumentParser(
        description="No-HLP baseline: full search loop, zero hyperbolic guidance")
    parser.add_argument("--model",  default="all",
                        help="Model name, 'all' (small→large), or prefix.")
    parser.add_argument("--split",  default="test", choices=["test","valid"])
    parser.add_argument("--n",      type=int, default=100,
                        help="Problems per model (100=good, 244=full test)")
    parser.add_argument("--seed",   type=int, default=SEED)
    parser.add_argument("--list",   action="store_true")
    args = parser.parse_args()

    if args.list:
        print(f"\nDetected models (small→large):")
        for name in sorted(all_models, key=model_size):
            sz  = model_size(name)
            hlp = HLP_RESULTS.get(name, "?")
            print(f"  {name:<38} {sz:.0f}B   HLP={hlp}%")
        return

    # Resolve model list (skip all-MiniLM — it's an embedding model)
    SKIP_MODELS = {"all-MiniLM-L6-v2"}
    if args.model == "all":
        model_list = [k for k in sorted(all_models, key=model_size)
                      if k not in SKIP_MODELS]
    else:
        if args.model in all_models:
            model_list = [args.model]
        else:
            matches = [k for k in all_models
                       if k.startswith(args.model) and k not in SKIP_MODELS]
            model_list = matches
            if not model_list:
                print(f"❌ '{args.model}' not found. Use --list."); return

    print(f"\n{'='*72}")
    print(f"  No-HLP Baseline v2  |  split={args.split}  n={args.n}")
    print(f"  Method: full search loop (40 steps, Lean REPL feedback)")
    print(f"  HLP disabled: no HGCN hints, tau=0, Lie=zeroed")
    print(f"  Models ({len(model_list)}, small→large): {model_list}")
    print(f"{'='*72}\n")

    problems = load_problems(args.split, args.n, args.seed)
    if not problems: return

    results      = []
    summary_path = os.path.join(OUTPUT_DIR, f"no_hlp_summary_{args.split}.json")

    for model_name in model_list:
        model_path = all_models[model_name]
        sz = model_size(model_name)
        print(f"\n{'─'*72}\n🤖  {model_name}  ({sz:.0f}B)  [no HLP]")

        res = run_one_model(model_name, model_path, problems, args.split)
        lo, hi = wilson(res["solved"], res["total"])
        hlp    = HLP_RESULTS.get(model_name)
        gain   = f"+{hlp - res['pass']*100:.2f}pp" if hlp else "—"
        print(f"  No-HLP Pass@1 = {res['pass']*100:.2f}%  "
              f"({res['solved']}/{res['total']})  "
              f"CI=[{lo*100:.1f}%, {hi*100:.1f}%]  "
              f"HLP gain: {gain}")
        results.append({**res, "lo": lo, "hi": hi})

        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)

    # Final summary
    print(f"\n{'='*72}")
    print(f"  NO-HLP BASELINE SUMMARY  (split={args.split})")
    print(f"{'='*72}")
    print(f"{'Model':<35} {'Sz':>4} {'No-HLP':>8}  {'HLP':>8}  {'Gain':>10}  {'CI (no-HLP)'}")
    print("-"*80)

    for r in sorted(results, key=lambda x: model_size(x["model"])):
        lo, hi = r["lo"]*100, r["hi"]*100
        hlp    = HLP_RESULTS.get(r["model"])
        gain   = f"+{hlp - r['pass']*100:.1f}pp" if hlp else "—"
        sz     = f"{model_size(r['model']):.0f}B"
        ci     = f"[{lo:.1f}%, {hi:.1f}%]"
        print(f"{r['model']:<35} {sz:>4} {r['pass']*100:>7.2f}%  "
              f"{hlp:>7.2f}%  {gain:>10}  {ci}")

    # Compute avg HLP gain
    gains = [(HLP_RESULTS[r["model"]] - r["pass"]*100)
             for r in results if r["model"] in HLP_RESULTS]
    if gains:
        print(f"\n  Average HLP gain: +{sum(gains)/len(gains):.1f}pp across "
              f"{len(gains)} models")

    print(f"\n📋 LaTeX (comparison table for paper):")
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(r"Model & Size & No-HLP (\%) & HLP (\%) & Gain \\ \midrule")
    for r in sorted(results, key=lambda x: model_size(x["model"])):
        hlp  = HLP_RESULTS.get(r["model"], 0)
        sz   = f"{model_size(r['model']):.0f}B"
        gain = f"+{hlp - r['pass']*100:.1f}"
        short = (r["model"].replace("-Instruct","").replace("-Prover","")
                           .replace("-Distill-Llama",""))
        print(f"{short} & {sz} & "
              f"{r['pass']*100:.2f} & {hlp:.2f} & {gain}\\,pp \\\\")
    print(r"\bottomrule" + "\n" + r"\end{tabular}")

    print(f"\n📂 Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()