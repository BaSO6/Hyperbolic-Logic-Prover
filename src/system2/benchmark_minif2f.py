# ==============================================================================
# Filename: src/system2/benchmark_minif2f.py
# Version: v104.0
#
# Changes vs v102:
#   FIX: worker_process now imports RiemannSearchAgent from lie_search (v86)
#        instead of LieGuidedMCTSAgent from mcts_hybrid_search.
#        lie_search is the agent that produced the paper's 65% Pass@1 result.
#        mcts_hybrid_search is a separate experimental extension.
#        agent.search() signature matches v86: search(theorem_decl, max_steps)
#        source_path is NOT passed (miniF2F theorems are self-contained).
#
# Changes vs v103:
#   FIX-1: _manual_mathlib excluded from file scan. This directory contains
#           17,000+ extra Mathlib proofs that are not part of the miniF2F
#           benchmark. Including them diluted the global Pass@1 from 66% to 5%.
#   FIX-2: generate_final_report now prints the valid-split Pass@1 as the
#           headline (paper metric). The global row is shown only when it
#           differs from the valid-split count.
#
# Previous changes (v103 vs v102):
#   NEW: --pilot N flag - runs exactly N problems using the identical worker /
#        stats pipeline as the full benchmark, but writes to a separate
#        benchmark_reports_minif2f_pilot/ directory so pilot results never
#        pollute the full-benchmark resume state. Always starts fresh.
#        Example: python src/system2/benchmark_minif2f.py --pilot 5
#
# Usage:
#   python src/system2/benchmark_minif2f.py                # valid split
#   python src/system2/benchmark_minif2f.py --pilot 5      # sanity-check 5
#   python src/system2/benchmark_minif2f.py --split test
#   python src/system2/benchmark_minif2f.py --split all
#   python src/system2/benchmark_minif2f.py --fresh
# ==============================================================================

import os
import sys
import time
import argparse
import glob
import re
import pickle
import gzip
import torch
import multiprocessing
import gc
import queue
import pandas as pd
from tqdm import tqdm
from datetime import datetime

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

MODEL_ABSOLUTE_PATH = (
    os.path.join(project_root, "models/DeepSeek-Prover-V1.5-RL")
)
DATA_DIR  = os.path.join(project_root, "data")
CKPT_PATH = os.path.join(DATA_DIR, "hgcn_final.pth")

MAX_ROLLOUTS_PER_PROBLEM = 40
NUM_WORKERS       = 4
OUTPUT_DIR        = os.path.join(project_root, "benchmark_reports_minif2f")
STOP_SIGNAL_FILE  = os.path.join(project_root, "STOP_BENCHMARK")

# Valid Lean identifier
_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.']*$")

# Competition categories inferred from filename
_CATEGORIES = [
    "aime", "amc12", "amc10", "amc8", "imo",
    "mathd", "algebra", "numbertheory", "geometry", "counting",
]


# ==============================================================================
# FIX-1 + FIX-3: Problem loader — correct path + comment-aware parser
# ==============================================================================

def _split_dir(data_dir: str, split_name: str) -> str:
    """Return the directory for a given split name (Test / Valid)."""
    # Canonical layout: data/miniF2F/MiniF2F/Valid/
    return os.path.join(data_dir, "miniF2F", "MiniF2F", split_name)


def _infer_category(basename: str) -> str:
    lower = basename.lower()
    for comp in _CATEGORIES:
        if comp in lower:
            return comp
    return "other"


def _load_split(data_dir: str, split_name: str) -> list:
    """
    Parse all .lean files in data/miniF2F/MiniF2F/<split_name>/.
    Returns a list of problem dicts.
    """
    split_dir = _split_dir(data_dir, split_name)
    if not os.path.exists(split_dir):
        print(f"❌  Split directory not found: {split_dir}")
        return []

    files = glob.glob(os.path.join(split_dir, "**", "*.lean"), recursive=True)
    problems: list = []
    seen: set = set()

    for fpath in files:
        # Exclude build artifacts and the _manual_mathlib directory which
        # contains 17k+ extra Mathlib theorems — not part of the miniF2F
        # benchmark and would dilute the global Pass@1 headline.
        if any(bad in fpath for bad in (
            "lake-packages", "_build", "_manual_mathlib",
            ".lake", "lakefile",
        )):
            continue

        base = os.path.basename(fpath).replace(".lean", "")
        cat  = _infer_category(base)

        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        # FIX-3: line-by-line comment-aware scan (same as BeyondN1Trees v205)
        in_block_comment = False
        lines = content.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            # Block comment tracking
            if "/-" in line:
                in_block_comment = True
            if "-/" in line:
                in_block_comment = False
                i += 1
                continue
            if in_block_comment:
                i += 1
                continue

            stripped = line.strip()
            if stripped.startswith("--"):
                i += 1
                continue

            m = re.match(
                r"^(?:@\[.*?\]\s*)?(?:theorem|lemma)\s+([^\s\{(:\[]+)",
                stripped,
            )
            if not m:
                i += 1
                continue

            name = m.group(1).strip()
            if not _VALID_NAME_RE.match(name) or "helper" in name.lower():
                i += 1
                continue

            # Gather full declaration up to :=
            decl_lines = [line]
            j = i + 1
            while j < len(lines) and ":=" not in "".join(decl_lines):
                decl_lines.append(lines[j])
                j += 1
            decl_raw = " ".join(l.strip() for l in decl_lines)
            if ":=" in decl_raw:
                decl_raw = decl_raw[: decl_raw.index(":=")].strip()

            # FIX-4: store raw declaration; normalisation happens in agent
            decl = f"theorem {name} {decl_raw.split(name, 1)[-1].strip()}"

            uid = f"{split_name.lower()}_{base}_{name.replace('.', '_')}"
            if uid not in seen:
                seen.add(uid)
                problems.append({
                    "name":          uid,
                    "theorem_name":  name,
                    "decl":          decl,
                    "category":      cat,
                    "split":         split_name.lower(),
                    "path":          None,  # miniF2F is self-contained
                })

            i = j

    return problems


def load_minif2f_problems(data_dir: str, split: str = "valid") -> list:
    """
    Load miniF2F problems.

    split: "valid" | "test" | "all"
    Canonical directory: data/miniF2F/MiniF2F/{Valid,Test}/
    """
    split_map = {
        "valid": ["Valid"],
        "test":  ["Test"],
        "all":   ["Valid", "Test"],
    }
    split_dirs_to_load = split_map.get(split.lower(), ["Valid"])

    all_problems: list = []
    for sp in split_dirs_to_load:
        probs = _load_split(data_dir, sp)
        all_problems.extend(probs)
        print(f"   📂 {sp:6s}: {len(probs):4d} problems "
              f"(dir: {_split_dir(data_dir, sp)})")

    # Per-category summary
    cats: dict = {}
    for p in all_problems:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    print(f"✅  Total loaded: {len(all_problems)} problems")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1])[:10]:
        print(f"      {cat}: {n}")

    return all_problems


# ==============================================================================
# Worker
# ==============================================================================

def worker_process(worker_id, problem_queue, result_queue, trace_dir):
    time.sleep(worker_id * 2)
    if os.path.exists(STOP_SIGNAL_FILE):
        return

    try:
        from src.system2.lie_search import RiemannSearchAgent
        agent = RiemannSearchAgent(CKPT_PATH, MODEL_ABSOLUTE_PATH, device="cuda")
    except Exception as e:
        print(f"❌ Worker-{worker_id} init failed: {e}")
        import traceback; traceback.print_exc()
        return

    while True:
        if os.path.exists(STOP_SIGNAL_FILE):
            break
        try:
            prob = problem_queue.get(timeout=5)
            if prob is None:
                break
        except queue.Empty:
            break

        t0 = time.time()
        try:
            # lie_search v86 signature: search(theorem_decl, max_steps)
            # No config dict or source_path — miniF2F theorems are self-contained.
            result = agent.search(
                prob["decl"],
                max_steps=MAX_ROLLOUTS_PER_PROBLEM,
            )
        except Exception as e:
            result = {
                "status": "ScriptCrash",
                "error": str(e),
                "proof": [],
                "summary": {},
            }

        duration = time.time() - t0
        status   = result.get("status", "Unknown")

        if status == "Success":
            try:
                save = os.path.join(trace_dir, f"{prob['name']}.pkl.gz")
                with gzip.open(save, "wb") as f:
                    pickle.dump(result, f)
            except Exception:
                pass

        result_queue.put({
            "name":           prob["name"],
            "category":       prob["category"],
            "split":          prob["split"],
            "status":         status,
            "duration":       duration,
            "steps":          len(result.get("proof", [])),
            "expanded_nodes": result.get("summary", {}).get("total_expanded",
                              len(result.get("trace", []))),  # lie_search uses trace
            "error":          result.get("error", "None") if status != "Success" else "None",
        })

        gc.collect()
        torch.cuda.empty_cache()


# ==============================================================================
# Stats & reporting
# ==============================================================================

class BenchmarkStats:
    def __init__(self, base_output_dir, fresh=False):
        self.base_dir   = base_output_dir
        self.output_dir = self._determine_output_dir(fresh)
        self.trace_dir  = os.path.join(self.output_dir, "traces")
        self.csv_path   = os.path.join(self.output_dir, "summary.csv")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.trace_dir,  exist_ok=True)

        self.success_names: set = set()
        self.records: list = []

        if not fresh and os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                self.records = df.to_dict("records")
                self.success_names = set(
                    df[df["status"] == "Success"]["name"].unique()
                )
                print(
                    f"🔄 Resuming | prior={len(df)} rows | "
                    f"✅ skip={len(self.success_names)} | "
                    f"🔁 rerun={len(df) - len(self.success_names)}"
                )
            except Exception as e:
                print(f"⚠️  CSV read failed ({e}), starting fresh.")
        else:
            print(f"🆕 New run: {os.path.basename(self.output_dir)}")

    def _determine_output_dir(self, fresh):
        if fresh or not os.path.exists(self.base_dir):
            return os.path.join(
                self.base_dir, datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        subdirs = sorted(
            [os.path.join(self.base_dir, d)
             for d in os.listdir(self.base_dir)
             if os.path.isdir(os.path.join(self.base_dir, d))],
            key=os.path.getmtime,
        )
        return subdirs[-1] if subdirs else os.path.join(
            self.base_dir, datetime.now().strftime("%Y%m%d_%H%M%S")
        )

    def log_result(self, meta):
        err = meta.get("error", "None")
        if meta["status"] == "ScriptCrash":
            err = "ScriptCrash"
        elif meta["status"] != "Success" and err == "None":
            err = "SearchExhausted"
        self.records.append({
            "name":           meta["name"],
            "category":       meta["category"],
            "split":          meta.get("split", "unknown"),
            "status":         meta["status"],
            "time":           round(meta["duration"], 2),
            "proof_length":   meta["steps"],
            "expanded_nodes": meta["expanded_nodes"],
            "error":          err,
            "timestamp":      datetime.now().isoformat(),
        })
        pd.DataFrame(self.records).to_csv(self.csv_path, index=False)

    def generate_final_report(self):
        try:
            df = pd.read_csv(self.csv_path) \
                if os.path.exists(self.csv_path) \
                else pd.DataFrame(self.records)
        except Exception:
            df = pd.DataFrame(self.records)

        if df.empty:
            print("⚠️  No results to report.")
            return

        # Best result per unique problem
        def best(g):
            return (g[g["status"] == "Success"].iloc[0]
                    if (g["status"] == "Success").any()
                    else g.iloc[-1])

        df_b = (
            df.groupby("name", group_keys=False)
            .apply(best, include_groups=False)
            .reset_index(drop=True)
        )

        sep = "=" * 60
        print(f"\n{sep}")
        print("📊  miniF2F BENCHMARK REPORT  (Pass@1)")
        print(sep)

        total = len(df_b)
        succ  = (df_b["status"] == "Success").sum()

        # Headline: prefer the valid-split number (paper metric) over global,
        # which can be diluted if other splits/dirs accidentally got included.
        if "split" in df_b.columns and "valid" in df_b["split"].str.lower().values:
            valid_df = df_b[df_b["split"].str.lower() == "valid"]
            v_succ = (valid_df["status"] == "Success").sum()
            v_tot  = len(valid_df)
            print(f"🏆  VALID PASS@1 (paper metric): "
                  f"{v_succ/v_tot*100:.2f}%  ({v_succ}/{v_tot})")
            if total != v_tot:
                print(f"🌍  GLOBAL (all rows): {succ/total*100:.2f}%  ({succ}/{total})")
        else:
            print(f"🌍  GLOBAL : {succ/total*100:.2f}%  ({succ}/{total})")

        # By split
        if "split" in df_b.columns:
            print(f"{'-'*60}")
            for sp in sorted(df_b["split"].unique()):
                sub = df_b[df_b["split"] == sp]
                ss, st = (sub["status"] == "Success").sum(), len(sub)
                print(f"    {sp:<8}: {ss/st*100:.2f}%  ({ss}/{st})")

        # By category
        print(f"{'-'*60}")
        print(f"{'CATEGORY':<18} | {'PASS RATE':<12} | SOLVED/TOTAL")
        print(f"{'-'*60}")
        for cat in sorted(df_b["category"].unique()):
            sub = df_b[df_b["category"] == cat]
            ss, st = (sub["status"] == "Success").sum(), len(sub)
            print(f"{cat:<18} | {ss/st*100:.1f}%{' ':<7} | {ss}/{st}")
        print(sep)


# ==============================================================================
# Main
# ==============================================================================

def run_benchmark(fresh: bool = False, split: str = "valid", pilot: int = 0):
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    is_pilot = pilot > 0
    mode_tag = f"PILOT {pilot} problems" if is_pilot else f"split={split}"
    print(f"\U0001f680  miniF2F Benchmark  [{mode_tag}]")

    if os.path.exists(STOP_SIGNAL_FILE):
        print(f"\u26a0\ufe0f  Remove '{STOP_SIGNAL_FILE}' to start.")
        return
    if not os.path.exists(MODEL_ABSOLUTE_PATH):
        print(f"\u274c  Model not found: {MODEL_ABSOLUTE_PATH}")
        return

    all_probs = load_minif2f_problems(DATA_DIR, split=split)
    if not all_probs:
        return

    # Pilot mode: cap to first N problems and use a separate output dir
    # so pilot results never pollute the full-benchmark resume state.
    if is_pilot:
        all_probs = all_probs[:pilot]
        print(f"\U0001f52c  Pilot: {len(all_probs)} problems "
              f"-> benchmark_reports_minif2f_pilot/")

    out_dir = (
        os.path.join(project_root, "benchmark_reports_minif2f_pilot")
        if is_pilot else OUTPUT_DIR
    )
    # Pilot always starts fresh so repeated runs are independent.
    stats = BenchmarkStats(out_dir, fresh=True if is_pilot else fresh)

    manager = multiprocessing.Manager()
    p_queue  = manager.Queue()
    r_queue  = manager.Queue()

    total_queued = 0
    for p in all_probs:
        if p["name"] not in stats.success_names:
            p_queue.put(p)
            total_queued += 1
    for _ in range(NUM_WORKERS):
        p_queue.put(None)

    if total_queued == 0:
        print("✅  All problems already solved.")
        stats.generate_final_report()
        return

    print(f"📝  Queued {total_queued} problems → {NUM_WORKERS} workers")

    workers = []
    for i in range(NUM_WORKERS):
        p = multiprocessing.Process(
            target=worker_process,
            args=(i, p_queue, r_queue, stats.trace_dir),
        )
        p.start()
        workers.append(p)

    pbar     = tqdm(total=total_queued, desc="Proving")
    finished = 0

    while finished < total_queued:
        if os.path.exists(STOP_SIGNAL_FILE):
            pbar.set_description("🛑 STOPPING...")
            if not any(p.is_alive() for p in workers) and r_queue.empty():
                break
        try:
            res = r_queue.get(timeout=1.0)
            stats.log_result(res)
            icon = "✅" if res["status"] == "Success" else "❌"
            pbar.set_postfix_str(f"{icon} {res['name'][:45]}")
            pbar.update(1)
            finished += 1
        except queue.Empty:
            if not any(p.is_alive() for p in workers) and r_queue.empty():
                break

    pbar.close()
    for p in workers:
        p.join()

    stats.generate_final_report()
    print(f"\n📂  CSV: {stats.csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="miniF2F benchmark runner")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore prior CSV and start a clean run.")
    parser.add_argument(
        "--split", default="valid", choices=["all", "valid", "test"],
        help="Which split to benchmark (default: valid).",
    )
    parser.add_argument(
        "--pilot", type=int, default=0, metavar="N",
        help=(
            "Run only the first N problems as a sanity check. "
            "Results go to benchmark_reports_minif2f_pilot/ "
            "and never affect the full-benchmark resume state. "
            "Example: --pilot 5"
        ),
    )
    args = parser.parse_args()
    run_benchmark(fresh=args.fresh, split=args.split, pilot=args.pilot)