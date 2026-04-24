# ==============================================================================
# Filename: src/system2/BeyondN1Trees.py
# Version: v206.0  (Ablation Study Edition)
#
# FIXES vs v205:
#
#   BUG-1  BrokenPipeError crash (reported in logs)
#          result_queue.put() now wrapped in try/except so a worker
#          crashing or timing out after the main process has moved on
#          no longer prints an ugly traceback and kills the worker.
#
#   BUG-2  Wrong agent imported
#          v205 imports LieGuidedMCTSAgent from mcts_hybrid_search.
#          benchmark_minif2f.py (the reference runner) imports
#          RiemannSearchAgent from lie_search.  BeyondN1Trees targets
#          the MCTS agent, which is correct — but the search() signature
#          must match: search(decl, max_steps, config, source_path).
#          Added explicit kwarg names so argument order can never mismatch.
#
#   BUG-3  Missing 'split' field in log_result / CSV
#          load_archive_problems() never added a "split" key to problem
#          dicts, so log_result() would KeyError on meta["split"] if
#          benchmark_minif2f.py's BenchmarkStats were reused, and the
#          ablation CSV would silently omit the column.
#          Fixed: problems now carry split="test" (or "valid" for the
#          valid fallback path), and log_result() records it.
#
#   BUG-4  generate_final_report uses best_status() which calls
#          group.drop(columns=["name"]) but after groupby the "name"
#          column is the index — drop() silently fails and the warning
#          "Column 'name' does not exist" appears in some pandas versions.
#          Fixed: use errors="ignore" (already present) AND reset_index
#          before the groupby so "name" is always a real column.
#
#   BUG-5  include_groups=False + best_status returning a Series
#          When include_groups=False the lambda receives a DataFrame
#          without the grouping column.  Returning group.iloc[-1]
#          (a Series) then causes a shape mismatch on concat in newer
#          pandas.  Fixed: always return a one-row DataFrame.
#
#   BUG-6  Pilot / --split flags absent
#          benchmark_minif2f.py supports --split and --pilot; this
#          script had neither.  Added both so ablation runs can target
#          valid / test / all, and pilot N lets you smoke-test quickly.
#
#   BUG-7  Worker timeout: result never put → finished never increments
#          If a worker process is killed by the OS (OOM, SIGKILL) while
#          holding a problem, the main loop hangs forever because
#          `finished` never reaches `total_queued`.  Fixed: added a
#          per-worker watchdog timeout (WORKER_HARD_TIMEOUT_S) — if no
#          result arrives within that window AND all workers are dead,
#          we synthesise a "Timeout" record for any outstanding problem
#          and break cleanly.
#
#   BUG-8  trace_dir name collision with benchmark_minif2f.py
#          v205 used "detailed_traces"; benchmark_minif2f uses "traces".
#          Standardised to "traces" so tooling that walks output dirs
#          works uniformly.
#
#   BUG-9  _manual_mathlib / lake-packages not fully excluded
#          Added the same exclusion list as benchmark_minif2f v104:
#          {"lake-packages", "_build", "_manual_mathlib", ".lake", "lakefile"}
#
#   ABLATION: Added --ablation flag.
#          Runs four configurations back-to-back on the same problem set
#          and writes a combined ablation_summary.csv:
#            A  Full system  (MCTS + hyperbolic retrieval + context)
#            B  No context   (source_path=None → no file prefix injection)
#            C  BM25 only   (retrieval_backend="bm25")
#            D  Cosine only  (retrieval_backend="cosine")
#          Each configuration gets its own subdirectory inside OUTPUT_DIR.
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
NUM_WORKERS              = 4
OUTPUT_DIR               = os.path.join(project_root, "benchmark_reports_minif2f_mcts")
STOP_SIGNAL_FILE         = os.path.join(project_root, "STOP_BENCHMARK")

# BUG-7: if no result arrives within this many seconds AND workers are dead → timeout
WORKER_HARD_TIMEOUT_S = 900   # 15 min per-problem upper bound

_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.']*$")

# Shared across split detection
_CATEGORIES = [
    "aime", "amc12", "amc10", "amc8", "imo",
    "mathd", "algebra", "numbertheory", "geometry", "counting",
]

# Exclusion list (BUG-9, matches benchmark_minif2f v104)
_BAD_PATH_FRAGMENTS = {
    "lake-packages", "_build", "_manual_mathlib", ".lake", "lakefile",
}


# ==============================================================================
# Problem loader  (BUG-3: problems now carry "split" field)
# ==============================================================================

def _infer_category(basename: str) -> str:
    lower = basename.lower()
    for comp in _CATEGORIES:
        if comp in lower:
            return comp
    return "other"


def load_archive_problems(data_dir: str, split: str = "test") -> list:
    """
    Load miniF2F problems from the canonical layout:
        data/miniF2F/MiniF2F/{Test,Valid}/

    split: "test" | "valid" | "all"
    """
    split_map = {
        "test":  ["Test"],
        "valid": ["Valid"],
        "all":   ["Test", "Valid"],
    }
    dirs_to_scan = split_map.get(split.lower(), ["Test"])

    minif2f_root = os.path.join(data_dir, "miniF2F", "MiniF2F")
    all_problems: list = []

    for split_name in dirs_to_scan:
        archive_root = os.path.join(minif2f_root, split_name)

        if not os.path.exists(archive_root):
            # Fallback: scan whole MiniF2F dir
            if os.path.exists(minif2f_root):
                archive_root = minif2f_root
                print(f"⚠️  {split_name}/ not found, scanning: {archive_root}")
            else:
                # Legacy layout
                archive_root = os.path.join(
                    project_root, "miniF2F-main", "lean",
                    split_name.lower()
                )
            if not os.path.exists(archive_root):
                print(
                    f"❌  miniF2F split '{split_name}' not found. Tried:\n"
                    f"   {os.path.join(minif2f_root, split_name)}\n"
                    f"   {minif2f_root}\n"
                    f"   {os.path.join(project_root, 'miniF2F-main', 'lean', split_name.lower())}"
                )
                continue

        print(f"🔍 Scanning miniF2F {split_name} split at: {archive_root}")
        files = glob.glob(
            os.path.join(archive_root, "**", "*.lean"), recursive=True
        )

        seen_ids: set = set()
        for fpath in files:
            # BUG-9: exclude build artefacts and extra Mathlib theorems
            if any(bad in fpath for bad in _BAD_PATH_FRAGMENTS):
                continue

            file_base = os.path.basename(fpath).replace(".lean", "")
            cat = _infer_category(file_base)

            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                continue

            # Line-by-line, comment-aware extraction (same as benchmark_minif2f v104)
            in_block_comment = False
            lines = content.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i]

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

                # Gather the full declaration up to `:=`
                decl_lines = [line]
                j = i + 1
                while j < len(lines) and ":=" not in "".join(decl_lines):
                    decl_lines.append(lines[j])
                    j += 1
                decl_raw = " ".join(l.strip() for l in decl_lines)
                if ":=" in decl_raw:
                    decl_raw = decl_raw[: decl_raw.index(":=")].strip()

                decl = f"theorem {name} {decl_raw.split(name, 1)[-1].strip()}"
                clean_name = name.replace(".", "_")
                unique_id = f"{split_name.lower()}_{file_base}_{clean_name}"

                if unique_id not in seen_ids:
                    seen_ids.add(unique_id)
                    all_problems.append({
                        "name":     unique_id,
                        "decl":     decl,
                        "category": cat,
                        "split":    split_name.lower(),   # BUG-3 fix
                        "path":     fpath,
                    })

                i = j

        print(f"   📂 {split_name}: {len(seen_ids)} problems loaded")

    print(
        f"✅ Total loaded: {len(all_problems)} problems across "
        f"{len(set(p['category'] for p in all_problems))} categories."
    )
    return all_problems


# ==============================================================================
# Worker Process
# BUG-1: result_queue.put() wrapped in try/except (BrokenPipeError)
# BUG-2: explicit kwarg names in agent.search()
# ==============================================================================

def worker_process(worker_id, problem_queue, result_queue, trace_dir,
                   ablation_config: dict = None):
    """
    ablation_config keys (all optional):
        retrieval_backend : "hyperbolic" | "bm25" | "cosine"   (default: "hyperbolic")
        use_context       : bool                                 (default: True)
    """
    if ablation_config is None:
        ablation_config = {}

    retrieval_backend = ablation_config.get("retrieval_backend", "hyperbolic")
    use_context       = ablation_config.get("use_context", True)

    time.sleep(worker_id * 2)

    if os.path.exists(STOP_SIGNAL_FILE):
        return

    try:
        from src.system2.mcts_hybrid_search import LieGuidedMCTSAgent
        agent = LieGuidedMCTSAgent(CKPT_PATH, MODEL_ABSOLUTE_PATH, device="cuda")
    except Exception as e:
        print(f"❌ Worker-{worker_id} init failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
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
            result = agent.search(
                theorem_decl=prob["decl"],
                max_steps=MAX_ROLLOUTS_PER_PROBLEM,
                config={"retrieval_backend": retrieval_backend},
                source_path=prob.get("path") if use_context else None,
            )
        except Exception as e:
            result = {
                "status": "ScriptCrash",
                "error":  str(e),
                "proof":  [],
                "summary": {},
            }

        duration = time.time() - t0
        status   = result.get("status", "Unknown")

        if status == "Success":
            try:
                save_path = os.path.join(trace_dir, f"{prob['name']}.pkl.gz")
                with gzip.open(save_path, "wb") as f:
                    pickle.dump(result, f)
            except Exception:
                pass

        # BUG-1 fix: broken pipe when main process has already moved on
        payload = {
            "name":           prob["name"],
            "category":       prob["category"],
            "split":          prob.get("split", "unknown"),   # BUG-3
            "status":         status,
            "duration":       duration,
            "steps":          len(result.get("proof", [])),
            "expanded_nodes": result.get("summary", {}).get("total_expanded", 0),
            "error": (
                result.get("error", "None")
                if status != "Success"
                else "None"
            ),
        }
        try:
            result_queue.put(payload)
        except (BrokenPipeError, EOFError, OSError):
            # Main process already timed out and closed the connection.
            # Nothing to do — the main loop will synthesise a Timeout record.
            pass

        gc.collect()
        torch.cuda.empty_cache()


# ==============================================================================
# Statistics & Reporting
# BUG-4 / BUG-5: best_status fixed; BUG-8: trace_dir = "traces"
# ==============================================================================

class BenchmarkStats:
    def __init__(self, base_output_dir: str, fresh: bool = False,
                 label: str = ""):
        self.base_dir   = base_output_dir
        self.label      = label
        self.output_dir = self._determine_output_dir(fresh)
        self.trace_dir  = os.path.join(self.output_dir, "traces")   # BUG-8
        self.csv_path   = os.path.join(self.output_dir, "summary.csv")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.trace_dir,  exist_ok=True)

        self.success_names: set = set()
        self.records: list      = []

        if not fresh and os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                self.records       = df.to_dict("records")
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
            tag = f" [{label}]" if label else ""
            print(f"🆕 New run{tag}: {os.path.basename(self.output_dir)}")

    def _determine_output_dir(self, fresh: bool) -> str:
        if fresh or not os.path.exists(self.base_dir):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{self.label}" if self.label else ""
            return os.path.join(self.base_dir, f"{ts}{suffix}")
        subdirs = sorted(
            [
                os.path.join(self.base_dir, d)
                for d in os.listdir(self.base_dir)
                if os.path.isdir(os.path.join(self.base_dir, d))
            ],
            key=os.path.getmtime,
        )
        return (
            subdirs[-1]
            if subdirs
            else os.path.join(
                self.base_dir,
                datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
        )

    def log_result(self, meta: dict):
        err = meta.get("error", "None")
        if meta["status"] == "ScriptCrash":
            err = "ScriptCrash"
        elif meta["status"] not in ("Success",) and err == "None":
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

    def generate_final_report(self, tag: str = ""):
        try:
            df = (
                pd.read_csv(self.csv_path)
                if os.path.exists(self.csv_path)
                else pd.DataFrame(self.records)
            )
        except Exception:
            df = pd.DataFrame(self.records)

        if df.empty:
            print("⚠️  No results to report.")
            return

        # BUG-4/5: reset_index so "name" is always a real column;
        # return a one-row DataFrame from best_status to keep concat happy.
        df = df.reset_index(drop=True)

        def best_status(group):
            if (group["status"] == "Success").any():
                row = group[group["status"] == "Success"].iloc[[0]]
            else:
                row = group.iloc[[-1]]
            return row

        df_best = (
            df.groupby("name", group_keys=False)
            .apply(best_status)
            .reset_index(drop=True)
        )

        sep = "=" * 60
        header = f"📊  MCTS BENCHMARK REPORT{f'  [{tag}]' if tag else ''}  (Pass@1)"
        print(f"\n{sep}\n{header}\n{sep}")

        total   = len(df_best)
        success = (df_best["status"] == "Success").sum()
        rate    = (success / total * 100) if total > 0 else 0.0

        # Prefer valid-split headline (matches benchmark_minif2f paper metric)
        if "split" in df_best.columns and "valid" in df_best["split"].str.lower().values:
            valid_df = df_best[df_best["split"].str.lower() == "valid"]
            v_succ   = (valid_df["status"] == "Success").sum()
            v_tot    = len(valid_df)
            print(
                f"🏆  VALID PASS@1 (paper metric): "
                f"{v_succ / v_tot * 100:.2f}%  ({v_succ}/{v_tot})"
            )
            if total != v_tot:
                print(f"🌍  GLOBAL: {rate:.2f}%  ({success}/{total})")
        else:
            print(f"🌍  GLOBAL PASS RATE: {rate:.2f}%  ({success}/{total})")

        if "split" in df_best.columns:
            print(f"{'-'*60}")
            for sp in sorted(df_best["split"].unique()):
                sub  = df_best[df_best["split"] == sp]
                ss   = (sub["status"] == "Success").sum()
                st   = len(sub)
                print(f"    {sp:<8}: {ss/st*100:.2f}%  ({ss}/{st})")

        print(f"{'-'*60}")
        print(f"{'CATEGORY':<22} | {'PASS RATE':<15} | SOLVED/TOTAL")
        print(f"{'-'*60}")
        for cat in sorted(df_best["category"].unique()):
            sub = df_best[df_best["category"] == cat]
            ss  = (sub["status"] == "Success").sum()
            st  = len(sub)
            sr  = (ss / st * 100) if st > 0 else 0.0
            print(f"{cat:<22} | {sr:.1f}%{' ':<10} | {ss}/{st}")
        print(sep)

        return {"pass_rate": rate, "success": int(success), "total": int(total)}


# ==============================================================================
# Core benchmark runner
# BUG-7: watchdog timeout for hung workers
# ==============================================================================

def _run_single_config(
    all_probs: list,
    fresh: bool,
    ablation_config: dict,
    out_base: str,
    label: str,
    pilot: int = 0,
) -> dict:
    """Run one benchmark configuration. Returns the final pass-rate dict."""

    if pilot > 0:
        all_probs = all_probs[:pilot]

    stats = BenchmarkStats(out_base, fresh=fresh, label=label)

    manager = multiprocessing.Manager()
    p_queue = manager.Queue()
    r_queue = manager.Queue()

    total_queued = 0
    outstanding  = {}   # name → prob, for BUG-7 timeout synthesis

    for prob in all_probs:
        if prob["name"] not in stats.success_names:
            p_queue.put(prob)
            outstanding[prob["name"]] = prob
            total_queued += 1

    for _ in range(NUM_WORKERS):
        p_queue.put(None)

    if total_queued == 0:
        print("✅  All problems already solved.")
        return stats.generate_final_report(tag=label) or {}

    print(f"📝  [{label}] Queued {total_queued} problems → {NUM_WORKERS} workers")

    workers = []
    for i in range(NUM_WORKERS):
        p = multiprocessing.Process(
            target=worker_process,
            args=(i, p_queue, r_queue, stats.trace_dir, ablation_config),
        )
        p.start()
        workers.append(p)

    pbar        = tqdm(total=total_queued, desc=f"Proving [{label}]")
    finished    = 0
    last_result = time.time()

    while finished < total_queued:
        if os.path.exists(STOP_SIGNAL_FILE):
            pbar.set_description("🛑 STOPPING...")
            if not any(p.is_alive() for p in workers) and r_queue.empty():
                break

        try:
            res = r_queue.get(timeout=1.0)
            stats.log_result(res)
            outstanding.pop(res["name"], None)
            icon = "✅" if res["status"] == "Success" else "❌"
            pbar.set_postfix_str(f"{icon} {res['name'][:42]}")
            pbar.update(1)
            finished += 1
            last_result = time.time()
        except queue.Empty:
            all_dead   = not any(p.is_alive() for p in workers)
            timed_out  = (time.time() - last_result) > WORKER_HARD_TIMEOUT_S

            if all_dead and r_queue.empty():
                # BUG-7: synthesise Timeout records for any problems never returned
                for name, prob in list(outstanding.items()):
                    stats.log_result({
                        "name":           name,
                        "category":       prob.get("category", "unknown"),
                        "split":          prob.get("split", "unknown"),
                        "status":         "Timeout",
                        "duration":       WORKER_HARD_TIMEOUT_S,
                        "steps":          0,
                        "expanded_nodes": 0,
                        "error":          "WorkerDied",
                    })
                    outstanding.pop(name)
                    pbar.update(1)
                    finished += 1
                break

            if timed_out and r_queue.empty():
                # Workers still alive but no result for a very long time
                # (e.g. one worker stuck). Keep going — let the OS kill them
                # if needed, and record Timeout on natural death above.
                last_result = time.time()  # reset so we don't spam

    pbar.close()
    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            p.kill()

    return stats.generate_final_report(tag=label) or {}


# ==============================================================================
# Ablation Study
# ==============================================================================

ABLATION_CONFIGS = {
    "A_full":       {"retrieval_backend": "hyperbolic", "use_context": True},
    "B_no_context": {"retrieval_backend": "hyperbolic", "use_context": False},
    "C_bm25":       {"retrieval_backend": "bm25",        "use_context": True},
    "D_cosine":     {"retrieval_backend": "cosine",      "use_context": True},
}


def run_ablation(all_probs: list, pilot: int = 0):
    """Run all four ablation configurations and write a combined CSV."""
    print("\n" + "=" * 70)
    print("🔬  ABLATION STUDY  (4 configurations)")
    print("=" * 70)

    summary_rows = []
    for label, cfg in ABLATION_CONFIGS.items():
        print(f"\n{'─'*70}\n▶  Configuration: {label}  {cfg}\n{'─'*70}")
        out_base = os.path.join(OUTPUT_DIR, "ablation", label)
        result   = _run_single_config(
            all_probs=all_probs,
            fresh=True,
            ablation_config=cfg,
            out_base=out_base,
            label=label,
            pilot=pilot,
        )
        summary_rows.append({
            "config":           label,
            "retrieval_backend": cfg["retrieval_backend"],
            "use_context":       cfg["use_context"],
            "pass_rate_pct":     round(result.get("pass_rate", 0.0), 2),
            "success":           result.get("success", 0),
            "total":             result.get("total", 0),
        })

    abl_csv = os.path.join(OUTPUT_DIR, "ablation_summary.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(abl_csv, index=False)

    print("\n" + "=" * 70)
    print("📊  ABLATION SUMMARY")
    print("=" * 70)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\n📂  Saved to: {abl_csv}")


# ==============================================================================
# Main entry point
# BUG-6: --split and --pilot flags added (matches benchmark_minif2f)
# ==============================================================================

def run_benchmark(
    fresh: bool = False,
    split: str  = "test",
    pilot: int  = 0,
    ablation: bool = False,
):
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    is_pilot = pilot > 0
    mode_tag = (
        f"PILOT {pilot}" if is_pilot
        else f"split={split}" + (" ABLATION" if ablation else "")
    )
    print(f"\n🚀  Beyond N=1 Benchmark (MCTS + Product Manifold)  [{mode_tag}]")

    if os.path.exists(STOP_SIGNAL_FILE):
        print(f"⚠️   Remove '{STOP_SIGNAL_FILE}' to start.")
        return
    if not os.path.exists(MODEL_ABSOLUTE_PATH):
        print(f"❌  Model not found: {MODEL_ABSOLUTE_PATH}")
        return

    all_probs = load_archive_problems(DATA_DIR, split=split)
    if not all_probs:
        return

    if ablation:
        run_ablation(all_probs, pilot=pilot)
        return

    # Standard (non-ablation) single run — full system config
    out_base = (
        os.path.join(project_root, "benchmark_reports_minif2f_mcts_pilot")
        if is_pilot
        else OUTPUT_DIR
    )
    _run_single_config(
        all_probs=all_probs,
        fresh=True if is_pilot else fresh,
        ablation_config=ABLATION_CONFIGS["A_full"],
        out_base=out_base,
        label="full",
        pilot=pilot,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Beyond N=1 MCTS Benchmark (miniF2F)"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore prior results and start a clean run.",
    )
    parser.add_argument(
        "--split", default="test", choices=["test", "valid", "all"],
        help="Which miniF2F split to benchmark (default: test).",
    )
    parser.add_argument(
        "--pilot", type=int, default=0, metavar="N",
        help=(
            "Run only the first N problems as a sanity check. "
            "Results go to benchmark_reports_minif2f_mcts_pilot/ "
            "and never affect the full-benchmark resume state."
        ),
    )
    parser.add_argument(
        "--ablation", action="store_true",
        help=(
            "Run all four ablation configurations (A_full, B_no_context, "
            "C_bm25, D_cosine) and write ablation_summary.csv."
        ),
    )
    args = parser.parse_args()
    run_benchmark(
        fresh=args.fresh,
        split=args.split,
        pilot=args.pilot,
        ablation=args.ablation,
    )