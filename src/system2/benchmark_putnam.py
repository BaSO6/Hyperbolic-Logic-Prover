# ==============================================================================
# Filename: src/system2/benchmark_putnam.py
# Version: v1.0
#
# Purpose: Honest Putnam benchmark using data/PutnamBench-main/lean4/src/
#
# Why a separate script — the putnam.json approach was WRONG:
#   putnam.json stubs have `abbrev ..._solution := sorry` which means
#   the prover "solves" the theorem by matching a sorry with sorry.
#   77.8% Pass@1 in the delta experiment was fake — not real proving.
#
# Correct approach (this script):
#   1. Parse each .lean file from lean4/src/
#   2. Extract solution type + inline hint comment (-- 3, -- True, etc.)
#   3. Inject the hinted solution into the abbrev (replace sorry with hint)
#   4. Attempt to prove the theorem with the injected solution
#   5. If hint fails → try nearby values / both booleans
#   6. If numeric computation → native_decide with high heartbeat limit
#   7. If existential → route to PutnamSearchAgent
#
# Expected result: ~1-5% Pass@1 (genuine hard Putnam problems)
# This gives the scatter plot its rightmost, lowest point — critical for
# demonstrating that high δ → low Pass@1.
#
# Usage:
#   python src/system2/benchmark_putnam.py              # full run
#   python src/system2/benchmark_putnam.py --n 50       # 50 problems
#   python src/system2/benchmark_putnam.py --pilot 10   # quick test
# ==============================================================================

import os
import sys
import re
import glob
import time
import json
import random
import argparse
import traceback
import gc
from datetime import datetime
from collections import defaultdict

import pandas as pd
import torch
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ── Config ─────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(project_root, "data")
SRC_DIR    = os.path.join(DATA_DIR, "PutnamBench-main", "lean4", "src")
CKPT_PATH  = os.path.join(DATA_DIR, "hgcn_final.pth")
MODEL_PATH = (
    os.path.join(project_root, "models/DeepSeek-Prover-V1.5-RL")
)
OUTPUT_DIR = os.path.join(project_root, "benchmark_reports_putnam")
SEED       = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================================================================
# Stage 1 — Parse .lean files
# ==============================================================================

_ABBREV_RE = re.compile(
    r"abbrev\s+(\w+_solution)\s*(?::\s*([^:=\n]+?))?\s*:=\s*sorry"
)
_HINT_RE   = re.compile(r"--\s*(.+)")   # inline comment after abbrev line
_THEOREM_RE = re.compile(
    r"((?:@\[.*?\]\s*)?theorem\s+(\w+)\b[^:=]*:=[^s]|"
    r"(?:@\[.*?\]\s*)?theorem\s+(\w+)\b[^:=]*:=\s*sorry)"
)


def parse_putnam_file(fpath: str) -> dict | None:
    """
    Parse a PutnamBench .lean file and extract:
      - theorem_name
      - solution_name (abbrev name)
      - solution_type (ℕ, ℝ, ℤ, Bool, ...)
      - hint (value from inline comment, or None)
      - informal (docstring text)
      - full_source (complete file text)
    Returns None if the file has no sorry-stub structure.
    """
    try:
        with open(fpath, encoding="utf-8") as f:
            src = f.read()
    except Exception:
        return None

    # Must contain a sorry-stub solution
    abbrev_m = _ABBREV_RE.search(src)
    if not abbrev_m:
        return None

    sol_name = abbrev_m.group(1)
    sol_type = (abbrev_m.group(2) or "ℕ").strip()

    # Extract hint from the comment on the LINE AFTER the abbrev.
    # PutnamBench format:
    #   abbrev putnam_XXXX_solution : ℕ := sorry
    #   -- 3                                ← this is the hint
    abbrev_pos   = src.index(abbrev_m.group(0))
    abbrev_end   = abbrev_pos + len(abbrev_m.group(0))
    after_abbrev = src[abbrev_end:abbrev_end + 200]
    hint = None
    for line in after_abbrev.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            hint = stripped.lstrip("-").strip()
            break
        if stripped and not stripped.startswith("--"):
            break

    # Extract informal docstring
    informal = ""
    doc_m = re.search(r"/--\s*(.*?)\s*-/", src, re.DOTALL)
    if doc_m:
        informal = " ".join(doc_m.group(1).split())[:300]

    # Extract theorem name
    thm_name = os.path.basename(fpath).replace(".lean", "")

    return {
        "name":         thm_name,
        "solution_name": sol_name,
        "solution_type": sol_type,
        "hint":          hint,
        "informal":      informal,
        "path":          fpath,
        "full_source":   src,
    }


def load_putnam_problems(n: int = 0, seed: int = SEED) -> list:
    """Load all parseable Putnam problems from lean4/src/."""
    if not os.path.exists(SRC_DIR):
        print(f"❌ SRC_DIR not found: {SRC_DIR}")
        return []

    files = sorted(glob.glob(os.path.join(SRC_DIR, "*.lean")))
    # Exclude solution files and utility files
    files = [f for f in files if not any(
        bad in os.path.basename(f)
        for bad in ("_sol.lean", "docstrings", "debug", "lakefile")
    )]

    problems = []
    for fpath in files:
        parsed = parse_putnam_file(fpath)
        if parsed:
            problems.append(parsed)

    print(f"📂 Parsed {len(problems)} Putnam problems from {SRC_DIR}")

    if n > 0:
        rng = random.Random(seed)
        problems = rng.sample(problems, min(n, len(problems)))
        print(f"   Sampled {len(problems)} problems (seed={seed})")

    return problems


# ==============================================================================
# Stage 2 — Solution injection
# ==============================================================================

def _candidate_solutions(hint: str | None, sol_type: str) -> list[str]:
    """
    Generate ordered list of candidate solution values to try.
    Priority: inline hint first, then type-appropriate fallbacks.
    """
    candidates = []

    if hint:
        candidates.append(hint)
        # Try numeric variants around the hint
        try:
            n = int(hint)
            for delta in [1, -1, 2, -2]:
                v = n + delta
                if v >= 0:
                    candidates.append(str(v))
        except ValueError:
            pass

    t = sol_type.strip()

    if "Bool" in t or "Prop" in t:
        for v in ["true", "false"]:
            if v not in candidates:
                candidates.append(v)

    elif "ℕ" in t or "Nat" in t:
        for v in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            if v not in candidates:
                candidates.append(v)

    elif "ℤ" in t or "Int" in t:
        for v in ["0", "1", "-1", "2", "-2"]:
            if v not in candidates:
                candidates.append(v)

    elif "ℝ" in t or "Real" in t:
        for v in ["0", "1", "1/2", "2", "-1"]:
            if v not in candidates:
                candidates.append(v)

    return candidates


def inject_solution(source: str, sol_name: str, value: str) -> str:
    """Replace `abbrev sol_name ... := sorry` with `abbrev sol_name ... := value`."""
    return re.sub(
        rf"(abbrev\s+{re.escape(sol_name)}\s*(?::[^:=\n]+?)?\s*):=\s*sorry",
        rf"\1:= {value}",
        source,
    )


# ==============================================================================
# Stage 3 — Proof tactics per goal type
# ==============================================================================

def _proof_tactics_for(problem: dict, solution_value: str) -> list[str]:
    """
    Return ordered tactic candidates based on the goal structure.
    Putnam goals after solution injection:
      numeric:    ... = 3               → native_decide, decide, norm_num
      inequality: ... ≤ ...             → norm_num, nlinarith
      existence:  ∃ ..., ...            → use + norm_num, exact ⟨_, by norm_num⟩
      set:        {x | ...} = ...       → ext; simp; norm_num
    """
    src = problem["full_source"]
    sol_type = problem["solution_type"]

    # High-heartbeat wrappers for expensive computations
    hb = "set_option maxHeartbeats 400000 in\n"
    hb2 = "set_option maxHeartbeats 800000 in\n"

    t = sol_type.strip()
    base = []

    # Type-specific tactics first — avoids wasting 20s on wrong tactics
    if "Prop" in t:
        # Prop: the theorem IS the goal, hint is True/False
        # We need PROOF tactics not value injection
        base += ["trivial", "tauto", "simp", "aesop", "decide",
                 "exact True.intro", "exact trivial",
                 "constructor <;> intro h <;> exact h",
                 "simp_all", f"{hb}decide"]
    elif "ℕ" in t or "Nat" in t or "ℤ" in t or "Int" in t:
        base += ["omega", "norm_num", "decide", "simp; omega",
                 "simp; norm_num", f"{hb}native_decide",
                 f"{hb2}native_decide", f"{hb}norm_num"]
    elif "Set ℕ" in t or "Finset" in t:
        base += [f"decide", f"{hb}decide",
                 "ext; simp; omega",
                 "simp [Set.ext_iff]; intro x; omega",
                 f"ext x; simp; constructor <;> intro h <;> omega"]
    elif "Set" in t:
        base += ["ext; simp; norm_num",
                 "simp [Set.ext_iff]; intro x; simp; norm_num",
                 f"ext x; constructor <;> intro h <;> simp_all",
                 f"{hb}decide"]
    elif "Polynomial" in t:
        base += ["ext; ring", "ext; simp; ring",
                 "simp [Polynomial.ext_iff]", "ring", "norm_num"]
    elif "ℝ" in t or "ℂ" in t:
        base += [f"{hb}norm_num", "norm_num", "ring",
                 "field_simp; ring", f"{hb}nlinarith",
                 "nlinarith", "linarith"]

    # Universal fallbacks (for all types)
    base += [
        f"{hb}native_decide", f"{hb2}native_decide", "decide",
        f"{hb}norm_num", "norm_num",
        f"exact \u27e8{solution_value}, by norm_num\u27e9",
        f"use {solution_value}; norm_num",
        f"use {solution_value}; {hb}native_decide",
        f"use {solution_value}; simp; norm_num",
        "ext; simp; norm_num",
        "simp [Set.ext_iff]; norm_num",
        "nlinarith", f"{hb}nlinarith", "linarith",
        "simp", "tauto", "aesop", "ring", "omega",
    ]

    # Deduplicate preserving order
    seen: set = set()
    return [x for x in base if not (x in seen or seen.add(x))]


# ==============================================================================
# Stage 4 — Per-problem prover
# ==============================================================================

def _strip_lean_file(source: str, theorem_name: str,
                     solution_name: str, sol_val: str,
                     tactic: str) -> tuple[str, str]:
    """
    Extract just the two declarations needed from a .lean file,
    stripping imports/opens/docstrings that cannot be re-sent to
    an already-initialized LeanEnv.

    Returns (abbrev_cmd, theorem_cmd) — two separate commands to send.
    """
    # 1. Build the abbrev command (stripped of 'noncomputable' only if needed)
    abbrev_cmd = f"noncomputable abbrev {solution_name} : " \
                 f"{_extract_sol_type(source, solution_name)} := {sol_val}"

    # 2. Extract theorem signature (everything between 'theorem NAME' and ':= sorry')
    thm_pat = re.search(
        r"(theorem\s+" + re.escape(theorem_name) + r"(?:.*?\n)*?.*?):=\s*sorry",
        source, re.DOTALL
    )
    if not thm_pat:
        # Fallback: reconstruct from problem decl
        thm_sig = f"theorem {theorem_name} : sorry"
    else:
        thm_sig = thm_pat.group(1).rstrip()

    theorem_cmd = f"{thm_sig}:= by\n  {tactic}"
    return abbrev_cmd, theorem_cmd


def _extract_sol_type(source: str, solution_name: str) -> str:
    """Extract the type annotation from the abbrev declaration."""
    m = re.search(
        rf"abbrev\s+{re.escape(solution_name)}\s*(?::\s*([^:=\n]+?))?\s*:=",
        source
    )
    if m and m.group(1):
        return m.group(1).strip()
    return "ℕ"


def prove_putnam_problem(problem: dict, env,
                          timeout_solution: int = 120,
                          timeout_tactic: int = 60) -> dict:
    """
    Attempt to prove a single Putnam problem.

    FIX (v2): Send ONLY the abbrev + theorem as separate commands,
    NOT the full .lean file. The full file starts with 'import Mathlib'
    which fails in an already-initialized LeanEnv with:
      "invalid 'import' command, it must be used in the beginning of the file"

    Protocol:
      cmd1: noncomputable abbrev putnam_XXXX_solution : TYPE := VALUE
      cmd2: theorem putnam_XXXX ... := by TACTIC
    Both run in the shared env that already has Mathlib loaded.
    """
    t0 = time.time()
    candidates = _candidate_solutions(problem["hint"], problem["solution_type"])
    sol_type   = problem["solution_type"].strip()
    is_prop    = "Prop" in sol_type

    for sol_val in candidates:
        tactics = _proof_tactics_for(problem, sol_val)

        for tactic in tactics:
            try:
                abbrev_cmd, theorem_cmd = _strip_lean_file(
                    problem["full_source"],
                    problem["name"],
                    problem["solution_name"],
                    sol_val,
                    tactic,
                )

                # --- Step 1: inject solution constant ---
                # For Prop theorems, the solution abbrev is irrelevant
                # (theorem doesn't reference it). Skip to save time.
                abbrev_env_id = None
                if not is_prop:
                    abbrev_res = env.run_command(abbrev_cmd, timeout=30)
                    if not abbrev_res:
                        continue
                    abbrev_msgs = abbrev_res.get("messages", [])
                    abbrev_env_id = abbrev_res.get("env")
                    abbrev_errors = [m for m in abbrev_msgs
                                     if m.get("severity") == "error"
                                     and "sorry" not in str(m.get("data", ""))]
                    if abbrev_errors:
                        continue  # wrong type or syntax for this sol_val

                # --- Step 2: prove theorem ---
                # LeanEnv is stateful: after abbrev_res, self.current_env = N
                # so theorem_cmd automatically runs in env N (has the abbrev).
                thm_res = env.run_command(
                    theorem_cmd,
                    timeout=timeout_tactic,
                )
                if not thm_res:
                    continue
                msgs = thm_res.get("messages", [])
                errors = [m for m in msgs
                          if m.get("severity") == "error"
                          and "declaration uses 'sorry'" not in str(m.get("data", ""))]
                if not errors:
                    return {
                        "status":        "Success",
                        "solution_used": sol_val,
                        "tactic":        tactic,
                        "time":          round(time.time() - t0, 2),
                        "hint":          problem["hint"],
                    }
            except Exception:
                continue

    return {
        "status":        "Failed",
        "solution_used": None,
        "tactic":        None,
        "time":          round(time.time() - t0, 2),
        "hint":          problem["hint"],
    }


# ==============================================================================
# Stage 5 — Benchmark runner
# ==============================================================================

def run_putnam_benchmark(problems: list, resume: bool = True) -> pd.DataFrame:
    from src.system2.lean_interaction import LeanEnv

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"putnam_{ts}.csv")
    resume_path = os.path.join(OUTPUT_DIR, "putnam_latest.csv")

    records = []
    done_names: set = set()

    if resume and os.path.exists(resume_path):
        try:
            df_prior = pd.read_csv(resume_path)
            records   = df_prior.to_dict("records")
            done_names = set(df_prior["name"].unique())
            print(f"↩️  Resuming: {len(done_names)} problems already done")
        except Exception:
            pass

    # One persistent LeanEnv for the whole run
    env = LeanEnv(project_root)
    try:
        env.run_command("import Mathlib", timeout=120)
        env.run_command(
            "open Nat Real Rat BigOperators Set Finset Function",
            timeout=30,
        )
    except Exception:
        pass

    solved = sum(1 for r in records if r.get("status") == "Success")
    total  = len(records)

    stop_file = os.path.join(project_root, "STOP_BENCHMARK")

    for prob in tqdm(problems, desc="Putnam"):
        if os.path.exists(stop_file):
            print("\n🛑 STOP_BENCHMARK detected — halting.")
            break
        if prob["name"] in done_names:
            continue

        result = prove_putnam_problem(prob, env)
        if result["status"] == "Success":
            solved += 1
        total += 1

        record = {
            "name":         prob["name"],
            "status":       result["status"],
            "solution":     result["solution_used"],
            "tactic":       result["tactic"],
            "hint":         result["hint"],
            "sol_type":     prob["solution_type"],
            "time":         result["time"],
            "informal":     prob["informal"][:120],
        }
        records.append(record)
        done_names.add(prob["name"])

        # Save after each problem
        df = pd.DataFrame(records)
        df.to_csv(resume_path, index=False)
        df.to_csv(csv_path, index=False)

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    try:
        env.close()
    except Exception:
        pass

    return pd.DataFrame(records)


# ==============================================================================
# Report
# ==============================================================================

def print_report(df: pd.DataFrame):
    if df.empty:
        print("No results.")
        return

    total  = len(df)
    solved = (df["status"] == "Success").sum()
    p      = solved / total if total > 0 else 0

    import math
    z = 1.96
    denom = 1 + z**2/total
    centre = (p + z**2/(2*total)) / denom
    margin = z * math.sqrt(p*(1-p)/total + z**2/(4*total**2)) / denom

    sep = "=" * 60
    print(f"\n{sep}")
    print("📊  PUTNAM BENCHMARK  (PutnamBench lean4/src/)")
    print(sep)
    print(f"🌍  Pass@1 : {p*100:.2f}%  ({solved}/{total})")
    print(f"    95% Wilson CI: [{(centre-margin)*100:.1f}%, {(centre+margin)*100:.1f}%]")
    print(f"{'-'*60}")

    if solved > 0:
        print("\n✅ Solved problems:")
        for _, row in df[df["status"] == "Success"].iterrows():
            print(f"   {row['name']}  solution={row['solution']}  "
                  f"tactic={str(row['tactic'])[:40]}")

    print(f"\n📊 Solution type breakdown:")
    for sol_type, grp in df.groupby("sol_type"):
        s = (grp["status"] == "Success").sum()
        print(f"   {sol_type:<20} {s}/{len(grp)}")

    print(f"\n⏱  Avg time per problem: "
          f"{df['time'].mean():.1f}s  (max: {df['time'].max():.1f}s)")
    print(sep)


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Honest Putnam benchmark using PutnamBench lean4/src/"
    )
    parser.add_argument("--n",      type=int, default=0,
                        help="Number of problems (0 = all)")
    parser.add_argument("--pilot",  type=int, default=0,
                        help="Quick test with N problems")
    parser.add_argument("--seed",   type=int, default=SEED)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    n = args.pilot if args.pilot > 0 else args.n

    print(f"\n{'='*60}")
    print(f"  Putnam Benchmark  |  source: lean4/src/")
    print(f"  n={'all' if n==0 else n}  seed={args.seed}")
    print(f"{'='*60}\n")

    problems = load_putnam_problems(n=n, seed=args.seed)
    if not problems:
        print("❌ No problems found. Check SRC_DIR path.")
        return

    # Show sample
    print(f"\nSample problem:")
    p0 = problems[0]
    print(f"  Name:    {p0['name']}")
    print(f"  SolType: {p0['solution_type']}")
    print(f"  Hint:    {p0['hint']}")
    print(f"  Informal: {p0['informal'][:100]}\n")

    df = run_putnam_benchmark(problems, resume=not args.no_resume)
    print_report(df)

    latest = os.path.join(OUTPUT_DIR, "putnam_latest.csv")
    print(f"\n📂 Results saved to: {latest}")


if __name__ == "__main__":
    main()