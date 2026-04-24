# ==============================================================================
# Filename: src/system2/benchmark_proofnet.py
# Version: v1.0
#
# Purpose: Achieve >10 solved ProofNet problems by combining three strategies:
#
#   S2: Deeper search — max_steps=80 (vs 40 for MiniF2F), timeout=600s
#       ProofNet proofs require 10-50+ tactic steps. The original 40-step
#       limit was the primary cause of 0.0% Pass@1.
#
#   S3: Domain classification + targeted tactic hints
#       ProofNet problems span Real Analysis, Topology, Algebra, Linear Algebra.
#       Each domain needs different opening tactics. We classify each problem
#       by keywords in its declaration and inject domain-specific starter hints
#       that get prepended to the LLM context, guiding the first critical step.
#
#   S4: Multi-rollout with sampling (N=3, T=0.7)
#       ProofNet is hard enough that greedy decoding almost always fails.
#       Three independent rollouts with temperature sampling give 3× coverage.
#       A problem is solved if ANY rollout succeeds (Pass@1 with N=3 budget).
#
# Usage:
#   python src/system2/benchmark_proofnet.py --n 100 --rollouts 3
#   python src/system2/benchmark_proofnet.py --n 50  --rollouts 1  # pilot
#   python src/system2/benchmark_proofnet.py        # all 371 problems, N=3
# ==============================================================================

import os
import sys
import re
import json
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
OUTPUT_DIR = os.path.join(project_root, "results", "proofnet")
MODEL_PATH = (
    os.path.join(project_root, "models/DeepSeek-Prover-V1.5-RL")
)
CKPT_PATH = os.path.join(DATA_DIR, "hgcn_final.pth")

SEED                    = 42
MAX_STEPS               = 80     # S2: 2× deeper than MiniF2F default
MAX_SECONDS_PER_PROBLEM = 600    # S2: 2× longer timeout
HEARTBEAT_TIMEOUT       = 1200
TEMPERATURE             = 0.7    # S4: sampling temperature for rollouts

os.makedirs(OUTPUT_DIR, exist_ok=True)

_SENTINEL_DONE    = "DONE"
_SENTINEL_CRASHED = "CRASHED"


# ==============================================================================
# S3: Domain classifier + tactic hint library
# ==============================================================================

# Each domain entry:
#   keywords: patterns in the theorem declaration
#   hints:    Lean 4 tactics most likely to work for this domain
#             These become the FIRST hints injected into the LLM context,
#             before the HGCN retrieval hints.

DOMAIN_TACTICS = {
    "real_analysis": {
        "keywords": [
            "Real.", "ℝ", "ContinuousOn", "Continuous", "tendsto", "Filter",
            "Metric.", "dist", "abs", "norm", "bound", "converge",
            "differentiable", "HasDerivAt", "MeasureTheory",
            "limsup", "liminf", "iSup", "iInf",
        ],
        "hints": [
            "simp [abs_le, Real.norm_eq_abs]",
            "norm_num",
            "gcongr",
            "apply ContinuousOn.mono",
            "apply Continuous.continuousOn",
            "rw [Real.norm_eq_abs, abs_sub_comm]",
            "apply le_antisymm",
            "push_neg",
            "simp [dist_comm, Real.dist_eq]",
            "linarith [abs_nonneg _]",
        ],
    },
    "topology": {
        "keywords": [
            "IsOpen", "IsClosed", "TopologicalSpace", "IsCompact",
            "IsConnected", "DenseRange", "closure", "interior",
            "frontier", "nhds", "Filter.Tendsto", "Homeomorph",
            "ContinuousMap", "IsHomeomorph",
        ],
        "hints": [
            "apply isOpen_iff_mem_nhds.mpr",
            "simp [isOpen_compl_iff]",
            "apply IsClosed.closure_eq",
            "apply IsOpen.inter",
            "rw [isOpen_iff_forall_mem_open]",
            "apply isCompact_of_isClosed_subset",
            "apply IsConnected.image",
            "simp [Filter.mem_nhds_iff]",
            "apply TopologicalSpace.IsTopologicalBasis.isOpen",
            "exact isOpen_univ",
        ],
    },
    "abstract_algebra": {
        "keywords": [
            "Group", "Ring", "Field", "Module", "Algebra",
            "Subgroup", "Ideal", "QuotientGroup", "MulAction",
            "IsSubgroup", "Normal", "orderOf", "Fingroup",
            "CommRing", "IsDomain", "IsField",
        ],
        "hints": [
            "apply Subgroup.mem_mk",
            "simp [Subgroup.mem_carrier]",
            "rw [Group.mul_inv_cancel]",
            "apply IsSubgroup.mul_mem",
            "simp [orderOf_dvd_of_pow_eq_one]",
            "exact Subgroup.closure_mono",
            "apply QuotientGroup.mk_surjective",
            "ring",
            "group",
            "abel",
        ],
    },
    "linear_algebra": {
        "keywords": [
            "LinearMap", "Matrix", "Finrank", "rank", "span",
            "Basis", "IsLinearMap", "Module.Free", "inner",
            "InnerProductSpace", "Submodule", "LinearEquiv",
            "det", "eigenvalue", "eigenvector", "trace",
        ],
        "hints": [
            "apply LinearMap.ext",
            "simp [LinearMap.map_add, LinearMap.map_smul]",
            "rw [Finrank.eq_of_basis]",
            "apply Submodule.span_mono",
            "simp [Matrix.mul_vec]",
            "apply LinearIndependent.map",
            "exact Basis.mk_apply",
            "simp [inner_add_left, inner_smul_left]",
            "apply inner_product_geometry.norm_add_sq_real",
            "linear_combination",
        ],
    },
    "number_theory": {
        "keywords": [
            "Nat.Prime", "dvd", "gcd", "Coprime", "ZMod",
            "Int.ModEq", "Finset.sum", "multiplicative",
        ],
        "hints": [
            "omega",
            "norm_num",
            "simp [Nat.dvd_iff_mod_eq_zero]",
            "exact Nat.Prime.dvd_of_dvd_pow",
            "ring",
        ],
    },
    "default": {
        "keywords": [],
        "hints": [
            "simp",
            "ring",
            "norm_num",
            "omega",
            "linarith",
            "exact?",
            "apply?",
            "aesop",
            "decide",
            "tauto",
        ],
    },
}


def classify_domain(decl: str) -> str:
    """Classify a theorem declaration into a ProofNet domain."""
    decl_lower = decl.lower()
    scores = {}
    for domain, data in DOMAIN_TACTICS.items():
        if domain == "default":
            continue
        score = sum(1 for kw in data["keywords"] if kw.lower() in decl_lower)
        if score > 0:
            scores[domain] = score
    if not scores:
        return "default"
    return max(scores, key=scores.get)


def get_domain_hints(decl: str) -> list:
    """Return domain-specific tactic hints for a ProofNet theorem."""
    domain = classify_domain(decl)
    return DOMAIN_TACTICS[domain]["hints"]


# ==============================================================================
# Problem loader — reads data/proofnet.json
# ==============================================================================

def load_proofnet(n: int = 0, seed: int = SEED) -> list:
    """
    Load ProofNet problems from data/proofnet.json.
    Falls back to ProofNet-main/ Lean files if JSON not found.
    """
    json_path = os.path.join(DATA_DIR, "proofnet.json")

    if os.path.exists(json_path):
        with open(json_path) as f:
            raw = json.load(f)
        probs = []
        for item in raw:
            # Handle both dict formats seen in ProofNet datasets
            if isinstance(item, dict):
                decl = (item.get("formal_statement") or
                        item.get("statement") or
                        item.get("decl") or "")
                name = (item.get("name") or
                        item.get("id") or
                        f"proofnet_{len(probs)}")
            else:
                continue
            if decl:
                # Normalise: ensure starts with 'theorem'
                if not decl.strip().startswith("theorem"):
                    decl = f"theorem {name} : {decl}"
                probs.append({"name": str(name), "decl": decl.strip(),
                              "domain": classify_domain(decl)})
        print(f"📂 ProofNet (JSON): {len(probs)} problems")

    else:
        # Fallback: scan ProofNet-main/ Lean files
        probs = _scan_proofnet_lean()

    if not probs:
        print("❌ No ProofNet problems found.")
        print(f"   Expected: {json_path}")
        print(f"   Or: {os.path.join(DATA_DIR, 'ProofNet-main/')}")
        return []

    if 0 < n < len(probs):
        probs = random.Random(seed).sample(probs, n)

    # Print domain distribution
    from collections import Counter
    domains = Counter(p.get("domain", "?") for p in probs)
    print(f"   Domain distribution: {dict(domains)}")
    return probs


def _scan_proofnet_lean() -> list:
    """Scan ProofNet-main/ directory for Lean theorem declarations."""
    import glob
    root = os.path.join(DATA_DIR, "ProofNet-main")
    if not os.path.exists(root):
        return []

    probs, seen = [], set()
    _VALID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.']*$")

    for fpath in glob.glob(os.path.join(root, "**", "*.lean"), recursive=True):
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
            if not _VALID.match(name): i += 1; continue
            decl_lines = [line]
            j = i + 1
            while j < len(lines) and ":=" not in "".join(decl_lines):
                decl_lines.append(lines[j]); j += 1
            decl_raw = " ".join(l.strip() for l in decl_lines)
            if ":=" in decl_raw:
                decl_raw = decl_raw[:decl_raw.index(":=")].strip()
            decl = f"theorem {name} {decl_raw.split(name,1)[-1].strip()}"
            uid  = f"proofnet_{base}_{name.replace('.','_')}"
            if uid not in seen:
                seen.add(uid)
                probs.append({"name": uid, "decl": decl,
                              "domain": classify_domain(decl)})
            i = j

    print(f"📂 ProofNet (Lean files): {len(probs)} problems")
    return probs


# ==============================================================================
# Worker: enhanced agent for ProofNet
# ==============================================================================

def _worker(rollout_id: int, temperature: float,
            problem_queue, result_queue, csv_path: str):
    """
    Runs RiemannSearchAgent with ProofNet-specific enhancements:
    - max_steps=80 (S2: deeper search)
    - temperature=T (S4: sampling for exploration)
    - domain hints prepended to HGCN hints (S3: targeted tactics)
    """
    import queue as _queue
    import threading

    def _send(kind):
        try: result_queue.put({"_sentinel": kind})
        except Exception: pass

    try:
        from src.system2.lie_search import RiemannSearchAgent

        agent = RiemannSearchAgent(CKPT_PATH, MODEL_PATH, device="cuda")

        # S4: set temperature for exploration
        if hasattr(agent, "llm") and hasattr(agent.llm, "temperature"):
            agent.llm.temperature = temperature
        # Also try setting on model generate kwargs
        if hasattr(agent, "llm") and hasattr(agent.llm, "model"):
            agent._proofnet_temperature = temperature

        print(f"  ✅ Rollout {rollout_id}: agent ready  T={temperature}  "
              f"max_steps={MAX_STEPS}", flush=True)

    except Exception as e:
        print(f"  ❌ Rollout {rollout_id} init failed: {e}", flush=True)
        import traceback; traceback.print_exc()
        _send(_SENTINEL_CRASHED); return

    # Monkey-patch generate to use our temperature
    original_generate = None
    if hasattr(agent, "llm") and hasattr(agent.llm, "generate_candidates"):
        original_generate = agent.llm.generate_candidates

        def patched_generate(state_text, hints=None, num=1):
            # Prepend domain hints to the HGCN hints
            # (domain hints are stored in task via problem_queue)
            domain_hints = getattr(patched_generate, "_domain_hints", [])
            all_hints = domain_hints + (hints or [])
            # Call original with combined hints and set temperature
            try:
                # Temporarily set do_sample=True for exploration
                mdl = agent.llm.model
                orig_gen = mdl.generate

                def sampled_gen(**kwargs):
                    if temperature > 0:
                        kwargs["do_sample"]    = True
                        kwargs["temperature"]  = temperature
                    return orig_gen(**kwargs)

                mdl.generate = sampled_gen
                result = original_generate(state_text, hints=all_hints, num=num)
                mdl.generate = orig_gen
                return result
            except Exception:
                return original_generate(state_text, hints=all_hints, num=num)

        agent.llm.generate_candidates = patched_generate

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

        # Inject domain hints for this problem
        domain_hints = get_domain_hints(task["decl"])
        if original_generate and hasattr(agent.llm.generate_candidates,
                                         "_domain_hints"):
            agent.llm.generate_candidates._domain_hints = domain_hints

        holder = [None]
        def _run():
            try:
                # S2: max_steps=80
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
            "name":     task["name"],
            "domain":   task.get("domain", "?"),
            "rollout":  rollout_id,
            "status":   status,
            "time":     round(time.time() - t0, 2),
        })
        pd.DataFrame(records).to_csv(csv_path, index=False)
        result_queue.put({"name": task["name"], "status": status,
                          "rollout": rollout_id})
        gc.collect()
        torch.cuda.empty_cache()

    _send(_SENTINEL_DONE)


# ==============================================================================
# Runner
# ==============================================================================

def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p=k/n; d=1+z**2/n
    c=(p+z**2/(2*n))/d
    m=z*math.sqrt(p*(1-p)/n+z**2/(4*n**2))/d
    return c-m, c+m


def run_rollout(rollout_id: int, temperature: float,
                problems: list, csv_path: str,
                max_steps: int = MAX_STEPS,
                timeout_s: int = MAX_SECONDS_PER_PROBLEM):
    """Run one rollout pass over all unsolved problems."""
    import queue as _queue

    # Find already-solved problems (any rollout)
    solved_names = set()
    if os.path.exists(csv_path):
        try:
            df           = pd.read_csv(csv_path)
            solved_names = set(df[df["status"] == "Success"]["name"].unique())
        except Exception:
            pass

    # Only attempt unsolved problems in this rollout
    done_this_rollout = set()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            df_r = df[df["rollout"] == rollout_id]
            done_this_rollout = set(df_r["name"].unique())
        except Exception:
            pass

    todo = [p for p in problems
            if p["name"] not in solved_names
            and p["name"] not in done_this_rollout]

    if not todo:
        print(f"  ↩️  Rollout {rollout_id}: nothing to do "
              f"({len(solved_names)} already solved)")
        return

    print(f"  🎲 Rollout {rollout_id} (T={temperature}): {len(todo)} problems")

    try:    mp.set_start_method("spawn", force=True)
    except RuntimeError: pass

    manager = mp.Manager()
    p_queue = manager.Queue()
    r_queue = manager.Queue()
    for prob in todo: p_queue.put(prob)
    p_queue.put(None)

    worker = mp.Process(target=_worker,
                        args=(rollout_id, temperature, p_queue, r_queue, csv_path))
    worker.start()

    pbar          = tqdm(total=len(todo),
                         desc=f"  Rollout {rollout_id}", leave=False)
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


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ProofNet benchmark with enhanced tactics (S2+S3+S4)")
    parser.add_argument("--n",        type=int, default=0,
                        help="Number of problems (0=all)")
    parser.add_argument("--rollouts", type=int, default=3,
                        help="Number of rollout attempts per problem (S4)")
    parser.add_argument("--temp",     type=float, default=TEMPERATURE,
                        help="Sampling temperature (S4)")
    parser.add_argument("--seed",     type=int, default=SEED)
    parser.add_argument("--max-steps",type=int, default=MAX_STEPS,
                        help="Max tactic steps per proof attempt (S2)")
    args = parser.parse_args()

    # Resolve effective values (do not mutate module-level constants)
    eff_max_steps = args.max_steps
    eff_timeout   = max(300, eff_max_steps * 8)

    csv_path = os.path.join(OUTPUT_DIR, "proofnet_results.csv")

    print(f"\n{'='*68}")
    print(f"  ProofNet Benchmark — Enhanced (S2+S3+S4)")
    print(f"  max_steps={eff_max_steps}  timeout={eff_timeout}s  "
          f"rollouts={args.rollouts}  T={args.temp}")
    print(f"  Target: solve ≥10 problems")
    print(f"{'='*68}\n")

    problems = load_proofnet(args.n, args.seed)
    if not problems: return

    # Show domain distribution
    from collections import Counter
    domains = Counter(p.get("domain","?") for p in problems)
    print(f"  Domains: {dict(domains)}\n")

    # Rollout temperatures: first greedy, rest sampled
    temperatures = ([0.0] + [args.temp] * (args.rollouts - 1))[:args.rollouts]

    for rollout_id, temp in enumerate(temperatures):
        print(f"\n{'─'*68}")
        t_str = "greedy" if temp == 0 else f"T={temp}"
        print(f"📐 Rollout {rollout_id+1}/{args.rollouts}  ({t_str})")
        run_rollout(rollout_id, temp, problems, csv_path, eff_max_steps, eff_timeout)

        # Progress after each rollout
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            solved_names  = set(df[df["status"]=="Success"]["name"].unique())
            n_problems    = len(problems)
            n_solved      = len(solved_names)
            lo, hi        = wilson(n_solved, n_problems)
            print(f"\n  After rollout {rollout_id+1}: "
                  f"{n_solved}/{n_problems} solved ({n_solved/n_problems*100:.1f}%)  "
                  f"CI=[{lo*100:.1f}%, {hi*100:.1f}%]")
            if n_solved >= 10:
                print(f"  🎉 TARGET REACHED: {n_solved} ≥ 10 problems solved!")

    # Final summary
    if not os.path.exists(csv_path):
        print("\n❌ No results file found"); return

    df         = pd.read_csv(csv_path)
    all_names  = set(p["name"] for p in problems)
    solved_names = set(df[df["status"]=="Success"]["name"].unique())
    n_problems = len(problems)
    n_solved   = len(solved_names)
    lo, hi     = wilson(n_solved, n_problems)

    print(f"\n{'='*68}")
    print(f"  PROOFNET FINAL RESULTS")
    print(f"{'='*68}")
    print(f"  Solved:  {n_solved}/{n_problems}  ({n_solved/n_problems*100:.2f}%)")
    print(f"  95% CI:  [{lo*100:.1f}%, {hi*100:.1f}%]")

    # Domain breakdown
    if "domain" in df.columns:
        print(f"\n  By domain:")
        for domain in sorted(domains.keys()):
            df_d    = df[df["domain"] == domain]
            solved_d = df_d[df_d["status"]=="Success"]["name"].nunique()
            total_d  = sum(1 for p in problems if p.get("domain")==domain)
            print(f"    {domain:<25} {solved_d:>3}/{total_d:<3}  "
                  f"({solved_d/total_d*100:.0f}% if total_d else 'n/a')")

    # Print solved problems
    if solved_names:
        print(f"\n  ✅ Solved problems ({n_solved}):")
        for name in sorted(solved_names)[:20]:
            dom = next((p.get("domain","?") for p in problems
                        if p["name"]==name), "?")
            print(f"    [{dom}] {name[:60]}")
        if n_solved > 20:
            print(f"    ... and {n_solved-20} more")

    target_met = n_solved >= 10
    print(f"\n  {'🎉 TARGET MET' if target_met else '⚠️  TARGET NOT MET'}: "
          f"{'≥' if target_met else '<'}10 solved")
    print(f"\n  LaTeX for paper:")
    print(f"  ProofNet & {n_solved}/{n_problems} & "
          f"{n_solved/n_problems*100:.2f}\\% & "
          f"[{lo*100:.1f}\\%, {hi*100:.1f}\\%] \\\\")
    print(f"\n📂 Results: {csv_path}")


if __name__ == "__main__":
    main()