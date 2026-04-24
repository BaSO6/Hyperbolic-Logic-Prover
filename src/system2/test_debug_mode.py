# ==========================================
# Filename: src/system2/test_debug_mode.py
# Version: v15.3 (System 1 Monitor Edition)
# Functionality: 
#   1. [NEW] System 1 semantic alignment self-check
#   2. Deep debugging for 4 specific hard problems
#   3. Real-time display of System 1 retrieval results (synchronized with lie_search prints)
# ==========================================

import os
import sys
import time
import json
import torch
import traceback
import torch.multiprocessing as mp
import numpy as np

# --- 1. Path and Environment Configuration ---
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Force set HuggingFace mirror
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 2. Core Module Imports ---
try:
    from src.system2.lie_search import RiemannSearchAgent, GoalEncoder
    # Introduce hyperbolic geometry library for diagnosis
    from src.system1.manifold_math import PoincareBall
    from src.system2.lean_interaction import LeanEnv
    print("✅ Successfully imported core modules")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# --- 🎯 Core Configuration ---
REAL_MODEL_PATH = os.path.join(project_root, "models", "DeepSeek-Prover-V1.5-RL")
CKPT_PATH = os.path.join(project_root, "data", "hgcn_refined.pth")
TRACE_OUTPUT_PATH = os.path.join(project_root, "data", "debug_traces.jsonl")

# [CRITICAL] Force single process to facilitate real-time observation of System 1 output in the console
NUM_WORKERS = 1 

# --- 🧪 Specialized Debugging Targets ---
TARGET_PROBLEMS = [
    {
        "name": "imo_1959_p1",
        "decl": "theorem imo_1959_p1 (n : ℕ) (h₀ : 0 < n) : Nat.gcd (21*n + 4) (14*n + 3) = 1 := by",
        "desc": "Number Theory/GCD - Monitor whether gcd_sub or Euclidean algorithm is retrieved"
    },
    {
        "name": "aime_1983_p1",
        "decl": "theorem aime_1983_p1 (x y z w : ℕ) (ht : 1 < x ∧ 1 < y ∧ 1 < z) (hw : 0 ≤ w) (h0 : Real.log w / Real.log x = 24) (h1 : Real.log w / Real.log y = 40) (h2 : Real.log w / Real.log (x * y * z) = 12) : Real.log w / Real.log z = 60 := by",
        "desc": "Algebra/Logarithm - Monitor whether log_mul is retrieved"
    },
    {
        "name": "amc12b_2002_p4",
        "decl": "theorem amc12b_2002_p4 (n : ℕ) (h₀ : 0 < n) (h₁ : ((1 / 2 + 1 / 3 + 1 / 7 + 1 / n) : ℚ).den = 1) : n = 42 := by",
        "desc": "Rational Numbers - Tests Solver's handling of denominators"
    },
    {
        "name": "algebra_2varlineareq_fp3zeq11_3tfm1m5zeqn68_feqn10_zeq7",
        "decl": "theorem algebra_2varlineareq_fp3zeq11_3tfm1m5zeqn68_feqn10_zeq7 (f z : ℂ) (h₀ : f + 3*z = 11) (h₁ : 3*(f - 1) - 5*z = -68) : f = -10 ∧ z = 7 := by",
        "desc": "Complex Equations - Tests Solver's support for the complex field"
    }
]

# --- 🔍 System 1 Specialized Diagnosis (New Feature) ---

def test_system1_alignment():
    print("\n" + "="*50)
    print("🔬 [System 1 Diagnostic] Checking HGCN semantic alignment...")
    print("="*50)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        encoder = GoalEncoder(CKPT_PATH, device)
    except Exception as e:
        print(f"❌ System 1 load failed: {e}")
        return

    # Construct test pair: check if "log(xy)" and "log_mul" are close
    goal_text = "Real.log (x * y)"
    theorem_text = "log_mul" 
    
    emb_goal = encoder.encode(goal_text, mode="hyperbolic")
    emb_thm = encoder.encode(theorem_text, mode="hyperbolic")
    
    manifold = PoincareBall(c=encoder.c)
    dist = manifold.dist(emb_goal, emb_thm).item()
    
    print(f"    Test Case: Goal='{goal_text}' vs Theorem='{theorem_text}'")
    print(f"    Hyperbolic Distance: {dist:.4f}")
    
    if dist < 5.0:
        print("    ✅ Distance Judgment: Excellent (Strong System 1 correlation)")
    elif dist < 8.0:
        print("    ⚠️ Distance Judgment: Average (Multi-step retrieval may be needed)")
    else:
        print("    ❌ Distance Judgment: Poor (System 1 may produce hallucinations)")
    
    print("-" * 50 + "\n")

# --- 🛠️ Worker Logic ---

worker_agent = None

def init_worker(ckpt_path, model_path):
    global worker_agent
    try:
        print(f"    [Worker-{os.getpid()}] Loading Agent...")
        worker_agent = RiemannSearchAgent(ckpt_path, model_path, device="cuda")
    except Exception as e:
        print(f"    [Worker-{os.getpid()}] ❌ Load failed: {e}")
        traceback.print_exc()
        worker_agent = None

def solve_problem_wrapper(prob_data):
    global worker_agent
    if worker_agent is None:
        return {"name": prob_data['name'], "status": "SystemError", "trace": []}

    print(f"\n🚀 [Worker-{os.getpid()}] Proving: {prob_data['name']} ({prob_data['desc']})")
    t0 = time.time()
    
    try:
        # max_steps=30 gives System 2 sufficient fallback space
        result = worker_agent.search(prob_data['decl'], max_steps=30)
    except Exception as e:
        result = {"status": "ScriptCrash", "error": str(e), "trace": []}
        traceback.print_exc()
        
    duration = time.time() - t0
    print(f"📊 [Worker-{os.getpid()}] Done {prob_data['name']}: {result.get('status')} ({duration:.1f}s)")
    
    return {
        "name": prob_data['name'],
        "status": result.get("status"),
        "trace": result.get("trace", []),
        "proof": result.get("proof", [])
    }

# --- 🏁 Main Program ---

def warmup_lean():
    print("-" * 40)
    print("🔥 Warming up Lean environment...")
    t0 = time.time()
    try:
        env = LeanEnv(project_root)
        # Full warmup
        res = env.run_command("import Mathlib\nopen Nat Real Rat BigOperators Set Finset Function", timeout=120) 
        print(f"    Warmup Result: {str(res)[:50]}...")
        env.close()
    except Exception as e:
        print(f"    ⚠️ Warmup Warning: {e}")
    print(f"    ✅ Warmup finished in {time.time()-t0:.1f}s")
    print("-" * 40)

def run_debug_session():
    test_system1_alignment()
    warmup_lean()

    print(f"🔥 Starting System 2 Debug Session (Integrated Mode)")
    print(f"    Model:  {REAL_MODEL_PATH}")
    
    if os.path.exists(TRACE_OUTPUT_PATH):
        os.remove(TRACE_OUTPUT_PATH)

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    with mp.Pool(processes=NUM_WORKERS, initializer=init_worker, initargs=(CKPT_PATH, REAL_MODEL_PATH)) as pool:
        try:
            results = pool.map(solve_problem_wrapper, TARGET_PROBLEMS)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            return

    # Save logs
    print("\n💾 Saving Trace data...")
    success_count = 0
    with open(TRACE_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for res in results:
            if res['status'] == "Success":
                success_count += 1
            log_entry = {
                "goal_id": res['name'],
                "status": "success" if res['status'] == "Success" else "fail",
                "raw_trace": res['trace']
            }
            f.write(json.dumps(log_entry) + "\n")

    print("-" * 60)
    print(f"Total: {len(results)} | Success: {success_count}")
    print(f"✅ Trace saved to: {TRACE_OUTPUT_PATH}")
    print(f"👉 Pay close attention to [System 1] 🧠 Retrieved in console output")
    print("-" * 60)

if __name__ == "__main__":
    run_debug_session()