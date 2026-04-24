# ==========================================
# Filename: src/system2/model.py
# Version: v105.3 (Single Model Eval: InternLM2 with 4 Workers)
# Compatibility: lie_search.py v104+ / llm_engine.py v7.3
# Functionality: 
#   1. Automatically iterate through models in MODEL_ZOO for evaluation
#   2. Process-level isolation to prevent VRAM pollution between different models
#   3. Automatically archive reports to benchmark_reports/<model_name>/
# ==========================================

import os
import sys
import time
import pandas as pd
import glob
import re
import pickle
import gzip
import torch
import multiprocessing
import traceback
import gc
import queue
from tqdm import tqdm
from datetime import datetime

# 1. Environment Setup
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 2. Path Adaptation
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 3. Core Config & Model Repository
DATA_DIR = os.path.join(project_root, "data")
CKPT_PATH = os.path.join(DATA_DIR, "hgcn_final.pth")
MODELS_ROOT = "PROJECT_ROOT_PLACEHOLDER/models"

# --- 🦁 Model Zoo: Define all models available ---
# [Note] Llemma-7B removed, keeping only downloaded models
MODEL_ZOO = {
    "DeepSeek-Prover-V1.5-RL":    f"{MODELS_ROOT}/DeepSeek-Prover-V1.5-RL",
    "InternLM2-StepProver":       f"{MODELS_ROOT}/InternLM2-StepProver",
    # "Llemma-7B":                 f"{MODELS_ROOT}/Llemma-7B",
    "Qwen2.5-Math-7B-Instruct":   f"{MODELS_ROOT}/Qwen2.5-Math-7B-Instruct",
    "Llama-3.1-8B-Instruct":      f"{MODELS_ROOT}/Llama-3.1-8B-Instruct",
    "DeepSeek-R1-Distill-Llama-8B": f"{MODELS_ROOT}/DeepSeek-R1-Distill-Llama-8B"
}

# --- 🎯 Current Run Plan ---
# Select models to run (Comment out models to skip)
MODELS_TO_RUN = [
    # "DeepSeek-Prover-V1.5-RL",    # Baseline (Already evaluated)
    "InternLM2-StepProver",       # Competitor 1 (Active)
    # "Llama-3.1-8B-Instruct",      # General Agent (Skipped)
    # "Qwen2.5-Math-7B-Instruct",   # Math SOTA (Skipped)
    # "DeepSeek-R1-Distill-Llama-8B" # Frontier (Skipped)
]

# --- ⚙️ Strategy Configuration ---
SEARCH_MODE = "relaxed" 
MAX_STEPS_PER_PROBLEM = 30 
NUM_WORKERS = 4   # Updated to 4 workers for InternLM2
BASE_OUTPUT_DIR = os.path.join(project_root, "benchmark_reports")
TARGET_NAMES = [] # Empty list = Run full Test set

# --- 🛠️ Helper Functions ---

def load_problems(data_dir, split="Test"):
    target_path = os.path.join(data_dir, "miniF2F/lean/src/test.lean")
    if not os.path.exists(target_path):
        print(f"🔍 Searching for .lean files in {data_dir}...")
        files = glob.glob(os.path.join(data_dir, "miniF2F", "**", "*.lean"), recursive=True)
    else:
        files = [target_path]
        
    problems = []
    regex = re.compile(r'(?:@\[.*?\]\s*)?(?:theorem|lemma|example)\s+([^\s\{]+)\s*(.*?)\s*:\s*(.*?)\s*:=', re.DOTALL)

    for fpath in files:
        if "_manual" in fpath or "lake-packages" in fpath: continue
        try:
            with open(fpath, 'r', encoding='utf-8') as f: content = f.read()
            category = "Test" if "test" in fpath.lower() else "Valid"
            if split.lower() not in category.lower() and len(files) > 1: continue

            for match in regex.finditer(content):
                name, args, target = match.groups()
                if "helper" in name.lower() or "import" in name: continue
                decl = f"theorem {name.strip()} {args.strip()} : {target.strip()}"
                problems.append({
                    "name": name.strip(),
                    "decl": decl,
                    "category": category,
                    "path": fpath
                })
        except: continue

    print(f"✅ Found {len(problems)} problems.")
    return problems

# --- 🏃 Worker Process Logic ---

def worker_process(worker_id, problem_queue, result_queue, trace_dir, model_path):
    """Worker Process: Execute search and save results directly"""
    time.sleep(worker_id * 5) # Staggered start
    
    print(f"🔧 Worker-{worker_id} initializing with model: {os.path.basename(model_path)}...")
    try:
        from src.system2.lie_search import RiemannSearchAgent
        # [Update] Pass specific model_path
        agent = RiemannSearchAgent(CKPT_PATH, model_path, device="cuda")
    except Exception as e:
        print(f"❌ Worker-{worker_id} init failed: {e}")
        traceback.print_exc()
        return

    while True:
        try:
            prob = problem_queue.get(timeout=5)
            if prob is None: break 
        except: break

        t0 = time.time()
        try:
            search_config = {
                "mode": SEARCH_MODE,
                "model_name": os.path.basename(model_path) # Log model name
            }
            result = agent.search(prob['decl'], max_steps=MAX_STEPS_PER_PROBLEM, config=search_config)
        except Exception as e:
            result = {"status": "ScriptCrash", "error": str(e), "trace": [], "experience": []}
        
        duration = time.time() - t0
        
        # [Decentralized Save]
        status = result.get('status', 'Unknown')
        
        if status == "Success" or len(result.get('trace', [])) > 5:
            save_path = os.path.join(trace_dir, f"{prob['name']}.pkl.gz")
            try:
                clean_res = result.copy()
                if "experience" in clean_res:
                    clean_res["experience"] = [
                        {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v for k, v in exp.items()}
                        for exp in clean_res["experience"]
                    ]
                with gzip.open(save_path, "wb") as f:
                    pickle.dump(clean_res, f)
            except Exception as e:
                print(f"⚠️ Worker-{worker_id} save trace failed: {e}")

        # Meta data
        meta_pack = {
            "name": prob['name'],
            "status": status,
            "duration": duration,
            "steps": len(result.get('proof', [])),
            "layer_stats": str(result.get('summary', {}).get('layer_counts', {})),
            "avg_radius": result.get('summary', {}).get('avg_radius', 0.0),
            "exp_count": len(result.get('experience', [])),
            "min_trust": min([t.get('trust_score', 0) for t in result.get('trace', [])] or [0])
        }
        
        result_queue.put(meta_pack)
        gc.collect()
        torch.cuda.empty_cache()

# --- 📊 Main Process Stats Logic ---

class BenchmarkStats:
    def __init__(self, base_dir, model_name):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # [Update] Directory structure: reports / ModelName / Timestamp
        self.output_dir = os.path.join(base_dir, model_name, self.timestamp)
        self.trace_dir = os.path.join(self.output_dir, "detailed_traces")
        self.csv_path = os.path.join(self.output_dir, "summary.csv")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.trace_dir, exist_ok=True)
        
        self.records = []
        self.done_names = set()
        
        print(f"📂 Report Directory: {self.output_dir}")

    def log_result(self, meta):
        status = meta['status']
        error_type = "None"
        
        if status != "Success":
            if "InitFail" in status: error_type = "InitFail"
            elif "EnvDeath" in status: error_type = "EnvDeath"
            elif "Crash" in status: error_type = "ScriptCrash"
            elif meta.get('min_trust', 0) < 0: error_type = "TrustGateReject"
            else: error_type = "SearchExhausted"

        entry = {
            "name": meta['name'],
            "status": status,
            "time": round(meta['duration'], 2),
            "steps": meta['steps'],
            "avg_radius": round(meta['avg_radius'], 4),
            "min_trust": round(meta.get('min_trust', 0), 4),
            "layer_stats": meta['layer_stats'],
            "error": error_type,
            "exp_count": meta['exp_count'],
            "timestamp": datetime.now().isoformat()
        }
        
        self.records.append(entry)
        df = pd.DataFrame(self.records)
        df.to_csv(self.csv_path, index=False)

    def generate_final_report(self, model_name):
        df = pd.DataFrame(self.records)
        if df.empty: return

        total = len(df)
        success = df[df['status'] == 'Success']
        pass_rate = (len(success) / total) * 100 if total > 0 else 0
        total_exp = df['exp_count'].sum()
        
        print("\n" + "="*50)
        print(f"📊 Final Report for [{model_name}]:")
        print(f"   Pass Rate: {pass_rate:.2f}% ({len(success)}/{total})")
        print(f"   Total Samples: {total}")
        print("="*50 + "\n")

# --- 🚀 Entry Point ---

def warmup_lean():
    print("-" * 40)
    print("🔥 Warming up Lean environment...")
    t0 = time.time()
    try:
        from src.system2.lean_interaction import LeanEnv
        env = LeanEnv(project_root)
        res = env.run_command("import Mathlib\nopen Nat Real Rat BigOperators Set Finset Function", timeout=600) 
        env.close()
    except Exception as e:
        print(f"   ⚠️ Warmup Warning: {e}")
    print(f"   ✅ Warmup finished in {time.time()-t0:.1f}s")
    print("-" * 40)

def run_single_benchmark(model_name, model_path):
    print(f"\n🚀 STARTING BENCHMARK: {model_name}")
    print(f"   Path: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model path not found for {model_name}! Skipping...")
        return

    all_probs = load_problems(DATA_DIR, split="Test")
    
    # Filter problems
    if TARGET_NAMES:
        target_probs = [p for p in all_probs if any(t in p['name'] for t in TARGET_NAMES)]
    else:
        # Default to algebra/number theory or full set
        target_probs = [p for p in all_probs if any(x in p['name'] for x in ["algebra", "imo", "numbertheory", "amc"])]
    
    if not target_probs:
        print("❌ No problems to run.")
        return

    # Initialize stats
    stats = BenchmarkStats(BASE_OUTPUT_DIR, model_name)
    
    # Prepare queue
    manager = multiprocessing.Manager()
    p_queue = manager.Queue()
    r_queue = manager.Queue()
    
    for p in target_probs: p_queue.put(p)
    for _ in range(NUM_WORKERS): p_queue.put(None) # Poison pills

    print(f"   Spawning {NUM_WORKERS} workers for {model_name}...")
    workers = []
    for i in range(NUM_WORKERS):
        p = multiprocessing.Process(
            target=worker_process, 
            args=(i, p_queue, r_queue, stats.trace_dir, model_path)
        )
        p.start()
        workers.append(p)

    # Monitor progress
    pbar = tqdm(total=len(target_probs), desc=f"Benchmarking {model_name}")
    finished = 0
    while finished < len(target_probs):
        try:
            res = r_queue.get(timeout=1.0)
            stats.log_result(res)
            pbar.update(1)
            finished += 1
        except queue.Empty:
            if not any(p.is_alive() for p in workers) and r_queue.empty():
                print("⚠️ All workers died unexpectedly.")
                break
            continue

    pbar.close()
    for p in workers: p.join()
    
    stats.generate_final_report(model_name)
    
    # [Cleanup]
    del p_queue, r_queue, manager, workers
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5) # Cooldown

def main():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass

    warmup_lean()

    print(f"📋 Planned Benchmarks: {MODELS_TO_RUN}")
    
    for model_name in MODELS_TO_RUN:
        if model_name not in MODEL_ZOO:
            print(f"⚠️ Warning: {model_name} not found in MODEL_ZOO. Skipping.")
            continue
            
        path = MODEL_ZOO[model_name]
        run_single_benchmark(model_name, path)

    print("\n🎉 ALL BENCHMARKS COMPLETED!")

if __name__ == "__main__":
    main()