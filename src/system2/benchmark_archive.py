# ==============================================================================
# Filename: src/system2/benchmark_archive.py
# Version: v200.0 (Mathlib Archive Edition)
# 
# Features:
#   1. [Target] Scans 'data/mathlib4/Archive' recursively.
#   2. [Categorization] Automatically groups results by folder (IMO, Putnam, etc.).
#   3. [Reporting] Outputs Pass@1 per category + Global Pass@1.
#   4. [Resume] Auto-resumes from summary.csv.
# ==============================================================================

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

# 1. Environment Settings
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 2. Path Adaptation
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 3. Core Configuration
# [Check] Ensure this points to your BEST model (InternLM2 or DeepSeek)
MODEL_ABSOLUTE_PATH = "PROJECT_ROOT_PLACEHOLDER/models/DeepSeek-Prover-V1.5-RL"
DATA_DIR = os.path.join(project_root, "data")
# [Check] Ensure this points to your standard hgcn checkpoint
CKPT_PATH = os.path.join(DATA_DIR, "hgcn_final.pth")

# --- ⚙️ Strategy Configuration ---
SEARCH_MODE = "relaxed" 
MAX_STEPS_PER_PROBLEM = 40  # Archive problems are harder, give them depth
NUM_WORKERS = 4 
OUTPUT_DIR = os.path.join(project_root, "benchmark_reports_archive")

# --- 🛑 Control Signals ---
STOP_SIGNAL_FILE = os.path.join(project_root, "STOP_BENCHMARK")

# --- ⏯️ Resume Configuration ---
RESUME_LATEST = True 

# --- 🛠️ Data Loading Logic (Archive Specific) ---

def load_archive_problems(data_dir):
    """
    Scans data/mathlib4/Archive and extracts problems with category labels.
    Category is determined by the immediate subfolder name (e.g., 'Imo', 'Putnam').
    """
    archive_root = os.path.join(data_dir, "mathlib4", "Archive")
    
    if not os.path.exists(archive_root):
        print(f"❌ Error: Archive directory not found at: {archive_root}")
        print("   Please ensure 'data/mathlib4/Archive' exists.")
        return []
    
    print(f"🔍 Scanning Archive at: {archive_root}")
    files = glob.glob(os.path.join(archive_root, "**", "*.lean"), recursive=True)
        
    problems = []
    # Regex to capture theorem declarations
    regex = re.compile(r'(?:@\[.*?\]\s*)?(?:theorem|lemma|example)\s+([^\s\{]+)\s*(.*?)\s*:\s*(.*?)\s*:=', re.DOTALL)

    for fpath in files:
        if "_manual" in fpath or "lake-packages" in fpath: continue
        
        # Extract Category (e.g., "Imo" from "Archive/Imo/...")
        rel_path = os.path.relpath(fpath, archive_root)
        category = rel_path.split(os.sep)[0]
        
        # Simplify filename for ID
        file_base = os.path.basename(fpath).replace(".lean", "")

        try:
            with open(fpath, 'r', encoding='utf-8') as f: content = f.read()

            for match in regex.finditer(content):
                name, args, target = match.groups()
                if "helper" in name.lower() or "import" in name: continue
                
                # Create a unique ID: Category_Filename_Theorem
                clean_name = name.strip().replace(".", "_")
                unique_id = f"{category}_{file_base}_{clean_name}"
                
                decl = f"theorem {name.strip()} {args.strip()} : {target.strip()}"
                
                problems.append({
                    "name": unique_id,
                    "decl": decl,
                    "category": category, # e.g., "Imo", "Putnam", "Mielnik"
                    "path": fpath
                })
        except: continue

    print(f"✅ Found {len(problems)} problems across {len(set(p['category'] for p in problems))} categories.")
    return problems

# --- 🏃 Worker Process ---

def worker_process(worker_id, problem_queue, result_queue, trace_dir):
    time.sleep(worker_id * 2) 
    
    if os.path.exists(STOP_SIGNAL_FILE): return

    try:
        # Using standard lie_search (NOT the ablation version)
        from src.system2.lie_search import RiemannSearchAgent
        agent = RiemannSearchAgent(CKPT_PATH, MODEL_ABSOLUTE_PATH, device="cuda")
    except Exception as e:
        print(f"❌ Worker-{worker_id} init failed: {e}")
        return

    while True:
        if os.path.exists(STOP_SIGNAL_FILE): break

        try:
            task_data = problem_queue.get(timeout=5)
            if task_data is None: break 
            prob = task_data
        except: break

        t0 = time.time()
        try:
            search_config = {"mode": SEARCH_MODE}
            result = agent.search(prob['decl'], max_steps=MAX_STEPS_PER_PROBLEM, config=search_config)
        except Exception as e:
            result = {"status": "ScriptCrash", "error": str(e), "trace": [], "experience": []}
        
        duration = time.time() - t0
        status = result.get('status', 'Unknown')
        
        # Save trace if successful
        if status == "Success":
            save_path = os.path.join(trace_dir, f"{prob['name']}.pkl.gz")
            try:
                with gzip.open(save_path, "wb") as f:
                    pickle.dump(result, f)
            except: pass

        meta_pack = {
            "name": prob['name'],
            "category": prob['category'], # Pass category through
            "status": status,
            "duration": duration,
            "steps": len(result.get('proof', [])),
            "avg_radius": result.get('summary', {}).get('avg_radius', 0.0),
            "error": result.get("error", "None") if status != "Success" else "None"
        }
        
        result_queue.put(meta_pack)
        gc.collect()
        torch.cuda.empty_cache()

# --- 📊 Statistics & Reporting ---

class BenchmarkStats:
    def __init__(self, base_output_dir, resume=False):
        self.base_dir = base_output_dir
        self.output_dir = self._determine_output_dir(resume)
        self.trace_dir = os.path.join(self.output_dir, "detailed_traces")
        self.csv_path = os.path.join(self.output_dir, "summary.csv")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.trace_dir, exist_ok=True)
        
        self.records = []
        self.done_names = set()
        
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                self.records = df.to_dict('records')
                self.done_names = set(df['name'].unique())
                print(f"🔄 Resuming from: {os.path.basename(self.output_dir)}")
                print(f"📊 Found {len(self.done_names)} completed problems.")
            except: pass
        else:
            print(f"🆕 New Benchmark: {os.path.basename(self.output_dir)}")

    def _determine_output_dir(self, resume):
        if not os.path.exists(self.base_dir):
            return os.path.join(self.base_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        if resume:
            subdirs = [os.path.join(self.base_dir, d) for d in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, d))]
            subdirs.sort(key=os.path.getmtime)
            if subdirs: return subdirs[-1]
        return os.path.join(self.base_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))

    def log_result(self, meta):
        # Normalize error field
        err = meta.get('error', 'None')
        if meta['status'] == 'ScriptCrash': err = 'ScriptCrash'
        elif meta['status'] != 'Success' and err == 'None': err = 'SearchExhausted'

        entry = {
            "name": meta['name'],
            "category": meta['category'], # Critical for grouping
            "status": meta['status'],
            "time": round(meta['duration'], 2),
            "steps": meta['steps'],
            "avg_radius": round(meta['avg_radius'], 4),
            "error": err,
            "timestamp": datetime.now().isoformat()
        }
        self.records.append(entry)
        df = pd.DataFrame(self.records)
        df.to_csv(self.csv_path, index=False)

    def generate_final_report(self):
        df = pd.DataFrame(self.records)
        if df.empty: return

        print("\n" + "="*60)
        print(f"📊 ARCHIVE BENCHMARK REPORT (Pass@1)")
        print("="*60)
        
        # 1. Global Stats
        total = len(df)
        success = len(df[df['status'] == 'Success'])
        global_rate = (success / total) * 100 if total > 0 else 0
        print(f"🌍 GLOBAL PASS RATE: {global_rate:.2f}% ({success}/{total})")
        print("-" * 60)
        
        # 2. Per-Category Stats
        if 'category' in df.columns:
            print(f"{'CATEGORY':<20} | {'PASS RATE':<15} | {'SOLVED/TOTAL':<15}")
            print("-" * 60)
            cats = df['category'].unique()
            for cat in sorted(cats):
                sub_df = df[df['category'] == cat]
                sub_total = len(sub_df)
                sub_succ = len(sub_df[sub_df['status'] == 'Success'])
                sub_rate = (sub_succ / sub_total) * 100 if sub_total > 0 else 0
                print(f"{cat:<20} | {sub_rate:.2f}%{' ':<9} | {sub_succ}/{sub_total}")
        print("="*60)

# --- 🚀 Main ---

def run_benchmark():
    try: multiprocessing.set_start_method('spawn', force=True)
    except: pass
    
    print(f"🚀 Starting Mathlib Archive Benchmark")
    if os.path.exists(STOP_SIGNAL_FILE):
        print(f"⚠️  Remove '{STOP_SIGNAL_FILE}' to start.")
        return

    # Check Environment
    if not os.path.exists(MODEL_ABSOLUTE_PATH):
        print(f"❌ Model not found: {MODEL_ABSOLUTE_PATH}")
        return

    # Load Data
    all_probs = load_archive_problems(DATA_DIR)
    if not all_probs: return

    # Init Stats
    stats = BenchmarkStats(OUTPUT_DIR, resume=RESUME_LATEST) 
    
    # Queue Management
    manager = multiprocessing.Manager()
    p_queue = manager.Queue()
    r_queue = manager.Queue()
    
    total_queued = 0
    for p in all_probs:
        if p['name'] not in stats.done_names:
            p_queue.put(p)
            total_queued += 1

    for _ in range(NUM_WORKERS): p_queue.put(None)

    if total_queued == 0:
        print("✅ All tasks completed.")
        stats.generate_final_report()
        return

    print(f"📝 Queued {total_queued} problems. Spawning {NUM_WORKERS} workers...")
    
    workers = []
    for i in range(NUM_WORKERS):
        p = multiprocessing.Process(target=worker_process, args=(i, p_queue, r_queue, stats.trace_dir))
        p.start()
        workers.append(p)

    # Monitoring Loop
    pbar = tqdm(total=total_queued)
    finished = 0
    
    while finished < total_queued:
        if os.path.exists(STOP_SIGNAL_FILE):
            pbar.set_description("🛑 STOPPING...")
            if not any(p.is_alive() for p in workers) and r_queue.empty(): break
        
        try:
            res = r_queue.get(timeout=1.0)
            stats.log_result(res)
            pbar.update(1)
            finished += 1
        except queue.Empty:
            if not any(p.is_alive() for p in workers) and r_queue.empty(): break
            continue

    pbar.close()
    for p in workers: p.join()
    
    stats.generate_final_report()
    print(f"\n📂 CSV Report: {stats.csv_path}")

if __name__ == "__main__":
    run_benchmark()