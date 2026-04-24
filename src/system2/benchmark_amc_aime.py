# ==========================================
# Filename: src/system2/benchmark_amc_aime.py
# Function: Aggregate AMC/AIME problems from miniF2F and Compfiles for specialized testing
# Compatibility: lie_search.py v104.0
# ==========================================

import os
import sys
import time
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

# --- Environment Settings ---
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.setrecursionlimit(2000)

# --- Path Adaptation ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# --- Key Configurations ---
# Your model path
MODEL_PATH = "PROJECT_ROOT_PLACEHOLDER/models/DeepSeek-Prover-V1.5-RL"
CKPT_PATH = os.path.join(project_root, "data", "hgcn_final.pth")
DATA_DIR = os.path.join(project_root, "data")

# [Strategy] AMC/AIME are harder, use Relaxed mode + more steps
SEARCH_MODE = "relaxed"
MAX_STEPS = 40  # Provide deeper search depth
NUM_WORKERS = 4 # A100 parallelism count

# --- Core: Problem Loader ---
def load_amc_aime_problems(data_dir):
    problems = []
    
    # 1. Scan source: miniF2F (Primary source)
    minif2f_path = os.path.join(data_dir, "miniF2F")
    # 2. Scan source: Compfiles (High-quality source)
    compfiles_path = os.path.join(data_dir, "compfiles")
    
    files_to_scan = []
    if os.path.exists(minif2f_path):
        files_to_scan.extend(glob.glob(os.path.join(minif2f_path, "**", "*.lean"), recursive=True))
    if os.path.exists(compfiles_path):
        files_to_scan.extend(glob.glob(os.path.join(compfiles_path, "**", "*.lean"), recursive=True))
        
    print(f"🔍 Scanning {len(files_to_scan)} files for AMC/AIME problems...")
    
    # Regex: Match theorem, lemma, or the Compfiles-specific 'problem' keyword
    # Capture groups: 1=Type, 2=Name, 3=Args, 4=Target
    regex = re.compile(r'(?:@\[.*?\]\s*)?(theorem|lemma|problem|example)\s+([^\s\{]+)\s*(.*?)\s*:\s*(.*?)\s*:=', re.DOTALL)
    
    seen_names = set()
    
    for fpath in files_to_scan:
        # Exclude non-problem files
        if any(x in fpath for x in ["_manual", "lake-packages", "test/"]): continue
        
        try:
            with open(fpath, 'r', encoding='utf-8') as f: content = f.read()
            
            for match in regex.finditer(content):
                p_type, name, args, target = match.groups()
                name = name.strip()
                
                # 🎯 Core filtering: Only keep AMC or AIME
                name_lower = name.lower()
                is_target = "amc" in name_lower or "aime" in name_lower
                
                # Compfiles file paths usually contain the year/competition name, which can serve as supplementary criteria
                if not is_target and ("AMC" in fpath or "AIME" in fpath):
                    is_target = True

                if not is_target: continue
                if "helper" in name_lower: continue
                if name in seen_names: continue # De-duplication
                
                # Compfiles sometimes uses the 'problem' keyword; we need to convert it to standard Lean theorem format for the Prover to process
                # Simple handling: If it's a 'problem', we construct a theorem string
                decl = f"theorem {name} {args.strip()} : {target.strip()}"
                
                # Determine category
                category = "Other"
                if "aime" in name_lower: category = "AIME"
                elif "amc12" in name_lower: category = "AMC12"
                elif "amc10" in name_lower: category = "AMC10"
                elif "amc8" in name_lower: category = "AMC8"
                
                problems.append({
                    "name": name,
                    "decl": decl,
                    "category": category,
                    "source": "Compfiles" if "compfiles" in fpath else "miniF2F",
                    "path": fpath
                })
                seen_names.add(name)
        except Exception: continue

    # Sorting: AIME first, then AMC12
    problems.sort(key=lambda p: (p['category'] != 'AIME', p['category'] != 'AMC12'))
    
    print(f"✅ Found {len(problems)} Unique AMC/AIME Problems.")
    print(f"   - AIME: {len([p for p in problems if p['category']=='AIME'])}")
    print(f"   - AMC12: {len([p for p in problems if p['category']=='AMC12'])}")
    print(f"   - Other: {len(problems) - len([p for p in problems if p['category'] in ['AIME', 'AMC12']])}")
    return problems

# --- Worker & Stats (Reuse logic) ---

def worker_process(worker_id, problem_queue, result_queue, trace_dir):
    time.sleep(worker_id * 2)
    try:
        from src.system2.AMC_AIME_search import RiemannSearchAgent
        # Ensure Agent can find the checkpoint
        agent = RiemannSearchAgent(CKPT_PATH, MODEL_PATH, device="cuda")
    except Exception as e:
        print(f"❌ Worker-{worker_id} Init Fail: {e}")
        return

    while True:
        try:
            prob = problem_queue.get(timeout=3)
            if prob is None: break
        except: break

        try:
            # Start search
            cfg = {"mode": SEARCH_MODE}
            res = agent.search(prob['decl'], max_steps=MAX_STEPS, config=cfg)
            status = res.get('status', 'Error')
            
            # Save Trace (Success or long paths only)
            if status == "Success" or len(res.get('trace', [])) > 5:
                fname = f"{prob['source']}_{prob['name']}.pkl.gz"
                with gzip.open(os.path.join(trace_dir, fname), "wb") as f:
                    # Simplify experience for serialization
                    clean_res = res.copy()
                    if "experience" in clean_res:
                        clean_res["experience"] = [] # Reduce size; exp buffer is not needed for benchmark for now
                    pickle.dump(clean_res, f)
            
            # Send back results
            result_queue.put({
                "name": prob['name'],
                "category": prob['category'],
                "status": status,
                "steps": len(res.get('proof', [])),
                "time": 0.0, # Simplified
                "avg_radius": res.get('summary', {}).get('avg_radius', 0),
                "min_trust": min([t.get('trust_score', 0) for t in res.get('trace', [])] or [0])
            })
            
        except Exception as e:
            print(f"⚠️ Worker-{worker_id} Crash on {prob['name']}: {e}")
            result_queue.put({"name": prob['name'], "status": "Crash", "category": prob['category']})
        
        gc.collect()

class StatsTracker:
    def __init__(self, out_dir):
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(out_dir, f"AMC_AIME_Run_{self.ts}")
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join(self.dir, "traces"), exist_ok=True)
        self.csv_path = os.path.join(self.dir, "report.csv")
        self.records = []
        print(f"📂 Results will be saved to: {self.dir}")

    def log(self, res):
        self.records.append(res)
        pd.DataFrame(self.records).to_csv(self.csv_path, index=False)

    def report(self):
        df = pd.DataFrame(self.records)
        if df.empty: return
        print("\n" + "="*40)
        print("📊 AMC / AIME Benchmark Report")
        for cat in sorted(df['category'].unique()):
            sub = df[df['category'] == cat]
            succ = len(sub[sub['status'] == 'Success'])
            total = len(sub)
            print(f"   {cat.ljust(8)}: {succ}/{total} ({succ/total*100:.1f}%)")
        print("="*40)

# --- Main ---
def run():
    try: multiprocessing.set_start_method('spawn', force=True)
    except: pass
    
    # 1. Warm up Lean
    print("🔥 Warming up Lean Env...")
    from src.system2.lean_interaction import LeanEnv
    env = LeanEnv(project_root)
    env.run_command("import Mathlib", timeout=120)
    env.close()

    # 2. Load problems
    problems = load_amc_aime_problems(DATA_DIR)
    if not problems: return

    # 3. Prepare queues
    mgr = multiprocessing.Manager()
    q_in = mgr.Queue()
    q_out = mgr.Queue()
    
    for p in problems: q_in.put(p)
    for _ in range(NUM_WORKERS): q_in.put(None)
    
    stats = StatsTracker(os.path.join(project_root, "benchmark_reports"))
    
    # 4. Start Workers
    workers = []
    for i in range(NUM_WORKERS):
        p = multiprocessing.Process(target=worker_process, args=(i, q_in, q_out, os.path.join(stats.dir, "traces")))
        p.start()
        workers.append(p)
        
    # 5. Monitoring
    pbar = tqdm(total=len(problems))
    cnt = 0
    while cnt < len(problems):
        try:
            res = q_out.get(timeout=2)
            stats.log(res)
            pbar.update(1)
            cnt += 1
        except queue.Empty:
            if not any(w.is_alive() for w in workers): break
            
    pbar.close()
    for w in workers: w.terminate()
    stats.report()

if __name__ == "__main__":
    run()