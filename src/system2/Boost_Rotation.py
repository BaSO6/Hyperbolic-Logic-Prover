# ==============================================================================
# 文件名: src/system2/Boost_Rotation.py
# 用途: "Physics of Logic" 消融实验主脚本
# 实验内容: 
#   1. Variant F: No-Rotation (Frozen Semantics, S=0) ONLY
#   (Variant E: No-Boost 已完成测试，结果符合预期)
# 依赖: src/system2/lie_search_divide.py (v108.0)
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

# 1. 环境设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 2. 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# 3. 核心配置
# [CRITICAL FIX] 必须使用 InternLM2 以匹配 Baseline (63.01%)
MODEL_ABSOLUTE_PATH = "PROJECT_ROOT_PLACEHOLDER/models/InternLM2-StepProver"
DATA_DIR = os.path.join(project_root, "data")
CKPT_PATH = os.path.join(DATA_DIR, "hgcn_final.pth")

# --- ⚙️ 消融实验配置 ---
# [MODIFIED] 只运行 No-Rotation，节省时间
ABLATION_MODES = ["no_rotation"] 

SEARCH_MODE = "relaxed" 
MAX_STEPS_PER_PROBLEM = 30 
NUM_WORKERS = 4   # A100 Safe Limit
BASE_OUTPUT_DIR = os.path.join(project_root, "benchmark_reports")
TARGET_NAMES = [] # 空列表 = 跑全量

# --- 🛠️ 辅助函数 (保持不变) ---

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

# --- 🏃 工作进程逻辑 (注入消融参数) ---

def worker_process(worker_id, problem_queue, result_queue, trace_dir, ablation_mode):
    """Worker 进程: 执行带有消融参数的搜索"""
    time.sleep(worker_id * 5) 
    
    print(f"🔧 Worker-{worker_id} initializing [Mode: {ablation_mode}]...")
    try:
        # [CRITICAL] 确保导入的是 v108.0 修复版 lie_search_divide
        from src.system2.lie_search_divide import RiemannSearchAgent
        agent = RiemannSearchAgent(CKPT_PATH, MODEL_ABSOLUTE_PATH, device="cuda")
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
            # 注入 ablation 模式
            search_config = {
                "mode": SEARCH_MODE,
                "ablation": ablation_mode 
            }
            result = agent.search(prob['decl'], max_steps=MAX_STEPS_PER_PROBLEM, config=search_config)
        except Exception as e:
            result = {"status": "ScriptCrash", "error": str(e), "trace": [], "experience": []}
        
        duration = time.time() - t0
        status = result.get('status', 'Unknown')
        
        # Save Trace (Success or Long Failures)
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
            "min_trust": min([t.get('trust_score', 0) for t in result.get('trace', [])] or [0]),
            "ablation": ablation_mode
        }
        
        result_queue.put(meta_pack)
        gc.collect()
        torch.cuda.empty_cache()

# --- 📊 统计逻辑 (支持子目录) ---

class BenchmarkStats:
    def __init__(self, base_dir, ablation_mode):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # [Output] benchmark_reports/ablation_no_rotation/Timestamp
        self.output_dir = os.path.join(base_dir, f"ablation_{ablation_mode}", self.timestamp)
        self.trace_dir = os.path.join(self.output_dir, "detailed_traces")
        self.csv_path = os.path.join(self.output_dir, "summary.csv")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.trace_dir, exist_ok=True)
        
        self.records = []
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
            "ablation": meta['ablation'],
            "timestamp": datetime.now().isoformat()
        }
        
        self.records.append(entry)
        df = pd.DataFrame(self.records)
        df.to_csv(self.csv_path, index=False)

    def generate_final_report(self, ablation_mode):
        df = pd.DataFrame(self.records)
        if df.empty: return

        total = len(df)
        success = df[df['status'] == 'Success']
        pass_rate = (len(success) / total) * 100 if total > 0 else 0
        
        print("\n" + "="*50)
        print(f"📊 Final Report [Ablation: {ablation_mode}]:")
        print(f"   Pass Rate: {pass_rate:.2f}% ({len(success)}/{total})")
        print("="*50 + "\n")

# --- 🚀 启动入口 ---

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

def run_single_ablation(ablation_mode):
    print(f"\n🧪 STARTING ABLATION EXPERIMENT: {ablation_mode}")
    print(f"   Model: {MODEL_ABSOLUTE_PATH}")

    all_probs = load_problems(DATA_DIR, split="Test")
    
    if TARGET_NAMES:
        target_probs = [p for p in all_probs if any(t in p['name'] for t in TARGET_NAMES)]
    else:
        target_probs = [p for p in all_probs if any(x in p['name'] for x in ["algebra", "imo", "numbertheory", "amc"])]
    
    if not target_probs:
        print("❌ No problems to run.")
        return

    stats = BenchmarkStats(BASE_OUTPUT_DIR, ablation_mode)
    manager = multiprocessing.Manager()
    p_queue = manager.Queue()
    r_queue = manager.Queue()
    
    for p in target_probs: p_queue.put(p)
    for _ in range(NUM_WORKERS): p_queue.put(None) 

    print(f"   Spawning {NUM_WORKERS} workers...")
    workers = []
    for i in range(NUM_WORKERS):
        p = multiprocessing.Process(
            target=worker_process, 
            args=(i, p_queue, r_queue, stats.trace_dir, ablation_mode)
        )
        p.start()
        workers.append(p)

    pbar = tqdm(total=len(target_probs), desc=f"Ablation: {ablation_mode}")
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
    
    stats.generate_final_report(ablation_mode)
    
    del p_queue, r_queue, manager, workers
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)

def main():
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError: pass

    warmup_lean()

    print(f"📋 Planned Ablations: {ABLATION_MODES}")
    
    for mode in ABLATION_MODES:
        run_single_ablation(mode)

    print("\n🎉 ALL ABLATION EXPERIMENTS COMPLETED!")

if __name__ == "__main__":
    main()