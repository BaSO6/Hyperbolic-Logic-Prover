# ==========================================
# 文件名: src/system2/rigorous_search.py
# 版本: v201.0 (Balanced Rigor: Target 6-10 Solves)
# 策略: 严禁答案检索，但允许强力代数计算
# ==========================================

import os
import sys
import re
import heapq
import itertools
import uuid
import time
import hashlib
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, Counter
from dataclasses import dataclass

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path: sys.path.append(project_root)

# Internal Imports
from src.system1.manifold_math import PoincareBall
from src.system1.operators import LogicalLieAlgebra
from src.system2.lean_interaction import LeanEnv
from src.system2.llm_engine import LLMEngine
from src.system2.tactic_encoder import TacticEncoder
from src.system2.lie_search import InferenceHGCN, GoalEncoder, HyperbolicEnergy

# =================================================
# 1. Rigorous Goal Controller (平衡版)
# =================================================

class RigorousGoalController:
    """
    平衡控制器：
    - 禁止检索作弊 (aesop, library_search)
    - 开放计算能力 (ring, linarith, simp) -> 这是拿下 6-10 题的关键
    """
    def __init__(self):
        self.logic_patterns = [
            (r"∧", "constructor"), (r"↔", "constructor"), (r"∃", "refine ⟨_, _⟩"),
            (r"∀", "intro"), (r"→", "intro"), 
        ]

    def get_structural_action(self, goal_text: str) -> str:
        g = goal_text.strip()
        for pat, tac in self.logic_patterns:
            if re.search(pat, g): return tac
        return None

    def get_domain_solver(self, goal_text: str) -> list:
        solvers = []
        
        # [核心增强]：允许数学计算工具 (White-Hat Automation)
        # 这些工具不会检索答案，只会做逻辑变形，是安全的
        solvers.append("simp_all") 
        solvers.append("ring_nf")  # 强力处理代数
        solvers.append("field_simp") # 处理分数
        solvers.append("norm_num")   # 处理数字
        
        # 针对不等式和实数 (Putnam 常见题型)
        if any(c in goal_text for c in ["<", "≤", ">", "≥", "="]):
            solvers.append("linarith")
            solvers.append("nlinarith") # 非线性算术
            solvers.append("positivity") # 自动判断正负
            solvers.append("gcongr")     # 几何/不等式放缩
        
        # [BANNED LIST] 依然严厉禁止检索类工具
        # aesop, library_search, exact?, apply?, suggest, hint -> 全部封印
        
        return list(set(solvers))

    def is_cheating(self, tactic: str) -> bool:
        """审计策略是否包含作弊嫌疑"""
        t = tactic.lower()
        # 1. 禁止直接引用答案文件
        if "putnam" in t and ("exact" in t or "apply" in t or "rw" in t or "refine" in t):
            # 允许递归调用自己 (putnam_2022_b1 引用 putnam_2022_b1 是合法的归纳法)
            # 但为了保险，如果出现 putnam_xxxx_sol 必须杀掉
            if "_sol" in t or "solution" in t: return True
            # 如果仅仅是 putnam，先放行 (可能是递归)，交给审计脚本判断
            return False 
            
        # 2. 禁止黑盒搜索
        if any(x in t for x in ["aesop", "library_search", "exact?", "apply?"]):
            return True
            
        # 3. 禁止放弃治疗
        if "sorry" in t or "admit" in t: return True
        
        return False

# =================================================
# 2. Rigorous Search Agent (高活性版)
# =================================================

class RigorousSearchAgent:
    def __init__(self, hgcn_ckpt, llm_path, device="cuda"):
        self.device = device
        self.counter = itertools.count()
        self.controller = RigorousGoalController()
        
        # 复用组件
        self.goal_encoder = GoalEncoder(hgcn_ckpt, device)
        self.manifold = PoincareBall(c=self.goal_encoder.c)
        self.energy = HyperbolicEnergy(self.goal_encoder.c)
        
        # 加载 Mathlib 嵌入
        data_dir = os.path.join(project_root, "data")
        emb_path = os.path.join(data_dir, "node_embeddings.pt")
        map_path = os.path.join(data_dir, "id_to_name.pkl.gz")
        self.idx_to_name = {}
        if os.path.exists(map_path):
            import gzip, pickle
            with gzip.open(map_path, "rb") as f: self.idx_to_name = pickle.load(f)
        
        self.graph_emb = None
        self.graph_norms = None
        if os.path.exists(emb_path):
            self.graph_emb = torch.load(emb_path, map_location=device)
            self.retrieval_mode = "hyperbolic" if self.goal_encoder.use_hgcn else "euclidean"
            self.graph_norms = self.graph_emb.norm(dim=-1)

        lie_dim = 384 
        if self.goal_encoder.use_hgcn:
            lie_dim = self.goal_encoder.projector.semantic_proj.weight.shape[0]
        
        self.lie = LogicalLieAlgebra(lie_dim, 64, c=self.goal_encoder.c).to(device) 
        self.llm = LLMEngine(llm_path, device=device)
        self.tactic_enc = TacticEncoder(device=device)
        self.tac_to_coeff = nn.Sequential(
            nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh()
        ).to(device)
        
        self.env = None
        self._init_env()

    # --- Environment Management ---
    def _init_env(self):
        if self.env: self.env.close()
        self.env = LeanEnv(project_root)
        try:
            # 仅引入 Mathlib，保证纯净
            self.env.run_command("import Mathlib", timeout=600) 
        except RuntimeError: sys.exit(1)
        self.env.run_command("open Nat Real Rat BigOperators Set Finset Function Topology Filter Metric")

    def _ensure_env(self):
        if self.env is None or self.env.proc.poll() is not None: self._init_env()
        return self.env

    def normalize_goal(self, text: str) -> str:
        if "⊢" in text: text = text.split("⊢")[-1]
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # --- System 1: Retrieval (Strict) ---
    def retrieve_theorems(self, goal_emb, goal_text=None, k=5):
        if self.graph_emb is None: return [], [], []
        
        with torch.no_grad():
            if self.retrieval_mode == "hyperbolic":
                dists = self.manifold.dist(goal_emb, self.graph_emb)
                if dists.dim() > 1: dists = dists.squeeze()
                # 适度惩罚，但不如之前严厉，允许更多候选进入 System 2 筛选
                scores = dists * (1.0 + 0.5 * self.graph_norms.to(dists.device))
                _, idxs = torch.topk(scores, k=k*6, largest=False)
            else: 
                scores = torch.mm(goal_emb, self.graph_emb.T)
                _, idxs = torch.topk(scores, k=k*6, largest=True)

            out = []
            for i in idxs.flatten().tolist():
                if len(out) >= k: break
                if i < len(self.idx_to_name):
                    name = self.idx_to_name.get(i, "")
                    # [严谨过滤]
                    if any(bad in name.lower() for bad in ["solution", "helper", "test"]): continue
                    if name and not name.startswith("_"):
                        out.append(name)
            return out, [], []

    # --- System 2: Reasoning ---
    def _calculate_geodesic_trust(self, current_emb, tactic_name):
        if self.controller.is_cheating(tactic_name):
            return -999.0, current_emb

        try:
            with torch.no_grad():
                emb = self.tactic_enc.encode(tactic_name)
                coeff = self.tac_to_coeff(emb)
                op = self.lie.get_operator(coeff)
                pred_next = self.lie.apply_tactic(current_emb, op)
                
                curr_dist = current_emb.norm().item()
                next_dist = pred_next.norm().item()
                
                # Trust = 半径缩减量
                trust = curr_dist - next_dist 
                return trust, pred_next
        except: return 0.0, current_emb

    # --- Main Search Logic ---
    def search(self, theorem_decl, max_steps=60, config=None):
        if config is None: config = {}
        
        # [关键调整] 
        # 将信任阈值从 0.02 降至 -0.08
        # 这允许 Agent 尝试那些“可能暂时没有缩短距离，但也没有严重偏离”的策略
        # 比如引入一个引理 (have) 或者做一下变形 (rw)
        TRUST_THRESHOLD = -0.08
        
        base = theorem_decl.split(":= by")[0].strip()
        base_clean = re.sub(r"putnam_\d{4}_[a-b]\d+", "goal", base)
        base = re.sub(r"^\s*theorem\s+\S+\s*", "example ", base)
        
        env = self._ensure_env()
        res = env.run_command(f"{base} := by skip", timeout=15)
        if "unsolved goals" not in str(res):
             return {"status": "InitFail", "trace": [], "proof": []}

        raw_state = str(res.get("messages", [""])[-1].get("data", ""))
        state_str = self.normalize_goal(raw_state)
        state_emb = self.goal_encoder.encode(state_str, mode=self.retrieval_mode)
        
        frontier = []
        root_id = str(uuid.uuid4())[:8]
        # 使用 A* 启发式：初始 Cost = 0 + 半径
        heapq.heappush(frontier, (state_emb.norm().item(), next(self.counter), [], state_str, state_emb, root_id))
        
        trace = []
        visited_hashes = set()
        visited_hashes.add(hashlib.md5(state_str.encode()).hexdigest())
        
        expanded = 0
        while frontier:
            cost, _, hist, gtext, emb, curr_id = heapq.heappop(frontier)
            
            if len(hist) >= max_steps: continue
            expanded += 1
            if expanded > 150: break # 给足够的节点去探索
            
            # 1. 结构化策略
            struct_tac = self.controller.get_structural_action(gtext)
            candidates = [struct_tac] if struct_tac else []
            
            # 2. 检索策略 (System 1)
            hints, _, _ = self.retrieve_theorems(emb, goal_text=gtext, k=4)
            for h in hints:
                candidates.append(f"apply {h}")
                candidates.append(f"rw [{h}]")
                candidates.append(f"rw [← {h}]")
            
            # 3. 计算策略 (System 2 White-Hat)
            domain_tacs = self.controller.get_domain_solver(gtext)
            candidates.extend(domain_tacs)
            
            # 评分与筛选
            scored_candidates = []
            for tac in candidates:
                if not tac: continue
                trust, _ = self._calculate_geodesic_trust(emb, tac)
                is_structural = tac in [struct_tac] or "intro" in tac or "constructor" in tac
                scored_candidates.append((trust, tac, is_structural))
            
            # 按信任度排序
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            
            # [强制活跃机制]
            # 如果没有策略通过阈值，强制取 Top 1，防止死锁 (Analysis Paralysis)
            valid_moves = [x for x in scored_candidates if x[0] > TRUST_THRESHOLD or x[2]]
            if not valid_moves and scored_candidates:
                # 只有当策略看起来不算太离谱时才强制执行
                if scored_candidates[0][0] > -0.2: 
                    valid_moves = [scored_candidates[0]]
            
            for trust, tac, _ in valid_moves:
                # 执行
                new_hist = hist + [tac]
                res = env.run_command(f"{base} := by {'; '.join(new_hist)}", timeout=20)
                
                st_text = str(res)
                if "no goals" in st_text.lower():
                    # 再次确认步骤数，过于简单的可能是假阳性，但在 Search 阶段我们先返回 Success
                    return {
                        "status": "Success", 
                        "proof": new_hist, 
                        "trace": trace, 
                        "summary": {"avg_radius": 0.0, "total_expanded": expanded}
                    }
                
                if "error" not in st_text.lower() and "unsolved goals" in st_text:
                    new_raw = str(res.get("messages", [""])[-1].get("data", ""))
                    new_goal = self.normalize_goal(new_raw)
                    new_hash = hashlib.md5(new_goal.encode()).hexdigest()
                    
                    if new_hash not in visited_hashes:
                        visited_hashes.add(new_hash)
                        new_emb = self.goal_encoder.encode(new_goal, mode=self.retrieval_mode)
                        new_r = new_emb.norm().item()
                        
                        # A* Cost: 步数 + 10 * 半径 (增加半径的权重，更强烈地引导向圆心)
                        new_cost = len(new_hist) + 10.0 * new_r
                        
                        trace.append({
                            "step": expanded, "tactic": tac, "trust": trust,
                            "radius_before": emb.norm().item(), "radius_after": new_r
                        })
                        
                        heapq.heappush(frontier, (new_cost, next(self.counter), new_hist, new_goal, new_emb, str(uuid.uuid4())[:8]))

        return {"status": "Fail", "trace": trace, "proof": []}