# ==========================================
# 文件名: src/system2/AMC_AIME_search.py
# 版本: v105.0 (AMC/AIME Optimized)
# 基于: lie_search.py v104
# 核心升级:
#   1. [Anti-Hallucination] 增加幻觉检测，遇到 "unknown identifier" 立即熔断
#   2. [Algebra-First] 针对 AMC 强化代数策略，限制微积分策略滥用
#   3. [Aesop] 引入 aesop 自动化策略作为兜底
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
from sentence_transformers import SentenceTransformer

# =================================================
# Layer 1 & 2: Goal Shape Controller (AMC Optimized)
# =================================================

class GoalShapeController:
    """
    AMC/AIME 专用控制器：
    严格区分代数域和分析域，防止模型在代数题上尝试微积分。
    """
    def __init__(self):
        # Layer 1: Logical Structure
        self.logic_patterns = [
            (r"∧", "constructor"), (r"↔", "constructor"), (r"∃", "refine ⟨_, _⟩"),
        ]
        # Layer 2: Domain Solvers (AMC Focused)
        self.arithmetic_patterns = [
            (r"Nat", "omega"), (r"Int", "omega"), 
            (r"Real", "linarith"), (r"ℚ", "linarith"),
            (r"∑", "simp [Finset.sum]"), (r"mod", "omega"), (r"dvd", "omega"),
        ]

    def get_structural_action(self, goal_text: str) -> str:
        g = goal_text.strip()
        for pat, tac in self.logic_patterns:
            if re.search(pat, g): return tac
        return None

    def get_domain_solver(self, goal_text: str) -> list:
        solvers = []
        is_rel = any(c in goal_text for c in ["=", "<", "≤", ">", "≥"])
        
        # [Universal Solvers]
        solvers.append("simp")
        solvers.append("ring_nf") # 核心代数化简
        solvers.append("field_simp") # 处理分数/有理数的神器
        solvers.append("aesop") # Lean 4 的自动化大锤 (新加入)
        
        # [Algebra / Number Theory]
        # 只要有不等式或等式，就尝试强力算术策略
        if is_rel:
            solvers.append("linarith")
            if "Real" in goal_text or "ℝ" in goal_text:
                solvers.append("nlinarith") # 非线性算术 (处理平方等)
                solvers.append("gcongr")    # 几何/不等式放缩 (Mathlib 新神器)
            
            if "Complex" in goal_text or "ℂ" in goal_text: 
                solvers.append("linear_combination")
                solvers.append("ring")

        # [Anti-Hallucination Filter]
        # 只有明确出现积分符号时，才允许微积分策略
        # 你的日志显示它在代数题上乱用 integral_inv_of_pos，这里进行了屏蔽
        if "∫" in goal_text or "deriv" in goal_text or "Integral" in goal_text:
            solvers.append("apply integral_inv_of_pos") # 仅在需要时开启
            solvers.append("simp [FundamentalTheoremOfCalculus]")

        return list(set(solvers))

# =================================================
# System-1 Models (HGCN Integration)
# =================================================

class InferenceHGCN(nn.Module):
    def __init__(self, in_dim, out_dim, c):
        super().__init__()
        self.manifold = PoincareBall(c)
        self.semantic_proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        z = self.semantic_proj(x)
        # AMC 题目通常更复杂，允许更大的半径变化
        return self.manifold.expmap0(z)

class GoalEncoder:
    def __init__(self, hgcn_ckpt, device):
        self.device = device
        st_path = os.path.join(project_root, "models", "all-MiniLM-L6-v2")
        self.bert = SentenceTransformer(st_path, device=device)
        self.bert.eval()
        self.c = 1.0
        self.use_hgcn = False
        self.projector = None
        self.hgcn_ckpt_path = hgcn_ckpt
        
        if os.path.exists(hgcn_ckpt):
            try:
                ckpt = torch.load(hgcn_ckpt, map_location=device)
                w = None
                prefix = ""
                if "layer.semantic_proj.weight" in ckpt["model"]:
                    w = ckpt["model"]["layer.semantic_proj.weight"]
                    prefix = "layer."
                elif "semantic_proj.weight" in ckpt["model"]:
                    w = ckpt["model"]["semantic_proj.weight"]
                    prefix = ""
                else: w = torch.zeros(64, 384) 
                
                out_dim = w.shape[0]
                self.c = ckpt.get("c", 1.0)
                self.projector = InferenceHGCN(384, out_dim, self.c).to(device)
                
                state_dict = {}
                for k, v in ckpt["model"].items():
                    k_clean = k.replace(prefix, "") if prefix else k
                    if k_clean in self.projector.state_dict():
                        state_dict[k_clean] = v
                
                self.projector.load_state_dict(state_dict, strict=False)
                self.projector.eval()
                self.use_hgcn = True
            except Exception: pass

    def encode(self, text, mode="euclidean"):
        if not text:
            dim = 384
            if self.use_hgcn and mode == "hyperbolic":
                dim = self.projector.semantic_proj.weight.shape[0]
            return torch.zeros(1, dim, device=self.device)

        with torch.no_grad():
            emb = self.bert.encode(text, convert_to_tensor=True, show_progress_bar=False)
            emb = F.normalize(emb.unsqueeze(0), dim=-1)
            if self.use_hgcn and mode == "hyperbolic":
                return self.projector(emb)
            return emb * 0.9

class HyperbolicEnergy:
    def __init__(self, c):
        self.manifold = PoincareBall(c)
    def h(self, a, b):
        d = self.manifold.dist(a, b)
        return d.item() if isinstance(d, torch.Tensor) else float(d)

# =================================================
# Search Agent (With Hallucination Guard)
# =================================================

@dataclass
class GoalStruct:
    raw: str; is_searchable: bool; kind: str; domain: str

class GoalAnalyzer:
    def __init__(self): self.ban_patterns = [r"^\d+\s*=\s*\d+$"]
    def analyze(self, text):
        domain = "unknown"
        if "Nat" in text: domain = "nat"
        elif "Real" in text: domain = "real"
        elif "Complex" in text: domain = "complex"
        return GoalStruct(text, True, "unknown", domain)

class RiemannSearchAgent:
    def __init__(self, hgcn_ckpt, llm_path, device="cuda"):
        self.device = device
        self.counter = itertools.count()
        self.controller = GoalShapeController()
        self.analyzer = GoalAnalyzer()
        self.goal_encoder = GoalEncoder(hgcn_ckpt, device)
        self.manifold = PoincareBall(c=self.goal_encoder.c)
        self.energy = HyperbolicEnergy(self.goal_encoder.c)
        
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
            print(f"   [Memory] Loaded {len(self.graph_emb)} nodes.")

        lie_dim = 384 
        if self.goal_encoder.use_hgcn:
            lie_dim = self.goal_encoder.projector.semantic_proj.weight.shape[0]
        
        self.lie = LogicalLieAlgebra(lie_dim, 64, c=self.goal_encoder.c).to(device) 
        self.llm = LLMEngine(llm_path, device=device)
        self.tactic_enc = TacticEncoder(device=device)
        self.tac_to_coeff = nn.Sequential(
            nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh()
        ).to(device)
        self.optimizer = torch.optim.SGD(self.tac_to_coeff.parameters(), lr=0.01, momentum=0.9)
        self.state_visits = defaultdict(int)
        self.env = None
        self._init_env()

    # --- Robust State Predicates ---
    def _is_proof_state(self, res) -> bool:
        if not res: return False
        if isinstance(res, dict):
            msgs = res.get("messages", [])
            for m in msgs:
                if "unsolved goals" in str(m.get("data", "")) or "⊢" in str(m.get("data", "")): return True
        return "unsolved goals" in str(res) or "⊢" in str(res)

    def _is_env_failure(self, res) -> bool:
        text = str(res).lower()
        return res is None or "panic" in text or "segmentation" in text or "crash" in text

    # [NEW] 幻觉检测器
    def _is_hallucination(self, error_msg: str) -> bool:
        """检测 Lean 是否报告了未知标识符"""
        text = error_msg.lower()
        return "unknown identifier" in text or "function expected" in text or "failed to synthesize" in text

    def _init_env(self):
        if self.env: self.env.close()
        self.env = LeanEnv(project_root)
        try:
            # 引入更多的库以支持 AMC
            self.env.run_command("import Mathlib", timeout=600)
        except RuntimeError: sys.exit(1)
        # 预先打开常用命名空间
        self.env.run_command("open Nat Real Rat BigOperators Set Finset Function Topology")

    def _ensure_env(self):
        if self.env is None or self.env.proc.poll() is not None: self._init_env()
        return self.env

    def __del__(self):
        if hasattr(self, 'env') and self.env: self.env.close()

    def _get_state_hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:8]

    def normalize_goal(self, text: str) -> str:
        if "⊢" in text: text = text.split("⊢")[-1]
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # --- Mechanics ---
    def retrieve_theorems(self, goal_emb, goal_text=None, k=5):
        if self.graph_emb is None: return [], [], []
        
        forbidden_keywords = []
        if goal_text:
            if any(w in goal_text for w in ["Nat", "Int", "gcd", "dvd", "mod"]):
                forbidden_keywords = ["Measure", "Topology", "Manifold", "Lie", "Integral"]
            elif "Geometry" in goal_text:
                forbidden_keywords = ["Category", "Probability"]
        
        with torch.no_grad():
            if self.retrieval_mode == "hyperbolic":
                goal_r = goal_emb.norm().item()
                dists = self.manifold.dist(goal_emb, self.graph_emb)
                if dists.dim() > 1: dists = dists.squeeze()
                
                alpha = 0.0
                if goal_r > 0.7: alpha = 1.0 
                
                scores = dists * (1.0 + alpha * self.graph_norms.to(dists.device))
                _, idxs = torch.topk(scores, k=k*4, largest=False)
            else: 
                scores = torch.mm(goal_emb, self.graph_emb.T)
                _, idxs = torch.topk(scores, k=k*4, largest=True)

            out = []
            radii = []
            ranks = [] 
            raw_rank = 0
            for i in idxs.flatten().tolist():
                raw_rank += 1
                if len(out) >= k: break
                if i < len(self.idx_to_name):
                    name = self.idx_to_name.get(i, "")
                    if any(ban in name for ban in forbidden_keywords): continue
                    if name and not name.startswith("_") and "test" not in name:
                        out.append(name)
                        ranks.append(raw_rank)
                        if self.retrieval_mode == "hyperbolic":
                            radii.append(self.graph_norms[i].item())
            return out, radii, ranks

    def _generate_smart_heuristics(self, hints: list, goal_struct) -> list:
        tactics = []
        for i, h in enumerate(hints[:3]):
            tactics.append(f"rw [{h}]")
            tactics.append(f"rw [← {h}]")
            tactics.append(f"apply {h}")
            if i == 0: tactics.append(f"refine {h} _") 
        return list(set(tactics))

    def tactic_operator(self, tactic):
        with torch.no_grad():
            emb = self.tactic_enc.encode(tactic)
            coeff = self.tac_to_coeff(emb)
            return self.lie.get_operator(coeff)

    def parse_result(self, res):
        if not res: return "error", ""
        text = str(res)
        if isinstance(res, dict):
            msgs = res.get("messages", [])
            text = "\n".join(str(m.get("data", "")) for m in msgs)
            if "env" in res and not msgs: return "success", ""
        low = text.lower()
        if "no goals" in low: return "success", ""
        if "error" in low and "unsolved goals" not in low: return "error", text
        if "⊢" in text: return "continue", text.split("⊢")[-1].strip()
        return "continue", text

    def _calculate_geodesic_trust(self, current_emb, tactic_name):
        try:
            with torch.no_grad():
                op = self.tactic_operator(tactic_name)
                pred_next = self.lie.apply_tactic(current_emb, op)
                curr_dist = current_emb.norm().item()
                next_dist = pred_next.norm().item()
                trust = curr_dist - next_dist 
                return trust, pred_next
        except: return -1.0, current_emb

    def _online_memory_update(self, trace_path):
        self.tac_to_coeff.train()
        total_loss = 0
        try:
            for step in trace_path:
                if step.get('status') not in ['success', 'continue']: continue
                curr = torch.tensor(step['current_coord'], device=self.device)
                target = torch.tensor(step['actual_coord'], device=self.device)
                tac_emb = self.tactic_enc.encode(step['tactic'])
                coeff = self.tac_to_coeff(tac_emb)
                op = self.lie.get_operator(coeff)
                pred = self.lie.apply_tactic(curr, op)
                loss = self.manifold.dist(pred, target).mean()
                total_loss += loss
            
            if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.tac_to_coeff.parameters(), 1.0)
                self.optimizer.step()
        except Exception: pass
        self.tac_to_coeff.eval()

    def _finalize_result(self, status, proof, trace, exp, config, expanded, failure_info=None):
        layer_counts = Counter(t["layer"] for t in trace)
        radii = [t["radius"] for t in trace if "radius" in t]
        summary = {
            "status": status, "total_expanded": expanded, "proof_len": len(proof),
            "layer_counts": dict(layer_counts), "avg_radius": float(np.mean(radii)) if radii else 0.0,
            "final_radius": radii[-1] if radii else 0.0
        }
        ret = {"status": status, "proof": proof, "trace": trace, "experience": exp, "config": config, "summary": summary}
        if failure_info: ret.update(failure_info)
        return ret

    # =================================================
    # Main Search Loop (Optimized)
    # =================================================
    def search(self, theorem_decl, max_steps=40, config=None):
        import uuid
        run_id = str(uuid.uuid4())
        
        if config is None: config = {}
        search_mode = config.get("mode", "relaxed") 
        
        # [Strategy] AMC 需要更深的步数，但对 Trust 要求适中
        TRUST_THRESHOLD = 0.005 if search_mode == "relaxed" else 0.02
        MAX_TRUST_VIOLATIONS = 8 if search_mode == "relaxed" else 0
        
        config.update({
            "run_id": run_id,
            "trust_threshold": TRUST_THRESHOLD,
            "max_violations": MAX_TRUST_VIOLATIONS
        })
        
        trace = []
        experience_buffer = [] 
        
        base = theorem_decl.split(":= by")[0].strip()
        base = re.sub(r"^\s*theorem\s+\S+\s*", "example ", base)
        
        env = self._ensure_env()
        if not env: return {"status": "EnvDeath", "trace": [], "config": config}
        
        res = env.run_command(f"{base} := by skip", timeout=15) 
        if self._is_env_failure(res):
            self._init_env()
            res = env.run_command(f"{base} := by skip", timeout=15)
            
        if not self._is_proof_state(res):
            return {"status": "InitFail", "trace": [], "config": config}
        
        raw_state_str = str(res.get("messages", [""])[-1].get("data", ""))
        if "unsolved goals" in raw_state_str: raw_state_str = raw_state_str.split("unsolved goals")[-1].strip()

        start_goal = self.normalize_goal(raw_state_str)
        start_emb = self.goal_encoder.encode(start_goal, mode=self.retrieval_mode)
        start_hash = self._get_state_hash(raw_state_str)
        
        root_id = str(uuid.uuid4())[:8]
        lineage = {root_id: None} 
        
        # Frontier: (Cost, Tie, History, GoalText, GoalEmb, NodeID, RawState, Hash, Violations)
        frontier = []
        init_r = start_emb.norm().item()
        heapq.heappush(frontier, (init_r, next(self.counter), [], start_goal, start_emb, root_id, raw_state_str, start_hash, 0))
        
        expanded = 0
        failed_tactics = set()
        hallucination_penalty_set = set() # 记录幻觉词，防止反复尝试
        
        while frontier:
            cost, _, hist, gtext, emb, current_id, raw_state, current_hash, viol = heapq.heappop(frontier)
            
            if len(hist) >= max_steps: continue
            expanded += 1
            if expanded > 80: break # AMC 需要更多节点
            
            current_radius = emb.norm().item()
            goal_struct = self.analyzer.analyze(gtext)
            
            def log_step(layer, tac, status, next_id=None, trust=-99, hint_radii=[], hint_ranks=[], pred_coord=None, actual_coord=None):
                step_info = {
                    "step": expanded, "layer": layer, "node_id": current_id, 
                    "next_id": next_id, "parent_id": lineage.get(current_id),
                    "tactic": tac, "status": status, "radius": current_radius,
                    "trust_score": trust, "hint_avg_radius": np.mean(hint_radii) if hint_radii else 0.0,
                    "hint_min_rank": min(hint_ranks) if hint_ranks else -1,
                    "current_coord": emb.detach().cpu().numpy().tolist(), "actual_coord": actual_coord
                }
                trace.append(step_info)
                return step_info

            # === Layer 1: Structural ===
            struct_tac = self.controller.get_structural_action(gtext)
            if struct_tac:
                new_hist = hist + [struct_tac]
                res = env.run_command(f"{base} := by {'; '.join(new_hist)}")
                msg = str(res)
                st, out = self.parse_result(res)
                
                next_id = str(uuid.uuid4())[:8] if st in ["success", "continue"] else None
                log_step("Structure", struct_tac, st, next_id)
                
                if st == "success": 
                    return self._finalize_result("Success", new_hist, trace, experience_buffer, config, expanded)
                if st == "continue":
                    lineage[next_id] = current_id
                    new_goal = self.normalize_goal(out)
                    new_emb = self.goal_encoder.encode(new_goal, mode=self.retrieval_mode)
                    heapq.heappush(frontier, (cost, next(self.counter), new_hist, new_goal, new_emb, next_id, out, self._get_state_hash(out), viol))
                continue

            # === Layer 2 & 3: Synergy ===
            hints, hint_radii, hint_ranks = self.retrieve_theorems(emb, goal_text=gtext, k=3)
            memory_tactics = self._generate_smart_heuristics(hints, goal_struct)
            reasoning_tactics = self.controller.get_domain_solver(gtext)
            
            chosen_tactics = []
            chosen_tactics.extend(reasoning_tactics) 
            
            high_trust_cands = []
            for tac in memory_tactics:
                # 过滤掉已知的幻觉词
                if any(bad in tac for bad in hallucination_penalty_set): continue

                trust, _ = self._calculate_geodesic_trust(emb, tac)
                if trust > TRUST_THRESHOLD: 
                    high_trust_cands.append(tac)
                elif search_mode == "relaxed" and trust > -0.15: # 放宽限制
                    if viol < MAX_TRUST_VIOLATIONS:
                        high_trust_cands.append(tac)
            
            chosen_tactics.extend(high_trust_cands)

            for tac in chosen_tactics:
                if (current_id, tac) in failed_tactics: continue
                
                trust, pred_emb = self._calculate_geodesic_trust(emb, tac)
                pred_list = pred_emb.detach().cpu().numpy().tolist()
                
                new_hist = hist + [tac]
                res = env.run_command(f"{base} := by {'; '.join(new_hist)}", timeout=20) # 稍微增加超时
                st, out = self.parse_result(res)
                
                # [CRITICAL] 幻觉熔断机制
                if st == "error":
                    if self._is_hallucination(out):
                        # 提取错误信息中的关键幻觉词 (简单处理: 只要报错 unknown 就惩罚该 tactic)
                        print(f"🚫 [Hallucination Detected] Banning tactic branch: {tac}")
                        failed_tactics.add((current_id, tac))
                        # 如果是 apply xxx，尝试把 xxx 加入全局惩罚 (这里简化为只在当前节点放弃)
                        continue 
                    
                    failed_tactics.add((current_id, tac))
                    continue

                next_id = str(uuid.uuid4())[:8] if st in ["success", "continue"] else None
                actual_list = None
                if st == "success": actual_list = [0.0]*len(pred_list) 
                elif st == "continue":
                    new_emb = self.goal_encoder.encode(self.normalize_goal(out), mode=self.retrieval_mode)
                    actual_list = new_emb.detach().cpu().numpy().tolist()

                log_step("Synergy", tac, st, next_id, trust, hint_radii, hint_ranks, pred_list, actual_list)
                
                if st == "success":
                    if not high_trust_cands: self._online_memory_update(trace)
                    return self._finalize_result("Success", new_hist, trace, experience_buffer, config, expanded)
                
                if st == "continue":
                    if "no goals" in str(out):
                         return self._finalize_result("Success", new_hist, trace, experience_buffer, config, expanded)
                    
                    lineage[next_id] = current_id
                    new_viol = viol + 1 if (trust <= TRUST_THRESHOLD) else viol
                    
                    r_val = np.linalg.norm(actual_list) if actual_list else 0
                    new_cost = len(new_hist) + r_val
                    
                    heapq.heappush(frontier, (new_cost, next(self.counter), new_hist, self.normalize_goal(out), new_emb, next_id, out, self._get_state_hash(out), new_viol))

        return self._finalize_result("Fail", [], trace, experience_buffer, config, expanded, 
                                     {"failure_goal": theorem_decl, "failure_radius": 0.0})