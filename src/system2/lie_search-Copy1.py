# ==========================================
# 文件名: src/system2/lie_search.py
# 版本: v86.0 (Final Complete - with Data Logging & Advanced Tactics)
# ==========================================

import os
import sys
import re
import heapq
import gzip
import pickle
import itertools
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

# -------------------------------------------------
# Path & Env
# -------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------------------------------------
# Internal Imports
# -------------------------------------------------
from src.system1.manifold_math import PoincareBall
from src.system1.operators import LogicalLieAlgebra
from src.system2.lean_interaction import LeanEnv
from src.system2.llm_engine import LLMEngine
from src.system2.tactic_encoder import TacticEncoder

# =================================================
# Goal Analysis (Integrated)
# =================================================

@dataclass
class GoalStruct:
    raw: str
    is_searchable: bool
    kind: str        # equality / inequality / prop / unknown
    domain: str      # nat / int / rat / real / complex / prop / unknown

class GoalAnalyzer:
    def __init__(self):
        self.ban_patterns = [
            r"^\d+\s*=\s*\d+$",
            r"^\d+\s*<\s*\d+$",
            r"Nat\.gcd\s+\d+\s+\d+",
        ]

    def analyze(self, goal_text: str) -> GoalStruct:
        g = goal_text.strip()
        
        # 1. Searchability
        is_searchable = True
        for pat in self.ban_patterns:
            if re.match(pat, g):
                is_searchable = False
                break

        # 2. Kind
        if "=" in g: kind = "equality"
        elif "≤" in g or "<" in g or "≥" in g or ">" in g: kind = "inequality"
        elif "→" in g or "∧" in g or "∨" in g: kind = "prop"
        elif "∃" in g or "Exists" in g: kind = "prop" # Treat exists as prop for apply
        else: kind = "unknown"

        # 3. Domain
        if "Nat." in g or "ℕ" in g: domain = "nat"
        elif "Int." in g or "ℤ" in g: domain = "int"
        elif "Rat." in g or "ℚ" in g: domain = "rat"
        elif "Real." in g or "ℝ" in g: domain = "real"
        elif "Complex." in g or "ℂ" in g: domain = "complex"
        elif kind == "prop": domain = "prop"
        else: domain = "unknown"

        return GoalStruct(g, is_searchable, kind, domain)

# =================================================
# System-1 Models
# =================================================

class InferenceHGCN(nn.Module):
    def __init__(self, in_dim, out_dim, c):
        super().__init__()
        self.manifold = PoincareBall(c)
        self.semantic_proj = nn.Linear(in_dim, out_dim)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        z = self.semantic_proj(x)
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        r = 0.9 * torch.tanh(self.scale)
        return self.manifold.expmap0(z * r)

class GoalEncoder:
    def __init__(self, hgcn_ckpt, device):
        self.device = device
        st_path = os.path.join(project_root, "models", "all-MiniLM-L6-v2")
        if not os.path.exists(st_path):
            st_path = "sentence-transformers/all-MiniLM-L6-v2"

        self.bert = SentenceTransformer(st_path, device=device)
        self.bert.eval()

        self.use_hgcn = False
        self.c = 1.0
        self.projector = None

        if os.path.exists(hgcn_ckpt):
            try:
                ckpt = torch.load(hgcn_ckpt, map_location=device)
                
                if "layer.semantic_proj.weight" in ckpt["model"]:
                    w = ckpt["model"]["layer.semantic_proj.weight"]
                    prefix = "layer."
                elif "semantic_proj.weight" in ckpt["model"]:
                    w = ckpt["model"]["semantic_proj.weight"]
                    prefix = ""
                else:
                    # Fallback
                    w = torch.zeros(64, 384)
                    prefix = ""
                
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
                print(f"   [GoalEncoder] HGCN loaded (dim={out_dim}, c={self.c})")
            except Exception as e:
                print(f"   ⚠️ HGCN load failed: {e}")

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
            return emb

class HyperbolicEnergy:
    def __init__(self, c):
        self.manifold = PoincareBall(c)

    def h(self, a, b):
        d = self.manifold.dist(a, b)
        return d.item() if isinstance(d, torch.Tensor) else float(d)

# =================================================
# Search Agent
# =================================================

class RiemannSearchAgent:
    def __init__(self, hgcn_ckpt, llm_path, device="cuda"):
        self.device = device
        self.counter = itertools.count() 

        # ---- Knowledge Graph ----
        data_dir = os.path.join(project_root, "data")
        refined = os.path.join(data_dir, "hgcn_refined.pth")
        if os.path.exists(refined):
            hgcn_ckpt = refined
            print("   [System1] Using refined checkpoint")

        self.idx_to_name = []
        self.graph_emb = None
        self.retrieval_mode = "none"

        id_map = os.path.join(data_dir, "id_to_name.pkl.gz")
        old_map = os.path.join(data_dir, "node_text_map.pkl.gz")
        
        if os.path.exists(id_map):
            with gzip.open(id_map, "rb") as f:
                m = pickle.load(f)
                if isinstance(m, dict):
                    max_idx = max(m.keys())
                    self.idx_to_name = [""] * (max_idx + 1)
                    for k, v in m.items(): self.idx_to_name[k] = v
                else:
                    self.idx_to_name = m
        elif os.path.exists(old_map):
            with gzip.open(old_map, "rb") as f:
                self.idx_to_name = list(pickle.load(f).keys())

        emb_path = os.path.join(data_dir, "node_embeddings.pt")
        if self.idx_to_name and os.path.exists(emb_path):
            self.graph_emb = torch.load(emb_path, map_location=device)
            self.retrieval_mode = "hyperbolic"
            print(f"   [System1] Hyperbolic Embeddings Loaded ({len(self.graph_emb)} nodes)")

        # ---- Models ----
        self.goal_encoder = GoalEncoder(hgcn_ckpt, device)
        self.c = self.goal_encoder.c
        self.manifold = PoincareBall(self.c)
        self.energy = HyperbolicEnergy(self.c)

        lie_dim = 16
        if self.goal_encoder.use_hgcn:
            lie_dim = self.goal_encoder.projector.semantic_proj.weight.shape[0]

        self.lie = LogicalLieAlgebra(lie_dim, 64, c=self.c).to(device)
        self.llm = LLMEngine(llm_path, device=device)
        self.tactic_enc = TacticEncoder(device=device)

        self.tac_to_coeff = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh()
        ).to(device)

        self.analyzer = GoalAnalyzer()
        self.state_visits = defaultdict(int)

    # ---------------------------
    # Normalize Lean goal
    # ---------------------------
    def normalize_goal(self, text: str) -> str:
        if "⊢" in text:
            text = text.split("⊢")[-1]
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\b_[0-9]+\b", "_", text)
        return text.strip()

    # ---------------------------
    # Retrieve Theorems
    # ---------------------------
    def retrieve_theorems(self, goal_text, k=3):
        if self.graph_emb is None: return []

        with torch.no_grad():
            q = self.goal_encoder.encode(goal_text, mode=self.retrieval_mode)
            
            if self.retrieval_mode == "hyperbolic":
                d = self.manifold.dist(q, self.graph_emb)
                if d.dim() == 1: d = d.unsqueeze(0)
                _, idxs = torch.topk(d, k=k*5, largest=False)
            else: # Euclidean fallback
                scores = torch.mm(q, self.graph_emb.T)
                _, idxs = torch.topk(scores, k=k*5, largest=True)

            out = []
            for i in idxs.flatten().tolist():
                if len(out) >= k: break
                if i < len(self.idx_to_name):
                    name = self.idx_to_name[i]
                    if not name or name.startswith("_") or ".proof_" in name: continue
                    if "delab" in name or "repr" in name or "inst" in name: continue
                    if len(name) < 2: continue
                    out.append(name)
            return out

    # ---------------------------
    # Smart Heuristics (Advanced)
    # ---------------------------
    def _generate_smart_heuristics(self, hints: list, goal_struct) -> list:
        tactics = []
        if not hints: return []
        
        # We process top 3 hints to give more chances
        top_hints = hints[:3] 
        
        for i, h in enumerate(top_hints):
            is_top_1 = (i == 0)
            
            # 1. Equality Strategies
            if goal_struct.kind == "equality" or "iff" in h or "eq" in h:
                tactics.append(f"rw [{h}]")
                tactics.append(f"rw [← {h}]")
                if is_top_1:
                    tactics.append(f"simp only [{h}]")

            # 2. Application Strategies (Critical Fix)
            tactics.append(f"apply {h}")
            
            # [Fix: Refine] Handle missing parameters
            if is_top_1:
                tactics.append(f"refine {h} _") 
                tactics.append(f"refine {h} _ _")

            # 3. Type-Specific Handling
            if "dvd" in h:
                tactics.append(f"apply dvd_trans _ {h}") 
            if ("le" in h or "lt" in h) and goal_struct.kind == "inequality":
                tactics.append(f"linarith [{h}]")

        # 4. Domain Solvers
        if goal_struct.domain == "nat":
            tactics.append("omega")
        elif goal_struct.domain in ("real", "complex"):
            tactics.append("nlinarith")
            if goal_struct.domain == "complex":
                # [Fix: Nuclear Option] Try to solve context
                tactics.append("linear_combination") 
                tactics.append("simp at *; linear_combination")

        # Deduplicate
        seen = set()
        unique_tactics = []
        for t in tactics:
            if t not in seen:
                seen.add(t)
                unique_tactics.append(t)
                
        return unique_tactics

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
        if "panic" in low or "too big" in low: return "panic", text
        if "error" in low and "unsolved goals" not in low: return "error", text
        if "⊢" in text: return "continue", text.split("⊢")[-1].strip()
        return "continue", text

    def _restart_env(self, env):
        try: env.close()
        except: pass
        env = LeanEnv(project_root)
        env.run_command("import Mathlib", timeout=120)
        # Import LinearCombination explicitly
        env.run_command("import Mathlib.Tactic.LinearCombination")
        env.run_command("open Nat Real Rat BigOperators Set Finset Function")
        return env

    # =================================================
    # Main Search
    # =================================================
    def search(self, theorem_decl, max_steps=15):
        trace = []

        base = theorem_decl.split(":= by")[0].strip()
        base = re.sub(r"^\s*theorem\s+\S+\s*", "example ", base)
        goal_text = base.split(":", 1)[1].strip()

        goal = self.analyzer.analyze(goal_text)
        trace.append({"stage": "classify", "struct": str(goal)})

        if not goal.is_searchable:
            return {"status": "Skipped", "trace": trace}

        env = LeanEnv(project_root)
        try:
            print("   [Env] Initializing...", flush=True)
            env.run_command("import Mathlib", timeout=120)
            env.run_command("import Mathlib.Tactic.LinearCombination")
            env.run_command("open Nat Real Rat BigOperators Set Finset Function")

            # 1. Fast Solvers
            solvers = ["simp", "aesop"]
            if goal.domain == "nat": solvers = ["omega"] + solvers
            if goal.domain == "real": solvers = ["nlinarith"] + solvers
            if goal.domain == "complex": solvers = ["linear_combination"] + solvers

            for s in solvers:
                res = env.run_command(f"{base} := by {s}", timeout=20)
                st, _ = self.parse_result(res)
                if st == "success":
                    return {"status": "Success", "proof": [s], "trace": trace}
                elif st == "panic":
                    env = self._restart_env(env)

            # 2. A* Search
            start_norm = self.normalize_goal(goal_text)
            goal_emb = self.goal_encoder.encode(start_norm, mode="hyperbolic")
            target = torch.zeros_like(goal_emb)

            frontier = []
            heapq.heappush(frontier, (0.0, next(self.counter), [], start_norm, goal_emb))

            expanded = 0
            while frontier:
                _, _, hist, gtext, emb = heapq.heappop(frontier)
                
                if len(hist) >= max_steps: continue
                expanded += 1
                if expanded > 50: break

                # --- Structural Expansion ---
                if "∧" in gtext:
                    t = "constructor"
                    code = f"{base} := by {'; '.join(hist + [t])}"
                    res = env.run_command(code)
                    out = str(res)
                    if "⊢" in out and "error" not in out.lower():
                        ng = self.normalize_goal(out)
                        heapq.heappush(frontier, (len(hist), next(self.counter), hist + [t], ng, emb))
                    continue

                # --- Retrieval & Candidates ---
                hints = self.retrieve_theorems(gtext, k=3)
                if hints: print(f"      [RAG] Hints: {hints}", flush=True)

                smart_cands = self._generate_smart_heuristics(hints, goal)
                
                try: llm_cands = self.llm.generate_candidates(gtext, hints=hints, num=1)
                except: llm_cands = []

                candidates = smart_cands + llm_cands

                for t in candidates:
                    if not t: continue
                    
                    try:
                        M = self.tactic_operator(t)
                        pred = self.lie.apply_tactic(emb, M)
                        h_val = self.energy.h(pred, target)
                    except: pred, h_val = emb, 0.0

                    code = f"{base} := by {'; '.join(hist + [t])}"
                    res = env.run_command(code, timeout=20)
                    st, out = self.parse_result(res)

                    # [新增] 详细的 System 1 数据埋点
                    step_data = {
                        "step": expanded,
                        "depth": len(hist),
                        "tactic": t,
                        "status": st,
                        "goal_text": gtext[:100], 
                        # --- System 1 核心指标 ---
                        "current_coord": emb.detach().cpu().numpy().tolist(), 
                        "predicted_coord": pred.detach().cpu().numpy().tolist(), 
                        "target_dist": float(h_val),
                        "retrieved_hints": hints if hints else [] 
                    }
                    trace.append(step_data)

                    if st == "success": 
                        return {"status": "Success", "proof": hist + [t], "trace": trace}
                    
                    if st == "continue":
                        ng = self.normalize_goal(out)
                        self.state_visits[ng] += 1
                        if self.state_visits[ng] <= 2:
                            heapq.heappush(
                                frontier,
                                (len(hist) + 1 + h_val, next(self.counter), hist + [t], ng, pred)
                            )
                    
                    if st == "panic": env = self._restart_env(env)

            return {"status": "Failed", "trace": trace}

        finally:
            env.close()