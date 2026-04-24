# ==========================================
# Filename: src/system2/mcts_hybrid_search.py
# Version: v208.0 (Full-State Edition)
#
# Core regression fixes vs v206:
#   FIX-1: CONTEXT INJECTION — theorems that reference local definitions
#           (Miustr, Derivable, Goodm, etc.) are proved inside their own file's
#           namespace by prepending the file source up to the theorem declaration.
#           This is how the working v86 lie_search achieved 65% pass@1.
#   FIX-2: DECLARATION EXTRACTION — the regex was matching partial text inside
#           comments and `have` blocks, producing names like "(h" and ":".
#           A stricter multiline-aware regex is used, and declarations are
#           validated before search begins.
#   FIX-3: PERSISTENT ENV PER PROBLEM — the old v86 kept one LeanEnv open for
#           the entire search of a single problem. The new MCTS was spawning a
#           new env per rollout (paying ~2s startup × 40 = 80s overhead).
#           Now a single env is reused across all rollouts; only restarted on panic.
#   FIX-4: SORRY-BASED INIT — restored correct goal extraction via the
#           `sorries` field in the REPL response (not `unsolved goals` in messages).
#   FIX-5: TACTIC BLACKLIST + CLEAN LLM OUTPUT — carried forward from v206.
# ==========================================
#
# Regression fixes vs v207:
#   FIX-A: _extract_goal_from_sorry now returns the FULL proof state
#           (hypotheses + turnstile + conclusion). The previous split("⊢")[-1]
#           discarded ALL hypotheses, so the LLM received "x = 3 := by" with
#           no knowledge of h0: 2*(2*(2*(2*x)))=48, making linarith impossible.
#   FIX-B: _parse_tactic_result similarly returns the full state for "continue"
#           so child nodes inherit the correct hypothesis context.
#   FIX-C: normalize_goal no longer strips hypotheses (whitespace-only cleanup).
#           A new _goal_conclusion() helper extracts just the conclusion and is
#           used exclusively for hyperbolic retrieval embedding — shorter text
#           gives a cleaner signal, preventing hallucinated hints like
#           "degree_cubic", "tfaeHaveCore", "lieCharpoly_monic".
#   FIX-D: All goal_encoder.encode() calls use _goal_conclusion() so the
#           hyperbolic radius reflects the mathematical difficulty of the
#           conclusion, not the length of the hypothesis list.

import os
import sys
import re
import math
import traceback
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.system1.manifold_math import PoincareBall
from src.system1.operators import LogicalLieAlgebra
from src.system2.lean_interaction import LeanEnv
from src.system2.llm_engine import LLMEngine
from src.system2.tactic_encoder import TacticEncoder
from src.system2.lie_search import GoalEncoder, GoalAnalyzer

# ------------------------------------------------------------------
# Hallucination blacklist
# ------------------------------------------------------------------
TACTIC_BLACKLIST = {
    "letTelescope", "leftMoves_mul", "isEmpty_nim_zero_leftMoves",
    "mul_div_mul_comm_of_dvd_dvd", "nndist_mul_left", "Odd.mod_even",
}

_TACTIC_LINE_RE = re.compile(
    r"^(apply|exact|refine|rw|simp|linarith|omega|ring|norm_num|"
    r"constructor|intro|cases|induction|use|have|obtain|push_neg|"
    r"contrapose|decide|tauto|trivial|assumption|field_simp|"
    r"linear_combination|nlinarith|positivity|gcongr|ext|funext|"
    r"congr|convert|calc|by_contra|by_cases|split|fin_cases|"
    r"interval_cases|aesop|norm_cast|push_cast|lift|rcases|"
    r"specialize|replace|rfl|symm|trans|left|right|exfalso|"
    r"contradiction|simp_all|ring_nf|norm_num1|native_decide)\b"
)


def _clean_llm_tactics(raw: list) -> list:
    out = []
    for item in raw:
        if not item:
            continue
        for line in item.splitlines():
            line = line.strip()
            if not line or line.startswith("/-") or line.startswith("--"):
                continue
            if not _TACTIC_LINE_RE.match(line):
                continue
            if any(bl in line for bl in TACTIC_BLACKLIST):
                continue
            out.append(line)
            break
    seen: set = set()
    return [t for t in out if not (t in seen or seen.add(t))]


# =================================================
# Module 1: Product Manifold Dynamics
# =================================================

class ProductManifoldAnalyzer(GoalAnalyzer):
    def get_manifold_mixing_weight(self, text: str) -> float:
        text = text or ""
        lateral = ["ring", "field", "linarith", "omega", "norm_num",
                   "Complex", "Real", "Matrix", "equiv", "iff", "eq"]
        hierarchical = ["subset", "union", "inter", "measure",
                        "continuous", "TopologicalSpace"]
        ls = sum(1 for w in lateral if w.lower() in text.lower())
        hs = sum(1 for w in hierarchical if w.lower() in text.lower())
        if ls > hs: return 0.2
        if hs > ls: return 0.8
        return 0.5


# =================================================
# Module 2: MCTS Node
# =================================================

class MCTSNode:
    def __init__(self, state_text: str, hist: list,
                 parent=None, tactic=None, prior=0.0):
        self.state_text = state_text or ""
        self.hist = hist
        self.parent = parent
        self.tactic = tactic
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.is_terminal = False
        self.is_success = False
        self.is_expanded = False
        self.radius = 0.0

    @property
    def value(self):
        return self.value_sum / self.visits if self.visits else 0.0


# =================================================
# Core Agent
# =================================================

class LieGuidedMCTSAgent:
    def __init__(self, hgcn_ckpt, llm_path, device="cuda"):
        self.device = device
        self.analyzer = ProductManifoldAnalyzer()
        self.goal_encoder = GoalEncoder(hgcn_ckpt, device)
        self.manifold = PoincareBall(c=self.goal_encoder.c)

        data_dir = os.path.join(project_root, "data")
        emb_path = os.path.join(data_dir, "node_embeddings.pt")
        map_path = os.path.join(data_dir, "id_to_name.pkl.gz")

        self.idx_to_name = {}
        if os.path.exists(map_path):
            import gzip, pickle
            with gzip.open(map_path, "rb") as f:
                raw = pickle.load(f)
                if isinstance(raw, dict):
                    self.idx_to_name = raw
                else:
                    self.idx_to_name = {i: v for i, v in enumerate(raw)}

        self.graph_emb = None
        if os.path.exists(emb_path):
            self.graph_emb = torch.load(emb_path, map_location=device)

        lie_dim = 64
        if self.goal_encoder.use_hgcn:
            lie_dim = self.goal_encoder.projector.semantic_proj.weight.shape[0]

        self.lie = LogicalLieAlgebra(lie_dim, 64, c=self.goal_encoder.c).to(device)
        self.llm = LLMEngine(llm_path, device=device)
        self.tactic_enc = TacticEncoder(device=device)
        self.tac_to_coeff = nn.Sequential(
            nn.Linear(384, 128), nn.ReLU(), nn.Linear(128, 64), nn.Tanh()
        ).to(device)

        self._bm25_index = None
        self._build_bm25_index()

    # ------------------------------------------------------------------
    def _build_bm25_index(self):
        if not self.idx_to_name:
            return
        try:
            from rank_bm25 import BM25Okapi
            corpus = [
                n.replace(".", " ").replace("_", " ").lower()
                for n in self.idx_to_name.values() if isinstance(n, str)
            ]
            self._bm25_index = BM25Okapi([doc.split() for doc in corpus])
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # FIX-1: Build a "context prefix" so local definitions are in scope.
    # We source everything in the file up to (but not including) the
    # theorem declaration line, stripped of import statements.
    # ------------------------------------------------------------------
    def _build_context_prefix(self, source_path: str, theorem_name: str) -> str:
        if not source_path or not os.path.exists(source_path):
            return ""
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception:
            return ""

        target_line = -1
        pat = re.compile(
            r"^\s*(?:@\[.*?\]\s*)?(?:theorem|lemma)\s+"
            + re.escape(theorem_name) + r"\b"
        )
        for i, line in enumerate(lines):
            if pat.match(line):
                target_line = i
                break
        if target_line == -1:
            for i, line in enumerate(lines):
                if theorem_name in line and (
                    "theorem " in line or "lemma " in line
                ):
                    target_line = i
                    break
        if target_line == -1:
            return ""

        # Strip import lines (already loaded by the REPL) and return everything else
        prefix_lines = []
        for line in lines[:target_line]:
            stripped = line.strip()
            if stripped.startswith("import "):
                continue
            prefix_lines.append(line)

        return "".join(prefix_lines)

    # ------------------------------------------------------------------
    # FIX-3: One env per problem, restarted only on panic.
    # ------------------------------------------------------------------
    def _init_env(self) -> LeanEnv:
        env = LeanEnv(project_root)
        try:
            env.run_command("import Mathlib", timeout=120)
            env.run_command(
                "open Nat Real Rat BigOperators Set Finset Function",
                timeout=30,
            )
        except RuntimeError:
            pass
        return env

    def _restart_env(self, env: LeanEnv, context_prefix: str = "") -> LeanEnv:
        try:
            env.close()
        except Exception:
            pass
        new_env = self._init_env()
        if context_prefix.strip():
            try:
                new_env.run_command(context_prefix, timeout=60)
            except Exception:
                pass
        return new_env

    # ------------------------------------------------------------------
    # FIX-4: Extract goal from the `sorries` field of the REPL response.
    # The `sorries` field is populated when `sorry` is used and the goal
    # is well-typed — this is the reliable way to get the initial state.
    # A response with only type errors (function expected, etc.) yields "".
    # ------------------------------------------------------------------
    def _extract_goal_from_sorry(self, res) -> str:
        """Return the FULL proof state (hypotheses + turnstile + conclusion).
        Stripping hypotheses here was the primary regression: the LLM received
        only the bare conclusion and had no basis to pick linarith/nlinarith."""
        if not res or not isinstance(res, dict):
            return ""

        msgs = res.get("messages", [])
        sorries = res.get("sorries", [])

        # Any real error = the decl itself is broken, don't proceed
        for m in msgs:
            sev = m.get("severity", "")
            data = str(m.get("data", ""))
            if sev == "error" and "declaration uses 'sorry'" not in data:
                return ""

        # Primary: use the sorries field — keep the WHOLE goal string
        if sorries:
            goal = sorries[0].get("goal", "")
            if "sorryAx" in goal:
                return ""
            return goal.strip()   # includes hypotheses AND turnstile

        # Fallback: unsolved goals in messages — keep everything after the header
        for m in msgs:
            data = str(m.get("data", ""))
            if "unsolved goals" in data and "sorryAx" not in data and "⊢" in data:
                # Strip the "unsolved goals" header line, keep the rest
                lines = data.split("\n")
                body = [l for l in lines if l.strip() and "unsolved goals" not in l]
                return "\n".join(body).strip()

        return ""

    def _parse_tactic_result(self, res):
        """Returns (status, goal_text): status in {success, continue, error, panic}"""
        if not res or not isinstance(res, dict):
            return "error", ""

        msgs = res.get("messages", [])
        sorries = res.get("sorries", [])

        if not msgs and not sorries:
            return ("success", "") if "env" in res else ("error", "")

        has_error = False
        unsolved = ""

        for m in msgs:
            data = str(m.get("data", ""))
            sev = m.get("severity", "")
            if "panic" in data.lower() or "too big" in data.lower():
                return "panic", ""
            if sev == "error":
                if "unsolved goals" in data:
                    if "sorryAx" in data:
                        return "error", ""
                    unsolved = data
                elif "declaration uses 'sorry'" not in data:
                    has_error = True

        if has_error:
            return "error", ""

        if sorries:
            goal = sorries[0].get("goal", "")
            if "sorryAx" in goal:
                return "error", ""
            if goal:
                return "continue", goal.strip()  # full state: hyps + turnstile

        if unsolved:
            # Strip "unsolved goals" header, keep hyps + turnstile
            lines = unsolved.split("\n")
            body = [l for l in lines if l.strip() and "unsolved goals" not in l]
            return "continue", "\n".join(body).strip()

        return "success", ""

    # ------------------------------------------------------------------
    def normalize_goal(self, text: str) -> str:
        """Normalise whitespace/metavars in a full proof state string.
        Does NOT strip hypotheses — used to store node.state_text."""
        text = text or ""
        text = re.sub(r"\s+", " ", text)
        return re.sub(r"\b_[0-9]+\b", "_", text).strip()

    def _goal_conclusion(self, text: str) -> str:
        """Return only the conclusion (after the last turnstile).
        Used as the retrieval query — shorter text gives better embeddings."""
        text = text or ""
        if "\u22a2" in text:   # ⊢
            text = text.split("\u22a2")[-1]
        text = re.sub(r"\s+", " ", text)
        return re.sub(r"\b_[0-9]+\b", "_", text).strip()

    # ------------------------------------------------------------------
    def retrieve_hybrid_theorems(self, goal_emb, goal_text,
                                 k=3, backend="hyperbolic"):
        if self.graph_emb is None:
            return []
        goal_text = goal_text or ""
        with torch.no_grad():
            if backend == "bm25" and self._bm25_index:
                tokens = (
                    goal_text.lower().replace(".", " ").replace("_", " ").split()
                )
                scores = self._bm25_index.get_scores(tokens)
                idxs = torch.tensor(np.argsort(scores)[::-1][: k * 5])
            elif backend == "cosine":
                sim = torch.mm(
                    F.normalize(goal_emb, dim=-1),
                    F.normalize(self.graph_emb, dim=-1).T,
                ).squeeze()
                _, idxs = torch.topk(sim, k=k * 5, largest=True)
            else:
                alpha = self.analyzer.get_manifold_mixing_weight(goal_text)
                dh = self.manifold.dist(goal_emb, self.graph_emb).squeeze()
                ndh = dh / (dh.max() + 1e-5)
                sc = torch.mm(
                    F.normalize(goal_emb, dim=-1),
                    F.normalize(self.graph_emb, dim=-1).T,
                ).squeeze()
                ndc = (1.0 - sc) / ((1.0 - sc).max() + 1e-5)
                hybrid = alpha * ndh + (1.0 - alpha) * ndc
                _, idxs = torch.topk(hybrid, k=k * 5, largest=False)

        out = []
        for i in idxs.flatten().tolist():
            if len(out) >= k:
                break
            name = self.idx_to_name.get(int(i), "")
            if name and not name.startswith("_") and "test" not in name and len(name) > 2:
                out.append(name)
        return out

    # ------------------------------------------------------------------
    def _generate_smart_heuristics(self, hints: list, goal_text: str) -> list:
        tactics = []
        goal_struct = self.analyzer.analyze(goal_text)

        for i, h in enumerate(hints[:3]):
            if any(bl in h for bl in TACTIC_BLACKLIST):
                continue
            is_top = i == 0
            if goal_struct.kind == "equality" or "iff" in h or "eq" in h:
                tactics.extend([f"rw [{h}]", f"rw [← {h}]"])
                if is_top:
                    tactics.append(f"simp only [{h}]")
            tactics.append(f"apply {h}")
            if is_top:
                tactics.extend([f"refine {h} _", f"refine {h} _ _"])
            if ("le" in h or "lt" in h) and goal_struct.kind == "inequality":
                tactics.append(f"linarith [{h}]")

        tactics.extend(["rfl", "trivial", "assumption", "decide", "aesop"])

        if goal_struct.domain == "nat":
            tactics.extend(["omega", "simp"])
        elif goal_struct.domain in ("real", "complex"):
            tactics.extend(["nlinarith", "ring", "norm_num"])
            if goal_struct.domain == "complex":
                tactics.extend(["linear_combination", "simp [*]"])
        elif goal_struct.domain == "prop":
            tactics.extend(["tauto", "simp [*]", "simp_all"])
        elif goal_struct.domain == "int":
            tactics.extend(["omega", "ring"])

        if "∧" in goal_text:
            tactics.append("constructor")
        if "∨" in goal_text:
            tactics.extend(["left", "right"])
        if "¬" in goal_text or "Not" in goal_text:
            tactics.extend(["push_neg", "contrapose!"])
        if "∃" in goal_text:
            tactics.append("exact ⟨_, rfl⟩")

        seen: set = set()
        return [t for t in tactics if not (t in seen or seen.add(t))]

    # ------------------------------------------------------------------
    def _geodesic_trust(self, emb, tactic_name: str) -> float:
        try:
            with torch.no_grad():
                op = self.lie.get_operator(
                    self.tac_to_coeff(self.tactic_enc.encode(tactic_name))
                )
                pred = self.lie.apply_tactic(emb, op)
                return emb.norm().item() - pred.norm().item()
        except Exception:
            return -1.0

    def puct_select(self, node: MCTSNode, c_puct: float = 1.5) -> MCTSNode:
        best, best_child = -float("inf"), None
        for child in node.children:
            u = c_puct * child.prior * (
                math.sqrt(node.visits) / (1 + child.visits)
            )
            s = child.value + u
            if s > best:
                best, best_child = s, child
        return best_child

    def _backprop(self, node: MCTSNode, value: float):
        curr = node
        while curr is not None:
            curr.visits += 1
            curr.value_sum += value
            curr = curr.parent

    # ------------------------------------------------------------------
    # FIX-1+3: search() accepts source_path for context injection.
    # A single LeanEnv lives for the entire problem.
    # ------------------------------------------------------------------
    def search(self, theorem_decl: str, max_steps: int = 30,
               config: dict = None, source_path: str = None):
        if config is None:
            config = {}
        retrieval_backend = config.get("retrieval_backend", "hyperbolic")
        max_rollouts = config.get("max_steps", max_steps)

        # Normalise declaration to `example ...`
        base_raw = theorem_decl.split(":=")[0].strip()
        name_match = re.match(r"(?:theorem|lemma)\s+(\S+)", base_raw)
        theorem_name = name_match.group(1) if name_match else ""
        base = re.sub(
            r"^\s*(?:theorem|lemma)\s+\S+\s*", "example ", base_raw
        ).strip()

        # FIX-1: Load file context
        context_prefix = self._build_context_prefix(source_path, theorem_name)

        # FIX-3: Single persistent env
        env = self._init_env()

        try:
            # Inject context
            if context_prefix.strip():
                try:
                    ctx_res = env.run_command(context_prefix, timeout=60)
                    ctx_msgs = (ctx_res or {}).get("messages", [])
                    ctx_errors = [
                        m for m in ctx_msgs
                        if m.get("severity") == "error"
                        and "declaration uses 'sorry'" not in str(m.get("data", ""))
                    ]
                    if ctx_errors:
                        print(
                            f"   ⚠️  Context injection failed for '{theorem_name}', "
                            f"proceeding without it.",
                            flush=True,
                        )
                        env = self._restart_env(env)  # clean env, no context
                        context_prefix = ""
                except Exception:
                    env = self._restart_env(env)
                    context_prefix = ""

            # Fast solvers (like v86 lie_search)
            type_text = base.split(":", 1)[-1].strip() if ":" in base else base
            goal_struct = self.analyzer.analyze(type_text)

            fast_solvers = ["rfl", "simp", "aesop", "decide", "tauto"]
            if goal_struct.domain == "nat":
                fast_solvers = ["omega"] + fast_solvers
            elif goal_struct.domain in ("real", "int"):
                fast_solvers = ["ring", "norm_num", "linarith"] + fast_solvers
            elif goal_struct.domain == "complex":
                fast_solvers = ["ring", "linear_combination"] + fast_solvers

            for solver in fast_solvers:
                try:
                    res = env.run_command(
                        f"{base} := by {solver}", timeout=15
                    )
                    st, _ = self._parse_tactic_result(res)
                    if st == "success":
                        env.close()
                        return {
                            "status": "Success",
                            "proof": [solver],
                            "summary": {"total_expanded": 0},
                            "token_count": 0,
                        }
                    if st == "panic":
                        env = self._restart_env(env, context_prefix)
                except Exception:
                    pass

            # Get initial proof state via sorry (FIX-4)
            try:
                init_res = env.run_command(
                    f"{base} := by sorry", timeout=20
                )
            except Exception:
                env.close()
                return {"status": "InitFail", "proof": [], "summary": {}}

            start_goal = self._extract_goal_from_sorry(init_res)
            if not start_goal:
                env.close()
                return {"status": "InitFail", "proof": [], "summary": {}}

            start_goal = self.normalize_goal(start_goal)
            if not start_goal or "sorryAx" in start_goal:
                env.close()
                return {"status": "InitFail", "proof": [], "summary": {}}

            # Build MCTS root
            root = MCTSNode(state_text=start_goal, hist=[])
            try:
                root.radius = self.goal_encoder.encode(
                    self._goal_conclusion(start_goal), mode="hyperbolic"
                ).norm().item()
            except Exception:
                root.radius = 1.0

            expanded_nodes = 0
            token_count = 0

            for _rollout in range(max_rollouts):
                # Selection
                node = root
                while node.is_expanded and not node.is_terminal:
                    if not node.children:
                        break
                    node = self.puct_select(node)

                # Terminal short-circuit
                if node.is_terminal:
                    self._backprop(node, 1.0 if node.is_success else -1.0)
                    if node.is_success:
                        env.close()
                        return {
                            "status": "Success",
                            "proof": node.hist,
                            "summary": {"total_expanded": expanded_nodes},
                            "token_count": token_count,
                        }
                    continue

                # Evaluate node with unknown state (FIX-3: reuse env)
                if not node.state_text and node.hist:
                    tactic_block = "\n  ".join(node.hist)
                    cmd = f"{base} := by\n  {tactic_block}"
                    try:
                        res = env.run_command(cmd, timeout=20)
                    except Exception:
                        env = self._restart_env(env, context_prefix)
                        node.is_terminal = True
                        self._backprop(node, -1.0)
                        continue

                    st, out_text = self._parse_tactic_result(res)
                    if st == "success":
                        node.is_success = True
                        node.is_terminal = True
                        self._backprop(node, 1.0)
                        env.close()
                        return {
                            "status": "Success",
                            "proof": node.hist,
                            "summary": {"total_expanded": expanded_nodes},
                            "token_count": token_count,
                        }
                    elif st == "panic":
                        env = self._restart_env(env, context_prefix)
                        node.is_terminal = True
                        self._backprop(node, -1.0)
                        continue
                    elif st == "error":
                        node.is_terminal = True
                        self._backprop(node, -1.0)
                        continue
                    else:
                        node.state_text = self.normalize_goal(out_text)
                        if not node.state_text or "sorryAx" in node.state_text:
                            node.is_terminal = True
                            self._backprop(node, -1.0)
                            continue
                        try:
                            new_r = self.goal_encoder.encode(
                                self._goal_conclusion(node.state_text), mode="hyperbolic"
                            ).norm().item()
                        except Exception:
                            new_r = node.parent.radius if node.parent else 1.0
                        node.radius = new_r
                        par_r = node.parent.radius if node.parent else new_r
                        self._backprop(node, (par_r - new_r) * 10.0)

                # Expansion — eager eval (FIX-3: env is already warm)
                if node.is_terminal or node.is_expanded or not node.state_text:
                    continue

                try:
                    # FIX-4: encode only the conclusion for retrieval
                    # (hypothesis strings pollute the embedding signal).
                    # The full state is still passed to the LLM below.
                    conclusion = self._goal_conclusion(node.state_text)
                    goal_emb = self.goal_encoder.encode(
                        conclusion, mode="hyperbolic"
                    )
                    hints = self.retrieve_hybrid_theorems(
                        goal_emb, conclusion,
                        k=3, backend=retrieval_backend,
                    )
                    smart_tacs = self._generate_smart_heuristics(
                        hints, node.state_text
                    )

                    try:
                        # Pass full proof state (with hypotheses) so the LLM
                        # can choose tactics like linarith [h₀], nlinarith, etc.
                        llm_raw = self.llm.generate_candidates(
                            node.state_text, hints, num=2
                        )
                        token_count += 512
                    except Exception:
                        llm_raw = []

                    if isinstance(llm_raw, str):
                        llm_raw = [llm_raw]
                    clean_llm = _clean_llm_tactics(llm_raw)
                    tactics = list(dict.fromkeys(
                        t for t in (smart_tacs + clean_llm) if t
                    ))

                    if not tactics:
                        node.is_terminal = True
                        self._backprop(node, -1.0)
                        continue

                    valid_children = []
                    for tac in tactics:
                        child_hist = node.hist + [tac]
                        tactic_block = "\n  ".join(child_hist)
                        cmd = f"{base} := by\n  {tactic_block}"
                        try:
                            c_res = env.run_command(cmd, timeout=15)
                        except Exception:
                            env = self._restart_env(env, context_prefix)
                            continue

                        c_st, c_out = self._parse_tactic_result(c_res)

                        if c_st == "success":
                            env.close()
                            return {
                                "status": "Success",
                                "proof": child_hist,
                                "summary": {"total_expanded": expanded_nodes + 1},
                                "token_count": token_count,
                            }
                        elif c_st == "panic":
                            env = self._restart_env(env, context_prefix)
                            continue
                        elif c_st == "error":
                            continue
                        else:
                            c_goal = self.normalize_goal(c_out)
                            if not c_goal or "sorryAx" in c_goal:
                                continue
                            trust = self._geodesic_trust(goal_emb, tac)
                            valid_children.append(
                                (tac, c_goal, math.exp(trust / 0.1))
                            )

                    if not valid_children:
                        node.is_terminal = True
                        self._backprop(node, -1.0)
                        continue

                    sum_trust = sum(v[2] for v in valid_children) + 1e-5
                    par_r = node.radius
                    for tac, c_goal, trust_exp in valid_children:
                        child = MCTSNode(
                            state_text=c_goal,
                            hist=node.hist + [tac],
                            parent=node,
                            tactic=tac,
                            prior=trust_exp / sum_trust,
                        )
                        try:
                            new_r = self.goal_encoder.encode(
                                self._goal_conclusion(c_goal), mode="hyperbolic"
                            ).norm().item()
                        except Exception:
                            new_r = par_r
                        child.radius = new_r
                        child.visits = 1
                        child.value_sum = (par_r - new_r) * 10.0
                        node.children.append(child)

                    node.is_expanded = True
                    expanded_nodes += 1
                    avg_val = (
                        sum(c.value for c in node.children) / len(node.children)
                    )
                    self._backprop(node, avg_val)

                except Exception as e:
                    print(f"\n🔥 MCTS EXPANSION CRASH: {e}", flush=True)
                    traceback.print_exc()
                    env.close()
                    return {
                        "status": "ScriptCrash",
                        "error": str(e),
                        "proof": [],
                        "summary": {"total_expanded": expanded_nodes},
                        "token_count": token_count,
                    }

            env.close()
            return {
                "status": "Fail",
                "proof": [],
                "summary": {"total_expanded": expanded_nodes},
                "token_count": token_count,
            }

        except Exception as e:
            try:
                env.close()
            except Exception:
                pass
            return {
                "status": "ScriptCrash",
                "error": str(e),
                "proof": [],
                "summary": {},
                "token_count": 0,
            }