# ==============================================================================
# Filename: src/system2/lie_search_putnam.py
# Version: v1.0
#
# Why this exists — structural differences from miniF2F:
#
#   miniF2F                         Putnam
#   ─────────────────────────────   ──────────────────────────────────────────
#   Concrete numeric goals          Abstract: ∃ constructions, ∀ chains
#   omega/nlinarith covers ~60%     omega/linarith almost never works directly
#   Depth 1-3 tactics               Depth 5-15+ tactics
#   Self-contained                  Requires intro/obtain/rcases/use/induction
#   Fast solvers dominate           Must enter A* search almost every problem
#   5-10 candidates per node        Needs richer structural candidates
#
# Key adaptations:
#   1. FAST SOLVERS — stripped to what can actually close a Putnam goal.
#      omega/nlinarith removed from fast list (waste compute). Added decide,
#      norm_num, simp [*] for rare but real quick wins.
#   2. STRUCTURAL OPENING — before retrieval, always try intro/constructor/
#      obtain to open the goal shape. Putnam goals almost always start with
#      ∀ or ∃ that must be discharged first.
#   3. TACTIC CANDIDATES — 5 priority tiers ordered by Putnam prevalence:
#        T1: Goal-shape openers  (intro, constructor, use, obtain)
#        T2: Existence witnesses (exact ⟨_, _⟩ variants)
#        T3: Induction/cases     (induction, rcases, fin_cases)
#        T4: Domain closers      (linarith, ring, norm_num after setup)
#        T5: Hint-based          (apply/rw from retriever, filtered)
#   4. DEEPER SEARCH — max_steps default raised to 64 (vs 40 for miniF2F).
#      Putnam proofs are longer; cutting off at 40 abandons viable paths.
#   5. TIMEOUT PER STEP — raised to 30s (vs 20s). Putnam tactics like
#      `decide` and `norm_num` on large expressions can be slow.
# ==============================================================================

import os
import sys
import re
import heapq
import itertools
import torch
import torch.nn.functional as F
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Reuse everything from lie_search — only override what differs
from src.system2.lie_search import (
    RiemannSearchAgent, GoalAnalyzer, GoalStruct,
)
from src.system2.lean_interaction import LeanEnv


# ==============================================================================
# Putnam-aware goal analyser
# ==============================================================================

class PutnamGoalAnalyzer(GoalAnalyzer):
    """
    Extended analyser that recognises Putnam-specific goal shapes.
    Adds: existential_construction, universal_chain, induction_target,
          divisibility, function_property.
    """

    def analyse_putnam(self, goal_text: str) -> dict:
        g = goal_text or ""
        return {
            # Shape
            "is_existential":      "∃" in g or "Exists" in g,
            "is_universal":        "∀" in g or "∈" in g,
            "is_iff":              "↔" in g,
            "is_conjunction":      "∧" in g,
            "is_negation":         "¬" in g or "Not " in g,
            "is_divisibility":     "∣" in g or "dvd" in g,
            "is_prime":            "Prime" in g or "Nat.Prime" in g,
            "is_induction_ready":  any(w in g for w in ("ℕ", "Nat", "Fin ", "List")),
            # Domain
            "has_function":        "→" in g and ("ℝ" in g or "ℤ" in g or "ℕ" in g),
            "has_real_analysis":   any(w in g for w in
                                       ("Differentiable", "Continuous", "Filter",
                                        "MeasureTheory", "Metric", "IsOpen")),
            "has_number_theory":   any(w in g for w in
                                       ("Nat.Coprime", "IsCoprime", "Finset",
                                        "Nat.factors", "multiplicity")),
            "has_algebra":         any(w in g for w in
                                       ("Group", "Ring", "Field", "Module",
                                        "Subgroup", "Ideal")),
            # Depth hint
            "has_hypothesis":      ":" in g and ("h" in g or "h₀" in g),
        }


# ==============================================================================
# Putnam tactic generator — the core change
# ==============================================================================

class PutnamTacticGenerator:
    """
    Five-tier tactic candidate generator tuned for Putnam proof structure.
    Called instead of _generate_smart_heuristics for Putnam problems.
    """

    # Retrieval blacklist (same as lie_search v87)
    _BLACKLIST = frozenset({
        "int_prod_range_pos", "refutationFor", "mkACProof", "getEqProof",
        "bddBelow_def", "cutExpand_add_right", "sdiff_ne_right",
        "lieCharpoly_monic", "card_support_binomial", "tfaeHaveCore",
        "of_projective", "StyleError", "zsmulArgs", "csSup_div",
        "divp_add_divp", "solution", "remainder",
        "degree_cubic", "nsmul_eq_nsmul", "natDegree_cubic",
        "mvPolynomial", "LieEquiv", "orbitRel", "MulRingNorm",
        "pointwise_smul_toAddSubgroup", "AddCommGroup.intIsScalarTower",
    })

    def __init__(self):
        self.analyzer = PutnamGoalAnalyzer()

    def candidates(self, goal_text: str, hints: list) -> list:
        """Return ordered tactic candidates for a Putnam goal state."""
        g = self.analyzer.analyse_putnam(goal_text)
        tactics = []

        # ── T1: Goal-shape openers ────────────────────────────────────────
        # These must come first — most Putnam goals start with ∀ or ∃ that
        # blocks all other tactics until discharged.
        if g["is_universal"]:
            tactics += ["intro x", "intro n", "intro h", "intros",
                        "intro x hx", "intro a b"]
        if g["is_existential"]:
            tactics += [
                "use 0", "use 1", "use ⟨_, _⟩",
                "refine ⟨?_, ?_⟩",
                "exact ⟨0, by norm_num⟩",
                "exact ⟨1, by norm_num⟩",
            ]
        if g["is_conjunction"]:
            tactics += ["constructor", "refine ⟨?_, ?_⟩", "exact ⟨_, _⟩"]
        if g["is_iff"]:
            tactics += ["constructor", "iff_intro"]
        if g["is_negation"]:
            tactics += ["intro h", "push_neg", "contrapose!", "by_contra h"]

        # ── T2: Destructuring / obtaining witnesses ────────────────────────
        tactics += [
            "obtain ⟨a, ha⟩ := h",
            "obtain ⟨a, b, hab⟩ := h",
            "rcases h with ⟨a, ha⟩",
            "rcases h with ⟨a, b, ha, hb⟩",
        ]

        # ── T3: Induction / case analysis ─────────────────────────────────
        if g["is_induction_ready"]:
            tactics += [
                "induction n with | zero => simp | succ n ih => ?_",
                "induction n",
                "cases n",
                "fin_cases n",
                "omega",
            ]
        if g["is_divisibility"]:
            tactics += [
                "exact dvd_refl _",
                "exact dvd_trans h₀ h₁",
                "apply Nat.dvd_antisymm",
                "omega",
                "norm_num",
            ]
        if g["is_prime"]:
            tactics += [
                "exact Nat.prime_def_lt_prime.mpr ⟨by norm_num, by norm_num⟩",
                "decide",
                "norm_num [Nat.Prime]",
                "apply Nat.Prime.dvd_of_dvd_pow",
            ]

        # ── T4: Domain closers (after the goal is simplified) ─────────────
        if g["has_real_analysis"]:
            tactics += [
                "exact h.continuous",
                "exact h.differentiableAt",
                "apply ContinuousOn.mono",
                "apply DifferentiableOn.mono",
                "simp [differentiableAt_const]",
                "fun_prop",
                "continuity",
                "linarith",
                "nlinarith",
            ]
        if g["has_number_theory"]:
            tactics += [
                "exact Nat.Coprime.symm h",
                "apply Nat.Coprime.mul_right",
                "simp [Nat.Coprime, Nat.gcd_comm]",
                "omega",
                "norm_num",
            ]
        if g["has_algebra"]:
            tactics += [
                "exact h.mul_mem ha hb",
                "apply Subgroup.mul_mem",
                "apply Ideal.add_mem",
                "ring",
                "group",
                "abel",
            ]

        # ── General closers always attempted ──────────────────────────────
        tactics += [
            "simp [*]", "simp_all", "aesop",
            "norm_num", "ring", "linarith",
            "exact h", "exact h₀", "assumption",
            "trivial", "rfl",
        ]

        # ── T5: Hint-based (filtered) ──────────────────────────────────────
        for i, h in enumerate(hints[:3]):
            if any(bl in h for bl in self._BLACKLIST):
                continue
            # Skip pure-internal Lean names
            if any(bad in h for bad in ("Meta.", "Tactic.", "Elab.", "Parser.")):
                continue
            is_top = (i == 0)
            tactics.append(f"apply {h}")
            if is_top:
                tactics += [f"exact {h}", f"refine {h} ?_", f"refine {h} ?_ ?_"]
            tactics.append(f"simp [{h}]")
            tactics.append(f"rw [{h}]")

        # Deduplicate preserving order
        seen: set = set()
        return [t for t in tactics if not (t in seen or seen.add(t))]


# ==============================================================================
# PutnamSearchAgent — overrides the three key methods
# ==============================================================================

class PutnamSearchAgent(RiemannSearchAgent):
    """
    Putnam-specialised version of RiemannSearchAgent.

    Inherits everything from lie_search v87 EXCEPT:
      - fast_solvers list
      - _generate_smart_heuristics
      - search() timeout and max_steps defaults
    """

    def __init__(self, hgcn_ckpt, llm_path, device="cuda"):
        super().__init__(hgcn_ckpt, llm_path, device)
        self.putnam_gen = PutnamTacticGenerator()
        # Putnam proofs are much longer — expand the state visit tolerance
        self.state_visits = defaultdict(int)

    # ------------------------------------------------------------------
    # Override: much richer heuristics for Putnam
    # ------------------------------------------------------------------
    def _generate_smart_heuristics(self, hints: list, goal_struct) -> list:
        """Replace miniF2F heuristics with Putnam-tuned tactic candidates."""
        # goal_struct.raw contains the full proof state text
        goal_text = goal_struct.raw if hasattr(goal_struct, "raw") else ""
        return self.putnam_gen.candidates(goal_text, hints)

    # ------------------------------------------------------------------
    # Override: search() with Putnam-tuned defaults
    # ------------------------------------------------------------------
    def search(self, theorem_decl: str, max_steps: int = 64):
        """
        Putnam search:
          - max_steps=64 (vs 40 for miniF2F) — proofs are longer
          - Fast solver list trimmed — omega/nlinarith rarely close Putnam
          - Per-tactic timeout raised to 30s
        """
        trace = []

        base = theorem_decl.split(":= by")[0].strip()
        base = re.sub(r"^\s*theorem\s+\S+\s*", "example ", base)

        # Extract goal type for domain analysis
        goal_text = base.split(":", 1)[-1].strip() if ":" in base else base
        goal_struct = self.analyzer.analyze(goal_text)

        env = LeanEnv(project_root)
        try:
            print("   [Env] Initializing...", flush=True)
            env.run_command("import Mathlib", timeout=120)
            env.run_command("import Mathlib.Tactic.LinearCombination", timeout=30)
            env.run_command(
                "open Nat Real Rat BigOperators Set Finset Function",
                timeout=30,
            )

            # ── Putnam fast solvers ───────────────────────────────────────
            # Only tactics that can actually close an abstract Putnam goal
            # without any prior simplification. omega/nlinarith removed.
            putnam_fast = []
            p = self.putnam_gen.analyzer.analyse_putnam(goal_text)

            if p["is_divisibility"] or p["is_prime"]:
                putnam_fast += ["decide", "norm_num", "omega"]
            if p["has_number_theory"]:
                putnam_fast += ["decide", "norm_num", "simp [Nat.Coprime]"]
            if p["has_algebra"]:
                putnam_fast += ["ring", "group", "abel", "simp"]

            # Universal fallbacks — cheap to try
            putnam_fast += ["rfl", "simp", "norm_num", "decide", "aesop"]

            # Deduplicate
            seen_fast: set = set()
            putnam_fast = [t for t in putnam_fast
                           if not (t in seen_fast or seen_fast.add(t))]

            for s in putnam_fast:
                res = env.run_command(f"{base} := by {s}", timeout=30)
                st, _ = self.parse_result(res)
                if st == "success":
                    print(f"   ✅ Fast solved: {s}", flush=True)
                    return {"status": "Success", "proof": [s], "trace": trace}
                elif st == "panic":
                    env = self._restart_env(env)

            # ── A* search with Putnam tactic distribution ─────────────────
            start_norm = self.normalize_goal(goal_text)
            goal_emb   = self.goal_encoder.encode(start_norm, mode="hyperbolic")
            target     = torch.zeros_like(goal_emb)

            frontier = []
            heapq.heappush(frontier,
                           (0.0, next(self.counter), [], start_norm, goal_emb))

            expanded     = 0
            failed_set   = set()  # (node_hash, tactic) to avoid retrying

            while frontier:
                cost, _, hist, gtext, emb = heapq.heappop(frontier)

                if len(hist) >= max_steps:
                    continue
                expanded += 1
                if expanded > 120:   # deeper budget than miniF2F's 50
                    break

                goal_s = self.analyzer.analyze(gtext)

                # Retrieve hints (conclusion-only encoding, from v87)
                conclusion = gtext.split("⊢")[-1].strip() if "⊢" in gtext else gtext
                hints_list = self.retrieve_theorems(conclusion, k=3)

                # Generate candidates via Putnam-tuned generator
                candidates = self.putnam_gen.candidates(gtext, hints_list)

                if hints_list:
                    print(f"      [RAG] {hints_list}", flush=True)

                # Also try LLM suggestions
                try:
                    llm_cands = self.llm.generate_candidates(
                        gtext, hints=hints_list, num=1
                    )
                except Exception:
                    llm_cands = []
                candidates = list(dict.fromkeys(candidates + (llm_cands or [])))

                node_hash = hash(gtext[:120])

                for tac in candidates:
                    if not tac:
                        continue
                    if (node_hash, tac) in failed_set:
                        continue

                    # Lie algebra trust score
                    try:
                        M     = self.tactic_operator(tac)
                        pred  = self.lie.apply_tactic(emb, M)
                        h_val = self.energy.h(pred, target)
                    except Exception:
                        pred, h_val = emb, 0.0

                    code = f"{base} := by {'; '.join(hist + [tac])}"
                    res  = env.run_command(code, timeout=30)  # 30s per tactic
                    st, out = self.parse_result(res)

                    trace.append({
                        "step": expanded, "depth": len(hist),
                        "tactic": tac, "status": st,
                        "goal": gtext[:120],
                        "trust": float(h_val),
                        "hints": hints_list,
                    })

                    if st == "success":
                        return {
                            "status": "Success",
                            "proof": hist + [tac],
                            "trace": trace,
                        }

                    if st == "continue":
                        ng = self.normalize_goal(out)
                        self.state_visits[ng] += 1
                        # Allow revisiting states up to 3× (vs 2× in miniF2F)
                        # because Putnam proofs may circle back through lemmas
                        if self.state_visits[ng] <= 3:
                            new_emb = self.goal_encoder.encode(
                                ng, mode=self.retrieval_mode
                            )
                            new_cost = len(hist) + 1 + h_val
                            heapq.heappush(
                                frontier,
                                (new_cost, next(self.counter),
                                 hist + [tac], ng, new_emb),
                            )
                    elif st == "error":
                        failed_set.add((node_hash, tac))

                    if st == "panic":
                        env = self._restart_env(env)

            return {"status": "Failed", "trace": trace}

        finally:
            env.close()


# ==============================================================================
# Convenience: drop-in replacement factory
# ==============================================================================

def make_agent(hgcn_ckpt: str, llm_path: str,
               mode: str = "auto", device: str = "cuda"):
    """
    Factory that returns the right agent for the dataset.

    mode = "putnam"  → PutnamSearchAgent
    mode = "minif2f" → RiemannSearchAgent (standard)
    mode = "auto"    → caller decides; returns PutnamSearchAgent by default
                       (safe to use on Archive problems too — richer tactics
                        don't hurt, they just expand the candidate set)
    """
    if mode in ("putnam", "auto"):
        return PutnamSearchAgent(hgcn_ckpt, llm_path, device)
    return RiemannSearchAgent(hgcn_ckpt, llm_path, device)
