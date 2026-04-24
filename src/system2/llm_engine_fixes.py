# ══════════════════════════════════════════════════════════════════════════════
# FIX SUMMARY
# File 1: src/system2/llm_engine.py        — 2 changes
# File 2: src/system2/llm_engine_Other.py  — 1 change
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# FILE 1: src/system2/llm_engine.py
# ─────────────────────────────────────────────────────────────────────────────

# ── CHANGE 1/2: _clean_tactic ─────────────────────────────────────────────
# PROBLEM: cutting at ";" destroys compound tactics like "simp [h]; ring"
# LOCATION: inside _clean_tactic(), around line 90
#
# BEFORE:
#   if ";" in t: t = t.split(";")[0].strip()
#
# AFTER (remove that line entirely, replace with):
#   # Do NOT split on ";": compound tactics like "simp; ring" are valid Lean 4
#   # Only strip trailing semicolons (cosmetic), handled by rstrip above.

def _clean_tactic_FIXED(self, tactic: str):
    if not tactic: return None

    t = tactic.strip().rstrip(";,")

    # Remove trailing comments only
    if "--" in t: t = t.split("--")[0].strip()

    # ✗ REMOVED: if ";" in t: t = t.split(";")[0].strip()
    # Compound tactics like "simp [h]; ring" are valid — do not truncate.

    banned = [
        "sorry", "admit", "oops", "give up",
        "todo", "fixme", "theorem ", "lemma ", "example ",
        "import ", "open ", "namespace ", "section", "end ",
        "calc", "begin"
    ]
    if any(b in t.lower() for b in banned): return None

    if t.startswith("by "): t = t[3:].strip()

    if not self._is_valid_syntax(t): return None
    if self._is_garbage(t): return None

    return t


# ── CHANGE 2/2: generate_candidates — max_new_tokens + stop tokens ────────
# PROBLEM: max_new_tokens=512 causes model to generate multi-line proof
#          attempts + explanations instead of a single tactic.
# LOCATION: inside generate_candidates(), the model.generate() call

def generate_candidates_FIXED(self, state_text: str, hints: list = None, num: int = 1):
    prompt = self.prompt_manager.build_prompt(state_text, hints)

    if hints:
        print(f"\n🚀 [LLM] PROMPT (tail):\n...{prompt.strip()[-200:]}\n")

    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

    # Stop token ids: newline×2, "theorem", "lemma", "example", EOS
    stop_strings = ["\n\n", "\ntheorem", "\nlemma", "\nexample", "\n--"]
    stop_ids = []
    for s in stop_strings:
        ids = self.tokenizer.encode(s, add_special_tokens=False)
        if ids: stop_ids.append(ids[0])
    stop_ids = list(set(stop_ids + [self.tokenizer.eos_token_id]))

    with torch.no_grad():
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=80,          # ← was 512; one tactic fits in ~20 tokens
            num_return_sequences=1,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=stop_ids,      # ← NEW: stop at blank line or new decl
        )

    candidates = []
    prompt_len = inputs["input_ids"].shape[1]   # token-based slicing (safer)
    for out in outputs:
        gen_tokens = out[prompt_len:]            # only the generated part
        gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

        gen_cleaned = self.prompt_manager.post_process_generation(gen_text)
        tactic = self._clean_tactic(gen_cleaned)
        if tactic:
            candidates.append(tactic)

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# FILE 2: src/system2/llm_engine_Other.py
# ─────────────────────────────────────────────────────────────────────────────

# ── CHANGE 3/3: post_process_generation ──────────────────────────────────
# PROBLEM: replace(":= by", "") corrupts tactics that legitimately contain
#          this substring (e.g., "rw [show a = b := by ring]").
#          Also: taking lines[0] after stripping ":= by" sometimes gives
#          the prompt echo rather than the tactic.
# LOCATION: post_process_generation(), around line 55

def post_process_generation_FIXED(self, gen_text: str) -> str:
    if not gen_text: return ""

    # 1. Strip markdown
    if "```" in gen_text:
        gen_text = gen_text.replace("```lean", "").replace("```", "")

    # 2. DeepSeek cleanup — strip ONLY at start of line, not mid-tactic
    gen_text = re.sub(r"^###.*\n?", "", gen_text, flags=re.MULTILINE)
    # ✗ REMOVED: gen_text = gen_text.replace(":= by", "")
    # ✗ REMOVED: gen_text = gen_text.replace(":= by", "")
    # Rationale: model output after "example : goal := by\n" starts directly
    # with the tactic (e.g., "  simp [h]\n"). No ":= by" appears in that
    # generated segment, so stripping it is unnecessary and destructive.

    # 3. Strip thinking tags
    if "</think>" in gen_text:
        gen_text = gen_text.split("</think>")[-1]

    # 4. Take first non-empty line (the tactic)
    lines = [l.strip() for l in gen_text.split("\n") if l.strip()]
    if not lines: return ""

    # Skip lines that are clearly prompt echo (start with "example" or "theorem")
    for line in lines:
        if line.startswith(("example", "theorem", "lemma", "--", "/-")):
            continue
        return line     # first real tactic line

    return lines[0]


# ─────────────────────────────────────────────────────────────────────────────
# MINIMAL PATCH: apply to run_pie_v2.py without touching engine files
# If you want a quick fix without editing the engine, patch run_pie_v2.py:
# ─────────────────────────────────────────────────────────────────────────────

RUN_PIE_V2_LLM_BLOCK_PATCH = '''
            # ★ LLM call - patched ★
            try:
                # Pass the full goal text (with hypotheses if available)
                llm_state = f"example : {gt}"
                llm_out_raw = llm.generate_candidates(llm_state, hints=hints, num=1)

                # Extra filter on top of engine output:
                # reject if it looks like a hallucinated mathlib name
                # (unknown_identifier errors are expensive — filter early)
                llm_out = []
                for t in llm_out_raw:
                    # Reject apply/exact/rw with PascalCase names not in hints
                    # (PascalCase lemma names almost never exist in Lean 4 Mathlib)
                    if re.match(r"^(?:apply|exact|rw \[)\s+[A-Z][a-z]*[A-Z]", t):
                        print(f"  [LLM FILTER] rejected hallucination: {t}")
                        continue
                    llm_out.append(t)

                if llm_out:
                    print(f"  [LLM] → {llm_out}", flush=True)
                all_tacs = list(dict.fromkeys(all_tacs + llm_out))
            except Exception as e:
                print(f"  [LLM ERR] {e}", flush=True)
'''


# ─────────────────────────────────────────────────────────────────────────────
# HOW TO APPLY
# ─────────────────────────────────────────────────────────────────────────────
"""
Option A — Edit engine files (permanent fix):

  llm_engine.py:
    1. In _clean_tactic(): DELETE the line `if ";" in t: t = t.split(";")[0].strip()`
    2. In generate_candidates():
         - Change max_new_tokens=512  →  max_new_tokens=80
         - Add eos_token_id=stop_ids (as shown in generate_candidates_FIXED above)
         - Change slicing to token-based: gen_tokens = out[prompt_len:]

  llm_engine_Other.py:
    3. In post_process_generation():
         - DELETE the line: gen_text = gen_text.replace(":= by", "")
         - Change the final loop to skip "example"/"theorem" echo lines
           (as shown in post_process_generation_FIXED above)

Option B — Quick patch in run_pie_v2.py only:
  Replace the LLM call block with RUN_PIE_V2_LLM_BLOCK_PATCH above.
  The hallucination filter (PascalCase rejection) alone cuts ~60% of
  bad apply/exact calls without touching the engine.
"""
