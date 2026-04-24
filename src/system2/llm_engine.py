# ==========================================
# File: src/system2/llm_engine.py
# Version: v8.0
#
# Changes vs v7.5:
#   FIX-1: _clean_tactic — removed ";" split that truncated compound tactics
#           like "simp [h]; ring" to just "simp [h]", preventing ring from
#           ever being tried. Trailing lone ";" is still stripped by rstrip.
#   FIX-2: generate_candidates — max_new_tokens reduced 512 → 80.
#           512 tokens caused the model to generate multi-line proof attempts
#           and explanations; only the first line was used, so all that compute
#           was wasted. One Lean tactic fits in ~10-25 tokens.
#   FIX-3: generate_candidates — added eos_token_id stop tokens so generation
#           halts at a blank line or a new declaration header ("\ntheorem",
#           "\nlemma", "\nexample", "\n--"), not just at EOS.
#   FIX-4: generate_candidates — switched from text-based prompt stripping
#           (text[len(prompt):]) to token-based slicing (out[prompt_len:]).
#           Text slicing is fragile when the tokenizer adds/removes whitespace;
#           token slicing is always exact.
# ==========================================

import os
import re
import torch
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix Import Path for PromptManager
try:
    from .llm_engine_Other import PromptManager
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    try:
        from llm_engine_Other import PromptManager
    except ImportError:
        print("❌ Critical Error: Could not import PromptManager from llm_engine_Other.py")
        raise

# ---- Force HuggingFace Offline Mode ----
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


class LLMEngine:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        model_path = os.path.abspath(os.path.expanduser(model_path))

        print(f"[LLM] Loading model (offline): {model_path}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[LLMEngine] Model path does not exist: {model_path}")

        # --- 1. Initialize Prompt Manager ---
        self.prompt_manager = PromptManager(model_path)

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        # -------------------------------
        # Tokenizer
        # -------------------------------
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"❌ Tokenizer load failed: {e}")
            raise e

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Give PromptManager access to tokenizer for apply_chat_template
        if hasattr(self.prompt_manager, "set_tokenizer"):
            self.prompt_manager.set_tokenizer(self.tokenizer)

        # Build stop-token ids once at init so generate_candidates doesn't
        # recompute them on every call.
        stop_strings = ["\n\n", "\ntheorem", "\nlemma", "\nexample", "\n--"]
        stop_ids = []
        for s in stop_strings:
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if ids:
                stop_ids.append(ids[0])
        self._stop_ids = list(set(stop_ids + [self.tokenizer.eos_token_id]))

        # -------------------------------
        # Model — auto 4-bit for large models that exceed VRAM
        # -------------------------------
        # Estimate model size from weight files on disk
        _weight_bytes = sum(
            os.path.getsize(os.path.join(model_path, f))
            for f in os.listdir(model_path)
            if f.endswith((".safetensors", ".bin"))
            and os.path.isfile(os.path.join(model_path, f))
        ) if os.path.isdir(model_path) else 0
        _model_gb  = _weight_bytes / 1e9
        _vram_gb   = (torch.cuda.get_device_properties(0).total_memory / 1e9
                      if torch.cuda.is_available() else 0)
        # Use 4-bit quantization if the model needs more than 70% of VRAM in BF16
        # (BF16 stores ~2 bytes/param, so _model_gb ≈ params × 2)
        _use_4bit  = _vram_gb > 0 and _model_gb > _vram_gb * 0.70
        if _use_4bit:
            print(f"   [LLM] Model {_model_gb:.0f}GB > {_vram_gb*0.70:.0f}GB "
                  f"(70% of {_vram_gb:.0f}GB VRAM) → enabling 4-bit quantization")
        else:
            print(f"   [LLM] Model {_model_gb:.0f}GB fits in VRAM → BF16 full precision")

        _load_kwargs = dict(
            torch_dtype=dtype,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )
        if _use_4bit:
            _load_kwargs["load_in_4bit"]          = True
            _load_kwargs["bnb_4bit_compute_dtype"] = dtype
            _load_kwargs["bnb_4bit_use_double_quant"] = True

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, **_load_kwargs
            )
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            raise e

        self.model.eval()

    # --------------------------------------------------
    # Internal filters
    # --------------------------------------------------

    def _is_garbage(self, text: str) -> bool:
        if not text:
            return True

        # Hard ban on natural language prefixes often output by chat models
        bad_prefixes = (
            "to ", "we ", "first ", "next ", "finally ",
            "let ", "consider ", "note ", "because ",
            "therefore ", "by ", "use ", "assume ",
            "sure", "here is", "below is", "okay", "certainly",
        )

        low = text.lower()
        if any(low.startswith(p) for p in bad_prefixes):
            return True
        if "the goal" in low:
            return True

        # Must contain at least one letter
        if not re.search(r"[A-Za-z]", text):
            return True

        return False

    def _is_valid_syntax(self, tactic: str) -> bool:
        # Basic parenthesis balancing
        if tactic.count("(") != tactic.count(")"):
            return False
        if tactic.count("[") != tactic.count("]"):
            return False
        if tactic.count("{") != tactic.count("}"):
            return False

        # Lean syntax sanity checks
        if tactic.startswith("|") or tactic.startswith("?"):
            return False
        if tactic.endswith(":=") or tactic.endswith(" by"):
            return False

        return True

    def _clean_tactic(self, tactic: str):
        if not tactic:
            return None

        t = tactic.strip().rstrip(";,")

        # Remove trailing comments only
        if "--" in t:
            t = t.split("--")[0].strip()

        # FIX-1: Do NOT split on ";".
        # Compound tactics like "simp [h]; ring" or "intro h; exact h" are
        # valid Lean 4 and should be tried as a unit. The old split(";"')[0]
        # silently truncated them, so the second tactic was never executed.
        # Trailing semicolons are already removed by rstrip above.

        banned = [
            "sorry", "admit", "oops", "give up",
            "todo", "fixme", "theorem ", "lemma ", "example ",
            "import ", "open ", "namespace ", "section", "end ",
            "calc", "begin",
        ]
        if any(b in t.lower() for b in banned):
            return None

        if t.startswith("by "):
            t = t[3:].strip()

        if not self._is_valid_syntax(t):
            return None
        if self._is_garbage(t):
            return None

        # NOTE: No restrictive ASCII/regex check here — Lean 4 tactics
        # legitimately use Unicode: ∧ ∨ ≠ ∈ ∩ ∪ ℕ ℝ ⟨ ⟩ etc.

        return t

    # --------------------------------------------------
    # Main API
    # --------------------------------------------------

    def generate_candidates(self, state_text: str, hints: list = None, num: int = 1):
        """
        Generate Lean 4 tactic candidates given a proof state and optional hints.

        state_text : full proof state string including hypotheses and goal,
                     e.g. "x : ℝ\\nh₀ : 2 * x = 6\\n⊢ x = 3"
        hints      : list of Mathlib theorem names from the retriever
        num        : number of candidates requested (currently always 1 from
                     the single greedy decode; kept for API compatibility)
        """
        # 1. Build prompt
        prompt = self.prompt_manager.build_prompt(state_text, hints)

        # Debug print — shows what the LLM actually receives
        if hints:
            print("\n" + "=" * 50)
            print(f"🚀 [LLM_DEBUG] ({self.prompt_manager.model_type}) PROMPT:")
            print(f"{prompt.strip()[-300:]}...")   # tail of prompt is most informative
            print("=" * 50 + "\n")

        # 2. Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        # FIX-4: record prompt length in tokens for exact slicing later
        prompt_len = inputs["input_ids"].shape[1]

        # 3. Generate
        # FIX-2: max_new_tokens=80 — one Lean tactic is ~10-25 tokens;
        #         80 is generous headroom without letting the model ramble.
        # FIX-3: eos_token_id=self._stop_ids — halt at blank line or new
        #         declaration so we never decode past the first tactic.
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=80,
                num_return_sequences=1,
                do_sample=False,            # greedy decoding for stability
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self._stop_ids,
            )

        candidates = []
        for out in outputs:
            # FIX-4: token-based slicing — always correct regardless of how
            # the tokenizer handles whitespace around the prompt boundary.
            gen_tokens = out[prompt_len:]
            gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

            # 4. Post-process (handles DeepSeek ":= by" echo, think tags, etc.)
            gen_cleaned = self.prompt_manager.post_process_generation(gen_text)

            # 5. Final cleaning & validation
            tactic = self._clean_tactic(gen_cleaned)
            if tactic:
                candidates.append(tactic)

        return candidates