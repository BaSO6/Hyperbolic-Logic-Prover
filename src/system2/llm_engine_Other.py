# ==============================================================================
# Filename: src/system2/llm_engine_Other.py
# Version: v3.0 (DeepSeek Trigger Fix & VLLM Integration)
#
# CRITICAL FIXES:
# 1. [Prompt] Appends ':= by' to DeepSeek prompts to trigger tactic generation.
# 2. [Cleaning] Removes common artifacts like '### Proof' and markdown blocks.
# 3. [Debug] Prints prompts to stdout to verify LLM is actually being called.
# ==============================================================================

import re
import sys
import os
import torch

class PromptManager:
    """
    Manages prompt templates and post-processing for different model families.
    """
    def __init__(self, model_path: str, verbose: bool = True):
        self.model_type = self._infer_model_type(model_path)
        self.verbose = verbose
        print(f"[PromptManager] Detected strategy: {self.model_type}")
        if self.verbose:
            print(f"[PromptManager] Verbose logging is ON.")

    def _infer_model_type(self, path: str) -> str:
        path = path.lower()
        if "deepseek-prover" in path:
            return "deepseek_prover"  # V1.5 Needs ':= by' trigger
        elif "internlm" in path:
            return "internlm_prover"
        elif "qwen" in path or "math" in path:
            return "chat_math"
        elif "llama" in path:
            return "chat_generic"
        else:
            return "default"

    def build_prompt(self, state_text: str, hints: list = None) -> str:
        """
        Dispatches to the correct prompt builder based on model type.
        """
        # 1. Format Hints
        hint_str = ""
        if hints:
            # Filter out non-string hints or too short ones
            clean_hints = [h for h in hints if h and isinstance(h, str) and len(h) > 2][:5]
            if clean_hints:
                # Lean comment style is universally safe
                hint_str = f"/- Hints: {', '.join(clean_hints)} -/\n"

        # 2. Select Template
        prompt = ""
        if self.model_type == "deepseek_prover":
            prompt = self._build_deepseek_prover(state_text, hint_str)
        elif self.model_type == "internlm_prover":
            prompt = self._build_internlm(state_text, hint_str)
        else:
            prompt = self._build_chat_instruct(state_text, hint_str)
            
        # [DEBUG] Critical: Verify what goes into the LLM
        if self.verbose:
            print("\n" + "="*20 + " [LLM Prompt Check] " + "="*20)
            # Print only the end of the prompt to see the trigger
            preview = prompt.strip()[-300:] if len(prompt) > 300 else prompt.strip()
            print(f"...{preview}")
            print("="*60 + "\n")

        return prompt

    def post_process_generation(self, gen_text: str) -> str:
        """
        Cleans up model output.
        """
        if not gen_text: return ""

        original = gen_text

        # 1. Strip Markdown Code Blocks
        if "```" in gen_text:
            gen_text = gen_text.replace("```lean", "").replace("```", "")
            
        # 2. DeepSeek Specific Cleanup
        # It often outputs "### Proof" or repeats ":= by"
        gen_text = gen_text.replace("### Proof", "")
        gen_text = gen_text.replace(":= by", "")
        
        # 3. Strip Thinking Tags (for R1/Reasoning models)
        if "</think>" in gen_text:
            gen_text = gen_text.split("</think>")[-1]
            
        # 4. Take the first valid line
        lines = [l.strip() for l in gen_text.split("\n") if l.strip()]
        if not lines: return ""
        
        final_tactic = lines[0]
        
        # [DEBUG] Verify output
        if self.verbose:
            print(f"[LLM Extracted]: {final_tactic}")

        return final_tactic

    # --- Templates ---

    def _build_deepseek_prover(self, state: str, hints: str) -> str:
        """
        DeepSeek-Prover V1.5 is a completion model.
        CRITICAL: It needs ':= by' at the end to know it should generate a tactic.
        """
        state = state.strip()
        # If the state already has the trigger, don't add it again
        if state.endswith(":= by"):
            return f"{hints}{state} "
        else:
            # Force the trigger
            return f"{hints}{state} := by\n"

    def _build_internlm(self, state: str, hints: str) -> str:
        return f"{hints}{state}\n-- Proof step:"

    def _build_chat_instruct(self, state: str, hints: str) -> str:
        return (
            "You are a Lean 4 expert. Given the state, output the next tactic.\n"
            "Output ONLY the code. Do not explain.\n\n"
            f"{hints}"
            "State:\n"
            f"{state}\n\n"
            "Tactic:"
        )

# ==============================================================================
# LLM Engine (VLLM Wrapper)
# ==============================================================================
class LLMEngine:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.prompt_mgr = PromptManager(model_path, verbose=True)
        
        print(f"[LLMEngine] Loading model from: {model_path}")
        try:
            from vllm import LLM, SamplingParams
            # Adjust gpu_memory_utilization if OOM occurs (e.g., 0.8 or 0.6)
            self.llm = LLM(
                model=model_path, 
                trust_remote_code=True, 
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                max_model_len=2048
            )
            self.sampling_params = SamplingParams(
                temperature=0.7, 
                top_p=0.9,
                max_tokens=256
            )
            self.backend = "vllm"
            print("[LLMEngine] VLLM loaded successfully.")
        except ImportError:
            print("[LLMEngine] VLLM not found. Falling back to Transformers (Slow!).")
            self.backend = "transformers"
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.float16,
                device_map="auto"
            )

    def generate_candidates(self, state, hints=[], num=1):
        prompt = self.prompt_mgr.build_prompt(state, hints)
        candidates = []
        
        try:
            if self.backend == "vllm":
                # VLLM Generation
                outputs = self.llm.generate([prompt] * num, self.sampling_params, use_tqdm=False)
                for output in outputs:
                    text = output.outputs[0].text
                    tac = self.prompt_mgr.post_process_generation(text)
                    if tac: candidates.append(tac)
            else:
                # Transformers Generation (Fallback)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=128, 
                        num_return_sequences=num,
                        do_sample=True,
                        temperature=0.7
                    )
                for out_seq in outputs:
                    text = self.tokenizer.decode(out_seq, skip_special_tokens=True)
                    # Strip input prompt from output
                    text = text[len(prompt):]
                    tac = self.prompt_mgr.post_process_generation(text)
                    if tac: candidates.append(tac)
                    
        except Exception as e:
            print(f"[LLMEngine] Generation Error: {e}")
            
        return list(set(candidates))