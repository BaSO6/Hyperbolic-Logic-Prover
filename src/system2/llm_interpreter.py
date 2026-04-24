# ==========================================
# Filename: src/system2/llm_interpreter.py
# Version: v7.0 (Clean & Robust)
# Fixes: 
#   1. [Critical] Removed redundant model.generate calls
#   2. [Fix] Fixed indentation errors and non-standard spaces
#   3. [Feature] Enhanced anti-interference capability for JSON extraction
# ==========================================

import os
import sys
import re
import json
import torch

# [Critical] Must be set first to ensure the use of domestic mirrors
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# [🚨 Critical Patch] Resolve PyTorch version compatibility issues
try:
    import torch.utils._pytree
    if not hasattr(torch.utils._pytree, 'register_pytree_node'):
        def _safe_register_pytree_node(cls, flatten_fn, unflatten_fn, serialized_type_name=None):
            return torch.utils._pytree._register_pytree_node(cls, flatten_fn, unflatten_fn)
        torch.utils._pytree.register_pytree_node = _safe_register_pytree_node
except ImportError:
    pass

from transformers import AutoModelForCausalLM, AutoTokenizer

# Path settings
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

class StrictInterpreter:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        print(f"🤖 Loading DeepSeek Reasoning Engine: {model_path}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, 
                device_map="auto",
                trust_remote_code=True
            )
            self.model.eval()
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise e
        
    def generate_label(self, cluster_id, examples):
        """
        Generate strict JSON format labels
        """
        # 1. Construct Few-Shot samples
        ex_str = ""
        for i, ex in enumerate(examples[:3]): 
            ex_str += f"Sample {i+1}:\n  Theorem: {ex.get('source_def', 'N/A')}\n  Uses Lemma: {ex.get('target_def', 'N/A')}\n\n"

        # 2. Construct forced Prompt
        prompt = (
            f"You are a mathematical logic expert. Analyze the following transformation patterns from Lean Mathlib.\n"
            f"These samples belong to Cluster {cluster_id}.\n\n"
            f"{ex_str}"
            f"--- TASK ---\n"
            f"Identify the specific logical operation or tactic being used.\n"
            f"Output result strictly in JSON format. NO conversational text.\n\n"
            f"```json\n"
            f"{{\n"
            f"  \"label\": \"<Short Name (3-5 words)>\",\n"
            f"  \"description\": \"<One sentence explanation>\",\n"
            f"  \"tactic_suggestion\": \"<Likely Lean tactic, e.g. rw, simp, apply>\"\n"
            f"}}\n"
            f"```\n"
            f"JSON Output:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # [Fix] Keep only one generate call
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=128,
                temperature=0.1,         # Low temperature to ensure format stability
                do_sample=False,         # Disable sampling to pursue determinism
                repetition_penalty=1.1,  # Penalize repetitive content
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
        return self._parse_json(output_text, cluster_id)

    def _parse_json(self, text, cid):
        """
        [Enhanced Version] Parse and clean LLM output (Anti-insanity version)
        """
        data = None
        
        # Strategy 1: Prioritize standard Markdown JSON code blocks
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
            except json.JSONDecodeError:
                pass 

        # Strategy 2: The Stack Method - The most robust extraction method
        if data is None:
            start_idx = text.find('{')
            if start_idx != -1:
                brace_count = 0
                json_str = ""
                for char in text[start_idx:]:
                    json_str += char
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                    
                    # When count reaches zero, it means a complete closed JSON object is found
                    if brace_count == 0:
                        try:
                            data = json.loads(json_str)
                            break
                        except json.JSONDecodeError:
                            # Might be an internal bracket issue, continue trying
                            continue
        
        # Strategy 3: Regex fallback (Handling cases where JSON format is completely corrupted)
        if data is None:
            label_match = re.search(r'"label":\s*"(.*?)"', text)
            label = label_match.group(1) if label_match else "General Logic Step"
            
            tactic_match = re.search(r'"tactic_suggestion":\s*"(.*?)"', text)
            tactic = tactic_match.group(1) if tactic_match else "simp"
            
            data = {
                "label": label, 
                "description": "Auto-generated description via regex fallback.", 
                "tactic_suggestion": tactic
            }

        # Final cleaning: Ensure critical fields exist
        if "label" not in data: data["label"] = "Unknown Pattern"
        if "tactic_suggestion" not in data: data["tactic_suggestion"] = "simp"
        
        return data

def main():
    MODEL_DIR = os.path.join(project_root, "models/deepseek-math-7b-rl")
    CLUSTER_FILE = os.path.join(project_root, "cluster_data_hierarchical.json")
    
    if not os.path.exists(CLUSTER_FILE):
        print(f"❌ Cluster file not found: {CLUSTER_FILE}")
        return
        
    try:
        interpreter = StrictInterpreter(MODEL_DIR)
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return
    
    with open(CLUSTER_FILE, "r") as f:
        cluster_data = json.load(f)
        
    print(f"\n🧠 Starting Strict Semantic Labeling on {len(cluster_data)} clusters...\n")
    
    full_report = {}
    nav_map = {} 
    
    for i, (cid, examples) in enumerate(cluster_data.items()):
        print(f"Processing Cluster {cid} ({i+1}/{len(cluster_data)})...", end="\r")
        try:
            info = interpreter.generate_label(cid, examples)
            
            full_report[cid] = info
            nav_map[cid] = {
                "label": info['label'],
                "tactic": info.get('tactic_suggestion', 'simp')
            }
            
            if i % 5 == 0:
                print(f"✅ Cls {cid}: {info['label']:<40}")
            
        except Exception as e:
            print(f"❌ Error on {cid}: {e}")
            nav_map[cid] = {"label": "Unknown Logic", "tactic": "simp"}
            
    report_path = os.path.join(project_root, "deep_semantic_report.json")
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=4)
        
    map_path = os.path.join(project_root, "cluster_nav_map.json")
    with open(map_path, "w") as f:
        json.dump(nav_map, f, indent=4)
        
    print(f"\n\n🎉 Done! Generated standardized data:")
    print(f"   1. Human Report: {report_path}")
    print(f"   2. Machine Map : {map_path} (Ready for System 2)")

if __name__ == "__main__":
    main()