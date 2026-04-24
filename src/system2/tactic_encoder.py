# ==========================================
# Filename: src/system2/tactic_encoder.py
# Version: v2.4 (Deterministic, Canonical, Lie-Safe)
# ==========================================

import os
import sys
import re
import hashlib
import warnings
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Environment Control
# ---------------------------------------------------------------------
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------
# Tactic Canonicalization
# ---------------------------------------------------------------------

def canonicalize_tactic(text: str) -> str:
    """
    Normalize Lean tactics into a stable form:
      - Strip leading/trailing whitespace
      - Merge redundant whitespace
      - Standardize spaces around brackets
      - Remove trailing semicolons
    """
    if not text:
        return ""

    t = text.strip()

    while t.endswith(";"):
        t = t[:-1].strip()

    t = re.sub(r"\s+", " ", t)

    t = re.sub(r"\(\s*", "(", t)
    t = re.sub(r"\s*\)", ")", t)
    t = re.sub(r"\[\s*", "[", t)
    t = re.sub(r"\s*\]", "]", t)

    return t


# ---------------------------------------------------------------------
# Tactic Encoder
# ---------------------------------------------------------------------

class TacticEncoder:
    def __init__(self, model_name_or_path="all-MiniLM-L6-v2", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_dummy = False
        self.model = None
        self.dim = 384

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))

        possible_paths = [
            os.path.join(project_root, "models", "all-MiniLM-L6-v2"),
            "models/all-MiniLM-L6-v2",
            model_name_or_path,
        ]

        target_path = model_name_or_path
        for p in possible_paths:
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
                target_path = p
                break

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(target_path, device=self.device)
            self.model.eval()
            print("✅ TacticEncoder loaded.")
        except Exception as e:
            warnings.warn(
                f"⚠️ TacticEncoder failed to load ({e}). Using deterministic dummy encoder."
            )
            self.use_dummy = True

    # -----------------------------------------------------------------
    # Dummy deterministic embedding
    # -----------------------------------------------------------------

    def _dummy_encode(self, text: str) -> torch.Tensor:
        canon = canonicalize_tactic(text)
        h = hashlib.md5(canon.encode("utf-8")).hexdigest()
        seed = int(h[:8], 16)

        g = torch.Generator(device=self.device)
        g.manual_seed(seed)

        vec = torch.randn(1, self.dim, generator=g, device=self.device)
        return F.normalize(vec, p=2, dim=1)

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def encode(self, text):
        """
        Input:
            - str | list[str]
        Output:
            - Tensor (batch, 384), normalized
        """
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts = list(text)
            single = False

        texts = [canonicalize_tactic(t) for t in texts]

        if self.use_dummy or self.model is None:
            embs = [self._dummy_encode(t) for t in texts]
            out = torch.cat(embs, dim=0)
            return out if not single else out[:1]

        try:
            with torch.no_grad():
                emb = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=self.device,
                )
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)

            emb = emb.float()
            emb = F.normalize(emb, p=2, dim=1)
            return emb if not single else emb[:1]

        except Exception as e:
            warnings.warn(f"TacticEncoder encode error: {e}. Falling back to zeros.")
            z = torch.zeros((len(texts), self.dim), device=self.device)
            return z if not single else z[:1]


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    enc = TacticEncoder(device="cpu")
    v1 = enc.encode("rw   [ add_comm ] ;")
    v2 = enc.encode("rw [add_comm]")
    sim = F.cosine_similarity(v1, v2).item()
    print("Canonicalization similarity:", sim)
    assert sim > 0.99