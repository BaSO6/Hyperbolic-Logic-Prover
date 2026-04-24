import os
import torch
import gzip
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
# These features were just generated with preprocess_integrated.py; they are perfect BERT vectors
FEAT_PATH = os.path.join(DATA_DIR, "node_features_euclidean.pt") 
LIST_PATH = os.path.join(DATA_DIR, "node_list.pkl.gz")
BERT_PATH = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")
if not os.path.exists(BERT_PATH): BERT_PATH = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Testing Pure BERT Baseline on {device}...")

    # 1. Load authoritative name list
    print("📂 Loading node list...")
    if not os.path.exists(LIST_PATH):
        print("❌ node_list.pkl.gz missing! Run preprocess_integrated.py first.")
        return
    with gzip.open(LIST_PATH, "rb") as f:
        idx_to_name = pickle.load(f)

    # 2. Load BERT features (As Graph Embedding)
    print("⚓ Loading BERT features (The Knowledge Base)...")
    if not os.path.exists(FEAT_PATH):
        print("❌ node_features_euclidean.pt missing!")
        return
    # Load and normalize to ensure cosine similarity is calculated correctly
    node_feats = torch.load(FEAT_PATH, map_location=device)
    node_feats = torch.nn.functional.normalize(node_feats, p=2, dim=1)
    
    # 3. Load BERT model (For Encoding Queries)
    print("🧠 Loading BERT Model...")
    bert = SentenceTransformer(BERT_PATH, device=device)

    # --- Test Cases ---
    test_cases = [
        {
            "name": "IMO 1959 (GCD)", 
            "goal": "Nat.gcd (21*n + 4) (14*n + 3) = 1"
        },
        {
            "name": "AIME 1983 (Log)", 
            "goal": "Real.log w / Real.log x = 24"
        },
        {
            "name": "AMC 12 (Rational)",
            "goal": "((1 / 2 + 1 / 3 + 1 / 7 + 1 / n) : ℚ).den = 1"
        }
    ]

    print("\n" + "="*60)
    for case in test_cases:
        print(f"🔎 Testing: {case['name']}")
        print(f"📝 Goal: {case['goal']}")
        
        with torch.no_grad():
            # Encode Query
            query_emb = bert.encode(case['goal'], convert_to_tensor=True)
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=0)
            
            # Cosine Similarity = Dot Product (because normalized)
            # [1, D] @ [N, D].T -> [1, N]
            scores = torch.matmul(query_emb.unsqueeze(0), node_feats.T)
            
            # Top K
            vals, indices = torch.topk(scores, k=15, dim=1, largest=True)
            
            print("\n🏆 Top 15 Retrieved (BERT Euclidean):")
            for i, idx in enumerate(indices[0]):
                name = idx_to_name[idx.item()]
                score = vals[0][i].item()
                
                # Highlight keywords
                highlight = "  "
                lower_name = name.lower()
                if "sub_mul" in lower_name or "log_mul" in lower_name or "rat.add" in lower_name:
                    highlight = "🔥🔥"
                elif "gcd" in lower_name or "log" in lower_name or "rat" in lower_name:
                    highlight = "✨"
                    
                print(f"   {highlight} [{i+1}] {name} (score={score:.4f})")
        print("="*60)

if __name__ == "__main__":
    main()