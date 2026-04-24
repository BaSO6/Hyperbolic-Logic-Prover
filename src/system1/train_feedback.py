# ==========================================
# Filename: src/system1/train_feedback.py
# Version: v3.0 (Compatible with ICML Residual Model)
# ==========================================

import os
import sys
import json
import gzip
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Path adaptation
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.system1.manifold_math import PoincareBall
# Must import the new model definition
from src.system1.train_final import FinalHGCN

# Configuration
DATA_DIR = os.path.join(project_root, "data")
MODEL_PATH = os.path.join(DATA_DIR, "hgcn_final.pth")
SAVE_PATH = os.path.join(DATA_DIR, "hgcn_refined.pth")
NODE_MAP_PATH = os.path.join(DATA_DIR, "node_text_map.pkl.gz")
NODE_EMB_PATH = os.path.join(DATA_DIR, "node_embeddings.pt")
# Use the new node_list path
NODE_LIST_PATH = os.path.join(DATA_DIR, "node_list.pkl.gz")

# --- CORE MODIFICATION: Wrapper for Residual Layer adaptation ---
class AlignmentModel(nn.Module):
    def __init__(self, original_model, c=1.0):
        super().__init__()
        self.manifold = PoincareBall(c)
        
        # Extract the semantic branch from the Residual Layer
        # Note: During the fine-tuning phase, we are tuning "how to map the Query into hyperbolic space"
        # The original model uses BERT -> Linear -> Hyperbolic
        # Here, semantic_proj is that Linear layer
        
        self.proj = original_model.layer.semantic_proj
        self.scale = original_model.layer.scale
        
        # Freeze the structural branch (we only fine-tune semantic mapping and don't want to destroy structural knowledge)
        for param in original_model.layer.structure_proj.parameters():
            param.requires_grad = False
        for param in original_model.layer.graph_conv.parameters():
            param.requires_grad = False
        
        # Does the gating also need fine-tuning? Maybe. For simplicity, we assume the Query is purely semantic,
        # so we are actually fine-tuning semantic_proj to make its output vector closer to the target.
        
    def forward(self, x):
        # Simulate the forward propagation of the Semantic Branch
        # Note: Here we ignore the structural branch (since the Query has no neighbors)
        # This is equivalent to the case where alpha=1
        
        z = self.proj(x) # [B, Out]
        
        # Does z directly enter ExpMap? No, the original model has a Gate.
        # But for a pure Query, there is no structural information, so the Gate's input is only the Semantic part.
        # To maintain consistency, we assume the Gate will lean towards the Semantic side.
        # For simplicity, we map directly.
        
        x_norm = z.norm(dim=-1, keepdim=True) + 1e-8
        x_unit = z / x_norm
        target_radius = 0.9 * torch.tanh(self.scale)
        
        return self.manifold.expmap0(x_unit * target_radius)

class FeedbackDataset:
    def __init__(self, feedback_path, node_name_to_idx, bert_model, device):
        self.samples = []
        self.device = device
        
        print(f"📥 Loading feedback from {feedback_path}...")
        with open(feedback_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        print("🧠 Pre-encoding queries with BERT...")
        count = 0
        for item in tqdm(raw_data):
            query_text = item['query']
            target_name = item.get('positive') or item.get('negative')
            label = 1 if 'positive' in item else -1
            
            # Fuzzy matching fix: if full name is missing, attempt suffix matching
            target_idx = -1
            if target_name in node_name_to_idx:
                target_idx = node_name_to_idx[target_name]
            else:
                # Attempt suffix matching (e.g. "Nat.gcd..." -> "gcd...")
                for name, idx in node_name_to_idx.items():
                    if name.endswith(target_name) or target_name in name:
                        target_idx = idx
                        print(f"   🔹 Auto-matched: '{target_name}' -> '{name}'")
                        break
            
            if target_idx == -1:
                print(f"   ⚠️ Warning: '{target_name}' not found in graph. Skipping.")
                continue

            with torch.no_grad():
                query_emb = bert_model.encode(query_text, convert_to_tensor=True, show_progress_bar=False)
                query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=0)

            self.samples.append({
                'query': query_emb.cpu(),
                'target': target_idx,
                'label': label,
                'weight': item.get('weight', 1.0)
            })
            count += 1
        print(f"✅ Loaded {count} samples.")

    def get_batch(self, batch_size=32):
        import random
        random.shuffle(self.samples)
        for i in range(0, len(self.samples), batch_size):
            batch = self.samples[i:i+batch_size]
            yield (
                torch.stack([x['query'] for x in batch]).to(self.device),
                torch.tensor([x['target'] for x in batch], dtype=torch.long).to(self.device),
                torch.tensor([x['label'] for x in batch]).to(self.device),
                torch.tensor([x['weight'] for x in batch]).to(self.device)
            )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.005)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load mapping
    print("📂 Loading node list...")
    # Prioritize loading the authoritative list
    if os.path.exists(NODE_LIST_PATH):
        with gzip.open(NODE_LIST_PATH, "rb") as f:
            node_list = pickle.load(f)
    else:
        with gzip.open(NODE_MAP_PATH, "rb") as f:
            node_list = list(pickle.load(f).keys())
            
    node_name_to_idx = {n: i for i, n in enumerate(node_list)}

    # 2. Load anchors (Graph Embeddings)
    graph_emb = torch.load(NODE_EMB_PATH, map_location=device)
    graph_emb.requires_grad = False

    # 3. Load model
    ckpt = torch.load(MODEL_PATH, map_location=device)
    # Dynamically retrieve dimensions
    h_dim = ckpt.get("hidden_dim", 256) # Defaults to the new version parameters
    o_dim = ckpt.get("out_dim", 64)
    c = ckpt.get("c", 1.0)
    
    print(f"🧠 Loading Model (H={h_dim}, O={o_dim})...")
    full_model = FinalHGCN(384, h_dim, o_dim, c).to(device)
    full_model.load_state_dict(ckpt["model"])
    
    model = AlignmentModel(full_model, c).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 4. Data
    bert = SentenceTransformer(os.path.join(project_root, "models/all-MiniLM-L6-v2"), device=device)
    dataset = FeedbackDataset(os.path.join(DATA_DIR, "system1_feedback.json"), node_name_to_idx, bert, device)

    if not dataset.samples:
        print("❌ No data to train.")
        return

    # 5. Training
    print("\n🚀 Fine-tuning...")
    manifold = PoincareBall(c)

    
    
    for epoch in range(args.epochs):
        total_loss = 0
        for q, t_idx, l, w in dataset.get_batch():
            optimizer.zero_grad()
            
            q_hyp = model(q)
            t_hyp = graph_emb[t_idx]
            
            dist = manifold.dist(q_hyp, t_hyp)
            
            # Loss: Positive -> min dist, Negative -> max dist (margin)
            loss = torch.where(l == 1, dist * w, torch.relu(2.0 - dist) * w).mean()
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"   Ep {epoch+1:03d} | Loss: {total_loss:.4f}")

    # 6. Saving
    # We only updated semantic_proj and scale; we need to write back to the checkpoint
    full_model.layer.semantic_proj = model.proj
    full_model.layer.scale = model.scale
    
    new_ckpt = ckpt.copy()
    new_ckpt["model"] = full_model.state_dict()
    new_ckpt["note"] = "Refined"
    
    torch.save(new_ckpt, SAVE_PATH)
    print(f"✅ Saved refined model to {SAVE_PATH}")

if __name__ == "__main__":
    main()