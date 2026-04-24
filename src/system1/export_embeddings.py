# ==========================================
# Filename: src/system1/export_embeddings.py
# Version: v3.0 (Auto-Dimension Detect)
# ==========================================
import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.system1.train_final import FinalHGCN

def main():
    DATA_DIR = os.path.join(project_root, "data")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("📥 Loading features...")
    x = torch.load(os.path.join(DATA_DIR, "node_features_euclidean.pt"), map_location=device)
    edge_index = torch.load(os.path.join(DATA_DIR, "edge_index.pt"), map_location=device)
    
    print("🧠 Loading model checkpoint...")
    ckpt_path = os.path.join(DATA_DIR, "hgcn_final.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Automatically detect dimension configurations from Checkpoint
    # Fallback to default values (64, 16) if the old model does not have stored dimensions
    h_dim = ckpt.get("hidden_dim", 64)
    o_dim = ckpt.get("out_dim", 16)
    c = ckpt.get("c", 1.0)
    
    print(f"   Model Config: Hidden={h_dim}, Out={o_dim}, c={c}")
    
    
    model = FinalHGCN(x.shape[1], h_dim, o_dim, c).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    print("🚀 Computing embeddings for 110k nodes...")
    with torch.no_grad():
        # The forward pass maps Euclidean features into the hyperbolic space
        z = model(x, edge_index)
        
    
    save_path = os.path.join(DATA_DIR, "node_embeddings.pt")
    torch.save(z.cpu(), save_path)
    print(f"✅ Saved embeddings {z.shape} to {save_path}")

if __name__ == "__main__":
    main()