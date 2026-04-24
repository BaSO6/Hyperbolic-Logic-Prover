# ==========================================
# Filename: src/system1/train_final.py
# Version: v12.1 (Fix Manifold Attribute Exposure)
# ==========================================

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import softmax

# Path adaptation
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.system1.manifold_math import PoincareBall

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURVATURE_C = 1.0 
HIDDEN_DIM = 256  # High Capacity
OUT_DIM = 64      # High Capacity

print(f"🔥 Device: {device} | Scaling Mode: Hidden={HIDDEN_DIM}, Out={OUT_DIM}")

# --- Component Definitions ---

class EuclideanGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        # Simple GCN variant for structure extraction
        
    def forward(self, x, edge_index):
        x_trans = self.linear(x)
        if edge_index.size(1) == 0: return x_trans
        
        row, col = edge_index
        # Mean Aggregation (More stable, prevents numerical explosion on large-scale graphs)
        out = torch.zeros_like(x_trans)
        deg = torch.zeros(x.size(0), 1, device=x.device)
        deg.index_add_(0, row, torch.ones(row.size(0), 1, device=x.device))
        
        out.index_add_(0, row, x_trans[col])
        out = out / (deg + 1e-8)
        return F.relu(out)

class HyperbolicResidualLayer(nn.Module):
    def __init__(self, in_dim, out_dim, c=1.0):
        super().__init__()
        self.manifold = PoincareBall(c)
        
        # Semantic branch
        self.semantic_proj = nn.Linear(in_dim, out_dim)
        
        # Structural branch
        self.structure_proj = nn.Linear(in_dim, out_dim)
        self.graph_conv = EuclideanGraphConv(out_dim, out_dim)
        
        # Gating
        self.gate = nn.Linear(out_dim * 2, 1) 
        
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_index):
        # 1. Semantic flow
        z_sem = self.semantic_proj(x)
        
        # 2. Structural flow
        z_struct = F.relu(self.structure_proj(x))
        z_struct = self.graph_conv(z_struct, edge_index)
        
        # 3. Dynamic fusion
        combined = torch.cat([z_sem, z_struct], dim=-1)
        alpha = torch.sigmoid(self.gate(combined))
        
        z_tan = alpha * z_sem + (1 - alpha) * z_struct
        
        # 4. Hyperbolic mapping
        x_norm = z_tan.norm(dim=-1, keepdim=True) + 1e-8
        x_unit = z_tan / x_norm 
        target_radius = 0.9 * torch.tanh(self.scale)
        
        return self.manifold.expmap0(x_unit * target_radius)

class FinalHGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, c=1.0):
        super().__init__()
        self.layer = HyperbolicResidualLayer(in_dim, out_dim, c)
        
        # [CRITICAL FIX] Expose the internal manifold of the layer to the exterior for use in Loss calculation
        self.manifold = self.layer.manifold 
        
        # Save parameters
        self.hidden_dim = hidden_dim 
        self.out_dim = out_dim
        self.c = c

    def forward(self, x, edge_index):
        return self.layer(x, edge_index)

    # Compatibility interface (for use with export_embeddings and debug)
    @property
    def proj(self): return self.layer.semantic_proj
    @property
    def final_proj(self): return self.layer.structure_proj # Placeholder only; actual logic is inside the layer
    @property
    def scale(self): return self.layer.scale

# ==========================================
# Main Training Process
# ==========================================
def main():
    DATA_DIR = os.path.join(project_root, "data")
    
    print("📥 Loading features & graph...")
    x = torch.load(os.path.join(DATA_DIR, "node_features_euclidean.pt"), map_location="cpu")
    edge_index = torch.load(os.path.join(DATA_DIR, "edge_index.pt"), map_location="cpu")
    
    # Initialize model
    model = FinalHGCN(x.shape[1], HIDDEN_DIM, OUT_DIM, CURVATURE_C).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    print(f"🚀 Training Start (Nodes={x.shape[0]})...")
    x = x.to(device)
    edge_index = edge_index.to(device)
    
    has_edges = edge_index.size(1) > 100
    
    for epoch in range(201):
        model.train()
        optimizer.zero_grad()
        z = model(x, edge_index)
        
        # Loss 1: Distribution Regularization
        norms = z.norm(dim=-1)
        loss_reg = torch.mean((norms - 0.8).pow(2))
        
        loss_link = torch.tensor(0.0, device=device)
        
        # Loss 2: Link Prediction
        if has_edges:
            perm = torch.randperm(edge_index.size(1))[:10000]
            u, v = edge_index[:, perm]
            
            # Calling model.manifold.dist here will no longer result in an error
            pos_dist = model.manifold.dist(z[u], z[v])
            
            neg_v = torch.randint(0, z.size(0), (len(u),), device=device)
            neg_dist = model.manifold.dist(z[u], z[neg_v])
            
            loss_link = F.relu(pos_dist - neg_dist + 1.0).mean()

        total_loss = loss_link + 0.1 * loss_reg
        total_loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Ep {epoch:03d} | Loss: {total_loss.item():.4f} (Link={loss_link:.4f}, Reg={loss_reg:.4f})")

    save_path = os.path.join(DATA_DIR, "hgcn_final.pth")
    torch.save({
        "model": model.state_dict(),
        "c": CURVATURE_C,
        "hidden_dim": HIDDEN_DIM,
        "out_dim": OUT_DIM,
        "note": "ICML_Oral_Version"
    }, save_path)
    print(f"✅ ICML Model saved to {save_path}")

if __name__ == "__main__":
    main()