# ==========================================
# Filename: src/system1/train_final.py
# Version: v10.0 (Scaled for 110k Nodes)
# ==========================================

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
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
HIDDEN_DIM = 128  # [Modified] Increased from 64 to 128 to accommodate larger graphs
OUT_DIM = 32      # [Modified] Increased from 16 to 32 to increase hyperbolic space capacity



class EuclideanGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.att_src = nn.Parameter(torch.Tensor(1, out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(1, out_dim))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, edge_index):
        x_trans = self.linear(x)
        if edge_index.size(1) == 0: # Handle disconnected graph case
            return F.relu(x_trans)
            
        row, col = edge_index
        alpha_src = (x_trans * self.att_src).sum(dim=-1)
        alpha_dst = (x_trans * self.att_dst).sum(dim=-1)
        alpha = F.leaky_relu(alpha_src[row] + alpha_dst[col], 0.2)
        alpha = softmax(alpha, row, num_nodes=x.size(0))
        
        out = torch.zeros_like(x_trans)
        out.index_add_(0, row, alpha.unsqueeze(-1) * x_trans[col])
        return F.relu(out)

class FinalHGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=16, c=1.0):
        super().__init__()
        self.manifold = PoincareBall(c)
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.conv1 = EuclideanGraphConv(hidden_dim, hidden_dim)
        self.conv2 = EuclideanGraphConv(hidden_dim, hidden_dim)
        self.final_proj = nn.Linear(hidden_dim, out_dim)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_index):
        x = self.proj(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.final_proj(x)
        
        x_norm = x.norm(dim=-1, keepdim=True) + 1e-8
        x_unit = x / x_norm 
        target_radius = 0.5 + 0.4 * torch.tanh(self.scale) 
        return self.manifold.expmap0(x_unit * target_radius)



def main():
    DATA_DIR = os.path.join(project_root, "data")
    x = torch.load(os.path.join(DATA_DIR, "node_features_euclidean.pt"), map_location="cpu")
    edge_index = torch.load(os.path.join(DATA_DIR, "edge_index.pt"), map_location="cpu")
    
    # Simple random sampling for Link Prediction training
    # For large-scale graphs, we don't need heavy operations like RandomLinkSplit; simply sample negative examples directly
    
    model = FinalHGCN(x.shape[1], HIDDEN_DIM, OUT_DIM, CURVATURE_C).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"🚀 Training HGCN on {x.shape[0]} nodes...")
    x = x.to(device)
    edge_index = edge_index.to(device)
    
    # Only perform meaningful training when there are enough edges
    has_edges = edge_index.size(1) > 100
    
    for epoch in range(101): # Fast training for 100 epochs
        model.train()
        optimizer.zero_grad()
        z = model(x, edge_index)
        
        loss = torch.tensor(0.0, device=device)
        
        # 1. Entailment Cone Loss - Always available
        # Distribute all node norms between 0.1 and 0.9 to prevent collapse
        norms = z.norm(dim=-1)
        loss_reg = torch.mean((norms - 0.5).pow(2))
        loss += loss_reg
        
        # 2. Link Prediction Loss (Only when edges exist)
        if has_edges:
            # Sample positive edges
            perm = torch.randperm(edge_index.size(1))[:5000]
            u, v = edge_index[:, perm]
            pos_dist = model.manifold.dist(z[u], z[v])
            
            # Sample negative examples
            neg_v = torch.randint(0, z.size(0), (len(u),), device=device)
            neg_dist = model.manifold.dist(z[u], z[neg_v])
            
            # Margin loss: pos should be < neg
            loss_link = F.relu(pos_dist - neg_dist + 0.5).mean()
            loss += loss_link

        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Ep {epoch:03d} | Loss: {loss.item():.4f}")

    # Save
    save_path = os.path.join(DATA_DIR, "hgcn_final.pth")
    torch.save({
        "model": model.state_dict(),
        "c": CURVATURE_C,
        "hidden_dim": HIDDEN_DIM,
        "out_dim": OUT_DIM
    }, save_path)
    print(f"✅ Model saved to {save_path}")

if __name__ == "__main__":
    main()