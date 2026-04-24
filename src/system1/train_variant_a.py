# ==========================================
# 文件名: src/system1/train_variant_a_heavy.py
# 功能: 深度欧氏 GCN (Deep Euclidean GCN)
# 目标: 利用 Oversmoothing 效应强行诱发 Semantic Collapse (SCI -> 0.9+)
# ==========================================

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import softmax

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# 复用组件
from src.system1.train_final import EuclideanGraphConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# [关键修改] 
# 1. 维度提升到 256 (匹配论文描述 R^256)
# 2. 层数加深 (利用 Deep GCN 的 Oversmoothing 问题)
HIDDEN_DIM = 256
OUT_DIM = 256 
NUM_LAYERS = 6  # 6层足以让欧氏空间“窒息”

class DeepEuclideanCollapse(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=6):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        
        # 堆叠多层 GCN 导致信息过度混合 (Collapse)
        self.convs = nn.ModuleList([
            EuclideanGraphConv(hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        self.final_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.proj(x))
        
        # 每一层都加剧平滑
        for conv in self.convs:
            x = conv(x, edge_index)
            # 欧氏空间中 Deep GCN 经常不加 Residual 以加速崩塌
            x = F.relu(x) 
            
        return self.final_proj(x)

def main():
    print(f"📉 Training Deep Variant A (Layers={NUM_LAYERS}) to force Maximum Collapse...")
    DATA_DIR = os.path.join(project_root, "data")
    
    # 1. 加载数据
    x = torch.load(os.path.join(DATA_DIR, "node_features_euclidean.pt"), map_location=device)
    edge_index = torch.load(os.path.join(DATA_DIR, "edge_index.pt"), map_location=device)
    
    model = DeepEuclideanCollapse(x.shape[1], HIDDEN_DIM, OUT_DIM, NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005) # 稍微加大 LR 加速收敛
    
    model.train()
    
    # 2. 训练循环 (80 Epochs)
    # 我们希望 Loss 降得很低，意味着所有点都挤在一起了
    for epoch in range(81):
        optimizer.zero_grad()
        z = model(x, edge_index)
        
        # 采样边
        perm = torch.randperm(edge_index.size(1))[:8000]
        u, v = edge_index[:, perm]
        
        # 负采样：随机点
        neg_v = torch.randint(0, z.size(0), (len(u),), device=device)
        
        # 欧氏距离
        pos_dist = (z[u] - z[v]).norm(dim=-1)
        neg_dist = (z[u] - z[neg_v]).norm(dim=-1)
        
        # [关键] 放宽 Margin，允许它们虽然分开一点点但整体很紧凑
        loss = F.relu(pos_dist - neg_dist + 0.5).mean()
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            # 监控 SCI 的代理指标：向量平均模长和方差
            z_norm = z.norm(dim=-1).mean().item()
            print(f"   Ep {epoch:02d} | Loss: {loss.item():.4f} | AvgNorm: {z_norm:.4f}")

    # 3. 保存覆盖旧文件
    save_path = os.path.join(DATA_DIR, "node_embeddings_variant_a.pt")
    torch.save(z.detach().cpu(), save_path)
    print(f"✅ Saved Deep Collapsed Embeddings to {save_path}")

if __name__ == "__main__":
    main()