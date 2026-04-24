# src/system1/models_euclidean.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class EuclideanGCN(nn.Module):
    """
    Baseline: Standard Euclidean GCN (No Hyperbolic Geometry)
    """
    def __init__(self, num_features, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(EuclideanGCN, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        
        # Layer 1
        self.layers.append(GCNConv(num_features, hidden_dim))
        
        # Hidden Layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            
        # Output Layer (Standard Linear Projection)
        self.layers.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        # Layer 1
        x = self.layers[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Hidden Layers
        for i in range(1, len(self.layers) - 1):
            x = self.layers[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
        # Output Layer
        x = self.layers[-1](x, edge_index)
        
        # [关键区别] 这里不使用 expmap，也不做双曲归一化
        # 只做标准的 L2 归一化以便计算余弦相似度，或者直接输出
        return F.normalize(x, p=2, dim=-1)

class EuclideanProjector(nn.Module):
    """
    Baseline Projector: Simple MLP in Euclidean Space
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        
    def forward(self, x):
        # 没有任何双曲映射
        return F.normalize(self.net(x), p=2, dim=-1)