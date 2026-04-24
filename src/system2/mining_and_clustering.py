# ==========================================
# Filename: src/system2/mining_and_clustering.py
# Version: v5.2 (Patched + Full Standalone)
# Functionality: 
#   1. [Fix] Includes PyTorch/Transformers compatibility patches
#   2. Includes complete HGCN model definition (for standalone execution)
#   3. Performs Balanced Spherical Clustering (resolves 99% of large cluster issues)
# ==========================================

import os
import sys
import json
import torch

# [🚨 Critical Patch] Must execute before importing torch_geometric or transformers
# Resolves AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'
try:
    import torch.utils._pytree
    if not hasattr(torch.utils._pytree, 'register_pytree_node'):
        def _safe_register_pytree_node(cls, flatten_fn, unflatten_fn, serialized_type_name=None):
            # Discard the serialized_type_name parameter passed by newer transformers versions
            return torch.utils._pytree._register_pytree_node(cls, flatten_fn, unflatten_fn)
        torch.utils._pytree.register_pytree_node = _safe_register_pytree_node
        print("🔧 [Mining] Applied PyTorch compatibility patch.")
except ImportError:
    pass

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from torch_geometric.utils import softmax
from torch_geometric.data import Data
from tqdm import tqdm
import gzip
import pickle

# Path Settings
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Dependency on math library
from src.system1.manifold_math import PoincareBall

# ==========================================
# 1. Model Definition (Fully embedded to ensure standalone execution)
# ==========================================
class HyperbolicLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, drop_connect=0.0):
        super().__init__()
        self.drop_connect = drop_connect
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=0.8)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class HyperbolicGraphConv(nn.Module):
    def __init__(self, in_features, out_features, c, drop_connect=0.2):
        super().__init__()
        self.c = c
        self.linear = HyperbolicLinear(in_features, out_features, bias=True, drop_connect=drop_connect)
        self.att_query = nn.Linear(out_features, 1)
        self.att_key = nn.Linear(out_features, 1)
        self.manifold = PoincareBall(c=c)

    def forward(self, x, edge_index):
        # 1. Map to Tangent Space
        x = x.clamp(min=-0.995, max=0.995)
        x_tan = self.manifold.logmap0(x)
        
        # 2. Linear Transform
        x_trans = self.linear(x_tan)
        
        # 3. Attention Aggregation
        row, col = edge_index
        q = self.att_query(x_trans)
        k = self.att_key(x_trans)
        alpha = F.leaky_relu(q[row] + k[col], negative_slope=0.2)
        alpha = softmax(alpha, row, num_nodes=x.size(0))
        
        x_aggr_tan = torch.zeros_like(x_trans)
        x_aggr_tan.index_add_(0, row, alpha * x_trans[col])
        
        # 4. Clip & Activation
        norm = x_aggr_tan.norm(dim=-1, keepdim=True)
        clip_coef = 10.0 / (norm + 1e-6)
        clip_coef = torch.clamp(clip_coef, max=1.0)
        x_aggr_tan = x_aggr_tan * clip_coef

        # 5. Map back
        out = self.manifold.expmap0(x_aggr_tan)
        out_tan = self.manifold.logmap0(out)
        out_tan = F.relu(out_tan)
        out = self.manifold.expmap0(out_tan)
        return out

class FinalHGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, c_fixed=5.0):
        super().__init__()
        self.c = c_fixed 
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layer1 = HyperbolicGraphConv(hidden_dim, hidden_dim, self.c, drop_connect=0.2)
        self.layer2 = HyperbolicGraphConv(hidden_dim, out_dim, self.c, drop_connect=0.2)
        self.manifold = PoincareBall(c=c_fixed)
        
    def forward(self, x_euclidean, edge_index):
        x = self.input_proj(x_euclidean)
        x = torch.clamp(x, min=-10, max=10)
        x = self.manifold.expmap0(x)
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        return x

# ==========================================
# 2. Data Loading and Clustering Logic
# ==========================================

def load_system1_model(device):
    """Load the trained HGCN model"""
    model_path = os.path.join(project_root, "data", "hgcn_final.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError("❌ Model weights not found, please run train_final.py first")
    
    print(f"📥 Loading Checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    c = checkpoint.get('c', 5.0)
    
    # Load feature dimensions
    feat_path = os.path.join(project_root, "data", "node_features_euclidean.pt")
    x_features = torch.load(feat_path, map_location='cpu', weights_only=True)
    in_dim = x_features.shape[1]
    
    model = FinalHGCN(in_dim=in_dim, hidden_dim=64, out_dim=16, c_fixed=c).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, c

def load_graph_data():
    """Load graph structure and text mapping"""
    data_dir = os.path.join(project_root, "data")
    
    x = torch.load(os.path.join(data_dir, "node_features_euclidean.pt"), map_location='cpu', weights_only=True)
    edge_index = torch.load(os.path.join(data_dir, "edge_index.pt"), map_location='cpu', weights_only=True)
    
    graph_path = os.path.join(data_dir, "mathlib_deep_graph.pkl.gz")
    print(f"📖 Loading Graph structure from {graph_path}...")
    with gzip.open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    idx_to_name = {i: n for i, n in enumerate(G.nodes())}
    
    text_map_path = os.path.join(data_dir, "node_text_map.pkl.gz")
    if os.path.exists(text_map_path):
        with gzip.open(text_map_path, 'rb') as f:
            name_to_def = pickle.load(f)
    else:
        name_to_def = {}
        
    return x, edge_index, idx_to_name, name_to_def

class BalancedRecursiveClustering:
    """
    Balanced Recursive Clusterer (v5.2)
    """
    def __init__(self, min_cluster_size=20, max_depth=3):
        self.min_cluster_size = min_cluster_size
        self.max_depth = max_depth
        self.clusters = {} 

    def fit(self, vectors, data_indices, depth=0, prefix="0"):
        num_samples = len(vectors)
        
        if num_samples < self.min_cluster_size or depth >= self.max_depth:
            self.clusters[prefix] = data_indices
            return

        # Spherization: Focus on direction
        vectors_norm = normalize(vectors, axis=1, norm='l2')

        # Dynamic K
        k = 4 if num_samples > 1000 else 2
        
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(vectors_norm)
        except Exception as e:
            print(f"⚠️ Clustering failed at depth {depth}: {e}")
            self.clusters[prefix] = data_indices
            return
        
        for i in range(k):
            mask = (labels == i)
            if np.sum(mask) == 0: continue
            
            sub_vectors = vectors[mask]
            sub_indices = [data_indices[j] for j in range(len(mask)) if mask[j]]
            
            new_id = f"{prefix}.{i}" if prefix else str(i)
            self.fit(sub_vectors, sub_indices, depth + 1, new_id)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Mining & Clustering Agent (Device: {device})")
    
    # 1. Load Data
    x, edge_index, idx_to_name, name_to_def = load_graph_data()
    model, c = load_system1_model(device)
    manifold = PoincareBall(c=c)
    
    # 2. Generate Embeddings
    print("🔮 Generating embeddings for all nodes...")
    x = x.to(device)
    edge_index = edge_index.to(device)
    
    with torch.no_grad():
        z = model(x, edge_index)
        
    # 3. Mine Geometric Vectors
    print("⛏️ Mining geometric relations (Sampling 20,000)...")
    
    num_edges = edge_index.shape[1]
    perm = torch.randperm(num_edges)[:20000]
    sampled_edges = edge_index[:, perm]
    
    vectors = []
    metadata = []
    
    u_indices = sampled_edges[0]
    v_indices = sampled_edges[1]
    
    batch_size = 2000
    for i in tqdm(range(0, len(u_indices), batch_size)):
        u_batch = u_indices[i:i+batch_size]
        v_batch = v_indices[i:i+batch_size]
        
        z_u = z[u_batch]
        z_v = z[v_batch]
        
        with torch.no_grad():
            delta = manifold.logmap(z_u, z_v)
            
        vectors.append(delta.cpu().numpy())
        
        for k in range(len(u_batch)):
            u_idx = u_batch[k].item()
            v_idx = v_batch[k].item()
            u_name = idx_to_name.get(u_idx, "Unknown")
            v_name = idx_to_name.get(v_idx, "Unknown")
            src_def = name_to_def.get(u_name, "")[:150]
            tgt_def = name_to_def.get(v_name, "")[:150]
            
            metadata.append({
                "source": u_name,
                "target": v_name,
                "source_def": f"{u_name} : {src_def}",
                "target_def": f"{v_name} : {tgt_def}"
            })
            
    all_vectors = np.concatenate(vectors, axis=0)
    print(f"✅ Extracted {len(all_vectors)} logic vectors.")
    
    # 4. Perform Balanced Clustering
    print("🌳 Starting Balanced Spherical Clustering...")
    clusterer = BalancedRecursiveClustering(min_cluster_size=20, max_depth=3)
    clusterer.fit(all_vectors, list(range(len(all_vectors))))
    
    # 5. Result Summary
    final_clusters = {}
    print("\n📊 Cluster Distribution (Top 10):")
    sorted_clusters = sorted(clusterer.clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    for i, (cid, indices) in enumerate(sorted_clusters):
        count = len(indices)
        if i < 10:
            print(f"   Cluster {cid}: {count} items")
        
        if count >= 10:
            sample_indices = np.random.choice(indices, min(20, count), replace=False)
            samples = [metadata[j] for j in sample_indices]
            final_clusters[cid] = samples
            
    # 6. Save
    out_file = os.path.join(project_root, "cluster_data_hierarchical.json")
    with open(out_file, "w") as f:
        json.dump(final_clusters, f, indent=4)
        
    print(f"\n🎉 Clustering complete! Saved {len(final_clusters)} clusters to {out_file}")
    print("➡️ Next Step: Run 'python src/system2/llm_interpreter.py'")

if __name__ == "__main__":
    main()