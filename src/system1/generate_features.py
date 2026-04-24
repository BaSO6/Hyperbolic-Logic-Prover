import os
import torch
import pickle
import gzip
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
DATA_DIR = os.path.join(project_root, "data")
MODEL_DIR = os.path.join(project_root, "models", "all-MiniLM-L6-v2")

# 1. Load the 110k-node graph
print("📥 Loading 110k Graph...")
with gzip.open(os.path.join(DATA_DIR, "mathlib_deep_graph.pkl.gz"), "rb") as f:
    G = pickle.load(f)
    
node_list = list(G.nodes())
print(f"✅ Nodes to process: {len(node_list)}") # Must see 110314

# 2. Load text mapping
with gzip.open(os.path.join(DATA_DIR, "node_text_map.pkl.gz"), "rb") as f:
    node_text_map = pickle.load(f)

# 3. Rebuild Edge Index

node_to_idx = {n: i for i, n in enumerate(node_list)}
edge_pairs = []
for u, v in G.edges():
    if u in node_to_idx and v in node_to_idx:
        edge_pairs.append((node_to_idx[u], node_to_idx[v]))
        edge_pairs.append((node_to_idx[v], node_to_idx[u]))
edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
torch.save(edge_index, os.path.join(DATA_DIR, "edge_index.pt"))
print("✅ Edge Index Rebuilt.")

# 4. Generate features
print("🧠 Generating BERT features (this takes time)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_DIR, device=device)

batch_size = 512
features = []
for i in tqdm(range(0, len(node_list), batch_size)):
    batch = node_list[i:i+batch_size]
    texts = [node_text_map.get(n, n) for n in batch]
    with torch.no_grad():
        emb = model.encode(texts, convert_to_tensor=True, show_progress_bar=False, normalize_embeddings=True)
    features.append(emb.cpu())

x = torch.cat(features, dim=0)
torch.save(x, os.path.join(DATA_DIR, "node_features_euclidean.pt"))
print(f"✅ Features Saved: {x.shape}") # Must be [110314, 384]