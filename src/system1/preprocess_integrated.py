# ==========================================
# Filename: src/system1/preprocess_integrated.py
# Version: v1.0 (The Unifier)
# Functionality: 
#   1. Read JSON data
#   2. Build Graph (NetworkX)
#   3. [CRITICAL] Lock node order and save ID mapping
#   4. Generate BERT features
#   5. Save all artifacts
# ==========================================

import os
import json
import gzip
import pickle
import torch
import networkx as nx
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
PROOF_DEP_PATH = os.path.join(DATA_DIR, "proof_local_deps.json")
MODEL_DIR = os.path.join(BASE_DIR, "models", "all-MiniLM-L6-v2")

# Output Paths
GRAPH_PATH = os.path.join(DATA_DIR, "mathlib_deep_graph.pkl.gz")
NODE_LIST_PATH = os.path.join(DATA_DIR, "node_list.pkl.gz")       # Definitive order
ID_MAP_PATH = os.path.join(DATA_DIR, "id_to_name.pkl.gz")         # ID -> Name mapping
FEATURE_PATH = os.path.join(DATA_DIR, "node_features_euclidean.pt")
EDGE_PATH = os.path.join(DATA_DIR, "edge_index.pt")
TEXT_MAP_PATH = os.path.join(DATA_DIR, "node_text_map.pkl.gz")

def main():
    print(f"🚀 Starting Integrated Preprocessing...")
    
    # 1. Load raw data
    print("📥 Loading raw json...")
    with open(PROOF_DEP_PATH, "r", encoding="utf-8") as f:
        proof_data = json.load(f)
    print(f"   Declarations: {len(proof_data)}")

    # 2. Build Graph & Text Map
    print("🧠 Building Graph & Text Map...")
    G = nx.DiGraph()
    node_text_map = {}
    
    # Strictly sort to guarantee determinism
    sorted_names = sorted(list(proof_data.keys()))
    
    for name in sorted_names:
        info = proof_data[name]
        G.add_node(name)
        
        # Construct rich text: Name + Type + (Optional Head Symbols)
        type_sig = info.get("type", "")
        heads = info.get("head_symbols", [])
        # [Suggestion 1: Adopted feedback, incorporating Head Symbols]
        text = f"{name} : {type_sig}"
        if heads:
            text += f" | HEADS: {' '.join(heads)}"
        node_text_map[name] = text
        
        # Add edges
        for dep in info.get("used_lemmas", []):
            if dep in proof_data: # Only connect known declarations
                G.add_edge(name, dep)

    print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    # 3. [CRITICAL] Lock node order and indices
    print("🔒 Locking Node Order...")
    # Use the sorted list as the definitive authority list
    node_list = sorted_names 
    node_to_idx = {name: i for i, name in enumerate(node_list)}
    idx_to_name = {i: name for i, name in enumerate(node_list)}

    # Build Edge Index in Tensor format
    
    edge_pairs = []
    for u, v in G.edges():
        if u in node_to_idx and v in node_to_idx:
            u_idx, v_idx = node_to_idx[u], node_to_idx[v]
            edge_pairs.append((u_idx, v_idx))
            edge_pairs.append((v_idx, u_idx)) # Bi-directional

    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()

    # 4. Generate features (BERT)
    print("🤖 Generating Features (BERT)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load with compatibility check
    model_path = MODEL_DIR if os.path.exists(MODEL_DIR) else "sentence-transformers/all-MiniLM-L6-v2"
    embedder = SentenceTransformer(model_path, device=device)
    
    batch_size = 512
    features = []
    
    # Must generate following the exact order of node_list!
    for i in tqdm(range(0, len(node_list), batch_size)):
        batch_names = node_list[i : i+batch_size]
        batch_texts = [node_text_map[n] for n in batch_names]
        
        with torch.no_grad():
            emb = embedder.encode(
                batch_texts, 
                convert_to_tensor=True, 
                show_progress_bar=False,
                normalize_embeddings=True
            )
        features.append(emb.cpu())
    
    x = torch.cat(features, dim=0)

    # 5. Save all Artifacts
    print("💾 Saving Artifacts...")
    
    # Graph
    with gzip.open(GRAPH_PATH, "wb") as f: pickle.dump(G, f)
    # Text Map
    with gzip.open(TEXT_MAP_PATH, "wb") as f: pickle.dump(node_text_map, f)
    # [CORE] Node List (Definitive order)
    with gzip.open(NODE_LIST_PATH, "wb") as f: pickle.dump(node_list, f)
    # [CORE] ID Map (Dedicated for System 2)
    with gzip.open(ID_MAP_PATH, "wb") as f: pickle.dump(idx_to_name, f)
    # Tensors
    torch.save(edge_index, EDGE_PATH)
    torch.save(x, FEATURE_PATH)

    print("✅ Preprocessing Integrated Complete. Consistency Guaranteed.")

if __name__ == "__main__":
    main()