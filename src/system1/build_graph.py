import os
import json
import networkx as nx
import pickle
import gzip

# ============================================================
# 1. Path Configuration (Retaining your smart fix logic)
# ============================================================

POSSIBLE_DATA_PATHS = [
    os.path.join(project_root, "data"),
    "data",
    "../../data"
]

DATA_DIR = None
for p in POSSIBLE_DATA_PATHS:
    if os.path.exists(p):
        DATA_DIR = p
        break

if DATA_DIR is None:
    raise RuntimeError("❌ data directory not found, please confirm path.")

print(f"📂 Using data directory: {os.path.abspath(DATA_DIR)}")

PROOF_DEP_PATH = os.path.join(DATA_DIR, "proof_local_deps.json")

if not os.path.exists(PROOF_DEP_PATH):
    raise RuntimeError(
        "❌ proof_local_deps.json not found, please run parse_mathlib_deep.py first"
    )

# ============================================================
# 2. Load proof-local dependency data
# ============================================================

print("📥 Loading proof-local dependency data...")
with open(PROOF_DEP_PATH, "r", encoding="utf-8") as f:
    proof_data = json.load(f)

print(f"✅ Loaded {len(proof_data)} declarations")

# ============================================================
# 3. Build Directed Graph (theorem -> used lemma)
# ============================================================

G = nx.DiGraph()
node_text_map = {}
node_head_map = {}  # Optional: for subsequent embedding use

print("🧠 Building proof-local logic graph...")



for name, info in proof_data.items():
    G.add_node(name)

    # node text = name + type (for SentenceTransformer)
    type_sig = info.get("type", "")
    node_text_map[name] = f"{name} : {type_sig}" if type_sig else name

    # Save head symbols (without breaking original flow, optional)
    node_head_map[name] = info.get("head_symbols", [])

# Construct edges
edge_count = 0
for name, info in proof_data.items():
    used = info.get("used_lemmas", [])
    for dep in used:
        if dep in G:  # Only connect known declarations
            G.add_edge(name, dep)
            edge_count += 1

print(f"🔗 Construction complete: {G.number_of_nodes()} nodes, {edge_count} edges")

# ============================================================
# 4. Save results (fully compatible with your original training flow)
# ============================================================

SAVE_DIR = DATA_DIR
os.makedirs(SAVE_DIR, exist_ok=True)

GRAPH_PATH = os.path.join(SAVE_DIR, "mathlib_deep_graph.pkl.gz")
TEXT_PATH = os.path.join(SAVE_DIR, "node_text_map.pkl.gz")
HEAD_PATH = os.path.join(SAVE_DIR, "node_head_symbols.pkl.gz")

print(f"💾 Saving graph structure -> {GRAPH_PATH}")
with gzip.open(GRAPH_PATH, "wb") as f:
    pickle.dump(G, f)

print(f"💾 Saving node text -> {TEXT_PATH}")
with gzip.open(TEXT_PATH, "wb") as f:
    pickle.dump(node_text_map, f)

print(f"💾 Saving head symbols -> {HEAD_PATH}")
with gzip.open(HEAD_PATH, "wb") as f:
    pickle.dump(node_head_map, f)

print("✅ build_graph.py execution finished")