import os
import gzip
import pickle
import sys

# Path adaptation
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
DATA_DIR = os.path.join(project_root, "data")
MAP_PATH = os.path.join(DATA_DIR, "node_text_map.pkl.gz")

if not os.path.exists(MAP_PATH):
    print(f"❌ File not found: {MAP_PATH}")
    sys.exit(1)

print("📂 Loading graph index...")
with gzip.open(MAP_PATH, "rb") as f:
    node_text_map = pickle.load(f)

all_names = list(node_text_map.keys())
print(f"📚 Total of {len(all_names)} theorems in the graph.")

# --- Core Modification: Relaxing search criteria and sorting by length ---

def search_and_print(keywords, label):
    print(f"\n🔍 Searching for theorems containing {keywords} (sorted by length)...")
    matches = []
    for name in all_names:
        # Convert all to lowercase for matching, ignoring case sensitivity
        name_lower = name.lower()
        if all(k.lower() in name_lower for k in keywords):
            matches.append(name)
    
    # Crucial: Sort by name length! Core lemmas usually have the shortest names.
    matches.sort(key=len)
    
    print(f"👉 '{label}' matched {len(matches)} results. Top 20 shortest:")
    for m in matches[:20]:
        print(f"   - {m}")
    
    return matches

# 1. Search for GCD related (searching only for 'gcd' and 'sub', removing 'mul' to account for naming variants)
search_and_print(["gcd", "sub"], "GCD Target")

# 2. Search for Log related (searching only for 'log' and 'mul', removing 'Real' to account for namespace differences)
search_and_print(["log", "mul"], "Log Target")

print("\n✅ Done. Please copy the [shortest and most likely] name from the list above.")