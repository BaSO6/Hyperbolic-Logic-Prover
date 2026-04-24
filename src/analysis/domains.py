import matplotlib
matplotlib.use('Agg') # 强制后台，防卡死

import os
import sys
import torch
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.patches as mpatches
import re

# ================= 配置区 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EMBEDDING_PATH = os.path.join(BASE_DIR, "data", "node_embeddings.pt")
NAME_MAP_PATH = os.path.join(BASE_DIR, "data", "id_to_name.pkl.gz")
OUTPUT_DIR = os.path.join(BASE_DIR, "benchmark_reports", "visual_analysis")
# =========================================

def get_detailed_domain(name):
    """
    深度语义推断：基于 100+ 个数学关键词进行领域分类
    """
    n = name.lower()
    
    # 1. 优先匹配顶级前缀 (如果存在)
    if "algebra" in n: return "Algebra"
    if "topology" in n: return "Topology"
    if "analysis" in n: return "Analysis"
    if "geometry" in n: return "Geometry"
    if "numbertheory" in n: return "Number Theory"
    if "categorytheory" in n: return "Category Theory"
    if "settheory" in n or "logic" in n: return "Logic & Set"
    if "measuretheory" in n: return "Measure Theory"
    
    # 2. 关键词反向推导 (Deep Inference)
    keywords = {
        "Algebra": ["group", "ring", "monoid", "module", "field", "linear", "matrix", "polynomial", "subgroup", "ideal", "hom", "iso", "tensor", "vector"],
        "Topology": ["continuous", "homeomorph", "metric", "compact", "connected", "open", "closed", "neighborhood", "filter", "limit", "cauchy"],
        "Analysis": ["deriv", "integr", "differentiab", "series", "sequence", "complex", "real", "convex", "normed", "banach", "hilbert", "fourier"],
        "Geometry": ["manifold", "bundle", "smooth", "affine", "euclidean", "angle", "triangle", "sphere", "curve"],
        "Number Theory": ["prime", "gcd", "lcm", "modular", "divisib", "fibonacci", "factorial", "bernoulli"],
        "Category Theory": ["functor", "category", "monad", "adjoint", "yoneda", "limit", "colimit"],
        "Logic & Set": ["decidable", "classical", "axiom", "cardinal", "ordinal", "relation", "equiv", "subset", "union", "inter"],
        "Combinatorics": ["perm", "combination", "graph", "tree", "finset", "fintype"]
    }
    
    for domain, keys in keywords.items():
        for k in keys:
            if k in n: return domain
            
    # 3. 兜底
    if "data" in n or "control" in n: return "CS / Data"
    return "Other"

def load_knowledge_base():
    print(f"📂 Loading System 1 Memory...")
    if not os.path.exists(EMBEDDING_PATH) or not os.path.exists(NAME_MAP_PATH):
        print(f"❌ Data files not found.\nExpected: {EMBEDDING_PATH}\nExpected: {NAME_MAP_PATH}")
        sys.exit(1)

    # 加载数据
    embeddings = torch.load(EMBEDDING_PATH, map_location='cpu').numpy()
    with gzip.open(NAME_MAP_PATH, "rb") as f:
        id_to_name = pickle.load(f)
    
    # [关键步骤] 打印前5个名字，人工确认格式！
    print(f"🔍 [DEBUG] First 5 theorem names in database:")
    sample_ids = list(id_to_name.keys())[:5]
    for i in sample_ids:
        print(f"   - ID {i}: {id_to_name[i]}")
        
    return embeddings, id_to_name

def plot_atlas(embeddings, id_to_name, output_dir):
    # 采样
    total_points = len(embeddings)
    MAX_POINTS = 60000 
    indices = np.arange(total_points)
    
    if total_points > MAX_POINTS:
        print(f"✂️ Sampling {MAX_POINTS} points from {total_points}...")
        np.random.shuffle(indices)
        indices = indices[:MAX_POINTS]

    # 提取数据
    selected_embs = embeddings[indices]
    selected_names = [str(id_to_name.get(i, "Unknown")) for i in indices]
    
    # 提取领域
    domains = [get_detailed_domain(n) for n in selected_names]
    norms = np.linalg.norm(selected_embs, axis=1)
    
    # 打印统计
    from collections import Counter
    counts = Counter(domains)
    print("\n📊 Knowledge Distribution (After Deep Matching):")
    for d, c in counts.most_common():
        print(f"   - {d}: {c} ({c/len(domains)*100:.1f}%)")

    print("\n🧠 Running PCA...")
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(selected_embs)
    
    # 归一化
    max_r = np.max(np.linalg.norm(X_2d, axis=1))
    if max_r > 0: X_2d = X_2d / (max_r * 1.05)

    print("🎨 Painting the Atlas...")
    plt.figure(figsize=(15, 15), facecolor='white', dpi=300)
    ax = plt.gca()

    # 1. 背景
    ax.add_patch(plt.Circle((0, 0), 1.0, color='#F9F9F9', fill=True))
    ax.add_patch(plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=2))
    
    # 2. 层级圆环 (核心特征)
    # r < 0.4: 基础层 (Foundations)
    # 0.4 < r < 0.7: 理论层 (Theories)
    # r > 0.7: 应用层 (Applications)
    for r, label in zip([0.4, 0.7, 0.95], ["Foundations", "Theories", "Specifics"]):
        ax.add_patch(plt.Circle((0, 0), r, color='gray', fill=False, linestyle='--', alpha=0.3))
        plt.text(0, -r+0.02, label, color='#666666', fontsize=12, ha='center', fontweight='bold', alpha=0.7)

    # 3. 高级配色 (Qualitative Set)
    color_map = {
        "Algebra": "#4E79A7",       # Blue
        "Analysis": "#59A14F",      # Green
        "Topology": "#EDC948",      # Yellow/Gold
        "Geometry": "#F28E2B",      # Orange
        "Number Theory": "#E15759", # Red
        "Logic & Set": "#76B7B2",   # Teal
        "Category Theory": "#B07AA1", # Purple
        "CS / Data": "#9C755F",     # Brown
        "Other": "#BAB0AC"          # Grey
    }

    # 4. 向量化绘图 (按领域分组绘制，极大加速)
    from collections import defaultdict
    groups = defaultdict(list)
    for i, d in enumerate(domains):
        groups[d].append(i)
        
    # 先画 Other
    if "Other" in groups:
        idx = groups["Other"]
        plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c=color_map["Other"], s=5, alpha=0.1, edgecolors='none', rasterized=True)
        
    # 再画主要领域
    for d, idxs in groups.items():
        if d == "Other": continue
        pts = X_2d[idxs]
        ns = norms[idxs]
        
        # 视觉大小：核心公理(Norm小)大，边缘定理(Norm大)小
        sizes = 40 * (1.15 - ns)
        sizes = np.clip(sizes, 5, 60)
        
        plt.scatter(pts[:, 0], pts[:, 1], c=color_map.get(d, "gray"), s=sizes, alpha=0.6, edgecolors='none', rasterized=True)

    # 5. 中心公理
    plt.scatter(0, 0, c='black', marker='+', s=400, linewidth=3, zorder=20)
    plt.text(0, 0.05, "Mathlib Root", ha='center', fontsize=14, fontweight='bold')

    # 6. 图例
    handles = [mpatches.Patch(color=c, label=f"{d} ({counts[d]})") for d, c in color_map.items() if d in counts and d != "Other"]
    plt.legend(handles=handles, loc='upper right', title="Mathematical Fields", fontsize=10, frameon=True)

    plt.axis('off')
    plt.title("Semantic Topology of Mathlib4\n(Hyperbolic Embedding via HGCN)", fontsize=22, pad=30, fontname='serif')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_DIR, "mathlib_atlas_v3.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Atlas saved to: {save_path}")

if __name__ == "__main__":
    embs, names = load_knowledge_base()
    plot_atlas(embs, names, OUTPUT_DIR)