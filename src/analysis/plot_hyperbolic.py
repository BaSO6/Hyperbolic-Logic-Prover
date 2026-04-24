import os
import sys
import glob
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.cm as cm

# ================= 配置区 =================
TARGET_DIR = "" # 通过命令行传入
# =========================================

def load_data(trace_dir):
    embeddings = []
    metadata = [] 
    
    files = glob.glob(os.path.join(trace_dir, "*.pkl.gz"))
    # 为了绘图清晰，随机采样 200 个失败案例 + 所有成功案例
    success_files = []
    fail_files = []
    
    print("Scanning files...")
    for f in files:
        if "exercise" in f: # 简单过滤
            success_files.append(f) # 先都放进去，后面再筛
            
    sampled_files = success_files[:500] # 限制总量
    
    for fpath in tqdm(sampled_files, desc="Loading"):
        try:
            with gzip.open(fpath, "rb") as f:
                res = pickle.load(f)
            
            trace = res.get('trace', [])
            status = res.get('status', 'Unknown')
            name = res.get('name', 'Unknown')
            
            trace_pts = []
            for step in trace:
                if 'current_coord' in step:
                    vec = np.array(step['current_coord']).flatten()
                    if np.linalg.norm(vec) < 1.05:
                        trace_pts.append(vec)
            
            if trace_pts:
                embeddings.extend(trace_pts)
                for i, pt in enumerate(trace_pts):
                    metadata.append({
                        'status': status,
                        'name': name,
                        'norm': np.linalg.norm(pt),
                        'step': i,
                        'trace_len': len(trace_pts)
                    })
        except: pass
        
    return np.array(embeddings), metadata

def plot_paper_figure(embeddings, metadata, output_dir):
    print("🎨 Generating ICML-Style Figure...")
    
    # PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(embeddings)
    
    # Normalize to Unit Disk
    max_val = np.max(np.linalg.norm(X_2d, axis=1))
    X_2d = X_2d / (max_val * 1.05)
    
    plt.figure(figsize=(10, 10), facecolor='white')
    ax = plt.gca()
    
    # 1. 绘制庞加莱盘背景
    circle = plt.Circle((0, 0), 1.0, color='black', fill=False, linewidth=2)
    ax.add_patch(circle)
    # 辅助网格
    for r in [0.2, 0.5, 0.8]:
        ax.add_patch(plt.Circle((0, 0), r, color='gray', fill=False, linestyle=':', alpha=0.3))
        plt.text(0, -r, f"{r}", color='gray', fontsize=8, ha='center', va='top')

    # 2. 分离数据
    succ_mask = [m['status'] == 'Success' for m in metadata]
    fail_mask = [not x for x in succ_mask]
    
    X_succ = X_2d[succ_mask]
    X_fail = X_2d[fail_mask]
    
    # 3. 绘制失败点 (背景噪音) - 极低透明度
    plt.scatter(X_fail[:, 0], X_fail[:, 1], c='lightgray', s=5, alpha=0.1, label='Failed Explorations', rasterized=True)
    
    # 4. 绘制成功轨迹 (核心证据)
    # 我们不只是画点，而是把同一个 trace 的点连起来
    unique_traces = list(set([m['name'] for i, m in enumerate(metadata) if succ_mask[i]]))
    
    # 挑选几条典型轨迹画线
    colors = cm.plasma(np.linspace(0, 1, len(unique_traces)))
    
    for idx, trace_name in enumerate(unique_traces[:10]): # 只画前10条成功轨迹，避免乱
        trace_indices = [i for i, m in enumerate(metadata) if m['name'] == trace_name]
        trace_indices.sort(key=lambda i: metadata[i]['step'])
        
        pts = X_2d[trace_indices]
        
        # 画线
        plt.plot(pts[:, 0], pts[:, 1], c=colors[idx], linewidth=1.5, alpha=0.8)
        # 画起点 (具体问题)
        plt.scatter(pts[0, 0], pts[0, 1], c='black', marker='x', s=30, zorder=5)
        # 画终点 (证毕)
        plt.scatter(pts[-1, 0], pts[-1, 1], c='gold', marker='*', s=80, edgecolors='black', zorder=5)

    # 5. 装饰
    plt.scatter(0, 0, c='red', marker='+', s=100, label='Axiom Origin', zorder=10)
    plt.title("Hyperbolic Geometry of Reasoning Traces", fontsize=14)
    plt.axis('off')
    plt.legend(loc='upper right')
    
    save_path = os.path.join(output_dir, "icml_hyperbolic_plot.pdf") # PDF for vector graphics
    plt.savefig(save_path, bbox_inches='tight')
    
    save_path_png = os.path.join(output_dir, "icml_hyperbolic_plot.png")
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/analysis/plot_hyperbolic_paper.py <TRACE_DIR>")
        sys.exit(1)
    
    X, M = load_data(sys.argv[1])
    if len(X) > 0:
        plot_paper_figure(X, M, os.path.dirname(sys.argv[1]))