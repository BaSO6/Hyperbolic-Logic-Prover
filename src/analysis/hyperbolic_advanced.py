import matplotlib
matplotlib.use('Agg') # 强制后台

import os
import sys
import glob
import gzip
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

# ================= 配置区 =================
TRACE_DIR = "PROJECT_ROOT_PLACEHOLDER/benchmark_reports/trace"
MAX_FAIL_TRACES = 300 
# =========================================

# ================= ICML 绘图风格设置 (保持不变) =================
plt.rcParams.update({
    'font.size': 22,               
    'font.family': 'serif',        
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'], 
    'axes.labelsize': 24,          
    'axes.titlesize': 28,          
    'legend.fontsize': 20,         
    'xtick.labelsize': 20,         
    'ytick.labelsize': 20,         
    'lines.linewidth': 2.5,        
    'lines.markersize': 10,        
    'figure.dpi': 300,             
    'savefig.bbox': 'tight',       
})
# ===================================================

def load_trajectory_data(trace_dir):
    # (此处代码与之前相同，省略以节省篇幅，请确保运行完整代码)
    all_points = []
    traces = []     
    files = glob.glob(os.path.join(trace_dir, "**/*.pkl.gz"), recursive=True)
    print(f"📂 Scanning {len(files)} files...")
    for fpath in tqdm(files, desc="Loading Traces"):
        try:
            with gzip.open(fpath, "rb") as f:
                res = pickle.load(f)
            trace_steps = res.get('trace', [])
            status = res.get('status', 'Unknown')
            name = res.get('name', 'Unknown')
            coords = []
            for step in trace_steps:
                if 'current_coord' in step:
                    vec = np.array(step['current_coord'])
                    vec = np.squeeze(vec).flatten()
                    if np.linalg.norm(vec) < 1.1: 
                        coords.append(vec)
                        all_points.append(vec)
            if len(coords) > 1: 
                traces.append({'coords': np.array(coords), 'status': status, 'name': name})
        except: pass
    return np.array(all_points), traces

def plot_dynamic_map(all_points, traces, output_dir):
    print("🧠 Computing PCA Projection...")
    if len(all_points) < 5:
        print("❌ Not enough points.")
        return

    pca = PCA(n_components=2)
    pca.fit(all_points)
    all_2d = pca.transform(all_points)
    max_norm = np.max(np.linalg.norm(all_2d, axis=1))
    scale = 1.0 / (max_norm * 1.05)

    print("🎨 Rendering Dynamic Trajectories...")
    
    # ================= 修改点在此处 =================
    # 将 facecolor 从 '#FAFAFA' 改为 'white' (纯白)
    plt.figure(figsize=(12, 12), facecolor='white') 
    ax = plt.gca()
    # 确保坐标轴背景也是纯白 (虽然会被圆盘覆盖，但这是好习惯)
    ax.set_facecolor('white') 
    # ============================================
    
    # 1. 绘制庞加莱盘背景
    # 【保留圆盘的浅灰色，与纯白背景形成对比】
    ax.add_patch(plt.Circle((0, 0), 1.0, color='#F0F0F0', fill=True, zorder=0))
    # 边框
    ax.add_patch(plt.Circle((0, 0), 1.0, color='#546E7A', fill=False, linewidth=3, zorder=10))
    
    # 层级网格
    for r in [0.3, 0.7]:
        ax.add_patch(plt.Circle((0, 0), r, color='white', fill=False, linestyle='--', linewidth=2, alpha=0.6))
        plt.text(0, -r+0.03, f"Level {r}", color='#90A4AE', fontsize=18, ha='center', fontweight='bold')

    # 2. 分离轨迹
    success_traces = [t for t in traces if t['status'] == 'Success']
    fail_traces = [t for t in traces if t['status'] != 'Success']
    
    import random
    if len(fail_traces) > MAX_FAIL_TRACES:
        fail_traces = random.sample(fail_traces, MAX_FAIL_TRACES)

    # 3. 绘制失败轨迹
    fail_segments = []
    for t in fail_traces:
        pts = pca.transform(t['coords']) * scale
        segs = np.concatenate([pts[:-1, None, :], pts[1:, None, :]], axis=1)
        fail_segments.extend(segs)
    
    lc_fail = LineCollection(fail_segments, colors='#B0BEC5', alpha=0.15, linewidths=1.2, zorder=1)
    ax.add_collection(lc_fail)

    # 4. 绘制成功轨迹
    for i, t in enumerate(success_traces):
        pts = pca.transform(t['coords']) * scale
        # A. 画线
        plt.plot(pts[:,0], pts[:,1], c='#D87C7C', linewidth=3.5, alpha=0.8, zorder=5)
        # B. 画箭头
        mid_idx = len(pts) // 2
        if len(pts) > 2:
            plt.arrow(pts[mid_idx-1,0], pts[mid_idx-1,1], 
                      pts[mid_idx,0]-pts[mid_idx-1,0], pts[mid_idx,1]-pts[mid_idx-1,1],
                      shape='full', lw=0, length_includes_head=True, head_width=0.06, color='#C0392B', zorder=6)
        # C. 标记点
        norms = np.linalg.norm(pts, axis=1)
        min_idx = np.argmin(norms)
        # 起点
        plt.scatter(pts[0,0], pts[0,1], c='#455A64', s=150, marker='o', zorder=7)
        # 终点
        plt.scatter(pts[-1,0], pts[-1,1], c='#F1C40F', s=350, marker='*', edgecolors='#B7950B', linewidth=1.5, zorder=8)
        # 抽象转折点
        if min_idx != 0 and min_idx != len(pts)-1 and norms[min_idx] < norms[0] * 0.9:
            plt.scatter(pts[min_idx,0], pts[min_idx,1], c='#00ACC1', s=250, marker='D', edgecolors='white', linewidth=1.5, zorder=9)
            plt.plot([0, pts[min_idx,0]], [0, pts[min_idx,1]], c='#00ACC1', linestyle=':', linewidth=2, alpha=0.5)

    # 5. 绘制公理原点
    plt.scatter(0, 0, c='#263238', marker='+', s=500, linewidth=4, zorder=20)

    # 6. 图例与装饰
    legend_elements = [
        mpatches.Patch(color='#F0F0F0', label='Math Space (Poincaré)'),
        plt.Line2D([0], [0], color='#D87C7C', lw=3.5, label='Successful Proof'),
        plt.Line2D([0], [0], color='#B0BEC5', lw=1.5, alpha=0.5, label='Failed Search'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#455A64', markersize=12, label='Problem Start'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#00ACC1', markersize=12, label='Abstraction Turn'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='#F1C40F', markersize=18, label='Q.E.D. (Solved)')
    ]
    # 图例背景默认是白色的，在纯白背景下很协调
    plt.legend(handles=legend_elements, loc='upper right', frameon=True, framealpha=0.95)
    
    plt.title("Dynamics of Logical Inference: The 'Centripetal' Hypothesis", pad=25, fontweight='bold', color='#37474F')
    plt.axis('off')
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "hyperbolic_trajectory_dynamics_white.png")
    save_path_pdf = os.path.join(output_dir, "hyperbolic_trajectory_dynamics_white.pdf")
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"✅ Saved Analysis: {save_path} & {save_path_pdf}")

if __name__ == "__main__":
    # (主函数逻辑保持不变)
    if len(sys.argv) > 1:
        trace_dir = sys.argv[1]
    else:
        trace_dir = TRACE_DIR 
        
    if not os.path.exists(trace_dir):
        print(f"❌ Path not found: {trace_dir}")
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        trace_dir = os.path.join(base_dir, "benchmark_reports", "Best", "detailed_traces")
        print(f"🔄 Trying auto-path: {trace_dir}")
    
    if os.path.exists(trace_dir):
        X, T = load_trajectory_data(trace_dir)
        output_dir = os.path.join(os.path.dirname(trace_dir), "visual_analysis")
        plot_dynamic_map(X, T, output_dir)
    else:
        print("❌ Could not find trace directory.")