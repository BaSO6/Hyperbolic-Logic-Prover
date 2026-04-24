# ==========================================
# Filename: src/system1/analyze_traces.py
# Version: v2.1 (Fix: 3D Array Dimension Error)
# ==========================================

import os
import sys
import gzip
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse

# -------------------------------------------------
# Configuration Section
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_ROOT = os.path.join(BASE_DIR, "benchmark_reports")

# -------------------------------------------------
# Core Logic
# -------------------------------------------------

def find_target_dir(target_name=None):
    """Smartly search for directories containing data"""
    if target_name:
        path = os.path.join(REPORT_ROOT, target_name, "detailed_traces")
        if os.path.exists(path): return path
        if os.path.exists(target_name): return target_name
    
    candidates = glob.glob(os.path.join(REPORT_ROOT, "*", "detailed_traces"))
    candidates.sort(key=os.path.getmtime, reverse=True)
    
    for path in candidates:
        if len(glob.glob(os.path.join(path, "*.pkl.gz"))) > 0:
            return path
    return None

def load_all_traces(trace_dir):
    files = glob.glob(os.path.join(trace_dir, "*.pkl.gz"))
    print(f"🔄 Loading {len(files)} traces from {trace_dir}...")
    
    all_data = []
    for f_path in files:
        try:
            with gzip.open(f_path, "rb") as f:
                data = pickle.load(f)
                if 'full_trace' in data and len(data['full_trace']) > 0:
                    all_data.append(data)
        except: pass
    return all_data

def extract_flat_coord(step):
    """[Fix] Safely extract and flatten coordinates, handling (1, D) and (D,) cases"""
    if 'current_coord' not in step:
        return None
    c = step['current_coord']
    # If it is a nested list [[...]], take the first element
    if isinstance(c, list) and len(c) > 0 and isinstance(c[0], list):
        return c[0]
    return c

# -------------------------------------------------
# Plotting Functions
# -------------------------------------------------

def plot_global_manifold(all_data, output_dir):
    """
    [Paper Tool] Global Manifold Map
    """
    print("🎨 Generating Global Manifold Map (This is for your Paper)...")
    
    all_coords = []
    success_endpoints = []
    
    for data in all_data:
        trace = data['full_trace']
        status = data['meta']['status']
        
        # [Fix] Use helper function to extract flattened coordinates
        trace_coords = []
        for t in trace:
            c = extract_flat_coord(t)
            if c: trace_coords.append(c)
            
        if not trace_coords: continue
        
        all_coords.extend(trace_coords)
        
        if status == 'Success':
            success_endpoints.append(trace_coords[-1])

    if not all_coords: 
        print("⚠️ No valid coordinates found!")
        return

    # 2. Global PCA dimensionality reduction
    X = np.array(all_coords)
    # [Double Check] Ensure it is 2D
    if X.ndim == 3: X = X.reshape(X.shape[0], -1)
    
    print(f"   PCA Input Shape: {X.shape}")
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)
    
    # Normalize to Poincaré disk (Unit Disk)
    scale = np.percentile(np.linalg.norm(X_2d, axis=1), 99)
    # Avoid division by zero
    if scale == 0: scale = 1.0
    X_2d = X_2d / (scale * 1.1)

    # 3. Plot Heatmap (Thinking Shape)
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    
    # Background Circle
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    # Heatmap
    hb = plt.hexbin(X_2d[:, 0], X_2d[:, 1], gridsize=50, cmap='inferno', mincnt=1, bins='log')
    cb = plt.colorbar(hb, label='Log Search Density')
    
    # Overlay Success Points (Green Stars)
    if success_endpoints:
        succ_arr = np.array(success_endpoints)
        if succ_arr.ndim == 3: succ_arr = succ_arr.reshape(succ_arr.shape[0], -1)
        
        succ_2d = pca.transform(succ_arr) / (scale * 1.1)
        plt.scatter(succ_2d[:, 0], succ_2d[:, 1], c='#00FF00', s=60, marker='*', 
                    edgecolors='white', linewidth=0.3, label='Solved Goals', alpha=0.9)

    plt.title(f"Hyperbolic Knowledge Manifold (N={len(all_data)} Problems)", fontsize=14)
    plt.legend(loc='upper right')
    plt.axis('equal')
    plt.axis('off')
    
    save_path = os.path.join(output_dir, "PAPER_global_manifold.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Global map saved to: {save_path}")
    plt.close()

def plot_single_trajectory(data, pca_model, scale, output_dir):
    """Plot single problem trajectory"""
    meta = data['meta']
    name = meta['name']
    trace = data['full_trace']
    
    coords = []
    for t in trace:
        c = extract_flat_coord(t)
        if c: coords.append(c)
        
    if len(coords) < 2: return

    X = np.array(coords)
    if X.ndim == 3: X = X.reshape(X.shape[0], -1)
    
    X_2d = pca_model.transform(X) / (scale * 1.1)
    
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.add_patch(plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', alpha=0.3))
    
    plt.plot(X_2d[:, 0], X_2d[:, 1], 'b-', alpha=0.4, linewidth=1)
    plt.scatter(X_2d[0, 0], X_2d[0, 1], c='blue', s=30, label='Start')
    
    end_color = 'green' if meta['status'] == 'Success' else 'red'
    plt.scatter(X_2d[-1, 0], X_2d[-1, 1], c=end_color, s=100, marker='*', label=meta['status'])
    
    plt.xlim(-1.1, 1.1); plt.ylim(-1.1, 1.1)
    plt.axis('off')
    plt.title(f"{name}\nSteps: {meta['steps']}")
    
    plt.savefig(os.path.join(output_dir, f"trace_{name}.png"), dpi=100)
    plt.close()

# -------------------------------------------------
# Main Entry Point
# -------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    args = parser.parse_args()

    # 1. Find directory
    trace_dir = find_target_dir(args.dir)
    if not trace_dir:
        print("❌ Could not find traces.")
        sys.exit(1)
        
    print(f"📂 Selected: {trace_dir}")
    
    # 2. Load data
    all_data = load_all_traces(trace_dir)
    if not all_data:
        print("❌ No valid data loaded.")
        sys.exit(1)

    # 3. Prepare output
    vis_dir = os.path.join(os.path.dirname(trace_dir), "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 4. Generate global map
    try:
        # To plot individual cases, we need a global PCA model
        all_coords_list = []
        for d in all_data:
            for t in d['full_trace']:
                c = extract_flat_coord(t)
                if c: all_coords_list.append(c)
        
        all_coords_np = np.array(all_coords_list)
        if all_coords_np.ndim == 3: all_coords_np = all_coords_np.reshape(all_coords_np.shape[0], -1)
        
        pca = PCA(n_components=2).fit(all_coords_np)
        scale = np.percentile(np.linalg.norm(pca.transform(all_coords_np), axis=1), 99)
        if scale == 0: scale = 1.0

        plot_global_manifold(all_data, vis_dir)
        
        # 5. Generate case study plots (Top 10 Hardest Success)
        print("🎨 Generating Case Study Plots...")
        success_cases = [d for d in all_data if d['meta']['status'] == 'Success']
        success_cases.sort(key=lambda x: x['meta'].get('nodes', 0), reverse=True)
        
        for case in success_cases[:10]:
            plot_single_trajectory(case, pca, scale, vis_dir)
            
    except Exception as e:
        print(f"❌ Error during plotting: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\n✨ All Done! Check output at: {vis_dir}")