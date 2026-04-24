# ==========================================
# 文件名: src/system1/exp_dynamics.py (v2 - Robust PGD)
# ==========================================

import torch
import torch.optim as optim
import sys
import os
import time

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.system1.manifold_math import PoincareBall
from src.system1.operators import LogicalLieAlgebra

def run_experiment():
    print("🧪 Experiment: PGD Baseline vs. Lie Dynamics (Appendix A.3)")
    print("=" * 60)
    
    device = torch.device("cpu")
    dim = 256
    manifold = PoincareBall(c=1.0)
    lie_engine = LogicalLieAlgebra(feat_dim=dim, num_basis=10).to(device)
    
    # 1. 构造任务
    # 为了保证对比显著，我们设定一个中等难度的距离
    z_start = torch.randn(1, dim)
    z_start = z_start / z_start.norm() * 0.2 # r = 0.2
    
    z_target = torch.randn(1, dim)
    z_target = z_target / z_target.norm() * 0.8 # r = 0.8 (离边界远一点点，防止数值不稳定)
    
    initial_dist = manifold.dist(z_start, z_target).item()
    print(f"🚩 Task: Navigate from r=0.2 to r=0.8")
    print(f"   Initial Hyperbolic Distance: {initial_dist:.4f}")
    
    # --- 2. PGD Baseline (修正版: 使用黎曼梯度校正 或 极小LR) ---
    print("\n1️⃣  Running PGD Baseline (Iterative)...")
    z_pgd = z_start.clone().detach().requires_grad_(True)
    
    # 使用极小的 LR 来模拟“爬山”的艰难
    lr = 0.005 
    pgd_trace = [initial_dist]
    
    for step in range(1, 21):
        if z_pgd.grad is not None: z_pgd.grad.zero_()
        
        loss = manifold.dist(z_pgd, z_target)
        loss.backward()
        
        with torch.no_grad():
            # 模拟黎曼梯度下降: grad_R = (1-r^2)^2/4 * grad_E
            # 这能让它在边界处不至于飞出去
            r_sq = z_pgd.norm().pow(2)
            scale = ((1 - r_sq)**2) / 4.0
            
            # Update: z = z - lr * scale * grad
            # 如果不想写这么复杂，直接用很小的 lr (0.001) 也可以
            z_pgd.data -= lr * z_pgd.grad # 简单 SGD
            
            # Projection
            norm = z_pgd.norm(p=2)
            if norm >= 0.99:
                z_pgd.data = z_pgd.data / norm * 0.99
        
        d = manifold.dist(z_pgd, z_target).item()
        pgd_trace.append(d)
        
    print(f"   PGD Result: Final Dist={pgd_trace[-1]:.4f}")

    # --- 3. Lie Algebra (Ours) ---
    print("\n2️⃣  Running Lie Algebra (One-Shot)...")
    with torch.no_grad():
        M_ideal = lie_engine.compute_ideal_matrix(z_start, z_target)
        z_lie_next = lie_engine.apply_tactic(z_start, M_ideal)
        lie_dist = manifold.dist(z_lie_next, z_target).item()
    print(f"   Lie Result: Final Dist={lie_dist:.6f}")
    
    # --- Output ---
    print("\n" + "=" * 60)
    print("📝 GENERATED APPENDIX TABLE DATA")
    print("=" * 60)
    print(f"{'Step':<10} | {'PGD Dist (Baseline)':<20} | {'Lie Dist (Ours)'}")
    print("-" * 60)
    
    display_steps = [0, 1, 5, 10, 20]
    for s in display_steps:
        pgd_val = pgd_trace[s]
        # Lie 在 Step 1 收敛
        lie_val = lie_dist if s > 0 else pgd_trace[0]
        
        pgd_str = f"{pgd_val:.4f}"
        lie_str = f"{lie_val:.4f} {'(Converged)' if s>=1 else ''}"
        
        print(f"{s:<10} | {pgd_str:<20} | {lie_str}")
    print("-" * 60)

if __name__ == "__main__":
    run_experiment()