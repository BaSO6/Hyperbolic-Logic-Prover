# ==========================================
# 文件名: src/system2/main_eval.py
# 功能: 系统主入口，负责整合各个模块并运行证明
# ==========================================

import os
import sys
import torch

# 1. 路径补丁 (最优先执行)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# 2. PyTorch 补丁
try:
    import torch.utils._pytree
    if not hasattr(torch.utils._pytree, 'register_pytree_node'):
        def _safe_register_pytree_node(cls, flatten_fn, unflatten_fn, serialized_type_name=None):
            return torch.utils._pytree._register_pytree_node(cls, flatten_fn, unflatten_fn)
        torch.utils._pytree.register_pytree_node = _safe_register_pytree_node
except ImportError:
    pass

# 3. 导入模块 (按依赖顺序)
from src.system2.lean_interaction import LeanEnv
from src.system2.lie_search import RiemannSearchAgent

def main():
    print("============================================================")
    print("🚀 Hyperbolic Neuro-Symbolic Solver (System 2 Launch)")
    print("============================================================")
    
    # 配置路径
    checkpoint_path = os.path.join(project_root, "data", "hgcn_final.pth")
    # 请根据实际情况修改你的 LLM 路径
    llm_path = os.path.join(project_root, "models/deepseek-math-7b-rl") 
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    # 1. 初始化 Lean 环境
    # 我们使用 with 语句确保退出时自动关闭进程
    lean_env = LeanEnv(project_root=project_root, debug=False)
    
    try:
        # 2. 初始化搜索 Agent
        agent = RiemannSearchAgent(
            checkpoint_path=checkpoint_path,
            llm_model_path=llm_path,
            project_root=project_root,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # 3. 定义目标定理
        target_thm = "example (a b c : Nat) : a + b + c = a + c + b"
        
        print(f"\n🧪 Target: {target_thm}")
        print("-" * 30)
        
        # 4. 执行搜索
        # 使用 v21.0 的全局搜索逻辑
        proof = agent.search(lean_env, target_thm, max_steps=20)
        
        print("\n" + "=" * 40)
        if proof:
            print("🏆 SUCCESS! Proof Found:")
            print("   by")
            for step in proof:
                print(f"     {step}")
        else:
            print("❌ Failed to find proof.")
            
    except Exception as e:
        print(f"\n❌ Runtime Error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print("\n[System] Closing Lean Environment...")
        lean_env.close()

if __name__ == "__main__":
    main()