# ==========================================
# 文件名: src/system2/test_system1_retrieval.py
# 功能: 验证双曲空间检索 (System 1) 的相关性
# ==========================================

import os
import sys
import torch
import gzip
import pickle

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.system2.lie_search import RiemannSearchAgent

def test_retrieval():
    # 1. 路径配置 [FIXED]
    # 指向新下载的 DeepSeek-Prover-V1.5-RL
    MODEL_DIR = os.path.join(project_root, "models/DeepSeek-Prover-V1.5-RL") 
    CKPT_PATH = os.path.join(project_root, "data/hgcn_final.pth")
    
    print(f"📂 Model Path: {MODEL_DIR}")
    if not os.path.exists(MODEL_DIR):
        print("❌ 错误: 模型路径不存在！请确认你已经运行了 download_model.sh")
        return

    print("🚀 Loading Riemann Agent (System 1)...")
    try:
        # 我们复用 Agent 的初始化逻辑来加载 HGCN 和 Embeddings
        # 注意：这也会加载 LLM，虽然这个脚本不用它，但为了复用代码必须加载
        agent = RiemannSearchAgent(CKPT_PATH, MODEL_DIR, device="cuda")
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    if agent.graph_emb is None:
        print("❌ 错误: 未找到知识库嵌入 (data/node_embeddings.pt)。")
        print("   System 1 无法工作。请先运行 embed_mathlib.py 生成嵌入。")
        return

    # 2. 定义测试用例 (模拟 Lean 的 Goal 状态)
    test_cases = [
        {
            "name": "素数定义",
            "goal": "n : ℕ\n⊢ Nat.Prime n ↔ 2 ≤ n ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n"
        },
        {
            "name": "平方差公式",
            "goal": "a b : ℝ\n⊢ a^2 - b^2 = (a + b) * (a - b)"
        },
        {
            "name": "GCD 性质",
            "goal": "a b : ℕ\n⊢ Nat.gcd a b ∣ a"
        },
        {
            "name": "AMC12 难题 (模运算)",
            "goal": "n : ℕ\n⊢ (2^n - 1) % 3 = 0"
        }
    ]

    # 3. 执行检索测试
    print(f"\n📚 Knowledge Base Size: {len(agent.idx_to_name)} theorems")
    print("=" * 60)

    for case in test_cases:
        print(f"\n🔍 Query: [{case['name']}]")
        print(f"   Goal: {case['goal'].replace(chr(10), ' ')}") # 去掉换行显示
        
        # 调用检索
        # 注意：这里我们手动调用 agent 内部逻辑
        hints = agent.retrieve_theorems(case['goal'], k=5)
        
        print(f"   Relevance Top-5:")
        for i, hint in enumerate(hints):
            print(f"     {i+1}. {hint}")
            
    print("\n" + "=" * 60)
    print("✅ 分析指南:")
    print("1. 如果检索结果包含 'Prime', 'sq_sub_sq', 'gcd_dvd' 等相关词，说明 HGCN 训练成功。")
    print("2. 如果检索结果全是随机的 'List.map' 或 'Category.id'，说明嵌入空间是混乱的。")

if __name__ == "__main__":
    test_retrieval()