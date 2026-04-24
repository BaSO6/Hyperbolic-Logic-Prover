import sys
import os
import json
import time

# ==========================================
# 脚本: src/system2/repl_lie_detector.py
# 功能: 发送一个绝对正确的 Mathlib 证明，验证环境是否正常工作
# ==========================================

# 1. 强制将项目根目录加入搜索路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.system2.lean_interaction import LeanEnv

print("🕵️ 启动 REPL 验真仪 (Truth Test)...")

# 2. 启动环境
try:
    env = LeanEnv(project_root=project_root, verbose=True)
    print(f"🔧 REPL 启动成功 (PID: {env.proc.pid})")
except Exception as e:
    print(f"❌ 启动失败: {e}")
    sys.exit(1)

# 3. 初始化 Mathlib
print("\n[Step 1] Import Mathlib...")
res = env.run_command("import Mathlib.Tactic", timeout=300)
print(f"📩 Import 响应: {res}")

if "env" not in res:
    print("❌ Import 失败，无法继续测试。")
    env.close()
    sys.exit(1)

# 记录当前环境 ID
env.current_env = res["env"]
print(f"✅ 环境已就绪，当前 Env ID: {env.current_env}")

# 4. 发送一个必对的证明 (使用 omega 证明 1+1=2)
# 这不仅测试基础逻辑，还测试 Mathlib 的 omega 战术是否可用
proof_code = "example : 1 + 1 = 2 := by omega"
print(f"\n[Step 2] 发送必对证明: {proof_code}")

res = env.run_command(proof_code, timeout=60)
print(f"📩 证明响应: {res}")

# 5. 分析结果
if "env" in res:
    # 检查是否有错误消息
    msgs = res.get("messages", [])
    has_error = False
    for m in msgs:
        if m.get("severity") == "error":
            has_error = True
            print(f"   ❌ 捕获到意外错误: {m.get('data')}")
    
    if not has_error:
        print("\n✅ 结论：REPL 正常工作！(Pass)")
        print("   它成功证明了 '1+1=2'，返回了新的 env ID，且没有报错。")
        print("   这意味着 Mathlib 加载正常，Solver 功能正常。")
    else:
        print("\n😱 结论：REPL 坏掉了！(Fail)")
        print("   虽然这是个正确的命题，但 Lean 报错了。")
        print("   可能是环境配置过严，或者 Mathlib 路径又有问题了。")
else:
    print("\n❓ 结论：REPL 崩溃或超时。")

env.close()