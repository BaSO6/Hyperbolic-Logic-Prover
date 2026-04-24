import os
import subprocess

# ==========================================
# 脚本: verify_reality.py
# 功能: 绕过 REPL，直接使用编译器验证环境的诚实度
# ==========================================

PROJECT_ROOT = os.getcwd()
WORK_DIR = os.path.join(PROJECT_ROOT, "data/miniF2F")
TOOLCHAIN = "/home/ma-user/.elan/toolchains/leanprover-lean4-v4.10.0"

# 1. 构造一个必定失败的 Lean 文件
# omega 绝对推不出 1 = 2
sanity_code = """
import Mathlib.Tactic

example : 1 = 2 := by omega
"""

file_path = os.path.join(WORK_DIR, "sanity_check.lean")

print("🧪 正在创建测谎文件 sanity_check.lean ...")
print(f"   内容: example : 1 = 2 := by omega")
with open(file_path, "w") as f:
    f.write(sanity_code)

# 2. 借用环境 (和 Wrapper 一样的逻辑)
print("\n🕵️  正在窃取 _manual_mathlib 的环境...")
manual_lib = os.path.join(PROJECT_ROOT, "data/miniF2F/_manual_mathlib")
try:
    # 获取正确的 LEAN_PATH
    lean_path = subprocess.check_output(
        [f"{TOOLCHAIN}/bin/lake", "env", "printenv", "LEAN_PATH"], 
        cwd=manual_lib, 
        text=True
    ).strip()
    print("   ✅ 成功获取 LEAN_PATH")
except Exception as e:
    print(f"   ❌ 获取环境失败: {e}")
    exit(1)

# 3. 直接调用 lean 编译器
# 如果环境正常，这里必须报错！
print("\n🔥 开始“火刑”测试 (直接编译)...")
cmd = [f"{TOOLCHAIN}/bin/lean", "sanity_check.lean"]
env = os.environ.copy()
env["LEAN_PATH"] = lean_path

try:
    # capture_output=True 会捕获 stdout/stderr
    result = subprocess.run(cmd, cwd=WORK_DIR, env=env, capture_output=True, text=True)
    
    print("\n================ 编译器输出 ================")
    print(result.stderr)
    print(result.stdout)
    print("============================================")
    
    if result.returncode != 0:
        print("\n✅ 测试通过！编译器报错了。")
        print("   这意味着底层的 Lean 还是诚实的。问题出在 REPL 或 Python 交互层。")
    else:
        print("\n💀 测试失败！编译器竟然通过了 1=2 ？？？")
        print("   这意味着世界毁灭了，或者 import Mathlib 根本没生效（导致 example 被忽略）。")

except Exception as e:
    print(f"❌ 执行出错: {e}")