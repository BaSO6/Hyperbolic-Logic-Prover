# ==========================================
# 文件名: src/system2/benchmark_debug_verbose.py
# 版本: v7.0 Debugger
# 功能: 验证 v13.2 Wrapper + 显式 Import 策略是否生效
# ==========================================

import os
import sys
import time
import json

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

try:
    from src.system2.lean_interaction import LeanEnv
except ImportError:
    print("❌ 无法导入 src.system2.lean_interaction，请检查路径")
    sys.exit(1)

def check_file_existence(path, name):
    if os.path.exists(path):
        print(f"✅ {name} 存在")
        return True
    else:
        print(f"❌ {name} 不存在: {path}")
        return False

def run_diagnostic():
    print(f"\n🔍 开始深度诊断 (Project Root: {project_root})")
    
    # 1. 静态检查
    wrapper_path = os.path.join(project_root, "src", "system2", "run_repl_wrapper.sh")
    check_file_existence(wrapper_path, "Wrapper script")
    
    repl_bin = os.path.join(project_root, "tools", "repl", ".lake", "build", "bin", "repl")
    check_file_existence(repl_bin, "REPL Binary")

    mathlib_olean = os.path.join(project_root, "data", "miniF2F", "_manual_mathlib", ".lake", "build", "lib", "Mathlib.olean")
    if check_file_existence(mathlib_olean, "Mathlib.olean (编译产物)"):
        print("   (这说明 Mathlib 至少编译成功了一部分，核心库是有的)")
    else:
        print("   ⚠️ 警告: Mathlib.olean 缺失，接下来的 import 可能会失败！")

    # 2. 启动进程
    print("\n[Step 1] 启动 LeanEnv (交互模式)...")
    env = None
    try:
        # debug=True 会打印底层 shell 命令
        env = LeanEnv(project_root=project_root, debug=True)
        print("✅ 进程启动成功 (Wrapper 已接管)")
        
        # 3. 显式导入 Mathlib
        print("\n[Step 2] 发送 'import Mathlib'...")
        print("   ⏳ 这可能需要 5-15 秒，因为正在从磁盘加载庞大的 Mathlib...")
        t0 = time.time()
        
        # 这里的 timeout 很关键，Mathlib 很大
        res_import = env.run_command("import Mathlib", timeout=60)
        dt = time.time() - t0
        
        print(f"   ⏱️ 耗时: {dt:.4f}s")
        print(f"   📩 原始响应: {json.dumps(res_import)}")
        
        if "error" in str(res_import).lower() and "messages" in res_import:
            print("❌ Import 返回了错误信息！")
        elif dt < 0.1:
            print("⚠️ 警告: Import 瞬间完成，这通常意味着路径配置错误，Lean 什么都没找到。")
        else:
            print("✅ Import 耗时合理，看起来加载成功了。")

        # 4. Warmup 测试 (战术检查)
        print("\n[Step 3] 执行 Warmup (测试 norm_num)...")
        # 这是一个需要 Mathlib 才能运行的战术
        warmup_code = "example : 1 < 2 := by norm_num"
        res_warmup = env.run_command(warmup_code, timeout=60)
        
        print(f"   📩 原始响应: {json.dumps(res_warmup)}")
        
        if "no goals" in str(res_warmup).lower():
            print("\n🎉🎉🎉 诊断通过！Warmup 成功！")
            print("这意味着 Mathlib 已正确加载，战术可用。")
            print("👉 你现在可以运行 benchmark_minif2f.py 了。")
        else:
            print("\n❌ Warmup 失败。")
            if "unknown tactic" in str(res_warmup):
                print("   原因: 依然找不到战术。Mathlib 虽然 import 了但可能没生效。")
            
    except Exception as e:
        print(f"\n❌ 发生异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if env:
            print("\n[Cleanup] 关闭环境")
            env.close()

if __name__ == "__main__":
    run_diagnostic()