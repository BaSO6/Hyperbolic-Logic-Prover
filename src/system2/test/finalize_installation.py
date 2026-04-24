import os
import subprocess
import glob
import sys

# ==========================================
# 脚本: src/system2/finalize_installation.py
# 版本: v2.3 (Variable Name Fixed)
# 功能: 
#   1. 自动定位 libleanshared.so (绝对路径)
#   2. 手动链接 REPL
#   3. 完成 Mathlib 编译
# ==========================================

PROJECT_ROOT = os.getcwd()
REPL_DIR = os.path.join(PROJECT_ROOT, "tools/repl")
FAKE_LIBS = os.path.join(REPL_DIR, "fake_libs")
LEAN_TOOLCHAIN = "/home/ma-user/.elan/toolchains/leanprover-lean4-v4.10.0"
MATHLIB_DIR = os.path.join(PROJECT_ROOT, "data/miniF2F/_manual_mathlib")

def setup_fake_libs():
    """建立库软链接，骗过链接器"""
    if not os.path.exists(FAKE_LIBS): os.makedirs(FAKE_LIBS)
    print(f"🔧 配置动态库链接: {FAKE_LIBS}")
    
    libs = {
        "libgmp.so": ["/usr/lib/x86_64-linux-gnu/libgmp.so.10", "/usr/lib64/libgmp.so.10"],
        "libstdc++.so": ["/usr/lib/x86_64-linux-gnu/libstdc++.so.6", "/usr/lib64/libstdc++.so.6"]
    }
    
    for link_name, sources in libs.items():
        link_path = os.path.join(FAKE_LIBS, link_name)
        if os.path.exists(link_path): os.remove(link_path)
        for src in sources:
            if os.path.exists(src):
                os.symlink(src, link_path)
                print(f"   ✅ {link_name} -> {src}")
                break
        else:
            print(f"   ⚠️ 未找到系统库: {link_name}")

def find_library_file(base_dir, lib_name):
    """递归查找库文件的绝对路径"""
    print(f"🔎 正在 {base_dir} 中搜索 lib{lib_name}.so ...")
    target = f"lib{lib_name}.so"
    for root, dirs, files in os.walk(base_dir):
        if target in files:
            full_path = os.path.join(root, target)
            print(f"   ✅ 锁定文件: {full_path}")
            return full_path
    return None

def link_repl():
    """手动调用 g++ 链接 REPL"""
    print("\n🔨 [1/2] 手动链接 REPL 二进制...")
    
    build_ir = os.path.join(REPL_DIR, ".lake", "build", "ir")
    output_bin = os.path.join(REPL_DIR, ".lake", "build", "bin", "repl")
    
    # 1. 寻找 libleanshared.so 的绝对路径
    leanshared_path = find_library_file(LEAN_TOOLCHAIN, "leanshared")
    
    # [修复] 修正了变量名拼写错误
    if not leanshared_path:
        # 尝试默认猜测
        guess = os.path.join(LEAN_TOOLCHAIN, "lib", "lean", "libleanshared.so")
        if os.path.exists(guess):
             leanshared_path = guess
             print(f"   ⚠️ 搜索失败，使用猜测路径: {guess}")
        else:
             print("❌ 致命错误：找不到 libleanshared.so")
             return False
    
    # 2. 收集对象文件
    objects = []
    for root, _, files in os.walk(build_ir):
        for f in files:
            if f.endswith(".o") or f.endswith(".o.export"):
                objects.append(os.path.join(root, f))
    
    if not objects:
        print("❌ 错误：找不到编译好的 .o 文件")
        return False
        
    print(f"   📦 找到 {len(objects)} 个对象文件")
    
    # 3. 构造 g++ 命令 (直接注入文件路径)
    cmd = [
        "g++", "-o", output_bin, "-rdynamic",
        *objects,
        leanshared_path, # 直接传入文件路径
        f"-L{FAKE_LIBS}",
        "-lgmp", "-lm"
    ]
    
    os.makedirs(os.path.dirname(output_bin), exist_ok=True)
    
    try:
        subprocess.run(cmd, check=True)
        print("   🎉 REPL 链接成功！")
        
        if os.path.exists(output_bin):
            st = os.stat(output_bin)
            os.chmod(output_bin, st.st_mode | 0o111)
            return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ 链接失败: {e}")
        return False

def build_mathlib():
    """继续编译 Mathlib"""
    print("\n🔨 [2/2] 继续编译 Mathlib...")
    lake_bin = os.path.join(LEAN_TOOLCHAIN, "bin", "lake")
    env = os.environ.copy()
    env["LEAN_CC"] = "gcc"
    env["LEAN_PATH"] = os.path.join(LEAN_TOOLCHAIN, "lib", "lean")
    if "ELAN_TOOLCHAIN" in env: del env["ELAN_TOOLCHAIN"]
    
    try:
        subprocess.run([lake_bin, "build"], cwd=MATHLIB_DIR, env=env, check=True)
        print("   🎉 Mathlib 编译成功！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Mathlib 编译失败: {e}")
        return False

def main():
    setup_fake_libs()
    if link_repl():
        build_mathlib()
        print("\n✅✅✅ 全流程修复完成！")
        print("👉 请立即运行: python src/system2/benchmark_debug_verbose.py")

if __name__ == "__main__":
    main()