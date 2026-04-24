import subprocess
import os

def test_lean_environment():
    """测试 Lean 环境是否正常"""
    
    # 测试1：直接运行 lake 命令
    print("🧪 测试1: 运行 lake 命令...")
    result = subprocess.run(
        ["cd PROJECT_ROOT_PLACEHOLDER/data/mathlib4 && lake env lean --version"],
        shell=True,
        capture_output=True,
        text=True
    )
    print(f"输出: {result.stdout[:100]}...")
    
    # 测试2：运行 REPL 并发送简单命令
    print("\n🧪 测试2: 运行 REPL...")
    script = '''
import json
import subprocess
import time

# 启动 REPL
proc = subprocess.Popen(
    ["PROJECT_ROOT_PLACEHOLDER/tools/repl/.lake/build/bin/repl"],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=0,
    cwd="PROJECT_ROOT_PLACEHOLDER/data/mathlib4"
)

# 发送命令
proc.stdin.write(json.dumps({"cmd": "#eval (1 : ℕ) + 1"}) + "\\n")
proc.stdin.flush()

# 读取响应
time.sleep(1)
output = proc.stdout.read(4096)
print("REPL 响应:", output[:200])

proc.terminate()
'''
    
    with open("/tmp/test_repl.py", "w") as f:
        f.write(script)
    
    result = subprocess.run(["python3", "/tmp/test_repl.py"], capture_output=True, text=True)
    print(f"测试结果: {result.stdout}")

if __name__ == "__main__":
    test_lean_environment()