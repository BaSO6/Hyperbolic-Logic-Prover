import subprocess
import time
import os
import threading

# 1. 定义 Wrapper 路径
wrapper_path = "src/system2/run_repl_wrapper.sh"

print(f"🚀 启动裸机调试: {wrapper_path}")

# 2. 启动进程 (管道全开)
process = subprocess.Popen(
    ["sh", wrapper_path],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=0 # 无缓冲，实时输出
)

# 3. 定义读取线程 (防止死锁)
def reader(pipe, name):
    for line in pipe:
        print(f"[{name}] {line.strip()}")

t_out = threading.Thread(target=reader, args=(process.stdout, "REPL_OUT"))
t_err = threading.Thread(target=reader, args=(process.stderr, "REPL_ERR"))
t_out.daemon = True
t_err.daemon = True
t_out.start()
t_err.start()

# 4. 交互函数
def send_cmd(cmd_str):
    print(f"\n👉 发送: {cmd_str}")
    try:
        process.stdin.write(cmd_str + "\n")
        process.stdin.flush()
        # 给一点时间让它反应
        time.sleep(2) 
    except Exception as e:
        print(f"❌ 发送失败: {e}")

# 5. 开始测试流程
time.sleep(1)
if process.poll() is not None:
    print("❌ 进程启动即崩溃！")
    exit(1)

# 测试 A: 加载 Mathlib.Tactic (这是问题的核心)
send_cmd('{"cmd": "import Mathlib.Tactic"}')

# 等待一会，看看有没有输出
time.sleep(5)

# 测试 B: 检查 Refl
# 注意：这里我们故意不带 env，看看它是个什么反应，或者带上 env: 0
send_cmd('{"cmd": "example : 1 = 1 := by refl", "env": 0}')

time.sleep(5)

print("\n🛑 测试结束，强制关闭进程...")
process.kill()
