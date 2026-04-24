import os
import shutil
import urllib.request
import tarfile
import sys

# 可能的 URL 列表 (穷举法)
CANDIDATE_URLS = [
    "https://github.com/leanprover-community/repl/releases/download/v4.10.0/repl-linux.tar.gz",
    "https://github.com/leanprover-community/repl/releases/download/v4.10.0-rc2/repl-linux.tar.gz",
    "https://github.com/leanprover-community/repl/releases/download/v4.10.0-rc1/repl-linux.tar.gz",
    "https://github.com/leanprover-community/repl/releases/download/test-lean-v4.10.0/repl-linux.tar.gz", # 有时会有这种非标 tag
]

PROJECT_ROOT = os.getcwd()
DEST_DIR = os.path.join(PROJECT_ROOT, "tools/repl/.lake/build/bin")
DEST_BIN = os.path.join(DEST_DIR, "repl")

print(f"🚑 正在尝试下载 REPL 二进制 (自动寻找正确版本)...")

tar_path = "repl-linux.tar.gz"
downloaded = False

for url in CANDIDATE_URLS:
    print(f"🌐 尝试下载: {url} ...")
    try:
        # 伪装 User-Agent 防止被 GitHub 拦截
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        
        urllib.request.urlretrieve(url, tar_path)
        print("   ✅ 下载成功！")
        downloaded = True
        break
    except Exception as e:
        print(f"   ❌ 失败: {e}")

if not downloaded:
    print("\n💀 所有链接都失败了。")
    print("可能的原因：")
    print("1. GitHub 真的没有为 v4.10.0 发布预编译的 REPL (需要从源码正确编译)。")
    print("2. 你的服务器网络完全连不上 GitHub (但你说下午能连)。")
    sys.exit(1)

# 2. 解压
print("📦 解压中...")
try:
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall("temp_repl")
except Exception as e:
    print(f"❌ 解压失败 (可能是下载了损坏的文件): {e}")
    sys.exit(1)

# 3. 替换
found = False
for root, dirs, files in os.walk("temp_repl"):
    if "repl" in files:
        src = os.path.join(root, "repl")
        if os.access(src, os.X_OK):
            print(f"✅ 找到二进制: {src}")
            os.makedirs(DEST_DIR, exist_ok=True)
            if os.path.exists(DEST_BIN):
                os.remove(DEST_BIN)
            shutil.copy2(src, DEST_BIN)
            os.chmod(DEST_BIN, 0o755)
            print(f"✅ 已替换到: {DEST_BIN}")
            found = True
            break

if not found:
    print("❌ 未在压缩包中找到 repl 二进制！")
    sys.exit(1)

# 4. 清理
if os.path.exists(tar_path): os.remove(tar_path)
if os.path.exists("temp_repl"): shutil.rmtree("temp_repl")

print("\n🎉 修复完成！请再次运行 python src/system2/benchmark_minif2f.py")