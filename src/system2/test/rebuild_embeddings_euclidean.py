# ==========================================
# 文件名: src/system2/rebuild_embeddings_euclidean.py
# 功能: 生成标准的 BERT 嵌入 (用于 RAG 基线)
# ==========================================
import os
import sys
import torch
import gzip
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

def main():
    DATA_DIR = os.path.join(project_root, "data")
    MAP_PATH = os.path.join(DATA_DIR, "node_text_map.pkl.gz")
    OUT_PATH = os.path.join(DATA_DIR, "node_embeddings_euclidean.pt") # 新文件名

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 使用设备: {device}")

    # 1. 加载文本数据
    print("📂 加载定理文本...")
    with gzip.open(MAP_PATH, "rb") as f:
        node_text = pickle.load(f)
    
    names = list(node_text.keys())
    texts = list(node_text.values())
    print(f"📚 共有 {len(texts)} 条定理。")

    # 2. 加载标准 BERT 模型 (不带 HGCN)
    print("🤖 加载 SentenceTransformer...")
    model_path = os.path.join(project_root, "models/all-MiniLM-L6-v2")
    if not os.path.exists(model_path):
        model_path = "sentence-transformers/all-MiniLM-L6-v2"
    
    model = SentenceTransformer(model_path, device=device)

    # 3. 批量生成嵌入
    print("⚡ 开始生成嵌入 (这可能需要几分钟)...")
    batch_size = 128
    all_embs = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        with torch.no_grad():
            embs = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
            # 归一化 (这一步对余弦相似度检索至关重要)
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
            all_embs.append(embs.cpu())

    # 4. 保存
    final_tensor = torch.cat(all_embs, dim=0)
    print(f"💾 保存嵌入矩阵: {final_tensor.shape} 到 {OUT_PATH}")
    torch.save(final_tensor, OUT_PATH)
    print("✅ 完成！现在你可以使用欧氏距离检索了。")

if __name__ == "__main__":
    main()