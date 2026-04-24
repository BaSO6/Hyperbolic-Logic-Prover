# gen_baseline_emb.py
import os
import sys
import torch
import gzip
import pickle
from sentence_transformers import SentenceTransformer

# 1. 路径设置
project_root = os.getcwd()
data_dir = os.path.join(project_root, "data")
model_path = os.path.join(project_root, "models", "all-MiniLM-L6-v2")

# 2. 加载模型
print(f"📥 Loading Raw BERT from {model_path}...")
model = SentenceTransformer(model_path)
model.eval()

# 3. 加载定理 ID 映射
map_path = os.path.join(data_dir, "id_to_name.pkl.gz")
print(f"📖 Loading ID map from {map_path}...")
with gzip.open(map_path, "rb") as f:
    id_to_name = pickle.load(f)

# 4. 准备文本列表 (保持顺序一致)
print(f"🔄 Preparing {len(id_to_name)} texts...")
texts = [id_to_name[i] for i in range(len(id_to_name))]

# 5. 批量编码 (Baseline Encoding)
print("🚀 Encoding with Raw BERT (Euclidean Baseline)...")
with torch.no_grad():
    # normalize_embeddings=True 对应 L2 归一化 (欧氏/余弦距离)
    embs = model.encode(texts, convert_to_tensor=True, show_progress_bar=True, normalize_embeddings=True)

# 6. 保存
save_path = os.path.join(data_dir, "raw_bert_embeddings.pt")
torch.save(embs.cpu(), save_path)
print(f"✅ Saved baseline memory to: {save_path}")