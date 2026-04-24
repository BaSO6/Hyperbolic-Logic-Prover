# ==========================================
# 文件名: src/system1/train_final.py
# 版本: Ultimate Final (基于 c=5.0 实验验证)
# 特性: Fixed High Curvature + DropConnect + Robust Optimization
# ==========================================
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import geoopt
from torch_geometric.utils import softmax
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data

# 1. 环境与种子设置 (确保可复现性)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔥 使用设备: {device} | 模式: Ultimate Final")

# ==========================================
# 2. 核心组件定义
# ==========================================
class HyperbolicLinear(nn.Module):
    """
    支持 DropConnect 的线性层：防止过拟合的核心组件
    """
    def __init__(self, in_features, out_features, bias=True, drop_connect=0.0):
        super().__init__()
        self.drop_connect = drop_connect
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # 使用 Xavier Uniform 初始化，gain 稍小以保持双曲稳定性
        nn.init.xavier_uniform_(self.weight, gain=0.8)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.training and self.drop_connect > 0:
            # DropConnect: 随机 mask 掉权重矩阵中的连接
            mask = torch.rand_like(self.weight) > self.drop_connect
            masked_weight = self.weight * mask.float() / (1 - self.drop_connect)
            return F.linear(x, masked_weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

class HyperbolicGraphConv(nn.Module):
    def __init__(self, in_features, out_features, c, drop_connect=0.2):
        super().__init__()
        self.c = c
        # 使用自定义的 Linear 层
        self.linear = HyperbolicLinear(in_features, out_features, bias=True, drop_connect=drop_connect)
        self.att_query = nn.Linear(out_features, 1)
        self.att_key = nn.Linear(out_features, 1)

    def forward(self, x, edge_index):
        manifold = geoopt.PoincareBall(c=self.c)
        
        # 1. Log Map (进入切空间)
        # 严格截断，防止边界爆炸
        x = x.clamp(min=-0.995, max=0.995)
        x_tan = manifold.logmap0(x)
        
        # 2. Linear Transform (with DropConnect)
        x_trans = self.linear(x_tan)
        
        # 3. Hyperbolic Attention (Attention 机制)
        row, col = edge_index
        q = self.att_query(x_trans)
        k = self.att_key(x_trans)
        # LeakyReLU 处理 attention logits
        alpha = F.leaky_relu(q[row] + k[col], negative_slope=0.2)
        alpha = softmax(alpha, row, num_nodes=x.size(0))
        
        # 4. Aggregation (切空间加权聚合)
        x_aggr_tan = torch.zeros_like(x_trans)
        x_aggr_tan.index_add_(0, row, alpha * x_trans[col])
        
        # [关键] Tangent Clipping (切空间范数截断)
        # 限制切向量长度，防止 expmap 后飞出流形
        norm = x_aggr_tan.norm(dim=-1, keepdim=True)
        clip_coef = 10.0 / (norm + 1e-6)
        clip_coef = torch.clamp(clip_coef, max=1.0)
        x_aggr_tan = x_aggr_tan * clip_coef

        # 5. Project (回到双曲空间)
        out = manifold.expmap0(x_aggr_tan)
        
        # Non-linearity (Hyp -> Tan -> ReLU -> Hyp)
        out_tan = manifold.logmap0(out)
        out_tan = F.relu(out_tan)
        out = manifold.expmap0(out_tan)
        
        return out

class FinalHGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, c_fixed=5.0):
        super().__init__()
        # 🏆 固化最佳曲率 c=5.0
        self.c = c_fixed 
        
        # 初始投影层
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        
        # HGCN Layers (DropConnect = 0.2 是经验最佳值)
        self.layer1 = HyperbolicGraphConv(hidden_dim, hidden_dim, self.c, drop_connect=0.2)
        self.layer2 = HyperbolicGraphConv(hidden_dim, out_dim, self.c, drop_connect=0.2)
        
    def forward(self, x_euclidean, edge_index):
        # 初始特征投影
        x = self.input_proj(x_euclidean)
        
        # 映射到双曲空间
        manifold = geoopt.PoincareBall(c=self.c)
        # 初始特征可能数值较大，先 clamp 再 expmap
        x = torch.clamp(x, min=-10, max=10)
        x = manifold.expmap0(x)
        
        # 图卷积
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        return x

    def decode_dist(self, z, edge_index):
        manifold = geoopt.PoincareBall(c=self.c)
        u = z[edge_index[0]]
        v = z[edge_index[1]]
        # 加上 1e-5 防止距离为 0 导致梯度 NaN
        dist = manifold.dist(u, v) + 1e-5
        return dist

# ==========================================
# 3. 数据加载与处理
# ==========================================
print("📂 加载数据...")
x_features = torch.load("PROJECT_ROOT_PLACEHOLDER/data/node_features_euclidean.pt", map_location='cpu', weights_only=True)
edge_index = torch.load("PROJECT_ROOT_PLACEHOLDER/data/edge_index.pt", map_location='cpu', weights_only=True)
data = Data(x=x_features, edge_index=edge_index)

print("🔪 划分数据集 (Train/Val/Test)...")
# 严格划分：Train 用于 BP，Val 用于早停，Test 用于最终论文汇报
transform = RandomLinkSplit(
    num_val=0.05, 
    num_test=0.1, 
    is_undirected=True, 
    add_negative_train_samples=False
)
train_data, val_data, test_data = transform(data)

# 将数据移入 GPU
train_data = train_data.to(device)
val_data = val_data.to(device)

# [关键] 生成固定的验证集负样本 (Fixed Validation Negatives)
# 确保验证集 Loss 的波动反映模型能力，而不是采样噪声
print("🔒 锁定验证集基准...")
val_pos_edge_index = val_data.edge_label_index
val_neg_edge_index = torch.randint(0, x_features.size(0), val_pos_edge_index.size(), device=device)

# ==========================================
# 4. 训练配置 (Best Practices)
# ==========================================
# 🏆 最佳配置：c=5.0
BEST_C = 5.0
model = FinalHGCN(in_dim=384, hidden_dim=64, out_dim=16, c_fixed=BEST_C).to(device)

# 解码器参数 (可学习)
decoder_r = nn.Parameter(torch.tensor(2.0).to(device))
decoder_t = nn.Parameter(torch.tensor(1.0).to(device))

# 优化器
# Weight Decay = 1e-4 防止过拟合
optimizer = optim.Adam([
    {'params': model.parameters(), 'lr': 0.001, 'weight_decay': 1e-4},
    {'params': [decoder_r, decoder_t], 'lr': 0.01}
])

# 学习率调度器：当 Loss 不降时减半
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)

# Fermi-Dirac Loss (Hyperbolic Link Prediction Standard)
def fermi_dirac_loss(dist, r, t, is_pos):
    t = t + 1e-5 # 数值稳定
    if is_pos:
        # 正样本：希望距离 < r
        return F.softplus((dist - r) / t).mean()
    else:
        # 负样本：希望距离 > r
        return F.softplus(-(dist - r) / t).mean()

# ==========================================
# 5. 训练主循环 (Production Ready)
# ==========================================
print(f"🚀 启动终极训练 (Curvature c={BEST_C})...")

best_val_loss = float('inf')
patience_counter = 0
MAX_PATIENCE = 50 # 允许 50 轮不更新最佳结果，充分收敛

for epoch in range(501): # 设置 500 轮，靠早停停止
    model.train()
    optimizer.zero_grad()
    
    # --- 1. Forward ---
    z = model(train_data.x, train_data.edge_index)
    
    # --- 2. Sampling ---
    pos_idx = train_data.edge_label_index
    # Mini-batch 采样 (显存优化)
    batch_size = 40000 
    if pos_idx.size(1) > batch_size:
        perm = torch.randperm(pos_idx.size(1))[:batch_size]
        pos_batch = pos_idx[:, perm]
        neg_batch = torch.randint(0, z.size(0), (2, batch_size), device=device)
    else:
        pos_batch = pos_idx
        neg_batch = torch.randint(0, z.size(0), pos_idx.size(), device=device)

    # --- 3. Loss Calculation ---
    pos_dist = model.decode_dist(z, pos_batch)
    neg_dist = model.decode_dist(z, neg_batch)
    
    r_val = F.softplus(decoder_r)
    t_val = F.softplus(decoder_t)
    
    loss = fermi_dirac_loss(pos_dist, r_val, t_val, True) + \
           fermi_dirac_loss(neg_dist, r_val, t_val, False)
    
    # --- 4. Backward & Update ---
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪
    optimizer.step()
    
    # --- 5. Validation ---
    # 每 5 轮验证一次，加快训练速度
    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            z_val = model(val_data.x, val_data.edge_index)
            # 使用固定的验证集样本
            v_p_d = model.decode_dist(z_val, val_pos_edge_index)
            v_n_d = model.decode_dist(z_val, val_neg_edge_index)
            
            val_loss = fermi_dirac_loss(v_p_d, r_val, t_val, True) + \
                       fermi_dirac_loss(v_n_d, r_val, t_val, False)
        
        # 调度器更新
        scheduler.step(val_loss)
        
        # 早停与模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 保存最终模型
            torch.save({
                'model': model.state_dict(),
                'r': decoder_r,
                't': decoder_t,
                'c': BEST_C
            }, "PROJECT_ROOT_PLACEHOLDER/data/hgcn_final.pth")
            save_msg = "💾 Saved"
        else:
            patience_counter += 1
            save_msg = ""
            
        print(f"Ep {epoch:03d} | Tr={loss:.4f} | Val={val_loss:.4f} | r={r_val.item():.2f} | Patience={patience_counter} {save_msg}")
        
        if patience_counter >= MAX_PATIENCE:
            print(f"⏹️ 早停触发！训练结束。最佳 Val Loss: {best_val_loss:.4f}")
            break

print("✅ 终极模型已就绪: data/hgcn_final.pth")
print("下一步：使用此 Embedding 进行 System 2 的 MCTS 搜索。")