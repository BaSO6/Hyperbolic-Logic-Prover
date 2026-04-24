# ==============================================================================
# Filename: src/system1/train_hgcn_dim.py
# Version: v1.0
#
# Purpose: Retrain HGCN at multiple OUT_DIM values for the dimension scaling
#          ablation. Saves each checkpoint as:
#            results/dimension_scaling/hgcn_d{D}.pth
#          NEVER overwrites hgcn_final.pth or any production file.
#
# Architecture: identical to train_final.py v12.1 (FinalHGCN with residual
#               hyperbolic layer, HIDDEN_DIM=256 fixed, only OUT_DIM varies).
#
# A100 notes:
#   - 80GB VRAM is far more than needed (~3GB per run for 110k nodes)
#   - Each dim trains in ~5–8 min on A100 (201 epochs)
#   - All dims run sequentially in one command
#
# Usage:
#   # First check your paper OUT_DIM:
#   python3 -c "import torch; c=torch.load('data/hgcn_final.pth',
#       map_location='cpu'); print(c.get('out_dim'),
#       c['model']['layer.semantic_proj.weight'].shape)"
#
#   # Train all dims (skip the paper dim — it already exists)
#   python src/system1/train_hgcn_dim.py --dims 16 32 128 256
#
#   # Force re-train even if checkpoint exists
#   python src/system1/train_hgcn_dim.py --dims 16 32 128 256 --force
# ==============================================================================

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.system1.manifold_math import PoincareBall   # type: ignore

DATA_DIR   = os.path.join(project_root, "data")
OUTPUT_DIR = os.path.join(project_root, "results", "dimension_scaling")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fixed hyperparameters matching train_final.py v12.1
HIDDEN_DIM  = 256
CURVATURE_C = 1.0
LR          = 0.005
EPOCHS      = 201


# ==============================================================================
# Architecture (verbatim from train_final.py v12.1)
# ==============================================================================

class EuclideanGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index):
        x_trans = self.linear(x)
        if edge_index.size(1) == 0:
            return x_trans
        row, col = edge_index
        out = torch.zeros_like(x_trans)
        deg = torch.zeros(x.size(0), 1, device=x.device)
        deg.index_add_(0, row, torch.ones(row.size(0), 1, device=x.device))
        out.index_add_(0, row, x_trans[col])
        return F.relu(out / (deg + 1e-8))


class HyperbolicResidualLayer(nn.Module):
    def __init__(self, in_dim, out_dim, c=1.0):
        super().__init__()
        self.manifold      = PoincareBall(c)
        self.semantic_proj = nn.Linear(in_dim, out_dim)
        self.structure_proj = nn.Linear(in_dim, out_dim)
        self.graph_conv    = EuclideanGraphConv(out_dim, out_dim)
        self.gate          = nn.Linear(out_dim * 2, 1)
        self.scale         = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_index):
        z_sem    = self.semantic_proj(x)
        z_struct = F.relu(self.structure_proj(x))
        z_struct = self.graph_conv(z_struct, edge_index)
        alpha    = torch.sigmoid(
            self.gate(torch.cat([z_sem, z_struct], dim=-1))
        )
        z_tan   = alpha * z_sem + (1 - alpha) * z_struct
        x_norm  = z_tan.norm(dim=-1, keepdim=True) + 1e-8
        radius  = 0.9 * torch.tanh(self.scale)
        return self.manifold.expmap0(z_tan / x_norm * radius)


class FinalHGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, c=1.0):
        super().__init__()
        self.layer      = HyperbolicResidualLayer(in_dim, out_dim, c)
        self.manifold   = self.layer.manifold
        self.hidden_dim = hidden_dim
        self.out_dim    = out_dim
        self.c          = c

    def forward(self, x, edge_index):
        return self.layer(x, edge_index)


# ==============================================================================
# Training
# ==============================================================================

def train_one_dim(out_dim: int, device: torch.device,
                  x: torch.Tensor, edge_index: torch.Tensor,
                  force: bool = False) -> str:
    """
    Train HGCN with OUT_DIM=out_dim. Returns path to saved checkpoint.
    Never overwrites hgcn_final.pth.
    """
    save_path = os.path.join(OUTPUT_DIR, f"hgcn_d{out_dim}.pth")

    if os.path.exists(save_path) and not force:
        print(f"  ↩️  Already exists: {save_path}  (use --force to retrain)")
        return save_path

    print(f"\n{'─'*60}")
    print(f"  🚀 Training d={out_dim}  "
          f"(HIDDEN={HIDDEN_DIM}, OUT={out_dim}, {EPOCHS} epochs)")
    print(f"{'─'*60}")

    model     = FinalHGCN(x.shape[1], HIDDEN_DIM, out_dim, CURVATURE_C).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    has_edges = edge_index.size(1) > 100

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        z = model(x, edge_index)

        norms    = z.norm(dim=-1)
        loss_reg = ((norms - 0.8) ** 2).mean()
        loss_link = torch.tensor(0.0, device=device)

        if has_edges:
            perm      = torch.randperm(edge_index.size(1), device=device)[:10_000]
            u, v      = edge_index[:, perm]
            pos_dist  = model.manifold.dist(z[u], z[v])
            neg_v     = torch.randint(0, z.size(0), (len(u),), device=device)
            neg_dist  = model.manifold.dist(z[u], z[neg_v])
            loss_link = F.relu(pos_dist - neg_dist + 1.0).mean()

        loss = loss_link + 0.1 * loss_reg
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"  Ep {epoch:03d} | loss={loss.item():.4f}  "
                  f"link={loss_link.item():.4f}  reg={loss_reg.item():.4f}")

    torch.save({
        "model":      model.state_dict(),
        "c":          CURVATURE_C,
        "hidden_dim": HIDDEN_DIM,
        "out_dim":    out_dim,
        "note":       f"dim_scaling_d{out_dim}",
    }, save_path)
    print(f"  ✅ Saved → {save_path}")
    return save_path


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Retrain HGCN at multiple OUT_DIM values (dim scaling ablation)")
    parser.add_argument(
        "--dims", nargs="+", type=int, required=True,
        help="List of OUT_DIM values to train, e.g. --dims 16 32 128 256")
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Training epochs per dim (default {EPOCHS})")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-train even if checkpoint already exists")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"   GPU: {props.name}  "
              f"VRAM: {props.total_memory / 1e9:.0f}GB")

    # Load features once — shared across all dim runs
    x_path  = os.path.join(DATA_DIR, "node_features_euclidean.pt")
    ei_path = os.path.join(DATA_DIR, "edge_index.pt")
    if not os.path.exists(x_path):
        print(f"❌ node_features_euclidean.pt not found in {DATA_DIR}")
        print("   Run src/system1/preprocess_integrated.py first.")
        return
    if not os.path.exists(ei_path):
        print(f"❌ edge_index.pt not found in {DATA_DIR}")
        return

    print(f"\n📥 Loading features (shared across all dims)...")
    x          = torch.load(x_path, map_location=device)
    edge_index = torch.load(ei_path, map_location=device)
    print(f"   x: {x.shape}   edge_index: {edge_index.shape}")

    # Warn if any requested dim matches production
    prod_path = os.path.join(DATA_DIR, "hgcn_final.pth")
    prod_dim  = None
    if os.path.exists(prod_path):
        ckpt     = torch.load(prod_path, map_location="cpu")
        prod_dim = ckpt.get("out_dim")
        if prod_dim is None:
            for key in ("layer.semantic_proj.weight", "semantic_proj.weight"):
                if key in ckpt.get("model", {}):
                    prod_dim = ckpt["model"][key].shape[0]; break
        print(f"\n📌 Production checkpoint: {prod_path}")
        print(f"   OUT_DIM = {prod_dim}  ← this will NOT be overwritten")

    print(f"\n📐 Dims to train: {sorted(args.dims)}")
    print(f"   Saving to:     {OUTPUT_DIR}/hgcn_d{{D}}.pth")
    print()

    for d in sorted(args.dims):
        if d == prod_dim:
            print(f"\n⏭️  Skipping d={d} — matches production OUT_DIM={prod_dim}.")
            print(f"   Production result is already at 65.75%. No retraining needed.")
            continue
        train_one_dim(d, device, x, edge_index, force=args.force)

    print(f"\n{'='*60}")
    print(f"  ✅ Done. Next step:")
    print(f"  python src/system1/export_embeddings_dim.py \\")
    print(f"      --dims {' '.join(str(d) for d in sorted(args.dims))}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
