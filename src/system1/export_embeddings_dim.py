# ==============================================================================
# Filename: src/system1/export_embeddings_dim.py
# Version: v1.0
#
# Purpose: Forward-pass the trained HGCN for each OUT_DIM and save the
#          resulting node embeddings as:
#            results/dimension_scaling/node_emb_d{D}.pt
#          NEVER overwrites node_embeddings.pt (production file).
#
# Run AFTER src/system1/train_hgcn_dim.py.
#
# Usage:
#   python src/system1/export_embeddings_dim.py --dims 16 32 128 256
#   python src/system1/export_embeddings_dim.py --dims 16 --force
# ==============================================================================

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.system1.manifold_math import PoincareBall   # type: ignore

DATA_DIR   = os.path.join(project_root, "data")
OUTPUT_DIR = os.path.join(project_root, "results", "dimension_scaling")
os.makedirs(OUTPUT_DIR, exist_ok=True)

HIDDEN_DIM = 256


# ==============================================================================
# Same architecture as train_hgcn_dim.py — must be identical
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
        z_tan  = alpha * z_sem + (1 - alpha) * z_struct
        x_norm = z_tan.norm(dim=-1, keepdim=True) + 1e-8
        radius = 0.9 * torch.tanh(self.scale)
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
# Export
# ==============================================================================

def export_one_dim(out_dim: int, device: torch.device,
                   x: torch.Tensor, edge_index: torch.Tensor,
                   force: bool = False) -> str:
    ckpt_path = os.path.join(OUTPUT_DIR, f"hgcn_d{out_dim}.pth")
    emb_path  = os.path.join(OUTPUT_DIR, f"node_emb_d{out_dim}.pt")

    # Check production dim — use production checkpoint if matching
    prod_ckpt = os.path.join(DATA_DIR, "hgcn_final.pth")
    prod_emb  = os.path.join(DATA_DIR, "node_embeddings.pt")
    prod_dim  = None
    if os.path.exists(prod_ckpt):
        ck = torch.load(prod_ckpt, map_location="cpu")
        prod_dim = ck.get("out_dim")
        if prod_dim is None:
            for key in ("layer.semantic_proj.weight", "semantic_proj.weight"):
                if key in ck.get("model", {}):
                    prod_dim = ck["model"][key].shape[0]; break

    if out_dim == prod_dim:
        # Just copy the production embeddings to our output dir
        if not os.path.exists(emb_path) or force:
            import shutil
            shutil.copy2(prod_emb, emb_path)
            print(f"  ✅ d={out_dim} (paper dim): copied {prod_emb} → {emb_path}")
        else:
            print(f"  ↩️  d={out_dim} (paper dim): already exists {emb_path}")
        return emb_path

    if os.path.exists(emb_path) and not force:
        print(f"  ↩️  Already exists: {emb_path}  (use --force to re-export)")
        return emb_path

    if not os.path.exists(ckpt_path):
        print(f"  ❌ Checkpoint not found: {ckpt_path}")
        print(f"     Run: python src/system1/train_hgcn_dim.py --dims {out_dim}")
        return ""

    print(f"\n{'─'*60}")
    print(f"  🚀 Exporting embeddings d={out_dim}  ({x.shape[0]} nodes)...")

    ckpt  = torch.load(ckpt_path, map_location=device)
    c     = ckpt.get("c", 1.0)
    model = FinalHGCN(x.shape[1], HIDDEN_DIM, out_dim, c).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        z = model(x, edge_index)

    torch.save(z.cpu(), emb_path)
    print(f"  ✅ Saved embeddings {z.shape} → {emb_path}")
    return emb_path


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export node embeddings for each dim (run after train_hgcn_dim.py)")
    parser.add_argument("--dims", nargs="+", type=int, required=True)
    parser.add_argument("--force", action="store_true",
                        help="Re-export even if embedding file exists")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 Device: {device}")

    x_path  = os.path.join(DATA_DIR, "node_features_euclidean.pt")
    ei_path = os.path.join(DATA_DIR, "edge_index.pt")
    if not os.path.exists(x_path):
        print(f"❌ {x_path} not found"); return

    print(f"\n📥 Loading features...")
    x          = torch.load(x_path, map_location=device)
    edge_index = torch.load(ei_path, map_location=device)
    print(f"   x: {x.shape}")

    print(f"\n📐 Dims to export: {sorted(args.dims)}")
    print(f"   Saving to: {OUTPUT_DIR}/node_emb_d{{D}}.pt")
    print()

    for d in sorted(args.dims):
        export_one_dim(d, device, x, edge_index, force=args.force)

    print(f"\n{'='*60}")
    print(f"  ✅ Done. Next step:")
    print(f"  python src/system2/benchmark_dim.py \\")
    print(f"      --dims {' '.join(str(d) for d in sorted(args.dims))} --n 240")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
