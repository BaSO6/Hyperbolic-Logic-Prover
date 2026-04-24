# ==============================================================================
# Filename: src/system1/train_hgcn_curvature.py
# Version: v1.0
#
# Purpose: Train HGCN variants with different Poincaré ball curvatures
#          c ∈ {0.5, 1.0, 2.0} for the curvature sensitivity ablation.
#
#          Saves to results/curvature_sensitivity/:
#            hgcn_c{C}.pth        — checkpoint with curvature c stored
#            node_emb_c{C}.pt     — node embeddings computed with curvature c
#
#          NEVER overwrites hgcn_final.pth or node_embeddings.pt.
#          c=1.0 reuses the paper checkpoint and embeddings (no retraining).
#
# Correct System 1 / System 2 separation:
#   This script (system1): train HGCN, export embeddings, validate
#   benchmark_curvature.py (system2): load assets, run Lean benchmark
#
# Usage:
#   # Train c=0.5 and c=2.0 (c=1.0 is reused from paper):
#   python src/system1/train_hgcn_curvature.py --curvatures 0.5 1.0 2.0
#
#   # Validate after training:
#   python src/system1/train_hgcn_curvature.py --validate-only
#
#   # Force retrain even if checkpoint exists:
#   python src/system1/train_hgcn_curvature.py --curvatures 0.5 2.0 --force
# ==============================================================================

import os
import sys
import argparse
import shutil
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.system1.manifold_math import PoincareBall  # type: ignore

DATA_DIR   = os.path.join(project_root, "data")
OUTPUT_DIR = os.path.join(project_root, "results", "curvature_sensitivity")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Hyperparameters — identical to train_final.py / train_hgcn_dim.py
HIDDEN_DIM = 256
OUT_DIM    = 64
EPOCHS     = 201
LR         = 0.005


# ==============================================================================
# Architecture — must be identical to train_final.py v12.1
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
        self.manifold       = PoincareBall(c)
        self.semantic_proj  = nn.Linear(in_dim, out_dim)
        self.structure_proj = nn.Linear(in_dim, out_dim)
        self.graph_conv     = EuclideanGraphConv(out_dim, out_dim)
        self.gate           = nn.Linear(out_dim * 2, 1)
        self.scale          = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, edge_index):
        z_sem    = self.semantic_proj(x)
        z_struct = F.relu(self.structure_proj(x))
        z_struct = self.graph_conv(z_struct, edge_index)
        alpha    = torch.sigmoid(self.gate(torch.cat([z_sem, z_struct], dim=-1)))
        z_tan    = alpha * z_sem + (1 - alpha) * z_struct
        x_norm   = z_tan.norm(dim=-1, keepdim=True) + 1e-8
        radius   = 0.9 * torch.tanh(self.scale)
        return self.manifold.expmap0(z_tan / x_norm * radius)


class FinalHGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, c=1.0):
        super().__init__()
        self.layer    = HyperbolicResidualLayer(in_dim, out_dim, c)
        self.manifold = self.layer.manifold
        self.out_dim  = out_dim
        self.c        = c

    def forward(self, x, edge_index):
        return self.layer(x, edge_index)


# ==============================================================================
# Helpers
# ==============================================================================

def c_str(c: float) -> str:
    """Convert curvature float to safe filename string: 1.0 → '1p0', 0.5 → '0p5'"""
    return f"{c:.1f}".replace(".", "p")


def ckpt_path(c: float) -> str:
    return os.path.join(OUTPUT_DIR, f"hgcn_c{c_str(c)}.pth")


def emb_path(c: float) -> str:
    return os.path.join(OUTPUT_DIR, f"node_emb_c{c_str(c)}.pt")


# ==============================================================================
# Step 1: Training
# ==============================================================================

def train_one(c: float, device: torch.device,
              x: torch.Tensor, edge_index: torch.Tensor,
              force: bool = False) -> bool:
    """Train HGCN with curvature c. Returns True if trained/ready, False on error."""

    paper_ckpt = os.path.join(DATA_DIR, "hgcn_final.pth")

    # c=1.0: reuse paper checkpoint, no retraining needed
    if abs(c - 1.0) < 1e-6:
        if not os.path.exists(paper_ckpt):
            print(f"  ❌ c=1.0: paper checkpoint not found: {paper_ckpt}")
            return False
        ckpt = torch.load(paper_ckpt, map_location="cpu")
        stored_c = ckpt.get("c", 1.0)
        if abs(stored_c - 1.0) > 0.01:
            print(f"  ⚠️  Paper checkpoint has c={stored_c}, expected 1.0")
        dest = ckpt_path(c)
        if not os.path.exists(dest):
            shutil.copy2(paper_ckpt, dest)
            print(f"  ↩️  c=1.0: copied paper checkpoint → {dest}")
        else:
            print(f"  ↩️  c=1.0: already exists at {dest}")
        return True

    # Other curvatures
    dest = ckpt_path(c)
    if os.path.exists(dest) and not force:
        print(f"  ↩️  c={c}: checkpoint exists (use --force to retrain)")
        return True

    print(f"\n{'─'*60}")
    print(f"  🚀 Training HGCN  c={c}  OUT_DIM={OUT_DIM}  epochs={EPOCHS}")
    print(f"{'─'*60}")

    model     = FinalHGCN(x.shape[1], HIDDEN_DIM, OUT_DIM, c).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        z = model(x, edge_index)

        norms    = z.norm(dim=-1)
        loss_reg = ((norms - 0.8) ** 2).mean()
        loss_link = torch.tensor(0.0, device=device)
        if edge_index.size(1) > 100:
            perm     = torch.randperm(edge_index.size(1), device=device)[:10_000]
            u, v     = edge_index[:, perm]
            pos_dist = model.manifold.dist(z[u], z[v])
            neg_v    = torch.randint(0, z.size(0), (len(u),), device=device)
            neg_dist = model.manifold.dist(z[u], z[neg_v])
            loss_link = F.relu(pos_dist - neg_dist + 1.0).mean()

        loss = loss_link + 0.1 * loss_reg
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"  Ep {epoch:03d} | loss={loss.item():.4f}  "
                  f"link={loss_link.item():.4f}  reg={loss_reg.item():.4f}")

    torch.save({
        "model":      model.state_dict(),
        "c":          c,           # ← curvature stored explicitly
        "out_dim":    OUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "note":       f"curvature_sensitivity_c{c}",
    }, dest)
    print(f"  ✅ Saved → {dest}")
    return True


# ==============================================================================
# Step 2: Export embeddings (separate from training — no OOM risk)
# ==============================================================================

def export_embeddings(c: float, device: torch.device,
                      x: torch.Tensor, edge_index: torch.Tensor,
                      force: bool = False) -> bool:
    """Export full-graph embeddings for curvature c. Always uses full edge_index."""

    paper_emb = os.path.join(DATA_DIR, "node_embeddings.pt")
    dest      = emb_path(c)

    # c=1.0: copy paper embeddings
    if abs(c - 1.0) < 1e-6:
        if not os.path.exists(paper_emb):
            print(f"  ❌ c=1.0: paper embeddings not found: {paper_emb}")
            return False
        if not os.path.exists(dest) or force:
            shutil.copy2(paper_emb, dest)
            print(f"  ↩️  c=1.0: copied paper embeddings → {dest}")
        else:
            print(f"  ↩️  c=1.0: already exists at {dest}")
        return True

    src_ckpt = ckpt_path(c)
    if not os.path.exists(src_ckpt):
        print(f"  ❌ c={c}: checkpoint not found. Run training first.")
        return False
    if os.path.exists(dest) and not force:
        print(f"  ↩️  c={c}: embeddings exist (use --force to re-export)")
        return True

    print(f"  📤 Exporting embeddings c={c}  ({x.shape[0]} nodes)...")
    ckpt  = torch.load(src_ckpt, map_location=device)

    # Load with the SAME curvature used during training
    stored_c = ckpt.get("c", c)
    if abs(stored_c - c) > 0.01:
        print(f"  ⚠️  Checkpoint has c={stored_c}, expected c={c}. Using stored.")
        c = stored_c

    model = FinalHGCN(x.shape[1], HIDDEN_DIM, OUT_DIM, c).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        z = model(x, edge_index)

    torch.save(z.cpu(), dest)
    print(f"  ✅ Saved embeddings {z.shape} → {dest}")
    return True


# ==============================================================================
# Step 3: Validation
# ==============================================================================

def validate_all(curvatures: list) -> bool:
    """Validate that all trained artifacts are self-consistent."""
    all_pass = True
    print(f"\n{'='*60}")
    print(f"  VALIDATION")
    print(f"{'='*60}")

    for c in curvatures:
        cp = ckpt_path(c)
        ep = emb_path(c)
        ok = True

        # Check files exist
        if not os.path.exists(cp):
            print(f"  ❌ c={c}: checkpoint missing: {cp}"); ok = False
        if not os.path.exists(ep):
            print(f"  ❌ c={c}: embeddings missing: {ep}"); ok = False
        if not ok:
            all_pass = False; continue

        # Validate checkpoint
        ckpt = torch.load(cp, map_location="cpu")
        stored_c = ckpt.get("c", None)
        w = ckpt["model"].get("layer.semantic_proj.weight",
                               ckpt["model"].get("semantic_proj.weight"))

        c_ok     = stored_c is not None and abs(float(stored_c) - c) < 0.01
        shape_ok = w is not None and tuple(w.shape) == (OUT_DIM, 384)
        nan_ok   = w is not None and not torch.isnan(w).any()

        # Validate embeddings
        emb = torch.load(ep, map_location="cpu")
        shape_emb_ok  = emb.shape[1] == OUT_DIM and emb.shape[0] > 100_000
        norm_ok       = emb.norm(dim=-1).max().item() < 1.0
        finite_ok     = torch.isfinite(emb).all().item()
        std_ok        = emb.std().item() > 0.01

        passes = [c_ok, shape_ok, nan_ok, shape_emb_ok, norm_ok, finite_ok, std_ok]
        labels = ["c stored", "weight shape", "no NaN", "emb shape",
                  "Poincaré ball", "finite", "diversity"]

        n_pass = sum(passes)
        status = "✅" if all(passes) else "❌"
        print(f"\n  {status} c={c}  ({n_pass}/{len(passes)} checks)")
        for lbl, p in zip(labels, passes):
            print(f"    {'✅' if p else '❌'} {lbl}")
        if not all(passes):
            all_pass = False

    print(f"\n{'='*60}")
    if all_pass:
        print(f"  ✅ ALL VALID — safe to run benchmark_curvature.py")
    else:
        print(f"  ❌ FAILURES DETECTED — fix before benchmarking")
    print(f"{'='*60}")
    return all_pass


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train HGCN at different curvatures for sensitivity ablation")
    parser.add_argument("--curvatures", nargs="+", type=float,
                        default=[0.5, 1.0, 2.0],
                        help="Curvature values to train")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if checkpoint exists")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip training, just validate existing artifacts")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"   GPU: {props.name}  {props.total_memory/1e9:.0f}GB")

    print(f"\n📐 Curvatures: {sorted(args.curvatures)}")
    print(f"   OUT_DIM={OUT_DIM}  HIDDEN_DIM={HIDDEN_DIM}  EPOCHS={args.epochs}")
    print(f"   Saving to: {OUTPUT_DIR}")
    print(f"   Paper c=1.0 will be reused from hgcn_final.pth (no retraining)")

    if args.validate_only:
        validate_all(sorted(args.curvatures))
        return

    # Load features once — shared across all curvature runs
    feat_path = os.path.join(DATA_DIR, "node_features_euclidean.pt")
    ei_path   = os.path.join(DATA_DIR, "edge_index.pt")
    if not os.path.exists(feat_path):
        print(f"❌ {feat_path} not found"); return
    if not os.path.exists(ei_path):
        print(f"❌ {ei_path} not found"); return

    print(f"\n📥 Loading features (shared across all curvature runs)...")
    x          = torch.load(feat_path,  map_location=device)
    edge_index = torch.load(ei_path, map_location=device)
    print(f"   x: {x.shape}  edge_index: {edge_index.shape}")

    # Step 1: Train all curvatures
    print(f"\n{'='*60}")
    print(f"  STEP 1: TRAINING")
    print(f"{'='*60}")
    for c in sorted(args.curvatures):
        success = train_one(c, device, x, edge_index, force=args.force)
        if not success:
            print(f"  ❌ Training failed for c={c}, stopping.")
            return

    # Step 2: Export embeddings (after all training, so GPU memory is freed)
    print(f"\n{'='*60}")
    print(f"  STEP 2: EXPORTING EMBEDDINGS")
    print(f"{'='*60}")

    # Free training memory before export
    if device.type == "cuda":
        torch.cuda.empty_cache()

    for c in sorted(args.curvatures):
        success = export_embeddings(c, device, x, edge_index, force=args.force)
        if not success:
            print(f"  ❌ Export failed for c={c}")
            return

    # Step 3: Validate everything
    validate_all(sorted(args.curvatures))

    print(f"\n{'='*60}")
    print(f"  NEXT STEP:")
    print(f"  cp benchmark_curvature.py src/system2/")
    print(f"  python src/system2/benchmark_curvature.py \\")
    cvs = " ".join(str(c) for c in sorted(args.curvatures))
    print(f"      --curvatures {cvs} --n 240")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
