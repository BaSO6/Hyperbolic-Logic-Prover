# Hyperbolic-Logic-Prover

A hyperbolic geometry-based automated theorem prover for Lean 4, combining a **Hyperbolic Graph Convolutional Network (HGCN)** with LLM-guided proof search over Mathlib.

> **Reproducibility status (July 2026):** the recovered executable is an
> A*/stepwise system with hyperbolic distance retrieval.  It is not a faithful
> implementation of the paper's single-trajectory entailment-cone Algorithm 1,
> and the available checkpoints do not contain a trained Lie policy.  New runs
> are therefore labelled `recovered_hlp_astar_stepwise`; see
> [REBUTTAL_README.md](REBUTTAL_README.md) and the machine-readable audit before
> citing results.

---

## Current reproducible path (recommended)

The `cloud/` suite pins the official DeepSeek-Prover-V1.5 repository, Mathlib
commit, Lean toolchain, datasets, and Python inference stack.  On a fresh Linux
CUDA machine:

```bash
git clone git@github.com:BaSO6/Hyperbolic-Logic-Prover.git
cd Hyperbolic-Logic-Prover
bash cloud/bootstrap.sh
bash cloud/smoke_gpu.sh
MAX_ATTEMPTS=1 bash cloud/run_rebuttal_n32.sh   # full MiniF2F-test pilot
MAX_ATTEMPTS=32 bash cloud/run_rebuttal_n32.sh  # resume through N=32
```

On a single server with four visible GPUs, use:

```bash
MAX_ATTEMPTS=1 bash cloud/run_rebuttal_4gpu.sh
MAX_ATTEMPTS=32 bash cloud/run_rebuttal_4gpu.sh
```

This runs four deterministic problem shards (one process and one model replica
per GPU), then refuses to aggregate unless all shards have the exact expected
problem/attempt coverage and matching manifests. `bootstrap_and_run.sh`
automatically selects this path when it sees at least four GPUs; set
`FORCE_SINGLE_GPU=1` only when intentionally testing the one-GPU path.

For an SSH-disconnect-safe full run, use `bash cloud/launch_huawei.sh`. The
runners retain every attempt and record deterministic seeds, proofs, verifier
outcomes, wall-clock allocation, token/LLM/Lean call counts, VRAM, Wilson
intervals, paired solved sets, and exact McNemar tests.

macOS is supported for audit, aggregation, and unit tests, but full model
inference requires Linux/CUDA:

```bash
PYTHONPYCACHEPREFIX=/tmp/hlp-pycache \
  python3 -m unittest -v tests.test_rebuttal_common
python3 -m rebuttal.audit_reproducibility
bash -n cloud/*.sh src/system2/run_repl_wrapper.sh
```

The manual instructions below describe the legacy environment.  They are kept
for historical scripts; the pinned `cloud/` path above is the supported route
for new cross-device experiments.

---

## Table of Contents

- [Current reproducible path (recommended)](#current-reproducible-path-recommended)
- [System Requirements](#system-requirements)
- [Project Structure](#project-structure)
- [Step 1 — Install Lean 4](#step-1--install-lean-4)
- [Step 2 — Clone the Project](#step-2--clone-the-project)
- [Step 3 — Python Environment](#step-3--python-environment)
- [Step 4 — Download Datasets](#step-4--download-datasets)
- [Step 5 — Model Weights](#step-5--model-weights)
- [Step 6 — Build the REPL](#step-6--build-the-repl)
- [Step 7 — Build Mathlib](#step-7--build-mathlib)
- [Step 8 — Verify Everything](#step-8--verify-everything)
- [Running Benchmarks](#running-benchmarks)
- [Data Directory Reference](#data-directory-reference)
- [Troubleshooting](#troubleshooting)

---

## System Requirements

| Component | Requirement |
|---|---|
| OS | Linux x86_64 (Ubuntu 20.04+) |
| GPU | NVIDIA with ≥24GB VRAM (A100 80GB used in paper) |
| CUDA | 12.1 or 12.8 |
| RAM | ≥64GB |
| Disk | ≥200GB free (Mathlib build alone ~50GB) |
| Python | 3.10 |
| Lean | **4.10.0 exactly** — no other version works |

> **macOS / Windows:** Not supported. The REPL uses Linux PTY interfaces.
> Use a Linux server or cloud GPU instance (e.g. Lambda Labs, RunPod, A100 node).

---

## Project Structure

```
Hyperbolic-Logic-Prover/
├── src/
│   ├── system1/        # HGCN training (hyperbolic graph neural network)
│   ├── system2/        # Proof search, benchmarks, LLM engine
│   └── analysis/       # Embedding analysis and visualisation
├── data/               # Trained artifacts + dataset placeholders (see Step 4)
├── models/             # LLM weights — download separately (see Step 5)
├── tools/repl/         # Lean 4 REPL source — must be built (see Step 6)
├── scripts/            # setup_env.sh helper
├── requirements.txt    # Python dependencies
└── environment.yml     # Full conda environment spec
```

---

## Step 1 — Install Lean 4

Lean **4.10.0** is hard-pinned across the repo via `lean-toolchain` files.
Any other version will fail at REPL build or Mathlib load time.

```bash
# Install elan (Lean version manager)
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh

# Reload shell so elan is on PATH
source ~/.elan/env
# — or restart your terminal —

# Install and activate exactly 4.10.0
elan toolchain install leanprover/lean4:v4.10.0
elan default leanprover/lean4:v4.10.0
```

Verify before continuing:

```bash
lean --version    # Lean (version 4.10.0, x86_64-unknown-linux-gnu, ...)
lake --version    # Lake version 5.0.0-...
```

---

## Step 2 — Clone the Project

```bash
git clone git@github.com:BaSO6/Hyperbolic-Logic-Prover.git
cd Hyperbolic-Logic-Prover
```

> From this point on, **all commands assume you are in the project root**
> (`Hyperbolic-Logic-Prover/`) unless stated otherwise.

---

## Step 3 — Python Environment

### 3.1 Create conda environment

```bash
conda create -n hyp_logic_prover python=3.10 -y
conda activate hyp_logic_prover
```

### 3.2 Check your CUDA version

```bash
nvidia-smi | grep "CUDA Version"
```

### 3.3 Install PyTorch — match your CUDA version

**CUDA 12.1:**
```bash
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 12.8:**
```bash
pip install torch==2.10.0+cu128 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

### 3.4 Install PyTorch Geometric (PyG)

Use the same CUDA version tag as above. Example for CUDA 12.1 / torch 2.4.0:

```bash
pip install torch_geometric

pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

For CUDA 12.8 / torch 2.10.0, replace the URL with:
```
https://data.pyg.org/whl/torch-2.10.0+cu128.html
```

### 3.5 Install DGL

```bash
# CUDA 12.1
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html

# CUDA 12.8
pip install dgl -f https://data.dgl.ai/wheels/torch-2.10/cu128/repo.html
```

### 3.6 Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 3.7 Verify GPU is visible

```bash
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# Expected: True   NVIDIA A100-SXM4-80GB  (or your GPU)
```

---

## Step 4 — Download Datasets

The trained HGCN artifacts (`hgcn_final.pth`, embeddings, graph files) are
**already included** in the repo under `data/`. You only need to clone the
public benchmarks below.

> Run all `git clone` commands from the **project root**, not from inside `data/`.

### 4.1 Mathlib4 — required for all proof search (~5.6 GB)

```bash
git clone https://github.com/leanprover-community/mathlib4.git data/mathlib4
```

> ⚠️ Do **not** run `lake build` here yet. Build order matters: REPL first (Step 6), then Mathlib (Step 7).

### 4.2 miniF2F — required for `benchmark_minif2f.py`

```bash
git clone https://github.com/yangky11/miniF2F-lean4.git data/miniF2F
```

Confirm the structure after cloning:

```
data/miniF2F/
├── MiniF2F/
│   ├── Valid/        ← 244 validation problems
│   ├── Test/         ← 244 test problems
│   └── MiniF2F.lean
├── lakefile.lean
└── lean-toolchain
```

### 4.3 PutnamBench — required for `benchmark_putnam.py` (~376 MB)

```bash
git clone https://github.com/trishullab/PutnamBench.git data/PutnamBench-main
```

### 4.4 ProofNet — required for `benchmark_proofnet.py` (~64 MB)

```bash
git clone https://github.com/zhangir-azerbayev/ProofNet.git data/ProofNet-main
```

### 4.5 Large embedding files — optional, regeneratable

Not included due to size. Only needed if retraining System 1 (HGCN).

| File | Size | Regenerate with |
|---|---|---|
| `data/node_embeddings_variant_a.pt` | 108 MB | `python src/system1/train_variant_a.py` |
| `data/node_features_euclidean.pt` | 162 MB | `python src/system1/train_euclidean.py` |
| `data/node_embeddings_euclidean.pt` | 31 MB | `python src/system1/train_euclidean.py` |
| `data/raw_bert_embeddings.pt` | 162 MB | `python src/system1/gen_baseline_emb.py` |

---

## Step 5 — Model Weights

Weights go in the `models/` directory. The two marked **required** must be
downloaded before any benchmark will run.

```bash
pip install huggingface_hub
```

### Required

```bash
# Primary prover model
huggingface-cli download deepseek-ai/DeepSeek-Prover-V1.5-RL \
    --local-dir models/DeepSeek-Prover-V1.5-RL

# Sentence embeddings used by the HGCN retriever
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
    --local-dir models/all-MiniLM-L6-v2
```

### Optional alternative provers

```bash
# Lightweight option — only ~4 GB VRAM needed
huggingface-cli download AI-MO/Kimina-Prover-Preview-Distill-1.5B \
    --local-dir models/Kimina-Prover-Preview-Distill-1.5B

huggingface-cli download AI-MO/Kimina-Prover-Distill-8B \
    --local-dir models/Kimina-Prover-Distill-8B

huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --local-dir models/DeepSeek-R1-Distill-Llama-8B

huggingface-cli download internlm/InternLM2-StepProver \
    --local-dir models/InternLM2-StepProver

huggingface-cli download mistralai/Mathstral-7B-v0.1 \
    --local-dir models/Mathstral-7B-v0.1

huggingface-cli download Qwen/Qwen2.5-Math-7B-Instruct \
    --local-dir models/Qwen2.5-Math-7B-Instruct

huggingface-cli download Qwen/Qwen2.5-Math-72B-Instruct \
    --local-dir models/Qwen2.5-Math-72B-Instruct

huggingface-cli download AI-MO/NuminaMath-7B-TIR \
    --local-dir models/NuminaMath-7B-TIR

huggingface-cli download AI-MO/NuminaMath-72B-TIR \
    --local-dir models/NuminaMath-72B-TIR

# Llama 3.1 requires HuggingFace account approval at meta-llama/Llama-3.1-8B-Instruct
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir models/Llama-3.1-8B-Instruct
huggingface-cli download meta-llama/Llama-3.1-70B-Instruct \
    --local-dir models/Llama-3.1-70B-Instruct
```

---

## Step 6 — Build the Lean REPL

The REPL is the Python↔Lean bridge. It must be compiled before any benchmark.

```bash
cd tools/repl

# Remove stale artifacts from any previous attempt
rm -rf .lake/build

# Build (2–5 minutes)
lake build

# Confirm binary was created (~10 MB)
ls -lh .lake/build/bin/repl

# Return to project root
cd ../..
```

If `lake build` fails with a linker or `LD_LIBRARY_PATH` error, isolate the
environment:

```bash
cd tools/repl
env -i HOME=$HOME PATH=$HOME/.elan/bin:/usr/bin:/bin lake build
cd ../..
```

---

## Step 7 — Build Mathlib

Pre-compiling Mathlib means the REPL loads theorems in milliseconds instead
of recompiling on every proof attempt. **This step is mandatory.**

```bash
cd data/mathlib4

# First run takes 30–90 minutes depending on CPU cores
# Use -j N to parallelise (e.g. -j8 for 8 cores)
lake build

# Verify — should print the Lean version with no errors
lake env lean --version

cd ../..
```

---

## Step 8 — Verify Everything

Run from the project root. Every line should show ✅.

```bash
python3 << 'EOF'
import os, sys, torch, pickle, gzip
sys.path.insert(0, '.')

print("=== GPU ===")
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device         : {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM           : {vram:.1f} GB  {'✅' if vram >= 24 else '⚠️  <24 GB — large models may OOM'}")

print("\n=== Trained artifacts ===")
for fname in [
    "hgcn_final.pth", "hgcn_refined.pth",
    "node_embeddings.pt", "edge_index.pt",
    "id_to_name.pkl.gz", "node_list.pkl.gz",
    "node_head_symbols.pkl.gz",
]:
    path = os.path.join("data", fname)
    print(f"{'✅' if os.path.exists(path) else '❌ MISSING'}  data/{fname}")

print("\n=== Datasets ===")
for path, label, step in [
    ("data/mathlib4",          "Mathlib4",    "Step 4.1"),
    ("data/miniF2F/MiniF2F",  "miniF2F",     "Step 4.2"),
    ("data/PutnamBench-main",  "PutnamBench", "Step 4.3"),
    ("data/ProofNet-main",     "ProofNet",    "Step 4.4"),
]:
    ok = os.path.isdir(path)
    print(f"{'✅' if ok else f'❌ MISSING — see {step}'}  {label}")

print("\n=== Model weights ===")
for model, note in [
    ("DeepSeek-Prover-V1.5-RL", "required"),
    ("all-MiniLM-L6-v2",        "required"),
]:
    ok = os.path.isdir(os.path.join("models", model))
    print(f"{'✅' if ok else '❌ MISSING — see Step 5'}  models/{model}  ({note})")

print("\n=== REPL binary ===")
repl = "tools/repl/.lake/build/bin/repl"
print(f"{'✅' if os.path.exists(repl) else '❌ MISSING — run Step 6'}  {repl}")

print("\n=== Mathlib compiled ===")
olean = "data/mathlib4/.lake/build"
ok = os.path.isdir(olean) and len(os.listdir(olean)) > 0
print(f"{'✅' if ok else '❌ NOT BUILT — run Step 7'}  data/mathlib4/.lake/build/")

print("\n=== Python imports ===")
for mod in ["torch", "transformers", "torch_geometric", "geoopt", "dgl", "sentence_transformers"]:
    try:
        __import__(mod)
        print(f"✅  {mod}")
    except ImportError as e:
        print(f"❌  {mod}: {e}")

print("\n=== src imports ===")
for mod in ["src.system2.lie_search", "src.system2.lean_interaction", "src.system1.models_euclidean"]:
    try:
        __import__(mod)
        print(f"✅  {mod}")
    except Exception as e:
        print(f"❌  {mod}: {e}")
EOF
```

---

## Running Benchmarks

All benchmarks are run from the **project root**.

### Start here — smoke test (~5–15 min)

```bash
python smoke_test_minif2f.py
```

Runs a small end-to-end check (GPU + REPL + Mathlib + model). Do this before
committing to a full benchmark run.

### miniF2F — main benchmark (reproduces 65% Pass@1)

```bash
# Quick pilot on 5 problems — good first real test (~15 min)
python src/system2/benchmark_minif2f.py --pilot 5

# Full validation split (244 problems — ~4–8 hours on A100)
python src/system2/benchmark_minif2f.py

# Test split
python src/system2/benchmark_minif2f.py --split test

# All splits
python src/system2/benchmark_minif2f.py --split all

# Discard saved checkpoint and start fresh
python src/system2/benchmark_minif2f.py --fresh
```

Results are saved to `benchmark_reports_minif2f/` and auto-resume if interrupted.

### Other benchmarks

```bash
python src/system2/benchmark_putnam.py      # PutnamBench
python src/system2/benchmark_proofnet.py    # ProofNet
python src/system2/benchmark_amc_aime.py   # AMC / AIME
```

---

## Data Directory Reference

After completing all steps, `data/` should look like this:

```
data/
├── mathlib4/                     ← git clone  (Step 4.1, ~5.6 GB)
│   └── .lake/build/              ← compiled   (Step 7, ~50 GB)
├── miniF2F/                      ← git clone  (Step 4.2)
│   └── MiniF2F/
│       ├── Valid/                   244 validation problems
│       └── Test/                    244 test problems
├── PutnamBench-main/             ← git clone  (Step 4.3)
├── ProofNet-main/                ← git clone  (Step 4.4)
│
│   ─── Trained artifacts (included in repo) ───
├── hgcn_final.pth                trained HGCN checkpoint
├── hgcn_refined.pth              refined HGCN checkpoint
├── node_embeddings.pt            64-dim hyperbolic embeddings (110,314 nodes)
├── edge_index.pt                 Mathlib dependency graph edges
├── id_to_name.pkl.gz             node ID → Mathlib theorem name
├── node_list.pkl.gz              ordered node list
├── node_head_symbols.pkl.gz      head symbol index per node
├── node_text_map.pkl.gz          node ID → theorem statement text
├── mathlib_deep_graph.pkl.gz     full Mathlib dependency graph
├── proof_local_deps.json         local proof dependency map
├── putnam.json                   processed Putnam problems
├── proofnet.json                 processed ProofNet problems
├── debug_traces.jsonl            proof search traces
├── system1_feedback.json         System 1 training feedback
├── compfiles/                    competition Lean source files
└── visualizations/               embedding visualisations
```

---

## Troubleshooting

### `REPL binary not found`
You need to build it. From the project root:
```bash
cd tools/repl && lake build && ls .lake/build/bin/repl && cd ../..
```

### `RuntimeError: Lean attempted to build Mathlib`
Mathlib is not pre-compiled. Run `lake build` inside `data/mathlib4/` (Step 7).
Do not skip this step — it is required before any benchmark.

### `Could not parse JSON` / PTY errors during proof search
Already handled in `lean_interaction.py` v16 by chunking PTY writes to 2048 bytes.
If still occurring, ensure you are using the REPL binary built from `tools/repl/`
in this repo, not a system-installed version.

### `torch_geometric` warnings about `pyg-lib`, `torch-scatter`
Harmless on some cloud platforms — these are runtime CUDA library path issues,
not import failures. All affected operations fall back to pure PyTorch automatically.

### GPU out of memory
- `llm_engine.py` auto-enables 4-bit quantisation when the model exceeds 70% of VRAM.
- Reduce `NUM_WORKERS` at the top of `benchmark_minif2f.py` (default: 4 → try 1 or 2).
- Switch to a smaller model: `Kimina-Prover-Preview-Distill-1.5B` needs only ~4 GB VRAM.

### `elan: command not found`
```bash
source ~/.elan/env
echo 'source ~/.elan/env' >> ~/.bashrc   # persist across sessions
```

### Wrong Lean version
```bash
elan toolchain install leanprover/lean4:v4.10.0
elan override set leanprover/lean4:v4.10.0
lean --version   # must now show 4.10.0
```

### `lake build` in `data/mathlib4` fails with `unknown package`
Make sure you cloned with exactly `git clone ... data/mathlib4` and that
`data/mathlib4/lean-toolchain` contains `leanprover/lean4:v4.10.0`.

### miniF2F benchmark reports `0 lean files found`
Check that `data/miniF2F/MiniF2F/Valid/` and `Test/` directories exist.
The benchmark intentionally excludes `_manual_mathlib/`.

---

## Citation

```bibtex
@misc{hyperbolic-logic-prover-2025,
  title  = {Hyperbolic Lie Prover: Hierarchical Reasoning for Logical Inference via Lie Group Dynamics},
  author = {Beibei Liu},
  year   = {2026},
  url    = {https://github.com/BaSO6/Hyperbolic-Logic-Prover}
}
```
