# Corrected entailment-cone rebuttal protocol

## Status and claim boundary

This suite is a new implementation reconstructed from the mathematical
definition after the original server artifacts were lost. It is not the
original training run and must be reported as
`reconstructed_corrected_entailment_cones`. The proof-search arms use the
recovered A* stepwise harness; they do not recover or validate the paper's
unavailable trained Lie navigator.

This distinction is written into every training and inference manifest through
`original_training_artifact_recovered=false` and
`paper_claim_compatible=false`.

## Direction being tested

For a Poincare-ball point \(x\), the cone aperture is

\[
\psi(x)=\arcsin\left(K\frac{1-\lVert x\rVert^2}{\lVert x\rVert}\right).
\]

The angular term is the angle at the cone apex:

\[
\Xi(x,y)=\pi-\angle(O,x,y).
\]

The code implements the closed-form apex angle and uses

\[
E(x,y)=\max(0,\Xi(x,y)-\psi(x)).
\]

The raw Mathlib dependency relation says that theorem \(t\) uses premise
\(p\). Training therefore orients every positive edge as \(p\rightarrow t\)
and minimizes \(E(z_p,z_t)\). When retrieving a premise for a current theorem,
the logically consistent test is

\[
z_t\in C(z_p),\quad\text{so score candidate }p\text{ with }E(z_p,z_t).
\]

That is the `corrected_inverse` arm. “Inverse” refers to reversing the old
query-to-candidate lookup, not reversing logical entailment.

## Leakage and held-out evaluation

- Input graph: `data/proof_local_deps.json`. It contains 110,314 declaration
  nodes and 47,941 additional names that appear only as premises. The latter
  receive name-only text features, so all 206,746 recorded dependency
  references remain usable.
- Benchmark test names are excluded from the complete graph by exact or
  namespace-suffix match before graph construction.
- Dependency edges incident to excluded declarations are removed.
- Remaining premise/theorem pairs are assigned by SHA-256 to fixed
  train/valid/test partitions in a 90/5/5 ratio.
- Message passing uses only the training edges. It is bidirectional for
  structural aggregation, while the cone loss remains directional.
- The audit currently finds 0 matching declarations for all 244 MiniF2F-test
  rows and all 186 ProofNet-test rows.
- ProofNet has 186 rows but only 181 unique theorem names. Runners preserve all
  rows with stable `__occurrence_XX` problem IDs, while retaining the original
  theorem name separately.

Run the source/data audit locally:

```bash
python3 -m rebuttal.audit_corrected_cones
```

## Pre-registered four arms

1. `corrected_distance`: distance-only retrieval using the newly trained
   embeddings.
2. `paper_origin_forward`: origin-centered angular comparator with the query as
   apex, retained to isolate the disputed old formula.
3. `corrected_apex_forward`: correct apex angle but the old forward
   query-to-candidate direction.
4. `corrected_inverse`: correct apex angle and candidate-premise-to-query
   direction.

Before proof search, `diagnose_corrected_cones.py` evaluates the same four arms
on held-out dependency edges and reports recall@32, hit rate, MRR, cone
containment counts, directional energy, and radial ordering.

## Fresh four-A100 launch

From a fresh Linux server with four visible GPUs:

```bash
git clone git@github.com:BaSO6/Hyperbolic-Logic-Prover.git
cd Hyperbolic-Logic-Prover
bash cloud/launch_corrected_cone_rebuttal.sh
tail -f results/rebuttal/corrected_cone_run.log
```

The default detached pipeline:

1. installs the pinned environment and model assets;
2. trains/resumes the corrected cone model on GPU 0;
3. runs link-prediction diagnostics and the provenance audit;
4. runs a two-problem, four-GPU smoke test in a separate directory;
5. runs the four proof-search arms at \(N=1\), one arm per GPU;
6. strictly validates and aggregates only complete outputs.

To also run the two primary \(N\leq32\) frontiers in the same detached job:

```bash
RUN_NATIVE_FRONTIER=1 RUN_CORRECTED_INVERSE_N32=1 \
  bash cloud/launch_corrected_cone_rebuttal.sh
```

The official native DeepSeek frontier and corrected inverse-cone frontier are
each divided into four deterministic problem shards. Existing valid rows are
resumed; no force operation or success-only filtering is used.

Individual phases can also be run directly:

```bash
CUDA_VISIBLE_DEVICES=0 bash cloud/train_corrected_cones.sh
MAX_ATTEMPTS=1 bash cloud/run_cone_ablation_4gpu.sh
MAX_ATTEMPTS=32 bash cloud/run_corrected_inverse_4gpu.sh
RUN_NATIVE=1 RUN_HLP=0 MAX_ATTEMPTS=32 bash cloud/run_rebuttal_4gpu.sh
```

## Output and decision rule

- Training: `results/rebuttal/corrected_cone/`
- Link diagnostics:
  `results/rebuttal/corrected_cone/diagnostics/summary.json`
- Smoke test: `results/rebuttal/smoke/corrected_cone_arms/`
- Four-arm proof results: `results/rebuttal/cone_arms/`
- Corrected inverse \(N\leq32\): `results/rebuttal/corrected_inverse_n32/`
- Native \(N\leq32\): `results/rebuttal/native/`

Use the cone correction as positive evidence only if `corrected_inverse`
improves held-out premise retrieval and/or verified proof success over both
distance and the two forward controls. If it does not, report the negative
result and restrict the rebuttal to the conceptual clarification. Do not use
these runs to claim recovery of the original Lie policy.
