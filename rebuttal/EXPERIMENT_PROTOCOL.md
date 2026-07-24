# FVxH rebuttal experiment protocol

This suite separates reproducible measurements from claims that the recovered
artifact cannot currently support.

## E0 — provenance gate

`python -m rebuttal.audit_reproducibility`

The audit hashes all geometry assets, verifies that MiniF2F-test contains exactly
244 official DeepSeek problems, and checks whether the recovered source implements
the paper's Algorithm 1. A result is labelled `paper-strict` only if all checks
pass. The current recovered artifact is expected to be labelled
`recovered-system`, because it uses A*, distance top-k retrieval, randomly
initialised Lie/tactic heads, and lacks a cone-filter implementation.

## E0b — reconstructed cone-direction gate (meta-review concern)

The corrected cone suite is specified in
`rebuttal/CORRECTED_CONE_PROTOCOL.md`. It is a new reconstruction, not the lost
original artifact. It corrects three independently testable issues:

- the angle is measured at the cone apex rather than at the origin;
- dependency positives are oriented premise-to-theorem;
- premise retrieval scores whether the query theorem lies in each candidate
  premise's cone, rather than whether candidates lie in the query cone.

The pre-registered four-arm comparison holds the reconstructed encoder and
proof-search harness fixed: distance, origin-angle forward, apex-angle forward,
and corrected inverse. First report held-out dependency retrieval; then report
verified proof success. This evidence may support the cone-direction
clarification, but cannot establish the unavailable trained Lie navigator.

## E1 — matched-compute MiniF2F frontier (reviewer Questions 2 and 6)

Hardware, model checkpoint, dataset order, and Lean/Mathlib commit are recorded in
machine-readable manifests.

- Native baseline: DeepSeek-Prover-V1.5-RL official CoT whole-proof prompt.
- Recovered HLP: current saved stepwise implementation, explicitly labelled as
  recovered and not Algorithm 1.
- Attempts: 1, 2, 4, 8, 16, 32.
- Dataset: official MiniF2F-test, exactly 244 problems.
- Sampling: temperature 1.0, top-p 0.95.
- Native completion limit: 2048 tokens.
- Recovered HLP limit: 64 search steps per trajectory.
- No success-only filtering; crashes and timeouts remain in the denominator.
- Every attempt is retained even after an earlier success, so Pass@k is
  comparable and is not an adaptive early-stopping estimate.
- Every `(problem, attempt)` uses a deterministic independent seed, so an
  interrupted run resumes without changing later samples.

Reported at each k:

- solved/244 and Wilson 95% interval;
- average cumulative wall-clock seconds per problem;
- LLM forward calls, prompt/completion tokens, Lean calls;
- peak CUDA allocation and independent `nvidia-smi` memory samples;
- raw proof code/tactics and verifier result per attempt.
- paired solved-set counts and exact McNemar tests.

## E2 — split and backbone correction (reviewer Questions 1 and 5)

Do not reuse Table 10. Any backbone lift must be recomputed on the identical
MiniF2F-test list with the same fast-solver and search policy. The current suite
only establishes the DeepSeek matched baseline; additional backbones should be
added after E1, not mixed across valid/test.

## E3 — depth > 10 / ProofNet (reviewer Questions 1 and 3)

The bundled official DeepSeek ProofNet Lean4 JSONL is runnable with
`cloud/run_proofnet_n1.sh` (official test split, 186 problems).
Five rows repeat theorem names; the runners now assign stable occurrence IDs so
all 186 rows remain distinct in resume, merge, and aggregation logic.
However, the recovered repository contains no executable Neural ODE/RGD variants.
A three-way SO(n,1)/ODE/RGD claim must remain blocked until implementations and
their trained checkpoints are supplied. Do not manufacture these variants from
the paper table.

When implementations become available:

1. define depth from a fixed reference proof before evaluating any method;
2. stratify as `<10` and `>=10`;
3. run all variants on the same examples/seeds/budget;
4. report paired solved-set differences and McNemar tests.

## E4 — qualitative trace (reviewer Question 5)

Successful recovered-HLP attempts are stored under
`results/rebuttal/hlp/traces/`. Select a theorem where native DeepSeek fails at
the same or greater measured time. Show every Lean goal, retrieved premise,
tactic, verifier response, LLM call count, and state coordinates. Never select
an example from a different split.
