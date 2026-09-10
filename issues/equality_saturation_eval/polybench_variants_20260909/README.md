# Controlled PolyBench IR-variation experiment

Status: matcher experiment complete; extracted-kernel CPU execution pilot passes.

## Question

When a raised PolyBench library candidate is expressed in a different but
algebraically equivalent scalar form, does equality saturation retain the
library identity more often than the exact-syntax ablation?

## Inputs and transformations

- 30 tracked LARGE FP64 PolyBench `raised_debufferized.mlir` inputs.
- Ten independent, name-agnostic transformations were offered: swap add
  operands, swap multiply operands, swap both, add zero on either side,
  multiply by one on either side, one composed identity stress case, and
  left-to-right reassociation of add or multiply at one deterministic site.
- 245 variants were applicable; 55 kernel/transformation pairs were explicitly
  not applicable. All 245 generated variants passed `polygeist-opt` parsing and
  verification.
- The primary campaign excludes the composed `(x + 0) * 1` stress case after
  its first 2mm Egglog run hit the 120-second whole-input watchdog. It contains
  30 originals plus 215 variants.

## Protocol

Each primary input ran in ten fresh processes: five Egglog and five exact-
syntax invocations, alternating arm order, sequentially on CPU 0. Semantic
fallback was disabled. The candidate limit remained 10 seconds (post-run
classification) and the whole-input watchdog was 120 seconds. Match sets were
deterministic across repetitions. E-graph serialization was disabled during
the timed campaign and collected in a separate one-repetition diagnostic for
the 63 Egglog-benefit variants.

Host provenance:

- Polygeist commit before experiment changes:
  `f37f06fbb8040ac158449c0e1521142d32fc8726`
- CPU: Intel Xeon Processor (Cascade Lake), 24 physical cores; CPU 0 used.
- Python 3.10.12; egglog 11.4.0.
- Clang/LLVM 18.0.0, LLVM revision
  `26eb4285b56edd8c897642078d91f16ff0fd3472`.

## Exact commands

```sh
python3 scripts/correctness/generate_eqsat_ir_variants.py \
  --manifest issues/equality_saturation_eval/polybench_variants_manifest.csv \
  --output issues/equality_saturation_eval/polybench_variants_20260909

/usr/bin/python3 scripts/correctness/run_eqsat_ablation.py \
  --manifest issues/equality_saturation_eval/polybench_variants_20260909/manifest.csv \
  --output issues/equality_saturation_eval/polybench_variants_20260909/campaign_main \
  --repetitions 5 --input-timeout 120 --cpu 0 \
  --exclude-input-regex '::identity_pair$'

/usr/bin/python3 scripts/correctness/run_eqsat_ablation.py \
  --manifest issues/equality_saturation_eval/polybench_variants_20260909/analysis/benefit_manifest.csv \
  --output issues/equality_saturation_eval/polybench_variants_20260909/egraph_diagnostic \
  --repetitions 1 --input-timeout 120 --cpu 0 --collect-egraph-sizes

python3 scripts/correctness/run_eqsat_variant_cpu_correctness.py \
  --analysis issues/equality_saturation_eval/polybench_variants_20260909/analysis/per_input_variant.csv \
  --variants issues/equality_saturation_eval/polybench_variants_20260909 \
  --output issues/equality_saturation_eval/polybench_variants_20260909/cpu_execution_extracted \
  --limit 10

python3 scripts/correctness/summarize_eqsat_ir_variants.py \
  --runs issues/equality_saturation_eval/polybench_variants_20260909/campaign_main/runs.csv \
  --generation issues/equality_saturation_eval/polybench_variants_20260909/generation.csv \
  --stress-runs issues/equality_saturation_eval/polybench_variants_20260909/campaign/runs.csv \
  --manifest issues/equality_saturation_eval/polybench_variants_20260909/manifest.csv \
  --cpu-summary issues/equality_saturation_eval/polybench_variants_20260909/cpu_execution_extracted/summary.json \
  --egraph-runs issues/equality_saturation_eval/polybench_variants_20260909/egraph_diagnostic/runs.csv \
  --output issues/equality_saturation_eval/polybench_variants_20260909/analysis
```

## Results

- Primary matcher processes: 2,450/2,450 successful, zero timeouts.
- Common-baseline target (identities both arms found before mutation): Egglog
  retained 207/221 (93.7%); exact syntax retained 98/221 (44.3%).
- Broader unmodified-Egglog target: Egglog retained 241/255 (94.5%); exact
  syntax retained 101/255 (39.6%).
- Paired Egglog-minus-syntax matcher cost: 15.018 ms median, with quartiles
  -6.639 and 68.924 ms over 1,225 paired repetitions.
- Paired peak-RSS delta: 316 KiB (0.309 MiB) median.
- Benefit diagnostic: largest individual proof contained 42 nodes and 17
  equivalence classes; the median per-input largest proof contained 18 nodes
  and 9 classes. No completed proof exceeded 10 seconds.
- The 14 missed recoveries are four Gesummv `cublasDaxpby` compositions, five
  SYR2K compositions, and five SYRK compositions. Their recognizers retain
  structural preconditions outside scalar equality saturation.

## CPU execution validation

Ten LARGE FP64 variants passed complete host execution and numerical comparison:
seven `2mm` variants and three `3mm` variants. Each build linked the extracted,
Egglog-matched kernel to a generated source-faithful PolyBench harness which
retains allocation, initialization, invocation, and dumping but only declares
the selected kernel. Symbol inspection confirms the harness leaves the kernel
undefined, so the transformed object is the sole definition and no weak-symbol
replacement or interposition is used. OpenBLAS/CBLAS ran with one thread; all
ten output comparisons passed at rtol 5e-4 and atol 1.1e-2. Commands, hashes,
outputs, and logs are retained under `cpu_execution_extracted/`.

The earlier whole-program ABI failure remains retained under
`cpu_execution_pilot_v2/` as a separate pipeline limitation; it is no longer a
blocker for validating extracted variants. Earlier invalid same-source harness
attempts remain under `cpu_correctness/` and are rejected by the project's
no-symbol-substitution guard.

The concise analysis is in `analysis/REPORT.md`; normalized results are in
`analysis/by_variant.csv` and `analysis/per_input_variant.csv`; every fresh
process is retained in `campaign_main/runs.csv`.
