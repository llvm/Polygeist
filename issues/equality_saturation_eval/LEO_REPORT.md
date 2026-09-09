# Equality saturation versus exact syntactic matching

Date: 2026-09-08/09 PDT

## Question

Does equality saturation enable backend-kernel matches that an otherwise
identical direct syntactic matcher misses, and what host-side matching cost
does it add?

## Experimental design

- Corpus: 687 already-raised, debufferized MLIR inputs: 598 ATen kernels,
  30 PolyBench programs, 20 MFEM kernels, 16 MFEM applications, 20 Llama
  forward operators, and 3 llama2.c operators.
- Arms:
  - `egglog`: capture binding plus equality checking after algebraic equality
    saturation.
  - `syntactic`: exact parsed-AST unification; no commutativity,
    reassociation, identity, factoring, or other algebraic rewrites.
- The handwritten semantic fallback was disabled in both arms. Structural,
  access-map, shape, and backend-legality checks common to both arms remained
  enabled.
- Five fresh processes per input per arm: 6,870 planned invocations. Arm order
  alternated by input and repetition.
- Runs were strictly sequential (`jobs=1`) and pinned to CPU 0. No parallel
  benchmark jobs were used.
- Measurements: fresh-process wall time, instrumented matcher time, aggregate
  equality-proof time, peak RSS, attempted proofs, rewrite matches, selected
  backend identity, covered Linalg bodies, and timeout status.

Egglog parameters:

- egglog 11.4.0.
- Eight saturation iterations per equality proof.
- Distributivity disabled for production matching.
- Input-expression ceiling: 32 AST nodes.
- Capture-binding proposal ceiling: 8.
- Nominal per-candidate limit: 10 seconds. This prototype records completed
  proofs exceeding the limit after they return; it cannot preempt an
  in-process Rust proof safely. A separate 120-second whole-input subprocess
  watchdog is enforced.
- No explicit e-graph node/class limit.
- No explicit memory limit.

Machine:

- Linux 5.15.0-151-generic, x86-64.
- Intel Xeon Cascadelake, 24 cores, one thread per core.
- 62 GiB RAM.
- Repository HEAD `1256b1da0b4d37377ccc8057fbfba2fd05c47d18` with a dirty
  working tree. The raw manifest stores a SHA-256 for every input; the exact
  experimental scripts are retained with the artifacts.

## Results

The coverage comparison excludes the one input for which Egglog never
completed, leaving 686 common-success inputs.

| Metric | Egglog | Exact syntax | Difference |
|---|---:|---:|---:|
| Selected match identities | 823 | 820 | +3 net |
| Covered Linalg bodies | 1,274 | 1,269 | +5 net |
| Arm-only identities | 5 | 2 | 5 Egglog-only, 2 syntax-only |

The two syntax-only identities are smaller scale operations that Egglog
replaces with larger two-body GEMM matches. The five Egglog-only identities
are:

- ATen `aten_addmm`, bodies `[0, 1]`: `cublasDgemm` (syntax selected only the
  body-0 `cublasDgeam_scale2D`).
- PolyBench `gemm`, bodies `[0, 1]`: `cublasDgemm` (syntax selected only the
  body-0 scale operation).
- PolyBench `2mm`, body `[1]`: `cublasDgemm_alpha_only`.
- PolyBench `gemver`, body `[1]`: `cublasDgemv_alpha`.
- PolyBench `gemver`, body `[3]`: `cublasDgemv_alpha`.

Thus equality saturation does not merely increase a counter: on `addmm` and
`gemm` it recognizes a larger fused backend kernel spanning two Linalg bodies
where exact syntax recognizes only a smaller one-body scale.

Across the 3,430 paired common-success repetitions:

| Metric | Egglog median [Q1, Q3] | Syntax median [Q1, Q3] | Paired Egglog - syntax median [Q1, Q3] |
|---|---:|---:|---:|
| Fresh-process wall time | 826.5 ms [738.3, 1,123.0] | 800.1 ms [722.6, 1,003.8] | +20.8 ms [-12.3, +93.4] |
| Matcher time | 365.1 ms [299.2, 561.9] | 331.8 ms [287.1, 462.0] | +16.8 ms [-4.3, +74.4] |
| Peak RSS | 66,570 KiB [65,520, 69,668] | 65,628 KiB [64,768, 65,896] | +288 KiB [-8, +4,159] |

The median hides important suite-dependent tails. Median paired matcher-time
overhead was +13.4 ms for ATen, +27.2 ms for PolyBench, +155.9 ms for
llama2.c, +235.6 ms for Llama forward, +718.4 ms for MFEM kernels, and
+2,981.6 ms for the 15 common-success MFEM applications.

There were 29,940 completed Egglog equality proofs. None of those completed
individual proofs exceeded 10 seconds. The largest aggregate proof time for a
successful whole input was 14.2 seconds across multiple proofs.

One very large MFEM application,
`mfem_app_navier_tgv_pa_operators_3d`, timed out in all five Egglog runs at the
120-second whole-input watchdog. All five exact-syntax runs completed in
97.1--105.1 seconds (median 100.2 seconds). Because a killed process cannot
flush per-proof telemetry, this experiment cannot determine whether a single
proof or the accumulation of many proofs caused those five timeouts.

## E-graph-size diagnostic

Graph serialization perturbs timing and RSS, so it was run separately for one
repetition on the four inputs containing the five Egglog-only identities.

| Input | Proofs | Maximum nodes | Maximum classes | Maximum proof time |
|---|---:|---:|---:|---:|
| PolyBench `2mm` | 16 | 21 | 12 | 29.7 ms |
| PolyBench `gemm` | 4 | 21 | 12 | 14.1 ms |
| PolyBench `gemver` | 8 | 25 | 15 | 15.4 ms |
| ATen `aten_addmm` | 4 | 21 | 12 | 38.9 ms |

These are diagnostic graph sizes for the benefit cases, not a corpus-wide
maximum. There is no configured e-graph-size cap.

## Interpretation and limitation

This experiment supports the narrow claim that algebraic equality saturation
finds useful larger backend matches that exact syntactic matching misses. The
gain is concentrated (five identities across four inputs), while typical
host-side overhead is small but MFEM applications have a substantial long
tail, including one reproducible whole-input timeout.

The timing is for the standalone matching/rewrite CLI over already-raised,
debufferized MLIR. It is not end-to-end time from C/C++ parsing through the
entire Polygeist compilation pipeline, and it should be labeled “matching
stage” or “host compilation/matching overhead” rather than total application
compilation time.

## Artifacts

- Primary raw runs: `campaign_20260908/runs.csv`
- Primary machine-readable summary: `campaign_20260908/summary.json`
- Primary generated summary: `campaign_20260908/SUMMARY.md`
- Resolved input paths and hashes: `campaign_20260908/manifest.resolved.json`
- E-graph diagnostic raw runs: `egraph_benefit_20260908/runs.csv`
- E-graph diagnostic summary: `egraph_benefit_20260908/SUMMARY.md`
- Canonical corpus manifest: `manifest.csv`
- Benefit-case manifest: `benefit_manifest.csv`
