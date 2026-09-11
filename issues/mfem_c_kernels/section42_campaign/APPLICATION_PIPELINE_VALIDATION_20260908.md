# MFEM-derived application pipeline validation — 2026-09-08

## Result

All 11 MFEM-derived application/operator hot paths were freshly regenerated at
`NE=1024`, lowered to both AArch64 CPU and SM87 GPU executables, and executed on
Orin #2 (`nvidia@192.168.57.1` through `pva-general`). All 11 pass complete
elementwise comparison with their direct extracted-C references on both raised
targets.

These are manually extracted numerical hot paths. They exclude MFEM mesh I/O,
MPI, global restriction/prolongation, and most solver control, so they are not
untouched full MFEM application runs.

- raised GPU: 11/11 built; 55/55 correctness-gated processes passed
- raised CPU: 11/11 built; 55/55 correctness-gated processes passed
- retained timings: 4,400 samples (11 paths × 2 campaign binaries × 5
  processes × 2 implementations × 20 samples)
- structural matches: 284 `kernel.launch` sites across the 11 paths
- lowered calls: 2,094 GPU-runtime calls and 698 CPU-runtime calls
- largest observed complete-output absolute error: `8.326672684688674e-17`
- median raised-CPU slowdown versus vanilla CPU: `290.32x` (range
  `28.75x`–`687.39x`)
- median raised-GPU slowdown versus vanilla CPU: `15.48x` (range
  `1.29x`–`23.21x`)

No current native-MFEM application baseline was run in this campaign. The
raised-GPU/vanilla-CPU ratios are useful pipeline measurements but are not a
replacement for native-MFEM GPU comparisons.

## Per-path medians

Times are milliseconds. Each value is the median of five independent process
medians. Every process performs five untimed warmups and retains 20 samples.

| path | matches | vanilla CPU | raised CPU | raised GPU | CPU/vanilla | GPU/vanilla |
|---|---:|---:|---:|---:|---:|---:|
| mtop elasticity 2D | 16 | 1.079296 | 380.114144 | 25.045120 | 352.19x | 23.21x |
| DFEM minimal surface 2D | 8 | 0.648688 | 188.326912 | 10.042896 | 290.32x | 15.48x |
| ex35p H1 3D | 19 | 4.162720 | 2050.021488 | 75.531904 | 492.47x | 18.14x |
| ex35p H(curl) 3D | 29 | 34.450144 | 2609.823472 | 211.359808 | 75.76x | 6.14x |
| ex35p H(div) 3D | 12 | 23.511040 | 966.635408 | 102.595568 | 41.11x | 4.36x |
| ex9p mass + convection 2D | 10 | 0.425488 | 194.688208 | 9.841968 | 457.56x | 23.13x |
| grad-div 3D | 12 | 23.521920 | 985.795248 | 102.101280 | 41.91x | 4.34x |
| abs-L1 mass 3D | 5 | 0.878912 | 604.157424 | 15.000240 | 687.39x | 17.07x |
| abs-L1 diffusion 3D | 14 | 3.363408 | 1443.978400 | 66.004752 | 429.32x | 19.62x |
| abs-L1 curl-curl 3D | 29 | 34.587952 | 2708.793120 | 212.937744 | 78.32x | 6.16x |
| Navier TGV PA operators 3D | 130 | 489.937504 | 14085.434912 | 630.432288 | 28.75x | 1.29x |

The complete machine-readable table is
`application_performance_20260908.csv`.

## Correctness protocol

The harness initializes nonconstant deterministic FP64 input and nonzero output
buffers. It runs the direct extracted-C reference and the transformed function,
then compares every output element with relative tolerance `1e-10`. It also
requires the reference output to differ nontrivially from its initial state;
this prevents two no-op paths from passing by equality.

The campaign uncovered and corrected an invalid historical timing path for
`abs_l1_mass_3d`: the old harness's session helper invoked a normally compiled
source helper rather than the transformed entry. The new harness invokes the
compiler-produced function directly. The discarded 1.6 microsecond sanity
timing is not present in the result ledger.

The Navier raised-CPU row required a resumed run because the first collection
was interrupted after two complete processes. Those two complete process logs
and three resumed process logs are combined; the interrupted partial third
process contributes no samples. A separate correctness-only Navier execution
also passed with maximum absolute error `8.326672684688674e-17`.

## Compiler and artifacts

- campaign source base at build start:
  `7b6437be4b98a785b76cd5f783fe33ec0e3b86dc`
- `cgeist` SHA-256:
  `d272de5e97d5347b57e0a392d064e1d1df83ca9447d1c296339d253cc9ba518b`
- `polygeist-opt` SHA-256:
  `77d7c31bf1ed1e207104b2a5865b0703e91f70e4183d125b42b900cbf2844789`
- matcher SHA-256:
  `2a1093d23a256bc4c303e8ecaf8cad88c4ddcec747d77b09dd5d5584f097fcf8`
- CUDA runtime source SHA-256:
  `7f9da424d8fdc86995b8b2e758d2d28ccbdfb7f2d44826e41e91ec28031a4bfa`
- pairwise cuTensorNet lowering was used; unvalidated Mass3D network
  composition and persistent-workspace lowering were disabled
- binaries, build intermediates, metadata, and raw logs:
  `/home/arjaiswal/mfem-derived-app-campaign-20260908/current-ne1024/`
- compact retained results archive (metadata, binaries, build logs, and raw
  run logs; build intermediates excluded):
  `/home/arjaiswal/mfem-derived-app-campaign-20260908/mfem-derived-app-ne1024-20260908-results.tar.gz`
- retained archive SHA-256:
  `70a26ae49269400559cfbe01f7742bb22cdc46111c7331e070b4e7c7693a5c32`

The source tree was dirty and advanced concurrently after the recorded build
base. Binary/tool hashes therefore define the exact evaluated compiler state;
a clean committed rerun is still required before freezing final paper numbers.
