# MFEM silicon table regeneration — 2026-08-27

Full re-timing on Jetson AGX Orin (`nvidia@192.168.58.1` via `pva-general`),
SM87 MAXN, CUDA 12.6, f64, `NE=1024`, `D1D=4`, `Q1D=5`. Reused surviving
aarch64 binaries via `run_jetson.sh --exe`, 5 processes each, median of runs
2–4. cuTensorNet libs from `/home/nvidia/polygeist_cuda_libs`.

## Kernel table (native_vs_raised_large_ne.csv) — FULLY REGENERATED

All 18 rows re-timed and updated. Every binary reproduced its recorded
checksum exactly (correctness gate). Binaries: PA = `/tmp/raised_*`
(+`_cache` for the plan-cache column), DFEM = `/tmp/mfem_dfem8_*`.
`raised_over_native` recomputed against the carried-over native baselines.
Values track the prior record within run-to-run noise (e.g. curlcurl_3d 5.583,
diffusion_3d 150.9, mass_3d 84.8). `.pre20260827.bak` holds the prior file.

## Application table (application_native_vs_raised_large_ne.csv) — PARTIAL

Faithfully regenerated (graph-optimized binaries, the "speedup technique" rows):

- `abs_l1_mass_3d`:      548.57 us  (graph-on composed cuTensorNet; recorded 658.8 non-graph / 549.4 graph). 3.976x slower than native.
- `abs_l1_diffusion_3d`: 14645.50 us (graph-on complete-residual CUDA graph; recorded 14628.30). 48.160x slower than native.
- `dfem_minimal_surface_2d`: still FAILs correctness (max_abs 0.20337545654012545, matches record). No timing.

NOT overwritten — surviving uniform build `/tmp/mfem_app_ne1024/*` diverges
from the recorded curated results and its provenance can't be confirmed to
match the original methodology (several recorded rows are COMPONENT_SUM against
native, not single-binary raised). Today's uniform-build re-measures (median
runs 2–4), for the record, vs recorded raised_runtime_us:

| row | uniform-build 2026-08-27 | recorded | note |
|---|---:|---:|---|
| mtop_iso_elasticity_dfem_2d | 19019 | 22692 | SAFE_PAIRWISE |
| ex35p_h1_3d | 57460 | 54797 | ~matches |
| ex35p_hcurl_3d | 485594 | 300324 | diverges (COMPONENT_SUM) |
| ex35p_hdiv_3d | 278444 | 190913 | diverges (COMPONENT_SUM) |
| ex9p_mass_convection_2d | 8567 | 10927 | diverges (PARTIAL_COMPONENT_SUM) |
| grad_div_3d | 246914 | 191170 | diverges (COMPONENT_SUM) |
| abs_l1_curlcurl_3d | 500846 | 300721 | diverges (COMPONENT_SUM) |
| navier_tgv_pa_operators_3d | 1084303 | 973032 | SAFE_PAIRWISE |

These 8 rows retain their recorded values pending identification of the exact
original per-row binaries. All uniform-build runs passed correctness
(roundoff-level max_abs); minimal_surface FAILs as recorded.

Raw sweep log: `/tmp/mfem_regen_20260827/sweep_results.txt`.
Silicon logs under `scripts/correctness/logs/regen_*20260827*.silicon.log`.
