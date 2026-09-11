# MFEM abs_l1_mass_3d — CUDA-graph speedup regeneration (2026-08-27)

Re-timing of the composed-cuTensorNet + CUDA-graph launch-transition speedup on
Jetson AGX Orin (`nvidia@192.168.58.1` via `pva-general`), SM87, CUDA 12.6.
Reused the surviving Aug-19 aarch64 binary
`/tmp/mfem_abs_mass_graph_20260819/abs_mass_graph_new_wrapped`; graph replay is
toggled at runtime with `POLYGEIST_CUDA_GRAPH`. f64, `NE=1024`, `D1D=4`,
`Q1D=5`, 20 warm iterations, 5 independent processes.

All 10 processes PASS, max abs/rel error `6.9388939039072284e-18`.

| config | raised_gpu_us per process | median (us) | speedup vs CPU ref |
|---|---|---:|---:|
| graph-disabled (`POLYGEIST_CUDA_GRAPH=0`) | 657.30, 651.71, 655.60, 659.55, 654.78 | 655.60 | ~1.19x |
| graph-enabled  (`POLYGEIST_CUDA_GRAPH=1`) | 550.61, 549.88, 546.20, 548.57, 546.24 | 548.57 | ~1.41x |

- CUDA-graph launch-transition speedup: `655.60 / 548.57 = 1.195x` (16.3% reduction).
- vs native MFEM resident CUDA baseline (137.9648 us): graph path is 3.98x slower.

Reproduces the 2026-08-19 record (655.12 us / 549.36 us / 1.1925x) within
run-to-run noise. Graph replay removes a measurable, stable part of the
launch/dispatch cost but does not close the native gap (the general cuTensorNet
network still moves intermediates through global memory vs MFEM's fused
element-local kernel).

## Scope / caveat

Only this row was regenerated. The full MFEM silicon tables are NOT turnkey
reproducible: the recorded numbers were produced by a bespoke, multi-session,
per-kernel manual pipeline (hand-authored ABI MLIR with `memref<?xf64>`
signatures that bypasses `gen_wrapper.py`, and scattered `/tmp` build variants).
The standard `polygeist_build.sh` path fails at `gen_wrapper.py`
(`Unrecognized arg: const double *` — the harness declares unnamed pointer
prototypes the wrapper generator can't parse), and the Polygeist runtime tree is
dirty with uncommitted changes. Faithfully regenerating all ~31 rows would be a
forensic rebuild, not a re-run.

Local logs:
- `scripts/correctness/logs/abs_mass_regen_graphon_20260827_20260827_095350.silicon.log`
- `scripts/correctness/logs/abs_mass_regen_graphdis_20260827_20260827_095434.silicon.log`
