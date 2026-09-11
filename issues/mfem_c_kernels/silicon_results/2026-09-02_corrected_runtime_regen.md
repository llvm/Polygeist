# MFEM corrected runtime regeneration — 2026-09-02

Measured on the Jetson AGX Orin at `nvidia@192.168.58.1` via
`pva-general`, SM87 MAXN, CUDA 12.6, f64, `NE=1024`, `D1D=4`, `Q1D=5`.

## Standalone kernels

All 18 raised/native pairs pass the checksum gate. Both implementations warm
three calls, then time 20 individual calls with `cudaDeviceSynchronize()` and
report the best. Published values are medians of five independent processes.
The raised ABI uses zero-copy mapped host buffers and includes residual host
loops; native MFEM uses resident CUDA buffers.

The corrected raised/native slowdown ranges from 2.611x
(`curlcurl_apply_3d`) to 100.565x (`interp_grad_3d`). Compared with the prior
table, 16 of 18 raised results improved; the median improvement is 1.69x and
the range is 0.87x–2.17x. No raised kernel beats native MFEM CUDA yet.

The first attempted performance build reused semantic MLIR specialized for the
`NE=2` correctness sweep. Its checksum mismatch exposed that it processed only
a prefix of the `NE=1024` buffers. Those timings were discarded; all published
raised binaries were regenerated from C with `MFEM_BENCH_NE=1024`.

The native DFEM timing harness previously measured queued asynchronous launch
overhead. It now uses the same synchronized best-of-N policy as the PA and
raised harnesses.

Raw Orin logs are retained at:

- `/home/nvidia/mfem_perf_20260902_raw.log` (native trials)
- `/home/nvidia/mfem_perf_ne1024_20260902_raised.log` (valid raised trials)

## Larger application paths

All 11 paths were rebuilt from C at `NE=1024` with the corrected compiler.
Ten pass correctness and report the median of five process-level means, each
using five application calls. `dfem_minimal_surface_2d` still fails only at
this production size (`max_abs=max_rel=0.19645498711635398`), so its timing is
withheld. The application table clearly labels exact native comparisons,
component-sum approximations, and unavailable native equivalents.

Raw application logs are retained at
`/home/nvidia/mfem_app_perf_ne1024_20260902.log`.
