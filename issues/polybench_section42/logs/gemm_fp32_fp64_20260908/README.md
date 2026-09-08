# GEMM FP32/FP64 direct-cuBLAS diagnostic (2026-09-08)

All AArch64 executables were cross-compiled on the x86 host and only executed
on the Orin through the configured `pva-general` silicon profile.  Dimensions
and initialization are canonical PolyBench LARGE GEMM: `NI=1000`, `NJ=1100`,
`NK=1200`, `alpha=1.5`, and `beta=1.2`.

## Result

- FP64: unchanged PolyBench/C raises to the GEMM library ABI.  In an isolated
  worktree, replacing the committed row-wise DGEMV fallback by one
  row-major-swapped `cublasDgemm` passes all 1,100,000 output values.  Median
  raised time is 40.242176 ms device compute and 63.597120 ms harness E2E.
  The retained FP64 native PolyBenchGPU baseline is 74.829246521 ms device and
  81.750000 ms E2E.
- FP32: unchanged PolyBench/C raises to exactly one SGEMM match and passes all
  1,100,000 output values.  Median raised time is 1.369664 ms device compute
  and 14.333920 ms harness E2E.  The datatype-normalized PolyBenchGPU kernel
  also passes and measures 33.011070251 ms device and 36.785000 ms E2E.

Five warmups precede five samples in one process.  Correctness uses
`rtol=5e-4`, `atol=1.1e-2`, rejects non-finite values, and compares the full
1,100,000-value output.

## Scope caveats

The direct FP64 runtime change exists only in `/tmp/polygeist-gemm-direct`; it
has not been applied to the shared worktree or committed.  The FP32 native
CUDA adapter is derived mechanically from the retained normalized FP64
PolyBenchGPU adapter by changing its scalar/storage type from `double` to
`float`; it is therefore a modified-source diagnostic.  The raised paths use
the canonical PolyBench/C source with only the datatype build macro changed.

The `ir/` directory retains the matched and ABI-level compiler evidence.  The
large correctness logs are gzip-compressed; timing and focused library probes
are retained verbatim.

## Key build settings

```
POLYGEIST_MINIMAL_CUDA_RUNTIME=1
POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE=1
POLYGEIST_GPU_ARCH=sm_87
scripts/correctness/polygeist_build.sh --target=jetson \
  --whole-program --function=kernel_gemm \
  gemm.c -O3 -DLARGE_DATASET -DDATA_TYPE_IS_FLOAT \
  -DPOLYBENCH_DUMP_ARRAYS
```

Timing executables use the repository's GPU residency and GPU-region timing
passes, a computation-free ABI harness, five warmups, and five samples.  The
native FP32 CUDA kernel was compiled on x86 with CUDA 12.6 `nvcc`,
`-arch=sm_87`, and `-ccbin aarch64-linux-gnu-g++`.

