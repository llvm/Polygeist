# FP32 PolyBench contraction-backend selector audit

This directory separates two questions that must not be conflated:

1. `raising_audit.csv` records whether unchanged PolyBench/C compiled with
   `-DDATA_TYPE_IS_FLOAT` currently raises and matches.
2. `selector_results.csv` exercises `--select-contraction-backend` after a
   semantic contraction has been established.  Its dimensions come directly
   from the canonical PolyBench MINI through EXTRALARGE dataset headers.

The selector sweep is a policy test.  A row is not evidence that the current
matcher produced an executable for that source occurrence.  The
`cublas-tf32` choice is implemented end to end through
`cublasGemmEx(..., CUBLAS_COMPUTE_32F_FAST_TF32, ...)`; direct MMA, WMMA, and
cuBLASDx remain policy labels whose code emission is follow-on work and must
not be reported as executed backends.

Run from the repository root:

```
python3 issues/polybench_section42/fp32_backend_selector/run_selector.py
```

The command retains both its generated semantic MLIR and selected MLIR.  TF32
is explicitly enabled for this experiment; the compiler default remains
strict FP32 and therefore stays on the vendor-library path.

Fresh LARGE source audit commands used the same canonical pipeline as the
Section 4.2 campaign, with only the documented datatype macro changed:

```
POLYGEIST_SKIP_LINK=1 \
POLYGEIST_BUILD_WORK_DIR=/tmp/polybench-fp32-selector/NAME/work \
POLYGEIST_MINIMAL_CUDA_RUNTIME=1 \
POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE=1 \
POLYGEIST_GPU_ARCH=sm_87 \
scripts/correctness/polygeist_build.sh --target=jetson --whole-program \
  --function=kernel_NAME SOURCE.c -O3 \
  -Itools/cgeist/Test/polybench/utilities -IKERNEL_DIRECTORY \
  -DLARGE_DATASET -DDATA_TYPE_IS_FLOAT
```

The fresh silicon evidence is summarized in `silicon_results.csv`.  Raw logs
are retained under `scripts/correctness/logs/` with
`*_fp32_{strict,tf32}_selector_timing_5x5_20260909_*.silicon.log`,
`*_tf32_selector_correctness_20260909_*.silicon.log`, and
`*_fp32_native_reference_20260909_*.silicon.log` names.

## Results

The canonical-size policy sweep contains 45 mathematical contraction terms:

- 23 select `cublas-tf32` through global `cublasGemmEx`.
- 15 symmetric rank-k terms retain their strict vendor-library backend.
- 5 select `direct-mma` as a not-yet-lowered policy label.
- 2 select `wmma` as a not-yet-lowered policy label.
- No canonical occurrence selects scalar CUDA FMA with the current cutoffs.

At LARGE, GEMM, 2MM, and 3MM select explicit TF32 cuBLAS.  SYRK and SYR2K
retain strict cuBLAS because controlled Orin measurements found no repeatable
benefit from cuBLAS TF32 math mode for those APIs.  MINI supplies the five
direct-MMA labels; MINI GEMM and SMALL 3MM supply the two WMMA labels.  These
small-case labels are verified selector outputs, not performance results for
generated backends.

The initial policy incorrectly selected cuBLASDx solely from total work. The
implementation audit corrected that rule: cuBLASDx is block-level and now
requires explicit structural legality metadata from an earlier planner.
Ordinary large or dynamic FP32 GEMMs use global cuBLAS instead.

The fresh unchanged-source audit now builds all five LARGE programs.  It
recovers one GEMM site, two 2MM GEMM sites, three 3MM GEMM sites, one SYRK
site, and one fused SYR2K site (the latter represents two product terms).
The previous f64-literal bug in FP32 2MM, the missed 3MM matches caused by
front-end inlining, the absent FP32 rank-k specifications, and a dynamic-slice
ABI rewiring failure were fixed with structural/type-driven logic.

Every executable was cross-compiled on x86 and only the completed AArch64
binary was transferred.  The Orin ran in MAXN with CPU, GPU (1.3005 GHz), and
EMC (3.199 GHz) clocks fixed and the fan at full speed.  Each timing is one
process with five warmups followed by five synchronized samples; the headline
is the minimum, while raw samples and medians remain in `silicon_results.csv`.

Against an independently cross-compiled unchanged PolyBench/C FP32 reference,
TF32 passes the complete printed output for GEMM, 3MM, SYRK, and SYR2K at
`rtol=5e-4, atol=1.1e-2`, with no NaN/Inf.  2MM fails 89 of 960,000 printed
values at that deliberately tight threshold (maximum relative error
7.5385e-4), so its 1.179x minimum-time speedup is not correctness-approved.
The correctness-approved improvements are GEMM (1.242x) and 3MM (1.126x).
SYRK is noise-level by minimum and slower by median; SYR2K is slightly slower,
so the final selector does not enable TF32 for either symmetric rank-k API.

The strict control is explicit rather than CUDA-default-dependent: GEMM uses
`CUBLAS_COMPUTE_32F_PEDANTIC`, and rank-k sets `CUBLAS_PEDANTIC_MATH` around
the library call.  The experimental routes use
`CUBLAS_COMPUTE_32F_FAST_TF32` or `CUBLAS_TF32_TENSOR_OP_MATH`.  Nsight Compute
is not installed on this board, so these results establish the explicit cuBLAS
compute policy and observed speedup, but do not claim instruction-counter
proof of HMMA execution.
