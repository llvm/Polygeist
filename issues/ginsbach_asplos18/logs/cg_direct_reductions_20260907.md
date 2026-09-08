CG direct reduction mapping, 2026-09-07 PDT

- Branch: `audit/ginsbach-reductions`
- Input: retained SNU NPB CG `cg.c` source mirror
- Source translation: pass
- affine-to-Linalg raising: pass
- Linalg generics: 24
- Production launches before this change: 4 cuSPARSE SpMV
- Production launches after this change: 8 total
  - 4 cuSPARSE SpMV
  - 3 cuBLAS Ddot
  - 1 CUB squared-L2 transform-reduce
- New exact patterns:
  - zero-seeded `sum(x*x)` -> `cublasDdot(x, x)`
  - two-output `sum(y*y)` and `sum(x*y)` -> two `cublasDdot` calls
  - zero-seeded `sum((x-y)^2)` -> CUB transform-reduce plus sum
- Semantic guards:
  - FP32/FP64 must agree across operands and outputs
  - outputs must be rank-one logical views aliasing rank-zero storage
  - scalar storage must be provably initialized to zero
  - nonzero-seeded reductions remain in Linalg
- ABI lowering: 3 calls to `polygeist_cublas_dot_f64`, 1 call to
  `polygeist_cub_squared_l2_distance_f64`, 4 calls to
  `polygeist_cusparse_spmv_csr_f64*`, and zero remaining `kernel.launch` ops
- Focused compiler tests: 3 passed
  - `kernel-match-cublas-ddot-memref.mlir`
  - `lower-kernel-launch-cublas-dot-memref.mlir`
  - `lower-kernel-launch-cub-squared-l2.mlir`
- CPU runtime smoke: `PASS squared L2: f32=34.0 f64=34.0`
- Cross-compiled CUB companion: CUDA 12.6, AArch64, `sm_87`.
- Orin #2 runtime smoke: 3/3 passed for both FP32 and FP64 with the expected
  value 34.0. Retained log:
  `scripts/correctness/logs/cg_squared_l2_fixed_20260907_20260907_165833.silicon.log`.
- The first deployment attempt correctly failed before execution because the
  loader found the board's older `libpolygeist_cub.so`, which did not export
  the new symbol. Restaging the newly cross-compiled companion under its
  canonical filename fixed the deployment. Retained failure log:
  `scripts/correctness/logs/cg_squared_l2_20260907_20260907_165559.silicon.log`.
- No source, MLIR, or CUDA code was compiled on the Jetson; only the completed
  AArch64 executable and shared library were transferred.
- Full-application follow-up: the original NPB CG Class S driver linked with
  the raised helper containing 2 SpMV, 2 Ddot, and 1 squared-L2 distinct static
  sites passed NASA verification 3/3. The whole-file count of eight is a
  post-inlining call-site count, not eight distinct source occurrences.
- Full-application log:
  `scripts/correctness/logs/npb_cg_five_routes_20260907_20260907_173727.silicon.log`.
