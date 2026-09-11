# Ginsbach cross-compiled silicon status

Last updated: 2026-09-05 PDT

## Required project rule

A successful match must lower to a pre-existing external library or platform
API. Project-authored computational CUDA/CPU kernels do not count. Compiler
and runtime code may adapt layouts, descriptors, pointers, and memory for an
external API, but may not reimplement the matched computation.

Everything for Jetson must be cross-compiled on the x86 development host. The
board receives only finished AArch64 executables and required shared
libraries. The current target is Orin #2 through profile `pva-general` at
`nvidia@192.168.57.1`, AArch64 / `sm_87`. Do not store credentials here.

## Current external-only audit

A fresh audit after the compiler fixes and external cuSPARSE/cuDNN routes
reports:

- 21 benchmark programs
- 103 real translation units (MRI-Q `computeQ.cc` is textually included by
  `main.c` and is no longer double-counted)
- 103 frontend successes
- 103 successful raising pipelines
- 677 raised `linalg.generic` operations
- 28 executable external/platform launch sites (21 computational; 7
  memory-initialization sites excluded from the compute comparison)

The twenty-eight launch sites are:

- NPB BT: 6 `memset_zero_2D` launches lowered to CUDA runtime memset, plus
  rerolled `matvec_sub`/`matmul_sub` matches lowered to cuBLAS GEMV/GEMM with
  `alpha=-1`, `beta=1`
- NPB CG: 4 `cusparseSpMV_CSR_f64_memref` launches lowered to cuSPARSE
- NPB IS: 6 `cubHistogramEvenI32ShiftZero_memref` launches lowered to CUB
- NPB LU: 1 `memset_zero_1D` launch lowered to CUDA runtime memset
- NPB MG: 3 factorized symmetric 3D stencil regions (`resid`, `rprj3`, and
  `psinv`) lowered to cuDNN convolution with operation-specific coefficients,
  strides, offsets, and alpha/beta epilogues
- NPB UA: 3 `cublasDaxpby` launches lowered to `cublasDscal` + `cublasDaxpy`
- NPB UA: 14 scalar-alias dot products are recognized, but the profitability
  guard leaves these statically tiny length-5 operations in Linalg
- Parboil SGEMM: 1 launch lowered to cuBLAS SGEMM
- Parboil SpMV: 1 whole-region JDS match lowered through a storage-only
  JDS-to-CSR adapter to 50 NVIDIA cuSPARSE SpMV calls
- Parboil stencil: 1 seven-point match lowered to a sparse 3x3x3 cuDNN
  convolution

No project-authored computational launch is present in this count.

The analysis-only structured inventory remains useful and reports:

- 69 Egglog-proved structured regions
- 29 reduction-shaped regions
- 26 stencil-shaped regions
- 12 histogram candidates
- 6 CSR SpMV candidates

These are recognition opportunities, not executable library matches.

## Retired custom routes

The former project-authored implementations and their match/lowering routes
for MG residual, MG inverse smoothing, saturating histogram, TPACF histogram,
JDS SpMV, CSR SpMV, and seven-point stencil have been removed. Their old
silicon logs prove only retired custom code and must not be cited as external
library results.

The associated CUDA sources, public runtime shims, kernel definitions,
matcher emission, ABI lowering, and custom-lowering tests were deleted.

## Valid external-library silicon evidence

- Parboil SGEMM remains a complete original-source path through cuBLAS and
  passed 3/3 on Orin #2. Log:
  `scripts/correctness/logs/ginsbach_parboil_sgemm_alpha_beta_cross_20260904_215100.silicon.log`.
  It also passed 3/3 after the CUDA-library repair and board reboot. Log:
  `scripts/correctness/logs/parboil_sgemm_post_reboot_20260905_202618.silicon.log`.
- NPB BT `lhsinit` remains a complete original-source path through CUDA
  memset and passed 3/3 on Orin #2. Log:
  `scripts/correctness/logs/npb_bt_lhsinit_source_cross_20260904_224213.silicon.log`.
- NPB LU exposes one external CUDA memset match, but the complete `l2norm`
  path is still blocked by reduction write-back lowering and is not a passing
  application result.
- The cuSPARSE CSR SpMV runtime smoke passed on Orin #2. Log:
  `scripts/correctness/logs/cusparse_csr_smoke_20260905_010216.silicon.log`.
- The complete original NPB CG Class S application now passes NASA's built-in
  verification with its extracted `conj_grad` routine lowered to two static
  cuSPARSE CSR SpMV calls and one cuBLAS dot call (416 and 400 dynamic calls,
  respectively, including the untimed warmup). The verified zeta is
  `8.5971775078648` with relative error `1.2397255990878e-15`. Log:
  `scripts/correctness/logs/npb_cg_cusparse_repro_20260905_120103.silicon.log`.
  The complete application passed verification again in all three runs after
  the CUDA-library repair and board reboot (`0.13 s` benchmark time each).
  Log: `scripts/correctness/logs/npb_cg_cusparse_post_reboot_20260905_202610.silicon.log`.
  The runtime-traced run proving all 416 cuSPARSE and 400 cuBLAS dynamic calls
  is `scripts/correctness/logs/npb_cg_cusparse_20260905_115854.silicon.log`.
- The CUB integer-histogram ABI smoke passed 3/3 runs on Orin #2. Its median
  time over six input samples is `2.891 ms` host / `2.829 ms` device. This
  validates the emitted ABI and external-library route, not a complete NPB IS
  application run. Log:
  `scripts/correctness/logs/ginsbach_histogram_factorizations_20260905_124034.silicon.log`.
- The external cuSten 2D convolution smoke passed on Orin #2. Log:
  `scripts/correctness/logs/custen_conv2d_smoke_20260905_010717.silicon.log`.
  This does not cover the benchmark's 3D stencil.
- The Parboil-form flattened 3D seven-point cuDNN adapter passed on Orin #2.
  Log: `issues/ginsbach_asplos18/logs/cudnn_stencil3d_7pt_cross_20260905_0831.silicon.log`.
  A three-run timed smoke on a `7x6x5` grid reports medians of `117.171 ms`
  host / `19.469 ms` device (initialization and descriptor preparation are
  included in host time). Log:
  `scripts/correctness/logs/ginsbach_cudnn_stencil3d_7pt_timed_20260905_120750.silicon.log`.
- The FP64 AXPBY composition (`cublasDscal` + `cublasDaxpy`) passed on Orin
  #2. Log: `issues/ginsbach_asplos18/logs/cublas_daxpby_cross_20260905_0838.silicon.log`.
- The FP64 dot-product adapter (`cublasDdot`) passed on Orin #2. Log:
  `issues/ginsbach_asplos18/logs/cublas_ddot_cross_20260905.silicon.log`.
- Timed three-run ABI smokes report median DAXPBY times of `1.503 ms` host /
  `1.320 ms` device and median Ddot times of `1.242 ms` host / `1.059 ms`
  device for `N=4`. Logs:
  `scripts/correctness/logs/ginsbach_cublas_daxpby_timed_20260905_120653.silicon.log`
  and
  `scripts/correctness/logs/ginsbach_cublas_ddot_timed_20260905_120701.silicon.log`.
- NPB MG's three factorized stencil forms pass an exact external-cuDNN ABI
  smoke, and each source function was separately substituted into the complete
  original Class S application. More importantly, one binary containing all
  three raised replacements passes NASA verification in 3/3 Orin runs. Its
  median benchmark time is `0.25 s` and median process time is `0.646 s`.
  The same-source CPU baseline also verifies 3/3 and has a median process time
  of `0.008 s`; its benchmark time is below NPB's two-decimal resolution. Thus
  this transfer-heavy Class-S GPU composition is about `80.8x` slower end to
  end; correctness and coverage are the result, not a speedup.
  Logs:
  `scripts/correctness/logs/cudnn_mg_stencil_external_20260905_234257.silicon.log`,
  `scripts/correctness/logs/npb_mg_class_s_resid_cudnn_20260905_235255.silicon.log`,
  `scripts/correctness/logs/npb_mg_class_s_psinv_cudnn_20260905_235327.silicon.log`,
  and
  `scripts/correctness/logs/npb_mg_class_s_rprj3_cudnn_20260905_235336.silicon.log`.
  Combined and CPU-baseline logs:
  `scripts/correctness/logs/npb_mg_class_s_all_cudnn_20260905_235659.silicon.log`
  and
  `scripts/correctness/logs/npb_mg_class_s_cpu_current_20260905_235715.silicon.log`.

## Unmatched external-library opportunities

- NPB CG CSR SpMV: complete Class S composition and silicon correctness are
  validated. The corpus-wide whole-file audit reports four static source
  sites; the source-faithful `conj_grad` application path contains the two
  executed static sites described above.
- NPB UA: fourteen scalar-alias reductions are recognized (nine in
  `convect.c`, five in `transfer.c`), but their statically proven length-5
  workloads are intentionally not dispatched as individual cuBLAS calls.
  The three DAXPBY sites verify in the official application composition.
- NPB MG residual/smoother/restriction: fixed. Egglog proves the three-stage
  raised regions as factorized linear stencils, and the renderer emits an
  ordinary cuDNN 3D convolution ABI. All three replacements compose and verify
  in one application. The next MG task is device-resident multigrid storage
  and call fusion to remove the dominant per-stage transfers.
- Parboil stencil: the original application passes on a deterministic
  128x128x128 grid for five iterations. Cached cuDNN descriptors, algorithm,
  workspace, and buffers reduce median compute time to `0.289735 s`, versus
  `0.251026 s` for the CPU (`1.15x` slower); transfers remain.
- NPB IS: six corpus sites lower to CUB. The full Class-S application now
  preserves the original driver and `full_verify`, replaces only `rank` with a
  source-faithful extracted core, and executes its two static histogram sites
  through CUB. All three Orin runs pass NASA verification (median 110.00
  Mop/s; benchmark-reported time 0.01 s). Each process makes 22 CUB calls:
  two sites in the warm-up rank plus ten timed ranks. Log:
  `scripts/correctness/logs/npb_is_cub_full_repro_20260905_133503.silicon.log`.
  It also passed verification in all three post-reboot runs when the required
  cross-compiled `libpolygeist_cub.so` companion was staged explicitly
  (`120.76` to `123.12` Mop/s). Log:
  `scripts/correctness/logs/npb_is_cub_post_reboot_with_companion_20260905_202650.silicon.log`.
- Parboil histogram: its saturating packed-byte form is detected, but does not
  satisfy the semantics of the new integer-count CUB route.
- Parboil JDS SpMV: fixed. The exact repeated JDS gather/reduction/permutation
  loop is matched as one region. A runtime adapter converts storage metadata
  to CSR once and preserves the source's 50 numerical repetitions as external
  NVIDIA cuSPARSE calls. Its cross-compiled ABI smoke passed 3/3 on Orin #2
  with expected output `[11, 14, 18]`. Log:
  `scripts/correctness/logs/cusparse_jds_external_20260905_230355.silicon.log`.
  A complete original-main executable is not yet claimed: downstream generic
  MLIR lowering rejects Parboil's raised opaque `FILE`-structure memref before
  linking. This is outside the matched SpMV region; the real corpus IR match,
  ABI lowering test, cross-build, and vendor-library silicon smoke all pass.
- Parboil TPACF: detected as an indirect histogram; remains unmatched until a
  suitable external implementation exists.

The authoritative generated audit for this round is
`/tmp/ginsbach_complete_final_20260905/program_summary.csv`.

## Active compiler-gap queue

- Fixed: guarded mixed SCF/affine nests now normalize before Linalg raising.
  Exact redundant non-empty guards are removed, immutable dimension loads are
  hoisted across loops using distinct-global alias checks, and loop-domain
  arithmetic stays outside nested generic payloads. Safe, statically in-bounds
  global-bound prefixes are also hoisted out of affine guards. With
  `exact_solution.c` supplied for application-faithful interprocedural
  inlining, `exact_rhs.c` improves from 2 to 42 generics and has no residual
  SCF loops. Fixed-size straight-line algebra recovery raises the fully
  unrolled `matvec_sub` and `matmul_sub` helpers to two Linalg contractions,
  which Egglog matches to external cuBLAS subtraction updates. NPB BT now has
  90 generics and 8 executable launch sites; all 14 audited units pass. See
  `issues/ginsbach_asplos18/bt_raise_results_2026-09-05.csv`.
- The full source-faithful NPB BT Class S application (12x12x12, 60 steps)
  passed NASA verification on Orin #2 in all three runs. Median wall time was
  `0.083 s` (median benchmark-reported time `0.08 s`). This is the CPU source
  baseline, not a GPU result. Two computational cuBLAS helper matches now
  exist, but the line solves still need application-level batching and device
  residency before they form a complete GPU BT result.
  Log: `scripts/correctness/logs/npb_bt_class_s_baseline_20260905_130426.silicon.log`.

- The unbatched BT GPU composition is not a passing application result: its
  one-step run hit the watchdog after `63.372 s`. The combined external
  strided-batched GEMM+GEMV ABI passes 3/3 on Orin #2, and the loop-aware
  Egglog prototype lowers both operations without a residual parent loop.
  Full BT still requires helper-first raising, per-line workspace
  privatization, and an external batched block-solve route.
  Log: `scripts/correctness/logs/cublas_batched_gemm_gemv_subtract_20260905_221620.silicon.log`.

- The new subtract GEMV/GEMM matcher and ABI lowering route passed its
  numerical silicon smoke in all three runs on Orin #2. The executable was
  cross-compiled on the host, uses `cudaMalloc`/`cudaMemcpy`, checks a raw
  cuBLAS GEMV reference result, and then checks the Polygeist subtract GEMV
  and GEMM wrappers (`alpha=-1`, `beta=1`) against exact expected values. The
  board uses the compatible Jetson CUDA 12.6 libraries staged under
  `/home/nvidia/jetson-cuda-libs`; the incompatible SBSA library directory is
  last in the fallback search path. Log:
  `scripts/correctness/logs/cublas_subtract_default_path_20260905_202530.silicon.log`.

- Fixed: NPB IS no longer crashes in `FoldSCFIf`; a branch-local
  `memref.get_global` target is cloned and remapped before the old `scf.if` is
  erased. IS now raises to 55 Linalg generics. Dynamic pointer memrefs can now
  use the CUB histogram route when a constant count-loop bound and a complete
  zero-filled submap prove the sample and bin extents.
- Fixed: NPB FT `fft3d.c` now accepts aggregate assignment between a
  fixed-size complex pair and its VLA-derived dynamic view. It completes both
  frontend and raising pipelines; the FFT loops still need cuFFT recognition.
- Fixed: descriptor-array loops no longer produce non-dominating SSA uses;
  CUTCP is 5/5 raised and SAD is 4/4 raised.
- Fixed: scalar reduction fusion no longer removes the only indexing map; UA
  is 11/11 raised.
- Fixed: descriptor arrays remain in memory form instead of being converted to
  invalid tensors. MRI-Q and the SGEMM I/O unit now raise successfully.
- Fixed: the audit asks `cgeist` for MLIR's generic operation syntax, so
  numbered multi-results in `cf.switch` successor arguments survive textual
  round trips without requiring a private LLVM-submodule patch. DC
  `jobcntl.c` now raises successfully.
- Fixed: while-to-for rewrites no longer capture values defined inside the old
  condition region, and the while-and rewrite clones/remaps its condition
  dependencies. DC `adc.c` and LBM `lbm.c` now complete the frontend.
- Fixed: shorter C aggregate/string initializers copy and zero-fill their
  destination tail, and recursive-struct memref globals receive the same
  opaque type conversion as their users. LBM `main.c` now completes both the
  frontend and raising pipeline.
- All 103 translation units now complete both frontend translation and the
  raising pipeline. There are no remaining corpus-wide frontend/raising
  failures in this audit.
- External-library gaps after raising: non-dot reductions and saturating or
  general indirect histograms, FT's residual FFT loop nests, LBM's residual
  loop bodies, and remaining full-application composition/validation. MG's
  three factorized 3D stages and Parboil JDS SpMV now have external routes.

## Dense solver validation

- PolyBench/C Cholesky MEDIUM (`n=400`) is recognized as one whole cuSOLVER
  DPOTRF operation despite the source recurrence. It passes a high-precision
  source-reference comparison in all three Orin runs (`max_rel` 2.743e-14).
  Median library-call timing is 13.737 ms host / 13.124 ms device.
- PolyBench/C Trisolv MEDIUM (`n=400`) is recognized as one whole cuBLAS DTRSV
  operation. It passes in all three Orin runs (`max_rel` 2.464e-13). Median
  library-call timing is 3.492 ms host / 2.729 ms device.
- These source kernels currently use the memref-Linalg route (`--no-debuf`).
  The default tensor/debufferized representation retains tensor submaps and is
  not yet accepted by the whole-algorithm factorization matcher.
- Logs:
  `scripts/correctness/logs/polybench_cholesky_medium_raised_timed_20260905_133515.silicon.log`
  and
  `scripts/correctness/logs/polybench_trisolv_medium_raised_timed_20260905_133520.silicon.log`.
