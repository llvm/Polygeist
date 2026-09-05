# Ginsbach cross-compiled silicon status

Last updated: 2026-09-05 PDT

## Required project rule

A successful match must lower to a pre-existing external library or platform
API. Project-authored computational CUDA/CPU kernels do not count. Compiler
and runtime code may adapt layouts, descriptors, pointers, and memory for an
external API, but may not reimplement the matched computation.

Everything for Jetson must be cross-compiled on the x86 development host. The
board receives only finished AArch64 executables and required shared
libraries. The current target is Orin #2 through profile `configured-host` at
`nvidia@192.0.2.1`, AArch64 / `sm_87`. Do not store credentials here.

## Current external-only audit

A fresh audit after the compiler fixes and external cuSPARSE/cuDNN routes
reports:

- 21 benchmark programs
- 103 real translation units (MRI-Q `computeQ.cc` is textually included by
  `main.c` and is no longer double-counted)
- 103 frontend successes
- 103 successful raising pipelines
- 555 raised `linalg.generic` operations
- 36 executable external/platform launch sites

The thirty-six launch sites are:

- NPB BT: 6 `memset_zero_2D` launches lowered to CUDA runtime memset
- NPB CG: 4 `cusparseSpMV_CSR_f64_memref` launches lowered to cuSPARSE
- NPB IS: 6 `cubHistogramEvenI32ShiftZero_memref` launches lowered to CUB
- NPB LU: 1 `memset_zero_1D` launch lowered to CUDA runtime memset
- NPB UA: 3 `cublasDaxpby` launches lowered to `cublasDscal` + `cublasDaxpy`
- NPB UA: 14 scalar-alias dot-product launches lowered to `cublasDdot`
- Parboil SGEMM: 1 launch lowered to cuBLAS SGEMM
- Parboil stencil: 1 seven-point match lowered to a sparse 3x3x3 cuDNN
  convolution

No project-authored computational launch is present in this count.

The analysis-only structured inventory remains useful and reports:

- 55 Egglog-proved structured regions
- 26 reduction-shaped regions
- 17 stencil-shaped regions
- 15 histogram candidates
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

## Unmatched external-library opportunities

- NPB CG CSR SpMV: complete Class S composition and silicon correctness are
  validated. The corpus-wide whole-file audit reports four static source
  sites; the source-faithful `conj_grad` application path contains the two
  executed static sites described above.
- NPB UA: fourteen scalar-alias reductions now lower to cuBLAS Ddot (nine in
  `convect.c`, five in `transfer.c`); full benchmark composition/correctness
  validation is still pending.
- NPB MG residual/smoother: detected stencil structure; needs a composition of
  permitted external operations or remains unmatched.
- Parboil stencil: its primary seven-point compute unit now has a cuDNN route;
  complete benchmark build/run validation remains.
- NPB IS: six direct/shifted integer histogram sites now lower to CUB; complete
  benchmark composition/correctness validation remains.
- Parboil histogram: its saturating packed-byte form is detected, but does not
  satisfy the semantics of the new integer-count CUB route.
- Parboil JDS SpMV: detected; cuSPARSE has no direct JDS operation, so it
  remains unmatched unless an external supported conversion route is found.
- Parboil TPACF: detected as an indirect histogram; remains unmatched until a
  suitable external implementation exists.

The authoritative generated audit for this round is
`/tmp/ginsbach_external_hist_fact/program_summary.csv`.

## Active compiler-gap queue

- Fixed: NPB IS no longer crashes in `FoldSCFIf`; a branch-local
  `memref.get_global` target is cloned and remapped before the old `scf.if` is
  erased. IS now raises to 55 Linalg generics.
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
  indirect histograms, MG's
  factorized 3D residual stages, FT's residual FFT loop nests, JDS SpMV, LBM's
  residual loop bodies, and full-application composition/validation.
