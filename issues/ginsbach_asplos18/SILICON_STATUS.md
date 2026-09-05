# Ginsbach cross-compiled silicon status

Last updated: 2026-09-04 22:54 PDT

## Required testing rule

Everything must be compiled on the x86 development host.  The Jetson receives
only finished AArch64 executables (and explicitly staged shared libraries when
needed).  Do not compile source or MLIR on the Jetson.

Target board:

- profile: `configured-host`
- user/host: `nvidia@192.0.2.1` (Orin #2)
- hostname: `configured-host.invalid`
- architecture/GPU target: AArch64 / `sm_87`
- board-side development/runtime installation belongs under the 1 TB `/home`
  filesystem, especially `/home/nvidia/cuda-12.6`

The shared runner and ATen resident sweep currently default to this board and
include the `/home/nvidia/cuda-12.6` library paths:

- `scripts/correctness/run_jetson.sh`
- `issues/aten_c_kernels/native_cuda_results/run_resident_sweep.sh`

Do not put passwords or other credentials in this file.

## Current recognition denominator

A fresh audit was run with:

```
/usr/bin/python3 scripts/correctness/ginsbach_asplos18_audit.py \
  --out /tmp/ginsbach_asplos18_audit_current --jobs 8 --timeout 180
```

It reports 104 translation units and 18 executable launch sites:

- BT `initialize.c`: 6 `memset_zero_2D` launches
- CG `cg.c`: 4 CSR SpMV launches
- LU `l2norm.c`: 1 `memset_zero_1D` launch
- MG `mg.c`: 1 residual and 1 inverse-smoother launch
- Parboil histo: 1 saturating histogram launch
- Parboil SGEMM: 1 cuBLAS SGEMM launch
- Parboil SpMV: 1 JDS SpMV launch
- Parboil stencil: 1 seven-point stencil launch
- Parboil TPACF: 1 histogram launch

These 18 compiler sites are not claimed as a one-to-one subset of the paper's
60 published idioms because the paper does not publish source locations.

## Completed cross-compiled silicon tests

### 1. Source-faithful Parboil SGEMM: complete end to end

This is the only family currently tested through the complete source path:

```
C source -> cgeist -> Linalg -> structured matcher -> kernel.launch
-> CUDA ABI call -> local AArch64 link -> unchanged executable on Orin #2
```

Results:

- matcher emitted 1 `kernel.launch`
- ABI lowering emitted the cuBLAS runtime pipeline
- local artifact verified as AArch64 ELF
- dependencies reduced to `libcublas.so.12` and `libcudart.so.12`
- nontrivial `alpha=2`, `beta=3` numerical test passed 3/3 times
- silicon runtime called `cublasSgemm_transpose`
- the board reported `nvcc not found`, confirming that it did not compile

Log:

`scripts/correctness/logs/ginsbach_parboil_sgemm_alpha_beta_cross_20260904_215100.silicon.log`

Fixture:

`issues/ginsbach_asplos18/extracted/parboil_sgemm_harness.c`

### 2. Cross-built custom CUDA device suite: passed

The shared CUDA implementation was compiled locally with NVIDIA's SBSA cross
compiler and verified as an AArch64 object containing `sm_87` device code:

`/tmp/polygeist_stencil3d_7pt_sm87_aarch64.o`

The following direct device implementations passed 3/3 silicon runs using
CUDA allocations:

- MG residual
- MG inverse smoother
- saturating byte histogram
- TPACF histogram
- JDS SpMV
- CSR SpMV
- FP32 flat seven-point stencil

Log:

`scripts/correctness/logs/ginsbach_custom_families_cross_20260904_220333.silicon.log`

Fixture:

`issues/ginsbach_asplos18/ginsbach_cuda_smoke.cu`

This proves the cross-built device implementations, but it is not a substitute
for compiling each original benchmark source through the complete matcher.

### 3. Cross-built runtime-wrapper suite: passed

One locally cross-built AArch64 executable tested all active Ginsbach runtime
routes with page-aligned host buffers.  It passed 3/3 silicon runs:

- FP64 1-D zeroing (LU route)
- FP64 2-D zeroing (BT route)
- MG residual and inverse smoother
- saturating histogram and TPACF histogram
- sized JDS and CSR SpMV ABIs
- FP32 custom stencil
- cuBLAS SGEMM with nontrivial alpha/beta

Log:

`scripts/correctness/logs/ginsbach_all_raised_routes_cross_20260904_220629.silicon.log`

Fixture:

`issues/ginsbach_asplos18/ginsbach_runtime_cuda_smoke.c`

This validates the runtime and device layers for every active family.  It does
not yet prove that each original source translation unit produces a complete
cross-linked executable.

### 4. Focused ABI lowering tests: passed

The lowering fixtures for histogram/TPACF, MG, sparse SpMV, stencil, and the
source-faithful SGEMM matcher/lowering pass.  After the latest stencil change,
both stencil forms were rerun and passed:

- tensor-result launch: `lower-kernel-launch-stencil-f32.mlir`
- bufferized zero-result/destination launch:
  `lower-kernel-launch-stencil-f32-bufferized.mlir`

## Fixes made while running the tests

- `scripts/correctness/polygeist_build.sh`
  - production builds now pass `--enable-structured-rewrite`
- `scripts/correctness/gen_wrapper.py`
  - added scalar `char` ABI support for Parboil SGEMM
  - FP32 inference now recognizes `float *` and `const float *`
  - scalar integer dimensions are collected before pointer extent inference
  - pointer APIs with `nx`, `ny`, and `nz` infer `nx * ny * nz`
- `scripts/correctness/run_jetson.sh`
  - defaults to Orin #2 (`192.0.2.1`)
  - searches the CUDA installation on the 1 TB `/home` filesystem first
- `lib/polygeist/Passes/LowerKernelLaunchToCuBLAS.cpp`
  - tested change teaches custom stencil lowering to accept a
    bufferized zero-result destination launch in addition to a tensor result
- `test/polygeist-opt/lower-kernel-launch-stencil-f32-bufferized.mlir`
  - regression coverage for the bufferized stencil launch

The host was missing the actual nvcc executable even though SBSA cross-package
metadata was installed.  `cuda-nvcc-12-6` and its compiler-only dependencies
were installed locally.  No GPU driver was installed or replaced.

## Work in progress when interrupted

The exact Parboil `cpu_stencil` source successfully reached:

- raised Linalg
- 1 structured `customStencil3D7pt_f32_tensor` match

Its first build left tensor submaps after ABI lowering.  Enabling pre-ABI
bufferization converted the launch to a zero-result destination form, which
the stencil lowering did not accept.  Support for that form and a regression
test were added, and `polygeist-opt` was rebuilt successfully.

Both stencil lowering regression tests now pass.  The exact source rebuild
then reached one matched launch and one lowered runtime call, but failed in
generic LLVM lowering because eight input/output affine view operations and
the final output write-back remained as tensor `polygeist.submap` operations.
The retained work directory is `/tmp/tmp.b7fi15izLo`; the exact diagnostic is
in `mlir.err`.  This confirms that the launch ABI fix works and isolates the
remaining failure to flat-C-array view cleanup.

A targeted follow-up is now in progress: evaluate each flat affine submap at
logical element zero to produce the runtime pointer directly, then reconnect
the in-place runtime result to the flat output base and erase the obsolete
`submapInverse`.  A regression fixture was added as
`lower-kernel-launch-stencil-f32-flat-submap.mlir`.  All three stencil ABI
fixtures now pass, including this flat-submap case.

With that fix, the exact Parboil source completed all nine local build stages:

- 1 structured stencil launch matched
- 1 custom stencil runtime call emitted
- final artifact is an AArch64 ELF with SHA-256
  `24c32c6a07f303b967fc2125091fffef61e1a3c96dc1c12493a73a83818d0ea7`
- only the finished executable was transferred; Orin again reports no `nvcc`

The first silicon correctness run failed at interior flat index 37: got `0`,
expected `38`.  Log:
`scripts/correctness/logs/parboil_stencil_source_cross_20260904_221813.silicon.log`.
The active problem is therefore runtime pointer/extent semantics, not raising,
matching, ABI conversion, cross compilation, transfer, or CUDA availability.

Root cause of that numerical failure is confirmed: the old flat runtime treated
the 2x2x2 logical interior as eight consecutive addresses.  In a 4x4x4 flat C
array the actual interior addresses are `21,22,25,26,37,38,41,42`, not
`21..28`.  A strided f32 ABI/device route has now been added.  The compiler
derives three physical affine strides independently for all seven input views
and the output view.  Local validation currently passes:

- `polygeist-opt` rebuilt successfully
- tensor, bufferized, and flat-affine-submap stencil tests all pass
- AArch64 CUDA runtime C compiles
- AArch64/SM87 custom CUDA object compiles

The original-source rebuild and silicon rerun with this new route are next and
now complete:

- generated ABI calls `polygeist_custom_stencil3d_7pt_strided_f32`
- executable contains both the runtime shim and SM87 device symbol
- AArch64 ELF SHA-256:
  `d2abd00589cb0cbd46da032cde7c5f334effc725e5235d3f6a9fa1a6851cf655`
- original-source numerical harness passed 3/3 on Orin #2
- board reported no `nvcc`, confirming execution-only deployment

Passing log:
`scripts/correctness/logs/parboil_stencil_strided_source_cross_20260904_222530.silicon.log`.

Parboil stencil is now a complete source-faithful end-to-end silicon path.

### 5. Source-faithful extracted NPB-MG compute cores: passed

The exact `resid` and `psinv` computational loop nests were extracted from
NPB3.3-SER-C `MG/mg.c`.  Timers, debug printing, and the separate `comm3`
boundary-exchange stage are intentionally not part of these core tests, so
this is not claimed as a full MG application run.

Each core independently completed:

```
C core -> cgeist -> Linalg/structured recognition -> 1 kernel.launch
-> MG CUDA ABI -> local AArch64/SM87 link -> Orin #2
```

Both numerical harnesses passed 3/3:

- residual:
  `scripts/correctness/logs/npb_mg_resid_core_cross_20260904_222824.silicon.log`
- inverse smoother:
  `scripts/correctness/logs/npb_mg_psinv_core_cross_20260904_222824.silicon.log`

Sources/harnesses:

- `issues/ginsbach_asplos18/extracted/npb_mg_kernels.c`
- `issues/ginsbach_asplos18/extracted/npb_mg_resid_harness.c`
- `issues/ginsbach_asplos18/extracted/npb_mg_psinv_harness.c`

### 6. Exact NPB-LU l2norm: compiler blocker isolated

The original `NPB3.3-SER-C/LU/l2norm.c` was attempted with a numerical
harness.  It reaches Linalg, matches the one five-element zero-initialization
launch, and lowers the runtime call.  Final `polygeist.submap` cleanup fails
before cross-linking.

The failing reduction write-back has a rank-4 logical iteration domain but
only a rank-1, five-element result.  `LowerPolygeistSubmap` incorrectly forms
a `tensor.insert_slice` using a dynamic reduction-domain size instead of the
five-element result extent, and MLIR rejects the size mismatch.  Retained work:
`/tmp/tmp.Fr9RdxRhlN`; diagnostic: `abi.err` at `abi.mlir:50`.

This path has not run on silicon and is not passing.  The active decision is
whether to fix compact reduction-result write-back now or proceed through the
other independently executable families first.

### 7. Original Parboil TPACF doCompute: complete end to end

The unmodified benchmark source
`third_party/gpu-parboil/benchmarks/tpacf/src/base/model_compute_cpu.c` now
completes the full path:

- original `struct cartesian` ABI represented as Nx3 FP32 memrefs
- 64-bit histogram bins preserved by the generated wrapper
- 1 structured TPACF launch matched
- lowered call is `polygeist_tpacf_histogram_f32`
- locally linked AArch64/SM87 executable SHA-256:
  `14675057e595eaa8f32da397a2f54295153f79b669c793e34bf4ee529eea21bd`
- non-self input, pre-seeded bins, and independent reference passed 3/3 on
  Orin #2

Log:
`scripts/correctness/logs/parboil_tpacf_source_cross_20260904_223152.silicon.log`.

Harness:
`issues/ginsbach_asplos18/extracted/parboil_tpacf_harness.c`.

### 8. Original NPB-BT lhsinit: complete end to end

The unmodified `lhsinit` in
`third_party/ginsbach-snu-npb/NPB3.3-SER-C/BT/initialize.c` now completes the
full source path:

- all 6 expected `memset_zero_2D` sites matched
- all kernel launches lowered through the CUDA runtime ABI
- the locally linked executable is AArch64 and has SHA-256
  `d65cce363fa884407422bde3caa57ec92c878735faf4354bef75bf9d553e3a25`
- an independent numerical harness passed 3/3 on Orin #2

This required two general compiler corrections in
`LowerPolygeistSubmap.cpp`:

- injectivity analysis now replaces affine symbols with zero while proving
  only the compact view-domain map, instead of creating an invalid map that
  retains undeclared symbols
- diagonal reads such as `(d0)[s0] -> (s0, 1, d0, d0)` are materialized as
  affine tensor gathers; they are not incorrectly treated as rectangular
  `extract_slice` operations

The submap regression test and the retained exact BT intermediate both pass
the final submap-lowering pass with no remaining submaps.

Log:
`scripts/correctness/logs/npb_bt_lhsinit_source_cross_20260904_224213.silicon.log`.

Harness:
`issues/ginsbach_asplos18/extracted/npb_bt_lhsinit_harness.c`.

### 9. Source-faithful extracted NPB-CG conj_grad core: passed

The two CSR matrix-vector loop nests from NPB `conj_grad` were preserved in a
core with the original 25-iteration solver structure.  Only the private NPB
partition globals were made explicit as `n`/`nnz` arguments so an independent
harness could call it.

- both CSR source sites matched `customCsrSpmv_f64_memref`
- both launches lowered to the sized CUDA runtime route
- locally linked AArch64/SM87 executable SHA-256:
  `3add8dc3f03b060648a90e394a58de062de3e614ee9de1e624025b1179c0634a`
- a nontrivial 64-row SPD CSR system passed array and residual-norm comparison
  against an independent CPU implementation 3/3 on Orin #2

This test found and fixed a Class-S-specific matcher error: the CSR launch had
hardcoded `memref<101xi32>`, `memref<3600xf64>`, and `memref<102xf64>` types.
The kernel specification and matcher now use dynamic rank-1 memrefs and insert
casts for fixed-size source globals.  The generated C ABI wrapper also now
supports annotated `int *` memrefs.

The full original CG `main` was then retried.  It now reaches all 4 expected
CSR matches and all 4 runtime calls.  It stops later because the whole-program
IR also contains C-library pointer bridges for `fopen`, strings, and reporting
(`polygeist.pointer2memref`).  The generic MLIR pipeline does not recognize
that Polygeist operation, while applying the broad
`--convert-polygeist-to-llvm` pass currently asserts on a non-memref value.
Retained work: `/tmp/tmp.WR58DGpnmo`; diagnostic: `mlir.err`.  Thus the CSR
computation is tested, but full CG remains blocked on whole-program C ABI
cleanup rather than sparse matching or the CUDA implementation.

Log:
`scripts/correctness/logs/npb_cg_conj_grad_core_cross_20260904_224717.silicon.log`.

Sources/harness:

- `issues/ginsbach_asplos18/extracted/npb_cg_conj_grad_core.c`
- `issues/ginsbach_asplos18/extracted/npb_cg_conj_grad_harness.c`

### 10. Source-faithful extracted Parboil histogram core: passed

The saturating byte-bin update loop from Parboil `histo/src/base/main.c` was
placed in a callable core; file I/O, timers, allocation, and the surrounding
iteration loop are intentionally excluded.

- the affine loop matched `customHistogramSaturatingU8_memref`
- the launch lowered to `polygeist_histogram_saturating_u8`
- locally linked AArch64/SM87 executable SHA-256:
  `0c842bffece03de4a04207c0cdddb84926214dceda93868ca39e87dc7bfe24c9`
- a 12,000-update input with pre-seeded bins and deliberate saturation passed
  an independent reference comparison 3/3 on Orin #2

The original recognizer only handled the `scf.while` spelling embedded in the
full Parboil `main`.  It now also recognizes the direct `affine.for` spelling
using the same semantic checks and derives runtime counts from the actual loop
bound and destination memref.  The generated wrapper now accepts
`unsigned int` VLA arguments as i32 memrefs.

Log:
`scripts/correctness/logs/parboil_histo_core_cross_20260904_225130.silicon.log`.

Sources/harness:

- `issues/ginsbach_asplos18/extracted/parboil_histo_core.c`
- `issues/ginsbach_asplos18/extracted/parboil_histo_harness.c`

### 11. Source-faithful extracted Parboil JDS SpMV core: passed

The nested JDS row-reduction loop from Parboil `spmv/src/cpu/main.c` was placed
in a callable core.  Dataset conversion, file I/O, timers, and the outer 50-run
benchmark repetition are excluded.

- the complete row/reduction/permutation nest matched
  `customJdsSpmv_f32_memref`
- the launch lowered to the sized JDS CUDA runtime route
- locally linked AArch64/SM87 executable SHA-256:
  `0b450a49a10dbf3cce09a9b9ec796e3ab944efb232ef81f647470ddb9f6fef32`
- a 17-row, four-diagonal nonuniform JDS matrix with an output permutation
  passed an independent reference comparison 3/3 on Orin #2

Log:
`scripts/correctness/logs/parboil_spmv_jds_core_cross_20260904_225415.silicon.log`.

Sources/harness:

- `issues/ginsbach_asplos18/extracted/parboil_spmv_jds_core.c`
- `issues/ginsbach_asplos18/extracted/parboil_spmv_jds_harness.c`

```
POLYGEIST_BUFFERIZE_BEFORE_ABI=1 \
POLYGEIST_MINIMAL_CUDA_RUNTIME=1 \
POLYGEIST_CUSTOM_CUDA_OBJ=/tmp/polygeist_stencil3d_7pt_sm87_aarch64.o \
scripts/correctness/polygeist_build.sh \
  --target=jetson --function=cpu_stencil \
  --harness=issues/ginsbach_asplos18/extracted/parboil_stencil_harness.c \
  -o /tmp/parboil_stencil_aarch64 \
  third_party/gpu-parboil/benchmarks/stencil/src/cpu/kernels.c \
  -Ithird_party/gpu-parboil/benchmarks/stencil/src/cpu
```

After verifying the result with `file` and `aarch64-linux-gnu-readelf`, deploy
only the finished executable:

```
POLYGEIST_SILICON_PROFILE=configured-host POLYGEIST_JETSON_RUNS=3 \
scripts/correctness/run_jetson.sh \
  --exe /tmp/parboil_stencil_aarch64 parboil_stencil_source_cross
```

## Remaining exact source-to-silicon work

SGEMM, stencil, TPACF, and BT `lhsinit` are complete original-source paths.
The MG, CG, Parboil histogram, and Parboil JDS SpMV compute cores also pass as
explicitly labelled source-faithful extractions.  The following paths still
need their own
complete cross-built executable and silicon correctness run:

- full CG `main` application context (the two source-faithful CSR sites pass in
  the extracted `conj_grad` core; inlining at two call sites produces the four
  audit sites)
- LU `l2norm` (one zeroing site)
- full MG `resid` and `psinv` functions/application context beyond the tested
  extracted compute cores
- Parboil histo full `main` context (the extracted source-faithful compute
  core passes)
- Parboil SpMV full `main` context (the extracted source-faithful JDS compute
  core passes)

For each path, record separately:

1. source and selected function
2. raised Linalg count
3. matched launch name/count
4. lowered runtime symbol
5. local AArch64/SM87 artifact verification
6. silicon checksum/correctness
7. silicon log path

Do not report the family-level runtime smoke tests as completed original-source
end-to-end tests.
