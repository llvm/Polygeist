*Ginsbach follow-up: sequential Orin #2 results (2026-09-05 PDT)*

This is the raw non-HTML handoff requested for the five remaining application
experiments. Every executable was cross-compiled on the x86 host and only the
finished AArch64 binary and input files were staged on Orin #2
(`nvidia@192.168.57.1` through `pva-general`). No project-authored
computational kernel is counted: the GPU routes below use cuBLAS or cuDNN.

## 1. NPB UA

- Official Class S CPU baseline: NASA verification passes 3/3; median
  benchmark time `0.39 s`.
- Replacing all three DAXPBY sites with `cublasDscal` + `cublasDaxpy`: NASA
  verification passes; benchmark time `2.42 s` (`6.21x` slower than the CPU
  median).
- All nine `convect` Ddot sites now compile and link after lowering its scalar
  accumulator broadcast, dynamic identity views, flattened C-array calls, and
  C external-call ABI correctly.
- Both transfer helpers now compile too: `transfb_cor_e` emits three Ddot
  sites and `transfb_cor_f` emits two. Together with `convect` and the three
  DAXPBY sites, the complete 17-site AArch64 application binary links.
- The full Ddot application is not a passing result. A `convect`-only run was
  stopped after `376 s` without reaching the official checksum. Every match is
  a length-5 dot product, producing millions of individually synchronized
  cuBLAS calls. The next required transformation is outer-loop batching or
  fusion; dispatching these scalar blocks independently is not viable.

Evidence:

- `scripts/correctness/logs/npb_ua_class_s_official_baseline_20260905_205435.silicon.log`
- `scripts/correctness/logs/npb_ua_three_daxpby_gpu_20260905_205845.silicon.log`
- `scripts/correctness/logs/npb_ua_convect_gpu_v2_20260905_211319.silicon.log`

## 2. Original Parboil SGEMM full application

- The original `main.cc`, Parboil argument parser/timers, matrix readers, and
  output writer ran around the raised source computation and its one cuBLAS
  SGEMM call.
- A deterministic `512x512` input was used because the separately distributed
  official Parboil dataset is not present in the checkout.
- Correctness: all `262,144` output values exactly match the original CPU
  application (`max_abs = 0`).
- Three-run median compute time: CPU `0.523127 s`; raised/cuBLAS `0.133982 s`;
  compute speedup `3.90x`.
- Three-run median complete timer wall time (including file I/O): CPU
  `0.876697 s`; raised/cuBLAS `0.439798 s`; application speedup `1.99x`.
- Median runtime-shim SGEMM timing: host `57.698592 ms`, device
  `56.409313 ms`.

Evidence:

- `scripts/correctness/logs/parboil_sgemm_full_app_cpu_512_20260905_212339.silicon.log`
- `scripts/correctness/logs/parboil_sgemm_full_app_gpu_512_20260905_212354.silicon.log`

## 3. Original Parboil stencil full application

- The original C driver, Parboil argument parser/timers, binary input reader,
  iteration/pointer-swap loop, and binary output writer ran with the raised
  seven-point compute function lowered to cuDNN.
- Test: deterministic `128x128x128` grid, five iterations.
- Correctness: all `2,097,152` output values agree with the original CPU
  application; maximum absolute error `7.4505806e-09`, with zero values above
  `1e-5` error.
- Three-run median compute time: CPU `0.251026 s`; raised/cuDNN `0.303858 s`.
  The current GPU path is `1.21x` slower (`0.826x` speedup).
- Cause: every iteration recreates cuDNN descriptors/plan state, allocates
  device buffers, and copies the complete grid. Steady-state convolution
  device calls are roughly `19.2-22.0 ms`, while the first call in each
  process is roughly `49.7-60.4 ms`.

Follow-up setup-hoisting result:

- The runtime now caches the cuDNN descriptors, selected algorithm, workspace,
  filter allocation, and input/output device buffers across equal-shaped
  calls. On the same Orin #2 workload, three-run median compute time improved
  from `0.303858 s` to `0.289735 s` (`4.65%`).
- Correctness still passes: all `2,097,152` values match the CPU application
  with maximum absolute error `7.4505806e-09` and zero errors above `1e-5`.
- This remains slower than the CPU because the complete grid is still copied
  to and from the device every iteration. Removing that cost requires matching
  the outer time-step loop and retaining its ping-pong buffers on the device.
- Follow-up evidence:
  `scripts/correctness/logs/parboil_stencil_cached_128_20260905_220022.silicon.log`.

Evidence:

- `scripts/correctness/logs/parboil_stencil_full_app_cpu_128_20260905_212558.silicon.log`
- `scripts/correctness/logs/parboil_stencil_full_app_gpu_128_20260905_212608.silicon.log`

## 4. NPB BT full composition

- Both raised helpers now compile and link into the official Class S
  application: `matvec_sub` lowers to cuBLAS GEMV and `matmul_sub` lowers to
  cuBLAS GEMM. A tensor/memref static-shape mismatch in the GEMM ABI lowering
  was fixed during this run.
- The full 60-step run did not reach NASA verification: CUDA terminated it via
  the board watchdog after `263.195 s`.
- A one-step diagnostic also timed out after `63.372 s`; the untouched CPU
  one-step application completed in `0.006 s` and printed its RMS norms.
- This is not a valid passing GPU application result. BT invokes the helpers
  for individual 5-vectors and 5x5 blocks throughout three directional line
  solves. These calls must be collected into a batched cuBLAS operation (and
  kept resident) before full application testing is meaningful.

Follow-up batching infrastructure:

- Added a semantic rank-3 `C[b] -= A[b] * B[b]` library specification,
  Egglog-backed matcher entry, ABI lowering, CPU reference implementation,
  and a CUDA implementation using `cublasDgemmStridedBatched`.
- The complete synthetic path now works: rank-3 `linalg.generic` ->
  `kernel.launch @cublasDgemm_strided_batched_subtract` -> runtime ABI call.
- The direct CUDA ABI validation passes 3/3 on Orin #2. Evidence:
  `scripts/correctness/logs/cublas_batched_subtract_20260905_220736.silicon.log`.
- The remaining BT transformation is not allowed to batch the inner `i`
  sweep because each iteration consumes the preceding block solve. It must
  batch independent outer grid lines and privatize the currently reused
  `lhs` workspace for those lines.

Second follow-up (loop lifting and integration audit):

- Added the matching FP64 strided-batched GEMV-subtract route. The combined
  direct ABI validation for batched GEMM and GEMV passes 3/3 on Orin #2.
  Evidence:
  `scripts/correctness/logs/cublas_batched_gemm_gemv_subtract_20260905_221620.silicon.log`.
- Structured Egglog now proves both legal loop forms over disjoint leading
  slices and rewrites them to the external cuBLAS strided-batched ABIs:
  `looped_gemm_as_batched_gemm` and `looped_gemv_as_batched_gemv`. The
  regression lowers through `polygeist-opt` to the two runtime calls with no
  residual parent loop.
- Translating upstream `solve_subs.c` and `x_solve.c` together with frontend
  inlining enabled does expose all helper arithmetic, but does so too early:
  the 5x5 operations become hundreds of scalar loads/stores and no longer
  raise as GEMM/GEMV. With `--no-inline`, the helper boundaries survive and
  `matvec_sub` and `matmul_sub` independently raise to the expected generics.
- Raising every function before inlining is not yet usable: the current
  raise pipeline combines part of `x_solve` into an invalid 200+ operand
  generic whose inferred extent is 14 for a physical extent-13 buffer. The
  next integration step must apply raising only to the helper functions,
  then inline those already-raised bodies into the untouched caller.
- That ordering alone is insufficient for a complete BT run. The upstream
  caller reuses one global `lhs/fjac/njac` workspace for every `(k,j)` line.
  A legal batched schedule must first privatize it by line and distribute the
  solve into (1) per-line coefficient construction, (2) an `i`-sequential but
  line-batched forward solve, and (3) an `i`-sequential but line-batched back
  solve. The block solve in each forward step should map to an external
  batched LU/solve implementation; it must not become a project-authored CUDA
  kernel.

Evidence:

- `scripts/correctness/logs/npb_bt_class_s_two_cublas_20260905_212940.silicon.log`
- `scripts/correctness/logs/npb_bt_class_s_two_cublas_1step_20260905_213450.silicon.log`
- `scripts/correctness/logs/npb_bt_class_s_cpu_1step_20260905_213438.silicon.log`

## 5. NPB FT / cuFFT

- The newly cross-built official Class S CPU application passes NASA
  verification 3/3 with the expected six checksums. Benchmark-reported time
  is `0.10 s` in each run; median process elapsed time is `0.165 s`.
- `fftXYZ` completes frontend translation and raising, but the production
  matcher emits zero library launches. Therefore there is no cuFFT application
  binary to run yet.
- The precise gap is algorithmic recognition: NPB implements a staged
  Stockham FFT using surrounding loops, scratch arrays, twiddle-factor table
  accesses, and complex butterfly updates. The existing cuFFT matcher only
  recognizes an explicit direct-DFT `linalg.generic`; scalar Egglog body
  equivalence cannot prove that the multi-stage loop program is an FFT.
- Required next work: represent FFT stages, loop domains, permutation, and
  twiddle indexing in the loop-aware equivalence model; match the complete 3D
  transform; then lower it to a pre-existing cuFFT `PlanMany`/3D operation
  that understands NPB's padded `n1+1` layout.

Evidence:

- `scripts/correctness/logs/npb_ft_class_s_cpu_20260905_213723.silicon.log`
- `/tmp/npb_ft_fftxyz_lift.log` (zero matches; transient local build log)

## Consolidated outcome

- NPB MG now has three external cuDNN matches. The exact ABI smoke passes all
  three semantics, and one original Class S application binary containing all
  three raised replacements passes NASA verification in 3/3 runs. Median
  benchmark time is `0.25 s`; median process time is `0.646 s`, versus `0.008
  s` for the same-source CPU baseline (`80.8x` slower end to end). Per-stage
  transfers dominate this deliberately small class.
- Full application correctness and useful speedup: Parboil SGEMM (`3.90x`
  compute, `1.99x` including I/O on the deterministic 512 input).
- Full application correctness but slower: Parboil stencil (`1.15x` slower
  after caching cuDNN setup).
- Official correctness for a partial composition: UA DAXPBY (passes, `6.21x`
  slower).
- Tiny-call regressions are now profitability-gated: UA's recognized length-5
  Ddot operations remain in Linalg. BT's combined strided-batched GEMM+GEMV
  ABI passes 3/3, but the full application still needs workspace
  privatization and loop-to-batch integration.
- No executable external-library match yet: FT Stockham FFT.

The dominant next compiler task is batching/fusing library matches across
outer loops. After that, add whole-algorithm Stockham-to-cuFFT recognition.
