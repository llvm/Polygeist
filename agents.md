# Agent Notes

## Proxy-App Raising Fixes: miniAMR and HyPar

- miniAMR direct-source `stencil_calc` no longer fails in `cgeist` on the
  local 3-D VLA:
  `double work[x_block_size+2][y_block_size+2][z_block_size+2];`.
- Root cause: cgeist created `memref<?x?x?xf64>` but passed only one dynamic
  size operand to `memref.alloca`, so MLIR verification failed with
  "dimension operand count does not equal memref dynamic dimension count".
- Fix: in `tools/cgeist/Lib/clang-mlir.cc`, nested `VariableArrayType`
  allocation now walks the array type chain and passes every dynamic dimension
  to `memref.alloca`. The miniAMR repro now emits
  `memref.alloca(%x, %y, %z) : memref<?x?x?xf64>`.
- Current miniAMR status after the fix: `cgeist` succeeds and the raise
  pipeline runs, but the result is still `no-linalg+loops+if` because the
  source still contains global state, `scf.while`, struct/LLVM loads, and
  indirect indexing through AMR metadata. Next step is kernel extraction or
  canonicalization of the actual stencil loop body.
- HyPar `LinearADRAdvection` no longer crashes in
  `--raise-affine-to-linalg-pipeline`.
- Root cause: `FoldSCFIf` rewrote an `scf.if` with existing scalar results and
  lifted branch stores, but erased the old `scf.if` without replacing uses of
  the old results. The enclosing `scf.yield` still referenced a destroyed op,
  causing MLIR to abort with "operation destroyed but still has uses".
- Fix: in `lib/polygeist/Passes/FoldSCFIf.cpp`, map each old `scf.if` result to
  the corresponding new `scf.if` result before erasing the old op.
- Current HyPar status after the fix: the full proxy probe reports
  `memref-linalg+loops+if` with 6 `linalg.generic` ops.
- Latest full proxy probe summary is in `/tmp/proxy_app_raise_mlir/summary.txt`.

## cgeist Whisper/GGML Debugging Focus

- Do not spend debugging time on C++ / STL-heavy Whisper translation units for
  the current kernel-raising work.
- Treat files such as `whisper.cpp`, `gguf.cpp`, `ggml-backend-meta.cpp`,
  `ggml-backend-reg.cpp`, `ggml-backend.cpp`, `ggml-threading.cpp`, and other
  libstdc++/STL-heavy `.cpp` files as out of scope unless the user explicitly
  asks to resume C++ frontend work.
- Prefer C files and kernel-relevant code paths that are closer to the paper's
  linalg raising / kernel matching story:
  - `third_party/whisper.cpp/ggml/src/ggml.c`
  - `third_party/whisper.cpp/ggml/src/ggml-quants.c`
  - `third_party/whisper.cpp/ggml/src/ggml-alloc.c`
  - `third_party/whisper.cpp/examples/stb_vorbis.c`
  - `third_party/whisper.cpp/tests/test-c.c`
- For timeout-heavy full translation units, prefer function-level selection or
  extracted kernels over `--function='*'`.
- Keep C++/STL fixes already made recorded, but do not use them as the main
  success criterion for the CGO/kernel ISA narrative.

## C-File Timeout Triage Notes

- `ggml-quants.c` has a large frontend baseline because `GGML_COMMON_IMPL_C`
  pulls huge IQ lookup/grid tables from `ggml-common.h`; Clang syntax checking
  alone takes about 19 seconds. Do not classify simple quant/dequant kernels as
  failed with a 20-second timeout. Use a 60-120 second bound for targeted
  functions.
- Simple quant kernels such as `quantize_row_q4_0_ref` do compile when given
  enough budget. IQ table initialization helpers such as `iq2xs_init_impl` and
  likely `iq3xs_init_impl` are true timeout suspects and are not representative
  linalg/kernel-matching targets.
- Sampled `ggml.c` compute/graph functions compile individually
  (`ggml_mul_mat`, `ggml_soft_max_ext`, `ggml_rope`, `ggml_conv_1d`,
  `ggml_conv_2d`, `ggml_build_forward_impl`). Treat full-file timeouts as
  breadth/infrastructure issues unless a specific function is isolated.
- `ggml-alloc.c` slowdowns/timeouts are dominated by recursive MLIR type
  expansion in backend/tensor/callback structs, producing 100-200+ MB MLIR
  outputs with very few functions. Skip allocator/backend functions for the
  kernel-raising story unless ABI/type-opaquing work is explicitly requested.
- `stb_vorbis.c` lower-level decode/math functions compile individually
  (`inverse_mdct`, `decode_residue`, codebook decode helpers), while
  `stb_vorbis_decode_frame_pushdata` is a true high-level pipeline timeout.

### Current C Timeout Repros

- Allocator/backend type expansion repros:
  - `issues/ggml_alloc_signature_public_probe.c`
  - `issues/ggml_alloc_signature_internal_probe.c`
- `ggml-alloc.c` public-header empty-body signatures emit only 1-2 KB, while
  internal-header empty-body signatures emit 8-21 MB in only 6-7 MLIR lines.
  This confirms the allocator timeout is recursive internal type expansion, not
  allocator loop logic.
- `stb_vorbis.c` packet-present repros:
  - `issues/stb_vorbis_packet_present_probe.c`
  - `issues/stb_vorbis_sentinel_loop_probe.c`
- The generic sentinel loop compiles. The minimized packet-present timeout is
  the cross-page scan loop plus the continued-packet flag branch; `memcmp` alone
  is not the trigger.
- `vorbis_decode_packet_rest` still times out with
  `STB_VORBIS_NO_INLINE_DECODE`; its major helpers (`decode_residue`,
  `inverse_mdct`, `do_floor`, `vorbis_decode_initial`) compile individually.
  Treat it as a composed high-level packet pipeline timeout.

### Timeout Fix Priorities and Kernel Extraction Rule

- Do not treat all timeouts as one pass failure:
  - `ggml-alloc.c` is a recursive internal type expansion / signature printing
    problem. Fix by opaquing backend/tensor/callback structs or lowering them
    through pointer-style ABI when they are used as handles.
  - `ggml-quants.c` simple quant/dequant kernels compile; skip IQ table init
    helpers (`iq2xs_init_impl`, `iq3xs_init_impl`) for the current
    linalg/kernel-matching story.
  - `stb_vorbis.c::is_whole_packet_present` has a minimized loop+flag-branch
    lowering repro. This is a compiler/debugging issue, not a useful compute
    kernel.
  - `vorbis_decode_packet_rest` is a high-level parser/decode pipeline; its
    math helpers compile individually, so extract kernels instead of raising the
    full composed control-flow body.
- For paper-facing Whisper/GGML experiments, extract the compute kernels that
  map to optimized libraries or clean linalg forms:
  - vector dot / GEMV-style dot products
  - softmax, including max-reduce + exp/sum + normalize
  - RMSNorm
  - GELU tanh approximation
  - 1D convolution as repeated dot products, with full conv1d composition left
    as matcher/library future work
  - optional Vorbis math kernels such as `decode_residue` and `inverse_mdct`,
    but not the pushdata packet parser
- Reuse `third_party/cnn-extracted/whisper_ops.c` and
  `scripts/correctness/bake_whisper_ops_mlir.sh` for isolated Whisper kernel
  raising. This fixture is the current source-level extraction path for
  avoiding C++/STL, SIMD, allocator/backend, and parser pipeline noise.

### Isolated Whisper Kernel Raise Status

- Fresh run output:
  `/tmp/whisper_ops_mlir_isolated_20260602_211202`
- All extracted kernels in `third_party/cnn-extracted/whisper_ops.c` compile,
  raise, and debufferize with multi-root:
  - `whisper_vec_dot`: 1 tensor `linalg.generic`, no loops/ifs
  - `whisper_vec_softmax`: 1 tensor `linalg.generic`, no loops/ifs
  - `whisper_softmax_full`: 3 tensor `linalg.generic`, no loops/ifs
  - `whisper_rms_norm`: 2 tensor `linalg.generic`, no loops/ifs
  - `whisper_gelu`: 1 tensor `linalg.generic`, no loops/ifs
  - `whisper_conv1d`: 1 tensor `linalg.generic` plus one residual loop
- Matcher dry-run works with `/usr/bin/python3` because default `python3` lacks
  `egglog`.
- Matcher dry-run reports:
  - `whisper_vec_dot` -> `cublasDdot` (but IR is f32, so audit symbol naming)
  - `whisper_vec_softmax` -> `whisperExpShiftSum_f32_tensor`
  - `whisper_softmax_full` -> `cudnnSoftmaxForwardOut_tensor`
  - `whisper_rms_norm` -> `rmsnorm_unweighted_f32`
  - `whisper_gelu` -> `gelu_tanh_f32_tensor`
  - `whisper_conv1d` -> inner dot match only; full conv1d composition remains
    future matcher/library work
- Original-source probe output:
  `/tmp/whisper_original_c_kernel_raise_20260602_211402`
  - `quantize_row_q4_0_ref`: raises but produces no linalg, leaves loops/ifs
  - `decode_residue`: raise fails on mixed LLVM/memref `llvm.load`
  - `inverse_mdct`: selected output has no useful raised function body

## Proxy App Standalone Kernel Extraction Status

- Added a standalone extraction suite for the five selected C proxy apps under
  `issues/proxy_kernel_extractions/`.
- Source fixture:
  `issues/proxy_kernel_extractions/proxy_kernel_extractions.c`
- Runner:
  `issues/proxy_kernel_extractions/run_proxy_kernel_extractions.sh`
- Results note:
  `issues/proxy_kernel_extractions/RESULTS.md`
- Latest generated MLIR output:
  `/tmp/proxy_kernel_extractions_mlir`
- Latest summary:
  `/tmp/proxy_kernel_extractions_mlir/summary.txt`
- Coverage: 85 standalone probes.
  - `miniAMR`: 13 kernels covering stencil averages, weighted/directional
    stencils, material pointwise updates, and halo/block pack/unpack.
  - `HPGMG`: 29 kernels covering 7-point/27-point apply, residual, Jacobi,
    GSRB, BLAS1, reductions, restriction, interpolation, FV flux, and solver
    updates.
  - `HyPar`: 28 kernels covering finite derivatives, reconstruction/WENO,
    limiters, LinearADR, Burgers, and Euler flux/upwind bodies.
  - `SWFFT`: 6 local redistribution, slab, and transpose/layout kernels.
  - `ExaSP2`: 9 dense matrix/SP2/trace/AXPBY/SpMV/CG-step kernels.
- 2026-06-04 raising fixes for the remaining standalone proxy issues:
  - Fixed `mayAlias` bookkeeping in `lib/polygeist/Ops.cpp`: the second value's
    block-argument/noalias state now updates `isArg[1]` and `isNoAliasArg[1]`
    instead of accidentally overwriting slot 0.
  - Extended `lib/polygeist/Passes/FoldSCFIf.cpp` with a single-store
    conditional rewrite. An `scf.if` with one store and no else can now become
    `select(condition, candidate, old_output_value)` plus one store. This is
    what lets the branchy HPGMG red-black smoother
    `hpgmg_gsrb_smooth_7pt` raise to tensor Linalg.
  - Added scalar `scf.if` and `affine.if` result folding to selects. The
    affine path materializes each integer-set constraint as
    `affine.apply + arith.cmpi` and combines them with `arith.andi`. This fixes
    the HyPar branch/upwind cases and the previous
    `exasp2_normalize_dense` raise failure caused by an `affine.if` reaching a
    `linalg.generic` body with `linalg.index` operands.
  - Extended store-disjointness checks in
    `lib/polygeist/Passes/RaiseToLinalg.cpp`. Multiple stores are now accepted
    when they target distinct memrefs, or when affine constant result positions
    prove different fixed components of the same memref. This fixes
    `hpgmg_cg_update`, `hpgmg_bicgstab_update`,
    `hypar_weno_weights_js`, HyPar Euler flux bodies, and
    `exasp2_conjugate_gradient_step`.
  - Extended the hybrid affine-for raiser so affine self-loads from the exact
    same address as the final store become the Linalg `outs` block argument
    instead of being preserved as illegal affine loads after `linalg.index`
    substitution. This turns `hpgmg_interpolation_p1` into loop-free memref
    Linalg.
- Latest run status:
  - 0 `cgeist` failures.
  - 82 tensor-form Linalg results.
  - 2 loop-free memref-form Linalg results:
    `hpgmg_interpolation_p1`, `hpgmg_interpolation_p2`.
  - 1 memref-form Linalg result with residual loops:
    `miniamr_stencil_calc_27`.
  - 0 no-Linalg loop/control-flow results.
  - 0 raise failures.
- Important successful story:
  after stripping app ABI/MPI/BML/solver structs, most regular compute and
  local data-layout kernels raise cleanly. This supports the paper point that
  extracted kernels can form a useful ISA-like layer plus a fallback Linalg
  lowering path.
- Important remaining limitations:
  - `miniamr_stencil_calc_27` still leaves the explicit nested 3x3x3
    accumulation loops. It reaches memref Linalg around the outer update, but
    the fixed-size inner reduction is not composed into one tensor Linalg body.
  - `hpgmg_interpolation_p1` and `hpgmg_interpolation_p2` are loop-free
    memref-Linalg, not tensor-Linalg. They keep dynamic coarse-grid
    `memref.load` payloads inside the Linalg body, so the current debufferizer
    does not tensorize them.

### Proxy Kernel Correctness Verification

- 2026-06-04 execution verifier:
  `python3 /tmp/proxy_kernel_correctness.py --repo /home/arjaiswal/Polygeist --mlir-dir /tmp/proxy_kernel_extractions_mlir`
- Latest verifier workspace:
  `/tmp/proxy_kernel_correctness_1780640949`
- Result after rebuilding `polygeist-opt` and regenerating
  `/tmp/proxy_kernel_extractions_mlir`:
  - 85/85 standalone proxy probes still reach Linalg structurally.
  - 81/85 lower to LLVM, link into the generated C harness, and match the
    original C reference output.
  - The executable subset reports `SUMMARY failures=0 tests=81` and
    `ALL_PROXY_KERNEL_CORRECTNESS_PASS`.
- Four probes are not yet execution-verified because verifier lowering rejects
  their current Linalg/submap form:
  - `miniamr_stencil_calc_27`
  - `miniamr_stencil_27_weighted`
  - `hpgmg_apply_op_27pt`
  - `hpgmg_interpolation_p0`
- Lowering-blocker details:
  - `miniamr_stencil_27_weighted` and `hpgmg_apply_op_27pt` now have the
    correct overwrite-reduction seed, but the raised IR still contains no-op
    identity `linalg.generic` reductions such as `outs(...) { yield %out }`
    over `polygeist.submap` views. These are semantically removable, but the
    standard Linalg lowering rejects the reduction-shaped output maps with
    "expected the shape-to-loops map to be non-null".
  - `miniamr_stencil_calc_27` remains the residual-loop/memref-Linalg case and
    still carries a symbol-bearing `polygeist.submap` in the executable
    lowering path.
  - `hpgmg_interpolation_p0` raises as a 2x2x2 fanout Linalg map into the fine
    grid. This is structurally useful but not directly lowerable by standard
    Linalg because the output map is not a projected permutation/invertible
    shape-to-loop map.
- Verification-time fixes:
  - `lib/polygeist/Passes/RemoveIterArgs.cpp`: direct overwrite reductions
    (`sum = init; ...; out = sum`) now seed the destination with the iter_arg
    init before the rewritten loop. This fixes reductions that previously
    accumulated from the old output value, which was only correct for
    `out += ...` forms. The helper skips loop-carried region args so nested
    accumulator loops are not seeded from enclosing loop-carried values.
  - `issues/proxy_kernel_extractions/proxy_kernel_extractions.c`:
    `hypar_interp_second_order_muscl` now declares `fC[HL + 3][HNV]`. The old
    `HL + 2` bound made the last iteration read `fC[i + 2]` one row past the
    array, so the reference C and lowered path compared undefined behavior.

### Proxy Kernel Matcher Coverage and Operand-Role Lessons

- 2026-06-05 matcher sweep over `/tmp/proxy_kernel_extractions_mlir/summary.txt`
  reached full structural kernel-dialect coverage:
  - 85/85 standalone proxy kernels emit at least one `kernel.launch`.
  - Total emitted launches: 88.
  - Matcher exits: 0 nonzero return codes.
  - Latest sweep result file:
    `/tmp/proxy_kernel_kernel_match_after_1780674500/results.tsv`.
- Important distinction for the paper/matcher discussion: the same scalar
  algebra can differ in operand role, and that determines whether a cuBLAS
  call is valid or whether the matcher must emit a separate residual/custom
  kernel symbol.
- Example 1, in-place scale vs out-of-place scale:
  - Existing `cublasDscal` template correctly matches the in-place form
    `x[i] = alpha * x[i]`.
  - Linalg body shape for in-place scale:
    `yield alpha * Out(0)`.
  - HPGMG `scale_vector` is out-of-place:
    `out[i] = alpha * in[i]`.
  - Linalg body shape for HPGMG:
    `yield alpha * In(0)`.
  - Plain `cublasDscal` cannot represent this by itself because it mutates one
    vector in place. A CUDA lowering would need either `copy(in, out)` followed
    by `scal(out)`, or a fused custom elementwise kernel. The matcher therefore
    emits a separate semantic symbol, `elemwise_scale_input_1D`, rather than
    overclaiming `cublasDscal`.
- Example 2, in-place AXPBY vs two-input AXPBY:
  - Existing `_axpby` template matches:
    `out[i] = alpha * x[i] + beta * out[i]`.
  - Linalg body shape:
    `yield alpha * In(0) + beta * Out(0)`.
  - HPGMG `add_vectors` and ExaSP2 `axpby` use two separate source vectors:
    `out[i] = alpha * x[i] + beta * y[i]`.
  - Linalg body shape:
    `yield alpha * In(0) + beta * In(1)`.
  - The old template should not fire because the second source is not the old
    output value. The matcher now emits `elemwise_axpby_inputs_1D` for this
    out-of-place/two-input variant.
- 2026-06-05 batched-GEMV-to-GEMM probe:
  - Probe source: `issues/gemv_to_gemm_probe.c`.
  - Shape tested:
    `for b, i: Y[b][i] = 0; for b, i, k: Y[b][i] += X[b][k] * A[i][k]`.
  - Mathematically this is a batch of GEMVs, equivalently
    `Y = X * A^T`.
  - Raised/debufferized IR has two tensor `linalg.generic` ops:
    a 2-D zero fill and a contraction with iterator types
    `["parallel", "parallel", "reduction"]`.
  - Kernel matcher emits:
    `memset_zero_2D` followed by `cublasDgemm_simple`.
  - This exposed a matcher ordering issue: the alpha-capture GEMM template
    could previously match no-alpha GEMM by implicitly binding `alpha = 1`.
    Exact no-alpha GEMM/GEMV templates now run before alpha-capture variants,
    so no-alpha contractions emit ABI-compatible simple symbols.
- Other matcher-side fixes that made the 85/85 sweep possible:
  - Added parser support for egglog pretty-printed `_Term_N = ...` aliases and
    trailing constructor commas. Without this, templates that were algebraically
    identical, such as `hpgmg_norm` and `hypar_limiter_minmod`, failed because
    repeated subexpressions were not inlined before unification.
  - Added shape-gated tensor copy symbols for rank-2, rank-3, and rank-6 copy
    bodies. The existing `cublasDcopy_tensor` algebra matched some pack/unpack
    kernels, but rewrite policy rejected them because the cuBLAS copy ABI was
    1D-only.
  - Added exact semantic templates for proxy kernels that are useful as a
    kernel-ISA layer: miniAMR 7/27-point stencils, HPGMG apply/residual/smooth/
    restriction/interpolation, HyPar derivatives/limiters/WENO/Euler fluxes,
    SWFFT copies/transposes, and ExaSP2 normalize/SP2/CG update kernels.
- Runtime/ABI caveat:
  - This is structural kernel-dialect coverage, not full CUDA execution
    coverage.
  - `kernel_match_rewrite.py` can emit symbols such as
    `elemwise_scale_input_1D`, `hypar_weno_weights_js`, or
    `hpgmg_gsrb_smooth_7pt_tensor`.
  - For actual GPU execution, each emitted symbol still needs a matching
    `LowerKernelLaunchToCuBLAS.cpp` lowering case plus a runtime implementation
    or decomposition. For example, `elemwise_scale_input_1D` could lower to
    `copy + cublasDscal` or to a fused custom CUDA elementwise kernel; until
    that ABI/runtime path exists, the IR explorer can show a successful kernel
    match but CUDA lowering may still reject or leave the launch unsupported.

## Proxy Pipeline Fixtures

### miniAMR Pipeline

- 2026-06-05 added `issues/proxy_kernel_pipelines/miniamr_pipeline.c`.
- The fixture is one inline pipeline-shaped function, not a call wrapper, so
  `--select-func=miniamr_pipeline` sees the loop bodies directly.
- Included loop families:
  - halo face pack/unpack
  - block pack/unpack
  - 7-point average stencil
  - 27-point average stencil
  - material coupled sum
  - pointwise material update
  - x/y/z directional stencils
  - weighted 7-point stencil
  - weighted 27-point stencil
- Latest fixture run:
  `/tmp/miniamr_pipeline_1780675056`
- Pipeline fixture result:
  - `cgeist` succeeds.
  - `--raise-affine-to-linalg-pipeline` produces 16 `linalg.generic` ops.
  - The selected raised artifact still has 8 residual loops and 0 ifs.
  - Both default and multi-root `--linalg-debufferize` fail.
  - Running the matcher on the non-debufferized Linalg artifact emits 6
    launches: four `cublasDcopy`, one `reduce_sum_1D`, and one
    `reduce_weighted_sum_1D`.
- Primary composed-pipeline blocker:
  - The 27-point average stencil still lowers as outer affine loops containing
    a scalar `memref.alloca` accumulator and an inner reduction
    `linalg.generic`.
  - During debufferization this becomes an invalid tensor/affine region with a
    dominance failure:
    `operand #0 does not dominate this use`, where the offending operand is the
    scalar alloca defined inside the child loop region.
  - This is the same structural limitation as isolated
    `miniamr_stencil_calc_27`: the fixed 3x3x3 reduction is not collapsed into
    one clean tensor Linalg op.
- Secondary matcher issue in the composed fixture:
  - Because debufferization fails, most stencil bodies remain memref-form.
  - The isolated-kernel matcher templates for miniAMR directional/weighted
    stencils are mostly tensor-form symbols, so the non-debufferized pipeline
    artifact does not get the same 85/85-style coverage.
  - This is not an algebra miss; body inspection shows the same shapes as the
    isolated kernels.
- Fresh original-source miniAMR probe:
  `/tmp/miniamr_original_probe_1780675165`
- Original `third_party/miniAMR/ref/stencil.c::stencil_calc` result:
  - `cgeist` succeeds.
  - Raised artifact has 0 `linalg.generic`, 6 loops, 7 ifs, and 152 memref
    type mentions.
  - The original app function remains worse than the pipeline fixture because
    it contains branchy `stencil_in` selection, `scf.while` over
    `sorted_index[num_refine+1]`, global block metadata, `blocks[sorted_list]`
    indirection, LLVM GEP/load on `block` structs, and `double ****array`
    pointer-chasing before reaching the stencil values.
  - Problem chain observed in the original miniAMR source:
    `global lookup -> struct pointer -> array pointer -> dynamic loop bounds ->
    branchy control flow -> temporary VLA -> copy-back loop`.
  - Compiler-facing meaning of that chain:
    - Global lookup: the useful stencil body is reached through global AMR
      state such as sorted block/refinement metadata, not direct kernel
      arguments.
    - Struct pointer: the selected block is loaded through a `block` struct,
      which introduces LLVM GEP/load traffic before the numeric array is
      visible.
    - Array pointer: the data payload is a pointer-rich layout such as
      `double ****array`, so the frontend sees pointer chasing instead of a
      simple strided memref.
    - Dynamic loop bounds: block dimensions and refinement metadata drive loop
      bounds, so the regular stencil extents are not obvious constants at the
      kernel boundary.
    - Branchy control flow: stencil selection and boundary/control branches
      survive as `scf.if`/`scf.while`, blocking the clean affine/linalg shape.
    - Temporary VLA: the local work array
      `work[x_block_size+2][y_block_size+2][z_block_size+2]` becomes a dynamic
      stack allocation. The frontend VLA allocation bug is fixed, but the
      remaining temporary still complicates tensorization.
    - Copy-back loop: the stencil result is first staged into the temporary and
      then copied back, so the useful update is split across producer and
      consumer loops rather than appearing as one direct linalg operation.
  - This confirms the current paper-facing path should use extracted or
    pipeline-shaped kernels for miniAMR, while treating the full original app
    as future frontend/control-flow/metadata work.

### miniAMR Easy Pipeline

- 2026-06-05 added
  `issues/proxy_kernel_pipelines/miniamr_pipeline_easy.c`.
- This is a cleaned, paper-facing miniAMR-style pipeline fixture. It keeps the
  relevant pipeline families but removes the original app ABI/control-flow
  barriers and the one reduction shape that still trips debufferization:
  global block metadata, `double ****` pointer chasing, AMR sorted-list
  indirection, branchy stencil selection, and the unweighted local-scalar
  27-point accumulator.
- Included loop families:
  - halo face pack/unpack
  - block pack/unpack
  - 7-point average stencil
  - x/y/z directional stencils
  - weighted 7-point stencil
  - weighted 27-point stencil expressed with a 3x3x3 coefficient tensor
  - final pointwise six-input combine
- Latest fixture run:
  `/tmp/miniamr_pipeline_easy_1780678211`
- Pipeline result:
  - `cgeist` succeeds.
  - `--raise-affine-to-linalg-pipeline` produces 14 `linalg.generic` ops.
  - Multi-root `--linalg-debufferize` succeeds.
  - The debufferized artifact has 14 tensor `linalg.generic` ops, 0 residual
    loops, and 0 ifs.
  - Kernel matcher dry-run reports 10 matched bodies out of 11 semantic bodies:
    two `tensor_copy_2D`, two `tensor_copy_3D`, one
    `miniamr_average_7pt_tensor`, three
    `miniamr_directional_stencil_tensor`, one
    `miniamr_weighted_7pt_tensor`, and one composed
    `miniamr_weighted_27pt_tensor` spanning bodies 9-12.
  - Rewritten kernel dialect artifact has 10 `kernel.launch` ops.
  - The only unmatched body is the final clean 3D pointwise sum of six tensors:
    `final = avg7 + dir_x + dir_y + dir_z + weighted7 + weighted27`.
- Interpretation:
  - This fixture demonstrates a clean end-to-end route for a miniAMR-style
    extracted pipeline: linalg raising, tensor debufferization, and kernel-ISA
    matching all work for the stencil/copy pieces.
  - The final pointwise combine is a good residual-Linalg example for the paper:
    it can be lowered through ordinary Linalg tiling/vectorization/buffering, or
    later matched to a small generated elementwise kernel if we want 11/11
    kernel-dialect coverage.

### Five Easy Proxy Pipelines

- 2026-06-05 added four more easy composed-pipeline fixtures:
  - `issues/proxy_kernel_pipelines/hpgmg_pipeline_easy.c`
  - `issues/proxy_kernel_pipelines/hypar_pipeline_easy.c`
  - `issues/proxy_kernel_pipelines/swfft_pipeline_easy.c`
  - `issues/proxy_kernel_pipelines/exasp2_pipeline_easy.c`
- Together with `miniamr_pipeline_easy.c`, these are the five paper-facing
  pipeline fixtures:
  - miniAMR: halo/block movement, 7-point and weighted stencils, final
    pointwise combine.
  - HPGMG: apply-op, residual, Jacobi smoother, restriction, simple
    prolongation/injection, CG vector update.
  - HyPar: fourth-order derivative, WENO weights/reconstruction, limiter,
    upwind flux, reaction/update.
  - SWFFT: local pack/unpack, slab movement, and local transposes.
  - ExaSP2: dense normalization, dense square/SP2 update, diagonal trace-term
    extraction, SpMV, AXPBY, and CG update.
- Latest all-pipeline sweep:
  `/tmp/proxy_pipeline_easy_all_1780679229/summary.txt`
- Common result across all five:
  - `cgeist` succeeds.
  - `--raise-affine-to-linalg-pipeline` succeeds.
  - Multi-root `--linalg-debufferize` succeeds.
  - Rewriting through `kernel_match_rewrite.py` succeeds.
  - Every debufferized pipeline has 0 residual `affine.for`/`scf.for` loops and
    0 residual `affine.if`/`scf.if` branches.
- Final structural/matcher coverage:
  - miniAMR: 14 tensor `linalg.generic`, 10 `kernel.launch`, 10/11 semantic
    bodies matched. The remaining body is the final six-input pointwise combine
    `avg7 + dir_x + dir_y + dir_z + weighted7 + weighted27`.
  - HPGMG: 6 tensor `linalg.generic`, 6 `kernel.launch`, 6/6 matched:
    `hpgmg_apply_op_7pt_tensor`, `hpgmg_residual_7pt_tensor`,
    `hpgmg_jacobi_smooth_7pt_tensor`, `hpgmg_restriction_cell_tensor`,
    `tensor_copy_3D`, and `cg_update_3out`.
  - HyPar: 7 tensor `linalg.generic`, 5 `kernel.launch`, 5/7 matched:
    `derivative_fourth_order`, `hypar_weno_weights_js`,
    `hypar_weno_interp5`, `elemwise_avg2`, and `hypar_upwind_var_flux`.
    The two residual bodies are:
    - composed slope-difference plus minmod limiter, which does not match the
      existing standalone minmod template because the slope differences are
      computed inside the same body instead of passed as direct inputs.
    - final fused reaction plus conservative update, which is a good
      residual-Linalg lowering example.
  - SWFFT: 6 tensor `linalg.generic`, 6 `kernel.launch`, 6/6 matched. All are
    local copy/layout movement bodies (`tensor_copy_2D`/`tensor_copy_3D`) for
    pack/unpack/slab/transpose shapes.
  - ExaSP2: 10 tensor `linalg.generic`, 9 `kernel.launch`, 9/10 matched:
    `exasp2_neg_div`, `elemwise_add_out_scalar_1D`, `memset_zero_2D`,
    dense square lowered as `cublasDsyrk_alias`, diagonal trace-term extraction
    as `elemwise_scale_input_1D`, `memset_zero_1D`, `cublasDgemv`,
    `elemwise_axpby_inputs_1D`, and `cg_update_3out`. The residual body is the
    scalar-conditioned SP2 selection:
    `x = take_square ? x2 : 2*rho - x2`.
- HPGMG adjustment note:
  - An earlier HPGMG version included P2 interpolation and one-element stats
    reductions. P2 interpolation is one of the known memref-only Linalg cases,
    so it prevented useful tensor-form kernel matching for the composed
    pipeline.
  - The one-element stats reductions exposed a scalar-output reduction hazard:
    in the composed form, the raised reduction body did not use the accumulator
    block argument. Keep scalar reductions out of the easy composed fixtures
    until that lowering is fixed or verified through the correctness harness.
  - The final HPGMG easy fixture therefore uses simple injection/prolongation
    and keeps the CG update but leaves scalar norm/dot/mean reductions as a
    separate issue.
- ExaSP2 trace adjustment note:
  - The first ExaSP2 pipeline used a scalar trace reduction and exposed the same
    scalar-output reduction hazard. The current fixture extracts diagonal trace
    terms into a vector instead. This keeps the pipeline correct and
    tensorizable while leaving scalar trace reduction as a separate fix.
