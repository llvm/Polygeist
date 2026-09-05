# Agent Notes

## Paper / Overleaf Location

- The local Overleaf Git clone for the current paper is
  `/home/arjaiswal/cgo-paper`.
- The authoritative LaTeX source is `/home/arjaiswal/cgo-paper/main.tex`.
- The evaluation begins at `main.tex:350`; Section 4.2, **Performance
  Improvements**, begins at `main.tex:373` and contains the intended
  PolyBench, ATen, MFEM, Llama, KernelFaRer, Polly, and optional
  Ginsbach/NPB/Parboil benchmark matrix.
- The clone's `origin` is the Overleaf Git project
  `https://git@git.overleaf.com/6a18cf44ff71410da93f7aa6`.
- Check the timestamps and Git history before treating `main.pdf` as current;
  the local PDF may lag behind `main.tex`. Pull/fetch only when the user asks
  to synchronize with Overleaf, and preserve untracked local figure assets.

## External-Library-Only Matching Rule

- Do not write custom CUDA, HIP, OpenCL, CPU, or other computational kernels
  to make a raised operation appear matched or executable.
- A successful kernel match must lower to a pre-existing implementation from
  an external library or platform API (for example cuBLAS, cuDNN, cuSPARSE,
  cuFFT, cuTENSOR, cuTensorNet, or a standard CUDA runtime primitive).
- Compiler/runtime adapter code may marshal arguments, descriptors, layouts,
  and memory for an external API, but it must not reimplement the matched
  computation with project-authored loops or device kernels.
- Matches backed only by project-authored computational implementations must
  be labeled `custom/non-library` and must not count as supported library
  lowerings, passing library matches, or paper-comparison successes.
- Do not add benchmark names, source-function names, or benchmark-specific
  arithmetic checks to force a match. Matching must be derived from the
  raised IR and a real external-library specification.
- Before adding or retaining a lowering, record the external library symbol
  that implements it. If no such implementation exists, leave the operation
  unmatched and report the missing library route honestly.
- The former project-authored 7-point stencil, MG, histogram, TPACF, JDS
  SpMV, and CSR SpMV CUDA routes were removed in September 2026. Keep their
  structural/Egglog recognition analysis-only until a permitted external
  library implementation is wired.

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
  - `whisper_vec_dot` -> `cublasSdot` (dtype-gated; f64 dot uses `cublasDdot`)
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
    `copy + cublasDscal`; a project-authored CUDA kernel is not permitted. Until
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

### Elementwise Semantic Matcher Prototype

- 2026-06-05 implemented a first semantic-recognition fallback in
  `scripts/correctness/kernel_match.py` and
  `scripts/correctness/kernel_match_rewrite.py`.
- Design:
  - Existing whole-body/multi-step `CompositionEntry` matching still runs first.
  - Only previously unmatched all-parallel tensor `linalg.generic` bodies reach
    the semantic fallback.
  - The fallback normalizes selected lowered scalar trees to semantic
    `External`-style nodes in the Python tuple AST, rather than adding egglog
    constructors yet.
  - This separates semantic recognition from lowering/scheduling: recognizing a
    semantic root returns a normal one-body match plan and the existing rewriter
    emits a named `kernel.launch`.
- Implemented semantic roots:
  - lowered `select/cmp/abs` minmod tree ->
    `External("minmod", a, b)`.
  - `External("minmod", center-left, right-center)` ->
    `hypar_slope_minmod`.
  - scalar-conditioned SP2 select
    `select(pred, square, 2*current - square)` ->
    `exasp2_select_square_inputs`.
- Also added a normal multi-yield composition for HyPar's final fused
  reaction/update body:
  - `reaction = source - lambda*u`
  - `next = u - dt*(flux_r - flux_l) + dt*reaction`
  - emitted as `hypar_reaction_update`.
- Rewriter fixes found while validating the semantic path:
  - `render_launch` now preserves MLIR multi-result binding syntax, e.g.
    `%19:3 = kernel.launch ...`, which is required for bodies like
    `hypar_weno_weights_js` and `hypar_reaction_update`.
  - `_scan_scalar_types` now records boolean constants printed as
    `arith.constant false`/`true` without an explicit `: i1`, avoiding `!any`
    in launch signatures for boolean captures.
- Latest semantic all-pipeline sweep:
  `/tmp/proxy_pipeline_semantic_all_1780700150/summary.txt`
- Semantic sweep result:
  - miniAMR: unchanged, 10/11 matched; the remaining body is the generic final
    six-input pointwise combine.
  - HPGMG: 6/6 matched.
  - HyPar: improved from 5/7 to 7/7 matched. New symbols:
    `hypar_slope_minmod` and `hypar_reaction_update`.
  - SWFFT: 6/6 matched.
  - ExaSP2: improved from 9/10 to 10/10 matched. New symbol:
    `exasp2_select_square_inputs`.
  - All five still have 0 residual loops and 0 residual ifs after
    debufferization.
- Standalone proxy-kernel matcher regression:
  `/tmp/proxy_kernel_semantic_match_1780700214/results.tsv`
  - 84 current `*_debuf_mr.mlir` files were swept.
  - 0 nonzero matcher exits.
  - 0 zero-match files.
  - 87 total matched bodies.
  - 0 `no_match` reports.
- Runtime/ABI caveat:
  - These are matcher/rewrite symbols. Jetson execution still needs
    `LowerKernelLaunchToCuBLAS.cpp` and runtime/kernel implementations for new
    semantic symbols such as `hypar_slope_minmod`,
    `hypar_reaction_update`, and `exasp2_select_square_inputs`, or a generic
    fused elementwise kernel lowering path.

### Kernel Runtime Pipeline Scope Pass

- 2026-06-05 added a first compiler-side hook for avoiding per-kernel runtime
  setup when matched kernels appear in a pipeline:
  `--wrap-kernel-launch-pipeline`.
- Files added/updated:
  - `lib/polygeist/Passes/WrapKernelLaunchPipeline.cpp`
  - `include/polygeist/Passes/Passes.td`
  - `include/polygeist/Passes/Passes.h`
  - `lib/polygeist/Passes/CMakeLists.txt`
  - `runtime/polygeist_cublas_rt.h`
  - `runtime/polygeist_cublas_rt_cpu.c`
  - `runtime/polygeist_cublas_rt_cuda.c`
  - `test/polygeist-opt/wrap-kernel-launch-pipeline.mlir`
- Pass behavior:
  - Inserts `func.call @polygeist_cublas_pipeline_begin()` at the start of any
    non-declaration function containing matched CUDA/cuBLAS/cuDNN runtime shim
    calls.
  - Inserts `func.call @polygeist_cublas_pipeline_end()` before each
    `func.return` in that function.
  - Declares the begin/end functions privately if missing.
  - Is idempotent: if a function already contains begin/end calls, the pass
    leaves it unchanged.
  - Recognizes post-lowering calls with prefixes:
    `polygeist_cublas_`, `polygeist_cudnn_`, `polygeist_cuda_`,
    `polygeist_rmsnorm_`, and `polygeist_whisper_`.
  - Also recognizes raw `kernel.launch` ops, but the safest intended placement
    is after `--lower-kernel-launch-to-cublas`, so only actually lowered CUDA
    runtime calls are scoped.
- Runtime behavior today:
  - CPU runtime: begin/end are no-ops.
  - CUDA runtime: begin calls `polygeist_cublas_init()` and increments a nesting
    depth; end decrements depth and synchronizes the CUDA stream at the
    outermost scope.
  - This was initially conservative; see the next memory section for the later
    runtime-cache/sync update that removes per-shim sync inside an active
    pipeline scope.
- Validation:
  - `ninja -C build polygeist-opt` passed.
  - `gcc -I runtime -c runtime/polygeist_cublas_rt_cpu.c -o
    /tmp/polygeist_cublas_rt_cpu.o` passed.
  - `build/bin/polygeist-opt --wrap-kernel-launch-pipeline
    test/polygeist-opt/wrap-kernel-launch-pipeline.mlir | FileCheck ...`
    passed.
  - The same test with the pass run twice passed, confirming idempotency.
  - Smoke-tested intended placement:
    `build/bin/polygeist-opt --lower-kernel-launch-to-cublas
    --wrap-kernel-launch-pipeline --split-input-file
    test/polygeist-opt/lower-llm-kernel-launches.mlir`.
- Important limitation / next step:
  - This pass creates the pipeline scope but does not yet implement true
    device-resident tensor dataflow.
  - To safely remove per-shim synchronization, either narrow the compiler pass
    to provably adjacent shim-only regions or add dataflow analysis proving
    there are no host reads of GPU-written buffers inside the scope.

### Pipeline-Scoped Runtime Cache/Sync Update

- 2026-06-05 completed the first runtime side of
  `--wrap-kernel-launch-pipeline` in `runtime/polygeist_cublas_rt_cuda.c`.
- Runtime behavior inside an active pipeline scope:
  - `timing_gpu_end` no longer performs a per-shim
    `cudaStreamSynchronize(g_stream)` when `POLYGEIST_RT_TIMING` is disabled.
  - If timing is enabled inside a pipeline, it records host enqueue timing only;
    per-op device timing is intentionally not collected because that would
    require synchronizing every shim and erase the pipeline benefit.
  - `DEVICE_MALLOC`/`DEVICE_FREE` now route shim-local `cudaMalloc`/`cudaFree`
    through a pipeline-aware temporary-device-buffer cache.
  - Freed device temporaries are marked reusable inside the same stream-ordered
    pipeline instead of being returned to CUDA immediately.
  - Device frees for non-cache pointers encountered inside a pipeline are
    deferred until the outermost pipeline-end sync, avoiding an immediate
    `cudaFree` synchronization on those paths.
  - Temporary host staging buffers that feed async H2D copies are freed through
    `pipeline_host_free`; inside a pipeline they are deferred until the
    outermost `polygeist_cublas_pipeline_end` synchronization.
  - At outermost pipeline end, the runtime synchronizes the stream and flushes
    deferred device and host frees. Device temporary buffers stay cached for
    reuse across future pipeline scopes and are released in
    `polygeist_cublas_destroy`.
- Runtime behavior outside a pipeline scope:
  - Allocation/free and synchronization remain conservative, matching previous
    per-shim behavior.
- Validation performed locally:
  - CPU runtime still compiles:
    `gcc -I runtime -c runtime/polygeist_cublas_rt_cpu.c -o
    /tmp/polygeist_cublas_rt_cpu.o`.
  - `--wrap-kernel-launch-pipeline` FileCheck test still passes.
  - `--lower-kernel-launch-to-cublas --wrap-kernel-launch-pipeline` smoke test
    on `test/polygeist-opt/lower-llm-kernel-launches.mlir` still produces
    begin/end scopes.
- Validation still needed on Jetson/CUDA machine:
  - Compile `runtime/polygeist_cublas_rt_cuda.c` with CUDA/cuBLAS/cuDNN headers
    and libraries.
  - Run llama forward with and without `--wrap-kernel-launch-pipeline`.
  - Keep `POLYGEIST_RT_TIMING=0` for the speed test; enabling it intentionally
    changes timing behavior and may perturb performance.

### Jetson Silicon Runner Script Update

- 2026-06-05 tested and fixed `scripts/correctness/run_jetson.sh`.
- Initial test result:
  - `--dry-run --mlir /tmp/run_jetson_smoke_abi.mlir smoke` worked, but showed
    the stale default route `nvidia@jetson-orin` with a forced bounce through
    `arjaiswal@10.176.207.72`.
  - Even with `POLYGEIST_JETSON_HOST=enmity` and
    `POLYGEIST_JETSON_USER=ubuntu`, the old script still printed the bounce
    route because `ST_TRACKER_DEV_HOST` defaulted internally.
- Script fixes:
  - Added direct SSH/SCP mode when `ST_TRACKER_DEV_HOST` is unset.
  - Kept bounce mode available when `ST_TRACKER_DEV_HOST` is explicitly set.
  - Added `--exe <path> [tag]` / `--binary <path> [tag]` mode for arbitrary
    prebuilt Jetson/aarch64 executables.
  - Broadened explicit-MLIR sanity check from `polygeist_cublas_*` to any
    `polygeist_*` runtime shim call, so cuDNN/custom/PVA-style ABI calls are
    accepted.
  - Added `POLYGEIST_JETSON_RUNS` to run an executable multiple times.
  - Added `POLYGEIST_JETSON_LD_LIBRARY_PATH` with a default that includes
    `/home/<user>/venv/lib/python3.12/site-packages/nvidia/cudnn/lib`, needed
    on `ubuntu@enmity` because `libcudnn.so.9` is installed in the Python
    package path, not the default linker path.
  - Cleaned the accelerator status probe to fall back cleanly from `nvidia-smi`
    to `tegrastats` on Jetson.
- Validation:
  - `bash -n scripts/correctness/run_jetson.sh` passed.
  - Built `/tmp/jetson_smoke` with `aarch64-linux-gnu-gcc`.
  - Real silicon smoke run passed:
    `POLYGEIST_JETSON_HOST=enmity POLYGEIST_JETSON_USER=ubuntu
    POLYGEIST_JETSON_RUNS=1 scripts/correctness/run_jetson.sh --exe
    /tmp/jetson_smoke jetson_smoke`.
  - The smoke run staged to `/tmp/polygeist_jetson_runs/...`, printed
    `polygeist jetson smoke ok`, and exited 0.
- Llama/CUDA blocker discovered:
  - The Llama suffix binaries built successfully locally for Jetson:
    `/tmp/llama_pipeline_scope_20260605_172506/llama_suffix_baseline`
    and `/tmp/llama_pipeline_scope_20260605_172506/llama_suffix_wrapped`.
  - Running the CUDA binary on `ubuntu@enmity` aborts in CUDA initialization:
    `cuda error: no CUDA-capable device is detected`.
  - Independent C runtime check on the Jetson also reports:
    `cudaGetDeviceCount err=100 name=no CUDA-capable device is detected`.
  - Therefore the runner is fixed, but this Jetson currently cannot execute
    CUDA workloads until its CUDA driver/device visibility issue is resolved.

### Proxy Five-Pipeline Raised Build/Silicon Fixes

- 2026-06-06 fixed the target=Jetson build/run path for the five proxy C
  pipelines:
  - `issues/proxy_kernel_pipelines/miniamr_pipeline_easy.c`
  - `issues/proxy_kernel_pipelines/hpgmg_pipeline_easy.c`
  - `issues/proxy_kernel_pipelines/hypar_pipeline_easy.c`
  - `issues/proxy_kernel_pipelines/swfft_pipeline_easy.c`
  - `issues/proxy_kernel_pipelines/exasp2_pipeline_easy.c`
- Key lowering fixes in `lib/polygeist/Passes/LowerPolygeistSubmap.cpp`:
  - Added constant-stride tensor submap lowering, covering HPGMG coarse/fine
    maps like `(d0, d1, d2) -> (2*d0 + c0, 2*d1 + c1, 2*d2 + c2)`.
  - Added row-major flatten/unflatten lowering using
    `tensor.expand_shape`/`tensor.collapse_shape` with static shape casts,
    covering MiniAMR/SWFFT maps like
    `(d0,d1,d2) -> d2 + d0*stride0 + d1*stride1`.
  - Fixed inverse submap size handling in the pass: for
    `polygeist.submapInverse`, use `map.getNumDims()` view sizes rather than
    the result/base rank. The generated op accessor currently reports only
    result-rank sizes, which broke flatten inverse maps.
  - Added a materialized linalg fallback for rank-expanding tensor submaps that
    are legal linalg indexing maps but not slices, notably MiniAMR's 27-point
    sliding-window view `grid[i+di][j+dj][k+dk]`.
  - Added lowering for rank-expanding projection inverses by extracting the
    first slice along ignored dimensions. This handles the identity-init
    scaffolding generated around MiniAMR's temporary higher-rank views.
- Runtime/build fixes:
  - Added `runtime/polygeist_mlir_runner_utils.c`, a small C implementation of
    MLIR's `memrefCopy(int64_t elemSize, unranked_memref *src, *dst)` ABI.
    This is needed when residual `memref.copy` lowers through MLIR's runner
    utility call, especially for MiniAMR/HPGMG/SWFFT pack/copy stages.
  - Updated `scripts/correctness/polygeist_build.sh` to compile and link
    `polygeist_mlir_runner_utils.o` for both host and Jetson targets.
  - Updated the harness compile in `polygeist_build.sh` to `-O0
    -fno-inline -fno-inline-functions`. This prevents the smoke harness from
    optimizing around an included kernel definition that is later replaced by
    the generated wrapper. Without this, the optimized SWFFT smoke binary
    segfaulted on Jetson in `checksum_1d` with an invalid pointer (`x0 = 0x4`),
    while the lowered SWFFT implementation itself passed under an O0/debug
    harness.
- Matcher/build policy retained:
  - Unsupported semantic matches are rejected by the ABI allowlist in
    `scripts/correctness/kernel_match_rewrite.py` and left as residual Linalg
    instead of emitting `kernel.launch` symbols without lowering/runtime
    support.
  - Final target=Jetson build coverage:
    - MiniAMR: 0 ABI-lowerable launches, residual Linalg lowered and linked.
    - HPGMG: 0 ABI-lowerable launches, residual Linalg lowered and linked.
    - HyPar: 0 ABI-lowerable launches, residual Linalg lowered and linked.
    - SWFFT: 0 ABI-lowerable launches, residual Linalg lowered and linked.
    - ExaSP2: 4 ABI-lowerable launches, 4 runtime shim calls emitted.
- Final validation:
  - `cmake --build build --target polygeist-opt -j$(nproc)` passed after the
    pass changes.
  - `bash -n scripts/correctness/polygeist_build.sh` passed.
  - `runtime/polygeist_mlir_runner_utils.c` compiled for host and aarch64:
    `cc -O2 -c ...` and `aarch64-linux-gnu-gcc -O2 -c ...`.
  - Final build root:
    `/tmp/proxy_pipeline_final_20260606_023215`.
  - All five final aarch64 executables built:
    `miniamr_raised`, `hpgmg_raised`, `hypar_raised`, `swfft_raised`,
    `exasp2_raised`.
  - Final silicon smoke root:
    `/tmp/proxy_pipeline_final_silicon_20260606_023343`.
  - All five final binaries ran on the Jetson through
    `scripts/correctness/run_jetson.sh --exe` and matched the known CPU
    reference checksums:
    - `miniamr 49.901192783357`
    - `hpgmg 5.626727135200`
    - `hypar 15.654281386856`
    - `swfft -0.483973000000`
    - `exasp2 13.101903706667`
    - `total 83.800132012080`
- Remaining technical caveat:
  - This proves buildability and correctness on silicon for the five proxy
    pipelines. It is not a performance claim for MiniAMR/HPGMG/HyPar/SWFFT,
    since those four currently run as residual Linalg/CPU-loop lowers in this
    conservative ABI-lowerable flow. ExaSP2 is the only one in this final sweep
    with lowered runtime shim calls.

### MiniAMR 27-Point cuDNN 3D Convolution Lowering

- 2026-06-06 added ABI support for the constant/shared-filter 3D ntap
  convolution subset exposed by MiniAMR's raised 27-point weighted stencil.
- Matcher change:
  - `scripts/correctness/kernel_match_rewrite.py` now treats the recognized
    `miniamr_weighted_27pt_tensor` four-step composition as a concrete
    `cudnnConvolution3D_ntap_tensor` / `_f32_tensor` launch when the last
    reduction consumes:
    - a rank-3 coefficient/filter tensor,
    - a rank-6 `polygeist.submap` window whose trailing three dimensions are a
      constant odd filter width, and
    - a rank-3 dense output tensor.
  - The rewrite recovers the haloed rank-3 input base from the rank-6 window
    submap and emits `kernel.launch @cudnnConvolution3D_ntap_tensor(input,
    output, weights, K) -> output`.
- ABI/lowering/runtime change:
  - Added `kernel.defn` declarations for `cudnnConvolution3D_ntap_tensor` and
    `cudnnConvolution3D_ntap_f32_tensor` in
    `generic_solver/kernel_library_phase2.mlir`.
  - Added `LowerKernelLaunchToCuBLAS.cpp` lowering to runtime shims
    `polygeist_cudnn_conv3d_ntap_f64` / `_f32`, passing explicit
    `inD, inH, inW, outD, outH, outW, K, W*, A*, B*`.
  - Added CPU reference loops and CUDA/cuDNN NCDHW runtime implementations in
    `runtime/polygeist_cublas_rt_{cpu,cuda}.c`.
- Validation performed:
  - Rebuilt `build/bin/polygeist-opt` successfully with `ninja -C build
    polygeist-opt`.
  - MiniAMR host build now reports `matched 1 kernel.launch op(s)` and
    `emitted 1 func.call to runtime shim`.
  - Host smoke run of `/tmp/miniamr_conv3d_host` matched the previous checksum:
    `miniamr 49.901192783357`, total `83.800132012080`.
  - Jetson aarch64 cross-build `/tmp/miniamr_conv3d_jetson` linked
    successfully and also emitted one lowered runtime call.
  - Jetson silicon smoke using
    `scripts/correctness/run_jetson.sh --exe /tmp/miniamr_conv3d_jetson
    miniamr_conv3d_smoke` passed with exit code 0. Runtime timing confirmed the
    new cuDNN path executed:
    `POLYGEIST_RT_TIMING op=cudnnConvolution3D_ntap_f64 m=12 n=80 k=27
    host_ms=501.300608 device_ms=62.056961`, with checksum
    `miniamr 49.901192783357` and total `83.800132012080`.
- Scope/caveat:
  - This fixes the constant-filter 27-point convolution route. It does not make
    variable-coefficient stencil pieces standard cuDNN convolutions; those stay
    residual Linalg until an external-library route exists.

### CUDA Library Clones And First cuFFT Integration

- 2026-06-06 cloned additional CUDA/HPC library sources and samples into
  `third_party/` for matcher/runtime expansion:
  - `third_party/VkFFT` from `https://github.com/DTolm/VkFFT.git`
    (header-oriented FFT library with CUDA backend).
  - `third_party/finufft` from `https://github.com/flatironinstitute/finufft.git`
    (includes `include/cufinufft.h` for nonuniform FFT).
  - `third_party/CUDALibrarySamples` from
    `https://github.com/NVIDIA/CUDALibrarySamples.git` (reference examples for
    cuFFT/cuSPARSE/cuSOLVER/cuRAND/cuFFTMp/cuSPARSELt).
  - `third_party/AMGX` from `https://github.com/NVIDIA/AMGX.git` (sparse
    iterative solver library, useful for HPGMG/CG future directions).
  - Pre-existing local libraries retained:
    `third_party/cutlass` and `third_party/cuda_headers/cuda_cccl`.
- Installed the CUDA 12.6 cross-SBSA dev packages needed to compile/link
  against more NVIDIA binary libraries locally:
  - `libcufft-cross-sbsa-12-6`
  - `libcusparse-cross-sbsa-12-6`
  - `libcusolver-cross-sbsa-12-6`
  - Verified headers/libs now exist under
    `/usr/local/cuda-12.6/targets/sbsa-linux/include` and `lib/stubs`, including
    `cufft.h`, `cusparse.h`, `cusolverDn.h`, `libcufft.so`,
    `libcusparse.so`, and `libcusolver.so`.
  - `apt-get` returned a nonzero status because of an unrelated pre-existing
    `shim-signed` postinstall failure (`/var/lib/grub/esp` missing
    `/dev/sda1`). The CUDA packages themselves are installed (`dpkg -l` shows
    `ii` for all three).
- First runtime integration added for cuFFT:
  - Added runtime ABI declarations:
    `polygeist_cufft_z2z_1d(int32_t N, int32_t inverse, const double *A,
    double *B)` and `polygeist_cufft_c2c_1d(...)`.
  - Complex values are represented as interleaved real/imag pairs:
    `A[2*i+0]`, `A[2*i+1]`.
  - CPU runtime has an O(N^2) DFT fallback with cuFFT-compatible signs and
    unnormalized inverse semantics.
  - CUDA runtime includes guarded `cufft.h` support and uses `cufftPlan1d`,
    `cufftSetStream`, `cufftExecZ2Z`, and `cufftExecC2C` when the cuFFT dev
    headers are present.
  - `scripts/correctness/polygeist_build.sh` now links Jetson target binaries
    with `-lcufft -lcusparse -lcusolver` in addition to existing
    cuDNN/cuBLAS/CUDA runtime libraries.
- Kernel-launch lowering support added:
  - New library defs in `generic_solver/kernel_library_phase2.mlir`:
    `@cufftZ2Z_1D_tensor` and `@cufftC2C_1D_tensor`, using
    `tensor<?x2xf64>` / `tensor<?x2xf32>` interleaved complex layout.
  - `LowerKernelLaunchToCuBLAS.cpp` maps those symbols to
    `polygeist_cufft_z2z_1d` / `polygeist_cufft_c2c_1d` and lowers launch
    operands `(input, output, inverse)` to shim args `(N, inverse, A*, B*)`.
  - Added `test/polygeist-opt/lower-kernel-launch-cufft.mlir`; manual run of
    `polygeist-opt --lower-kernel-launch-to-cublas` lowers both launch symbols
    to the expected runtime calls.
- Validation:
  - `ninja -C build polygeist-opt` passed after the lowering changes.
  - `cc -O2 -D_POSIX_C_SOURCE=199309L -I runtime -c
    runtime/polygeist_cublas_rt_cpu.c` passed.
  - `aarch64-linux-gnu-gcc -O2 -I/usr/local/cuda-12.6/targets/sbsa-linux/include
    -I/usr/include/aarch64-linux-gnu -c runtime/polygeist_cublas_rt_cuda.c`
    passed.
  - Added `issues/cufft_runtime_smoke.c` to directly test the new ABI.
  - Host smoke using CPU fallback passed with `cufft_runtime_smoke ok`.
  - Jetson silicon smoke using
    `scripts/correctness/run_jetson.sh --exe /tmp/cufft_runtime_smoke_jetson
    cufft_runtime_smoke` passed with exit code 0. Runtime timing confirmed
    actual cuFFT execution:
    `POLYGEIST_RT_TIMING op=cufftZ2Z_1D m=4 n=1 k=1 ...` and
    `POLYGEIST_RT_TIMING op=cufftC2C_1D m=4 n=1 k=1 ...`.

### cuFFT Matcher For Raised Direct DFT

- 2026-06-06 added first matcher route from raised Linalg DFT code to cuFFT.
- Added extracted FFT/DFT fixture:
  - `issues/fft_dft1d_extracted.c`
  - `issues/fft_dft1d_harness.c`
  - The cleanly raised matcher target is
    `fft_dft1d_z2z_forward_interleaved`, which uses interleaved complex layout
    `double out[N][2]`.
- Raising observations:
  - The scalar-accumulator DFT form
    `fft_dft1d_z2z_forward` raises into a nested affine/linalg form that fails
    dominance verification during the raise pipeline.
  - The two-scalar-slice accumulation form
    `fft_dft1d_z2z_forward_inplace_accum` raises, but the residual tensor IR
    only reinserts one of the two multi-result slices. This is not a good
    residual path.
  - The interleaved-output form raises cleanly to:
    - one full `tensor<?x2xf64>` zeroing linalg.generic,
    - one DFT reduction linalg.generic over `(k, component, n)`, with
      `math.cos`, `math.sin`, and an `arith.select` selecting real vs imaginary
      contribution,
    - one final `tensor.insert_slice` into the output tensor.
- Matcher changes:
  - Added `_is_dft1d_z2z_body` in `scripts/correctness/kernel_match.py`.
  - Added `_cufft_z2z_1d_tensor` composition: zero full complex tensor +
    special direct-DFT reduction.
  - The predicate is intentionally narrow: two input component slices, one
    output, two parallel dims, one reduction dim, `math.cos`, `math.sin`,
    `arith.cmpi eq`, `arith.select`, and the expected DFT arithmetic markers.
- Rewriter changes:
  - Added custom cuFFT rewrite in `scripts/correctness/kernel_match_rewrite.py`
    that replaces the zero+DFT generics plus trailing `tensor.insert_slice`
    with one full-tensor `kernel.launch`.
  - The rewrite recovers the input base tensor from real/imag
    `tensor.extract_slice` operands and passes the original output tensor as
    the cuFFT destination.
  - It preserves the static complex lane dimension by normalizing to
    `tensor<?x2xf64>` / `tensor<?x2xf32>`, not all-dynamic `tensor<?x?xf64>`.
  - It detects forward vs inverse from the sign of the captured `2*pi`
    constant and passes an `i32` inverse flag to the runtime.
- Validation:
  - `python3 -m py_compile scripts/correctness/kernel_match.py
    scripts/correctness/kernel_match_rewrite.py` passed.
  - Dry-run on the raised fixture reports:
    `match body#[0, 1] cufftZ2Z_1D_tensor`.
  - Rewritten MLIR contains:
    `kernel.launch @cufftZ2Z_1D_tensor(%1, %0, %inverse)`.
  - `polygeist-opt --lower-kernel-launch-to-cublas` lowers this to:
    `call @polygeist_cufft_z2z_1d`.
  - Full host build via `scripts/correctness/polygeist_build.sh` emitted
    `matched 1 kernel.launch op(s)` and `emitted 1 func.call to runtime shim`;
    `/tmp/fft_dft1d_host` passed with `fft_dft1d_interleaved ok`.
  - Full Jetson build `/tmp/fft_dft1d_jetson` linked successfully.
  - Jetson silicon smoke using
    `scripts/correctness/run_jetson.sh --exe /tmp/fft_dft1d_jetson
    fft_dft1d_cufft_match` passed with exit code 0 and confirmed real cuFFT
    execution:
    `POLYGEIST_RT_TIMING op=cufftZ2Z_1D m=4 n=1 k=1 ...`.
- Scope/caveat:
  - This is a semantic direct-DFT-to-cuFFT matcher for raised Linalg, not a
    general Cooley-Tukey FFT recognizer yet.
  - The current SWFFT proxy pipeline still only contains redistribution,
    slab copy, and transpose kernels; it does not include the FFT computation
    itself. This matcher is ready for extracted SWFFT local FFT stages once
    those are isolated in C.

### Semantic Candidate Planner For Kernel Matching

- 2026-06-06 added the first explicit semantic-candidate enumeration layer,
  inspired by the Leo discussion about separating semantic recognition from
  backend lowering/scheduling.
- New matcher-side data/API in `scripts/correctness/kernel_match.py`:
  - `SemanticCandidate` records one possible interpretation of a raised
    Linalg body:
    - `name`
    - `body_indices`
    - `match_kind` (`whole`, `composition`, `subterm`, or `completion`)
    - `coverage` (`whole` or `partial`)
    - `bindings`
    - optional `defaults`, `subterm_path`, and `source`
  - `composition_semantic_candidates(...)` tries every registered
    `CompositionEntry` at a body index instead of returning only the first
    greedy match.
  - `elementwise_semantic_candidates(...)` walks scalar subexpressions and
    reports semantic nodes such as minmod/slope-minmod even when the known
    semantic kernel is only a subterm of a larger elementwise body.
  - `enumerate_semantic_candidates(...)` combines whole/composition matches,
    subterm semantic nodes, and completion/specialization candidates.
- First completion rule:
  - A whole-body `miniamr_average_7pt_tensor` match now also produces a
    candidate `conv3d_sparse_3x3x3`.
  - The completion explicitly records defaults:
    `missing_filter_taps = 20 zeros`,
    `nonzero_filter_taps = center + six axial neighbors`,
    `tap_scale = 1/7`.
  - This is the architecture hook for mapping a 7-point 3D average stencil to
    cuDNN 3D convolution by materializing a sparse 3x3x3 filter. It is not
    emitted yet; current default rewriting still leaves the 7-point average as
    residual Linalg.
- Rewriter-side diagnostic support in
  `scripts/correctness/kernel_match_rewrite.py`:
  - Added `--dry-run --show-candidates`.
  - Normal build/rewrite behavior is unchanged unless `--show-candidates` is
    passed.
  - Candidate reports distinguish:
    - exact ABI-lowerable names,
    - semantic-only nodes,
    - backend candidates where a semantic node can be routed through a
      different backend symbol later.
  - Current backend hints:
    - `miniamr_weighted_27pt_tensor -> cudnnConvolution3D_ntap_tensor`
    - `conv3d_sparse_3x3x3 -> cudnnConvolution3D_ntap_tensor`
- Validation:
  - `/usr/bin/python3 -m py_compile scripts/correctness/kernel_match.py
    scripts/correctness/kernel_match_rewrite.py` passed.
  - `kernel_match_rewrite.py /tmp/tmp.2tmoPtgkRE/linalg.mlir --dry-run
    --show-candidates` reports both:
    - `miniamr_average_7pt_tensor` as the direct semantic match, and
    - `conv3d_sparse_3x3x3` as a completion/backend candidate with 20 zero
      default taps.
  - Five proxy pipeline candidate sweep:
    - MiniAMR: 10 selected semantic matches / 20 report entries,
      15 candidates, including the sparse 3D conv completion and the existing
      27-point cuDNN3D backend candidate.
    - HPGMG: 6 selected semantic matches / 12 report entries, 7 candidates.
    - HyPar: 7 selected semantic matches / 14 report entries, 7 candidates.
    - SWFFT: 6 selected semantic matches / 12 report entries, 12 candidates.
    - ExaSP2: 10 selected semantic matches / 16 report entries,
      15 candidates.
  - Default host builds still behave as before:
    - MiniAMR emits 1 `kernel.launch` and 1 runtime shim.
    - ExaSP2 emits 4 `kernel.launch` ops and 4 runtime shims.
    - Both host smoke binaries produce the unchanged checksum set:
      `miniamr 49.901192783357`, `hpgmg 5.626727135200`,
      `hypar 15.654281386856`, `swfft -0.483973000000`,
      `exasp2 13.101903706667`, `total 83.800132012080`.

### Kernel Definitions As The ISA: Matcher Design Clarification

- 2026-06-06 discussion outcome: the stronger design is not to make
  semantic-only labels the main compiler result. The useful abstraction is:
  **library/kernel definitions are the ISA**.
- Current semantic-only labels such as `miniamr_average_7pt_tensor`,
  `hypar_weno_interp5`, or `cg_update_3out` are useful as debugging/evaluation
  annotations, but they are weak as compiler targets when they were designed
  one-to-one by observing benchmark bodies and do not have a backend route.
- The desired matcher result should be a match against a real kernel
  definition with an implementation path:
  - vendor library definitions, e.g. cuBLAS GEMM/GEMV, cuDNN convolution,
    cuFFT FFT;
  - externally implemented optimized kernel definitions only;
  - residual Linalg only for unmatched leftovers.
- Revised flow:
  - Raise C to Linalg.
  - Encode each Linalg body plus Linalg context: iterator types, indexing
    maps/subviews, dtype, rank/layout, constants, aliases.
  - Match against registered kernel/library definitions, not benchmark-only
    labels.
  - Allow generalized matching modes under one matcher:
    - exact whole-body match,
    - specialization/default completion, e.g. missing convolution taps become
      explicit zero weights, omitted RMSNorm weight becomes one, beta defaults
      to zero, etc.,
    - subterm/partial match, e.g. recognize `minmod(a,b)` inside a larger
      elementwise expression and then decide whether to split/lower it.
  - Produce a candidate list containing only backend-capable or potentially
    backend-capable matches for actual rewriting/planning.
  - Select a non-overlapping set of matches using legality and cost:
    backend availability, coverage, number of materializations, fusion loss,
    data residency, target hardware, and expected performance.
- Example: MiniAMR 7-point average should not primarily become a semantic
  label `miniamr_average_7pt_tensor`. It should match the generic
  `conv3d_ntap`/`cudnnConvolution3D_ntap_tensor` kernel definition by
  completing a sparse 3x3x3 filter:
  - center and six axial neighbor taps are `1/7`;
  - the other 20 taps are explicit zero defaults.
- Example: HyPar minmod should not primarily become `hypar_slope_minmod`
  unless that name is backed by a real implementation. The useful target is a
  library/custom-kernel definition such as `external_elementwise_minmod(a,b)`
  only once there is a permitted external-library lowering for it.
- Paper framing:
  - Linalg raising exposes algebraic loop bodies and tensor access structure.
  - Optimized kernels are treated as an ISA.
  - The matcher maps raised Linalg fragments to that ISA using equality,
    specialization/defaults, and subexpression matching.
  - Matched ISA nodes lower to pre-existing external libraries or platform APIs.
  - Unmatched residual Linalg lowers through standard MLIR codegen, retaining a
    systematic path for code that does not map to a library definition.
- Follow-up implementation adjustment:
  - `kernel_match_rewrite.py --dry-run --show-candidates` now reports only
    backend-capable kernel-definition candidates by default.
  - Semantic-only labels are no longer reported as ordinary candidates. They
    can still be inspected explicitly with `--show-semantic-only` for debug or
    evaluation bookkeeping.
  - Report labels now distinguish:
    - `kernel_candidate`: a direct ABI-lowerable definition or a semantic
      completion with a declared backend route;
    - `semantic_debug`: a semantic-only match with no current backend route.
  - MiniAMR candidate report now shows only:
    - `conv3d_sparse_3x3x3`, a completion candidate routed to
      `cudnnConvolution3D_ntap_tensor`;
    - `miniamr_weighted_27pt_tensor`, routed to
      `cudnnConvolution3D_ntap_tensor`.
  - ExaSP2 candidate report shows backend-capable candidates for zeroing,
    GEMM, and GEMV; duplicate same-body alternatives such as
    `cublasDgemm_simple` vs `cublasDgemm_alpha_only` are still visible for the
    future planner/cost model to choose between.
  - HPGMG, HyPar, and current SWFFT proxy reports have zero backend-capable
    candidates in the default view because their current recognitions do not
    yet have real backend definitions/lowerings.
