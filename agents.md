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
