# Llama Section 4.2 Reproduction Status

Last updated: 2026-09-08 PDT

## 2026-09-08 protocol sweep

A fresh one-process, five-warm-up, five-timed-iteration sweep executed native
CPU, raised CPU/NVPL, raised resident GPU, and ggml CUDA on Orin #2. Complete
outputs revealed that both fresh Polygeist configurations fail the existing
strict `atol=1e-3, rtol=1e-4` gate, so their timings are retained but excluded
from paper-facing results. See `protocol_1x5x5_20260908/RUN_STATUS.md` and
`protocol_1x5x5_20260908/performance.csv` for the evidence and next action.

## Objective

Produce a correctness-gated three-way comparison for the paper's Section 4.2:
optimized native C on CPU, the current Polygeist raised path on Jetson AGX
Orin, and a shape-equivalent ggml CUDA implementation on the same Jetson.

## Fixed scope

- Primary workload: the FP32, one-token, one-layer, 7B-size extracted fixture
  in `third_party/cnn-extracted/llama2_extended_forward_bench.c`.
- Dimensions: model 4096, FFN 11008, vocabulary 32000, sequence 2048, 32
  heads; token 7 at position 1024.
- This is not a full 32-layer Llama 2 run and is not a quantized GGUF result.
- The fixture deliberately uses split even/odd RoPE storage and branchless
  causal masking. These source changes must remain explicit in the paper and
  viewer.

## Source revisions

- Polygeist starting revision: `c4dda7f5` (`raisetolinalg`).
- llama2.c revision: `350e04fe35433e6d2941dce5a1f53308f87058eb`.
- whisper.cpp/ggml revision: `f24588a272ae8e23280d9c220536437164e6ed28`.

## Current audit

- Existing viewer numbers come from May/June 2026 runs and are not accepted as
  final Section 4.2 measurements.
- The old raised executable used project-authored computational CUDA shims for
  RoPE, masking, and SwiGLU. Those calls cannot be counted as external-library
  matches under the current project rule.
- Fresh current-infrastructure IR contains 47 `linalg.generic` operations.
  The matcher emits 17 launches by default, but four are project-authored
  computational CUDA helpers (`cudaMaskSelect`, two `cudaAdd`, and
  `cudaSwiGLU`) and therefore are not external-library matches.
- With those helpers disabled, the provenance-clean hybrid emits 13 launches:
  CUDA/cuBLAS/cuTENSOR/cuDNN calls plus residual CPU code. An initial run
  failed because the Darknet-oriented rank-3 SGEMM adapter misclassified
  Llama's `[head,pair,model] * [model] -> [head,pair]` contraction. The fixed
  adapter uses a flattened `(head*pair) x model` GEMV. Three corrected process
  medians are 473.546, 481.474, and 470.788 ms.
- The default path initially exposed an incorrect 128-byte cuTENSOR alignment
  promise for host-mapped arrays. The runtime now reports the actual pointer
  alignment; this removes the `misaligned address` failure.
- Strict native C (`-O3 -march=native`) process medians are 460.074, 432.023,
  and 439.698 ms. The separately labelled fast-math medians are 226.090,
  233.476, and 236.706 ms; fast-math changes the checksum and is not the
  strict numerical reference.
- The same native C source cross-compiled for the Orin CPU provides the
  apples-to-apples hardware baseline. Three strict `-O3` medians are 342.414,
  342.515, and 343.150 ms. Three explicitly labelled `-ffast-math` medians are
  144.247, 144.152, and 144.111 ms. The latter again changes the checksum.
- The independent ggml CUDA graph at revision `f24588a2` has process medians
  15.889, 15.954, and 16.0565 ms.
- The CPU-library experiment raises 11 dense operations to NVPL BLAS (four
  flattened SGEMMs and seven SGEMVs). Its three process medians are 344.352,
  354.043, and 348.448 ms, versus 340.581, 341.617, and 341.702 ms for the
  contemporaneous native-C rerun. The median-of-process-medians comparison is
  therefore 348.448 ms versus 341.617 ms (NVPL is 1.02x slower end to end).
- The scalar reference runtime measured 744.332 ms in one timing process. NVPL
  is 2.14x faster than this correctness-oriented reference implementation; a
  full-logit dump was not retained for the scalar timing run.
- Full-output gate: all 32,000 Polygeist logits pass `atol=1e-3, rtol=1e-4`
  against strict Orin CPU (maximum absolute error 8.201e-4). All 32,000 ggml
  logits pass the documented cross-implementation FP32 tolerance
  `atol=1e-2, rtol=1e-4` (maximum absolute error 4.5185e-3).
- Authoritative measurements are in `performance.csv`; operation provenance is
  in `kernel_provenance.csv`; raw full-output comparisons are retained under
  `logs/`.

## Progress

- [x] Located the Overleaf Section 4.2 requirement.
- [x] Fixed the primary benchmark dimensions and source revisions.
- [x] Regenerate current raised and matched IR.
- [x] Classify default lowered operations by implementation provenance.
- [x] Build and measure optimized native CPU (strict and fast-math separated).
- [x] Cross-build and run the provenance-clean raised executable on Orin #2.
- [x] Diagnose the initial clean raised failure, fix the rank-3 GEMV lowering,
  and pass the full-output correctness gate.
- [x] Cross-build and run the ggml CUDA baseline on the same Orin.
- [x] Compare all 32,000 logits and relevant checksums.
- [x] Write the authoritative CSV and retained logs.
- [x] Update the HTML viewer source from the retained data.
- [x] Retain the earlier Section 4.2 draft and historical measurements as
  superseded evidence rather than mixing them with the current protocol.
- [x] Regenerate the HTML viewer with a dedicated `llama-paper.html`, current
  baseline graphs, complete raw samples, and explicit raised-row exclusions.
- [ ] Repair the fresh CPU match regression and GPU `gpu.alloc` lowering,
  pass the strict full-output gate, and rerun the 1-process 5+5 sweep before
  adding a Polygeist speedup claim to the paper.
