# B300 one-layer ggml and Polygeist results

## Common workload

Both rows use the deterministic FP32 one-token, one-layer Llama-style fixture:
model dimension 4096, FFN dimension 11008, vocabulary 32000, sequence length
2048, 32 heads, token 7, and position 1024. Each process performs five warmups
and five timed iterations; the minimum is the headline.

## Results

| Implementation | Scope | Minimum ms | Median ms | Maximum ms | Status |
|---|---|---:|---:|---:|---|
| Native ggml CUDA from pinned llama.cpp | resident synchronized graph compute | 0.387000 | 0.388000 | 0.421000 | PASS |
| Polygeist raised | synchronized per-call E2E | 331.101067 | 349.546811 | 450.875139 | PASS at declared FP32 tolerance |

The Polygeist samples are `331.101067, 333.645131, 349.546811, 380.584778,
450.875139 ms`. Its IQR is 83.356859 ms.

## Correctness

All 32000 logits were checked.

- Raised versus native llama2.c CPU: zero failures at `atol=1e-2,
  rtol=1e-4`; maximum absolute difference 0.0056457.
- Raised versus B300 ggml: zero failures at the same tolerance; maximum
  absolute difference 0.00251984.

The raised row does not pass the older, tighter implementation-specific
`atol=1e-3` gate, consistent with the previously diagnosed FP32 reduction
ordering differences. The implementation-neutral tolerance was declared
before this B300 run.

## Why the current raised number is not an apples-to-apples speed comparison

The ggml interval is a resident device graph. The currently reproducible
Polygeist build has 22 external-library launches but also retains 20
`linalg.generic` operations that lower to CPU loops. On a discrete B300 this
causes repeated synchronization and host/device staging. Its 22 matches are
cuBLAS GEMM/GEMV, cuDNN RMSNorm/softmax and pointwise graphs, and cuTENSOR
permutations; no handwritten CUDA kernel was added.

Accordingly, the current raised E2E minimum is 855.56x the ggml resident
compute minimum. This primarily measures the missing whole-pipeline residency
and residual-loop lowering, not the relative quality of the underlying vendor
library kernels. A fair resident comparison requires lowering the remaining 20
generics to supported external-library operations and keeping intermediates on
the device for the complete call.

## B300-specific fixes required to execute

- Used the existing Egglog environment under the 1-TB software tree.
- Extracted NVIDIA cuTENSOR 2.7.0.5 under the 1-TB tree rather than installing
  it into the root filesystem.
- Added host-CUDA cuTENSOR linkage to the temporary B300 build driver.
- Corrected cuDNN RMSNorm copy direction for device-resident pointers.
- Gave cached cuDNN RMSNorm plan storage a lifetime independent of the
  pipeline temporary allocator.
- Used pinned host staging for a discrete-GPU RMSNorm result copied back to a
  compiler-visible host destination.

These B300 runtime/build-driver changes are retained as explicit patches and
are not committed.
