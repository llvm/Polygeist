# Native llama.cpp/ggml one-layer CUDA baseline on B300

## Result

The exact FP32 one-token, one-layer fixture was compiled against the ggml
libraries built by the pinned upstream `llama.cpp` checkout at
`6d9c82ea2bb34e277c0664b8dd3434bfb4dcfb27`. It ran on one NVIDIA B300 SXM6
AC through the native ggml CUDA backend. No custom CUDA kernel was added.

The retained valid samples are:

```text
0.387, 0.388, 0.388, 0.421, 0.396 ms
```

- Headline minimum: 0.387 ms
- Median: 0.388 ms
- Maximum: 0.421 ms
- IQR: 0.021 ms

One process performed five graph-compute warmups followed by five synchronized
timed graph computations. Inputs and weights were resident before each timing
interval. All eight B300 GPUs were idle and had no compute processes at the
valid timing preflight. An earlier five-sample run is retained but explicitly
named `timing.preliminary-concurrent-other-gpu.csv`, because another benchmark
process was resident on GPU 2 during that preflight.

## Exact scope

The graph uses the same dimensions and deterministic FP32 data as
`llama2_extended_forward_bench.c`: model dimension 4096, FFN dimension 11008,
vocabulary 32000, sequence length 2048, 32 heads, token 7, and position 1024.
It computes one transformer block, final RMS normalization, and the complete
32000-element vocabulary projection.

This is the exact fixture represented with upstream ggml operations and run by
the `llama.cpp` checkout's scheduler and CUDA libraries. It is not the standard
`llama.cpp` GGUF model loader's internal Llama graph: the fixture deliberately
uses split even/odd Q/K matrices, synthetic RoPE values, and preinitialized KV
caches, which cannot be expressed as an unmodified standard Llama GGUF layer.

## Correctness

All 32000 logits were compared.

- Against strict llama2.c native CPU reference: 0 failures at
  `atol=1e-2, rtol=1e-4`; maximum absolute difference 0.0046749.
- Against the previous Orin ggml CUDA output: 0 failures at the same tolerance;
  maximum absolute difference 0.0006295.

The small differences are consistent with floating-point reduction ordering
across CPU and CUDA implementations.

## Interpretation

The previous Orin ggml minimum was 9.659 ms for the identical graph, making the
B300 result 24.96x faster. The complete quantized 32-layer `llama.cpp` B300
generation measurement was 3.227806 ms/token, or 8.34x the runtime of this
one-layer FP32 fixture. That ratio is not a layer-scaling factor: the full model
uses Q4_K_M weights and a different standard Llama graph, while this fixture is
FP32 and includes the full vocabulary projection after its single layer.
