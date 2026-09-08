# Llama protocol sweep: 2026-09-08

## Scope

- Workload: one FP32 token through one 7B-size extracted Llama layer.
- Shape: model 4096, FFN 11008, vocabulary 32000, sequence 2048, 32 heads,
  token 7, position 1024.
- Hardware: selected Jetson AGX Orin, non-secret campaign identity `orin2`.
- CPU affinity: core 3.
- Threading: BLAS/OpenMP thread counts fixed to 1.
- Board mode: MAXN; all four final timing preflights observed an idle CPU and
  0% integrated-GPU activity before execution.
- Protocol: one process, five warm-ups, five timed iterations.

## Outcome

The sweep executed all four requested configurations and retained all five
timing samples. Only the native reference and ggml expert baseline currently
pass their previously declared correctness roles. The two Polygeist rows are
excluded from paper-facing results because the fresh worktree output no longer
passes the strict `atol=1e-3, rtol=1e-4` Polygeist gate.

- Native CPU: median 272.887648001 ms, IQR 0.246399991 ms.
- Raised CPU/NVPL: median 562.344927996 ms, IQR 0.594240009 ms; 10,456 of
  32,000 logits fail the strict gate; maximum absolute error 0.0063534.
- Raised resident GPU: synchronized host-wall median 12.825504000 ms, IQR
  0.005952000 ms. CUDA-event median 12.813055992 ms, IQR 0.005631447 ms;
  7,922 logits fail the strict gate; maximum absolute error 0.0051174.
- ggml CUDA: median 9.691000000 ms, IQR 0.018000000 ms; all 32,000 logits
  pass the cross-implementation `atol=1e-2, rtol=1e-4` gate.

All three non-reference implementations pass at `atol=1e-2, rtol=1e-4`, but
that observation does not retroactively change the declared Polygeist gate.
The raised timings must remain excluded until the numerical difference is
explained or a tolerance change is justified before measurement.

## Important provenance qualification

The checkout was highly dirty during this sweep (1,146 porcelain entries), so
the Git commit alone does not identify the compiler state. The raised-CPU
pipeline also produced 18 `kernel.launch` operations and 39 runtime calls,
which differs from the earlier audited 11-NVPL-call configuration. This is a
second reason not to publish the raised rows yet.

The fresh raised-GPU pipeline currently fails while lowering a `gpu.alloc`
conversion. To isolate measurement from that unrelated regression, the sweep
reused the retained GPU-targeted object from the previously validated
automatic-residency build and rebuilt only the ABI/runtime/timing harness. The
timing-only LLVM edit changes the aggregate event interval into five individual
event intervals; it does not alter computational operations.

## Correctness evidence

Complete 32,000-logit outputs are retained under `full_outputs/`. Comparisons:

```text
raised CPU: failures=10456, max_abs=0.0063534, atol=0.001, rtol=0.0001
raised GPU: failures=7922,  max_abs=0.0051174, atol=0.001, rtol=0.0001
ggml CUDA:  failures=0,     max_abs=0.0051174, atol=0.01,  rtol=0.0001
```

## Final raw timing samples

```text
native CPU host ms: 272.636000000, 272.875168003, 272.887648001,
                    273.121567994, 273.208672002
raised CPU host ms: 559.572864003, 562.256511994, 562.344927996,
                    562.850752003, 563.431295998
raised GPU host ms: 12.862400000, 12.830656000, 12.824704000,
                    12.823840000, 12.825504000
raised GPU event ms: 12.836288452, 12.816351891, 12.810720444,
                     12.809760094, 12.813055992
ggml graph wall ms: 9.707000000, 9.659000000, 9.720000000,
                    9.689000000, 9.691000000
```

## Build hashes

```text
Git HEAD: 90dac2bc06a1abbe187f39090dfaf584eab37d67
native CPU executable: 9472e05b492b38d6a282840e2ab42e769df41d1d3956a8c18b00167eda57b0ee
raised CPU executable: 118336d85c116201b813461d9146f867ac76e7acdcd6e2291ed6cce3825fa8be
raised GPU executable: 91eac7e5dd114474d8dd4abbda872459d89608d0995b9946488351793ad703a8
ggml CUDA executable: 883460a1faf767bfb5c6d708fda9eab4c7536056296669cdbd0ed32054d14e9d
fixture source: 8959d283d2fd03fcbc0d2f5c3cf6c7a046ee633f47f70fb62d35625f5cac9af2
polygeist-opt: 77d7c31bf1ed1e207104b2a5865b0703e91f70e4183d125b42b900cbf2844789
mlir-opt: 5d324a26e89d7ecc611991b85ef5cc94c2b2c5be1ea19f44ca99c64b841d3574
```

## Next action

Diff the fresh raised CPU matched/ABI IR against the earlier proof-gated
configuration, identify why the launch/runtime-call count changed, and locate
the first intermediate tensor whose full values diverge. Separately repair the
current `gpu.alloc` lowering failure so the raised GPU can be regenerated from
source rather than a retained compiler object. Rerun the one-process 5+5 sweep
only after both raised paths pass the declared correctness gate.
