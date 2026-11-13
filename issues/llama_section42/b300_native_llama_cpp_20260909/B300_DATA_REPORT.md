# B300 benchmark data report

Generated 2026-09-09 on the temporary Modal B300 system.

## Platform and limitations

- 8 x NVIDIA B300 SXM6 AC, compute capability 10.3, 275040 MiB each.
- NVIDIA driver 580.95.05; reported CUDA capability 13.1; Ubuntu 24.04 x86-64.
- 96 logical Intel Xeon Platinum 8559C CPUs and approximately 3.9 TiB RAM.
- GPU clock locking was denied inside the container even as root. Treat these as secondary, non-fixed-clock measurements.
- Headline values are minima after five warmups; median, maximum, IQR, raw logs, correctness output, and generated artifacts are retained.

## Coverage and correctness

- Native single-core PolyBench FP64/LARGE: pass=30 (30 kernels).
- Polygeist raised PolyBench FP64/LARGE: pass=15 (15 currently supported kernels).
- PolyBenchGPU native CUDA FP64/LARGE: correctness_failed=1, pass=12 (13 kernels).
- PPCG FP64/LARGE: build_failed=1, correctness_failed=1, pass=28 (30 attempted).
- PPCG FP32/LARGE: build_failed=1, correctness_failed=2, pass=27 (30 attempted).
- PolyBench-ACC artifact native CUDA FP32/STANDARD, after parser reruns: correctness_failed=1, pass=20 (21 kernels).
- Ginsbach exact computational regions: PASS=4 (4 CPU/GPU rows).
- Native `llama.cpp` CUDA, Llama-2-7B Q4_K_M: PASS=2 benchmark
  scopes plus one deterministic generation smoke test.
- Native `llama.cpp`/ggml CUDA, exact FP32 one-layer fixture: PASS=1 timing
  scope and 32000/32000-logit full-output validation.
- Polygeist raised FP32 one-layer fixture on B300: PASS=1 synchronized E2E
  timing scope and 32000/32000-logit full-output validation.

Known failures are retained, not filtered: PPCG FP64 Gram-Schmidt fails full-output comparison and Nussinov fails to build; PPCG FP32 Durbin and Gram-Schmidt fail correctness and Nussinov fails to build; PolyBenchGPU Gram-Schmidt fails correctness; PolyBench-ACC ADI fails correctness.

## Polygeist raised FP64/LARGE comparisons

CPU speedup uses native single-core kernel time / raised GPU end-to-end time. Native-GPU ratio uses native CUDA end-to-end time / raised GPU end-to-end time; values above 1 mean raised is faster. Device-only columns are intentionally not mixed because their scopes differ.

| Kernel | CPU native min ms | Raised GPU E2E min ms | Speedup vs CPU | Native GPU E2E min ms | Ratio vs native GPU |
|---|---:|---:|---:|---:|---:|
| 2mm | 770.983 | 20.874 | 36.94x | 8.471 | 0.41x |
| 3mm | 899.524 | 44.329 | 20.29x | 8.711 | 0.20x |
| atax | 2.495 | 2.063 | 1.21x | 3.026 | 1.47x |
| bicg | 1.983 | 2.255 | 0.88x | 3.259 | 1.45x |
| covariance | 5075.249 | 5931.516 | 0.86x | 171.699 | 0.03x |
| doitgen | 81.751 | 727.346 | 0.11x | 10.322 | 0.01x |
| gemm | 297.608 | 13.682 | 21.75x | 8.641 | 0.63x |
| gemver | 6.205 | 5.025 | 1.23x | 6.010 | 1.20x |
| gesummv | 0.925 | 0.865 | 1.07x | 2.711 | 3.14x |
| mvt | 5.160 | 2.019 | 2.56x | 3.881 | 1.92x |
| symm | 1542.294 | 325.542 | 4.74x | — | — |
| syr2k | 414.161 | 17.784 | 23.29x | — | — |
| syrk | 386.819 | 7.617 | 50.79x | — | — |
| trisolv | 1.099 | 10.051 | 0.11x | — | — |
| trmm | 1524.705 | 304.621 | 5.01x | — | — |

## Clean Ginsbach exact-region baselines

| Workload | Shape | Variant | Minimum ms | Median ms | Status |
|---|---|---|---:|---:|---|
| parboil_sgemm | 512x512x512 | native_cpu_single_core | 155.990388 | 156.188559 | PASS |
| parboil_sgemm | 512x512x512 | native_cuda | 0.106094 | 0.106575 | PASS |
| parboil_stencil | 128x128x128 | native_cpu_single_core | 18.181000 | 18.292270 | PASS |
| parboil_stencil | 128x128x128 | native_cuda | 0.118325 | 0.118641 | PASS |

These are benchmark/native target reproductions, not automatically raised Ginsbach outputs: the original author compiler is unavailable, and manually linked library adapters are not labeled as raised results.

## Native llama.cpp CUDA baseline

Upstream `llama.cpp` ran the complete Llama-2-7B Q4_K_M model on one B300
with all layers requested on CUDA, split mode `none`, and FlashAttention on.
Each row used one process, five full-scope warmups, and five timed repetitions;
the headline is the minimum. This full-model quantized baseline is not directly
comparable to the repository's FP32 one-layer fixture.

| Scope | Minimum | Median | Maximum | Throughput at minimum | Status |
|---|---:|---:|---:|---:|---|
| Prompt processing, 512 tokens | 80.313784 ms | 80.325402 ms | 80.331726 ms | 6375.000 tok/s | PASS |
| Token generation, 128 tokens | 413.159377 ms total / 3.227806 ms/token | 413.250730 ms | 413.473164 ms | 309.808 tok/s | PASS |

The generation smoke establishes a functional native full-model execution,
not cross-implementation numerical equivalence. `llama-bench` excludes model
load, tokenization, sampling, and surrounding interactive I/O.

## Native llama.cpp/ggml exact one-layer baseline

The repository's FP32 one-token, one-layer fixture was compiled against the
ggml scheduler and CUDA libraries from the pinned `llama.cpp` checkout. It
uses model dimension 4096, FFN dimension 11008, vocabulary 32000, sequence
length 2048, 32 heads, token 7, and position 1024. No custom CUDA kernel was
added.

| Scope | Minimum | Median | Maximum | IQR | Status |
|---|---:|---:|---:|---:|---|
| One FP32 block + final norm + 32000-logit projection | 0.387 ms | 0.388 ms | 0.421 ms | 0.021 ms | PASS |

All 32000 logits pass `atol=1e-2, rtol=1e-4` against both the llama2.c CPU
reference and the prior Orin ggml output. Maximum absolute differences are
0.0046749 and 0.0006295, respectively. The valid timing used one process,
five warmups, and five synchronized timed graph computations after an
all-device idle preflight. An earlier run with an unrelated process resident
on GPU 2 is retained and explicitly marked preliminary.

This graph exactly mirrors the extracted fixture using upstream ggml
operations, but is not the unmodified standard GGUF Llama graph builder: the
fixture has synthetic RoPE values and preinitialized KV caches.

## Polygeist raised exact one-layer fixture

The B300 host-CUDA build produces 22 external-library launches and retains 20
residual `linalg.generic` operations that lower to CPU loops. It passes the
declared implementation-neutral correctness criterion against both the native
llama2.c reference and the B300 ggml output.

| Scope | Minimum | Median | Maximum | IQR | Status |
|---|---:|---:|---:|---:|---|
| Synchronized per-call E2E | 331.101067 ms | 349.546811 ms | 450.875139 ms | 83.356859 ms | PASS |

This row must not be presented as resident device compute. The 855.56x gap
relative to the 0.387 ms ggml row is dominated by CPU residual execution,
discrete-GPU staging, and synchronization. It is evidence for the remaining
whole-pipeline residency problem, not a vendor-library kernel comparison.

## Timing-scope cautions

- Native CPU uses the PolyBench internal kernel timer and excludes allocation/initialization.
- Polygeist raised timing is a focused raised-function end-to-end scope with harness-owned host inputs and includes runtime staging. It is not a whole-program compiler result.
- PolyBenchGPU end-to-end time includes adapter allocation, transfers, kernel execution, and free. Its separate CUDA-event field is kernel-only.
- PPCG uses its stock whole-kernel-function timer (allocation + H2D + kernels + D2H + free) and fresh processes. It is not directly interchangeable with same-process resident timing.
- PolyBench-ACC uses the artifact device timer with fresh processes and the STANDARD dataset, so it is not a direct FP64/LARGE comparison.

## Provenance

- Polygeist: c40c9b7bf64 (LLVM submodule 26eb428).
- PolyBench-ACC: 70ea4ca.
- PPCG: 5e07c52.
- PolyBench/C: d3a15bb.
- PolyBenchGPU: 5584aaa.
- Ginsbach SNU NPB source checkout: 4f2aa1b.
- llama.cpp: 6d9c82ea2bb34e277c0664b8dd3434bfb4dcfb27.
- Llama-2-7B Q4_K_M model SHA-256:
  4567208c2221da5a9f2ded6cc26ce58dd47d0410902c3f57a4a3ed104ce51b0b.
- Remote-only host-CUDA driver patch: /mnt/forge1/arpitj/patches/polygeist-host-cuda.patch. This is not committed upstream support.

## Artifacts

- Master normalized CSV: /mnt/forge1/arpitj/results/b300-master-results.csv
- PolyBench comparison CSV: /mnt/forge1/arpitj/results/b300-polybench-comparison.csv
- Native CPU: /mnt/forge1/arpitj/results/polybench-native-cpu-large-fp64/summary.csv
- Polygeist raised: /mnt/forge1/arpitj/results/polygeist-polybench-large-fp64/summary.csv
- PolyBenchGPU native CUDA: /mnt/forge1/arpitj/results/polybenchgpu-native-large-fp64/summary.csv
- PPCG FP64: /mnt/forge1/arpitj/results/ppcg-polybench-large-fp64/summary.csv
- PPCG FP32: /mnt/forge1/arpitj/results/ppcg-polybench-large-fp32/summary.csv
- PolyBench-ACC original: /mnt/forge1/arpitj/results/polybench-acc-artifact-native-fp32-standard/summary.csv
- PolyBench-ACC corrected parser reruns: /mnt/forge1/arpitj/results/polybench-acc-artifact-native-fp32-standard-parser-rerun/summary.csv
- Ginsbach regions: /mnt/forge1/arpitj/results/ginsbach-b300/summary.csv
- Native llama.cpp CUDA: /mnt/forge1/arpitj/results/llama-cpp-native-b300/summary.csv
- Native llama.cpp retained bundle: /mnt/forge1/arpitj/artifacts/llama-cpp-native-b300-20260909.tar.gz
- Native llama.cpp/ggml exact one-layer result: /mnt/forge1/arpitj/results/llama-cpp-one-layer-b300/timing_valid.csv
- Polygeist raised exact one-layer result: /mnt/forge1/arpitj/results/llama-polygeist-b300/final_e2e_timing.csv
- Preliminary or invalid campaigns remain in sibling directories with `.preliminary-overlapped-setup` or `.invalid-cpu-constprop-bypass` suffixes and must not be used as headline results.
