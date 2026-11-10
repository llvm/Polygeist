# Native llama.cpp CUDA on NVIDIA B300 (2026-09-09)

## Scope

- Implementation: upstream `llama.cpp` CUDA backend at commit
  `6d9c82ea2bb34e277c0664b8dd3434bfb4dcfb27`.
- Model: Llama-2-7B GGUF, Q4_K_M, 6,738,415,616 parameters.
- Model SHA-256:
  `4567208c2221da5a9f2ded6cc26ce58dd47d0410902c3f57a4a3ed104ce51b0b`.
- Device: one NVIDIA B300 SXM6 AC (`CUDA_VISIBLE_DEVICES=0`), with all
  32 model layers requested on the CUDA device (`-ngl 999`, split mode
  `none`) and FlashAttention enabled.
- This is a native, full-model `llama.cpp` measurement. It is not directly
  comparable to the repository's FP32 one-layer fixture because this run uses
  a quantized Q4_K_M model and a different measurement scope.

## Protocol

Each scope ran in one process. A measurement-only patch changed
`llama-bench`'s warmup loop from one warmup to five full-scope warmups; the
model implementation and CUDA kernels were not changed. The five warmups were
followed by the tool's five synchronized timed repetitions. The minimum is the
headline value, with all five samples retained. GPU 0 was idle immediately
before each process (0% utilization and 4 MiB allocated). B300 fixed clocks
were unavailable, so this is a secondary non-fixed-clock result.

## Results

| Scope | Headline minimum | Median | Maximum | IQR | Throughput at minimum |
|---|---:|---:|---:|---:|---:|
| Prompt processing, 512 tokens | 80.313784 ms | 80.325402 ms | 80.331726 ms | 0.016872 ms | 6375.000 tok/s |
| Token generation, 128 tokens | 413.159377 ms total / 3.227806 ms/token | 413.250730 ms | 413.473164 ms | 0.227302 ms | 309.808 tok/s |

Prompt-processing samples (ms):
`80.329968, 80.325402, 80.313784, 80.314166, 80.331726`.

Generation samples for 128 tokens (ms):
`413.473164, 413.250730, 413.159377, 413.364935, 413.224118`.

## Correctness and interpretation

`llama-cli` successfully loaded the complete model through the CUDA backend
and completed a deterministic generation smoke test. This establishes a
functional native full-model run, not cross-implementation numerical
equivalence. `llama-bench` measures full-model prompt processing and token
generation; it does not include tokenization, sampling, model load, or an
interactive application's surrounding I/O.

The result bundle retains the raw JSON, stderr/progress logs, idle preflights,
model and executable hashes, source/environment provenance, normalized sample
CSV, summary CSV, and the exact benchmark instrumentation diff.
