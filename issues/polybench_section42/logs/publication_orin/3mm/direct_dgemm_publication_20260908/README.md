# Fresh direct-DGEMM publication rerun

Canonical PolyBench/C LARGE/FP64 `3mm`, raised and matched to three real cuBLAS
DGEMM calls. The generic build driver retried Cgeist with translation-only
`-Dstatic=` after verifying that explicit selection had omitted the static
`kernel_3mm`. Both binaries were cross-compiled on the x86 host for Orin
`sm_87`; no compilation occurred on the Jetson.

Complete-output correctness passes all 880,000 values exactly. Timing used one
process, five untimed warmups, and five synchronized samples. The paper
headline is the minimum synchronized resident compute wall time: 82.504160 ms.
The minimum CUDA-event compute time is 82.513443 ms. Full samples are in the
parent `raised-gpu-samples.csv`; the prior stale sample file is retained here
as `superseded-raised-gpu-samples.csv`.

Fixed Orin power/clock/fan state could not be verified by the unprivileged
runner, so the result remains publication-pending.
