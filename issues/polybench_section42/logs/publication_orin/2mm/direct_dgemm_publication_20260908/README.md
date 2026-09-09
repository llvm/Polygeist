# Fresh direct-DGEMM publication rerun

Canonical PolyBench/C LARGE/FP64 `2mm`, raised and matched to two real cuBLAS
DGEMM calls. Both binaries were cross-compiled on the x86 host for Orin
`sm_87`; no compilation occurred on the Jetson.

Complete-output correctness passes all 960,000 values (`max_abs=0.01`,
`max_rel=5.52852566e-08`, `rtol=5e-4`, `atol=1.1e-2`). Timing used one
process, five untimed warmups, and five synchronized samples. The paper
headline is the minimum synchronized resident compute wall time: 48.909536 ms.
The minimum CUDA-event compute time is 48.919712 ms. Full samples are in the
parent `raised-gpu-samples.csv`; the prior stale sample file is retained here
as `superseded-raised-gpu-samples.csv`.

Fixed Orin power/clock/fan state could not be verified by the unprivileged
runner, so the result remains publication-pending.
