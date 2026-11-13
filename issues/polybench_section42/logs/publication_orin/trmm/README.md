# Publication Orin campaign: trmm

Raised GPU correctness and one-process 5+5 timing are complete at canonical
LARGE/FP64. The real-cuBLAS DTRMV/DSCAL composition passes all 1,200,000
values (`max_abs=0.01`, `max_rel=0.01`).

- Compute-device median: 236.725891 ms (228.869247–241.292709,
  IQR 2.554184).
- Compute-wall median: 236.716352 ms.
- Memory-device median: 14.514624 ms.
- End-to-end median: 251.838464 ms (243.798496–255.949248,
  IQR 2.431168).
- Native GPU: unavailable; PolyBenchGPU has no TRMM implementation.

All AArch64/CUDA compilation occurred on x86. Hardware-state reporting was
unavailable, so timing remains publication-pending.
