# Publication Orin campaign: syrk

Native and raised GPU correctness and one-process 5+5 timing are complete at
canonical LARGE/FP64. Both pass all 1,440,000 values exactly under the
canonical lower-triangle output contract.

- Native normalized PolyBenchGPU: device median 83.418144226 ms
  (83.409149170–83.438400269, IQR 0.007072449); end-to-end median
  90.033 ms (89.991–90.066, IQR 0.015).
- Raised real-cuBLAS DSYRK: compute-device median 22.748960 ms
  (22.700001–22.763872, IQR 0.046688); submission-wall median
  0.110944 ms; memory-device median 10.520000 ms; end-to-end median
  33.770720 ms (33.215840–33.812128, IQR 0.206912).
- Device/device speedup: 3.666899x.

The native kernel computes the full matrix, while both paths are checked on
the complete benchmark output with the untouched upper triangle preserved.
All AArch64/CUDA compilation occurred on x86. Hardware-state reporting was
unavailable, so timing remains publication-pending.
