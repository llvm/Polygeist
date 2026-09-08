# Publication Orin campaign: syr2k

Native and raised GPU correctness and one-process 5+5 timing are complete at
canonical LARGE/FP64. Both pass all 1,440,000 values under the canonical
lower-triangle output contract.

- Native normalized PolyBenchGPU: correctness `max_abs=0.01`,
  `max_rel=1.36584033e-05`; device median 207.753952026 ms
  (207.734329224–207.771163940, IQR 0.023132325); end-to-end median
  216.215 ms (216.179–216.241, IQR 0.037).
- Raised real-cuBLAS DSYR2K: correctness `max_abs=0.01`,
  `max_rel=1.47073963e-05`; compute-device median 41.871967 ms
  (41.853855–41.893250, IQR 0.012192); submission-wall median
  0.141184 ms; memory-device median 13.884128 ms; end-to-end median
  56.270304 ms (56.125312–56.428832, IQR 0.218976).
- Device/device speedup: 4.961648x.

The native kernel computes the full matrix, while both paths are checked on
the complete benchmark output with the untouched upper triangle preserved.
All AArch64/CUDA compilation occurred on x86. Hardware-state reporting was
unavailable, so timing remains publication-pending.
