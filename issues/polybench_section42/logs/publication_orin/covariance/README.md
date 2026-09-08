# Publication Orin campaign: covariance

Raised GPU correctness and one-process 5+5 timing are complete. Native GPU is
unavailable because the retained adapter defines three project-authored CUDA
computational kernels.

- Correctness: PASS, 1,440,000 values, `max_abs=0.01`,
  `max_rel=0.000203915171`.
- Compute-device median: 128.429794 ms (128.239166–129.067429,
  IQR 0.461983).
- Synchronized compute-wall median: 128.419264 ms.
- Memory-device median: 20.345312 ms.
- End-to-end median: 149.188416 ms (149.111296–150.606752,
  IQR 0.916176).

All compilation occurred on x86. No native ratio is reported. Hardware-state
reporting was unavailable, so timing remains publication-pending.
