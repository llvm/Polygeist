# Publication Orin campaign: symm

Raised GPU correctness and one-process 5+5 timing are complete at canonical
LARGE/FP64. The equivalent real-cuBLAS DSYMV composition passes all 1,200,000
values (`max_abs=0.01`, `max_rel=0.00319488818`).

- Compute-device median: 96.199745 ms (95.481758–97.104156,
  IQR 0.507103).
- Compute-wall median: 96.189088 ms.
- Memory-device median: 22.040512 ms.
- End-to-end median: 118.612480 ms (118.257088–119.693408,
  IQR 0.495040).
- Native GPU: unavailable; PolyBenchGPU has no SYMM implementation.

All AArch64/CUDA compilation occurred on x86. Hardware-state reporting was
unavailable, so timing remains publication-pending.
