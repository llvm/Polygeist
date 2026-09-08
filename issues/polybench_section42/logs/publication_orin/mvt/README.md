# Publication Orin campaign: mvt

Native and raised GPU correctness and one-process 5+5 timing are complete at
canonical LARGE/FP64.

- Native PolyBenchGPU: correctness passes all 4,000 values
  (`max_abs=0.01`, `max_rel=2.1360675e-05`); device median 2.137023926 ms
  (2.130847931–2.141664028, IQR 0.001056194); end-to-end median
  8.907 ms (8.884–8.968, IQR 0.025).
- Raised external-library path: correctness passes all 4,000 values
  (`max_abs=0.01`, `max_rel=2.28242759e-05`); compute-device median
  0.977696 ms (0.950624–0.999584, IQR 0.017440); synchronized
  compute-wall median 0.793152 ms; memory-device median 13.800864 ms;
  end-to-end median 15.361344 ms (15.237504–15.470944, IQR 0.084512).
- Device/device speedup: 2.185775x.

All AArch64/CUDA compilation occurred on x86. Native includes the upstream
PolyBenchGPU source through a normalization wrapper and is marked
modified-source. The raised path uses real cuBLAS and compiler-generated GPU
code. The rejected first local build used an obsolete include root and never
ran. Hardware-state reporting was unavailable, so timing remains
publication-pending.
