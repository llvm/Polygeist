# Publication Orin campaign: gesummv

Native and raised GPU correctness and one-process 5+5 timing are complete at
canonical LARGE/FP64. Both pass all 1,300 live-out values exactly.

- Native PolyBenchGPU: device median 0.883328021 ms
  (0.872575998–0.888448000, IQR 0.014944017); end-to-end median
  6.792 ms (6.724–6.888, IQR 0.055).
- Raised external-library path: compute-device median 0.445472 ms
  (0.440224–0.448512, IQR 0.005408); synchronized compute-wall median
  0.437952 ms; memory-device median 12.289088 ms; end-to-end median
  13.283392 ms (13.152288–13.615808, IQR 0.090816).
- Device/device speedup: 1.982904x.

All AArch64/CUDA compilation occurred on x86. The native adapter contains the
verbatim upstream PolyBenchGPU kernel and is marked modified-source. The raised
path uses real cuBLAS and compiler-generated GPU code. Hardware-state reporting
was unavailable, so these values remain publication-pending.
