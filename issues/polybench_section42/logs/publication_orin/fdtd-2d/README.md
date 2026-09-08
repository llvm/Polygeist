# Publication Orin campaign: fdtd-2d

Native GPU correctness and one-process 5+5 timing are complete at canonical
LARGE/FP64. The normalized upstream PolyBenchGPU path passes all 1,200,000
live-out values (`max_abs=0.01`, `max_rel=0.000755857899`).

- Native device median: 261.527954102 ms
  (260.833129883–261.872131348, IQR 0.482513427).
- Native end-to-end median: 271.517 ms
  (270.730–271.947, IQR 0.575).
- Raised GPU: unavailable; the compiler has no eligible external-library
  computational match for the FDTD stencil sequence. Its residual CPU path is
  a separate result and is not substituted here.

All AArch64/CUDA compilation occurred on x86. Native includes the upstream
PolyBenchGPU source through a normalization wrapper and is marked
modified-source. Hardware-state reporting was unavailable, so timing remains
publication-pending.
