# Publication Orin campaign: cholesky

Raised GPU correctness and one-process 5+5 timing are complete at canonical
LARGE/FP64. The validated blocked external-library composition uses cuSOLVER
Xpotrf for 512-wide diagonal panels, real cuBLAS DTRSV for panel solves, and
real cuBLAS DGEMV for trailing updates. It passes the complete 2,001,000-value
canonical lower triangle with `max_abs=1.921796055626146e-13` and
`max_rel=1.921796055626146e-13`.

- Compute-device median: 520.483276 ms (520.298523–520.839355,
  IQR 0.395813).
- External-library-region host median: 525.572640 ms.
- End-to-end median: 530.254496 ms (530.020608–530.688032,
  IQR 0.393984).
- Native GPU: unavailable; PolyBenchGPU has no Cholesky implementation.

The timing harness performs the canonical initializer once, restores identical
input outside each measured call, discards five warmups, and records five
samples. The device event covers the full external-library composition. E2E
covers the raised call, including host/device staging and orchestration. All
AArch64/CUDA compilation occurred on x86. Hardware-state reporting was
unavailable, so timing remains publication-pending.

Artifact hashes:

- correctness executable: `d5178756606e2cfb4c372b703cdc44d750400112a2bc05278e79aa607d3a8010`
- timing executable: `71781dbfc58d93623f3e0ccfe16d04647e79f0c12d124a088c2cbee7417e0653`
- ABI IR: `9732c5adfe388af85eebb05b770a267b925235ba9835871ea44a7a51741be953`
- matched IR: `2c437556f366dd96470b6f0fb0d30a5beda798544c461f23947472c8823112b0`
- CUDA runtime source: `7f9da424d8fdc86995b8b2e758d2d28ccbdfb7f2d44826e41e91ec28031a4bfa`
- correctness harness: `1d6a2fe7e06b1c420d9721abdc3689e3c89a4dc1bec35c9cd329a96bfe6b89d1`
- timing harness: `f7792ae97f8d0ff6949670de52d37a34bd7cd414e464464c99f4357b927a319d`
