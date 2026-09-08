# Publication Orin CPU campaign: bicg

All three CPU configurations pass the complete 4,000-value canonical
LARGE/FP64 output. Native uses AArch64 GCC 11 `-O3`; residual is
compiler-generated scalar CPU code; the library path contains compiler-
generated calls to real AArch64 OpenBLAS CBLAS.

- Native CPU median: 18.101 ms (18.095–18.138 ms, IQR 0.024 ms).
- Raised residual median: 6.131936 ms (6.077728–6.146016 ms,
  IQR 0.041376 ms), 2.951923x versus native.
- Raised OpenBLAS median: 4.870880 ms (4.821376–4.923840 ms,
  IQR 0.014912 ms), 3.716166x versus native.

Each configuration ran in one process pinned to Orin CPU core 0 with five
warmups plus five samples and BLAS/OpenMP threading fixed to one. All AArch64
compilation occurred on x86. Hardware-state reporting remains unavailable, so
the timing is publication-pending.
