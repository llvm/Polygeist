# Publication Orin CPU campaign: gesummv

Native and raised OpenBLAS CPU configurations both pass the complete
1,300-value canonical LARGE/FP64 output exactly.

- Native AArch64 GCC 11 `-O3`: 7.730 ms median (7.697–7.779 ms,
  IQR 0.028 ms).
- Raised real-OpenBLAS CBLAS: 2.006464 ms median
  (1.968448–2.018880 ms, IQR 0.044672 ms).
- Correctness-gated speedup: 3.852549x.

Each configuration ran in one process pinned to Orin CPU core 0 with five
warmups plus five samples and library threading fixed to one. All AArch64
compilation occurred on x86. Hardware-state reporting remains unavailable, so
these values are publication-pending.
