# Publication Orin campaign: correlation

Raised GPU correctness and one-process 5+5 timing are complete. Native GPU is
unavailable: the retained adapter defines four project-authored computational
CUDA kernels and is ineligible for paper data.

- Raised correctness: exact pass for 1,440,000 values against the canonical
  external-library output.
- Compute-device median: 315.176270 ms (306.372864–318.835052,
  IQR 6.846878).
- Synchronized compute-wall median: 315.164928 ms.
- Memory-device median: 19.866400 ms.
- End-to-end median: 335.643712 ms (326.728160–339.232160,
  IQR 6.983264).

All compilation occurred on x86. No native speedup is reported. Fixed
hardware-state reporting was unavailable, so raised timing is retained but
publication-pending.
