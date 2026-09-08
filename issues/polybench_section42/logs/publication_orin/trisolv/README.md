# Publication Orin campaign: trisolv

Raised GPU correctness and one-process 5+5 timing are complete at canonical
LARGE/FP64. Real cuBLAS DTRSV passes the canonical output exactly (1,999 parsed
values; PolyBench concatenates the first printed value to its dump label).

- Compute-device median: 0.740096 ms (0.731968–0.744160,
  IQR 0.005120).
- Compute-wall median: 0.730784 ms.
- Memory-device median: 25.085408 ms.
- End-to-end median: 26.387872 ms (26.261824–27.637568,
  IQR 1.164320).
- Native GPU: unavailable; PolyBenchGPU has no Trisolv implementation.

The first comparison attempt used a noncanonical newline after the dump label
and was rejected as a 2,000-versus-1,999 formatting mismatch. The corrected
ABI-only harness exactly reproduces canonical formatting. All compilation
occurred on x86. Hardware-state reporting was unavailable, so timing remains
publication-pending.
