# Publication Orin campaign: 2mm

Native and raised GPU correctness and one-process 5+5 timing are complete.
Both use canonical LARGE/FP64 inputs and complete 960,000-value output checks.

- Native PolyBenchGPU: exact correctness; device median 71.743362427 ms
  (71.729347229–71.761245728, IQR 0.021183014); end-to-end median
  79.172 ms (79.135–79.232, IQR 0.066).
- Raised real-cuBLAS path: correctness passes with `max_abs=0.01` and
  `max_rel=2.19043655e-05`; compute-device median 94.298943 ms
  (94.211586–94.324959, IQR 0.0921595); synchronized compute-wall median
  94.289536 ms; memory-device median 30.307904 ms; end-to-end median
  125.136384 ms.
- Device/device ratio: 0.760807x (native is faster).

The first native timing attempt is rejected because compiler IPA cloned the
source CPU kernel and produced no CUDA timing record. The retained timing was
rebuilt with no-IPA guards and a relocation audit proving the harness calls the
strong upstream-CUDA entry.

All AArch64/CUDA compilation occurred on the x86 host. The native baseline is
explicitly `modified_source=true`; the raised path uses an ABI-only harness and
is not described as untouched whole-program transformation. Hardware-state
reporting remained unavailable, so these values are publication-pending.
