# Publication Orin campaign: gemver

Native and raised GPU correctness and one-process 5+5 timing are complete.
Both use canonical LARGE/FP64 inputs and check all 2,000 live-out `w` values.

- Native PolyBenchGPU: exact correctness; device median 2.646879911 ms
  (2.642623901–2.653984070, IQR 0.004927873); end-to-end median
  13.210 ms (13.197–13.396, IQR 0.055).
- Raised external-library path: correctness passes with
  `max_abs=18721.87`, `max_rel=7.85809757e-05`; compute-device median
  15.091168 ms (15.073344–15.107520, IQR 0.016992); synchronized
  compute-wall median 15.084064 ms; memory-device median 14.472064 ms;
  end-to-end median 30.173888 ms.
- Device/device ratio: 0.175393x (native is faster).

The first native correctness build is rejected: GCC reused an interprocedural
return-register fact from the source CPU definition after the external CUDA
call, causing a zero-length output dump. The accepted build adds the generic
`-fno-ipa-ra` interposition guard; its relocation and explicit post-call
`n=2000` materialization were audited before execution.

All AArch64/CUDA compilation occurred on x86. The native adapter contains the
verbatim upstream PolyBenchGPU kernels and is marked modified-source. The
raised path uses an ABI-only canonical initializer/driver and real cuBLAS plus
compiler-generated GPU code. Hardware-state reporting was unavailable, so the
values remain publication-pending.
