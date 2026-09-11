# 2mm direct-DGEMM rerun (2026-09-08)

Status: **initial run rejected; unchanged-binary rerun passes; performance rerun still pending**.

The fresh LARGE/FP64 source-to-matcher build was cross-compiled on the x86 host for AArch64/sm_87 and emitted the expected external-library launches.  On Orin, all 960,000 final-output values failed (`max_abs=266862.87`, `max_rel=1`; first value reference `419.21`, candidate `0`).  Rebuilding the previously retained semantic matched IR with the current runtime failed identically, so the failure is not explained by the fresh matcher's result-destination attributes.  A device-resident control also failed identically.

The device-resident run's runtime trace observed two direct `cublasDgemm` calls, for `(m,n,k)=(800,900,1100)` and `(800,1200,900)`.  Its reported timing is diagnostic only because correctness failed; it must not be copied into a performance table.

A follow-up isolated those calls from the generated program.  Raw cuBLAS passes both exact shapes as a dependent chain, and the current `polygeist_cublas_dgemm` wrapper also passes the chain when both calls share one pipeline scope.  Relative errors are `1.4388462296676626e-15` and `1.6159480704195366e-16`.

After those probes exercised the CUDA stack, the exact same previously failing resident executable was deployed again without rebuilding and passed all 960,000 outputs (`max_abs=0.01`, `max_rel=5.52852566e-08`).  Its compute work changed from the failed run's implausible 2.836 ms/no-update behavior to about 78 ms with correct updates.  This retracts the initial compiler-buffer diagnosis: the evidence instead indicates transient cold/stale CUDA driver or cuBLAS execution state.  No competing process was present.  The precise vendor-runtime trigger is not yet established, so no new performance value is accepted until the warm-up/preflight behavior is made repeatable.

Evidence:

- `raised-correctness-build.log`: complete fresh cross-build log.
- `raised-correctness.silicon.log.gz` and `raised-correctness.compare.log`: fresh Orin output and comparison.
- `stored-raised-correctness-*`: control built from the retained semantic IR.
- `resident-correctness-*`: device-resident control.
- `resident-runtime-timing-diagnostic.log`: rejected diagnostic trace.
- `ir/`: fresh ABI, matched, and residency-transformed IR.
- `build-provenance.sha256`: retained artifact hashes.
- `direct-dgemm-isolation.log`: exact-shape raw-cuBLAS and runtime-wrapper probe results.

Execution used the local `pva-general` transport profile.  Private route and credential material are intentionally not retained in the repository.
