# Publication Orin campaign: BICG

Current status: native and raised GPU correctness plus one-process 5+5 timing
complete.

- Canonical source: `tools/cgeist/Test/polybench/linear-algebra/kernels/bicg/bicg.c`
- Dataset/type: `LARGE_DATASET`, FP64
- Native computation: upstream PolyBenchGPU CUDA commit
  `5584aaa7d0be810ff5eb0b61c49fb64ecc81ba4c`
- Integration: modified-source normalization adapter retaining the upstream
  CUDA computational kernels (`modified_source=true`).
- Correctness: PASS, 4,000 values, `max_abs=0.01`,
  `max_rel=2.53890878e-05`, `rtol=5e-4`, `atol=1.1e-2`.
- Repetitions: one process, five warmups then five samples.
- Device median: 2.218847990 ms (min 2.211424112, max 2.230207920,
  IQR 0.012863875).
- End-to-end median: 8.872000 ms (min 8.820000, max 8.931000,
  IQR 0.067000).

All AArch64 and CUDA compilation ran on the x86 host. Private board access
details are excluded. The runner returned no fixed hardware-state record, so
this measurement is retained but publication-pending.

Raised GPU uses compiler-generated real-cuBLAS transposed-DGEMV plus DGEMV
calls. The ABI-only harness initializes the canonical inputs and invokes the
generated function; it contains no BICG computation. Correctness passes all
4,000 values (`max_abs=0.01`, `max_rel=2.54045677e-05`). Its results are:

- Compute-device median: 0.874336 ms (min 0.576736, max 0.926144,
  IQR 0.321024).
- Synchronized resident compute-wall median: 0.584928 ms (min 0.568928,
  max 0.914688, IQR 0.175360).
- Compiler-visible memory-device median: 22.965504 ms.
- End-to-end median: 23.978720 ms (min 23.619328, max 24.600704,
  IQR 0.789136).
- Device/device speedup over native PolyBenchGPU: 2.537752x.

The five raised samples have materially higher compute variability than ATAX;
the complete range is retained and no sample was silently replaced.

## SHA-256

```
6aaa2ce6e5eac66feaedbf79f2b6d08c8d7ae92b998a71eed96969a0f75fd611  canonical bicg.c
eb1ce2d1a538c5a1cf9afb527457e336eab3ccfa08ce7d3f9d4778a75b9f6d60  native normalization adapter
719ef186a8e60f669cdda167450b4a37d1726329d7dbce194e37b792f60c43ba  repetition driver
ff48e7b3785272b78c25dc384842be58df7a381671f145d90943146b0283b0aa  native correctness executable
ca68b459b42c16cbdd9b18675d844db1d29cce2693435bcdf004f155e8ea9a6a  native timing executable
f538f73d35f0b8453aa211d7724b09f660d196b4189a01ffd414514d655cfed0  raised matched IR
8fad3e3ebc5faeb2816c5035fddf7e28aed2f8d36473d0d83062925d24018b76  raised ABI harness
11b93c2ecd105007b466aec1459a60e07d67e741001932b817901a0730dd4aec  raised timing executable
```
