# Publication Orin campaign: ATAX

Current status: native and raised GPU correctness plus one-process 5+5 timing
complete.

- Canonical source: `tools/cgeist/Test/polybench/linear-algebra/kernels/atax/atax.c`
- Dataset/type: `LARGE_DATASET`, FP64
- Native computation: upstream PolyBenchGPU CUDA, commit
  `5584aaa7d0be810ff5eb0b61c49fb64ecc81ba4c`
- Integration: the retained normalization adapter changes dimensions, datatype,
  initialization/outputs, and timing glue but includes the pinned upstream CUDA
  computational kernels. This row is explicitly `modified_source=true`.
- Build placement: every AArch64/CUDA object and executable was cross-compiled
  on the x86 host; Orin only executed the finished binaries.
- Correctness: PASS, all 2,100 values exact against the canonical reference.
- Repetitions: one process, five warmups followed by five timed samples.
- Device median: 2.388704062 ms (min 2.385792017, max 2.391999960,
  IQR 0.003583908).
- End-to-end median: 8.836000 ms (min 8.818000, max 8.913000,
  IQR 0.076500).

Raised GPU uses the compiler-generated ATAX function and real cuBLAS DGEMV plus
transposed-DGEMV calls. A dedicated ABI harness supplies the canonical input
and invokes that function; it does not implement ATAX computation. Correctness
passes all 2,100 values exactly. Its five accepted samples are:

- Compute-device median: 0.594080 ms (min 0.592672, max 0.600512,
  IQR 0.005904).
- Synchronized resident compute-wall median: 0.586560 ms (min 0.582656,
  max 0.590848, IQR 0.005920).
- Compiler-visible memory-device median: 25.258208 ms.
- End-to-end median: 26.394144 ms (min 26.298912, max 27.107200,
  IQR 0.507824).
- Device/device speedup over native PolyBenchGPU: 4.0208x.

The raised timing uses stored automatically matched semantic IR and an ABI-only
harness; it is not described as an untouched whole-program transformation.

The runner reported accelerator status as `N/A`; therefore this measurement is
retained but not yet publication-accepted under the fixed-power/clock/fan rule.
Private board access details were removed from the retained artifacts.

## SHA-256

```
f31f15b65b19e7d779c21ea9ad5c017fbb51278a333d9b541b5fa34705525e9b  canonical atax.c
da4ae0c582726319803fd800c7e6f203de53de9df6caddbaa85f5489f0a29254  native normalization adapter
719ef186a8e60f669cdda167450b4a37d1726329d7dbce194e37b792f60c43ba  repetition driver
85f6314d49dc27570ac0eb819d37ab7061da3f819449a15605e24414b5047ee2  native correctness executable
a30727c8768983044f9b98484937f06d2d4040c9632157c62db2e61a05792951  native timing executable
0f444ea8b6df31f5c03f2c6141e88cd8f535ff6027b3bf93c305a896c0452126  raised matched IR
528f974fab068f9adab655c12f4922a762b0ef5b642327a28a7e34b26971f068  raised ABI harness
793117c2343065ddfa4aa3492515f0ed5bb39bb8f99c926497be55d489a8968c  raised timing executable
```
