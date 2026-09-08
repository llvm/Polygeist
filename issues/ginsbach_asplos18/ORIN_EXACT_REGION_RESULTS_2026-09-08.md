# Ginsbach workloads: Orin exact-region measurements (2026-09-08)

These are exact-region measurements, not whole-application runtimes and not a
reproduction of all 60 opportunities reported by Ginsbach et al. They compare
the variants that are currently executable from the checked-out sources.

## Protocol

- NVIDIA Jetson AGX Orin Developer Kit, MAXN mode.
- Native CPU pinned to one core; binaries compiled with `-O3
  -mcpu=cortex-a78`.
- Variants were run sequentially with five warmups followed by five timed,
  synchronized samples.
- The table reports median, minimum, maximum, and IQR. For five samples, Q1
  and Q3 are the second and fourth sorted observations.
- Preflight and postflight found no competing CPU or GPU workload. CPU and GPU
  temperatures remained approximately 48.4--48.7 C and 42.7--43.0 C.
- Every executable variant passed complete-output correctness checks.
- The board was in MAXN, but `jetson_clocks` could not be locked because the
  benchmark account lacks passwordless sudo. These measurements are therefore
  clean comparison data but not final publication-grade fixed-clock data.

## Results

| Workload/region | Native CPU (ms) | Native GPU (ms) | Raised GPU (ms) | CPU / raised | Raised vs native GPU |
|---|---:|---:|---:|---:|---:|
| Parboil SGEMM, 512^3 | 492.369632 | 0.208352 | 0.150080 | 3280.71x | 1.39x faster |
| Parboil Stencil, 128^3 | 36.090848 | 0.278784 | 4.820064 | 7.49x | 17.29x slower |
| NPB CG core | 11.827264 | unavailable | 12.844128 | 0.92x | unavailable |
| NPB IS rank core | 0.237952 | unavailable | 0.384352 | 0.62x | unavailable |
| NPB UA DAXPBY | 1.129824 | unavailable | blocked | unavailable | unavailable |
| Parboil CUTCP atom extent | 1.165184 | unavailable | blocked | unavailable | unavailable |

The complete dispersion data is in
`orin_exact_region_results_2026-09-08.csv`.

## Interpretation and gaps

- SGEMM is the strongest result: semantic raising to cuBLAS is both much faster
  than single-core CPU and 1.39x faster than Parboil's shipped handwritten CUDA
  kernel for this region and shape.
- Stencil raising is correct and beats CPU, but cuDNN's route is 17.29x slower
  than Parboil's specialized CUDA stencil. This regression must remain visible.
- CG and IS are too small for the recovered library calls to amortize launch
  and runtime overhead; both are slower than the pinned native CPU region.
- No exact native-GPU implementation for the extracted NPB CG/IS regions was
  available in the current checkout, so a three-way comparison would require
  acquiring an authoritative baseline rather than creating a new one.
- UA reaches a DAXPBY library operation, but executable lowering currently
  mishandles the tensor-result ABI/writeback boundary.
- CUTCP currently fails before executable raising because its struct-pointer
  input becomes an unsupported dynamic memref load during lowering.
