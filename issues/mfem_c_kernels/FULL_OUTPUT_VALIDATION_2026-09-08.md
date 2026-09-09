# MFEM NE=1024 full-output validation — 2026-09-08

## Result

The live 18-kernel MFEM matcher corpus was regenerated from normalized C at
`MFEM_BENCH_NE=1024`, cross-compiled on the x86 host, and executed on Orin #2.
The validator compares every FP64 output element against a separately compiled
copy of the normalized C source; it does not use checksums.

- 128 fresh static `kernel.launch` contraction sites were recovered.
- 18/18 matched kernels built and passed complete-output comparison.
- 128/128 sites are therefore lowered, built, executed, and elementwise
  validated in their complete operator pipelines.
- 1,305,600 output elements were compared across those 18 kernels.
- Worst absolute error was `5.5511151231257827e-17`.
- Tolerance was `abs_error <= 1e-11 + 1e-10 * abs(reference)`.
- Inputs, basis/gradient matrices, operators, and initial output buffers were
  nonconstant deterministic values.

The final two sites in `mfem_integrate_value_3d_scratch_sliced` initially
failed even with matching disabled: all 65,536 outputs differed from the
reference (`max_abs=0.0051067106369118986`). The cause was a tensor
`submapInverse` permutation `(d0,d1,d2,d3)->(d0,d3,d1,d2)` being lowered as
`tensor.insert_slice`, which cannot transpose its source dimensions. That path
also hid the incompatible static shapes behind a dynamic tensor cast.

`LowerPolygeistSubmap` now routes non-identity full-rank permutations through
the exact affine elementwise writeback. A focused FileCheck regression passes.
Fresh host control and matched pipelines both pass, and the x86-cross-compiled
AArch64 matched executable passes on Orin with both expected
`cutensornetContraction2_f64` launches. The complete-output result is 0 failing
elements and `max_abs=5.5511151231257827e-17`. The retained machine-readable
record and raw Orin log are in
`section42_campaign/postfix_full_output_validation_20260908.csv` and
`section42_campaign/integrate_value_3d_fixed_orin.log`.

## Network-composition finding

The default cuTensorNet network-composition optimization is not correctness-safe
for Mass3D in this configuration. The composed five-match pipeline failed all
65,536 outputs (`max_abs=2.1567120620263891e-05`). Prefixes containing zero
through four matches all passed, and a fresh pairwise five-match build with
`POLYGEIST_COMPOSE_CUTENSORNET_NETWORKS=0` passed with maximum absolute error
`5.5511151231257827e-17`. Therefore the 126-site validation used pairwise
lowering and the composed Mass3D result must not be used.

## Per-kernel complete-output results

All rows below passed with zero failing elements:

```
integrate_grad_2d   sites=4  elements=16384  max_abs=0
integrate_grad_3d   sites=9  elements=65536  max_abs=0
integrate_value_2d  sites=1  elements=16384  max_abs=5.5511151231257827e-17
integrate_value_3d  sites=2  elements=65536  max_abs=5.5511151231257827e-17
convection_2d       sites=5  elements=16384  max_abs=3.4694469519536142e-18
convection_3d       sites=10 elements=65536  max_abs=3.4694469519536142e-18
curlcurl_2d         sites=5  elements=24576  max_abs=4.3368086899420177e-19
curlcurl_3d         sites=29 elements=147456 max_abs=1.3877787807814457e-17
diffusion_2d        sites=6  elements=16384  max_abs=1.3877787807814457e-17
diffusion_3d        sites=14 elements=65536  max_abs=6.9388939039072284e-18
divdiv_2d           sites=6  elements=24576  max_abs=8.6736173798840355e-19
divdiv_3d           sites=12 elements=110592 max_abs=5.5511151231257827e-17
interp_grad_2d      sites=4  elements=51200  max_abs=0
interp_grad_3d      sites=8  elements=384000 max_abs=0
interp_value_2d     sites=2  elements=25600  max_abs=0
interp_value_3d     sites=3  elements=128000 max_abs=0
mass_2d             sites=3  elements=16384  max_abs=5.5511151231257827e-17
mass_3d             sites=5  elements=65536  max_abs=5.5511151231257827e-17
```

## Reproduction details

Validator: `benchmarks/mfem_full_output_validation.c`

Each kernel was built with `scripts/correctness/polygeist_build.sh`, target
`jetson`, `-DMFEM_BENCH_NE=1024`, its corresponding `BENCH_*` macro, and:

```
POLYGEIST_COMPOSE_CUTENSORNET_NETWORKS=0
POLYGEIST_MINIMAL_CUTENSORNET_RUNTIME=1
POLYGEIST_CUTENSORNET_ROOT=/home/arjaiswal/cutensornet-aarch64-root/cuquantum
```

Retained host work directories and binaries:
`/home/arjaiswal/mfem-ne1024-validation/all`

Orin deployment directory:
`/home/nvidia/polygeist-results/mfem-full-output-ne1024-pairwise`
