# MFEM normalized-kernel library matching

- kernels: 20
- matcher successes: 20
- kernels with at least one match: 18
- matched stage groups: 35
- emitted kernel.launch operations: 35

Matches are stage-level unless a report explicitly names a whole composition.


## FP64 contraction lowering

- 128 matches are ABI-legal two-input FP64 contractions:
  - 64 rank `4 x 5 -> 4`
  - 4 rank `5 x 4 -> 4`
  - 20 rank `5 x 5 -> 4`
- 40 iterator/rank-generic launches, comprising all 36 2D contraction stages
  plus 4 3D stages whose physical output views compact a broadcast mode
- All 128 lower to `polygeist_cutensornet_contraction2_f64`, with the original
  affine indexing maps and physical `polygeist.submap` strides encoded as
  extent/stride/mode metadata.
- A reduction dimension may occur in the logical output map only when its
  physical `polygeist.submap` stride is proven zero; ABI lowering then omits
  that broadcast mode from the output descriptor.
- The remaining 6 emitted launches are older structural matches (5
  `cublasDaxpby`, 1 `cudnnAddTensor_batched`) and are not included in the
  cuTensorNet lowering count.

Host compilation, focused pass tests, and the CPU reference contraction test
pass.

## Silicon validation

On 2026-07-24, all three compiler-generated FP64 variants ran through
cuTensorNet on an aarch64 Tegra target:

- `r4 x r5 -> r4`: `max_error=0`
- `r5 x r4 -> r4`: `max_error=0`
- `r5 x r5 -> r4` with compacted broadcast modes: `max_error=0`

The first call paid cuTensorNet/CUDA initialization and planning cost. The
subsequent two small contractions took about `0.085-0.088 ms` of device time.
See `../silicon_results/2026-07-24_cutensornet_variants.log`.
