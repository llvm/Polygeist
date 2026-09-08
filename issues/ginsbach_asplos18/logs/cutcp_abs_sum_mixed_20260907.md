# CUTCP mixed-precision absolute-sum route — 2026-09-07

## Source and semantics

- Exact source: `third_party/gpu-parboil/benchmarks/cutcp/src/base/output.c`
- Function: `write_lattice_summary`
- Source expression: `abspotential += fabs((double) lattice_data[i]);`
- Raised retained IR:
  `/tmp/ginsbach_expected_matches_20260907/units/third_party_gpu-parboil_benchmarks_cutcp_src_base_output.c/linalg.mlir`
- Required semantics: FP32 input is extended to FP64 before absolute value and
  accumulated into a zero-seeded FP64 scalar. `cublasSasum` is not exact
  because its result is FP32.

## Implemented route

The isolated `audit/ginsbach-reductions` branch adds the generic semantic
pattern `Out(0) + Abs(In(0))` with an FP32 input memref and FP64 scalar-output
memref. It lowers through:

```text
kernel.launch @cubAbsSum_f32_f64_memref
  -> polygeist_cub_abs_sum_f32_f64
  -> CUB DeviceReduce::Sum over an FP32-to-FP64 absolute-value transform
```

The exact retained CUTCP IR produces one launch, one runtime call, and zero
residual launches after ABI lowering.

## Verification

- `test/polygeist-opt/kernel-match-cub-abs-sum-mixed.mlir`: PASS
- `test/polygeist-opt/lower-kernel-launch-cub-abs-sum-mixed.mlir`: PASS
- `test/runtime/cub-abs-sum-mixed-reference.c`: PASS
- AArch64 CUDA 12.6 `sm_87` companion and executable cross-compiled on x86.
- Orin #2 standalone ABI/runtime reference: PASS 3/3.
- Independently expected and observed result: `16.375`.
- Median host-observed runtime: `2.569824 ms`.
- Median CUDA-event device runtime: `2.117280 ms`.
- Retained raw silicon log:
  `issues/ginsbach_asplos18/logs/cutcp_abs_sum_mixed_20260907.silicon.log`

No source, MLIR, or CUDA code was compiled on the Jetson.

## Scope and claim discipline

This validates the exact source expression, generated ABI route, numerical
result, and Orin runtime independently. It is not yet a full CUTCP application
run. Together with the six previously validated `get_atom_extent` min/max
folds, Polygeist now has seven reconstructed CUTCP scalar sites. Their
correspondence to the seven CUTCP occurrences reported by Ginsbach et al. is a
high-confidence inference, not confirmation from the authors' matcher.
