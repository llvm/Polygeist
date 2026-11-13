# Latest Proxy Kernel Extraction Results

Run command:

```sh
issues/proxy_kernel_extractions/run_proxy_kernel_extractions.sh
```

Latest output directory:

```sh
/tmp/proxy_kernel_extractions_mlir
```

## Summary

- Total standalone probes: 85
- `cgeist` failures: 0
- Tensor-form Linalg: 82
- Loop-free memref-form Linalg: 2
- Memref-form Linalg with residual loops: 1
- No Linalg, but raise completed: 0
- Raise failure: 0
- Kernels with residual loops: 1
- Kernels with residual ifs: 0

Project breakdown:

- `miniAMR`: 13 total, 12 tensor-Linalg, 1 memref-Linalg with residual loops, 0 no-Linalg, 0 raise failures.
- `HPGMG`: 29 total, 27 tensor-Linalg, 2 loop-free memref-Linalg, 0 no-Linalg, 0 raise failures.
- `HyPar`: 28 total, 28 tensor-Linalg, 0 no-Linalg, 0 raise failures.
- `SWFFT`: 6 total, 6 tensor-Linalg, 0 no-Linalg, 0 raise failures.
- `ExaSP2`: 9 total, 9 tensor-Linalg, 0 no-Linalg, 0 raise failures.

## Useful Successes

- `miniAMR` isolated stencils, material updates, and halo pack/unpack mostly
  raise cleanly to tensor Linalg. The only partial result is
  `miniamr_stencil_calc_27`, which raises to memref Linalg but leaves the
  explicit 3x3x3 accumulation loops.
- `HPGMG` 7-point/27-point apply, residual, Jacobi and red-black smoothers,
  BLAS1 kernels, reductions, restriction, interpolation, FV flux extracts, and
  CG/BiCGSTAB multi-vector updates raise.
- `HyPar` first/second/fourth finite differences, central/upwind
  reconstruction, WENO reconstruction and weights, limiters, LinearADR pointwise
  kernels, Burgers kernels, Euler fluxes, and LLF upwind fluxes raise.
- `SWFFT` local redistribution pack/unpack, slab copy, and transpose kernels
  raise cleanly. This separates local layout movement from the full MPI-heavy
  SWFFT application.
- `ExaSP2` dense normalization, dense square, SP2 update/select, trace, AXPBY,
  SpMV, and CG-step extracts raise to tensor Linalg.

## Current Partial Cases

- `miniamr_stencil_calc_27`: raises to memref Linalg but leaves the explicit
  small 3x3x3 accumulation loops. This is now the only residual-loop case.
- `hpgmg_interpolation_p1`: raises to loop-free memref Linalg. It uses
  parity-dependent coarse-grid indexing and dynamic `memref.load` operations
  inside the Linalg payload, so the current debufferizer does not convert it to
  tensor form.
- `hpgmg_interpolation_p2`: raises to loop-free memref Linalg for the same
  hybrid-payload reason.

## Interpretation

The isolated results are strong for the paper narrative: once the application
ABI, MPI, BML, and solver structs are removed, most regular compute and layout
kernels from these proxy apps raise to Linalg. The remaining cases point to
specific next compiler/matcher work: tensorizing hybrid Linalg bodies that keep
dynamic memref payload loads, and composing nested fixed-size stencil reductions
such as the explicit 27-point accumulation in `miniamr_stencil_calc_27`.
