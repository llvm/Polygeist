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
- Tensor-form Linalg: 68
- Memref-form Linalg: 2
- No Linalg, but raise completed: 14
- Raise failure: 1
- Kernels with residual loops: 15
- Kernels with residual ifs: 6

Project breakdown:

- `miniAMR`: 13 total, 12 tensor-Linalg, 1 memref-Linalg, 0 no-Linalg, 0 raise failures.
- `HPGMG`: 29 total, 24 tensor-Linalg, 1 memref-Linalg, 4 no-Linalg, 0 raise failures.
- `HyPar`: 28 total, 20 tensor-Linalg, 0 memref-Linalg, 8 no-Linalg, 0 raise failures.
- `SWFFT`: 6 total, 6 tensor-Linalg, 0 no-Linalg, 0 raise failures.
- `ExaSP2`: 9 total, 6 tensor-Linalg, 2 no-Linalg, 1 raise failure.

## Useful Successes

- `miniAMR` isolated stencils, material updates, and halo pack/unpack mostly
  raise cleanly to tensor Linalg. The only partial result is
  `miniamr_stencil_calc_27`, which raises to memref Linalg but leaves the
  explicit 3x3x3 accumulation loops.
- `HPGMG` 7-point/27-point apply, residual, Jacobi smoother, BLAS1 kernels,
  reductions, restriction, p0/p2 interpolation, and FV flux extracts raise.
- `HyPar` first/second/fourth finite differences, central reconstruction,
  WENO reconstruction, limiters, LinearADR pointwise kernels, Burgers kernels,
  and LLF upwind fluxes raise.
- `SWFFT` local redistribution pack/unpack, slab copy, and transpose kernels
  raise cleanly. This separates local layout movement from the full MPI-heavy
  SWFFT application.
- `ExaSP2` dense square, SP2 update, trace, AXPBY, SpMV, and CG-step extracts
  either raise or identify clear multi-store composition limits.

## Current Non-Linalg / Failure Cases

- `hpgmg_gsrb_smooth_7pt`: branchy red-black update leaves loops/if.
- `hpgmg_interpolation_p1`: parity-dependent indexing leaves loops.
- `hpgmg_cg_update`, `hpgmg_bicgstab_update`: multiple stores to different
  vectors in one loop are not raised as one Linalg op.
- `hypar_interp_first_order_upwind`: upwind branch leaves loops/if.
- `hypar_weno_weights_js`: three output stores and nonlinear scalar chain remain
  as loops.
- `hypar_linear_adr_diffusion_h`, `hypar_linear_adr_upwind_const`,
  `hypar_linear_adr_upwind_var`: branch/upwind selection leaves loops/ifs.
- `hypar_euler1d_flux`, `hypar_euler2d_flux_x`, `hypar_euler2d_flux_y`:
  fixed-component flux writes remain as scalar loops with multiple stores.
- `exasp2_normalize_dense`: direct diagonal branch fails during raise. The pass
  creates a `linalg.generic` body containing an `affine.if` whose condition uses
  `linalg.index` values; MLIR rejects those values as affine dimension ids.
- `exasp2_normalize_dense_split`: the same normalization decomposed as full
  matrix scale plus diagonal add raises to two tensor Linalg ops.
- `exasp2_sp2_select_square`: branch selecting `X = X^2` vs `X = 2X - X^2`
  leaves loops/if.
- `exasp2_conjugate_gradient_step`: multi-store vector update remains a loop.

## Interpretation

The isolated results are strong for the paper narrative: once the application
ABI, MPI, BML, and solver structs are removed, most regular compute and layout
kernels from these proxy apps raise to Linalg. The remaining cases point to
specific next compiler/matcher work: diagonal/conditional updates inside Linalg,
multi-output loop bodies, branch/upwind selection, parity-dependent indexing,
and nested small reductions.
