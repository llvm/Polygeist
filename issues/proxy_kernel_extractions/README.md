# Proxy Kernel Extractions

This directory contains standalone, minimized C kernels extracted from the five
C proxy apps selected for the CGO paper experiments:

- `miniAMR`: stencil, material update, and halo pack/unpack kernels.
- `HPGMG`: stencil operators, residual/smoother kernels, BLAS1 kernels,
  restriction/interpolation, flux kernels, and solver update kernels.
- `HyPar`: finite differences, reconstruction/WENO, limiters, LinearADR,
  Burgers, and Euler flux/upwind kernels.
- `SWFFT`: redistribution, slab copy, and transpose/data-layout kernels.
- `ExaSP2`: dense matrix normalization, square/GEMM-like SP2 kernels, trace,
  AXPBY, SpMV, and CG update kernels.

The extraction intentionally removes MPI, BML, and solver-specific structs so
the run answers a narrower question: whether the computational loop/dataflow
shape can be raised by the Polygeist affine-to-Linalg pipeline.

Run:

```sh
issues/proxy_kernel_extractions/run_proxy_kernel_extractions.sh
```

By default the generated MLIR and summary go to:

```sh
/tmp/proxy_kernel_extractions_mlir
```
