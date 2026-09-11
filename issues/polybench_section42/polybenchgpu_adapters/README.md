# Canonical PolyBenchGPU adapters

These adapters make the independent PolyBenchGPU CUDA baselines comparable to
the PolyBench/C evaluation. They use the canonical PolyBench/C `LARGE` harness,
FP64 datatype, initialization, argument order, and output dump. The CUDA
computational kernel bodies are copied from PolyBenchGPU commit
`5584aaa7d0be810ff5eb0b61c49fb64ecc81ba4c`; Polygeist does not supply a
replacement computational kernel.

The adapter code supplies allocation, transfers, error checks, canonical launch
geometry, CUDA-event device timing, and the PolyBench/C ABI. These are therefore
reported as `modified_source=true`. The original upstream files and SHA-256
values are:

- `CUDA/GEMM/gemm.cu`: `8d458e662aeff25003c5efad74e22ef4414d6655c098aa28a7e8c5bf61e82880`
- `CUDA/2MM/2mm.cu`: `dec4988b28f94c75dfdc3b3048e1dc01511fb8cfcdc0314552451002be9d4a0f`
- `CUDA/3MM/3mm.cu`: `52db8f0bb854d5fcfca0c3368aa6fcfd466f5793b2519d22943086576f83a313`
- `CUDA/GESUMMV/gesummv.cu`: `717c2bc6161a1d9b478282dabe994e0184821ee100996a898aba688302f384a4`
- `CUDA/GEMVER/gemver.cu`: `c53e178dd285504f9dd29479460f85f25fa8fcd2e40a8429bf9f5e16e14de704`
- `CUDA/ATAX/atax.cu`: `5966799837bce3d7fce603c7876f78c2c1bc487f97f590bf242e28ab2acbd230`
- `CUDA/BICG/bicg.cu`: `e6a480c75f939958edd2633d7dc1be0a14d9890c6adb7b916b0db5e2c75965bd`
- `CUDA/CORR/correlation.cu`: `6e027ff963ebd24de31757cc9a407d422282bfd9e24c49c9087e7306fdacbe9e`
- `CUDA/COVAR/covariance.cu`: `236cc00fb1f8a13246314682eb371973fd9363bc3ca911f22aef42726d585c64`
- `CUDA/DOITGEN/doitgen.cu`: `ddd0b526691b6ad5f0afc495dccf2e8e97e12ee2e30ad84771836a95519df3a2`
- `CUDA/FDTD-2D/fdtd2d.cu`: `aa73d1266fdd173be5e2531c7ff2e8271e8d64ca9ce5409dd5b161924f05ddba`
- `CUDA/GRAMSCHM/gramschmidt.cu`: `30f9138e7433dee3a9ce1324ca5d53ec84932b3b9ae49fb26ba2676114a26b89`
- `CUDA/MVT/mvt.cu`: `44aa960a339f3702e82f4e9b2e7b29c90fae19b5a095fa1ccd0f8e6f6b76a87d`

The repository contains six additional same-named CUDA programs that are not
valid canonical baselines: ADI uses an older formulation; Jacobi-1D and
Jacobi-2D replace the canonical second stencil with a copy; LU produces a
different factor layout; and SYR2K/SYRK update the full matrix rather than the
canonical lower triangle. They are recorded as unavailable, not silently
adapted into different algorithms.

`run_polybenchgpu_native.sh` verifies the external repository commit before it
cross-compiles. Nothing is compiled on the Jetson.
