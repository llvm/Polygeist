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

`run_polybenchgpu_native.sh` verifies the external repository commit before it
cross-compiles. Nothing is compiled on the Jetson.
