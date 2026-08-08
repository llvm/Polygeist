# Exhaustive ATen CUDA-library audit

This audit adjudicates every provenance-linked standalone ATen C fixture against public NVIDIA libraries. It separately records whether the current rewrite covers the complete function or only an initialization/copy stage. The machine-readable CSV is the authoritative per-kernel list.

- Fixtures reviewed: 598
- Complete current rewrite candidates: 40
- Partial stage-only current matches: 101
- No current launch: 457

## What exists in NVIDIA libraries

- One fixed public call: 121
- One configurable generic primitive: 139
- Complete multi-node library graph/composition: 182
- Only some stages have library primitives: 121
- No direct tensor-library implementation: 35

A CUB or Thrust result means NVIDIA ships the generic device algorithm; it is not a stable host C ABI and would require a new template-backed runtime wrapper. A cuDNN graph result requires graph construction/lowering. Neither should be described as merely a missing Egglog pattern.

## Compiler diagnosis

- `ALREADY_FOUND`: 40
- `BACKEND_AND_MATCHER_GAP`: 256
- `COMPOSITION_REQUIRED_NOT_MATCHER_ONLY`: 52
- `NO_LIBRARY_MATCH_EXPECTED`: 31
- `PARTIAL_MATCH_ONLY_RESIDUAL_IR_REMAINS`: 101
- `RAISING_BLOCKS_WHOLE_OP_RECOGNITION`: 118

Only the `MATCHER_COVERAGE_GAP` rows are clean, whole-operation cases for which a selected runtime-wrapper family is already present locally. The remaining positive library candidates need raising work, a new API backend, graph composition, or some combination.

## Clean matcher-coverage candidates


## Candidate-library census

- cuDNN: 205
- CUB: 83
- NPP: 45
- Thrust: 43
- cuDNN Resample: 42
- none: 35
- cuTENSOR: 33
- cuBLAS: 32
- cuSPARSE: 32
- cuRAND: 29
- CUDA Runtime: 14
- cuSOLVER: 3
- cuDNN CTC: 2

## Per-kernel results

See [`cuda_library_audit.csv`](cuda_library_audit.csv). Every row includes the source provenance, current matcher scope, semantic family, candidate library/API, whole/partial availability, evidence URL, local backend status, and the precise compiler gap. The same fields are rendered on the paginated ATen Compiler Explorer pages.

## Official capability sources

- [cuBLAS](https://docs.nvidia.com/cuda/cublas/)
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/operations.html)
- [cuDNN Resample](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Resampling.html)
- [cuDNN CTC](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-adv-library.html)
- [cuTENSOR](https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html)
- [cuSPARSE](https://docs.nvidia.com/cuda/cusparse/)
- [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/contents.html)
- [cuFFT](https://docs.nvidia.com/cuda/cufft/contents.html)
- [cuRAND](https://docs.nvidia.com/cuda/curand/index.html)
- [CUB](https://nvidia.github.io/cccl/cub/api/device.html)
- [Thrust](https://nvidia.github.io/cccl/thrust/api/)
- [NPP](https://docs.nvidia.com/cuda/npp/index.html)
- [CUDA Runtime](https://docs.nvidia.com/cuda/cuda-runtime-api/)
