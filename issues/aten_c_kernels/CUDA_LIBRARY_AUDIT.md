# Exhaustive ATen CUDA-library audit

This audit adjudicates every provenance-linked standalone ATen C fixture against public NVIDIA libraries. It separately records whether the current rewrite covers the complete function or only an initialization/copy stage. The machine-readable CSV is the authoritative per-kernel list.

- Fixtures reviewed: 598
- Complete current rewrite candidates: 249
- Partial stage-only current matches: 89
- No current launch: 260
- Complete rewrites using genuine library/runtime algorithms: 249
- Complete generated/custom GPU fallbacks (not library matches): 0

## What exists in NVIDIA libraries

- One fixed public call: 126
- One configurable generic primitive: 92
- Complete multi-node library graph/composition: 175
- Only some stages have library primitives: 162
- No direct tensor-library implementation: 43

A named CUB algorithm means NVIDIA ships the substantive generic algorithm. Compiler-authored GPU functors are excluded from library-reuse coverage. A cuDNN graph result requires graph construction/lowering but executes vendor graph operations. None should be described as merely a missing Egglog pattern.

## Current implementation provenance

- `CUDA_RUNTIME_PRIMITIVE`: 81
- `DIRECT_VENDOR_API`: 229
- `LIBRARY_API_COMPOSITION`: 13
- `NO_IMPLEMENTATION`: 260
- `STANDARD_LIBRARY_ALGORITHM`: 15

## Compiler diagnosis

- `ALREADY_FOUND`: 249
- `BACKEND_AND_MATCHER_GAP`: 62
- `COMPOSITION_REQUIRED_NOT_MATCHER_ONLY`: 58
- `NO_LIBRARY_MATCH_EXPECTED`: 36
- `PARTIAL_MATCH_ONLY_RESIDUAL_IR_REMAINS`: 89
- `RAISING_BLOCKS_WHOLE_OP_RECOGNITION`: 104

Only the `MATCHER_COVERAGE_GAP` rows are clean, whole-operation cases for which a selected runtime-wrapper family is already present locally. The remaining positive library candidates need raising work, a new API backend, graph composition, or some combination.

## Clean matcher-coverage candidates


## Candidate-library census

- cuDNN: 202
- CUB: 110
- cuDNN Resample: 49
- NPP: 45
- none: 43
- cuBLAS: 34
- cuSPARSE: 32
- cuRAND: 27
- CUDA Runtime: 26
- cuTENSOR: 23
- cuSOLVER: 3
- cuTensorNet: 2
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
- [NPP](https://docs.nvidia.com/cuda/npp/index.html)
- [CUDA Runtime](https://docs.nvidia.com/cuda/cuda-runtime-api/)
