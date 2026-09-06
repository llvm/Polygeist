# Exhaustive ATen CUDA-library audit

This audit adjudicates every provenance-linked standalone ATen C fixture against public NVIDIA libraries. It separately records whether the current rewrite covers the complete function or only an initialization/copy stage. The machine-readable CSV is the authoritative per-kernel list.

- Fixtures reviewed: 598
- Complete current rewrite candidates: 269
- Partial stage-only current matches: 85
- No current launch: 244
- Complete rewrites using genuine library/runtime algorithms: 269
- Complete generated/custom GPU fallbacks (not library matches): 0

## What exists in NVIDIA libraries

- One fixed public call: 96
- One configurable generic primitive: 90
- Complete multi-node library graph/composition: 173
- Only some stages have library primitives: 171
- No direct tensor-library implementation: 68

A named CUB algorithm means NVIDIA ships the substantive generic algorithm. Compiler-authored GPU functors are excluded from library-reuse coverage. A cuDNN graph result requires graph construction/lowering but executes vendor graph operations. None should be described as merely a missing Egglog pattern.

## Current implementation provenance

- `CUDA_RUNTIME_PRIMITIVE`: 77
- `DIRECT_VENDOR_API`: 242
- `LIBRARY_API_COMPOSITION`: 13
- `NO_IMPLEMENTATION`: 244
- `STANDARD_LIBRARY_ALGORITHM`: 22

## Compiler diagnosis

- `ALREADY_FOUND`: 269
- `BACKEND_AND_MATCHER_GAP`: 41
- `COMPOSITION_REQUIRED_NOT_MATCHER_ONLY`: 62
- `NO_LIBRARY_MATCH_EXPECTED`: 48
- `PARTIAL_MATCH_ONLY_RESIDUAL_IR_REMAINS`: 85
- `RAISING_BLOCKS_WHOLE_OP_RECOGNITION`: 93

Only the `MATCHER_COVERAGE_GAP` rows are clean, whole-operation cases for which a selected runtime-wrapper family is already present locally. The remaining positive library candidates need raising work, a new API backend, graph composition, or some combination.

## Clean matcher-coverage candidates


## Candidate-library census

- cuDNN: 200
- CUB: 112
- none: 68
- NPP: 46
- cuBLAS: 34
- cuSPARSE: 31
- cuRAND: 27
- cuDNN Resample: 26
- CUDA Runtime: 25
- cuTENSOR: 22
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
