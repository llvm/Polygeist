# PolyBench library-match gap audit

Date: 2026-09-07

This audit covers every current raising gap and every kernel whose main
computation lacks a correctness-approved external-library match.  A related
routine is not called a match unless its mathematical contract covers the
canonical PolyBench operation.  All proposed implementations use external
libraries; no project-authored computational kernel is proposed.

## Priority summary

### Tier 1: direct, exact library operations available in the current stack

| Kernel | CPU candidate | Jetson candidate | Work still required |
|---|---|---|---|
| `trisolv` | OpenBLAS `cblas_dtrsv` | cuBLAS `cublasDtrsv` | Complete: integrated into the canonical tensor-raised sweep; LARGE CPU and Jetson correctness pass. |
| `symm` | OpenBLAS `cblas_dsymm` | equivalent cuBLAS `cublasDsymv` composition | Complete: structural whole-operation recognition and row/column-major mapping pass LARGE correctness. Direct Jetson `cublasDsymm` was non-functional and rejected. |
| `trmm` | OpenBLAS `cblas_dtrmm` | equivalent cuBLAS `cublasDtrmv` + `cublasDscal` composition | Complete: structural whole-operation recognition passes LARGE correctness. Direct Jetson `cublasDtrmm` produced zero output and was rejected. |
| `syrk` | OpenBLAS `cblas_dsyrk` | existing `cublasDsyrk` | Already implemented and correctness/performance tested on `fix/polybench-syrk-syr2k` (`a6c10e2a`); port the matcher/lowering dispatch and retained results into `raisetolinalg`. The CPU runtime shim itself is already present here. |
| `syr2k` | OpenBLAS `cblas_dsyr2k` | existing `cublasDsyr2k` | Already implemented and correctness/performance tested on `fix/polybench-syrk-syr2k` (`a6c10e2a`); port the matcher/lowering dispatch and retained results into `raisetolinalg`. The CPU runtime shim itself is already present here. |
| `cholesky` | LAPACK `dpotrf` | cuSOLVER `cusolverDnDpotrf` | The route already exists and emits one library call, but LARGE fails the strict canonical source-output check; investigate numerical/order behavior before accepting it. |

The installed OpenBLAS exports all six CPU symbols above.  The CUDA 12.6
SBSA headers used for host cross-compilation declare all corresponding cuBLAS
and cuSOLVER entry points.

### Tier 2: exact compositions of standard external-library operations

| Kernel | Candidate composition | Principal matcher requirement |
|---|---|---|
| `covariance` | GEMV for column means, GER for centering, then full GEMM for `X^T X/(n-1)` | Recognize the complete normalize-and-product dataflow. Full GEMM avoids an extra project-authored triangle-mirroring kernel. |
| `correlation` | GEMV + GER, strided `Dnrm2`/`Dscal` per column, residual epsilon clamp, then full GEMM | Fix the current raised residual correctness first, then compose the stages. A single SYRK is insufficient because both output triangles are live. |
| `gramschmidt` | Per outer step: `Dnrm2`, copy/scale, one GEMV over the trailing matrix, and one GER rank-1 update | Match dynamic trailing slices and preserve the Modified Gram-Schmidt order. LAPACK/cuSOLVER GEQRF is Householder QR and is not an output-equivalent replacement. |
| `durbin` | DOT over a reversed prefix plus COPY/AXPY over forward and negative-stride views | Lower dynamic submaps first, then recognize BLAS level-1 reductions/updates inside the sequential Levinson recurrence. No confirmed single-call routine exists in the installed stack. |
| `seidel-2d` | Build each row RHS with AXPY-like operations, then solve the left-to-right bidiagonal recurrence with `Dtbsv` or sparse triangular solve | Prove the in-place wavefront dependence and retain row ordering. A convolution is invalid because it ignores newly written left/top values. |
| `adi` | Pointwise RHS formation plus cuSPARSE `cusparseDgtsv2_nopivot` (multiple RHS) or `cusparseDgtsv2StridedBatch` for each alternating sweep | Lower dynamic submaps, form the three diagonals, and map row/column storage without changing boundary values. The installed CUDA 12.6 cross headers contain both FP64 APIs. CPU can use LAPACK `dgtsv`. |
| `ludcmp` | Non-pivoting GETRF followed by two triangular solves | Standard LAPACK/cuSOLVER GETRF uses partial pivoting and is not generally equivalent. cuSOLVERDx `getrf_no_pivot`/`getrs_no_pivot` or MAGMA non-pivoting LU is semantically suitable, but neither is installed in the current cross toolchain. |
| `lu` | Non-pivoting GETRF | Same pivoting restriction as `ludcmp`; cuSOLVERDx or MAGMA is a future external dependency rather than a current-stack match. |

### Tier 3: external primitives are possible, but not a direct dense-BLAS match

| Kernel | Possible external route | Assessment |
|---|---|---|
| `jacobi-1d` | OpenBLAS/cuBLAS `Dgbmv`, or sparse SpMV | Exact fixed three-diagonal operator with identity boundary rows. High confidence once the two ping-pong phases are recognized. |
| `jacobi-2d` | cuSPARSE SpMV or cuDNN convolution with a fixed five-point filter | Exact for the interior if boundary rows are preserved. The existing raised IR already has two five-point tensor stencil bodies. |
| `heat-3d` | cuSPARSE SpMV or cuDNN 3-D convolution with a fixed seven-point filter | Exact for the interior with explicit boundary preservation. The repository already has real-cuDNN 3-D stencil runtime infrastructure, but this FP64 PolyBench shape is not selected. |
| `fdtd-2d` | cuDNN convolution/pointwise graph or cuSPARSE derivative matrices plus AXPY | Three coupled derivative/update stages per time step; feasible as a composition, not one library call. Boundary assignment must remain outside the library calls. |
| `floyd-warshall` | SuiteSparse:GraphBLAS/LAGraph min-plus outer-product update; GraphBLAST is a possible GPU GraphBLAS backend | The Floyd-Warshall step is exactly `D = min(D, D(:,k) min-plus D(k,:))`. No GraphBLAS implementation is installed, and GraphBLAST's published build targets old CUDA versions, so this is not a near-term Jetson route. |
| `deriche` | Intel IPP order-2 IIR filtering per row | The checked source is an order-2 causal IIR recurrence, so IPP is a plausible CPU match. IPP is not installed and no exact CUDA library routine was confirmed. NPP/cuDNN convolution is not valid for the recursive dependence. |

### Tier 4: no credible external computational-library match found

| Kernel | Reason |
|---|---|
| `nussinov` | Interval dynamic programming with a data-dependent max-plus split reduction and wavefront dependencies. GraphBLAS max-plus primitives are related, but no standard library operation covers the complete recurrence without substantial algorithm synthesis. |

## Raising gaps versus matching gaps

The five current raising gaps remain `adi`, `durbin`, `ludcmp`, `nussinov`,
and `seidel-2d`.  Their candidate libraries do not remove the need to fix the
common submap/view lowering first.  The highest-value raising fix is therefore
generic dynamic submap lowering with preserved alias, offset, size, and stride
information; it unlocks ADI's tridiagonal systems, Durbin's prefix/reverse
views, and Ludcmp's triangular slices without benchmark-specific rewrites.

The raised-but-unmatched group contains several easier wins.  Trisolv and
Cholesky already have benchmark-independent whole-recurrence recognizers;
Trisolv is now correctness-approved at LARGE, while Cholesky is not.
SYMM/TRMM and CPU SYRK/SYR2K should be implemented before stencil and graph
libraries because they are direct standard calls with no algorithmic
reformulation.

## Required validation order

1. Prove semantic recognition on reduced generic IR, including negative tests
   for wrong triangle, transpose, diagonal, boundaries, or pivoting semantics.
2. Cross-build every Jetson binary on the x86 host.
3. Audit the final binary for the intended external-library symbol and absence
   of a local source-kernel implementation.
4. Compare the complete canonical LARGE/FP64 live-out against the identical
   native source.
5. Accept timing only after correctness, with GPU device and end-to-end scopes
   reported separately.

## Primary API references

- NVIDIA cuBLAS function reference (TRSV, SYMM, SYRK, SYR2K, TRMM):
  https://docs.nvidia.com/cuda/cublas/contents.html
- NVIDIA cuSPARSE tridiagonal solvers:
  https://docs.nvidia.com/cuda/cusparse/
- NVIDIA cuSOLVER dense factorizations:
  https://docs.nvidia.com/cuda/cusolver/
- NVIDIA cuSOLVERDx non-pivoting GETRF/GETRS:
  https://docs.nvidia.com/cuda/cusolverdx/api/description_ops.html
- NVIDIA cuDNN convolution operations:
  https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Convolutions.html
- Netlib BLAS operation contracts:
  https://www.netlib.org/blas/blasqr.pdf
- Netlib LAPACK routines and LAPACKE interface:
  https://www.netlib.org/lapack/explore-html/topics.html
  https://www.netlib.org/lapack/lapacke.html
- LAGraph algorithms, including Floyd-Warshall:
  https://graphblas.org/LAGraph-Docs/lagraph.pdf
- Intel IPP IIR operations:
  https://www.intel.com/content/www/us/en/docs/ipp/developer-guide-reference/2021-12/iir-filter-functions.html
- MAGMA non-pivoting LU availability:
  https://icl.utk.edu/projectsfiles/magma/doxygen/modules.html
