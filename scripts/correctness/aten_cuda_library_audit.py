#!/usr/bin/env python3
"""Classify every ATen fixture against existing NVIDIA CUDA-library APIs.

This is deliberately conservative about *direct* matches.  A fixed API is a
single public operation with the fixture's semantics.  A generic primitive is
an existing NVIDIA implementation (cuTENSOR/cuDNN graph/CUB) that needs
descriptor construction or template instantiation.  A partial API covers only
stages of the fixture and therefore needs graph composition.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from build_ce_viewer import ATEN_C_PROVENANCE

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "issues/aten_c_kernels"
SUMMARY = CORPUS / "results/summary.tsv"
OUTPUT = CORPUS / "cuda_library_audit.csv"
REPORT = CORPUS / "CUDA_LIBRARY_AUDIT.md"

EVIDENCE = {
    "cuBLAS": "https://docs.nvidia.com/cuda/cublas/",
    "cuDNN": "https://docs.nvidia.com/deeplearning/cudnn/latest/operations/operations.html",
    "cuDNN Resample": "https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Resampling.html",
    "cuDNN CTC": "https://docs.nvidia.com/deeplearning/cudnn/backend/latest/api/cudnn-adv-library.html",
    "cuTENSOR": "https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html",
    "cuSPARSE": "https://docs.nvidia.com/cuda/cusparse/",
    "cuSOLVER": "https://docs.nvidia.com/cuda/cusolver/contents.html",
    "cuFFT": "https://docs.nvidia.com/cuda/cufft/contents.html",
    "cuRAND": "https://docs.nvidia.com/cuda/curand/index.html",
    "CUB": "https://nvidia.github.io/cccl/cub/api/device.html",
    "NPP": "https://docs.nvidia.com/cuda/npp/index.html",
    "CUDA Runtime": "https://docs.nvidia.com/cuda/cuda-runtime-api/",
}


def hit(pattern: str, text: str) -> bool:
    return re.search(pattern, text) is not None


def classify(name: str, source: str, token: str) -> dict[str, str]:
    n = name.removeprefix("aten_")
    text = " ".join((n, token, Path(source).stem)).lower()

    def result(family: str, library: str, api: str, availability: str,
               scope: str, rationale: str) -> dict[str, str]:
        return {
            "semantic_family": family,
            "candidate_library": library,
            "candidate_api": api,
            "availability": availability,
            "coverage_scope": scope,
            "rationale": rationale,
            "evidence_url": EVIDENCE.get(library, ""),
        }

    # Exact and easily confused cases are kept ahead of the family rules.  In
    # particular, substring matching must not turn acos/acosh into cos, or an
    # arbitrary helper containing "add" into a cuDNN ADD operation.
    if n == "amp_update_scale_cpu":
        return result(
            "scalar_state_update", "", "none", "NO_DIRECT_LIBRARY_API",
            "none", "a two-scalar host control-flow update is not a tensor operation")
    if n in {"erfcx", "log_ndtr"}:
        return result(
            "opaque_special_function", "", "none", "NO_DIRECT_LIBRARY_API",
            "none", "the extraction calls an ATen scalar helper with no equivalent public NVIDIA tensor-library operation")
    if n == "cartesian_prod_cpu":
        return result("tensor_broadcast", "cuBLAS",
                      "SGER outer products with vectors of ones",
                      "FULL_GENERIC_API", "whole",
                      "the two outputs broadcast each input across the Cartesian grid")
    if n == "nested_sum_backward_cpu":
        return result("tensor_broadcast", "cuBLAS",
                      "SGER outer product with a vector of ones",
                      "FULL_GENERIC_API", "whole",
                      "sum backward replicates each row gradient; it performs no reduction")
    if n == "allany_dims_cpu":
        return result("boolean_reduction", "CUB",
                      "DeviceSegmentedReduce with logical AND/OR",
                      "FULL_GENERIC_API", "whole",
                      "the runtime flag selects an associative row-wise all or any reduction")
    if "histogram" in n or "histogramdd" in n:
        return result("histogram_count", "CUB", "DeviceHistogram",
                      "FULL_GENERIC_API", "whole",
                      "the extracted binning operation maps to a device histogram primitive")
    if "sparse" in n and hit(r"(^|_)(mm|mv|bmm|spmm|addmm|addmv)(_|$)", n):
        return result("sparse_linear_algebra", "cuSPARSE", "cusparseSpMV/SpMM/SpGEMM/SDDMM",
                      "FULL_FIXED_API", "whole",
                      "standard sparse-dense or sparse-sparse linear algebra")
    if "fft_conjugate_symmetry" in n:
        return result("complex_layout", "", "",
                      "NO_DIRECT_LIBRARY_API", "none",
                      "this symmetry-fill helper has no link-only NVIDIA tensor-library call")
    if hit(r"(^|_)(blas_axpy|blas_scale|linear_combination|flatten_nd_linear)(_|$)", n):
        return result("dense_vector_update", "cuBLAS", "cublasAxpy/cublasScal/cublasGemv",
                      "FULL_FIXED_API", "whole",
                      "the extracted loop is a standard BLAS vector update")
    if hit(r"(^|_)(ctc_loss)(_|$)", n):
        return result("ctc_loss", "cuDNN CTC", "cudnnCTCLoss_v8",
                      "PARTIAL_API", "fused_loss_and_gradient_only",
                      "cuDNN computes costs and gradients, but the extracted forward exposes its alpha DP table and the backward consumes that table plus grad_loss")
    if hit(r"(^|_)(argmax|argmin)(_|$)", n):
        return result("arg_reduction", "CUB", "DeviceSegmentedReduce ArgMax/ArgMin",
                      "FULL_GENERIC_API", "whole",
                      "each row is a segment reduced to a value/index pair")
    if "allany_dims" in n:
        return result("boolean_reduction", "cuDNN", "pointwise cast plus MIN/MAX reduction graph",
                      "FULL_GENERIC_API", "whole",
                      "boolean all/any is a graph-expressible reduction over each row")
    if "diff_cpu" in n:
        return result("adjacent_difference", "CUB", "DeviceAdjacentDifference",
                      "FULL_GENERIC_API", "whole",
                      "CUB directly implements adjacent differences")
    if n == "embedding" or "put_cpu" in n or "spdiags" in n:
        return result("indexed_data_movement", "CUB", "DeviceSelect/sort primitives",
                      "PARTIAL_API", "stages",
                      "CUB provides building blocks but no general gather/scatter tensor call")
    if n == "nested_select_cpu":
        return result("arbitrary_gather", "", "none",
                      "NO_DIRECT_LIBRARY_API", "none",
                      "each row uses a runtime index, so this is neither a contiguous copy nor a regular tensor permutation")
    if n in {"fftshift_cpu", "ifftshift_cpu"}:
        return result("cyclic_shift", "", "none",
                      "NO_DIRECT_LIBRARY_API", "none",
                      "a wraparound cyclic shift is not a cuTENSOR mode permutation")
    if n in {"lshift_i32", "rshift_i32"}:
        return result("variable_bit_shift", "", "none",
                      "NO_DIRECT_LIBRARY_API", "none",
                      "NPP exposes constant-shift signal calls, but this fixture supplies a distinct shift count for every element")
    if n == "copysign":
        return result("scalar_copysign", "", "none",
                      "NO_DIRECT_LIBRARY_API", "none",
                      "copying a sign bit is pointwise arithmetic, not tensor data movement")
    if hit(r"nested_(clone|squeeze)_cpu", n):
        return result("data_movement", "CUDA Runtime", "cudaMemcpyAsync",
                      "FULL_GENERIC_API", "whole",
                      "the standalone operation copies or reinterprets nested storage metadata")
    if hit(r"(^|_)(eq|ne|ge|gt|le|lt|fmax|fmin)(_|$)", n):
        return result("pointwise", "cuDNN", "relational or MIN/MAX pointwise graph operation",
                      "FULL_GENERIC_API", "whole",
                      "cuDNN has fixed relational and elementwise min/max pointwise modes")
    if "equal_cpu" in n:
        return result("compare_and_reduce", "cuDNN", "EQ pointwise plus MIN reduction graph",
                      "FULL_GENERIC_API", "whole",
                      "tensor equality is elementwise comparison followed by an all reduction")
    if hit(r"(^|_)(erf)(_|$)", n):
        return result("pointwise", "cuDNN", "ERF pointwise graph operation",
                      "FULL_GENERIC_API", "whole",
                      "cuDNN exposes an ERF pointwise mode")
    if hit(r"hardshrink|heaviside|huber_|mish|mse_|nan_to_num|shrink_backward|smooth_l1|softshrink|masked_scale", n):
        return result("pointwise_formula", "cuDNN", "multi-node pointwise graph",
                      "FULL_GENERIC_API", "whole",
                      "the formula is composed entirely from supported arithmetic, comparison, selection, and activation nodes")
    if hit(r"fused_(adagrad|adam|sgd)", n):
        return result("optimizer_update", "cuDNN", "pointwise/reduction operation graph",
                      "PARTIAL_API", "update_stages",
                      "the arithmetic stages are graph-expressible, but optimizer state/step semantics require composition")
    if hit(r"(^|_)cross(_|$)", n):
        return result("cross_product", "cuDNN", "MUL/SUB pointwise operation graph",
                      "FULL_GENERIC_API", "whole",
                      "a 3-vector cross product is six multiplies and three subtracts in one operation graph")
    if hit(r"gradient(_float)?_cpu", n):
        return result("finite_difference", "CUB", "DeviceAdjacentDifference plus boundary transform",
                      "PARTIAL_API", "interior_and_boundary_stages",
                      "the adjacent-difference primitive covers a stage, while centered and boundary formulas require composition")
    if "mode_cpu" in n:
        return result("statistical_mode", "CUB", "DeviceRadixSort plus DeviceRunLengthEncode/Reduce",
                      "PARTIAL_API", "sort_and_count_stages",
                      "CUB supplies the sort and run counting stages but not one mode call with ATen tie/index semantics")
    if "nansum" in n:
        return result("nan_ignoring_reduction", "CUB", "transform iterator plus DeviceSegmentedReduce::Sum",
                      "FULL_GENERIC_API", "whole",
                      "a transform iterator maps source NaNs to the additive identity before segmented sum")
    if n == "joint_scaling_cpu":
        return result("joint_maxabs_product", "cuBLAS",
                      "two cublasIsamax calls plus scalar product",
                      "FULL_GENERIC_API", "whole",
                      "two independent maximum-absolute reductions feed a scalar multiply")
    if "quant_col_offsets" in n:
        return result("column_reduction", "CUB", "DeviceSegmentedReduce",
                      "FULL_GENERIC_API", "whole",
                      "columns form regular reduction segments")
    if "rowwise_prune" in n:
        return result("rowwise_l1_threshold", "CUB",
                      "DeviceSegmentedReduce with transform input iterator",
                      "PARTIAL_API", "reduction_stage",
                      "the kernel emits a per-row threshold mask rather than a compacted "
                      "index list; CUB can compute the L1 row scores and leave the "
                      "comparison/cast as residual Linalg")
    if hit(r"flatten_indices|nested_to_mask|tril_indices|triu_indices|triu_mask|triu_tril_batch", n):
        return result("index_generation", "cuDNN", "GEN_INDEX plus arithmetic/comparison graph",
                      "FULL_GENERIC_API", "whole",
                      "cuDNN can generate coordinates and form masks or flattened indices in an operation graph")
    if "polar_scalarized" in n:
        return result("complex_construction", "cuDNN", "SIN/COS/MUL pointwise graph",
                      "FULL_GENERIC_API", "whole",
                      "polar conversion is a supported pointwise operation graph")
    if "transform_bias_rescale_qkv" in n:
        return result("qkv_transform", "cuDNN", "pointwise plus reshape/transpose graph",
                      "FULL_GENERIC_API", "whole",
                      "bias, scaling, and regular QKV layout transforms are graph-expressible")
    if "amp_update_scale" in n:
        return result("conditional_scalar_update", "cuDNN", "comparison/BINARY_SELECT pointwise graph",
                      "FULL_GENERIC_API", "whole",
                      "the scale update is a small comparison-and-selection graph")
    if hit(r"addcdiv|addcmul|addr_elementwise|angle_real|atanh|entr|erfcx?|frac|glu|hardsigmoid|hardswish|hardtanh|hypot|isneginf|isposinf|joint_scaling|ldexp|logaddexp|logit|powsum|quant_saturation|sinc|sinh|cosh|xlog", n):
        return result("pointwise_reduction_formula", "cuDNN", "pointwise and reduction operation graph",
                      "FULL_GENERIC_API", "whole",
                      "the complete formula can be assembled from documented cuDNN graph nodes")
    if "dropout_feature_noise" in n:
        return result("dropout", "cuDNN", "Bernoulli RNG plus MUL pointwise graph",
                      "FULL_GENERIC_API", "whole",
                      "cuDNN graph RNG and pointwise nodes express feature dropout")
    if "nested_matmul_broadcast" in n:
        return result("batched_matrix_multiply", "cuBLAS", "cublasGemmStridedBatchedEx",
                      "FULL_FIXED_API", "whole",
                      "the right matrix is broadcast across a regular GEMM batch")
    if "isin_default" in n:
        return result("set_membership", "CUB", "DeviceRadixSort building block",
                      "PARTIAL_API", "sort_stage",
                      "CUB sorts the values but provides no complete membership-search call")
    if "triu_tril_single" in n:
        return result("triangular_mask", "cuDNN", "GEN_INDEX/comparison/BINARY_SELECT graph",
                      "FULL_GENERIC_API", "whole",
                      "generated row/column indices form the triangular selection predicate")
    if "multinomial_with_replacement" in n:
        return result("categorical_sampling", "CUB", "segmented scan",
                      "PARTIAL_API", "stages",
                      "scan and search primitives exist, but sampling and batching require composition")
    if "dyn_quant_matmul_4bit" in n:
        return result("quantized_matrix_multiply", "cuBLAS", "cuBLASLt low-bit matmul plus scale/zero-point handling",
                      "PARTIAL_API", "matmul_stage",
                      "low-bit matmul exists, but this packed layout and per-column affine dequantization need validation/composition")
    if "sparse" in n and "softmax" in n:
        return result("sparse_softmax", "CUB", "segmented max/sum reductions plus pointwise transforms",
                      "PARTIAL_API", "stages",
                      "sparse rows can use segmented primitives, but there is no one sparse-softmax library call")
    if "nested" in n and "softmax" in n:
        return result("ragged_softmax", "CUB", "segmented max/sum reductions plus pointwise transforms",
                      "PARTIAL_API", "stages",
                      "ragged offsets define segments, but softmax needs a multi-stage composition")
    if "compressed_block_convert" in n:
        return result("tensor_permutation", "cuTENSOR", "cutensorPermute",
                      "FULL_GENERIC_API", "whole",
                      "the dense-to-block-major transform is a rank-4 mode permutation after a zero-copy logical reshape")
    if hit(r"convert_(coo|csr)", n):
        return result("sparse_format", "cuSPARSE", "cusparseXcoo2csr/csr2coo or conversion APIs",
                      "FULL_FIXED_API", "whole",
                      "cuSPARSE directly exposes standard sparse index/format conversions")

    # Dense BLAS and Einstein contractions.
    if hit(r"(^|_)(addmm|mm|bmm|gemm|gemv|mv|dot|outer|ger|syrk|trmm|trsm)(_|$)", n):
        api = "cuBLAS Level-1/2/3 or cuBLASLt Matmul"
        return result("dense_linear_algebra", "cuBLAS", api,
                      "FULL_FIXED_API", "whole",
                      "standard vector/matrix product or update")
    if "kron" in n:
        return result("tensor_product", "cuTENSOR",
                      "cutensorCreateElementwiseTrinary",
                      "FULL_GENERIC_API", "whole",
                      "broadcast multiply with explicit input/output modes and strides")
    if "upsample" not in n and hit(r"bilinear|trilinear", n):
        return result("tensor_contraction", "cuTensorNet",
                      "cutensornetCreateNetworkDescriptor/ContractionOptimizer",
                      "FULL_GENERIC_API", "whole",
                      "three-input Einstein network; a single binary cuTENSOR contraction is insufficient")
    if hit(r"sumproduct|contraction|tensor_product", n):
        return result("tensor_contraction", "cuTENSOR", "cutensorCreateContraction",
                      "FULL_GENERIC_API", "whole",
                      "Einstein-style multiply/reduce with explicit modes")
    if hit(r"matrix_power|(^|_)(eig|eigen|svd|cholesky|lu|qr)(_|$)|reflect_conj|unpack_pivots", text):
        return result("matrix_factorization", "cuSOLVER", "cuSolverDN dense LAPACK APIs",
                      "PARTIAL_API", "stage",
                      "cuSOLVER covers the factorization/solve; helper-only fixtures need composition")

    # Convolution, pooling, attention, normalization and resampling.
    if "im2col" in n or "columns" in n or "unfold" in n or "col2im" in n:
        return result("patch_extract_scatter", "cuDNN", "Convolution graph operation",
                      "PARTIAL_API", "containing_operation",
                      "cuDNN implements convolution but does not expose im2col/col2im as its public result")
    if hit(r"(^|_)(conv[123]d|conv_tbc|conv_transpose[123]d|convolution|depthwise_conv|dilated_convolution|slow_conv)", n):
        return result("convolution", "cuDNN", "ConvolutionFwd/BwdData/BwdFilter",
                      "FULL_FIXED_API", "whole",
                      "standard, transposed, or dilated convolution maps to cuDNN convolution descriptors")
    if "flash_attention" in n or "scaled_dot" in n or "attention" in n:
        return result("attention", "cuDNN", "Fused Flash Attention graph",
                      "FULL_FIXED_API", "whole",
                      "cuDNN frontend exposes attention forward/backward graphs")
    if hit(r"batch_norm|layer_norm|group_norm|rms_norm|weight_norm|renorm", n):
        return result("normalization", "cuDNN", "NormalizationForward/Backward graph",
                      "FULL_GENERIC_API", "whole",
                      "cuDNN normalization and graph pointwise/reduction nodes cover the operation")
    if "softmax" in n:
        return result("softmax", "cuDNN", "Softmax or pointwise+reduction graph",
                      "FULL_FIXED_API", "whole",
                      "dense softmax is fixed-function; sparse/nested forms require layout composition")
    if "fractional_max_pool" in n:
        return result("fractional_pooling", "cuDNN Resample", "max-pooling plus generated window positions",
                      "PARTIAL_API", "fixed_window_reduction_stage",
                      "cuDNN has max pooling, but fractional sample-dependent window origins require composition")
    if "max_unpool" in n:
        return result("indexed_scatter", "CUDA Runtime", "cudaMemset plus residual scatter",
                      "PARTIAL_API", "initialization_stage",
                      "zero-fill is available but indexed scatter has no link-only library call")
    if "adaptive" in n and "pool" in n:
        return result("adaptive_pooling", "cuDNN Resample", "average/max reduction over windows",
                      "PARTIAL_API", "regular_window_cases",
                      "cuDNN pooling uses fixed windows/strides; general adaptive pooling has output-dependent window boundaries")
    if "pool" in n:
        return result("pooling", "cuDNN Resample", "ResampleFwd/ResampleBwd",
                      "FULL_FIXED_API", "whole",
                      "cuDNN resample directly supports regular average/max pooling")
    if hit(r"upsample|grid_sampler|resize|resample", n):
        if "grid_sampler" in n:
            availability = "PARTIAL_API" if "backward" in n else "FULL_GENERIC_API"
            return result("resampling", "NPP", "nppiResize/nppiRemap",
                          availability, "whole_or_forward_stage",
                          "NPP provides 2D resize/remap interpolation, but does not expose the matching backward operator")
        if n == "upsample_bilinear2d":
            return result(
                "resampling", "cuDNN Resample", "ResampleFwd",
                "FULL_FIXED_API", "whole",
                "fixture is exactly cuDNN's supported FP32 2x bilinear, half-pixel, edge-clamped subset")
        if "bilinear2d" in n and "antialias" not in n and "_aa" not in n:
            return result(
                "resampling", "cuDNN Resample", "ResampleFwd/ResampleBwd",
                "PARTIAL_API", "exact_2x_subset",
                "cuDNN bilinear upsampling requires FP32, 2x spatial output, half-pixel coordinates, edge padding, and NHWC-compatible layout")
        if ("nearest2d" in n or "bicubic2d" in n or "lanczos" in n) and \
                "backward" not in n:
            return result(
                "resampling", "NPP", "nppiResize",
                "PARTIAL_API", "compatible_2d_forward_subset",
                "NPP may cover compatible 2D forward ROI/layout/interpolation cases; it does not provide ATen's general backward operation")
        return result(
            "resampling", "", "conventional Linalg/GPU lowering",
            "NO_DIRECT_LIBRARY_API", "none",
            "installed cuDNN does not support nearest upsampling or general 1D/3D interpolation; coordinate/backward semantics lack a defensible fixed call")

    if n == "sparse_sum_backward_cpu":
        return result("scalar_fill", "", "none",
                      "NO_DIRECT_LIBRARY_API", "none",
                      "the kernel broadcasts an arbitrary runtime float; cudaMemset only represents zero")

    # Sparse tensor norm operates on the stored value vector; the sparse index
    # structure is irrelevant to this extracted kernel.
    if n == "sparse_norm_cpu":
        return result("euclidean_norm", "cuBLAS", "cublasSnrm2",
                      "FULL_FIXED_API", "whole",
                      "the kernel computes sqrt(sum(value[i]^2)) over the stored values")

    # Sparse linear algebra and sparse format manipulation.
    if hit(r"sparse.*(mm|mv|bmm|spmm|addmv)|(^|_)spmm|sspaddmm|hspmm", n):
        return result("sparse_linear_algebra", "cuSPARSE", "cusparseSpMV/SpMM/SpGEMM/SDDMM",
                      "FULL_FIXED_API", "whole",
                      "standard sparse-dense or sparse-sparse linear algebra")
    if "sparse" in n or hit(r"coo|csr|bsr|compressed|coalesce", n):
        if hit(r"convert|csr_to_coo|coo_to_csr|sort|coalesce|flatten_indices", n):
            return result("sparse_format", "cuSPARSE", "format conversion and sorting APIs",
                          "FULL_GENERIC_API", "whole",
                          "cuSPARSE exposes sparse format conversion/sorting primitives")
        if hit(r"reduce|sum|norm", n):
            return result("sparse_reduction", "CUB", "DeviceSegmentedReduce",
                          "FULL_GENERIC_API", "whole",
                          "CSR/COO offsets define segments for a library segmented reduction")
        return result("sparse_indexed_elementwise", "cuSPARSE", "SpVec/SpMat plus generic operation",
                      "PARTIAL_API", "stage",
                      "descriptor/storage handling exists, but arbitrary indexed elementwise semantics need composition")

    if hit(r"sobol_(initialize|scramble)", n):
        return result("sobol_state_transform", "", "",
                      "NO_DIRECT_LIBRARY_API", "none",
                      "these helpers transform direction/state arrays; cuRAND does not expose them")

    # Random-number generation.  Transforms not offered by cuRAND are partial.
    if hit(r"uniform|normal_cpu|log_normal|poisson|sobol", n):
        return result("random_generation", "cuRAND", "host/device generation APIs",
                      "FULL_FIXED_API", "whole",
                      "cuRAND directly provides uniform, normal, log-normal, Poisson, and Sobol generation")
    if hit(r"bernoulli|binomial|gamma|dirichlet|cauchy|exponential|geometric|random|randperm", n):
        return result("random_distribution", "cuRAND", "base RNG plus distribution transform",
                      "PARTIAL_API", "random_draw_stage",
                      "cuRAND supplies random bits/uniform/normal draws but not this complete transform in its host API")

    # FFTs and spectral rearrangement.
    if hit(r"(^|_)(fft|dft)(_|$)", n):
        return result("fourier_transform", "cuFFT", "cufftExec* and PlanMany",
                      "FULL_FIXED_API", "whole",
                      "cuFFT directly supports batched 1D/2D/3D real and complex transforms")
    if hit(r"fftshift|conjugate_symmetry|as_complex|complex_scalarized|conj_complex", n):
        return result("complex_layout", "cuTENSOR", "permutation or elementwise conjugate",
                      "FULL_GENERIC_API", "whole",
                      "cuTENSOR supports permutation and conjugate unary operators")

    # Device-wide algorithms: sort/select/scan/reduce/histogram/indexing.
    if hit(r"cumsum|cumprod|cummax|cummin|scan|prefix|batch_offsets", n):
        return result("scan", "CUB", "DeviceScan or DeviceSegmentedScan",
                      "FULL_GENERIC_API", "whole",
                      "device-wide inclusive/exclusive and segmented scans are implemented")
    if hit(r"sort|topk|kth|median|quick_select|unique", n):
        return result("ordering_selection", "CUB", "DeviceRadixSort/MergeSort/TopK/Select/RunLengthEncode",
                      "FULL_GENERIC_API", "whole",
                      "CUB provides device-wide ordering and selection primitives")
    if n == "bincount_cpu":
        return result("weighted_bincount", "CUB",
                      "DeviceRadixSort plus DeviceReduceByKey plus dense scatter",
                      "PARTIAL_API", "multi_stage",
                      "this fixture accumulates floating weights by runtime integer key; "
                      "DeviceHistogram only counts samples and is not a weighted histogram")
    if hit(r"hist|bincount|count_nonzero", n):
        return result("histogram_count", "CUB", "DeviceHistogram/DeviceReduce",
                      "FULL_GENERIC_API", "whole",
                      "histogram/count operations map to device-wide primitives")
    if hit(r"searchsorted|lower_bound|upper_bound|binary_search", n):
        return result("search", "CUB", "DeviceRadixSort building block",
                      "PARTIAL_API", "sort_stage",
                      "CUB has ordering primitives but no direct vectorized binary-search call")
    if hit(r"index|gather|scatter|take|masked_select|masked_scatter|nonzero|where", n):
        if hit(r"reduce|backward|add", n):
            return result("indexed_scatter_reduce", "CUB", "sort/reduce-by-key plus scatter",
                          "PARTIAL_API", "stages",
                          "collision-aware scatter needs ordering/reduction composition")
        return result("indexed_data_movement", "CUB", "DeviceSelect/sort building blocks",
                      "PARTIAL_API", "selection_or_sort_stage",
                      "CUB does not expose a complete arbitrary gather/scatter tensor call")
    if n == "embedding_bag_counts_cpu":
        return result("histogram_count", "CUB", "DeviceHistogram",
                      "FULL_GENERIC_API", "whole",
                      "bounded embedding IDs form a dense integer histogram")
    if n == "embedding_bag_counts_uniq_cpu":
        return result("frequency_by_key", "CUB",
                      "DeviceRadixSort plus DeviceRunLengthEncode plus key lookup",
                      "PARTIAL_API", "multi_stage",
                      "global frequency-by-key requires sorting/counting unique runs and "
                      "mapping counts back to the original order; it is not a segmented reduction")
    if n in {"embedding_bag_backward_sum_cpu", "embedding_bag_backward_max_cpu"}:
        return result("indexed_scatter_reduce", "CUB",
                      "DeviceRadixSort plus DeviceReduceByKey plus scatter",
                      "PARTIAL_API", "multi_stage",
                      "embedding IDs select destination rows and duplicate IDs require a "
                      "collision-aware scatter-add; bag boundaries do not define the reduction groups")
    if n == "embedding_bag_per_sample_backward_cpu":
        return result("indexed_gather_dot", "CUB",
                      "DeviceSegmentedReduce with transform input iterator",
                      "PARTIAL_API", "composed_whole",
                      "each output is a dot product between one bag-gradient row and an "
                      "embedding row selected by a runtime index")
    if hit(r"segment_reduce|segmented|embedding_bag", n):
        return result("segmented_reduction", "CUB", "DeviceSegmentedReduce",
                      "FULL_GENERIC_API", "whole",
                      "offset/length arrays define device-wide reduction segments")
    if hit(r"reduce|(^|_)(sum|mean|prod|all|any|min|max|aminmax|std_var|trace|norm)(_|$)", n):
        if hit(r"std|mean|norm|dot", n):
            return result("reduction", "NPP", "signal statistics/norm APIs",
                          "FULL_FIXED_API", "whole",
                          "NPP exposes mean, standard deviation, norm, dot, min/max, and sum operations")
        return result("reduction", "CUB", "DeviceReduce or DeviceSegmentedReduce",
                      "FULL_GENERIC_API", "whole",
                      "associative tensor reduction maps to a device-wide primitive")

    # Data movement and tensor layouts.
    if hit(r"copy|cat|stack|split|unbind|repeat|tile|pad|shuffle|transpose|permute|flip|narrow|select|block_diag|cartesian|combinations", n):
        if hit(r"transpose|permute|shuffle|repeat|tile|narrow", n):
            return result("tensor_permutation", "cuTENSOR", "cutensorPermute",
                          "FULL_GENERIC_API", "whole",
                          "mode permutation/broadcast covers regular affine layouts")
        if hit(r"flip|reverse", n):
            return result("reverse", "", "", "NO_DIRECT_LIBRARY_API", "none",
                          "no link-only NVIDIA tensor-library reverse call exists")
        if "pad" in n:
            return result("padding", "NPP", "copy-border/image geometry primitives",
                          "PARTIAL_API", "mode_dependent",
                          "constant/image borders exist; circular/reflection and arbitrary rank need composition")
        return result("data_movement", "CUDA Runtime", "cudaMemcpy*/cudaMemset",
                      "FULL_GENERIC_API", "whole",
                      "contiguous copies are fixed runtime calls; structured concatenation needs multiple copies")

    # Initializers and sequences.
    if hit(r"fill|zeros|eye|arange|range_out|linspace|logspace|sequence", n):
        return result("tensor_initialization", "CUDA Runtime", "cudaMemset for zero only",
                      "PARTIAL_API", "zero_fill_stage",
                      "general fill and sequence generation have no link-only runtime call")

    # Fixed pointwise modes and NPP signal routines.
    if hit(r"(^|_)(abs|sqrt|square|exp|exp2|expm1|log|log2|log10|log1p|add|sub|mul|div|remainder|fmod|clamp|threshold|relu|gelu|sigmoid|tanh|silu|swish|softplus|elu|logical|copysign|minimum|maximum|lerp|reciprocal|rsqrt|ceil|floor|round|trunc|neg|sign|signbit|pow)(_|$)", n):
        return result("pointwise", "cuDNN", "Pointwise graph operation",
                      "FULL_GENERIC_API", "whole",
                      "cuDNN exposes a broad fixed set of unary/binary/ternary pointwise modes")
    if hit(r"(^|_)(sin|cos|tan|cbrt|atan|atan2)(_|$)", n):
        return result("pointwise_math", "NPP", "signal arithmetic/transcendental API",
                      "FULL_FIXED_API", "whole",
                      "NPP provides fixed signal math routines for supported dtypes")
    if hit(r"bitwise|xor|and_reduce|or_reduce|lshift|rshift", n):
        return result("integer_pointwise", "NPP", "signal logical/shift API",
                      "FULL_FIXED_API", "whole",
                      "NPP exposes fixed integer logical and constant-shift signal routines")

    # Losses and multi-stage numerical formulas generally need graph assembly.
    if hit(r"loss|margin", n):
        return result("loss", "cuDNN", "pointwise+reduction graph",
                      "PARTIAL_API", "stages",
                      "primitive nodes exist but there is no matching single public loss operation")
    if hit(r"distance|pdist|cdist", n):
        return result("distance", "cuTENSOR", "contraction/reduction plus pointwise graph",
                      "PARTIAL_API", "stages",
                      "dot/reduction stages exist but distance requires composition and indexing")

    # Remaining special functions generally have libdevice scalar routines,
    # but no fixed host-callable tensor library operation.
    if hit(r"bessel|chebyshev|hermite|laguerre|legendre|zeta|digamma|trigamma|polygamma|erfc|gamm|airy|spherical|xlog|entr|i0|i1|ndtri", n):
        return result("special_function", "", "CUDA Math/libdevice scalar function",
                      "NO_DIRECT_LIBRARY_API", "scalar_only",
                      "a device scalar function may exist, but not a fixed tensor-library launch")

    return result("compound_or_specialized", "", "none identified",
                  "NO_DIRECT_LIBRARY_API", "none",
                  "no semantically equivalent public NVIDIA tensor-library operation identified")


def local_backend_status(name: str, audit: dict[str, str]) -> str:
    """Describe whether this repository can already emit the candidate API."""
    family = audit["semantic_family"]
    library = audit["candidate_library"]
    if library == "cuBLAS" and name in {
        "aten_addmm", "aten_blas_dot_naive_cpu", "aten_bf16_dot_cpu",
        "aten_dot", "aten_fp16_dot_cpu", "aten_mm", "aten_mv",
        "aten_blas_gemv_generic_cpu", "aten_linear_combination_cpu",
        "aten_nested_matmul_broadcast_cpu", "aten_outer",
        "aten_sparse_norm_cpu", "aten_joint_scaling_cpu",
    }:
        return "SELECTED_WRAPPERS_PRESENT"
    if library == "cuDNN" and name in {
        "aten_conv2d", "aten_conv3d", "aten_slow_conv3d_forward_cpu",
        "aten_softmax", "aten_conv_transpose2d",
        "aten_depthwise_conv3x3_cpu", "aten_conv_tbc_cpu",
    }:
        return "SELECTED_WRAPPERS_PRESENT"
    if library == "cuDNN" and family == "normalization" and name in {
        "aten_batch_norm", "aten_batch_norm_cpu_entry", "aten_rms_norm"
    }:
        return "SELECTED_WRAPPERS_PRESENT"
    if library == "cuDNN Resample" and family == "pooling":
        return "SELECTED_WRAPPERS_PRESENT"
    if library == "cuDNN Resample" and name == "aten_upsample_bilinear2d":
        return "SELECTED_WRAPPERS_PRESENT"
    if library == "cuFFT" and family == "fourier_transform":
        return "SELECTED_WRAPPERS_PRESENT"
    if library == "cuTENSOR" and audit["candidate_api"] == "cutensorPermute":
        return "SELECTED_WRAPPERS_PRESENT"
    if library == "cuTENSOR" and name in {
        "aten_kron_impl_cpu", "aten_kron_out_cpu"
    }:
        return "SELECTED_WRAPPERS_PRESENT"
    if library == "cuSPARSE" and name in {
        "aten_sparse_addmv_csr_cpu", "aten_sparse_csr_addmm_cpu",
        "aten_sparse_addmm_cpu", "aten_hspmm_cpu",
        "aten_sparse_addmv_bsr_cpu",
        "aten_sampled_addmm_sparse_csr_cpu",
        "aten_convert_coo_to_csr_cpu", "aten_sparse_coo_to_csr_cpu",
        "aten_convert_csr_to_coo_cpu", "aten_sparse_matmul_csr_to_coo_cpu",
    }:
        return "SELECTED_WRAPPERS_PRESENT"
    if library == "CUB" and family == "scan":
        return "SELECTED_WRAPPERS_PRESENT"
    if library == "CUB" and name in {
        "aten_and_reduce_cpu", "aten_count_nonzero_impl_cpu",
        "aten_quant_col_offsets_cpu", "aten_diff_cpu",
        "aten_embedding_bag_counts_cpu", "aten_nansum_cpu",
    }:
        return "SELECTED_WRAPPERS_PRESENT"
    if library == "cuDNN" and "graph" in audit["candidate_api"].lower():
        if name in {"aten_binary_cross_entropy",
                    "aten_transform_bias_rescale_qkv_cpu",
                    "aten_addr_elementwise",
                    "aten_log_sigmoid_cpu"}:
            return "SELECTED_WRAPPERS_PRESENT"
        return "GENERAL_GRAPH_BACKEND_ABSENT"
    if not library:
        return "NO_TENSOR_LIBRARY_API"
    return "API_BACKEND_ABSENT"


def implementation_form(audit: dict[str, str]) -> str:
    availability = audit["availability"]
    api = audit["candidate_api"].lower()
    if availability == "FULL_FIXED_API":
        return "SINGLE_FIXED_CALL"
    if availability == "NO_DIRECT_LIBRARY_API":
        return "NONE"
    if availability == "PARTIAL_API":
        return "PARTIAL_STAGES"
    if ("graph" in api or " plus " in api or
            audit["semantic_family"] in {
                "data_movement", "set_membership", "indexed_scatter",
                "max_unpool", "compare_and_reduce"
            }):
        return "MULTI_NODE_LIBRARY_GRAPH"
    return "SINGLE_CONFIGURED_PRIMITIVE"


def current_implementation_provenance(symbols: str) -> tuple[str, str, str]:
    """Classify whether emitted launches reuse public vendor implementations."""
    names = [s.strip() for s in symbols.split(",") if s.strip()]
    if not names:
        return "NO_IMPLEMENTATION", "no emitted launch", "no"
    classes: list[tuple[str, str, str]] = []
    for symbol in names:
        if symbol.startswith(("cublas", "cudnn", "cutensor", "cutensornet",
                              "cufft", "cusparse", "cusolver", "npp")):
            classes.append(("DIRECT_VENDOR_API",
                            "public vendor API or vendor operation graph", "yes"))
        elif symbol.startswith(("cudaCopy", "memset_zero")):
            classes.append(("CUDA_RUNTIME_PRIMITIVE",
                            "CUDA copy or memset runtime primitive", "yes"))
        elif symbol.startswith("cub"):
            classes.append(("STANDARD_LIBRARY_ALGORITHM",
                            "preimplemented CUB device algorithm", "yes"))
        elif symbol.startswith("custom") or symbol in {
                "gelu_tanh_f32_tensor", "rmsnorm_f32_tensor"}:
            classes.append(("CUSTOM_GENERATED_GPU_FALLBACK",
                            "project-authored GPU implementation", "no"))
        else:
            classes.append(("UNVERIFIED_IMPLEMENTATION",
                            "implementation provenance has not been audited", "no"))
    if all(entry[2] == "yes" for entry in classes):
        kind = (classes[0][0] if all(entry[0] == classes[0][0]
                                     for entry in classes)
                else "LIBRARY_API_COMPOSITION")
        detail = "; ".join(dict.fromkeys(entry[1] for entry in classes))
        return kind, detail, "yes"
    return next(entry for entry in classes if entry[2] == "no")


def diagnose(row: dict[str, str], audit: dict[str, str],
             current_scope: str, counts_as_library_reuse: str) -> str:
    if (current_scope == "COMPLETE_REWRITE_CANDIDATE" and
            counts_as_library_reuse == "yes"):
        return "ALREADY_FOUND"
    if current_scope == "COMPLETE_REWRITE_CANDIDATE":
        return "CUSTOM_GPU_FALLBACK_NOT_LIBRARY_MATCH"
    if current_scope == "PARTIAL_STAGE_ONLY":
        return "PARTIAL_MATCH_ONLY_RESIDUAL_IR_REMAINS"
    if row["status"] == "debufferize_failed":
        return "FRONTEND_BLOCKS_MATCHER"
    if int(row["residual_loops"]):
        return "RAISING_BLOCKS_WHOLE_OP_RECOGNITION"
    if audit["availability"] == "NO_DIRECT_LIBRARY_API":
        return "NO_LIBRARY_MATCH_EXPECTED"
    if audit["availability"] == "PARTIAL_API":
        return "COMPOSITION_REQUIRED_NOT_MATCHER_ONLY"
    if local_backend_status(row["kernel"], audit) == "SELECTED_WRAPPERS_PRESENT":
        return "MATCHER_COVERAGE_GAP"
    return "BACKEND_AND_MATCHER_GAP"


def main() -> None:
    with SUMMARY.open(newline="") as stream:
        summary = list(csv.DictReader(stream, delimiter="\t"))
    output = []
    for row in summary:
        source, token = ATEN_C_PROVENANCE[row["kernel"]]
        audit = classify(row["kernel"], source, token)
        launches = int(row["kernel_launches"])
        current = row["matched_symbols"] if row["matched_symbols"] != "-" else ""
        matched_path = CORPUS / "results" / row["kernel"] / "matched.mlir"
        matched_text = matched_path.read_text() if matched_path.exists() else ""
        remaining_linalg = len(re.findall(r"\blinalg\.(?:generic|matmul|conv)", matched_text))
        remaining_loops = len(re.findall(
            r"\b(?:affine|scf)\.(?:for|parallel|while)\b", matched_text))
        if not launches:
            current_scope = "NONE"
        elif remaining_linalg or remaining_loops:
            current_scope = "PARTIAL_STAGE_ONLY"
        else:
            current_scope = "COMPLETE_REWRITE_CANDIDATE"
        impl_class, impl_detail, is_library = current_implementation_provenance(current)
        gap = diagnose(row, audit, current_scope, is_library)
        output.append({
            "kernel": row["kernel"], "source": source, "source_token": token,
            "pipeline_status": row["status"], "linalg_ops": row["linalg_ops"],
            "residual_loops": row["residual_loops"],
            "current_match": current,
            "current_match_scope": current_scope,
            "remaining_linalg_after_match": remaining_linalg,
            "remaining_loops_after_match": remaining_loops,
            **audit,
            "implementation_form": implementation_form(audit),
            "current_implementation_class": impl_class,
            "current_implementation_detail": impl_detail,
            "counts_as_library_reuse": is_library,
            "local_backend_status": local_backend_status(row["kernel"], audit),
            "compiler_gap": gap,
        })
    fields = list(output[0])
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(output)
    from collections import Counter
    availability = Counter(row["availability"] for row in output)
    forms = Counter(row["implementation_form"] for row in output)
    scopes = Counter(row["current_match_scope"] for row in output)
    gaps = Counter(row["compiler_gap"] for row in output)
    implementation_classes = Counter(
        row["current_implementation_class"] for row in output)
    libraries = Counter(row["candidate_library"] or "none" for row in output)
    matcher_only = [
        row["kernel"] for row in output
        if row["compiler_gap"] == "MATCHER_COVERAGE_GAP"
    ]
    report = [
        "# Exhaustive ATen CUDA-library audit",
        "",
        "This audit adjudicates every provenance-linked standalone ATen C "
        "fixture against public NVIDIA libraries. It separately records whether "
        "the current rewrite covers the complete function or only an initialization/"
        "copy stage. The machine-readable CSV is the authoritative per-kernel list.",
        "",
        f"- Fixtures reviewed: {len(output)}",
        f"- Complete current rewrite candidates: {scopes['COMPLETE_REWRITE_CANDIDATE']}",
        f"- Partial stage-only current matches: {scopes['PARTIAL_STAGE_ONLY']}",
        f"- No current launch: {scopes['NONE']}",
        f"- Complete rewrites using genuine library/runtime algorithms: "
        f"{sum(r['current_match_scope']=='COMPLETE_REWRITE_CANDIDATE' and r['counts_as_library_reuse']=='yes' for r in output)}",
        f"- Complete generated/custom GPU fallbacks (not library matches): "
        f"{sum(r['current_match_scope']=='COMPLETE_REWRITE_CANDIDATE' and r['counts_as_library_reuse']!='yes' for r in output)}",
        "",
        "## What exists in NVIDIA libraries",
        "",
        f"- One fixed public call: {forms['SINGLE_FIXED_CALL']}",
        f"- One configurable generic primitive: {forms['SINGLE_CONFIGURED_PRIMITIVE']}",
        f"- Complete multi-node library graph/composition: {forms['MULTI_NODE_LIBRARY_GRAPH']}",
        f"- Only some stages have library primitives: {forms['PARTIAL_STAGES']}",
        f"- No direct tensor-library implementation: {forms['NONE']}",
        "",
        "A named CUB algorithm means NVIDIA ships the substantive generic "
        "algorithm. Compiler-authored GPU functors are excluded from library-reuse "
        "coverage. A "
        "cuDNN graph result requires graph construction/lowering but executes vendor "
        "graph operations. None should be described as merely a missing Egglog pattern.",
        "",
        "## Current implementation provenance",
        "",
    ]
    report.extend(
        f"- `{key}`: {value}" for key, value in sorted(implementation_classes.items())
    )
    report.extend([
        "",
        "## Compiler diagnosis",
        "",
    ])
    report.extend(
        f"- `{key}`: {value}" for key, value in sorted(gaps.items())
    )
    report.extend([
        "",
        "Only the `MATCHER_COVERAGE_GAP` rows are clean, whole-operation cases "
        "for which a selected runtime-wrapper family is already present locally. "
        "The remaining positive library candidates need raising work, a new API "
        "backend, graph composition, or some combination.",
        "",
        "## Clean matcher-coverage candidates",
        "",
    ])
    report.extend(f"- `{name}`" for name in matcher_only)
    report.extend([
        "",
        "## Candidate-library census",
        "",
    ])
    report.extend(
        f"- {library}: {count}" for library, count in libraries.most_common()
    )
    report.extend([
        "",
        "## Per-kernel results",
        "",
        "See [`cuda_library_audit.csv`](cuda_library_audit.csv). Every row includes "
        "the source provenance, current matcher scope, semantic family, candidate "
        "library/API, whole/partial availability, evidence URL, local backend status, "
        "and the precise compiler gap. The same fields are rendered on the paginated "
        "ATen Compiler Explorer pages.",
        "",
        "## Official capability sources",
        "",
    ])
    for library, url in EVIDENCE.items():
        report.append(f"- [{library}]({url})")
    REPORT.write_text("\n".join(report) + "\n")
    print(f"wrote {len(output)} rows to {OUTPUT}")
    print(f"wrote summary to {REPORT}")
    print("availability", dict(availability))
    print("gap", dict(gaps))


if __name__ == "__main__":
    main()
