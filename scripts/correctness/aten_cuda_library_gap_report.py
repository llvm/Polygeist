#!/usr/bin/env python3
"""Produce a conservative, per-kernel audit of unresolved ATen CUDA matches.

Unlike cuda_library_audit.py, this report distinguishes an exact public
library operation from a useful primitive and records the semantic conditions
that a compiler must prove before replacing the extracted C loop.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "issues/aten_c_kernels"
INPUT = CORPUS / "cuda_library_audit.csv"
OUTPUT = CORPUS / "cuda_library_gap_detailed.csv"
REPORT = CORPUS / "CUDA_LIBRARY_GAP_DETAILED.md"

DOC = {
    "cuBLAS": "https://docs.nvidia.com/cuda/cublas/contents.html",
    "cuDNN": "https://docs.nvidia.com/deeplearning/cudnn/latest/index.html",
    "cuTENSOR": "https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html",
    "cuSPARSE": "https://docs.nvidia.com/cuda/cusparse/index.html",
    "cuSOLVER": "https://docs.nvidia.com/cuda/cusolver/contents.html",
    "cuRAND": "https://docs.nvidia.com/cuda/curand/host-api-overview.html",
    "CUB": "https://nvidia.github.io/cccl/unstable/cub/api/device.html",
    "NPP": "https://docs.nvidia.com/cuda/npp/",
    "CUDA Runtime": "https://docs.nvidia.com/cuda/cuda-runtime-api/",
    "none": "",
}

# Operations actually listed by cutensorOperator_t.  These are configured
# tensor primitives, not one separately named C entry point per operator.
CUTENSOR_UNARY = {
    "abs": "CUTENSOR_OP_ABS", "acos": "CUTENSOR_OP_ACOS",
    "acosh": "CUTENSOR_OP_ACOSH", "asin": "CUTENSOR_OP_ASIN",
    "asinh": "CUTENSOR_OP_ASINH", "atan": "CUTENSOR_OP_ATAN",
    "atanh": "CUTENSOR_OP_ATANH", "ceil": "CUTENSOR_OP_CEIL",
    "cos": "CUTENSOR_OP_COS", "cosh": "CUTENSOR_OP_COSH",
    "exp": "CUTENSOR_OP_EXP", "floor": "CUTENSOR_OP_FLOOR",
    "log": "CUTENSOR_OP_LOG", "mish": "CUTENSOR_OP_MISH",
    "neg": "CUTENSOR_OP_NEG", "reciprocal": "CUTENSOR_OP_RCP",
    "relu": "CUTENSOR_OP_RELU", "sigmoid": "CUTENSOR_OP_SIGMOID",
    "silu": "CUTENSOR_OP_SWISH", "sin": "CUTENSOR_OP_SIN",
    "sinh": "CUTENSOR_OP_SINH", "softplus": "CUTENSOR_OP_SOFT_PLUS",
    "sqrt": "CUTENSOR_OP_SQRT", "tan": "CUTENSOR_OP_TAN",
    "tanh": "CUTENSOR_OP_TANH",
}


def base_name(kernel: str) -> str:
    n = kernel.removeprefix("aten_")
    for suffix in ("_cpu_backend", "_scalarized", "_cpu", "_out"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return n


def route(row: dict[str, str]) -> dict[str, str]:
    """Return the closest *semantically defensible* existing implementation."""
    k, fam = row["kernel"], row["semantic_family"]
    n = base_name(k)

    # Correct semantic-family errors inherited from the broad first-pass name
    # classifier before choosing an API.
    if k == "aten_binary_cross_entropy":
        fam = "loss"
    elif k == "aten_cartesian_prod_cpu":
        fam = "data_movement"
    elif k in {"aten_clamp_max_scalar_cpu", "aten_clamp_min_scalar_cpu"}:
        fam = "pointwise"
    elif k.startswith("aten_nested_softmax"):
        fam = "ragged_softmax"
    elif k == "aten_polar_scalarized":
        fam = "complex_construction"

    def r(lib: str, api: str, relation: str, coverage: str,
          semantic: str, layout: str, work: str, confidence: str = "HIGH",
          priority: str = "MEDIUM", note: str = "") -> dict[str, str]:
        return {
            "closest_library": lib, "closest_api": api,
            "relationship": relation, "whole_kernel_coverage": coverage,
            "semantic_constraints": semantic, "rank_layout_constraints": layout,
            "required_compiler_work": work, "confidence": confidence,
            "priority": priority, "notes": note,
            "evidence_url": DOC.get(lib, ""),
        }

    # Exact cuTENSOR unary operators.  The old audit missed several of these.
    if n in CUTENSOR_UNARY:
        return r("cuTENSOR", f"cutensorPermute/elementwise + {CUTENSOR_UNARY[n]}",
                 "EXACT_CONFIGURED_PRIMITIVE", "whole",
                 "real floating input; preserve ATen NaN/signed-zero behavior where relevant",
                 "explicit modes/strides; cuTENSOR real FP16/BF16/FP32/FP64",
                 "generic cuTENSOR descriptor lowering + semantic matcher",
                 priority="HIGH")

    # Fixed-function dense, convolutional, and neural-network operations.
    if fam == "dense_linear_algebra":
        if "int4" in n or "int8" in n or n.startswith("int_mm"):
            return r("cuBLAS", "cublasLtMatmul", "SUBSET_WITH_CONSTRAINTS", "whole when supported",
                     "quantization scale/zero-point, accumulation width, packing and overflow must agree",
                     "cuBLASLt-supported integer layouts and alignments",
                     "quantized-matmul recognizer + cuBLASLt descriptor/runtime backend", priority="HIGH")
        return r("cuBLAS", "GEMM/StridedBatchedGEMM/GemmBatched", "EXACT_FIXED_CALL", "whole",
                 "alpha/beta, transpose and floating reassociation policy must agree",
                 "matrix/batch strides representable by cuBLAS; nested/ragged batches need grouping",
                 "generalize GEMM matcher and device-resident cuBLAS ABI", priority="HIGHEST")
    if fam == "dense_vector_update":
        return r("cuBLAS", "Axpy/Scal", "EXACT_FIXED_CALL", "whole",
                 "standard BLAS update and supported scalar/type", "constant vector increments",
                 "recognize Level-1 BLAS + add resident wrappers", priority="HIGH")
    if fam == "convolution":
        return r("cuDNN", "Convolution forward/backward-data/backward-filter", "EXACT_FIXED_CALL", "whole",
                 "padding/dilation/groups/transposition and accumulation policy must match",
                 "cuDNN tensor/filter layouts and supported types; conv_tbc may need a layout transform",
                 "convolution descriptor extraction + missing forward/backward wrappers", priority="HIGHEST")
    if fam == "attention":
        return r("cuDNN", "SDPA forward/backward graph", "SUBSET_WITH_CONSTRAINTS", "whole for supported SDPA",
                 "mask, dropout, scale, RNG state, auxiliary statistics and backward contract must match",
                 "cuDNN SDPA head-size/layout/dtype/device restrictions",
                 "recognize complete attention graph + cuDNN frontend plan backend", priority="HIGH")
    if fam == "ctc_loss":
        return r("cuDNN", "CTC loss", "SUBSET_WITH_CONSTRAINTS", "whole",
                 "blank label, normalization, determinism, input lengths and gradient semantics",
                 "cuDNN-supported CTC tensor layout/type/algorithm",
                 "CTC matcher + API wrapper", priority="MEDIUM")
    if fam == "pooling":
        if "max_pool1d" in n:
            return r("CUB", "DeviceSegmentedReduce::ArgMax",
                     "SUBSET_WITH_CONSTRAINTS", "whole for explicit windows",
                     "window bounds, first-index/tie and NaN behavior must match ATen",
                     "each pooling window must be expressible by begin/end segment offsets",
                     "preserve the multi-output value/index reduction through debufferization; then lower to segmented ArgMax",
                     priority="HIGH",
                     note="cuDNN pooling returns values but not ATen's argmax-index output")
        return r("cuDNN", "Resample forward/backward (MAXPOOL/AVGPOOL)", "EXACT_FIXED_CALL", "whole",
                 "padding inclusion, NaN propagation, max-index and tie behavior must match",
                 "regular fixed windows/strides/dilations in supported layouts",
                 "pool descriptor matcher + generic forward/backward lowering", priority="HIGH")
    if fam == "adaptive_pooling":
        return r("cuDNN", "regular Resample/pooling", "SUBSET_WITH_CONSTRAINTS", "only divisible regular-window cases",
                 "adaptive bin boundaries generally vary by output index; max indices/ties must match",
                 "only cases reducible to a fixed window and stride",
                 "prove regular-window specialization; otherwise no one-call library route", priority="LOW")
    if "fractional_max_pool" in n:
        return r("cuDNN", "MAXPOOL Resample", "BUILDING_BLOCKS_ONLY", "window reduction only",
                 "sample-generated window origins and returned indices are outside cuDNN pooling",
                 "irregular per-output windows are not one cuDNN descriptor",
                 "multi-stage composition; not a matcher-only gap", priority="LOW")
    if fam == "softmax":
        return r("cuDNN", "Softmax forward/backward", "EXACT_FIXED_CALL", "whole",
                 "axis, log-softmax mode, scaling and NaN behavior", "dense regular tensor/axis flattening",
                 "softmax axis matcher + general resident wrapper", priority="HIGH")
    if fam == "normalization":
        rel = "SUBSET_WITH_CONSTRAINTS" if any(x in n for x in ("weight_norm", "renorm_scale", "collect_stats", "stats")) else "EXACT_GRAPH_IF_SUPPORTED"
        return r("cuDNN", "Batch/Layer/Group normalization graph", rel,
                 "whole for supported normalization; otherwise normalization stages",
                 "epsilon, training/inference, saved statistics, unbiased variance and backward outputs",
                 "cuDNN normalization layout/type/alignment restrictions",
                 "normalization semantic matcher + cuDNN graph-plan backend", priority="HIGH")

    # General dense tensor algebra.
    if fam == "tensor_product":
        return r("cuTENSOR", "cutensorCreateElementwiseTrinary",
                 "EXACT_CONFIGURED_PRIMITIVE", "whole",
                 "broadcast modes, physical strides, overwrite semantics and non-aliasing",
                 "two dense rank-2 inputs and the interleaved rank-4 output view are representable by descriptors",
                 "recognize the complete reshape/product/writeback region and emit the existing cuTENSOR wrapper",
                 priority="HIGH")
    if fam == "tensor_contraction" and "upsample" not in n:
        return r("cuTENSOR", "cutensorCreateContraction", "EXACT_CONFIGURED_PRIMITIVE", "whole",
                 "multiply-add reduction; alpha/beta and reassociation policy",
                 "modes/extents/strides express the affine accesses; real or complex supported types",
                 "iterator-count-independent contraction recognition + generic descriptor lowering", priority="HIGHEST")
    if fam == "reduction":
        if "aminmax" in n:
            return r("cuTENSOR", "two cutensorCreateReduction plans (MIN and MAX)", "BUILDING_BLOCKS_ONLY", "whole through two calls",
                     "ATen returns both extrema; NaN behavior and reduction reassociation must agree",
                     "regular affine tensor modes and supported floating data/compute type",
                     "recognize paired extrema and emit/cache two cuTENSOR plans", priority="MEDIUM")
        if any(x in n for x in ("and_reduce", "or_reduce", "xor_sum")):
            return r("CUB", "DeviceReduce with logical/bitwise operator", "SUBSET_WITH_CONSTRAINTS", "whole for a flat/segmented supported type",
                     "logical versus bitwise interpretation, identity, integer type and empty input",
                     "flattened contiguous range or explicit tensor-axis segments",
                     "CUB reduction backend + boolean/integer semantic matcher", priority="MEDIUM")
        if "cartesian_prod" in n or "clamp_" in n:
            raise AssertionError("reviewed family override failed")
        if any(x in n for x in ("std_var", "norm", "mean", "trace", "cartesian", "clamp_", "nested")):
            return r("cuTENSOR", "cutensorCreateReduction plus elementwise stages", "BUILDING_BLOCKS_ONLY", "reduction stage",
                     "variance/norm/mean scaling or nested metadata requires extra stages; reduction order may differ",
                     "regular affine tensor modes/strides and supported reduction operator",
                     "raise stages, partition graph, and lower generic reduction descriptors", priority="MEDIUM")
        return r("cuTENSOR", "cutensorCreateReduction", "EXACT_CONFIGURED_PRIMITIVE", "whole",
                 "associative ADD/MUL/MIN/MAX and permitted reassociation; boolean/integer types are restricted",
                 "regular affine tensor modes and supported data/compute type",
                 "generic reduction matcher + cuTENSOR descriptor lowering", priority="HIGH")
    if fam in {"arg_reduction", "statistical_mode"}:
        return r("CUB", "DeviceReduce/SegmentedReduce on value-index pairs; sort+RLE for mode",
                 "BUILDING_BLOCKS_ONLY", "algorithmic stages",
                 "ATen first-index/tie, NaN and stable-order rules need a custom pair comparator/composition",
                 "flatten/segment the requested tensor axis; strided axes may require permutation",
                 "CUB template backend + index-aware matcher + composition", priority="MEDIUM")
    if fam in {"boolean_reduction", "nan_ignoring_reduction", "compare_and_reduce", "pointwise_reduction_formula"}:
        # Some members below are actually single cuTENSOR unary ops and were handled above.
        return r("cuDNN", "pointwise operations + reduction operation graph", "EXACT_GRAPH_IF_SUPPORTED", "whole if graph accepted",
                 "body operations, NaN policy, reduction identity and reassociation must match",
                 "cuDNN supported modes/types/layout/alignment; fusion engines do not accept every arbitrary graph",
                 "extract expression DAG + graph legality/cost check + cuDNN plan lowering", priority="MEDIUM")

    # Pointwise. cuDNN is a graph API for most non-cuTENSOR operators.
    if fam in {"pointwise", "pointwise_formula", "pointwise_math"}:
        return r("cuDNN", "Pointwise operation graph", "EXACT_GRAPH_IF_SUPPORTED", "whole if every node is supported",
                 "rounding mode, integer division/modulo, NaN/signed-zero and backward formula must agree",
                 "broadcastable regular strides and cuDNN-supported data types/alignment",
                 "provenance-preserving expression DAG extraction + cuDNN graph backend", priority="HIGH")
    if fam == "integer_pointwise":
        return r("NPP", "signal logical/shift primitives", "SUBSET_WITH_CONSTRAINTS", "whole only for flat supported integer signals",
                 "signed shifts, overflow and scalar-vs-vector operands must agree",
                 "NPP fixed integer types and contiguous 1D signal representation",
                 "layout/type specialization + NPP wrapper; retain nonmatching cases", priority="LOW")
    if fam in {"special_function", "compound_or_specialized"}:
        if n in {"acos", "asin"}:
            raise AssertionError("cuTENSOR unary routing order failed")
        return r("none", "no public whole-tensor NVIDIA library operation", "NO_PUBLIC_LIBRARY_EQUIVALENT", "none",
                 "scalar CUDA math/libdevice or recurrences are not a host-callable tensor library implementation",
                 "not applicable", "retain raised code or permit a generated/custom GPU kernel", priority="NONE",
                 note="A scalar device function may exist; that is not a link-only tensor-library lowering.")

    if fam == "ragged_softmax":
        return r("CUB", "segmented max/sum reductions plus pointwise transforms",
                 "BUILDING_BLOCKS_ONLY", "softmax stages",
                 "ragged offsets, stable max-subtraction, empty rows, dropout RNG state and backward semantics",
                 "explicit segment offsets over contiguous values",
                 "CUB segmented-reduction backend + multi-stage composition", priority="MEDIUM")
    if fam == "complex_construction":
        return r("cuDNN", "SIN/COS/MUL pointwise operation graph", "EXACT_GRAPH_IF_SUPPORTED", "whole",
                 "construct real/imaginary output with the same polar convention and exceptional values",
                 "regular real input tensors and representable complex output storage",
                 "expression graph extraction + complex-layout-aware cuDNN graph lowering", priority="LOW")

    # Resampling and layout transforms.
    if fam == "resampling" or (fam == "tensor_contraction" and "upsample" in n):
        if "nearest" in n or "linear1d" in n or "bilinear" in n:
            return r("cuDNN", "Resample forward/backward", "SUBSET_WITH_CONSTRAINTS", "whole for supported coordinate mode",
                     "ATen align_corners, half-pixel/exact-nearest, antialias and backward accumulation must match",
                     "cuDNN supported rank/layout/dtype and interpolation modes",
                     "coordinate-mode proof + resample descriptor lowering", priority="HIGH")
        return r("NPP", "nppiResize/nppiRemap", "SUBSET_WITH_CONSTRAINTS", "forward 2D image subset",
                 "ATen grid normalization, padding mode, align_corners, antialias and backward are not generally identical",
                 "NPP 2D image channels/ROI/step and supported dtypes",
                 "specialize proven-compatible 2D forward cases; no generic one-call route", priority="LOW")
    if fam == "tensor_permutation":
        return r("cuTENSOR", "cutensorPermute", "EXACT_CONFIGURED_PRIMITIVE", "whole for affine permutation/broadcast",
                 "pure data rearrangement with no overlapping writes",
                 "positive explicit strides/modes; shuffle/repeat must be representable without index-dependent modulo",
                 "affine-map-to-mode extraction + generic permutation lowering", priority="HIGH")
    if fam == "reverse":
        return r("none", "no direct reverse API", "NO_PUBLIC_LIBRARY_EQUIVALENT", "none",
                 "multi-axis flip order is irrelevant but views/aliasing must be legal",
                 "contiguous flattened range; arbitrary strided axes require permutation/composition",
                 "leave as residual IR", priority="LOW")
    if fam == "padding":
        return r("NPP", "nppiCopy*Border", "SUBSET_WITH_CONSTRAINTS", "2D image constant/replicate border subset",
                 "reflection/circular rules and backward accumulation are not generally covered",
                 "NPP 2D ROI/channel/dtype layouts only",
                 "specialize compatible image cases; otherwise composition", priority="LOW")
    if fam in {"data_movement", "tensor_initialization", "index_generation"}:
        return r("CUDA Runtime" if fam == "data_movement" else "CUB",
                 "cudaMemcpy*/Memset or CUB building blocks", "BUILDING_BLOCKS_ONLY", "regular contiguous stages",
                 "concatenation, combinations, diagonal/mask/index formulas need multiple calls or transforms",
                 "contiguous/regular pitched copies; arbitrary indexing is not memcpy",
                 "shape specialization and multi-call composition", priority="LOW")
    if fam == "complex_layout":
        return r("cuTENSOR", "cutensorPermute with CONJ/IDENTITY", "SUBSET_WITH_CONSTRAINTS", "conjugate/permutation stage",
                 "angle/sign/conjugate-symmetry fill may contain formulas or overlapping writes",
                 "regular complex FP32/FP64 tensors and affine permutation",
                 "split pure conjugate/permutation stages; compose remaining work", priority="MEDIUM")

    # CUB device algorithms are existing NVIDIA implementations,
    # but using them requires a C++ template backend and often multiple calls.
    if fam == "scan":
        return r("CUB", "DeviceScan/DeviceSegmentedScan", "SUBSET_WITH_CONSTRAINTS", "whole for contiguous/segmented associative scans",
                 "axis, inclusive convention, dtype accumulation, logsumexp stability and cummax indices/ties",
                 "scan axis must be contiguous or converted to explicit segments",
                 "scan matcher + CUB template backend + axis specialization", priority="HIGH")
    if fam in {"ordering_selection", "search", "set_membership"}:
        return r("CUB", "DeviceRadixSort/SegmentedRadixSort/Select/RLE",
                 "BUILDING_BLOCKS_ONLY", "sort/search/select stages",
                 "stable ordering, NaNs, first-index/tie policy, multidimensional axes and returned indices",
                 "contiguous keys or explicit segments; arbitrary strided axes need layout conversion",
                 "CUB backend + operation-specific composition", priority="MEDIUM")
    if fam in {"histogram_count", "column_reduction"}:
        return r("CUB", "DeviceHistogram or DeviceReduce", "SUBSET_WITH_CONSTRAINTS", "whole for supported binning/reduction",
                 "bin-edge inclusivity, out-of-range/NaN handling and weighted/multidimensional bins",
                 "supported sample/bin types; histogramdd may require linearized keys",
                 "histogram matcher + CUB backend + semantic guards", priority="MEDIUM")
    if fam in {"indexed_data_movement", "indexed_scatter", "indexed_scatter_reduce", "patch_extract_scatter", "reduce_and_compact"}:
        return r("CUB", "DeviceSelect or sort/reduce-by-key primitives", "BUILDING_BLOCKS_ONLY", "supported indexing stages",
                 "bounds/negative indices, duplicate destinations, atomic reduction, determinism and write order",
                 "index arrays and flattened affine addressing; patch operations need index generation",
                 "indexed-op semantic matcher + collision proof or reduce-by-key composition", priority="MEDIUM")
    if fam in {"segmented_reduction", "adjacent_difference", "finite_difference"}:
        return r("CUB", "DeviceSegmentedReduce", "SUBSET_WITH_CONSTRAINTS", "whole for direct reduction primitive",
                 "empty segments, indices/ties, scale, boundary formula and backward accumulation",
                 "explicit contiguous segments/offsets",
                 "CUB backend + segment/boundary extraction", priority="MEDIUM")

    # Sparse APIs cover standardized matrix operations and formats, not every
    # loop that happens to contain sparse indices.
    if fam == "sparse_linear_algebra":
        return r("cuSPARSE", "SpMV/SpMM/SpGEMM/SDDMM", "SUBSET_WITH_CONSTRAINTS", "whole for standardized sparse algebra",
                 "reduction operator (usually plus-times), duplicate entries, sortedness, transpose and alpha/beta",
                 "supported COO/CSR/CSC/BSR formats, index widths, data types and layouts",
                 "sparse descriptor extraction + cuSPARSE generic-API backend", priority="HIGHEST")
    if fam == "sparse_format":
        return r("cuSPARSE", "COO/CSR conversion and sparse sorting/pruning APIs", "SUBSET_WITH_CONSTRAINTS", "standard conversion/sort stages",
                 "duplicate coalescing, value reduction, block packing and requested ordering",
                 "supported sparse formats/index widths; workspace required",
                 "format recognizer + cuSPARSE conversion backend + residual composition", priority="MEDIUM")
    if fam in {"sparse_reduction", "sparse_softmax", "sparse_indexed_elementwise"}:
        return r("cuSPARSE", "sparse descriptors plus CUB segmented/indexed primitives", "BUILDING_BLOCKS_ONLY", "storage and reduction stages",
                 "implicit zeros, duplicates, segment boundaries, gradients and reduction/softmax semantics",
                 "standard sparse storage; arbitrary indexed formulas remain outside cuSPARSE",
                 "mixed cuSPARSE+CUB graph composition; not a one-call matcher", priority="LOW")

    # RNG equivalence is deliberately conservative: distribution names alone
    # do not imply PyTorch generator/state/reproducibility equivalence.
    if fam in {"random_generation", "random_distribution", "categorical_sampling", "dropout"}:
        return r("cuRAND", "uniform/normal/lognormal/Poisson/Sobol generators", "SUBSET_WITH_CONSTRAINTS", "random draw stage or whole distribution subset",
                 "ATen Philox generator state, seed/offset advancement, reproducibility and exact transform must match",
                 "cuRAND output type/count/alignment; many distributions need a transform",
                 "RNG-state proof + cuRAND backend; compose unsupported transforms", priority="LOW")

    if fam == "matrix_factorization":
        return r("cuSOLVER", "dense eig/LU/QR helper APIs", "BUILDING_BLOCKS_ONLY", "factorization or helper stage",
                 "the extracted helper may only reflect/unpack pivots rather than perform the factorization",
                 "cuSOLVER column-major dense layouts/types/workspaces",
                 "recognize enclosing factorization; helper alone is not a cuSOLVER call", priority="LOW")
    if fam in {"loss", "distance", "optimizer_update", "cross_product", "qkv_transform", "triangular_mask", "conditional_scalar_update"}:
        return r("cuDNN", "pointwise/reduction/matmul operation graph", "BUILDING_BLOCKS_ONLY", "arithmetic stages",
                 "complete formula, index/label rules, state mutation, reductions and backward outputs",
                 "only graph nodes/layouts supported by cuDNN",
                 "extract and partition expression/stage graph; validate plan or keep raised code", priority="LOW")
    if fam == "quantized_matrix_multiply":
        return r("cuBLAS", "cublasLtMatmul", "SUBSET_WITH_CONSTRAINTS", "matmul stage",
                 "4-bit packing, per-group scales/zero-points, dequantization and accumulator semantics",
                 "cuBLASLt-supported quantized types/layouts/alignments",
                 "quantized pattern + pack/layout proof + cuBLASLt backend", priority="HIGH")

    return r("none", "no defensible public-library mapping identified",
             "NO_PUBLIC_LIBRARY_EQUIVALENT", "none",
             "operation-specific semantics exceed reviewed public APIs", "not applicable",
             "retain raised code; revisit only with new library evidence", "MEDIUM", "NONE")


def reviewed_family(row: dict[str, str]) -> str:
    k = row["kernel"]
    if k == "aten_binary_cross_entropy": return "loss"
    if k == "aten_cartesian_prod_cpu": return "data_movement"
    if k in {"aten_clamp_max_scalar_cpu", "aten_clamp_min_scalar_cpu"}: return "pointwise"
    if k.startswith("aten_nested_softmax"): return "ragged_softmax"
    if k == "aten_polar_scalarized": return "complex_construction"
    return row["semantic_family"]


def operation_summary(row: dict[str, str]) -> str:
    token = row.get("source_token", "").replace("_cpu", "").replace("_out", "")
    return f"{reviewed_family(row)}: {token or base_name(row['kernel'])}"


def backend_status(verdict: dict[str, str]) -> str:
    lib = verdict["closest_library"]
    if lib == "cuBLAS":
        return "RELATED_CUBLAS_WRAPPERS_PRESENT_NEED_GENERALIZATION"
    if lib == "cuDNN":
        if "graph" in verdict["closest_api"].lower():
            return "GENERAL_CUDNN_GRAPH_BACKEND_ABSENT"
        return "RELATED_CUDNN_WRAPPERS_PRESENT_NEED_GENERALIZATION"
    if lib == "cuTENSOR":
        return "GENERAL_CUTENSOR_BACKEND_ABSENT_CUTENSORNET_IS_NOT_EQUIVALENT"
    if lib == "none":
        return "NO_PUBLIC_LIBRARY_BACKEND_POSSIBLE"
    return "LIBRARY_BACKEND_ABSENT"


def alternatives(family: str, verdict: dict[str, str]) -> str:
    lib = verdict["closest_library"]
    if family == "dense_linear_algebra":
        return "cuDNN Matmul graph; CUTLASS templates"
    if family in {"tensor_contraction", "tensor_product"}:
        return "cuTensorNet for larger contraction networks; cuBLAS when flattenable to GEMM"
    if family in {"pointwise", "pointwise_formula", "pointwise_math", "pointwise_reduction_formula"}:
        return "cuTENSOR for its fixed unary/binary operator subset; NPP for flat supported signals"
    if "reduction" in family or family in {"normalization", "softmax"}:
        return "CUB segmented/device reduction; cuDNN graph; NPP flat-signal statistics"
    if family in {"convolution", "pooling", "resampling", "adaptive_pooling"}:
        return "NPP for compatible 2D image cases; implicit-GEMM via cuBLAS/CUTLASS for convolution"
    if family.startswith("sparse"):
        return "CUB sort/segmented-reduce for nonstandard sparse semantics"
    if lib == "CUB":
        return "cuTENSOR/cuDNN only when the indexing operation specializes to a regular affine tensor op"
    if lib == "none":
        return "scalar CUDA math/libdevice or generated kernel (not a link-only tensor API)"
    return "none identified with stronger whole-kernel semantics"


def fixture_metadata(kernel: str) -> tuple[str, str, str]:
    path = CORPUS / f"{kernel}.c"
    if not path.exists():
        return str(path.relative_to(ROOT)), "unknown", ""
    text = path.read_text(errors="replace")
    signature = re.search(
        rf"\b{re.escape(kernel)}\s*\((.*?)\)\s*\{{", text, re.S
    )
    type_text = signature.group(1) if signature else text
    types = []
    for typ in ("double", "float", "uint64_t", "uint32_t", "uint16_t", "uint8_t",
                "int64_t", "int32_t", "int16_t", "int8_t", "int", "bool"):
        if re.search(rf"\b{re.escape(typ)}\b", type_text):
            types.append(typ)
    defs = re.findall(r"^\s*#\s*define\s+([A-Z][A-Z0-9_]*)\s+([^\s/]+)", text, re.M)
    shapes = "; ".join(f"{key}={value}" for key, value in defs if key not in {"ATEN_CONST"})
    return str(path.relative_to(ROOT)), "/".join(types) or "unknown", shapes


def gap_class(old: dict[str, str], verdict: dict[str, str]) -> str:
    if (old["current_match_scope"] == "COMPLETE_REWRITE_CANDIDATE" and
            old.get("counts_as_library_reuse") != "yes"):
        return "CUSTOM_GPU_FALLBACK_REQUIRES_LIBRARY_ROUTE"
    if int(old["residual_loops"]) > 0:
        return "RAISING_THEN_LIBRARY_LOWERING"
    if old["current_match_scope"] == "PARTIAL_STAGE_ONLY":
        return "GRAPH_PARTITION_RESIDUAL_THEN_LIBRARY_LOWERING"
    rel = verdict["relationship"]
    if rel == "NO_PUBLIC_LIBRARY_EQUIVALENT":
        return "NO_LINK_ONLY_LIBRARY_ROUTE"
    if rel == "BUILDING_BLOCKS_ONLY":
        return "MULTI_CALL_COMPOSITION_NOT_MATCHER_ONLY"
    if rel == "SUBSET_WITH_CONSTRAINTS":
        return "LEGALITY_SPECIALIZATION_AND_BACKEND"
    return "SEMANTIC_MATCHER_AND_LIBRARY_BACKEND"


def main() -> None:
    rows = list(csv.DictReader(INPUT.open(newline="")))
    total_fixtures = len(rows)
    rows = [
        r for r in rows
        if not (r["current_match_scope"] == "COMPLETE_REWRITE_CANDIDATE" and
                r.get("counts_as_library_reuse") == "yes")
    ]
    out = []
    for old in rows:
        verdict = route(old)
        fixture, scalar_types, shape_macros = fixture_metadata(old["kernel"])
        raising = int(old["residual_loops"]) > 0
        work = verdict["required_compiler_work"]
        if raising:
            work = "finish raising residual loops; then " + work
        elif old["current_match_scope"] == "PARTIAL_STAGE_ONLY":
            work = "preserve current partial match and partition residual graph; then " + work
        out.append({
            "kernel": old["kernel"], "source": old["source"],
            "source_token": old["source_token"],
            "standalone_c": fixture, "fixture_scalar_types": scalar_types,
            "fixture_shape_macros": shape_macros,
            "operation_summary": operation_summary(old),
            "reviewed_semantic_family": reviewed_family(old),
            "current_match": old["current_match"],
            "current_match_scope": old["current_match_scope"],
            "current_implementation_class": old.get(
                "current_implementation_class", "UNVERIFIED_IMPLEMENTATION"),
            "current_implementation_detail": old.get(
                "current_implementation_detail", ""),
            "counts_as_library_reuse": old.get("counts_as_library_reuse", "no"),
            "linalg_ops": old["linalg_ops"], "residual_loops": old["residual_loops"],
            **verdict, "required_compiler_work": work,
            "alternative_libraries": alternatives(reviewed_family(old), verdict),
            "current_backend_support": backend_status(verdict),
            "compiler_gap_class": gap_class(old, verdict),
        })

    with OUTPUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader(); w.writerows(out)

    rel = Counter(r["relationship"] for r in out)
    libs = Counter(r["closest_library"] for r in out)
    priorities = Counter(r["priority"] for r in out)
    backends = Counter(r["current_backend_support"] for r in out)
    gaps = Counter(r["compiler_gap_class"] for r in out)
    families: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in out:
        families[r["operation_summary"].split(":", 1)[0]].append(r)
    high = [r for r in out if r["priority"] in {"HIGHEST", "HIGH"}]

    lines = [
        "# ATen unresolved CUDA-library matcher audit",
        "",
        "This report audits every ATen fixture that does **not** currently end in a "
        "complete rewrite backed by genuine library/runtime algorithms. It deliberately distinguishes an exact public "
        "library operation from a configured primitive, a constrained subset, and "
        "mere building blocks. The row-level CSV is the authoritative artifact.",
        "",
        "## Scope and headline",
        "",
        f"- Unresolved fixtures audited: **{len(out)}** (the other "
        f"{total_fixtures - len(out)}/{total_fixtures} already have a complete "
        "genuine library/runtime rewrite).",
        f"- Complete generated/custom GPU fallbacks still requiring a true library "
        f"route: **{sum(r['current_match_scope']=='COMPLETE_REWRITE_CANDIDATE' and r['counts_as_library_reuse']!='yes' for r in out)}**.",
        f"- No current match: **{sum(r['current_match_scope']=='NONE' for r in out)}**.",
        f"- Partial stage match with residual IR: **{sum(r['current_match_scope']=='PARTIAL_STAGE_ONLY' for r in out)}**.",
        f"- Residual loops still block whole-operation recognition: **{sum(int(r['residual_loops'])>0 for r in out)}**.",
        "- A related library primitive is not automatically a legal or profitable replacement. "
        "The compiler must prove the constraints recorded for that row.",
        "",
        "## Corrected availability classification",
        "",
    ]
    for key, count in rel.most_common():
        lines.append(f"- **{key}**: {count}")
    lines += ["", "By closest library:", ""]
    for key, count in libs.most_common():
        lines.append(f"- **{key}**: {count}")
    lines += ["", "Priority:", ""]
    for key, count in priorities.most_common():
        lines.append(f"- **{key}**: {count}")
    lines += ["", "By concrete compiler gap:", ""]
    for key, count in gaps.most_common():
        lines.append(f"- **{key}**: {count}")
    lines += ["", "Current local backend status:", ""]
    for key, count in backends.most_common():
        lines.append(f"- **{key}**: {count}")

    lines += [
        "", "## Important corrections to the previous audit", "",
        "- `acos`, `asin`, `atan`, `acosh`, `asinh`, `atanh`, trigonometric/hyperbolic "
        "functions, `mish`, `swish`, and `softplus` are explicit `cutensorOperator_t` "
        "values. They are generic cuTENSOR descriptor candidates, not missing CUDA APIs.",
        "- CUB supplies implementations of scans, sorts, reductions, and selection, "
        "but most ATen rows are not a one-call equivalence until "
        "axis layout, tie/index policy, collisions, and determinism are proven.",
        "- NPP is primarily a fixed-type 1D signal / 2D image API. It is relevant to "
        "specialized contiguous cases, not a general arbitrary-rank ATen tensor backend.",
        "- cuRAND having the same distribution name is insufficient for PyTorch equivalence: "
        "generator algorithm, seed/offset advancement, and transform reproducibility matter.",
        "- cuDNN graphs are promising for pointwise/reduction formula DAGs, but the backend "
        "must validate an execution plan; documentation does not promise every arbitrary graph fuses.",
        "", "## Library portfolio reviewed", "",
        "- **cuBLAS/cuBLASLt:** preferred for Level-1/2/3 dense algebra and supported quantized matmul. "
        "It does not cover arbitrary elementwise formulas or tensor-axis reductions.",
        "- **cuDNN:** preferred for convolution, regular pooling/resampling, dense softmax, "
        "normalization, attention, and supported pointwise/reduction graphs. Graph-plan acceptance "
        "and layout/type constraints still require a legality query.",
        "- **cuTENSOR:** preferred for arbitrary-rank affine contraction, permutation, supported "
        "unary elementwise operators, and ADD/MUL/MIN/MAX reductions. The repository currently "
        "does not have this general backend.",
        "- **cuTensorNet:** reviewed as an alternative for multi-tensor contraction networks. It "
        "is not a substitute for general pointwise/reduction lowering, and the repository's fixed "
        "cuTensorNet wrappers do not cover arbitrary ATen shapes. Simple contractions are better "
        "served by cuTENSOR or cuBLAS; larger contraction graphs may later select cuTensorNet.",
        "- **cuSPARSE:** preferred only where the loop is a standardized SpMV/SpMM/SpGEMM/SDDMM "
        "or supported sparse-format conversion. Sparse indexing alone does not make an operation "
        "a cuSPARSE call.",
        "- **CUB:** existing NVIDIA template implementations for scan/sort/reduce/select. "
        "They require a C++ template backend and frequently multi-call composition.",
        "- **NPP:** useful for specialized contiguous signal or 2D-image cases. It is not treated "
        "as a general tensor backend.",
        "- **cuRAND:** useful only when generator-state and sequence compatibility are proven; "
        "otherwise it covers merely the random-draw stage.",
        "- **cuSOLVER:** relevant to enclosing dense factorizations, not automatically to extracted "
        "pivot/reflection helper loops.",
        "- **cuFFT:** no unresolved fixture is an FFT execution. `fftshift`, conjugation, and symmetry "
        "fill helpers are layout/pointwise operations, so cuFFT is not their replacement.",
        "- **CUDA Runtime:** memcpy/memset cover regular contiguous transfers only. Concatenation, "
        "padding, gathers, combinations, and overlapping writes need more than a runtime copy.",
        "- **CUTLASS/cuDNN frontend templates:** reviewed as implementation frameworks, not counted "
        "as link-only fixed APIs. Selecting them requires code generation/template instantiation "
        "and therefore is a different backend strategy.",
        "", "## Highest-value missing work", "",
    ]
    for r in high:
        lines.append(
            f"- **`{r['kernel']}`** → {r['closest_library']} `{r['closest_api']}` "
            f"({r['relationship']}, {r['whole_kernel_coverage']}). "
            f"Work: {r['required_compiler_work']}."
        )

    lines += ["", "## Family-by-family, per-kernel appendix", "",
              "Each entry lists the closest reviewed implementation, the strength of the "
              "relationship, coverage, and required work. Exact legality constraints are in "
              "[`cuda_library_gap_detailed.csv`](cuda_library_gap_detailed.csv).", ""]
    for family in sorted(families):
        rs = families[family]
        lines += [f"### {family} ({len(rs)})", ""]
        for r in rs:
            lines.append(
                f"- `{r['kernel']}` — {r['closest_library']} / `{r['closest_api']}`; "
                f"**{r['relationship']}**; coverage: {r['whole_kernel_coverage']}; "
                f"work: {r['required_compiler_work']}."
            )
        lines.append("")

    lines += [
        "## Primary API evidence", "",
        "- [cuTENSOR operator/data types](https://docs.nvidia.com/cuda/cutensor/latest/api/types.html) "
        "and [operation descriptors](https://docs.nvidia.com/cuda/cutensor/latest/api/cutensor.html)",
        "- [cuDNN operation families](https://docs.nvidia.com/deeplearning/cudnn/latest/index.html), "
        "[pointwise/reduction](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Pointwise.html), "
        "and [graph/runtime-fusion constraints](https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html)",
        "- [cuBLAS APIs](https://docs.nvidia.com/cuda/cublas/contents.html)",
        "- [cuSPARSE generic APIs](https://docs.nvidia.com/cuda/cusparse/index.html)",
        "- [CUB device-wide primitives](https://nvidia.github.io/cccl/unstable/cub/api/device.html)",
        "- [cuRAND host API](https://docs.nvidia.com/cuda/curand/host-api-overview.html)",
        "- [NPP signal/image primitives](https://docs.nvidia.com/cuda/npp/)",
        "- [cuTensorNet overview](https://docs.nvidia.com/cuda/cuquantum/latest/cutensornet/overview.html)",
        "- [cuFFT APIs](https://docs.nvidia.com/cuda/cufft/)",
        "", "## Interpretation", "",
        "`EXACT_FIXED_CALL` is the strongest route. `EXACT_CONFIGURED_PRIMITIVE` means the "
        "mathematics exists but modes/strides/operators must be synthesized. "
        "`EXACT_GRAPH_IF_SUPPORTED` requires graph construction and successful plan validation. "
        "`SUBSET_WITH_CONSTRAINTS` is only legal after specialization. `BUILDING_BLOCKS_ONLY` "
        "is not a matcher-only fix. `NO_PUBLIC_LIBRARY_EQUIVALENT` means link-only lowering is "
        "not available in the reviewed NVIDIA libraries.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUTPUT} ({len(out)} rows)")
    print(f"wrote {REPORT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
