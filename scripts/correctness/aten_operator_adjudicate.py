#!/usr/bin/env python3
"""Adjudicate non-dispatch loop bodies after call-graph coverage analysis.

Every rule is intentionally source/symbol based and carries a reason. Anything
not proven to be plumbing, orchestration, external delegation, or already
covered remains NEEDS_PORT.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "issues/aten_c_kernels/operator_inventory.csv"
OUTPUT = ROOT / "issues/aten_c_kernels/operator_adjudication.csv"

WHOLE_SOURCE = {
    "aten/src/ATen/native/AutogradComposite.cpp": ("NON_NUMERICAL_PLUMBING", "autograd metadata allocation"),
    "aten/src/ATen/native/CPUFallback.cpp": ("NON_NUMERICAL_PLUMBING", "device fallback and argument traversal"),
    "aten/src/ATen/native/IndexingUtils.cpp": ("NON_NUMERICAL_PLUMBING", "index-width eligibility check"),
    "aten/src/ATen/native/Integration.cpp": ("NON_NUMERICAL_PLUMBING", "shape padding helper"),
    "aten/src/ATen/native/LegacyBatching.cpp": ("NON_NUMERICAL_PLUMBING", "batch-dimension metadata"),
    "aten/src/ATen/native/TensorIteratorReduce.cpp": ("NON_NUMERICAL_PLUMBING", "TensorIterator reduction scheduling"),
    "aten/src/ATen/native/TensorProperties.cpp": ("NON_NUMERICAL_PLUMBING", "storage alias/property test"),
    "aten/src/ATen/native/TypeProperties.cpp": ("NON_NUMERICAL_PLUMBING", "dtype inference"),
    "aten/src/ATen/native/UpSample.cpp": ("NON_NUMERICAL_PLUMBING", "output-shape calculation"),
    "aten/src/ATen/native/TestOps.cpp": ("NON_NUMERICAL_PLUMBING", "test-only argument materialization"),
    "aten/src/ATen/native/transformers/sdp_utils_cpp.cpp": ("NON_NUMERICAL_PLUMBING", "backend selection"),
}

EXTERNAL_DELEGATES = {
    "aten/src/ATen/native/BatchLinearAlgebra.cpp": {"apply_cholesky_solve"},
    "aten/src/ATen/native/BatchLinearAlgebraKernel.cpp": {
        "apply_cholesky", "apply_cholesky_inverse", "apply_linalg_eig",
        "apply_lapack_eigh", "apply_geqrf", "apply_orgqr", "apply_ormqr",
        "apply_triangular_solve", "apply_ldl_factor", "apply_ldl_solve",
        "apply_lu_factor", "apply_lu_solve", "apply_svd",
    },
    "aten/src/ATen/native/NNPACK.cpp": {"_nnpack_spatial_convolution"},
    "aten/src/ATen/native/QuantizedLinear.cpp": {
        "fbgemm_linear_int8_weight_fp32_activation",
    },
}

COMPOSITE_SOURCES = {
    "aten/src/ATen/native/Convolution.cpp",
    "aten/src/ATen/native/Copy.cpp",
    "aten/src/ATen/native/ForeachOpsKernels.cpp",
    "aten/src/ATen/native/FusedAdagrad.cpp",
    "aten/src/ATen/native/FusedAdam.cpp",
    "aten/src/ATen/native/FusedSGD.cpp",
    "aten/src/ATen/native/Histogram.cpp",
    "aten/src/ATen/native/MaxUnpooling.cpp",
}

PLUMBING_NAMES = re.compile(
    r"^(?:"
    r".*(?:check|validate|shape_check).*|"
    r"(?:can|should|use)_[A-Za-z0-9_]+|canUse32BitIndexMath|"
    r".*(?:size_stride|strides_for_view|output_memory_format).*|"
    r"compute_target_device|out_device|result_type|find_split_dim|"
    r"remove_existing_batch_dim|add_padding_to_shape|"
    r"allocate_bin_edges_tensors|histogramdd_prepare_out|"
    r"debug_assert_shape|aligned_tensor|to_meta|empty_permuted_symint|"
    r"set_storage_meta__symint|stack_meta|get_stack_inputs|check_stack_inputs|"
    r"compressed_count_blocks|_estimate_sparse_compressed_tensor_size|"
    r"num_bytes|NestedTensor_get_max_size_from_size_tensor|"
    r"cat_compute_output_memory_format|_permute_size_stride_estimation"
    r")$"
)

RNN_ORCHESTRATION = {
    "use_mkldnn", "pair_vec", "unpair_vec", "gather_params", "project",
    "operator", "_lstm_impl", "lstm", "quantized_lstm_input",
    "quantized_lstm_data",
}

# These bodies iterate over sizes, strides, Tensor lists, or dispatch choices;
# they do not implement the elementwise/reduction/contraction arithmetic that
# the standalone-C raising corpus is intended to preserve.
SOURCE_PLUMBING = {
    "aten/src/ATen/native/TensorShape.cpp": {
        "_reshape_from_tensor", "sparse_broadcast_to", "sizes_match_except",
        "tensor_split_sections_symint", "_tensor_split_indices", "tensor_split",
        "split", "unsafe_split", "split_with_sizes", "unsafe_split_with_sizes",
        "_pad_chunk", "inferSqueezeGeometry", "squeeze_qtensor", "flatten",
        "unbind", "meshgrid", "numpy_T", "movedim", "unflatten_dense_tensors",
        "tile_symint",
    },
    "aten/src/ATen/native/SpectralOps.cpp": {
        "resize_fft_input", "canonicalize_fft_shape_and_dim_args", "default_alldims",
    },
    "aten/src/ATen/native/nested/NestedTensorMath.cpp": {
        "_nested_tensor_from_tensor_list", "_nested_view_from_buffer",
        "reshape_as_nested", "cat_nested_as_jagged", "cat_nested_impl",
    },
    "aten/src/ATen/native/nested/NestedTensorMatmul.cpp": {
        "matmul_with_bmm_nested", "matmul_out_nested",
    },
    "aten/src/ATen/native/nested/NestedTensorFactories.cpp": {
        "NestedTensor_unbind",
    },
    "aten/src/ATen/native/nested/NestedTensorUtils.cpp": {
        "chunk_nested_tensor", "split_with_sizes_nested",
    },
    "aten/src/ATen/native/PackedSequence.cpp": {
        "_pack_padded_sequence", "_pack_padded_sequence_backward_symint",
        "_pad_packed_sequence", "pad_sequence",
    },
    "aten/src/ATen/native/Linear.cpp": {"einsum", "tensordot"},
    "aten/src/ATen/native/TensorAdvancedIndexing.cpp": {
        "build_index_op", "all_strides_match", "_scatter_via_index_put",
        "_gather_sparse_backward",
    },
    "aten/src/ATen/native/LinearAlgebra.cpp": {
        "matrix_chain_order", "multi_dot_impl", "mexp_impl",
        "linalg_matrix_power_impl", "compute_T18_scale_square",
    },
    "aten/src/ATen/native/TensorConversions.cpp": {"_to_cpu"},
    "aten/src/ATen/native/sparse/ValidateCompressedIndicesKernel.cpp": {"launch"},
    "aten/src/ATen/native/sparse/SparseTensor.cpp": {"sparse_coo_tensor"},
    "aten/src/ATen/native/EmbeddingBag.cpp": {"fbgemm_spmdm_report_error_"},
}


def adjudicate(row: dict[str, str]) -> tuple[str, str]:
    if row["status"] in {"EXTRACTED", "COVERED_BY_EXTRACTED_ENTRY"}:
        return row["status"], "provenance/call-graph evidence"
    source, symbol = row["source"], row["symbol"]
    if symbol == "constexpr":
        return "PARSER_ARTIFACT", "not a function symbol"
    if source in WHOLE_SOURCE:
        return WHOLE_SOURCE[source]
    if symbol in SOURCE_PLUMBING.get(source, set()):
        return "NON_NUMERICAL_PLUMBING", "shape/view/list orchestration; arithmetic is delegated to called operators"
    if symbol in EXTERNAL_DELEGATES.get(source, set()):
        return "EXTERNAL_LIBRARY_DELEGATION", "batch loop delegates arithmetic to LAPACK/NNPACK/FBGEMM"
    if PLUMBING_NAMES.match(symbol):
        return "NON_NUMERICAL_PLUMBING", "shape, validation, dtype, or backend-selection loop"
    if source in COMPOSITE_SOURCES:
        return "COVERED_COMPOSITE_ORCHESTRATION", "loops dispatch already-extracted backend kernels"
    if source == "aten/src/ATen/native/nested/NestedTensorBinaryOps.cpp" and symbol in {
        "get_elementwise_nested_tensor_impl", "NestedTensor_elementwise_Tensor",
    }:
        return "COVERED_COMPOSITE_ORCHESTRATION", "iterates nested components and delegates arithmetic to dense elementwise operators"
    if source == "aten/src/ATen/native/RNN.cpp" and symbol in RNN_ORCHESTRATION:
        return "COVERED_COMPOSITE_ORCHESTRATION", "tensor-list orchestration over RNN primitives"
    if source == "aten/src/ATen/native/transformers/attention.cpp" and symbol in {
        "debug_assert_shape", "aligned_tensor",
    }:
        return "NON_NUMERICAL_PLUMBING", "shape assertion/aligned allocation"
    return "NEEDS_PORT", "local iterative body not yet proven covered or non-numerical"


def main() -> None:
    rows = list(csv.DictReader(INPUT.open()))
    output = []
    for row in rows:
        final_status, rationale = adjudicate(row)
        output.append({**row, "final_status": final_status, "rationale": rationale})
    fields = list(rows[0]) + ["final_status", "rationale"]
    with OUTPUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    counts: dict[str, int] = {}
    for row in output:
        counts[row["final_status"]] = counts.get(row["final_status"], 0) + 1
    print(f"wrote {len(output)} adjudicated bodies to {OUTPUT}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
