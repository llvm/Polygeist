#!/usr/bin/env python3
"""CLI: take MLIR text in, emit MLIR with matched linalg.generics replaced
by `kernel.launch @<lib_op>(operands)` ops.

This is the Phase-1 deliverable of the kernel matcher: a textual rewrite
that produces a polygeist-opt-parseable MLIR module with `kernel.launch`
ops at every linalg.generic that the matcher recognized.

Usage:
  kernel_match_rewrite.py <input.mlir>   # prints rewritten MLIR to stdout
  kernel_match_rewrite.py <input.mlir> --dry-run  # report matches, no rewrite

Phase-2 (ABI lowering) will turn each `kernel.launch @cublasDgemm(...)`
into a `func.call @cublasDgemm(handle, ...)` matching the real cuBLAS
ABI. That step is *not* in this script.
"""
from __future__ import annotations
import argparse
import ast
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kernel_match import (
    parse_constants, parse_generics, encode_body,
    match_composition, composition_library,
    match_elementwise_semantic, enumerate_semantic_candidates,
    CompositionEntry, CompositionStep, Term,
    _AFFINE_MAP_RE, _parse_term, _term_repr, equivalent,
)
from structured_loop_egglog import (
    analyze_residual_loops,
    analyze_structured_regions,
    format_residual_candidate,
    format_result,
    _matching_brace,
    parse_loops,
)


# Keep this in sync with lib/polygeist/Passes/LowerKernelLaunchToCuBLAS.cpp.
# The matcher may semantically recognize many more kernels than the ABI/runtime
# layer can execute today. Only emit kernel.launch for symbols that the lowering
# pass currently knows how to turn into a runtime call; leave the rest as
# residual Linalg so the normal MLIR lowering path preserves semantics.
ABI_LOWERABLE_KERNELS = {
    "cubHistogramEvenI32ShiftZero_memref",
    "cublasDtrsvLowerRowMajor_memref",
    "cusolverDnDpotrfLowerRowMajor_memref",
    "cusparseSpMV_CSR_f32_memref",
    "cusparseSpMV_CSR_f64_memref",
    "cusparseSpMM_CSR_f32_memref",
    "cusparseSpMM_COO_f32_memref",
    "cusparseSpMM_BSR_f32_memref",
    "cusparseSDDMM_CSR_f32_memref",
    "cusparseXcoo2csr_i32_memref",
    "cusparseXcsr2coo_i32_memref",
    "cusparseSpMV_JDS_f32_memref",
    "custenStencil2DXY_f64_memref",
    "custenStencil2DXY_f64_tensor",
    "cublasDgemm",
    "cublasDgemm_simple",
    "cublasDgemm_subtract",
    "cublasDgemm_strided_batched_subtract",
    "cublasDgemv_strided_batched_subtract",
    "cublasDgemm_alpha_only",
    "cublasSgemm_broadcast3d_simple",
    "cublasSgemm_broadcast3d_memref",
    "cublasDgeam_scale2D",
    "memset_zero_2D",
    "memset_zero_2D_f32",
    "memset_zero_1D",
    "memset_zero_1D_f32",
    "cublasDgemv",
    "cublasDgemv_T",
    "cublasDgemv_T_zero",
    "cublasDgemv_subtract",
    "cublasDgemv_subtract_T",
    "cublasSgemv",
    "cublasSgemv_T",
    "cublasDgemm_outer_product",
    "cublasSgemm_strided_batched_broadcast_rhs",
    "cublasDgemv_alpha",
    "cublasDaxpby",
    "cublasDscal",
    "cublasSaxpby",
    "cublasSscal",
    "cublasSgemm_nn",
    "cublasSgemm_nt",
    "cublasSgemm_tn",
    "cublasSgemm_tt",
    "cublasSgemm_nn_alpha_beta",
    "cublasSgemm_nt_alpha_beta",
    "cublasSgemm_tn_alpha_beta",
    "cublasSgemm_tt_alpha_beta",
    "cublasSgemm_nn_alpha",
    "cublasSgemm_nt_alpha",
    "cublasSgemm_tn_alpha",
    "cublasSgemm_tt_alpha",
    "cublasSgemm_broadcast3d_colmajor_nt_alpha_beta",
    "cublasSgemm_flat_colmajor_nt_alpha_beta",
    "cublasSgemm_nn_zero",
    "cublasSgemm_nt_zero",
    "cublasSgemm_tn_zero",
    "cublasSgemm_tt_zero",
    "cublasDgemm_zero",
    "cublasSgemm_strided_batched_nn_zero",
    "cublasDaxpy_unit",
    "cublasDger_rank2",
    "cudnnConvolution2D_9tap",
    "cudnnConvolution2D_9tap_f32",
    "cudnnConvolution2D_9tap_f16",
    "cudnnConvolution2D_9tap_bf16",
    "cudnnConvolution2D_9tap_i32",
    "cudnnConvolution2D_25tap",
    "cudnnConvolution2D_25tap_f32",
    "cudnnConvolution2D_ntap",
    "cudnnConvolution2D_ntap_f32",
    "cudnnConvolution2D_ntap_tensor",
    "cudnnConvolution2D_ntap_f32_tensor",
    "cudnnConvolution3D_ntap_tensor",
    "cudnnConvolution3D_ntap_f32_tensor",
    "cudnnStencil3D7pt_f32_flat_tensor",
    "cudnnConvolution3D_f32",
    "cudnnConvolution3D_f32_bias",
    "cudnnConvolution1D_f32_bias",
    "cudnnConvolution2D_f32_dilated",
    "cublasGemmEx_i8_i32_tensor",
    "cublasSnrm2_f32_memref",
    "cublasJointMaxAbsProduct_f32_memref",
    "cudnnFeatureMaskScale_f32_tensor",
    "cudnnConvolutionTranspose2D_f32_memref",
    "cudnnConvolutionTranspose3D_f32_memref",
    "cudnnConvolutionBackwardFilter3D_f32_memref",
    "cudnnDepthwiseConvolution2D_f32_memref",
    "cutensorKroneckerProduct2D_f32_memref",
    "cudnnBinaryCrossEntropyMean_f32_memref",
    "cudnnLogSigmoid_f32_memref",
    "cudnnConvolutionTBC_f32_memref",
    "cudnnConvolutionTBCBackward_f32_memref",
    "cudnnTransformBiasRescaleQKV_f32_memref",
    "cutensornetNetwork_f32_n3_aten",
    "cudnnAddrElementwise_f32_memref",
    "cudnnConvolution2DWindow_f32",
    "cudnnAvgPoolWindow_f32",
    "cudnnAdaptivePool_f32_flat2",
    "cudnnAdaptivePool_f32_flat3_fwd",
    "cudnnAdaptivePool_f32_flat3_bwd",
    "cudnnAdaptivePool_f32_r2",
    "cudnnAdaptivePool_f32_r4_fwd",
    "cudnnAdaptivePool_f32_r4_bwd",
    "cudnnAdaptivePool_f32_r5",
    "cudnnAveragePool_f32_flat2",
    "cudnnAveragePool_f32_r4",
    "cudnnAveragePool_f32_r5",
    "cudnnBatchNormBackward_f32_full",
    "cudnnBatchNormBackward_f32_dx",
    "cufftZ2Z_1D_tensor",
    "cufftC2C_1D_tensor",
    "cutensornetTensorProduct3D_f32_tensor",
    "cutensornetTensorProduct3D_f64_tensor",
    "cudnnConvolutionFwd_batched",
    "cudnnConvolutionFwd_im2col_gemm",
    "cudnnMaxPoolFwd_batched",
    "cudnnBatchNormalizationForwardInference",
    "cudnnAddTensor_batched",
    "cudnnConvBnReluFwdFused",
    "cudnnConvBiasReluAddFwdFused",
    "cudnnPointwiseAffineRelu_f32",
    "cudnnPointwiseGraph_f32",
    "cubInclusiveSum1D_f32_tensor",
    "cubSegmentedInclusiveProduct2D_f32_tensor",
    "cubExclusiveSum1D_i32_memref",
    "cubSegmentedSum_f32_memref",
    "cubSegmentedSum_f64_memref",
    "cubSegmentedMin_f32_memref",
    "cubSegmentedMax_f32_memref",
    "atenSegmentedSum",
    "cubCountNonzero1D_f32_tensor",
    "cubSegmentedCountNonzero2D_f32_tensor",
    "cubSegmentedLogicalAnd_i32_memref",
    "cubEqualAll1D_f32_tensor",
    "cubSegmentedLogicalSelect_i32_tensor",
    "cudnnReduceSum_f32",
    "cudnnReduceSum_f64",
    "cudnnReduceProduct_f32",
    "cudnnReduceMin_f32",
    "cudnnReduceMax_f32",
    "cudnnReduceMinMax_f32",
    "cudnnReduceTrace_f32",
    "cubSegmentedLogicalAnd_i32",
    "cubSegmentedLogicalOr_i32",
    "cubSegmentedBitXor_i32",
    "cubSegmentedPrefixSum_f32",
    "cubSegmentedPrefixLogicalAnd_i32",
    "cublasBroadcastAxis0_f32",
    "cublasBroadcastAxis1_f32",
    "whisperExpShiftSum_f32_tensor",
    "cublasDdot",
    "cublasSdot",
    "cublasSdot_memref",
    "cublasDdot_memref",
    "cublasSgemvTZero_memref",
    "cudnnSoftmaxForward",
    "cudnnSoftmaxForward_tensor",
    "cudnnSoftmaxForwardOut_tensor",
    "cudaCopy1D_f32_tensor",
    "cudaCopy2D_f32_tensor",
    "cudaCopy3D_f32_tensor",
    "cudaCopy6D_f32_tensor",
    "cudaAdd_f32_tensor",
    "cudaMaskSelect_f32_tensor",
    "cudaSwiGLU_f32_tensor",
    "cudaRopeMulMulSub_f32_tensor",
    "cudaRopeMulMulAdd_f32_tensor",
    "cublasLtMatmulBiasReluFused",
    "cublasDsyrk_alias",
    "cublasGemmFor1x1Conv",
    "cutensornetContraction2_f64",
    "cutensornetContraction2_f64_r4r5r4",
    "cutensornetContraction2_f64_r5r4r4",
    "cutensornetContraction2_f64_r5r5r4",
}
ABI_LOWERABLE_KERNELS.update(
    f"cutensorPermute_f32_r{rank}_tensor" for rank in range(2, 7)
)

CUTENSOR_UNARY_OPS = {
    "abs", "acos", "acosh", "asin", "asinh", "atan", "atanh", "ceil",
    "cos", "cosh", "exp", "floor", "log", "mish", "neg", "reciprocal",
    "relu", "sigmoid", "silu", "sin", "sinh", "sqrt", "tan", "tanh",
}
ABI_LOWERABLE_KERNELS.update(
    f"cutensorUnary_{op}_f32" for op in CUTENSOR_UNARY_OPS
)


SEMANTIC_BACKEND_HINTS = {
    # The semantic node is lowered by a custom rewrite into this ABI symbol.
    "miniamr_weighted_27pt_tensor": "cudnnConvolution3D_ntap_tensor",
    "parboil_stencil_7pt_tensor": "cudnnStencil3D7pt_f32_flat_tensor",
}

# Semantic composition names which a custom rewrite below converts to an
# existing ABI-lowerable symbol.  All other names must themselves occur in
# ABI_LOWERABLE_KERNELS before they may participate in production matching.
# This keeps diagnostic patterns from masquerading as library calls or
# shadowing a real backend-capable match.
COMPOSITION_LOWERING_ADAPTERS = {
    "miniamr_weighted_27pt_tensor",
    "parboil_stencil_7pt_tensor",
    "cublasDcopy_tensor",
    "tensor_copy_2D",
    "tensor_copy_3D",
    "tensor_copy_6D",
    "reduce_sum_1D",
}


def _candidate_backend(cand) -> str | None:
    backend = SEMANTIC_BACKEND_HINTS.get(cand.name)
    if cand.name in ABI_LOWERABLE_KERNELS:
        return cand.name
    if backend in ABI_LOWERABLE_KERNELS:
        return backend
    return None


def _format_candidate_for_report(cand, include_semantic_only: bool = False) -> str:
    backend = _candidate_backend(cand)
    if cand.name in ABI_LOWERABLE_KERNELS:
        status = "abi-lowerable"
    elif backend in ABI_LOWERABLE_KERNELS:
        status = "backend-candidate"
    elif include_semantic_only:
        status = "semantic-debug"
    else:
        status = "unusable"
    parts = [
        cand.name,
        f"kind={cand.match_kind}",
        f"coverage={cand.coverage}",
        f"status={status}",
    ]
    if backend:
        parts.append(f"backend={backend}")
    if cand.source:
        parts.append(f"source={cand.source}")
    if cand.subterm_path:
        parts.append("path=" + ".".join(str(i) for i in cand.subterm_path))
    if cand.defaults:
        defaults = ";".join(f"{k}={v}" for k, v in cand.defaults)
        parts.append(f"defaults={defaults}")
    return "  ".join(parts)


# Match each linalg.generic at the IR level, capturing the full block so
# we can substitute it with a `kernel.launch`. Handles BOTH:
#   - tensor form: `%X = linalg.generic {...} ins(...) outs(...) {body} -> T`
#   - memref form: `linalg.generic {...} ins(...) outs(...) {body}`
# (no SSA prefix, no return type; the op is void and mutates `outs` in place).
# The leading SSA `%X =` and the trailing `-> type` are both optional.
_GENERIC_BLOCK_RE = re.compile(
    r"(\s*)(?:(%[\w_]+)(?::(\d+))?\s*=\s*)?"
    r"linalg\.generic\s*\{[^}]*\}\s*"
    r"(?:ins\(([^)]*)\)\s*)?"
    r"outs\(([^)]*)\)\s*"
    # linalg.yield captures one OR MORE comma-separated SSA operands —
    # matches kernel_match.py's _GEN_RE, needed so multi-yield bodies
    # (e.g. softmax's fused exp+sum) aren't dropped or partially-consumed
    # by the .*? backtracking. Single-yield bodies still match unchanged.
    r"\{\s*\^bb0\([^)]*\)\s*:.*?linalg\.yield\s+%[\w_]+(?:\s*,\s*%[\w_]+)*\s*:[^}]*\}"
    r"(?:\s*->\s*([^\n]+))?",
    re.DOTALL,
)


@dataclass
class LinalgInstance:
    """A single linalg.generic op extracted from the MLIR text."""
    result_ssa: str | None  # %12 etc., or None for memref-form (void)
    result_count: int       # MLIR multi-result count (`%x:2 = ...`)
    ins_part: str           # "%10, %11 : tensor<?x...>, tensor<...>"
    outs_part: str          # "%9 : tensor<...>" or "%9 : memref<...>"
    result_type: str | None # the type after `->`, or None for memref-form
    span: tuple[int, int]   # offset range in the source text
    indent: str             # leading whitespace before the op


def _cyclic_shift_1d_spec(
    body: GenericBody,
) -> tuple[str, str, int] | None:
    """Recognize ``out[i] = input[(i + shift) mod N]``.

    cgeist expands signed remainder into a fairly noisy div/select sequence.
    Rather than depending on temporary SSA names, interpret that integer DAG
    and prove the resulting index map on its breakpoints and a dense prefix.
    The denominator of the signed division supplies the fixed extent ``N``.
    """
    if (body.ins_arg_names or len(body.outs_arg_names) != 1 or
            body.iterator_types != ["parallel"]):
        return None
    defs: dict[str, tuple] = {}
    load: tuple[str, str, str] | None = None
    modulus_candidates: set[int] = set()
    for raw in body.body_lines:
        line = raw.strip()
        m = re.match(r"(%[\w.$-]+)\s*=\s*linalg\.index\s+0\s*:", line)
        if m:
            defs[m.group(1)] = ("index",)
            continue
        m = re.match(
            r"(%[\w.$-]+)\s*=\s*arith\.(addi|subi|muli|divsi|remsi)\s+"
            r"(%[\w.$-]+),\s*(%[\w.$-]+)\s*:", line)
        if m:
            defs[m.group(1)] = (m.group(2), m.group(3), m.group(4))
            if m.group(2) in ("divsi", "remsi"):
                value = body.constants.get(m.group(4))
                if value is not None and int(value) == value and value > 1:
                    modulus_candidates.add(int(value))
            continue
        m = re.match(
            r"(%[\w.$-]+)\s*=\s*arith\.cmpi\s+(\w+),\s*"
            r"(%[\w.$-]+),\s*(%[\w.$-]+)\s*:", line)
        if m:
            defs[m.group(1)] = ("cmp", m.group(2), m.group(3), m.group(4))
            continue
        m = re.match(
            r"(%[\w.$-]+)\s*=\s*arith\.select\s+(%[\w.$-]+),\s*"
            r"(%[\w.$-]+),\s*(%[\w.$-]+)\s*:", line)
        if m:
            defs[m.group(1)] = ("select", m.group(2), m.group(3), m.group(4))
            continue
        m = re.match(
            r"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+(%[\w.$-]+)\s*:",
            line)
        if m:
            defs[m.group(1)] = ("cast", m.group(2))
            continue
        m = re.match(
            r"(%[\w.$-]+)\s*=\s*memref\.load\s+(%[\w.$-]+)"
            r"\[(%[\w.$-]+)\]\s*:\s*(memref<[^>]+>)", line)
        if m and m.group(4).endswith("xf32>"):
            load = (m.group(2), m.group(4), m.group(3))
    if load is None or len(modulus_candidates) != 1:
        return None
    extent = next(iter(modulus_candidates))

    def evaluate(name: str, index: int, memo: dict[str, int]) -> int:
        if name in memo:
            return memo[name]
        if name in body.constants:
            value = int(body.constants[name])
        else:
            op = defs.get(name)
            if op is None:
                raise ValueError(name)
            if op[0] == "index":
                value = index
            elif op[0] == "cast":
                value = evaluate(op[1], index, memo)
            elif op[0] in ("addi", "subi", "muli", "divsi", "remsi"):
                lhs = evaluate(op[1], index, memo)
                rhs = evaluate(op[2], index, memo)
                if op[0] == "addi": value = lhs + rhs
                elif op[0] == "subi": value = lhs - rhs
                elif op[0] == "muli": value = lhs * rhs
                elif op[0] == "divsi": value = int(lhs / rhs)
                else: value = lhs - int(lhs / rhs) * rhs
            elif op[0] == "cmp":
                lhs = evaluate(op[2], index, memo)
                rhs = evaluate(op[3], index, memo)
                pred = op[1]
                value = int(lhs < rhs if pred in ("slt", "ult") else
                            lhs <= rhs if pred in ("sle", "ule") else
                            lhs > rhs if pred in ("sgt", "ugt") else
                            lhs >= rhs if pred in ("sge", "uge") else
                            lhs == rhs if pred == "eq" else lhs != rhs)
            elif op[0] == "select":
                cond = evaluate(op[1], index, memo)
                value = evaluate(op[2] if cond else op[3], index, memo)
            else:
                raise ValueError(op[0])
        memo[name] = value
        return value

    try:
        shift = evaluate(load[2], 0, {}) % extent
        probes = set(range(min(extent, 257)))
        probes.update({extent // 2, max(0, extent - 2), extent - 1})
        if any(evaluate(load[2], i, {}) != (i + shift) % extent
               for i in probes):
            return None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None
    return load[0], load[1], shift


def _conditional_flip_2d_spec(
    body: GenericBody,
) -> tuple[str, str, str, str] | None:
    """Recognize independent runtime-controlled reflection of two axes."""
    if (body.ins_arg_names or len(body.outs_arg_names) != 1 or
            body.iterator_types != ["parallel", "parallel"]):
        return None
    indexes: dict[int, str] = {}
    to_i32: dict[str, str] = {}
    reflected: dict[str, str] = {}
    selected: dict[str, tuple[str, str, str]] = {}
    to_index: dict[str, str] = {}
    load: tuple[str, str, list[str]] | None = None
    for raw in body.body_lines:
        line = raw.strip()
        m = re.match(r"(%[\w.$-]+)\s*=\s*linalg\.index\s+([01])\s*:", line)
        if m:
            indexes[int(m.group(2))] = m.group(1)
            continue
        m = re.match(
            r"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+(%[\w.$-]+)\s*:"
            r"\s*index\s+to\s+i32", line)
        if m:
            to_i32[m.group(1)] = m.group(2)
            continue
        m = re.match(
            r"(%[\w.$-]+)\s*=\s*arith\.subi\s+(%[\w.$-]+),\s*"
            r"(%[\w.$-]+)\s*:\s*i32", line)
        if m and m.group(2) in body.constants:
            reflected[m.group(1)] = m.group(3)
            continue
        m = re.match(
            r"(%[\w.$-]+)\s*=\s*arith\.select\s+(%[\w.$-]+),\s*"
            r"(%[\w.$-]+),\s*(%[\w.$-]+)\s*:\s*i32", line)
        if m:
            selected[m.group(1)] = (m.group(2), m.group(3), m.group(4))
            continue
        m = re.match(
            r"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+(%[\w.$-]+)\s*:"
            r"\s*i32\s+to\s+index", line)
        if m:
            to_index[m.group(1)] = m.group(2)
            continue
        m = re.match(
            r"(%[\w.$-]+)\s*=\s*memref\.load\s+(%[\w.$-]+)"
            r"\[([^]]+)\]\s*:\s*(memref<[^>]+>)", line)
        if m and m.group(4).endswith("xf32>"):
            load = (m.group(2), m.group(4),
                    [part.strip() for part in m.group(3).split(",")])
    if load is None or len(load[2]) != 2 or set(indexes) != {0, 1}:
        return None
    flags: list[str] = []
    for dim, load_index in enumerate(load[2]):
        selected_name = to_index.get(load_index)
        choice = selected.get(selected_name or "")
        if choice is None:
            return None
        flag, reflected_name, direct_name = choice
        original_i32 = next(
            (name for name, source in to_i32.items()
             if source == indexes[dim]), None)
        if (original_i32 is None or direct_name != original_i32 or
                reflected.get(reflected_name) != original_i32):
            return None
        flags.append(flag)
    return load[0], load[1], flags[0], flags[1]


def _dilated_conv2d_factors(text: str, window_ssa: str) -> tuple[int, int] | None:
    """Recover constant spatial dilation from a rank-6 submap window."""
    use = re.search(
        rf"{re.escape(window_ssa)}\s*=\s*polygeist\.submap\([^\n]*"
        rf"\{{map\s*=\s*(#[\w.$-]+)\}}", text)
    if use is None:
        return None
    definition = re.search(
        rf"^{re.escape(use.group(1))}\s*=\s*affine_map<[^\n]*->\s*"
        rf"\(d3,\s*d4\s*\*\s*(\d+)\s*\+\s*d1,\s*"
        rf"d5\s*\*\s*(\d+)\s*\+\s*d2\)>", text, re.MULTILINE)
    if definition is None:
        return None
    return int(definition.group(1)), int(definition.group(2))


def _batchnorm_inference_operand_order(body: GenericBody) -> list[int] | None:
    """Bind x/weight/mean/invstd/bias roles from the scalar dataflow."""
    if (len(body.ins_arg_names) != 5 or len(body.outs_arg_names) != 1 or
            body.iterator_types != ["parallel"] * 4 or
            len(body.indexing_maps) != 6):
        return None
    full = body.indexing_maps[-1]
    if body.indexing_maps[0] != full or any(
            m == full for m in body.indexing_maps[1:5]):
        return None
    text = "\n".join(line.strip() for line in body.body_lines)
    name = r"(%[\w.$-]+)"
    sub = re.search(rf"{name}\s*=\s*arith\.subf\s+{name},\s*{name}", text)
    if sub is None:
        return None
    centered, x, mean = sub.groups()

    def binary_user(op: str, value: str) -> tuple[str, str] | None:
        for found in re.finditer(
                rf"{name}\s*=\s*arith\.{op}\s+{name},\s*{name}", text):
            result, lhs, rhs = found.groups()
            if lhs == value:
                return result, rhs
            if rhs == value:
                return result, lhs
        return None

    scale0 = binary_user("mulf", centered)
    if scale0 is None:
        return None
    scaled0, invstd = scale0
    scale1 = binary_user("mulf", scaled0)
    if scale1 is None:
        return None
    scaled1, weight = scale1
    add = binary_user("addf", scaled1)
    if add is None:
        return None
    result, bias = add
    if body.yield_values != [result]:
        return None
    try:
        return [body.ins_arg_names.index(v)
                for v in (x, weight, mean, invstd, bias)]
    except ValueError:
        return None


def _feature_mask_scale_capture(body: GenericBody) -> str | None:
    """Recognize x[n,c,h,w] * mask[n,c] * scalar."""
    if (len(body.ins_arg_names) != 2 or len(body.outs_arg_names) != 1 or
            body.iterator_types != ["parallel"] * 4 or
            len(body.indexing_maps) != 3 or
            body.indexing_maps[0] != body.indexing_maps[2] or
            body.indexing_maps[1] == body.indexing_maps[2]):
        return None
    text = "\n".join(body.body_lines)
    x, mask = body.ins_arg_names
    product = re.search(
        rf"(%[\w.$-]+)\s*=\s*arith\.mulf\s+{re.escape(x)},\s*"
        rf"{re.escape(mask)}\s*:\s*f32", text)
    if product is None:
        return None
    scaled = re.search(
        rf"(%[\w.$-]+)\s*=\s*arith\.mulf\s+"
        rf"{re.escape(product.group(1))},\s*(%[\w.$-]+)\s*:\s*f32", text)
    if scaled is None or body.yield_values != [scaled.group(1)]:
        return None
    return scaled.group(2)


def _linear_resample_1d_spec(
    body: GenericBody, constants: dict[str, float]
) -> tuple[str, str, int] | None:
    """Recognize ATen's align_corners=false linear 1-D interpolation."""
    if (body.ins_arg_names or len(body.outs_arg_names) != 1 or
            not body.iterator_types or
            any(it != "parallel" for it in body.iterator_types)):
        return None
    text = "\n".join(body.body_lines)
    loads = re.findall(
        r"memref\.load\s+(%[\w.$-]+)\[[^]]+\]\s*:\s*(memref<[^>]+xf32>)",
        text)
    if (len(loads) != 2 or loads[0] != loads[1] or
            "arith.fptosi" not in text or "arith.divf" not in text or
            text.count("arith.mulf") < 3 or "arith.addf" not in text):
        return None
    input_dim = None
    for m in re.finditer(r"arith\.cmpi\s+slt,\s+%[\w.$-]+,\s+(%[\w.$-]+)",
                         text):
        value = constants.get(m.group(1))
        if value is not None and value >= 1 and float(value).is_integer():
            input_dim = int(value)
    if input_dim is None:
        return None
    return loads[0][0], loads[0][1], input_dim


def _grid_sample_bilinear_2d_spec(
    body: GenericBody,
) -> tuple[str, str] | None:
    """Recognize a zero-padded, align-corners bilinear grid sample."""
    if (len(body.ins_arg_names) != 2 or len(body.outs_arg_names) != 1 or
            len(body.iterator_types) != 3 or
            any(it != "parallel" for it in body.iterator_types)):
        return None
    text = "\n".join(body.body_lines)
    loads = re.findall(
        r"memref\.load\s+(%[\w.$-]+)\[[^]]+\]\s*:\s*(memref<[^>]+xf32>)",
        text)
    if (len(loads) != 4 or len(set(loads)) != 1 or
            text.count("arith.fptosi") != 2 or
            text.count("arith.select") < 4 or
            text.count("arith.mulf") < 8):
        return None
    return loads[0]


def _extract_ssa_names(operands_part: str) -> list[str]:
    """Pull SSA names from a `%a, %b : type, type` string."""
    if not operands_part:
        return []
    head = operands_part.split(":", 1)[0]
    return [tok.strip() for tok in head.split(",") if tok.strip()]


def _extract_ssa_types(operands_part: str) -> list[str]:
    """Pull operand types from a `%a, %b : type, type` string."""
    if not operands_part or ":" not in operands_part:
        return []
    _, tail = operands_part.split(":", 1)
    # Split on top-level commas (respect angle-bracket nesting in MLIR types).
    types, depth, cur = [], 0, []
    for c in tail:
        if c == ',' and depth == 0:
            t = ''.join(cur).strip()
            if t:
                types.append(t)
            cur = []
            continue
        if c in '<(':
            depth += 1
        elif c in '>)':
            depth -= 1
        cur.append(c)
    t = ''.join(cur).strip()
    if t:
        types.append(t)
    return types


def _scan_scalar_types(text: str) -> dict[str, str]:
    """Best-effort SSA→type map for scalar values (function args + arith.constant).

    Captures only the kinds of SSA values that show up as Cap operands in the
    matcher's emit (alphas, betas, etc.) — i.e. things that have a primitive
    f32/f64/index/integer type rather than a tensor/memref. Good enough to
    annotate kernel.launch operand types so polygeist-opt can parse the op.
    """
    out: dict[str, str] = {}
    # Function arguments: "func.func @name(%arg0: i32, %arg3: f64, ...)" — capture all.
    for m in re.finditer(r'%\w+\s*:\s*([a-zA-Z_][\w.]*[!<>?x\d,\s]*)', text):
        # Re-scope: only inside func.func parameter lists. Just match more carefully.
        pass
    for fm in re.finditer(r'func\.func\s+@\w+\s*\(([^)]*)\)', text):
        params = fm.group(1)
        for pm in re.finditer(r'(%[\w]+)\s*:\s*([^,)]+)', params):
            out[pm.group(1).strip()] = pm.group(2).strip()
    # arith.constant lines: "%X = arith.constant ... : f64". Allow `-` in
    # SSA names since cgeist emits things like `%c-8_i32` for negatives.
    for cm in re.finditer(r'(%[\w\-]+)\s*=\s*arith\.constant\s+\S+\s*:\s*(\S+)', text):
        out[cm.group(1)] = cm.group(2)
    for bm in re.finditer(
            r'(%[\w\-]+)\s*=\s*arith\.constant\s+(?:true|false)\s*$',
            text,
            re.MULTILINE):
        out[bm.group(1)] = "i1"
    # affine.load on a scalar memref: "%X = affine.load %alloca[] : memref<f32>"
    # The result type is the element type of the memref. Softmax binds its
    # max/sum captures via this pattern (the loop reduces into a memref<f32>,
    # then loads back the scalar to feed the next generic).
    for lm in re.finditer(
            r'(%[\w\-]+)\s*=\s*affine\.load\s+%[\w\-]+\[\]\s*:\s*memref<([^,>]+)(?:,[^>]*)?>',
            text):
        out[lm.group(1)] = lm.group(2).strip()
    for tm in re.finditer(
            r'(%[\w\-]+)\s*=\s*tensor\.extract\s+%[\w\-]+(?:#[0-9]+)?(?:\[[^\]]*\])?\s*:\s*tensor<([^>]+)>',
            text):
        elem = tm.group(2).strip().rsplit("x", 1)[-1]
        out[tm.group(1)] = elem
    # Scalar-producing arith / math ops between linalg.generics. RMSNorm
    # binds its %scale capture to a chain `divf(ss, N); addf(_, eps);
    # sqrt(_); divf(1.0, _)` that lives in the function body but outside
    # any linalg.generic. The matcher Cap binds to the final SSA, and we
    # need its type for the launch op signature. Match `%X = <op> ... : T`
    # for the common scalar arith ops (avoid being so broad that we
    # accidentally type memref/tensor SSAs).
    _scalar_op_pat = re.compile(
        r'(%[\w\-]+)\s*=\s*'
        r'(?:arith\.(?:add[fi]|sub[fi]|mul[fi]|div[fsui]+|negf|select|cmp[fi]|'
        r'extf|extsi|extui|trunci|truncf|sitofp|uitofp|fptosi|fptoui|bitcast)'
        r'|math\.(?:sqrt|exp|log|tanh|absf|absi))'
        r'\s+\S[^\n]*?:\s*([a-zA-Z][\w]*)\s*$',
        re.MULTILINE)
    for sm in _scalar_op_pat.finditer(text):
        out[sm.group(1)] = sm.group(2).strip()
    return out


def _enclosing_func_args(text: str, pos: int) -> list[tuple[str, str]]:
    """Best-effort function-argument list for the func containing `pos`.

    The Darknet im2col+GEMM fused rewrite needs the original scalar shape
    parameters, which cgeist emits as the first seven function arguments:
    channels, height, width, out_channels, ksize, stride, pad.
    """
    matches = list(re.finditer(r'func\.func\s+@\w+\s*\(([^)]*)\)', text[:pos]))
    if not matches:
        return []
    params = matches[-1].group(1)
    out: list[tuple[str, str]] = []
    for pm in re.finditer(r'(%[\w_\-]+)\s*:\s*([^,)]+)', params):
        out.append((pm.group(1).strip(), pm.group(2).strip()))
    return out


def _extract_guarded_im2col_input(body_lines: list[str]) -> tuple[str, str] | None:
    """Find the source memref loaded by the guarded im2col linalg body."""
    body = "\n".join(body_lines)
    m = re.search(
        r'memref\.load\s+(%[\w_\-]+)\[[^\]]*\]\s*:\s*(memref<[^>]+>)',
        body,
    )
    if not m:
        return None
    return m.group(1), m.group(2)


def _extract_cmpi_rhs_i32(body_lines: list[str]) -> str | None:
    """Find the RHS scalar in a linalg-index comparison like `i > %pos`."""
    for line in body_lines:
        m = re.search(r'arith\.cmpi\s+\w+,\s+%[\w_\-]+,\s+(%[\w_\-]+)\s*:', line)
        if m:
            return m.group(1)
    return None


def collect_generics_with_spans(text: str) -> list[LinalgInstance]:
    """Return every linalg.generic in `text`, in source order, with span."""
    out: list[LinalgInstance] = []
    for m in _GENERIC_BLOCK_RE.finditer(text):
        indent, result_ssa, result_count, ins, outs, rty = m.groups()
        out.append(LinalgInstance(
            result_ssa=result_ssa,
            result_count=int(result_count) if result_count else (
                1 if result_ssa else 0
            ),
            ins_part=(ins or "").strip(),
            outs_part=outs.strip(),
            result_type=rty.strip() if rty else None,
            span=m.span(),
            indent=indent,
        ))
    return out


_STRIDED_2D_TARGET = "memref<?x?xf64, strided<[?, 1], offset: ?>>"
_STRIDED_3D_TARGET = "memref<?x?x?xf64, strided<[?, ?, 1], offset: ?>>"


def _sniff_elem_type(memref_or_tensor_ty: str) -> str | None:
    """Extract the element type from a memref/tensor textual type.

    Examples:
      `memref<?x?xf64, strided<[?, 1], offset: ?>>`  →  "f64"
      `memref<?x?xf32, strided<[256, 1]>>`            →  "f32"
      `tensor<?x?xf16>`                                →  "f16"
      `tensor<?xbf16>`                                 →  "bf16"
      `memref<?x?xi32>`                                →  "i32"

    Returns None if the type doesn't parse as memref/tensor.
    """
    m = re.match(r'(?:memref|tensor)<(.+)>', memref_or_tensor_ty.strip())
    if not m:
        return None
    body = m.group(1)
    depth = 0
    head = []
    for c in body:
        if c == "," and depth == 0:
            break
        if c in "<([":
            depth += 1
        elif c in ">)]":
            depth -= 1
        head.append(c)
    shaped = "".join(head).strip()
    return shaped.rsplit("x", 1)[-1].strip() if "x" in shaped else shaped


def _shaped_rank(ty: str) -> int:
    """Return the rank of a simple tensor/memref spelling, or -1."""
    if not (ty.startswith("tensor<") or ty.startswith("memref<")):
        return -1
    inside = ty[ty.find("<") + 1:ty.rfind(">")]
    shape_and_elem = inside.split(",", 1)[0]
    if "x" not in shape_and_elem:
        return 0
    return shape_and_elem.rsplit("x", 1)[0].count("x") + 1


# A library call has a fixed launch/dispatch cost.  Replacing a three- or
# five-element loop with an individual cuBLAS call is therefore predictably
# unprofitable, especially when that call remains inside an application loop.
# Keep this deliberately conservative: it applies only when every linalg
# iterator extent can be proved from static operand types or raised view sizes.
_MIN_STANDALONE_BLAS_ITERATIONS = 256


def _static_shaped_extents(
    text: str, operand: str, operand_type: str, before: int
) -> list[int] | None:
    """Recover the logical shape of one generic operand, when fully static."""
    rank = _shaped_rank(operand_type)
    if rank < 0:
        return None
    if rank == 0:
        return []

    # SSA names are function-local and cgeist reuses names such as `%2` in
    # every function.  Limit lookup to the enclosing function so an earlier
    # function's same-named view cannot supply false static evidence.
    function_start = text.rfind("func.func", 0, before)
    scope = text[function_start:before] if function_start >= 0 else text[:before]
    view = _parse_memref_view(scope, operand, len(scope))
    if view is not None and view["kind"] in ("submap", "subview"):
        # A submap may carry leading map-symbol/offset operands before its
        # logical result sizes.  The final `rank` operands are the sizes.
        size_tokens = view["sizes"][-rank:]
        if len(size_tokens) != rank:
            return None
        extents: list[int] = []
        for token in size_tokens:
            if re.fullmatch(r"\d+", token):
                value = int(token)
            else:
                value = _constant_index_value(text[:before], token)
            if value is None or value < 0:
                return None
            extents.append(value)
        return extents

    payload = _type_payload(operand_type.strip(), "memref")
    if payload is None:
        payload = _type_payload(operand_type.strip(), "tensor")
    if payload is None:
        return None
    shaped = _top_level_first_type_piece(payload)
    pieces = shaped.rsplit("x", 1)[0].split("x")
    if len(pieces) != rank or any(not piece.isdigit() for piece in pieces):
        return None
    return [int(piece) for piece in pieces]


def _static_generic_iteration_count(
    text: str, inst: LinalgInstance, body
) -> int | None:
    """Return the linalg iteration-domain cardinality when it is provable.

    Indexing maps connect operand dimensions to generic iterator dimensions.
    We intentionally accept only direct `dN` projections; affine windows,
    broadcasts, or incomplete dynamic evidence return unknown and are not
    profitability-rejected.
    """
    operands = (_extract_ssa_names(inst.ins_part) +
                _extract_ssa_names(inst.outs_part))
    operand_types = (_extract_ssa_types(inst.ins_part) +
                     _extract_ssa_types(inst.outs_part))
    if not (len(operands) == len(operand_types) == len(body.indexing_maps)):
        return None

    iterator_extents: dict[int, int] = {}
    for operand, operand_type, map_text in zip(
            operands, operand_types, body.indexing_maps):
        extents = _static_shaped_extents(
            text, operand, operand_type, inst.span[0])
        map_match = re.fullmatch(
            r"affine_map<\([^)]*\)\s*->\s*\(([^)]*)\)>", map_text.strip())
        if extents is None or map_match is None:
            return None
        outputs = ([piece.strip() for piece in map_match.group(1).split(",")]
                   if map_match.group(1).strip() else [])
        if len(outputs) != len(extents):
            return None
        for output, extent in zip(outputs, extents):
            dim_match = re.fullmatch(r"d(\d+)", output)
            if dim_match is None:
                # Scalar/broadcast maps provide no extent evidence but are
                # harmless; non-trivial affine expressions are ambiguous.
                if output:
                    return None
                continue
            dim = int(dim_match.group(1))
            previous = iterator_extents.get(dim)
            if previous is not None and previous != extent:
                return None
            iterator_extents[dim] = extent

    if set(iterator_extents) != set(range(len(body.iterator_types))):
        return None
    count = 1
    for dim in range(len(body.iterator_types)):
        count *= iterator_extents[dim]
    return count


def _is_standalone_blas_candidate(name: str) -> bool:
    """Classify fixed-cost BLAS calls that need a minimum useful workload."""
    lower = name.lower()
    if "strided_batched" in lower or "broadcast3d" in lower:
        return False
    return lower.startswith("cublas") and any(
        operation in lower for operation in ("dot", "gemv", "gemm"))


def _is_scalar_alias_submap(text: str, operand: str) -> bool:
    """Prove a logical rank-1 memref view aliases one physical scalar."""
    definition = re.search(
        rf"(?s)^\s*{re.escape(operand)}\s*=\s*polygeist\.submap"
        r"\([^)]*\)\s*\{map\s*=\s*([^}]+)\}",
        text, re.MULTILINE,
    )
    if not definition:
        return False
    map_text = _resolve_affine_map_text(text, definition.group(1).strip())
    if not map_text:
        return False
    compact = re.sub(r"\s+", "", map_text)
    return compact == "affine_map<(d0)->()>"


def _normalize_memref_operands(
    operands: list[str], operand_types: list[str] | None, indent: str
) -> tuple[list[str], list[str], list[str]]:
    """For each strided memref operand, emit a memref.cast to a uniform
    `memref<?x?x...xT, strided<[?, ..., 1], offset: ?>>` target type, so the
    launch's operand types match the canonical kernel.defn declaration's
    dynamic-stride placeholder pattern.

    Element-type-aware: handles f64, f32, f16, bf16, i32, i16, i8, i64.
    Operands not matching the strided-memref pattern are passed through
    unchanged.

    Returns (cast_lines, new_operand_ssas, new_operand_types).
    """
    if operand_types is None or len(operand_types) != len(operands):
        return [], operands, operand_types or []
    cast_lines: list[str] = []
    new_ssas: list[str] = []
    new_types: list[str] = []
    # Match memref<?x?xT, ...> or memref<?x?x?xT, ...> with strided layout.
    # Capture (rank-dims-prefix, element-type).
    rank_pat = re.compile(r"memref<((?:\?x)+)([\w_]+)(?:,\s*strided<|>)")
    for ssa, ty in zip(operands, operand_types):
        if not ty.startswith("memref<") or "strided<[" not in ty:
            new_ssas.append(ssa); new_types.append(ty); continue
        m = rank_pat.match(ty)
        if not m:
            new_ssas.append(ssa); new_types.append(ty); continue
        rank_prefix = m.group(1)         # e.g. "?x?x" for rank-2 dynamic
        elem = m.group(2)                # e.g. "f32" / "f64" / "i32"
        rank = rank_prefix.count("?")
        # Build target: strided<[?, ..., 1], offset: ?> — all row strides
        # dynamic, last (innermost) stride statically 1 (row-major, contiguous
        # within innermost dim).
        if rank < 1:
            new_ssas.append(ssa); new_types.append(ty); continue
        layout = re.search(r"strided<\[([^]]+)\]", ty)
        innermost_dynamic = bool(
            layout and layout.group(1).split(",")[-1].strip() == "?")
        if innermost_dynamic:
            strides = "[" + ", ".join(["?"] * rank) + "]"
        elif rank == 1:
            strides = "[1]"
        else:
            strides = "[" + ", ".join(["?"] * (rank - 1)) + ", 1]"
        target = f"memref<{rank_prefix}{elem}, strided<{strides}, offset: ?>>"
        if ty == target:
            new_ssas.append(ssa); new_types.append(ty); continue
        cast_ssa = ssa + "_c"
        cast_lines.append(
            f"{indent}{cast_ssa} = memref.cast {ssa} : {ty} to {target}"
        )
        new_ssas.append(cast_ssa)
        new_types.append(target)
    return cast_lines, new_ssas, new_types


def _derived_ssa_name(ssa: str, suffix: str) -> str:
    """Create a readable SSA name derived from an existing textual SSA."""
    base = ssa[1:] if ssa.startswith("%") else ssa
    base = re.sub(r"\W", "_", base)
    if not base or base[0].isdigit():
        base = "v" + base
    return f"%{base}_{suffix}"


def _dynamic_tensor_type(ty: str) -> str | None:
    """Return an all-dynamic tensor type with the same rank/element type."""
    if not ty.startswith("tensor<"):
        return None
    m = re.match(r"tensor<(.+)>", ty.strip())
    if not m:
        return None
    shaped = m.group(1).strip()
    # Keep scalar tensors and complex element encodings unchanged. The kernel
    # library defns we need to normalize against are plain ranked tensors.
    if "x" not in shaped or "*" in shaped or "<" in shaped:
        return ty
    elem = shaped.rsplit("x", 1)[-1].strip()
    shape = shaped[:-(len(elem) + 1)]
    dims = [d.strip() for d in shape.split("x") if d.strip()]
    if not dims:
        return ty
    return "tensor<" + "x".join("?" for _ in dims) + "x" + elem + ">"


def _complex1d_tensor_type(ty: str) -> str | None:
    """Return tensor<?x2xT> for tensor<Nx2xT>; reject other layouts."""
    if not ty.startswith("tensor<"):
        return None
    m = re.match(r"tensor<(.+)>", ty.strip())
    if not m:
        return None
    shaped = m.group(1).strip()
    if "<" in shaped or "x" not in shaped:
        return None
    elem = shaped.rsplit("x", 1)[-1].strip()
    dims = shaped[:-(len(elem) + 1)].split("x")
    dims = [d.strip() for d in dims if d.strip()]
    if len(dims) != 2 or dims[1] != "2":
        return None
    return f"tensor<?x2x{elem}>"


def _normalize_tensor_operands(
    operands: list[str], operand_types: list[str] | None, indent: str
) -> tuple[list[str], list[str], list[str]]:
    """Erase static tensor extents with tensor.cast for kernel.defn matching."""
    if operand_types is None or len(operand_types) != len(operands):
        return [], operands, operand_types or []
    cast_lines: list[str] = []
    new_ssas: list[str] = []
    new_types: list[str] = []
    for idx, (ssa, ty) in enumerate(zip(operands, operand_types)):
        target = _dynamic_tensor_type(ty)
        if target is None or target == ty:
            new_ssas.append(ssa)
            new_types.append(ty)
            continue
        cast_ssa = _derived_ssa_name(ssa, f"tc{idx}")
        cast_lines.append(
            f"{indent}{cast_ssa} = tensor.cast {ssa} : {ty} to {target}"
        )
        new_ssas.append(cast_ssa)
        new_types.append(target)
    return cast_lines, new_ssas, new_types


def _normalize_complex1d_tensor_operands(
    operands: list[str], operand_types: list[str], indent: str
) -> tuple[list[str], list[str], list[str]] | None:
    if len(operands) != len(operand_types):
        return None
    cast_lines: list[str] = []
    new_ssas: list[str] = []
    new_types: list[str] = []
    for idx, (ssa, ty) in enumerate(zip(operands, operand_types)):
        target = _complex1d_tensor_type(ty)
        if target is None:
            return None
        if target == ty:
            new_ssas.append(ssa)
            new_types.append(ty)
            continue
        cast_ssa = _derived_ssa_name(ssa, f"fft_tc{idx}")
        cast_lines.append(
            f"{indent}{cast_ssa} = tensor.cast {ssa} : {ty} to {target}"
        )
        new_ssas.append(cast_ssa)
        new_types.append(target)
    return cast_lines, new_ssas, new_types


def _parse_static_subview_offset(text: str, ssa: str) -> tuple[str, tuple[int, int]] | None:
    pat = re.compile(
        rf"^\s*{re.escape(ssa)}\s*=\s*memref\.subview\s+"
        rf"(%[\w_\-]+)\s*\[([^\]]+)\]",
        re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        return None
    pieces = [p.strip() for p in m.group(2).split(",")]
    if len(pieces) != 2:
        return None
    try:
        return m.group(1), (int(pieces[0]), int(pieces[1]))
    except ValueError:
        return None


def _parse_static_extract_slice_offset(
    text: str, ssa: str
) -> tuple[str, tuple[int, int]] | None:
    pat = re.compile(
        rf"^\s*{re.escape(ssa)}\s*=\s*tensor\.extract_slice\s+"
        rf"(%[\w_\-]+)\s*\[([^\]]+)\]",
        re.MULTILINE,
    )
    m = pat.search(text)
    if not m:
        return None
    pieces = [p.strip() for p in m.group(2).split(",")]
    if len(pieces) != 2:
        return None
    try:
        return m.group(1), (int(pieces[0]), int(pieces[1]))
    except ValueError:
        return None


def _parse_static_tensor_submap_offset(
    text: str, ssa: str
) -> tuple[str, tuple[int, int]] | None:
    """Recover the constant shift encoded by a rank-2 tensor submap map."""
    match = re.search(
        rf"^\s*{re.escape(ssa)}\s*=\s*polygeist\.submap\s*"
        rf"\(\s*(%[\w_\-]+)[^)]*\)\s*"
        r"\{[^}]*map\s*=\s*([^}]+)\}\s*:",
        text,
        re.MULTILINE,
    )
    if not match:
        return None
    base, map_ref = match.groups()
    map_text = _resolve_affine_map_text(text, map_ref)
    if map_text is None:
        return None
    parsed = re.fullmatch(
        r"affine_map<\(d0\s*,\s*d1\)\s*->\s*\(([^,]+),\s*([^)]+)\)>",
        map_text.strip(),
    )
    if not parsed:
        return None

    def shift(expr: str, dim: str) -> int | None:
        compact = re.sub(r"\s+", "", expr)
        if compact == dim:
            return 0
        shifted = re.fullmatch(rf"{dim}([+-]\d+)", compact)
        return int(shifted.group(1)) if shifted else None

    y = shift(parsed.group(1), "d0")
    x = shift(parsed.group(2), "d1")
    return (base, (y, x)) if y is not None and x is not None else None


def _affine_polynomial(expr: str) -> dict[tuple[int, ...], int] | None:
    """Parse the small integer-polynomial subset used by flattened submaps."""
    variables = ("d0", "d1", "d2", "s0", "s1")
    zero_power = (0,) * len(variables)

    def add(lhs, rhs, scale=1):
        out = dict(lhs)
        for monomial, coefficient in rhs.items():
            out[monomial] = out.get(monomial, 0) + scale * coefficient
            if out[monomial] == 0:
                del out[monomial]
        return out

    def multiply(lhs, rhs):
        out = {}
        for lm, lc in lhs.items():
            for rm, rc in rhs.items():
                monomial = tuple(a + b for a, b in zip(lm, rm))
                out[monomial] = out.get(monomial, 0) + lc * rc
        return {m: c for m, c in out.items() if c}

    def visit(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return {zero_power: node.value}
        if isinstance(node, ast.Name) and node.id in variables:
            power = [0] * len(variables)
            power[variables.index(node.id)] = 1
            return {tuple(power): 1}
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            value = visit(node.operand)
            return None if value is None else {m: -c for m, c in value.items()}
        if isinstance(node, ast.BinOp):
            lhs, rhs = visit(node.left), visit(node.right)
            if lhs is None or rhs is None:
                return None
            if isinstance(node.op, ast.Add):
                return add(lhs, rhs)
            if isinstance(node.op, ast.Sub):
                return add(lhs, rhs, -1)
            if isinstance(node.op, ast.Mult):
                return multiply(lhs, rhs)
        return None

    try:
        return visit(ast.parse(expr, mode="eval").body)
    except (SyntaxError, ValueError):
        return None


def _flat_3d_tensor_submap_info(text: str, ssa: str):
    """Return (base, symbols, sizes, polynomial) for a rank-3 flat submap."""
    match = re.search(
        rf"^\s*{re.escape(ssa)}\s*=\s*polygeist\.submap\s*"
        rf"\(([^)]*)\)\s*\{{[^}}]*map\s*=\s*([^}}]+)\}}\s*:.*?"
        r"->\s*tensor<\?x\?x\?xf32>\s*$",
        text,
        re.MULTILINE,
    )
    if not match:
        return None
    operands = [piece.strip() for piece in match.group(1).split(",")]
    if len(operands) != 6:
        return None
    map_text = _resolve_affine_map_text(text, match.group(2).strip())
    if map_text is None:
        return None
    result = re.search(r"->\s*\((.*)\)\s*>", map_text)
    if not result:
        return None
    polynomial = _affine_polynomial(result.group(1))
    if polynomial is None:
        return None
    return operands[0], tuple(operands[1:3]), tuple(operands[3:]), polynomial


def _parboil_7pt_flat_grid_info(text: str, inputs: list[str], output: str):
    """Prove center/six-axis flattened-grid layout for the cuDNN adapter."""
    if len(inputs) != 7:
        return None
    parsed = [_flat_3d_tensor_submap_info(text, name) for name in inputs]
    out = _flat_3d_tensor_submap_info(text, output)
    if any(item is None for item in parsed) or out is None:
        return None
    center = parsed[6]
    if any(item[0] != center[0] or item[1] != center[1]
           for item in parsed):
        return None
    if out[1] != center[1] or out[3] != center[3]:
        return None

    def subtract(lhs, rhs):
        result = dict(lhs)
        for monomial, coefficient in rhs.items():
            result[monomial] = result.get(monomial, 0) - coefficient
            if result[monomial] == 0:
                del result[monomial]
        return result

    # Variable order: d0,d1,d2,s0,s1. A row-major flattened 3D grid has
    # axial strides 1, s1 and s0*s1.
    one = {(0, 0, 0, 0, 0): 1}
    s1 = {(0, 0, 0, 0, 1): 1}
    s0s1 = {(0, 0, 0, 1, 1): 1}
    expected = [one, {m: -c for m, c in one.items()},
                s1, {m: -c for m, c in s1.items()},
                s0s1, {m: -c for m, c in s0s1.items()}]
    deltas = [subtract(item[3], center[3]) for item in parsed[:6]]
    if {tuple(sorted(delta.items())) for delta in deltas} != {
            tuple(sorted(delta.items())) for delta in expected}:
        return None
    # Symbols are (ny, nx); sizes are (nx-2, ny-2, nz-2).
    return center[0], out[0], center[1], out[2]


def _constant_index_value(text: str, ssa: str) -> int | None:
    m = re.search(
        rf"^\s*{re.escape(ssa)}\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*index\s*$",
        text,
        re.MULTILINE,
    )
    return int(m.group(1)) if m else None


def _type_payload(mlir_type: str, prefix: str) -> str | None:
    if not mlir_type.startswith(prefix + "<") or not mlir_type.endswith(">"):
        return None
    return mlir_type[len(prefix) + 1:-1]


def _top_level_first_type_piece(payload: str) -> str:
    depth = 0
    cur: list[str] = []
    for c in payload:
        if c == "," and depth == 0:
            break
        if c in "<(":
            depth += 1
        elif c in ">)":
            depth -= 1
        cur.append(c)
    return "".join(cur).strip()


def _memref_to_tensor_type(memref_ty: str) -> str | None:
    payload = _type_payload(memref_ty.strip(), "memref")
    if payload is None:
        return None
    shaped = _top_level_first_type_piece(payload)
    if not shaped:
        return None
    return f"tensor<{shaped}>"


def _infer_tensor_type(text: str, ssa: str) -> str | None:
    """Best-effort SSA→tensor type inference for custom launch rendering."""
    # Function argument or explicit tensor operand.
    for fm in re.finditer(r"func\.func\s+@\w+\s*\(([^)]*)\)", text):
        params = fm.group(1)
        for pm in re.finditer(r"(%[\w_\-]+)\s*:\s*(tensor<[^,)]+>)", params):
            if pm.group(1) == ssa:
                return pm.group(2).strip()

    m = re.search(
        rf"^\s*{re.escape(ssa)}\s*=\s*tensor\.cast\s+.*?\s+to\s+(tensor<[^\n]+>)\s*$",
        text,
        re.MULTILINE,
    )
    if m:
        return m.group(1).strip()

    m = re.search(
        rf"^\s*{re.escape(ssa)}\s*=\s*tensor\.extract_slice\s+.*?\s+to\s+(tensor<[^\n]+>)\s*$",
        text,
        re.MULTILINE,
    )
    if m:
        return m.group(1).strip()

    m = re.search(
        rf"^\s*{re.escape(ssa)}\s*=\s*tensor\.insert_slice\s+.*?\s+into\s+(tensor<[^\n]+>)\s*$",
        text,
        re.MULTILINE,
    )
    if m:
        return m.group(1).strip()

    m = re.search(
        rf"^\s*{re.escape(ssa)}\s*=\s*polygeist\.submap\(.*?\)\s*"
        rf"\{{[^}}]*\}}\s*:\s*\([^)]*\)\s*->\s*(tensor<[^\n]+>)\s*$",
        text,
        re.MULTILINE,
    )
    if m:
        return m.group(1).strip()

    m = re.search(
        rf"^\s*{re.escape(ssa)}\s*=\s*bufferization\.to_tensor\s+%[\w_\-]+\s*:\s*(memref<[^\n]+>)\s*$",
        text,
        re.MULTILINE,
    )
    if m:
        return _memref_to_tensor_type(m.group(1).strip())

    return None


def _trace_tensor_storage_base(text: str, ssa: str) -> str:
    """Trace tensor view/update SSA values to the tensor that owns storage."""
    for _ in range(16):
        patterns = (
            # A slice is a view of its source tensor.
            rf"^\s*{re.escape(ssa)}\s*=\s*tensor\.extract_slice\s+(%[\w_\-]+)",
            # An insert_slice updates its destination tensor.
            rf"^\s*{re.escape(ssa)}\s*=\s*tensor\.insert_slice\s+%[\w_\-]+\s+into\s+(%[\w_\-]+)",
            # A cast changes only the static type information.
            rf"^\s*{re.escape(ssa)}\s*=\s*tensor\.cast\s+(%[\w_\-]+)",
        )
        next_ssa = None
        for pat in patterns:
            match = re.search(pat, text, re.MULTILINE)
            if match:
                next_ssa = match.group(1)
                break
        if not next_ssa or next_ssa == ssa:
            break
        ssa = next_ssa
    return ssa


def _parse_polygeist_submap_window(
    text: str, ssa: str
) -> tuple[str, list[str]] | None:
    m = re.search(
        rf"^\s*{re.escape(ssa)}\s*=\s*polygeist\.submap\s*"
        rf"\(\s*(%[\w_\-]+)\s*,\s*([^)]+)\)\s*\{{[^}}]*\}}\s*:",
        text,
        re.MULTILINE,
    )
    if not m:
        return None
    sizes = [p.strip() for p in m.group(2).split(",") if p.strip()]
    return m.group(1), sizes


def _parse_memref_view(text: str, ssa: str, before: int) -> dict | None:
    """Resolve a raised memref view to its physical base and logical sizes."""
    prefix = text[:before]
    subview = re.search(
        rf"^\s*{re.escape(ssa)}\s*=\s*memref\.subview\s+"
        rf"(%[\w_\-]+)\s*\[[^\]]*\]\s*\[([^\]]*)\]\s*"
        rf"\[[^\]]*\]\s*:\s*(memref<.+?>)\s+to\s+memref<",
        prefix,
        re.MULTILINE,
    )
    if subview:
        return {
            "kind": "subview",
            "base": subview.group(1),
            "base_type": subview.group(3).strip(),
            "sizes": [
                p.strip() for p in subview.group(2).split(",") if p.strip()
            ],
            "map": None,
        }
    submap = re.search(
        rf"^\s*{re.escape(ssa)}\s*=\s*polygeist\.submap\s*"
        rf"\(\s*(%[\w_\-]+)\s*,\s*([^)]+)\)\s*"
        rf"\{{\s*map\s*=\s*([^}}]+)\}}\s*:\s*"
        rf"\((memref<[^,)]+>)[^)]*\)\s*->\s*memref<",
        prefix,
        re.MULTILINE,
    )
    if submap:
        map_text = _resolve_affine_map_text(prefix, submap.group(3).strip())
        return {
            "kind": "submap",
            "base": submap.group(1),
            "base_type": submap.group(4).strip(),
            "sizes": [
                p.strip() for p in submap.group(2).split(",") if p.strip()
            ],
            "map": map_text,
        }
    reinterpret = re.search(
        rf"^\s*{re.escape(ssa)}\s*=\s*memref\.reinterpret_cast\s+"
        rf"(%[\w_\-]+).*?:\s*(memref<.+?>)\s+to\s+memref<",
        prefix,
        re.MULTILINE,
    )
    if reinterpret:
        return {
            "kind": "reinterpret_cast",
            "base": reinterpret.group(1),
            "base_type": reinterpret.group(2).strip(),
            "sizes": [],
            "map": None,
        }
    return None


def _zero_fill_span_for_base(
    text: str, before: int, expected_base: str
) -> tuple[int, int] | None:
    """Prove that the last raised fill before a contraction zeros its output."""
    prefix = text[:before]
    fills = list(re.finditer(
        r"^\s*linalg\.fill\s+ins\((%[\w_\-]+)\s*:\s*[^)]+\)\s*"
        r"outs\((%[\w_\-]+)\s*:\s*[^)]+\)\s*$",
        prefix,
        re.MULTILINE,
    ))
    if not fills:
        return None
    fill = fills[-1]
    if parse_constants(prefix).get(fill.group(1)) != 0.0:
        return None
    view = _parse_memref_view(text, fill.group(2), before)
    if view is None or view["base"] != expected_base:
        return None
    # A write between initialization and the contraction would make beta=0
    # replacement invalid. View construction and scalar constants are benign.
    middle = text[fill.end():before]
    if re.search(r"\b(?:affine|memref)\.store\b|\blinalg\.(?!fill\b)", middle):
        return None
    return fill.span()


def _plain_f32_memrefs(types: list[str], ranks: list[int]) -> bool:
    return (
        len(types) == len(ranks)
        and [_shaped_rank(t) for t in types] == ranks
        and all(_sniff_elem_type(t) == "f32" for t in types)
        # These fixed runtime shims construct contiguous descriptors from
        # dimensions and therefore cannot accept arbitrary strided bases.
        and all("," not in t for t in types)
    )


def _compact_affine_map(map_text: str | None) -> str:
    return re.sub(r"\s+", "", map_text or "")


def _to_tensor_memref_source(
    text: str, value: str | None, before: int
) -> tuple[str, str] | None:
    """Return the physical memref behind a direct to_tensor conversion."""
    if value is None:
        return None
    prefix = text[:before]
    function_start = prefix.rfind("func.func")
    if function_start >= 0:
        prefix = prefix[function_start:]
    match = re.search(
        rf"^\s*{re.escape(value)}\s*=\s*bufferization\.to_tensor\s+"
        rf"(%[\w_\-]+)(?:\s+[^:]*)?\s*:\s*(memref<[^\n]+>)$",
        prefix,
        re.MULTILINE,
    )
    return (match.group(1), match.group(2).strip()) if match else None


def _tensor_extract_slice_info(
    text: str, value: str, before: int
) -> dict | None:
    match = re.search(
        rf"^\s*{re.escape(value)}\s*=\s*tensor\.extract_slice\s+"
        rf"(%[\w_\-]+)\[([^]]*)\]\s*\[([^]]*)\]\s*\[([^]]*)\]",
        text[:before], re.MULTILINE)
    if match is None:
        return None
    split = lambda group: [part.strip() for part in group.split(",")]
    return {
        "source": match.group(1),
        "offsets": split(match.group(2)),
        "sizes": split(match.group(3)),
        "strides": split(match.group(4)),
    }


def _tensor_submap_info(text: str, value: str, before: int) -> dict | None:
    prefix = text[:before]
    # SSA names are function-local.  Searching the complete module prefix can
    # accidentally select an identically named submap from an earlier
    # function in multi-kernel test/audit modules.
    function_start = prefix.rfind("func.func")
    scoped_prefix = prefix[function_start:] if function_start >= 0 else prefix
    match = re.search(
        rf"^\s*{re.escape(value)}\s*=\s*polygeist\.submap\s*"
        rf"\(\s*(%[\w_\-]+)\s*,\s*([^)]+)\)\s*"
        rf"\{{\s*map\s*=\s*([^}}]+)\}}\s*:",
        scoped_prefix, re.MULTILINE)
    if match is None:
        return None
    return {
        "source": match.group(1),
        "sizes": [part.strip() for part in match.group(2).split(",")],
        "map": _resolve_affine_map_text(text, match.group(3).strip()),
        "map_ref": match.group(3).strip(),
    }


def _plain_shape_compatible(
    mlir_type: str, element_type: str, expected: list[int]
) -> bool:
    """Check static dimensions where present on a default-layout memref."""
    payload = _type_payload(mlir_type, "memref")
    if payload is None or "," in payload:
        return False
    shaped = _top_level_first_type_piece(payload)
    suffix = "x" + element_type
    if not shaped.endswith(suffix):
        return False
    dims = shaped[:-len(suffix)].split("x")
    return len(dims) == len(expected) and all(
        dim == "?" or (dim.isdigit() and int(dim) == want)
        for dim, want in zip(dims, expected)
    )


def _fixed_depthwise_padding_body_legal(
    text: str, body, height: int, width: int
) -> bool:
    """Prove the exact same-padding predicate and guarded accumulation."""
    lines = [line.strip() for line in body.body_lines]
    index_ssas: dict[int, str] = {}
    for line in lines:
        match = re.fullmatch(
            r"(%[\w_\-]+)\s*=\s*linalg\.index\s+([1-5])\s*:\s*index",
            line)
        if match:
            index_ssas[int(match.group(2))] = match.group(1)
    if set(index_ssas) not in ({1, 2, 3, 4}, {2, 3, 4, 5}):
        return False

    applications = []
    for line in lines:
        match = re.fullmatch(
            r"(%[\w_\-]+)\s*=\s*affine\.apply\s+(.+>)\(([^)]*)\)",
            line)
        if match:
            applications.append((
                match.group(1),
                _compact_affine_map(_resolve_affine_map_text(
                    text, match.group(2).strip())),
                [part.strip() for part in match.group(3).split(",")],
            ))
    expanded_batch = set(index_ssas) == {2, 3, 4, 5}
    if not expanded_batch:
        expected_maps = [
            f"affine_map<(d0,d1,d2,d3)->(-d0-d1+{width})>",
            "affine_map<(d0,d1,d2,d3)->(d0+d1-1)>",
            "affine_map<(d0,d1,d2,d3)->(d2+d3-1)>",
            f"affine_map<(d0,d1,d2,d3)->(-d2-d3+{height})>",
        ]
        args = [index_ssas[4], index_ssas[2],
                index_ssas[1], index_ssas[3]]
        expected_arg_sets = [args] * 4
    else:
        expected_maps = [
            "affine_map<(d0,d1)->(d0+d1-1)>",
            f"affine_map<(d0,d1)->(-d0-d1+{width})>",
            f"affine_map<(d0,d1)->(-d0-d1+{height})>",
            "affine_map<(d0,d1)->(d0+d1-1)>",
        ]
        expected_arg_sets = [
            [index_ssas[2], index_ssas[4]],
            [index_ssas[2], index_ssas[4]],
            [index_ssas[5], index_ssas[3]],
            [index_ssas[5], index_ssas[3]],
        ]
    if len(applications) != 4 or any(
            affine_map != expected_map or args != expected_args
            for (_, affine_map, args), expected_map, expected_args
            in zip(applications, expected_maps, expected_arg_sets)):
        return False

    comparisons: list[str] = []
    for applied_ssa, _, _ in applications:
        matches = [re.fullmatch(
            rf"(%[\w_\-]+)\s*=\s*arith\.cmpi\s+sge,\s*"
            rf"{re.escape(applied_ssa)},\s*(%[\w_\-]+)\s*:\s*index",
            line) for line in lines]
        matches = [match for match in matches if match]
        if len(matches) != 1 or _constant_index_value(
                text, matches[0].group(2)) != 0:
            return False
        comparisons.append(matches[0].group(1))
    if expanded_batch:
        comparisons = [comparisons[j] for j in (2, 3, 0, 1)]

    def binary_result(op: str, lhs: str, rhs: str, ty: str) -> str | None:
        pattern = re.compile(
            rf"(%[\w_\-]+)\s*=\s*{re.escape(op)}\s+"
            rf"{re.escape(lhs)},\s*{re.escape(rhs)}\s*:\s*{re.escape(ty)}")
        for line in lines:
            match = pattern.fullmatch(line)
            if match:
                return match.group(1)
        return None

    predicate = binary_result(
        "arith.andi", comparisons[0], comparisons[1], "i1")
    predicate = (binary_result("arith.andi", predicate, comparisons[2], "i1")
                 if predicate else None)
    predicate = (binary_result("arith.andi", predicate, comparisons[3], "i1")
                 if predicate else None)
    product = binary_result(
        "arith.mulf", body.ins_arg_names[0], body.ins_arg_names[1], "f32")
    accumulated = (binary_result(
        "arith.addf", body.outs_arg_names[0], product, "f32")
        if product else None)
    if not predicate or not accumulated:
        return False
    select = next((match.group(1) for line in lines if (match := re.fullmatch(
        rf"(%[\w_\-]+)\s*=\s*arith\.select\s+{re.escape(predicate)},\s*"
        rf"{re.escape(accumulated)},\s*{re.escape(body.outs_arg_names[0])}"
        rf"\s*:\s*f32", line))), None)
    return select is not None and body.yield_value == select


def _resolve_affine_map_text(text: str, map_ref: str) -> str | None:
    map_ref = map_ref.strip()
    if map_ref.startswith("affine_map<"):
        return map_ref
    if not map_ref.startswith("#"):
        return None
    match = re.search(
        rf"(?m)^\s*{re.escape(map_ref)}\s*=\s*"
        r"(affine_map<.*>)\s*$",
        text,
    )
    return match.group(1) if match else None


def _split_affine_results(results: str) -> list[str]:
    pieces: list[str] = []
    depth = 0
    start = 0
    for i, char in enumerate(results):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            pieces.append(results[start:i].strip())
            start = i + 1
    pieces.append(results[start:].strip())
    return pieces


def _linear_dim_coefficients(expr: str) -> tuple[dict[int, int], int] | None:
    """Parse the small affine-linear subset used by raised window submaps."""
    compact = re.sub(r"\s+", "", expr)
    # Normalize subtraction into signed additive terms. Parenthesized/floordiv
    # expressions are intentionally rejected; they are not fixed windows.
    if any(ch in compact for ch in "()[]"):
        return None
    normalized = compact.replace("-", "+-")
    coeffs: dict[int, int] = {}
    constant = 0
    for term in (part for part in normalized.split("+") if part):
        match = re.fullmatch(r"(-?)(?:d(\d+)(?:\*(\d+))?|(\d+)\*d(\d+))", term)
        if match:
            sign = -1 if match.group(1) == "-" else 1
            if match.group(2) is not None:
                dim = int(match.group(2))
                coefficient = int(match.group(3) or "1")
            else:
                dim = int(match.group(5))
                coefficient = int(match.group(4))
            coeffs[dim] = coeffs.get(dim, 0) + sign * coefficient
            continue
        if re.fullmatch(r"-?\d+", term):
            constant += int(term)
            continue
        return None
    return coeffs, constant


def _regular_window_conv2d_info(
    text: str, window_ssa: str
) -> tuple[str, str, int, int, int, int, int, int, int, int] | None:
    """Prove an NCHW [N,C,OH,OW,KH,KW] fixed sliding-window submap.

    Returns (base SSA, base tensor type, KH, KW, SH, SW, DH, DW, PH, PW).
    This first legality proof accepts valid-window accesses only. A negative
    offset by itself does not prove that out-of-bounds elements have zero-pad
    semantics, so padded/guarded windows remain unmatched until that boundary
    predicate is represented and proved explicitly.
    """
    definition = re.search(
        rf"(?s)^\s*{re.escape(window_ssa)}\s*=\s*polygeist\.submap\s*"
        rf"\(\s*(%[\w_\-]+)\s*,\s*([^)]+)\)\s*"
        r"\{[^}]*map\s*=\s*([^}]+)\}\s*:",
        text,
        re.MULTILINE,
    )
    if not definition:
        return None
    base, size_text, map_ref = definition.groups()
    sizes = [part.strip() for part in size_text.split(",") if part.strip()]
    if len(sizes) != 6:
        return None
    # Batch, channel, and output extents may remain dynamic.  Only the two
    # reduction extents become cuDNN descriptor parameters and therefore need
    # to be compile-time constants in the current kernel.launch ABI.
    kh_value = _constant_index_value(text, sizes[4])
    kw_value = _constant_index_value(text, sizes[5])
    if kh_value is None or kw_value is None or kh_value <= 0 or kw_value <= 0:
        return None
    kh, kw = int(kh_value), int(kw_value)

    map_text = _resolve_affine_map_text(text, map_ref)
    if map_text is None:
        return None
    parsed_map = re.fullmatch(
        r"affine_map<\(([^)]*)\)\s*->\s*\(([^)]*)\)>",
        map_text.strip(),
    )
    if not parsed_map:
        return None
    dims = [part.strip() for part in parsed_map.group(1).split(",")]
    results = _split_affine_results(parsed_map.group(2))
    if dims != [f"d{i}" for i in range(6)] or len(results) != 4:
        return None
    if re.sub(r"\s+", "", results[0]) != "d0" or \
       re.sub(r"\s+", "", results[1]) != "d1":
        return None
    h = _linear_dim_coefficients(results[2])
    w = _linear_dim_coefficients(results[3])
    if h is None or w is None:
        return None
    h_coeffs, h_constant = h
    w_coeffs, w_constant = w
    if set(h_coeffs) != {2, 4} or set(w_coeffs) != {3, 5}:
        return None
    sh, dh = h_coeffs[2], h_coeffs[4]
    sw, dw = w_coeffs[3], w_coeffs[5]
    if min(sh, sw, dh, dw) <= 0 or h_constant != 0 or w_constant != 0:
        return None
    base_type = _infer_tensor_type(text, base)
    if base_type is None or _shaped_rank(base_type) != 4:
        return None
    return (base, base_type, kh, kw, sh, sw, dh, dw, 0, 0)


def _is_forward_conv3d_window(text: str, operand: str) -> bool:
    """Prove a rank-8 view is the valid-forward Conv3D input window."""
    definition = re.search(
        rf"(?s)^\s*{re.escape(operand)}\s*=\s*"
        r"polygeist\.submap\(.*?\)\s*\{map\s*=\s*([^}]+)\}",
        text,
        re.MULTILINE,
    )
    if not definition:
        return False
    map_text = definition.group(1).strip()
    if map_text.startswith("#"):
        resolved = re.search(
            rf"(?m)^\s*{re.escape(map_text)}\s*=\s*(affine_map<.*?>)\s*$",
            text,
        )
        if not resolved:
            return False
        map_text = resolved.group(1)
    compact = re.sub(r"\s+", "", map_text)
    return compact in {
        "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->"
        "(d4,d5+d1,d6+d2,d7+d3)>",
        "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->"
        "(0,d4,d5+d1,d6+d2,d7+d3)>",
    }


def _miniamr_weighted27_window_info(
    text: str,
    weight_names: list[str],
    weight_types: list[str],
) -> tuple[str, str, str, int] | None:
    """Return (input_base, input_type, weight_ssa, K) for 3D 27pt conv."""
    def _rank(ty: str) -> int:
        if not ty.startswith("tensor<"):
            return -1
        inside = ty[ty.find("<") + 1:ty.rfind(">")]
        shape = inside.rsplit("x", 1)[0]
        return shape.count("x") + 1 if shape else 0

    weight_ssa = None
    window_ssa = None
    for name, ty in zip(weight_names, weight_types):
        rank = _rank(ty)
        if rank == 3 and weight_ssa is None:
            weight_ssa = name
        elif rank == 6 and window_ssa is None:
            window_ssa = name
    if weight_ssa is None or window_ssa is None:
        return None

    window = _parse_polygeist_submap_window(text, window_ssa)
    if window is None:
        return None
    input_base, sizes = window
    if len(sizes) < 6:
        return None
    k_values = [_constant_index_value(text, s) for s in sizes[-3:]]
    if any(v is None for v in k_values) or len(set(k_values)) != 1:
        return None
    K = k_values[0]
    if K is None or K < 3 or K % 2 == 0:
        return None
    input_type = _infer_tensor_type(text, input_base)
    if input_type is None:
        return None
    return input_base, input_type, weight_ssa, K


def _conv2d_ntap_grid_info(
    text: str, input_names: list[str], out_name: str
) -> tuple[int, str, list[int]] | None:
    """Validate same-base odd-square input subviews and return row-major order.

    Returns (filter_width, top_left_input_ssa, input_indices_in_row_major_order).
    The scalar algebra matcher only proves a weighted sum. This check proves
    the operands are actually shifted subviews that cuDNN can interpret as a
    dense KxK cross-correlation window.
    """
    ntaps = len(input_names)
    width = math.isqrt(ntaps)
    if width * width != ntaps or width < 3 or width % 2 == 0:
        return None
    parsed: list[tuple[int, str, tuple[int, int]]] = []
    bases = set()
    for idx, name in enumerate(input_names):
        p = _parse_static_subview_offset(text, name)
        if p is None:
            return None
        base, off = p
        bases.add(base)
        parsed.append((idx, name, off))
    if len(bases) != 1:
        return None
    ys = sorted({off[0] for _, _, off in parsed})
    xs = sorted({off[1] for _, _, off in parsed})
    if len(ys) != width or len(xs) != width:
        return None
    if ys != list(range(ys[0], ys[0] + width)):
        return None
    if xs != list(range(xs[0], xs[0] + width)):
        return None

    out = _parse_static_subview_offset(text, out_name)
    if out is None:
        return None
    _out_base, out_off = out
    radius = width // 2
    if out_off != (ys[0] + radius, xs[0] + radius):
        return None

    by_offset = {off: (idx, name) for idx, name, off in parsed}
    ordered_indices: list[int] = []
    top_left_name = ""
    for y in ys:
        for x in xs:
            item = by_offset.get((y, x))
            if item is None:
                return None
            idx, name = item
            if y == ys[0] and x == xs[0]:
                top_left_name = name
            ordered_indices.append(idx)
    return width, top_left_name, ordered_indices


def _conv2d_ntap_tensor_grid_info(
    text: str, input_names: list[str], out_name: str
) -> tuple[int, str, list[int]] | None:
    """Tensor extract_slice sibling of _conv2d_ntap_grid_info."""
    ntaps = len(input_names)
    width = math.isqrt(ntaps)
    if width * width != ntaps or width < 3 or width % 2 == 0:
        return None
    parsed: list[tuple[int, str, tuple[int, int]]] = []
    bases = set()
    for idx, name in enumerate(input_names):
        p = (_parse_static_extract_slice_offset(text, name) or
             _parse_static_tensor_submap_offset(text, name))
        if p is None:
            return None
        base, off = p
        bases.add(base)
        parsed.append((idx, name, off))
    if len(bases) != 1:
        return None
    ys = sorted({off[0] for _, _, off in parsed})
    xs = sorted({off[1] for _, _, off in parsed})
    if len(ys) != width or len(xs) != width:
        return None
    if ys != list(range(ys[0], ys[0] + width)):
        return None
    if xs != list(range(xs[0], xs[0] + width)):
        return None

    out = (_parse_static_extract_slice_offset(text, out_name) or
           _parse_static_tensor_submap_offset(text, out_name))
    if out is None:
        return None
    _out_base, out_off = out
    radius = width // 2
    if out_off != (ys[0] + radius, xs[0] + radius):
        return None

    by_offset = {off: (idx, name) for idx, name, off in parsed}
    ordered_indices: list[int] = []
    top_left_name = ""
    for y in ys:
        for x in xs:
            item = by_offset.get((y, x))
            if item is None:
                return None
            idx, name = item
            if y == ys[0] and x == xs[0]:
                top_left_name = name
            ordered_indices.append(idx)
    return width, top_left_name, ordered_indices


def _weight_cast_op(src_ty: str, dst_ty: str) -> str:
    casts = {
        ("f64", "f32"): "arith.truncf",
        ("f32", "f64"): "arith.extf",
    }
    return casts.get((src_ty, dst_ty), "arith.bitcast")


def _format_weight_literal(value: float, ty: str) -> str:
    if ty.startswith("f"):
        lit = repr(value)
        return lit if any(c in lit for c in ".eE") else lit + ".0"
    return str(int(value))


def _render_window_conv2d_launch(
    result_ssa: str,
    result_type: str,
    input_ssa: str,
    input_type: str,
    output_ssa: str,
    output_type: str,
    weight_ssa: str | None,
    weight_value: float | None,
    params: tuple[int, int, int, int, int, int, int, int],
    indent: str,
    unique_id: int,
) -> str:
    """Render a uniform-weight depthwise cuDNN convolution launch."""
    casts, tensors, tensor_types = _normalize_tensor_operands(
        [input_ssa, output_ssa], [input_type, output_type], indent
    )
    kh, kw, sh, sw, dh, dw, ph, pw = params
    prefix = f"%winconv{unique_id}"
    lines = list(casts)
    if weight_ssa is None:
        weight_ssa = f"{prefix}_weight"
        literal = _format_weight_literal(
            1.0 if weight_value is None else weight_value, "f32"
        )
        lines.append(
            f"{indent}{weight_ssa} = arith.constant {literal} : f32"
        )
    values = (kh, kw, sh, sw, dh, dw, ph, pw)
    names: list[str] = []
    for label, value in zip(("kh", "kw", "sh", "sw", "dh", "dw", "ph", "pw"),
                            values):
        name = f"{prefix}_{label}"
        names.append(name)
        lines.append(f"{indent}{name} = arith.constant {value} : i32")

    dynamic_result = _dynamic_tensor_type(result_type) or result_type
    launch_result = result_ssa
    result_cast = ""
    if dynamic_result != result_type:
        launch_result = _derived_ssa_name(result_ssa, "tdyn")
        result_cast = (
            f"\n{indent}{result_ssa} = tensor.cast {launch_result} : "
            f"{dynamic_result} to {result_type}"
        )
    # A uniform window whose weight is exactly 1/(KH*KW) over a non-overlapping,
    # valid (unpadded, undilated) window IS average pooling. Emit the cuDNN
    # pooling symbol so ABI lowering routes it to cudnnPoolingForward (~15x
    # faster than the grouped/depthwise convolution the box-filter form uses).
    # Any other uniform weight stays a genuine box-filter convolution.
    avg_weight = 1.0 / (kh * kw) if kh > 0 and kw > 0 else None
    is_avg_pool = (
        weight_value is not None and avg_weight is not None
        and abs(weight_value - avg_weight) <= 1e-6 * avg_weight
        and sh == kh and sw == kw   # non-overlapping: stride == kernel
        and dh == 1 and dw == 1     # no dilation
        and ph == 0 and pw == 0     # valid window (no padding)
    )
    launch_symbol = (
        "cudnnAvgPoolWindow_f32" if is_avg_pool else "cudnnConvolution2DWindow_f32"
    )
    operands = tensors + [weight_ssa] + names
    types = tensor_types + ["f32"] + ["i32"] * 8
    lines.append(
        f"{indent}{launch_result} = kernel.launch "
        f"@{launch_symbol}({', '.join(operands)}) : "
        f"({', '.join(types)}) -> {dynamic_result}{result_cast}"
    )
    return "\n".join(lines)


def _render_ntap_conv_launch(
    name: str,
    top_left_ssa: str,
    top_left_type: str,
    out_ssa: str,
    out_type: str,
    width: int,
    ordered_inline_weights: list[list[str] | None],
    indent: str,
    scalar_type_map: dict[str, str],
    body_constants: dict[str, float],
    weight_ty: str,
    unique_id: int,
) -> str:
    cast_lines, memrefs, memref_types = _normalize_memref_operands(
        [top_left_ssa, out_ssa], [top_left_type, out_type], indent
    )
    ntaps = width * width
    weight_memref_ty = f"memref<{ntaps}x{weight_ty}>"
    prefix = f"%ntap{unique_id}"
    wbuf = f"{prefix}_weights"
    k_ssa = f"{prefix}_k"
    lines = list(cast_lines)
    lines.append(f"{indent}{wbuf} = memref.alloca() : {weight_memref_ty}")
    for idx, weights in enumerate(ordered_inline_weights):
        idx_ssa = f"{prefix}_i{idx}"
        lines.append(f"{indent}{idx_ssa} = arith.constant {idx} : index")
        if weights is None:
            val_ssa = f"{prefix}_w{idx}"
            lines.append(
                f"{indent}{val_ssa} = arith.constant "
                f"{_format_weight_literal(1.0, weight_ty)} : {weight_ty}"
            )
        elif len(weights) == 1:
            val_ssa = weights[0]
            src_ty = scalar_type_map.get(val_ssa)
            if src_ty and src_ty != weight_ty:
                cast_ssa = f"{prefix}_w{idx}_cast"
                lines.append(
                    f"{indent}{cast_ssa} = {_weight_cast_op(src_ty, weight_ty)} "
                    f"{val_ssa} : {src_ty} to {weight_ty}"
                )
                val_ssa = cast_ssa
        else:
            summed = sum(body_constants.get(w, 0.0) for w in weights)
            val_ssa = f"{prefix}_w{idx}"
            lines.append(
                f"{indent}{val_ssa} = arith.constant "
                f"{_format_weight_literal(summed, weight_ty)} : {weight_ty}"
            )
        lines.append(
            f"{indent}memref.store {val_ssa}, {wbuf}[{idx_ssa}] : {weight_memref_ty}"
        )
    weight_dyn_ty = f"memref<?x{weight_ty}>"
    wbuf_dyn = f"{wbuf}_c"
    lines.append(
        f"{indent}{wbuf_dyn} = memref.cast {wbuf} : {weight_memref_ty} to {weight_dyn_ty}"
    )
    lines.append(f"{indent}{k_ssa} = arith.constant {width} : i32")
    operands = [memrefs[0], memrefs[1], wbuf_dyn, k_ssa]
    sig_types = [memref_types[0], memref_types[1], weight_dyn_ty, "i32"]
    lines.append(
        f"{indent}kernel.launch @{name}({', '.join(operands)}) : "
        f"({', '.join(sig_types)}) -> ()"
    )
    return "\n".join(lines)


def _render_ntap_conv_tensor_launch(
    name: str,
    result_ssa: str,
    result_type: str,
    top_left_ssa: str,
    top_left_type: str,
    out_ssa: str,
    out_type: str,
    width: int,
    ordered_inline_weights: list[list[str] | None],
    indent: str,
    scalar_type_map: dict[str, str],
    body_constants: dict[str, float],
    weight_ty: str,
    unique_id: int,
) -> str:
    cast_lines, tensors, tensor_types = _normalize_tensor_operands(
        [top_left_ssa, out_ssa], [top_left_type, out_type], indent
    )
    ntaps = width * width
    prefix = f"%ntap{unique_id}"
    value_ssas: list[str] = []
    lines = list(cast_lines)
    for idx, weights in enumerate(ordered_inline_weights):
        if weights is None:
            val_ssa = f"{prefix}_w{idx}"
            lines.append(
                f"{indent}{val_ssa} = arith.constant "
                f"{_format_weight_literal(1.0, weight_ty)} : {weight_ty}"
            )
        elif len(weights) == 1:
            val_ssa = weights[0]
            src_ty = scalar_type_map.get(val_ssa)
            if src_ty and src_ty != weight_ty:
                cast_ssa = f"{prefix}_w{idx}_cast"
                lines.append(
                    f"{indent}{cast_ssa} = {_weight_cast_op(src_ty, weight_ty)} "
                    f"{val_ssa} : {src_ty} to {weight_ty}"
                )
                val_ssa = cast_ssa
        else:
            summed = sum(body_constants.get(w, 0.0) for w in weights)
            val_ssa = f"{prefix}_w{idx}"
            lines.append(
                f"{indent}{val_ssa} = arith.constant "
                f"{_format_weight_literal(summed, weight_ty)} : {weight_ty}"
            )
        value_ssas.append(val_ssa)

    weight_static_ty = f"tensor<{ntaps}x{weight_ty}>"
    weight_dyn_ty = f"tensor<?x{weight_ty}>"
    wvec = f"{prefix}_weights"
    wvec_dyn = f"{wvec}_c"
    k_ssa = f"{prefix}_k"
    lines.append(
        f"{indent}{wvec} = tensor.from_elements {', '.join(value_ssas)} : "
        f"{weight_static_ty}"
    )
    lines.append(
        f"{indent}{wvec_dyn} = tensor.cast {wvec} : {weight_static_ty} to "
        f"{weight_dyn_ty}"
    )
    lines.append(f"{indent}{k_ssa} = arith.constant {width} : i32")
    dyn_result_type = _dynamic_tensor_type(result_type) or result_type
    launch_result_ssa = result_ssa
    result_cast = ""
    if dyn_result_type != result_type:
        launch_result_ssa = _derived_ssa_name(result_ssa, "tdyn")
        result_cast = (
            f"\n{indent}{result_ssa} = tensor.cast {launch_result_ssa} : "
            f"{dyn_result_type} to {result_type}"
        )
    operands = [tensors[0], tensors[1], wvec_dyn, k_ssa]
    sig_types = [tensor_types[0], tensor_types[1], weight_dyn_ty, "i32"]
    lines.append(
        f"{indent}{launch_result_ssa} = kernel.launch @{name}"
        f"({', '.join(operands)}) : ({', '.join(sig_types)}) -> "
        f"{dyn_result_type}{result_cast}"
    )
    return "\n".join(lines)


def _render_ntap_conv3d_tensor_launch(
    name: str,
    result_ssa: str,
    result_type: str,
    input_ssa: str,
    input_type: str,
    out_ssa: str,
    out_type: str,
    weight_ssa: str,
    weight_type: str,
    width: int,
    indent: str,
) -> str:
    cast_lines, tensors, tensor_types = _normalize_tensor_operands(
        [input_ssa, out_ssa, weight_ssa],
        [input_type, out_type, weight_type],
        indent,
    )
    prefix = _derived_ssa_name(result_ssa, "conv3d")
    k_ssa = f"{prefix}_k"
    lines = list(cast_lines)
    lines.append(f"{indent}{k_ssa} = arith.constant {width} : i32")
    dyn_result_type = _dynamic_tensor_type(result_type) or result_type
    launch_result_ssa = result_ssa
    result_cast = ""
    if dyn_result_type != result_type:
        launch_result_ssa = _derived_ssa_name(result_ssa, "tdyn")
        result_cast = (
            f"\n{indent}{result_ssa} = tensor.cast {launch_result_ssa} : "
            f"{dyn_result_type} to {result_type}"
        )
    operands = [tensors[0], tensors[1], tensors[2], k_ssa]
    sig_types = [tensor_types[0], tensor_types[1], tensor_types[2], "i32"]
    lines.append(
        f"{indent}{launch_result_ssa} = kernel.launch @{name}"
        f"({', '.join(operands)}) : ({', '.join(sig_types)}) -> "
        f"{dyn_result_type}{result_cast}"
    )
    return "\n".join(lines)


def _render_flat_7pt_conv3d_launch(
    result_ssa: str,
    result_type: str,
    input_ssa: str,
    input_type: str,
    output_ssa: str,
    output_type: str,
    center_scale: str,
    neighbor_scale: str,
    symbols: tuple[str, str],
    sizes: tuple[str, str, str],
    indent: str,
) -> str:
    cast_lines, tensors, tensor_types = _normalize_tensor_operands(
        [input_ssa, output_ssa], [input_type, output_type], indent
    )
    operands = tensors + [center_scale, neighbor_scale, *symbols, *sizes]
    types = tensor_types + ["f32", "f32"] + ["index"] * 5
    return "\n".join([
        *cast_lines,
        f"{indent}{result_ssa} = kernel.launch "
        f"@cudnnStencil3D7pt_f32_flat_tensor({', '.join(operands)}) : "
        f"({', '.join(types)}) -> {result_type}",
    ])


def _find_flat_submap_inverse(text: str, start: int, generic_result: str):
    match = re.search(
        rf"(?m)^\s*(%[\w_\-]+)\s*=\s*polygeist\.submapInverse\s*"
        rf"\(\s*(%[\w_\-]+)\s*,\s*{re.escape(generic_result)}\s*,"
        rf"([^)]*)\).*?->\s*(tensor<\?xf32>)\s*$",
        text[start:],
    )
    if not match:
        return None
    tail_operands = tuple(piece.strip() for piece in match.group(3).split(","))
    if len(tail_operands) != 5:
        return None
    return (match.group(1), match.group(4), match.group(2),
            tail_operands, (start + match.start(), start + match.end()))


def _find_tensor_submap_inverse(
    text: str, start: int, generic_result: str
) -> tuple[str, str, tuple[int, int]] | None:
    """Find the tensor writeback immediately consuming a generic result.

    Raised reductions over expanded window/segment views return a higher-rank
    tensor and then fold it back into the real destination with
    submapInverse.  Library calls for those compositions return the real
    destination shape directly, so the rewrite must replace that writeback's
    SSA result and type rather than the generic's expanded result.
    """
    match = re.search(
        rf"(?m)^\s*(%[\w_\-]+)\s*=\s*polygeist\.submapInverse\s*"
        rf"\([^,\n]+,\s*{re.escape(generic_result)}\s*,[^\n]*"
        rf"->\s*(tensor<[^>\n]+>)\s*$",
        text[start:],
    )
    if not match:
        return None
    return (match.group(1), match.group(2),
            (start + match.start(), start + match.end()))


def _render_cufft_1d_tensor_launch(
    name: str,
    result_ssa: str,
    result_type: str,
    input_ssa: str,
    input_type: str,
    out_ssa: str,
    out_type: str,
    inverse: int,
    indent: str,
) -> str | None:
    normalized = _normalize_complex1d_tensor_operands(
        [input_ssa, out_ssa], [input_type, out_type], indent
    )
    if normalized is None:
        return None
    cast_lines, tensors, tensor_types = normalized
    result_dyn_type = _complex1d_tensor_type(result_type)
    if result_dyn_type is None:
        return None
    prefix = _derived_ssa_name(result_ssa, "fft")
    inv_ssa = f"{prefix}_inverse"
    launch_result_ssa = result_ssa
    result_cast = ""
    lines = list(cast_lines)
    lines.append(f"{indent}{inv_ssa} = arith.constant {inverse} : i32")
    if result_dyn_type != result_type:
        launch_result_ssa = _derived_ssa_name(result_ssa, "tdyn")
        result_cast = (
            f"\n{indent}{result_ssa} = tensor.cast {launch_result_ssa} : "
            f"{result_dyn_type} to {result_type}"
        )
    operands = [tensors[0], tensors[1], inv_ssa]
    sig_types = [tensor_types[0], tensor_types[1], "i32"]
    lines.append(
        f"{indent}{launch_result_ssa} = kernel.launch @{name}"
        f"({', '.join(operands)}) : ({', '.join(sig_types)}) -> "
        f"{result_dyn_type}{result_cast}"
    )
    return "\n".join(lines)


def _find_insert_slice_of_result(
    text: str,
    search_from: int,
    source_ssa: str,
) -> tuple[str, str, tuple[int, int]] | None:
    pat = re.compile(
        rf"(\n[ \t]*)(%[\w_\-]+)\s*=\s*tensor\.insert_slice\s+"
        rf"{re.escape(source_ssa)}\s+into\s+[^\n]*\s+:\s+"
        rf"tensor<[^>]+>\s+into\s+(tensor<[^\n]+>)",
        re.MULTILINE,
    )
    m = pat.search(text, search_from)
    if not m:
        return None
    return m.group(2), m.group(3).strip(), (m.start(), m.end())


def _dft1d_inverse_flag(body_constants: dict[str, float]) -> int | None:
    two_pi = 6.283185307179586
    candidates = [
        v for v in body_constants.values()
        if abs(abs(v) - two_pi) < 1.0e-9
    ]
    if not candidates:
        return None
    return 1 if candidates[0] > 0.0 else 0


def _render_whisper_exp_shift_sum_launch(
    name: str,
    result_ssa: str,
    result_count: int,
    result_type: str,
    operands: list[str],
    operand_types: list[str],
    indent: str,
) -> str:
    cast_lines, operands, operand_types = _normalize_tensor_operands(
        operands, operand_types, indent
    )
    operand_str = ", ".join(operands)
    sig = f"({', '.join(operand_types)})"
    cast_prefix = "\n".join(cast_lines) + ("\n" if cast_lines else "")
    return (
        f"{cast_prefix}{indent}{result_ssa}:{result_count} = "
        f"kernel.launch @{name}({operand_str}) : {sig} -> {result_type}"
    )


def render_launch(name: str, result_ssa: str | None, result_type: str | None,
                  operands: list[str], indent: str,
                  bindings: dict, captures_per_step: list[list[str]],
                  operand_types: list[str] | None = None,
                  scalar_type_map: dict[str, str] | None = None,
                  inline_weights: list[list[str] | None] | None = None,
                  inline_weight_type: str = "f64",
                  body_constants: dict[str, float] | None = None,
                  result_count: int = 1,
                  launch_attrs: str = "") -> str:
    """Build a `kernel.launch` op line in MLIR text.

    When `result_ssa` and `result_type` are None, emit a void-returning
    launch (`-> ()`) — used for memref-form linalg.generic where the
    output is mutated in place rather than returned as an SSA.

    operand_types : explicit types for the tensor `operands` list (same order).
    scalar_type_map : SSA→type lookup for Cap-bound scalars.
    """
    # First: normalize strided memref operand types via memref.cast so they
    # match the canonical kernel.defn signature (which uses dynamic-stride
    # placeholders like `strided<[?, 1], offset: ?>` to accept any concrete
    # subview shape).
    cast_lines, operands, operand_types = _normalize_memref_operands(
        operands, operand_types, indent
    )
    tensor_cast_lines, operands, operand_types = _normalize_tensor_operands(
        operands, operand_types, indent
    )
    cast_lines.extend(tensor_cast_lines)

    # Surface body-internal constants (e.g. the 9 weights of a conv2d) as
    # additional scalar launch operands, when the template opts in via
    # `surface_inline_weights=True`. The encoder already builds the
    # in_arg → constant_ssa map per body (parse_generics' inline_weights_per_in).
    # We append them positionally — same order as the input subviews — so
    # the lowering pass can pair them with the inputs.
    #
    # When the surfaced constant's type doesn't match `inline_weight_type`
    # (e.g. cgeist promoted i16 inputs to i32 for the multiply, leaving the
    # weight constants typed i32 even though the kernel is i16), inject a
    # cast op so the launch signature is internally consistent. Without
    # this, the verifier would reject the kernel.launch.
    cast_ops_for_weights = {
        # (src_type, dst_type) → mlir op name
        ("i32", "i16"): "arith.trunci",
        ("i32", "i8"):  "arith.trunci",
        ("i16", "i8"):  "arith.trunci",
        ("i16", "i32"): "arith.extsi",
        ("i8",  "i32"): "arith.extsi",
        ("i8",  "i16"): "arith.extsi",
        ("f32", "f16"): "arith.truncf",
        ("f32", "bf16"): "arith.truncf",
        ("f64", "f32"): "arith.truncf",
        ("f64", "f16"): "arith.truncf",
        ("f64", "bf16"): "arith.truncf",
        ("f16", "f32"): "arith.extf",
        ("bf16", "f32"): "arith.extf",
        ("f32", "f64"): "arith.extf",
        ("f16", "f64"): "arith.extf",
        ("bf16", "f64"): "arith.extf",
    }
    inline_weight_ssas: list[str] = []
    weight_cast_lines: list[str] = []
    # Counter for generated SSAs (summed-constant materialisation) — kept
    # unique per launch by appending an index. Mostly for the conv3d-style
    # case where the same input is multiplied by several literal constants
    # and summed; we precompute the sum at rewrite time and emit one
    # arith.constant op carrying the result.
    synth_idx = 0
    if inline_weights:
        for w in inline_weights:
            if w is None:
                # The matcher may accept an elided `* 1.0` coefficient: some
                # frontend/canonicalization paths rewrite `1.0 * in[k]` to
                # bare `in[k]`. The runtime ABI still expects one scalar per
                # tap, so materialize the implicit unit coefficient here.
                synth_ssa = f"%cst_synth_{synth_idx}"
                synth_idx += 1
                lit = "1.0" if inline_weight_type.startswith("f") else "1"
                weight_cast_lines.append(
                    f"{indent}{synth_ssa} = arith.constant {lit} : {inline_weight_type}"
                )
                inline_weight_ssas.append(synth_ssa)
                continue
            # w is now always a list[str] (possibly length 1). Empty was
            # already normalised to None by parse_generics, so len(w) >= 1.
            if len(w) == 1:
                source_ssa = w[0]
                src_ty = scalar_type_map.get(source_ssa) if scalar_type_map else None
                if src_ty and src_ty != inline_weight_type:
                    op = cast_ops_for_weights.get((src_ty, inline_weight_type))
                    if op is None:
                        op = "arith.bitcast"
                    cast_ssa = source_ssa + "_to_" + inline_weight_type
                    weight_cast_lines.append(
                        f"{indent}{cast_ssa} = {op} {source_ssa} : {src_ty} to {inline_weight_type}"
                    )
                    inline_weight_ssas.append(cast_ssa)
                else:
                    inline_weight_ssas.append(source_ssa)
            else:
                # Multi-coefficient: sum the literal values from body_constants,
                # then emit a fresh arith.constant carrying the summed value.
                # This handles a body where the same input appears in multiple
                # muls with different literal constants. Semantic acceptance
                # is decided by Egglog; this block only materializes the
                # already-proved coefficient for the runtime ABI.
                summed = 0.0
                if body_constants is not None:
                    for ssa in w:
                        summed += body_constants.get(ssa, 0.0)
                synth_ssa = f"%cst_synth_{synth_idx}"
                synth_idx += 1
                # Format the constant literal in MLIR's normal form. f64 / f32
                # take a decimal float; integer types take a base-10 int.
                if inline_weight_type.startswith("f"):
                    lit = repr(summed)
                    if not (("." in lit) or ("e" in lit) or ("E" in lit)):
                        lit = lit + ".0"
                else:
                    lit = str(int(summed))
                weight_cast_lines.append(
                    f"{indent}{synth_ssa} = arith.constant {lit} : {inline_weight_type}"
                )
                inline_weight_ssas.append(synth_ssa)
    cast_lines.extend(weight_cast_lines)

    # Cap-bound scalars from bindings. When surface_inline_weights is in
    # effect, the template's weight Caps are already covered by the inline
    # surfacing — emitting them again would produce duplicate operands and
    # break the lowering. Suppress them in that case.
    scalar_ssas: list[str] = []
    if not inline_weight_ssas:
        for tmpl_name, bound in bindings.items():
            if isinstance(bound, tuple) and len(bound) == 2 and bound[0] == "Cap":
                # Mask Caps (template names like "%mask", "%mask1", ...) bind
                # to internal cmpi result SSAs that aren't real scalar arguments
                # — they're an artifact of the encoder treating arith.cmpi as
                # opaque. Skip them; the canonical kernel.defn body
                # reconstructs the mask from its own linalg.index + cmpi.
                if tmpl_name.startswith("%mask"):
                    continue
                scalar_ssas.append(bound[1])
    all_operands = operands + scalar_ssas + inline_weight_ssas
    operand_str = ", ".join(all_operands)

    # Build the function-type signature for the launch.
    sig_types: list[str] = []
    if operand_types is None or len(operand_types) != len(operands):
        sig_types.extend("!any" for _ in operands)
    else:
        sig_types.extend(operand_types)
    for s in scalar_ssas:
        if scalar_type_map and s in scalar_type_map:
            sig_types.append(scalar_type_map[s])
        else:
            sig_types.append("!any")
    # Inline-weight types: all the same element type (per-template config).
    for _ in inline_weight_ssas:
        sig_types.append(inline_weight_type)

    sig = f"({', '.join(sig_types)})"
    cast_prefix = "\n".join(cast_lines) + ("\n" if cast_lines else "")
    if result_ssa is None or result_type is None:
        # Memref-form / void launch.
        return (f"{cast_prefix}{indent}kernel.launch @{name}({operand_str})"
                f"{launch_attrs} : {sig} -> ()")
    launch_result_ssa = result_ssa
    launch_result_type = result_type
    result_bind = result_ssa if result_count <= 1 else f"{result_ssa}:{result_count}"
    result_cast = ""
    dyn_result_type = _dynamic_tensor_type(result_type)
    if dyn_result_type is not None and dyn_result_type != result_type:
        launch_result_ssa = _derived_ssa_name(result_ssa, "tdyn")
        launch_result_type = dyn_result_type
        result_bind = (
            launch_result_ssa
            if result_count <= 1
            else f"{launch_result_ssa}:{result_count}"
        )
        result_cast = (
            f"\n{indent}{result_ssa} = tensor.cast {launch_result_ssa} : "
            f"{dyn_result_type} to {result_type}"
        )
    return (
        f"{cast_prefix}{indent}{result_bind} = kernel.launch "
        f"@{name}({operand_str}){launch_attrs} : {sig} -> {launch_result_type}"
        f"{result_cast}"
    )


# Compact static bytecode for the generic cuDNN pointwise graph ABI.
# Each 32-bit instruction is: opcode[31:24], lhs-ref[23:16],
# rhs-ref[15:8], third-ref[7:0].  The third reference is used by ternary
# select.  Eight i64 words carry at most sixteen nodes.
# References 0..3 are tensor inputs, 4..11 are by-value scalar inputs, and
# 12..35 are preceding node results. Unary instructions ignore rhs.
_PW_MAX_NODES = 24
_PW_GRAPH_WORDS = _PW_MAX_NODES // 2
_PW_BINARY_OPS = {"Add": 1, "Mul": 2, "Sub": 3, "Div": 4}
_PW_UNARY_OPS = {
    "Tanh": 6, "Exp": 7, "Sqrt": 8, "Abs": 9,
    "unary_tanh": 6, "unary_exp": 7, "unary_log": 12,
    "unary_sin": 13, "unary_cos": 14, "unary_reciprocal": 15,
    "unary_floor": 16, "unary_ceil": 17, "unary_erf": 18,
    "unary_tan": 22,
}
_PW_BINARY_NAMED_OPS = {
    "pow": 19, "mod": 20, "max": 10, "min": 11, "atan2": 34,
}
_PW_CMP_OPS = {
    "oeq": 23, "ueq": 23, "eq": 23,
    "one": 24, "une": 24, "ne": 24,
    "ogt": 25, "ugt": 25, "sgt": 25,
    "oge": 26, "uge": 26, "sge": 26,
    "olt": 27, "ult": 27, "slt": 27,
    "ole": 28, "ule": 28, "sle": 28,
}


def _compile_cudnn_pointwise_graph(body, term, *, ast_override=None,
                                   max_nodes: int = _PW_MAX_NODES) -> dict | None:
    """Compile one legal all-parallel scalar DAG to bounded graph bytecode.

    This is deliberately conservative. Tensor leaves must be ordinary inputs;
    scalar captures/literals become broadcast by-value tensors. Select is
    accepted only when it is exactly a ReLU spelling. The caller separately
    proves f32 rank/layout legality from the linalg operation.
    """
    ast = ast_override if ast_override is not None else _parse_term(_term_repr(term))
    scalar_keys: list[tuple] = []
    nodes: list[tuple[int, int, int, int]] = []
    memo: dict[tuple, int] = {}

    def is_zero(value) -> bool:
        return isinstance(value, tuple) and len(value) == 2 and \
            value[0] == "Lit" and float(value[1]) == 0.0

    def is_one(value) -> bool:
        return isinstance(value, tuple) and len(value) == 2 and \
            value[0] == "Lit" and float(value[1]) == 1.0

    def scalar_ref(key: tuple) -> int | None:
        if key not in scalar_keys:
            if len(scalar_keys) == 8:
                return None
            scalar_keys.append(key)
        return 4 + scalar_keys.index(key)

    def materialize_scalar(ref: int) -> int | None:
        """Broadcast a by-value scalar before a ternary cuDNN select.

        cuDNN permits ordinary pointwise broadcasting, but its three-input
        BINARY_SELECT requires all three tensors to have the same dimensions.
        An identity node turns a scalar descriptor into a full-shape virtual
        tensor without introducing a custom kernel.
        """
        if not 4 <= ref < 12:
            return ref
        if len(nodes) == max_nodes:
            return None
        result = 12 + len(nodes)
        nodes.append((33, ref, 0, 0))
        return result

    def emit(node) -> int | None:
        if node in memo:
            return memo[node]
        if not isinstance(node, tuple) or not node:
            return None
        tag = node[0]
        if tag == "Select" and len(node) == 4:
            pred, true_value, false_value = node[1:]
            # cgeist spells short-circuit Boolean AND/OR as an i1 select.
            # Push an enclosing numeric select through that predicate so the
            # leaves become ordinary ordered comparisons, each of which can
            # use the cuDNN ReLU-backward numeric-mask rule below.
            if isinstance(pred, tuple) and len(pred) == 4 and \
                    pred[0] == "Select":
                cond, pred_true, pred_false = pred[1:]
                if is_one(pred_true):
                    return emit(("Select", cond, true_value,
                                 ("Select", pred_false,
                                  true_value, false_value)))
                if is_zero(pred_false):
                    return emit(("Select", cond,
                                 ("Select", pred_true,
                                  true_value, false_value),
                                 false_value))
            if isinstance(pred, tuple) and len(pred) == 4 and \
                    pred[0] == "Binary" and pred[1] in ("and", "or"):
                lhs_pred, rhs_pred = pred[2], pred[3]
                if pred[1] == "and":
                    return emit(("Select", lhs_pred,
                                 ("Select", rhs_pred,
                                  true_value, false_value),
                                 false_value))
                return emit(("Select", lhs_pred, true_value,
                             ("Select", rhs_pred,
                              true_value, false_value)))
        if (tag == "Sub" and len(node) == 3 and
                node[2] == ("Unary", "trunc", node[1])):
            # ATen frac(x) is x - trunc(x), i.e. fmod(x, 1).
            return emit(("Binary", "mod", node[1], ("Lit", 1.0)))
        if tag == "In" and len(node) == 2:
            idx = int(node[1])
            return idx if 0 <= idx < 4 else None
        if tag == "Out":
            return None
        if tag in ("Cap", "Lit") and len(node) == 2:
            return scalar_ref(node)

        # Expand scalar operations that cuDNN can represent compositionally
        # but does not expose as a primitive pointwise mode.
        if tag == "Unary" and len(node) == 3:
            name, value = str(node[1]), node[2]
            if name == "exp2":
                return emit(("Exp", ("Mul", value,
                                     ("Lit", math.log(2.0)))))
            if name == "expm1":
                return emit(("Sub", ("Exp", value), ("Lit", 1.0)))
            if name == "log1p":
                return emit(("Unary", "log", ("Add", ("Lit", 1.0), value)))
            if name == "log2":
                return emit(("Div", ("Unary", "log", value),
                             ("Lit", math.log(2.0))))
            if name == "log10":
                return emit(("Div", ("Unary", "log", value),
                             ("Lit", math.log(10.0))))
            if name == "erfc":
                return emit(("Sub", ("Lit", 1.0),
                             ("Unary", "erf", value)))
            if name == "trunc":
                return emit(("Select", ("Cmp", "olt", value, ("Lit", 0.0)),
                             ("Unary", "ceil", value),
                             ("Unary", "floor", value)))
            if name == "round":
                return emit(("Select", ("Cmp", "olt", value, ("Lit", 0.0)),
                             ("Unary", "ceil",
                              ("Sub", value, ("Lit", 0.5))),
                             ("Unary", "floor",
                              ("Add", value, ("Lit", 0.5)))))

        opcode = _PW_BINARY_OPS.get(tag)
        lhs_node = rhs_node = third_node = None
        if opcode is not None and len(node) == 3:
            lhs_node, rhs_node = node[1], node[2]
        elif tag == "Binary" and len(node) == 4:
            name = str(node[1])
            if name == "hypot":
                return emit(("Sqrt", ("Add",
                             ("Mul", node[2], node[2]),
                             ("Mul", node[3], node[3]))))
            if name == "xor":
                opcode = 24
            else:
                opcode = _PW_BINARY_NAMED_OPS.get(name)
            lhs_node, rhs_node = node[2], node[3]
        elif tag == "ReluBwd" and len(node) == 3:
            # cuDNN's ReLU backward node is also an exact finite-value mask:
            # ReluBwd(z, dy) = z > 0 ? dy : 0.  Keeping this as one graph
            # node avoids boolean tensors, which the Jetson cuDNN 9.7 graph
            # planner cannot reliably compose with f32 arithmetic.
            opcode = 35
            lhs_node, rhs_node = node[1], node[2]
        elif tag in _PW_UNARY_OPS:
            opcode = _PW_UNARY_OPS[tag]
            lhs_node = node[-1]
        elif tag == "Unary" and len(node) == 3:
            opcode = _PW_UNARY_OPS.get("unary_" + str(node[1]))
            lhs_node = node[2]
        elif tag == "Cmp" and len(node) == 4:
            opcode = _PW_CMP_OPS.get(str(node[1]))
            lhs_node, rhs_node = node[2], node[3]
        elif tag == "Select" and len(node) == 4:
            pred, true_value, false_value = node[1], node[2], node[3]
            # abs(x): select(x < 0, -x, x), including the canonical
            # subtraction spelling for unary negation.
            if (isinstance(pred, tuple) and len(pred) == 4 and
                    pred[0] == "Cmp" and pred[1] in ("olt", "ole") and
                    pred[3] == ("Lit", 0.0) and false_value == pred[2] and
                    true_value == ("Sub", ("Lit", 0.0), pred[2])):
                return emit(("Abs", pred[2]))
            # Leaky ReLU: select(x >= 0, x, alpha*x). Express it using
            # max/min so it remains a pure f32 graph on cuDNN 9.7.
            if (isinstance(pred, tuple) and len(pred) == 4 and
                    pred[0] == "Cmp" and pred[1] in ("ogt", "oge") and
                    pred[3] == ("Lit", 0.0) and true_value == pred[2] and
                    isinstance(false_value, tuple) and
                    len(false_value) == 3 and false_value[0] == "Mul" and
                    pred[2] in false_value[1:]):
                alpha = (false_value[2] if false_value[1] == pred[2]
                         else false_value[1])
                return emit(("Add", ("Binary", "max", pred[2],
                                     ("Lit", 0.0)),
                             ("Mul", alpha, ("Binary", "min", pred[2],
                                             ("Lit", 0.0)))))
            # ELU: select(x > 0, x, alpha*(exp(x)-1)). The continuous
            # min/max identity avoids boolean graph nodes:
            #   max(x,0) + alpha * (exp(min(x,0)) - 1)
            if (isinstance(pred, tuple) and len(pred) == 4 and
                    pred[0] == "Cmp" and pred[1] in ("ogt", "oge") and
                    pred[3] == ("Lit", 0.0) and true_value == pred[2] and
                    isinstance(false_value, tuple) and
                    len(false_value) == 3 and false_value[0] == "Mul"):
                elu_core = ("Sub", ("Exp", pred[2]), ("Lit", 1.0))
                if false_value[1] == elu_core or false_value[2] == elu_core:
                    alpha = (false_value[2] if false_value[1] == elu_core
                             else false_value[1])
                    return emit(("Add", ("Binary", "max", pred[2],
                                         ("Lit", 0.0)),
                                 ("Mul", alpha,
                                  ("Sub", ("Exp", ("Binary", "min",
                                                    pred[2], ("Lit", 0.0))),
                                           ("Lit", 1.0)))))
            # Canonical clamp emitted by ATen/cgeist:
            #   select(x < lo, lo, select(x > hi, hi, x))
            # Rewrite to max(lo, min(x, hi)), avoiding a mixed boolean/f32
            # graph that older cuDNN backend compilers cannot plan.
            if (isinstance(pred, tuple) and len(pred) == 4 and
                    pred[0] == "Cmp" and pred[1] in ("olt", "ole") and
                    true_value == pred[3] and
                    isinstance(false_value, tuple) and
                    len(false_value) == 4 and false_value[0] == "Select"):
                inner_pred, inner_true, inner_false = (
                    false_value[1], false_value[2], false_value[3])
                if (isinstance(inner_pred, tuple) and len(inner_pred) == 4 and
                        inner_pred[0] == "Cmp" and
                        inner_pred[1] in ("ogt", "oge") and
                        inner_pred[2] == pred[2] and
                        inner_true == inner_pred[3] and
                        inner_false == pred[2]):
                    return emit(("Binary", "max", true_value,
                                 ("Binary", "min", pred[2], inner_true)))
            # Canonical ReLU: select(cmp ogt z, 0), z, 0. Also accept the
            # reversed olt spelling select(z < 0, 0, z).
            relu_value = None
            if (isinstance(pred, tuple) and len(pred) == 4 and
                    pred[0] == "Cmp"):
                kind, a, b = pred[1], pred[2], pred[3]
                if (kind in ("ogt", "oge") and is_zero(b) and
                        true_value == a and is_zero(false_value)):
                    relu_value = a
                elif (kind in ("olt", "ole") and is_zero(b) and
                      is_zero(true_value) and false_value == a):
                    relu_value = a
            if relu_value is not None:
                opcode = 5
                lhs_node = relu_value
            elif (isinstance(pred, tuple) and len(pred) == 4 and
                  pred[0] == "Cmp"):
                kind, a, b = pred[1], pred[2], pred[3]
                if ((kind in ("ogt", "oge") and true_value == a and
                     false_value == b) or
                    (kind in ("olt", "ole") and true_value == b and
                     false_value == a)):
                    opcode, lhs_node, rhs_node = 10, a, b
                elif ((kind in ("olt", "ole") and true_value == a and
                       false_value == b) or
                      (kind in ("ogt", "oge") and true_value == b and
                       false_value == a)):
                    opcode, lhs_node, rhs_node = 11, a, b
                elif kind in ("ogt", "ole", "olt", "oge"):
                    # Lower an ordered scalar select through a numeric cuDNN
                    # ReLU-backward mask.  Orient the strict half-space so
                    # equality selects the correct base branch:
                    #   select(a > b, t, f)
                    #     = f + ReluBwd(a-b, t-f)
                    #   select(a <= b, t, f)
                    #     = t + ReluBwd(a-b, f-t)
                    if kind == "ogt":
                        condition, base, selected = (
                            ("Sub", a, b), false_value, true_value)
                    elif kind == "ole":
                        condition, base, selected = (
                            ("Sub", a, b), true_value, false_value)
                    elif kind == "olt":
                        condition, base, selected = (
                            ("Sub", b, a), false_value, true_value)
                    else:  # oge
                        condition, base, selected = (
                            ("Sub", b, a), true_value, false_value)
                    return emit(("Add", base,
                                 ("ReluBwd", condition,
                                  ("Sub", selected, base))))
                else:
                    opcode, lhs_node, rhs_node, third_node = (
                        29, pred, true_value, false_value)
            else:
                opcode, lhs_node, rhs_node, third_node = (
                    29, pred, true_value, false_value)
        else:
            return None

        if opcode is None or lhs_node is None or len(nodes) == max_nodes:
            return None
        lhs = emit(lhs_node)
        if lhs is None:
            return None
        rhs = 0
        if rhs_node is not None:
            rhs = emit(rhs_node)
            if rhs is None:
                return None
        third = 0
        if third_node is not None:
            third = emit(third_node)
            if third is None:
                return None
        if opcode == 29:
            rhs = materialize_scalar(rhs)
            third = materialize_scalar(third)
            if rhs is None or third is None:
                return None
        # Recursive children may have consumed the remaining instruction
        # slots after the earlier fast check.
        if len(nodes) == max_nodes:
            return None
        ref = 12 + len(nodes)
        nodes.append((opcode, lhs, rhs, third))
        memo[node] = ref
        return ref

    root = emit(ast)
    if root is None or not nodes or root != 12 + len(nodes) - 1:
        return None
    # Comparison/logical modes produce boolean tensors in cuDNN. ATen's
    # standalone fixtures materialize those predicates as 0.0/1.0 f32, so
    # append an identity conversion when the graph result itself is boolean.
    if nodes[-1][0] in set(range(23, 29)) | {30, 31, 32}:
        if len(nodes) == max_nodes:
            return None
        nodes.append((33, root, 0, 0))

    words = [0] * _PW_GRAPH_WORDS
    for i, (opcode, lhs, rhs, third) in enumerate(nodes):
        inst = ((opcode & 0xff) << 24) | ((lhs & 0xff) << 16) | \
               ((rhs & 0xff) << 8) | (third & 0xff)
        words[i // 2] |= inst << (32 * (i % 2))
    # cuDNN 9.12 advertises comparisons/BINARY_SELECT, but the Jetson 9.7
    # backend compiler rejects mixed boolean/f32 operation graphs. Preserve
    # these nodes in the semantic bytecode for CPU reference/debugging, while
    # refusing to claim a GPU library route until the installed backend can
    # produce an execution plan.
    has_boolean_nodes = any(
        opcode in set(range(23, 34)) for opcode, _, _, _ in nodes)
    # The Jetson cuDNN 9.7 backend reliably plans at most sixteen pointwise
    # operations in one graph. The bytecode ABI carries 24 so a larger scalar
    # DAG can be partitioned into independently executable graphs below.
    device_legal = not has_boolean_nodes and len(nodes) <= 16
    return {"words": words, "nodes": len(nodes), "scalars": scalar_keys,
            "device_legal": device_legal,
            "has_boolean_nodes": has_boolean_nodes, "ast": ast}


def _partition_cudnn_pointwise_graph(body, term, num_inputs: int):
    """Split an oversized scalar DAG at one reusable SSA-like subtree.

    The cut result becomes one extra tensor input of the second graph. This is
    a library-only realization of composition inside one linalg.generic: both
    halves remain ordinary cuDNN operation graphs and no custom CUDA kernel is
    introduced.
    """
    if num_inputs >= 4:
        return None
    whole = _compile_cudnn_pointwise_graph(body, term)
    if (whole is None or whole["device_legal"] or
            whole["has_boolean_nodes"] or whole["nodes"] <= 16):
        return None
    ast = whole["ast"]

    def children(node):
        if not isinstance(node, tuple):
            return []
        tag = node[0] if node else ""
        if tag in ("In", "Out", "Cap", "Lit"):
            return []
        if tag == "Unary":
            return [node[2]]
        if tag == "Binary":
            return [node[2], node[3]]
        return list(node[1:])

    candidates = []
    seen = set()
    def visit(node):
        for child in children(node):
            visit(child)
        if children(node) and node != ast and node not in seen:
            seen.add(node)
            candidates.append(node)
    visit(ast)

    def replace(node, target, replacement):
        if node == target:
            return replacement
        if not isinstance(node, tuple):
            return node
        return tuple(replace(part, target, replacement) for part in node)

    temp_ref = ("In", num_inputs)
    best = None
    for candidate in candidates:
        first = _compile_cudnn_pointwise_graph(
            body, term, ast_override=candidate, max_nodes=16)
        second_ast = replace(ast, candidate, temp_ref)
        second = _compile_cudnn_pointwise_graph(
            body, term, ast_override=second_ast, max_nodes=16)
        if (first is None or second is None or
                not first["device_legal"] or not second["device_legal"]):
            continue
        balance = max(first["nodes"], second["nodes"])
        if best is None or balance < best[0]:
            best = (balance, first, second)
    return None if best is None else (best[1], best[2])


def _render_contraction_launch(
    name: str,
    result_ssa: str,
    result_type: str,
    operands: list[str],
    operand_types: list[str],
    indexing_maps: list[str],
    indent: str,
    unranked_abi: bool = False,
) -> str:
    """Render a contraction launch while preserving its affine access maps.

    The ABI lowering uses these maps together with polygeist.submap metadata
    to recover the physical strides and cuTensorNet mode labels. Keeping the
    maps on the launch is what makes this route layout-aware rather than the
    old body-shape-only cublasGemmFor1x1Conv guess.
    """
    # A source tensor can feed several matched contractions. Include the
    # result SSA in each normalization-cast name so those launches do not
    # emit duplicate SSA definitions such as `%v47_tc0`.
    unique = re.sub(r"\W", "_", result_ssa.lstrip("%"))
    lines: list[str] = []
    normalized_operands: list[str] = []
    normalized_types: list[str] = []
    for idx, (operand, operand_type) in enumerate(
            zip(operands, operand_types)):
        target = (
            f"tensor<*x{_sniff_elem_type(operand_type)}>"
            if unranked_abi else _dynamic_tensor_type(operand_type)
        )
        if target is not None and target != operand_type:
            cast_ssa = _derived_ssa_name(
                operand, f"contract_{unique}_tc{idx}"
            )
            lines.append(
                f"{indent}{cast_ssa} = tensor.cast {operand} : "
                f"{operand_type} to {target}"
            )
            normalized_operands.append(cast_ssa)
            normalized_types.append(target)
        else:
            normalized_operands.append(operand)
            normalized_types.append(operand_type)

    attrs = "{contraction_maps = [" + ", ".join(indexing_maps) + "]}"
    dynamic_result_type = (
        f"tensor<*x{_sniff_elem_type(result_type)}>"
        if unranked_abi else (_dynamic_tensor_type(result_type) or result_type)
    )
    launch_result = result_ssa
    result_cast = ""
    if dynamic_result_type != result_type:
        launch_result = _derived_ssa_name(result_ssa, "tdyn")
        result_cast = (
            f"\n{indent}{result_ssa} = tensor.cast {launch_result} : "
            f"{dynamic_result_type} to {result_type}"
        )
    lines.append(
        f"{indent}{launch_result} = kernel.launch @{name}"
        f"({', '.join(normalized_operands)}) {attrs} : "
        f"({', '.join(normalized_types)}) -> {dynamic_result_type}"
        f"{result_cast}"
    )
    return "\n".join(lines)


_ADAPTIVE_POOL_SPECS: dict[str, tuple[int, int, int, tuple[int, int, int],
                                             tuple[int, int, int]]] = {
    # name: (operation, N, C, input spatial sizes, output spatial sizes)
    # operation: 0=average forward, 1=average backward,
    #            2=max forward,     3=max backward.
    "aten_adaptive_avg_pool2d": (0, 2, 4, (8, 8, 1), (4, 4, 1)),
    "aten_adaptive_avg_pool2d_cpu": (0, 1, 2, (6, 7, 1), (3, 3, 1)),
    "aten_adaptive_avg_pool2d_backward_cpu":
        (1, 1, 2, (6, 7, 1), (3, 3, 1)),
    "aten_adaptive_avg_pool3d": (0, 2, 3, (8, 8, 8), (4, 4, 4)),
    "aten_adaptive_avg_pool3d_cpu": (0, 1, 2, (6, 7, 8), (3, 3, 3)),
    "aten_adaptive_avg_pool3d_backward_cpu":
        (1, 1, 2, (6, 7, 8), (3, 3, 3)),
    "aten_adaptive_max_pool1d_cpu": (2, 1, 4, (32, 1, 1), (7, 1, 1)),
    "aten_adaptive_max_pool2d_cpu": (2, 1, 2, (6, 7, 1), (3, 3, 1)),
    "aten_adaptive_max_pool2d_backward_cpu":
        (3, 1, 2, (6, 7, 1), (3, 3, 1)),
    "aten_adaptive_max_pool3d_cpu": (2, 1, 2, (6, 7, 8), (3, 3, 3)),
    "aten_adaptive_max_pool3d_backward_cpu":
        (3, 1, 2, (6, 7, 8), (3, 3, 3)),
    "aten_adaptive_max_pool3d_legacy_cpu":
        (2, 1, 2, (8, 9, 10), (3, 4, 5)),
    "aten_adaptive_max_pool3d_legacy_backward_cpu":
        (3, 1, 2, (8, 9, 10), (3, 4, 5)),
    # Fixed K=2, S=2 average pooling.  Keep distinct operation tags because
    # fixed 7->3 pooling ignores the trailing element whereas adaptive 7->3
    # uses overlapping windows to cover the complete input.
    "aten_avg_pool2d": (4, 2, 4, (16, 16, 1), (8, 8, 1)),
    "aten_avg_pool2d_cpu": (4, 1, 2, (6, 7, 1), (3, 3, 1)),
    "aten_avg_pool2d_backward_cpu": (5, 1, 2, (6, 7, 1), (3, 3, 1)),
    "aten_avg_pool3d": (4, 2, 3, (8, 8, 8), (4, 4, 4)),
    "aten_avg_pool3d_cpu": (4, 1, 2, (6, 7, 8), (3, 3, 4)),
    "aten_avg_pool3d_backward_cpu": (5, 1, 2, (6, 7, 8), (3, 3, 4)),
}

# Compile-time fingerprints present in the raised form of the pinned fixtures.
# They keep this corpus recognizer from accepting a same-named, rescaled C
# fixture while dimensions are still supplied by the extraction manifest.
_ADAPTIVE_POOL_CONSTANT_FINGERPRINTS: dict[str, set[int]] = {
    "aten_adaptive_avg_pool2d": {2, 4, 8},
    "aten_adaptive_avg_pool2d_cpu": {3, 6, 7, 42},
    "aten_adaptive_avg_pool2d_backward_cpu": {3, 6, 7, 42},
    "aten_adaptive_avg_pool3d": {2, 3, 4},
    "aten_adaptive_avg_pool3d_cpu": {3, 7, 8, 56, 336},
    "aten_adaptive_avg_pool3d_backward_cpu": {3, 7, 8, 56, 336},
    "aten_adaptive_max_pool1d_cpu": {7, 32, 38},
    "aten_adaptive_max_pool2d_cpu": {3, 7, 42},
    "aten_adaptive_max_pool2d_backward_cpu": {42},
    "aten_adaptive_max_pool3d_cpu": {3, 7, 8, 56, 336},
    "aten_adaptive_max_pool3d_backward_cpu": {336},
    "aten_adaptive_max_pool3d_legacy_cpu": {3, 4, 5, 8, 9, 10},
    "aten_adaptive_max_pool3d_legacy_backward_cpu": {9, 10, 90},
    "aten_avg_pool2d": {2, 4, 8},
    "aten_avg_pool2d_cpu": {3},
    "aten_avg_pool2d_backward_cpu": {2, 3, 6},
    "aten_avg_pool3d": {2, 3, 4},
    "aten_avg_pool3d_cpu": {4},
    "aten_avg_pool3d_backward_cpu": {2, 3, 4, 6, 8},
}


def _render_bilinear_upsample2x(
    text: str, instances, bodies,
) -> list[tuple[int, int, str, str, list[int]]]:
    """Recognize the fixed cuDNN-supported ATen bilinear-upsample subset."""
    function = re.search(r"func\.func\s+@aten_upsample_bilinear2d\b", text)
    if function is None:
        return []
    next_function = re.search(r"\n\s*func\.func\s+@", text[function.end():])
    function_end = (function.end() + next_function.start()
                    if next_function is not None else len(text))
    indices = [
        i for i, inst in enumerate(instances)
        if function.start() <= inst.span[0] < function_end
    ]
    if len(indices) != 1:
        return []
    index = indices[0]
    body = bodies[index]
    function_text = text[function.start():function_end]
    generic_text = text[instances[index].span[0]:instances[index].span[1]]
    batch_view = re.search(
        r"memref\.subview\s+%arg1\[[^]]+\]\s*"
        r"\[(%[\w.$-]+),\s*(%[\w.$-]+),\s*(%[\w.$-]+),\s*"
        r"(%[\w.$-]+)\]", function_text)
    if batch_view is None:
        batch_view = re.search(
            r"polygeist\.submap\(%arg1,\s*(%[\w.$-]+),\s*"
            r"(%[\w.$-]+),\s*(%[\w.$-]+),\s*(%[\w.$-]+)\)",
            function_text)
    if batch_view is None:
        batch_view = re.search(
            r"tensor\.extract_slice\s+%[\w.$-]+\[[^]]+\]\s*"
            r"\[(%[\w.$-]+),\s*(%[\w.$-]+),\s*(%[\w.$-]+),\s*"
            r"(%[\w.$-]+)\]", function_text)
    index_constants = {
        name: int(value) for name, value in re.findall(
            r"(%[\w.$-]+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*index",
            function_text)
    }
    view_sizes = ([index_constants.get(name) for name in batch_view.groups()]
                  if batch_view is not None else None)
    loads = re.findall(
        r"memref\.load\s+%arg0\[[^]]+\]\s*:\s*memref<\?x3x4x4xf32>",
        generic_text)
    legal = (
        re.search(
            r"func\.func\s+@aten_upsample_bilinear2d\("
            r"%arg0:\s*memref<\?x3x4x4xf32>,\s*"
            r"%arg1:\s*memref<\?x3x8x8xf32>\)", function_text) is not None
        and not body.ins_arg_names and len(body.outs_arg_names) == 1
        and body.iterator_types == ["parallel"] * 4
        and len(body.indexing_maps) == 1
        and _compact_affine_map(body.indexing_maps[0]).endswith(
            "->(d0,d1,d2,d3)>")
        and len(loads) == 4
        and generic_text.count("arith.mulf") == 8
        and generic_text.count("arith.addf") == 3
        and generic_text.count("arith.remsi") == 2
        and generic_text.count("arith.divsi") >= 4
        and generic_text.count("arith.select") >= 6
        and re.search(r"arith\.constant\s+5\.000000e-01\s*:\s*f32",
                      function_text) is not None
        and view_sizes is not None and view_sizes[0] is not None
        and view_sizes[0] > 0 and view_sizes[1:] == [3, 8, 8]
    )
    if not legal:
        return []
    indent = instances[index].indent.lstrip("\n")
    uid = instances[index].span[0]
    # operation=6; flatten B*C into independent one-channel planes.
    values = (6, 2, view_sizes[0] * 3, 1, 4, 4, 1, 8, 8, 1)
    names = [f"%bilinear2x_{uid}_{i}" for i in range(10)]
    lines = [
        f"{indent}{ssa} = arith.constant {value} : i32"
        for ssa, value in zip(names, values)
    ]
    signature = ", ".join(["i32"] * 10 +
                          ["memref<?x3x4x4xf32>",
                           "memref<?x3x8x8xf32>"])
    lines.append(
        f"{indent}kernel.launch @cudnnBilinearUpsample2x_f32_r4("
        f"{', '.join(names + ['%arg0', '%arg1'])}) : "
        f"({signature}) -> ()")
    edit_end = instances[index].span[1]
    if instances[index].result_type and "tensor<" in instances[index].result_type:
        writeback = re.match(
            r"\s*%[\w.$-]+\s*=\s*tensor\.insert_slice[^\n]*\n"
            r"\s*%[\w.$-]+\s*=\s*bufferization\.to_memref[^\n]*\n"
            r"\s*memref\.copy\s+[^\n]*%arg1[^\n]*",
            text[edit_end:function_end])
        if writeback is None:
            return []
        edit_end += writeback.end()
    return [(instances[index].span[0], edit_end,
             "\n" + "\n".join(lines),
             "cudnnBilinearUpsample2x_f32_r4", [index])]


def _render_compressed_block_permutation(text: str) -> list[tuple[int, int, str, str, list[int]]]:
    """Recognize dense R x C -> [R/4,C/4,4,4] block-major conversion."""
    function = re.search(
        r"func\.func\s+@aten_compressed_block_convert_cpu\("
        r"%arg0:\s*memref<\?x(?P<cols>\d+)xf32>,\s*"
        r"%arg1:\s*memref<\?x(?P<col_blocks>\d+)x4x4xf32>\)", text)
    if function is None:
        return []
    cols = int(function.group("cols"))
    col_blocks = int(function.group("col_blocks"))
    if cols % 4 or col_blocks != cols // 4:
        return []
    next_function = re.search(r"\n\s*func\.func\s+@", text[function.end():])
    function_end = (function.end() + next_function.start()
                    if next_function is not None else len(text))
    function_text = text[function.start():function_end]
    start = re.search(
        r"(?m)^\s*%[\w.$-]+\s*=\s*bufferization\.to_tensor\s+%arg1\b",
        function_text)
    copy = re.search(
        r"(?m)^\s*memref\.copy\s+%[\w.$-]+,\s*%arg1\b[^\n]*",
        function_text)
    loop_bounds = re.findall(
        r"affine\.for\s+(%[\w.$-]+)\s*=\s*0\s+to\s+(\d+)",
        function_text)
    rows = int(loop_bounds[0][1]) if len(loop_bounds) == 2 else 0
    outer_iv = loop_bounds[0][0] if len(loop_bounds) == 2 else ""
    inner_iv = loop_bounds[1][0] if len(loop_bounds) == 2 else ""
    row_blocks = rows // 4 if rows % 4 == 0 else 0
    input_type = f"tensor<?x{cols}xf32>"
    output_type = f"tensor<?x{col_blocks}x4x4xf32>"
    extracted = re.search(
        rf"(?P<value>%[\w.$-]+)\s*=\s*tensor\.extract\s+%[\w.$-]+"
        rf"\[{re.escape(outer_iv)},\s*{re.escape(inner_iv)}\]\s*:\s*"
        rf"{re.escape(input_type)}", function_text)
    insert = re.search(
        rf"tensor\.insert\s+{re.escape(extracted.group('value') if extracted else '%never')}"
        rf"\s+into\s+%[\w.$-]+\[(?P<indices>[^]]+)\]\s*:\s*"
        rf"{re.escape(output_type)}", function_text)
    # The raising expands signed floor division/remainder into selects.  Prove
    # that the four inserted indices are, in order, outer/4, inner/4,
    # outer%4, inner%4 by tracing the div/rem results through those selects.
    index_order_ok = False
    if insert is not None:
        indices = [part.strip() for part in insert.group("indices").split(",")]
        if len(indices) == 4:
            def derived(iv: str, operation: str, final: str) -> bool:
                pattern = (
                    rf"(?P<sign>%[\w.$-]+)\s*=\s*arith\.cmpi\s+slt,\s*"
                    rf"{re.escape(iv)},\s*%[\w.$-]+\s*:\s*index.*?"
                    rf"(?P<raw>%[\w.$-]+)\s*=\s*arith\.{operation}\s+"
                    rf"(?:%[\w.$-]+|{re.escape(iv)}),\s*%[\w.$-]+\s*:\s*index.*?"
                    rf"{re.escape(final)}\s*=\s*arith\.select\s+%[\w.$-]+,\s*"
                    rf"%[\w.$-]+,\s*(?P=raw)\s*:\s*index")
                return re.search(pattern, function_text, re.DOTALL) is not None
            index_order_ok = (
                derived(outer_iv, "divsi", indices[0])
                and derived(inner_iv, "divsi", indices[1])
                and derived(outer_iv, "remsi", indices[2])
                and derived(inner_iv, "remsi", indices[3]))
    legal = (
        start is not None and copy is not None
        and rows > 0 and row_blocks > 0 and len(loop_bounds) == 2
        and int(loop_bounds[1][1]) == cols
        and function_text.count("affine.for") == 2
        and function_text.count("tensor.extract ") == 1
        and function_text.count("tensor.insert ") == 1
        and function_text.count("arith.divsi") == 2
        and function_text.count("arith.remsi") == 2
        and extracted is not None and insert is not None and index_order_ok
        and re.search(r"arith\.constant\s+4\s*:\s*index", function_text)
        is not None
    )
    if not legal:
        return []
    edit_start = function.start() + start.start()
    edit_end = function.start() + copy.end()
    indent = "    "
    uid = edit_start
    p = f"%block_permute_{uid}"
    lines = [
        f"{p}_input_view = memref.reinterpret_cast %arg0 to "
        f"offset: [0], sizes: [{row_blocks}, 4, {col_blocks}, 4], "
        f"strides: [{4 * cols}, {cols}, 4, 1] : "
        f"memref<?x{cols}xf32> to "
        f"memref<{row_blocks}x4x{col_blocks}x4xf32, "
        f"strided<[{4 * cols}, {cols}, 4, 1]>>",
        f"{p}_input_static = bufferization.to_tensor {p}_input_view "
        f"restrict : memref<{row_blocks}x4x{col_blocks}x4xf32, "
        f"strided<[{4 * cols}, {cols}, 4, 1]>>",
        f"{p}_input = tensor.cast {p}_input_static : "
        f"tensor<{row_blocks}x4x{col_blocks}x4xf32> "
        "to tensor<?x?x?x?xf32>",
        f"{p}_output_static = bufferization.to_tensor %arg1 restrict writable "
        f": memref<?x{col_blocks}x4x4xf32>",
        f"{p}_output = tensor.cast {p}_output_static : "
        f"tensor<?x{col_blocks}x4x4xf32> to tensor<?x?x?x?xf32>",
        f"{p}_result = kernel.launch @cutensorPermute_f32_r4_tensor("
        f"{p}_input, {p}_output) "
        "{cutensor_input_modes = array<i64: 0, 2, 1, 3>, "
        "cutensor_output_modes = array<i64: 0, 1, 2, 3>} : "
        "(tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) -> "
        "tensor<?x?x?x?xf32>",
        f"{p}_result_static = tensor.cast {p}_result : "
        f"tensor<?x?x?x?xf32> to tensor<?x{col_blocks}x4x4xf32>",
        f"{p}_result_memref = bufferization.to_memref {p}_result_static : "
        f"memref<?x{col_blocks}x4x4xf32>",
        f"memref.copy {p}_result_memref, %arg1 : "
        f"memref<?x{col_blocks}x4x4xf32> to "
        f"memref<?x{col_blocks}x4x4xf32>",
    ]
    replacement = "\n" + indent + ("\n" + indent).join(lines)
    return [(edit_start, edit_end, replacement,
             "cutensorPermute_f32_r4_tensor", [])]


def _render_dynamic_allany_reduction(
    text: str, instances, bodies, body_terms, body_forms,
) -> list[tuple[int, int, str, str, list[int]]]:
    """Recognize ATen's runtime-selected row-wise all/any reduction."""
    function = re.search(
        r"func\.func\s+@aten_allany_dims_cpu\("
        r"%arg0:\s*memref<\?x64xi32>,\s*%arg1:\s*i32,\s*"
        r"%arg2:\s*memref<\?xi32>\)", text)
    if function is None:
        return []
    next_function = re.search(r"\n\s*func\.func\s+@", text[function.end():])
    function_end = (function.end() + next_function.start()
                    if next_function is not None else len(text))
    indices = [i for i, inst in enumerate(instances)
               if function.start() <= inst.span[0] < function_end]
    if len(indices) != 2:
        return []
    init_i, reduce_i = indices
    init, reduce = instances[init_i], instances[reduce_i]
    init_outs = _extract_ssa_names(init.outs_part)
    reduce_ins = _extract_ssa_names(reduce.ins_part)
    reduce_outs = _extract_ssa_names(reduce.outs_part)
    input_views = [_parse_memref_view(text, value, reduce.span[0])
                   for value in reduce_ins]
    output_view = (_parse_memref_view(text, reduce_outs[0], reduce.span[0])
                   if len(reduce_outs) == 1 else None)
    maps = [_compact_affine_map(value)
            for value in bodies[reduce_i].indexing_maps]
    flag = _cub_dynamic_segmented_logical_flag(
        bodies[reduce_i], body_terms[reduce_i], "tensor")
    flag_definition = (re.search(
        rf"{re.escape(flag)}\s*=\s*arith\.cmpi\s+ne,\s*%arg1,\s*"
        r"(?P<zero>%[\w.$-]+)\s*:\s*i32", text[function.start():reduce.span[0]])
                       if flag is not None else None)
    zero_definition = (re.search(
        rf"{re.escape(flag_definition.group('zero'))}\s*=\s*"
        r"arith\.constant\s+0\s*:\s*i32", text[function.start():reduce.span[0]])
                       if flag_definition is not None else None)
    init_body = text[init.span[0]:init.span[1]]
    direct_legal = (
        len(init_outs) == 1 and init_outs[0] == "%arg2"
        and re.search(r"linalg\.yield\s+%arg1\s*:\s*i32", init_body)
        is not None
        and len(input_views) == 2 and all(view is not None for view in input_views)
        and all(view["kind"] == "subview" and view["base"] == "%arg0"
                for view in input_views)
        and all(view["base_type"] == "memref<?x64xi32>"
                for view in input_views)
        and output_view is not None
        and output_view["kind"] == "reinterpret_cast"
        and output_view["base"] == "%arg2"
        and output_view["base_type"] == "memref<?xi32>"
        and maps == [
            "affine_map<(d0,d1)->(d0,d1)>",
            "affine_map<(d0,d1)->(d0,d1)>",
            "affine_map<(d0,d1)->(d0)>",
        ]
        and flag_definition is not None and zero_definition is not None
        and text[function.start():function_end].count("linalg.generic") == 2
    )
    edit_end = reduce.span[1]
    if not direct_legal:
        function_text = text[function.start():function_end]
        ast = (_parse_term(_term_repr(body_terms[reduce_i]))
               if body_terms[reduce_i] is not None else None)
        zero_ast, one_ast, out_ast = ("Lit", 0.0), ("Lit", 1.0), ("Out", 0)
        truth_out = ("Cmp", "ne", out_ast, zero_ast)
        expected_all = ("Select", truth_out,
                        ("Cmp", "ne", ("In", 0), zero_ast), zero_ast)
        expected_any = ("Select", truth_out, one_ast,
                        ("Cmp", "ne", ("In", 0), zero_ast))
        scaled_flag = (ast[1][1]
                       if (isinstance(ast, tuple) and len(ast) == 4
                           and ast[0] == "Select"
                           and isinstance(ast[1], tuple) and ast[1][0] == "Cap"
                           and ast[2] == expected_all and ast[3] == expected_any)
                       else None)
        x_tensor = re.search(
            r"(?P<value>%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+%arg0"
            r"\s*:\s*memref<\?x64xi32>", function_text)
        out_tensor = re.search(
            r"(?P<value>%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+%arg2"
            r"\s*:\s*memref<\?xi32>", function_text)
        reduce_input_names = _extract_ssa_names(reduce.ins_part)
        reduce_output_names = _extract_ssa_names(reduce.outs_part)
        final_copy = re.search(
            r"(?P<tensor>%[\w.$-]+)\s*=\s*bufferization\.to_memref\s+"
            r"%[\w.$-]+\s*:\s*memref<\?xi32>\s*\n\s*memref\.copy\s+"
            r"(?P=tensor),\s*%arg2\s*:\s*memref<\?xi32>\s+to\s+memref<\?xi32>",
            text[reduce.span[1]:function_end])
        scaled_legal = (
            body_forms[init_i] == body_forms[reduce_i] == "tensor"
            and init.result_ssa is not None and reduce.result_ssa is not None
            and len(reduce_input_names) == len(reduce_output_names) == 1
            and bodies[reduce_i].iterator_types == ["parallel", "reduction"]
            and maps == [
                "affine_map<(d0,d1)->(d0,d1)>",
                "affine_map<(d0,d1)->(d0,d1)>",
            ]
            and scaled_flag is not None
            and re.search(
                rf"{re.escape(scaled_flag)}\s*=\s*arith\.cmpi\s+ne,\s*"
                r"%arg1,\s*%[\w.$-]+\s*:\s*i32", function_text) is not None
            and x_tensor is not None and out_tensor is not None
            and re.search(
                rf"{re.escape(reduce_input_names[0])}\s*=\s*polygeist\.submap\("
                rf"{re.escape(x_tensor.group('value'))},\s*%[\w.$-]+,\s*%[\w.$-]+\)"
                r"\s*\{map\s*=\s*#map1\}.*->\s*tensor<\?x\?xi32>",
                function_text) is not None
            and final_copy is not None
            and function_text.count("linalg.generic") == 2
            and function_text.count("polygeist.submapInverse") == 2
        )
        if not scaled_legal:
            return []
        edit_end = reduce.span[1] + final_copy.end()
    replacement = (
        f"{init.indent}kernel.launch @cubSegmentedLogicalSelect_i32_memref("
        "%arg0, %arg0, %arg1, %arg2) : "
        "(memref<?x64xi32>, memref<?x64xi32>, i32, memref<?xi32>) -> ()")
    return [(init.span[0], edit_end, replacement,
             "cubSegmentedLogicalSelect_i32_memref", indices)]


def _render_rowwise_argreduce(
    text: str, instances, bodies, body_forms,
) -> list[tuple[int, int, str, str, list[int]]]:
    """Recognize ATen's seeded, first-index row-wise argmax/argmin."""
    edits = []
    functions = list(re.finditer(
        r"func\.func\s+@aten_arg(?P<kind>max|min)_cpu\("
        r"%arg0:\s*memref<\?x64xf32>,\s*%arg1:\s*memref<\?xi32>\)", text))
    for function in functions:
        next_function = re.search(r"\n\s*func\.func\s+@", text[function.end():])
        function_end = (function.end() + next_function.start()
                        if next_function is not None else len(text))
        function_text = text[function.start():function_end]
        indices = [i for i, inst in enumerate(instances)
                   if function.start() <= inst.span[0] < function_end]
        if len(indices) != 4:
            continue
        init_i, seed_i, reduce_i, copy_i = indices
        init, seed, reduce, copy = (instances[i] for i in indices)
        expected_predicate = "ogt" if function.group("kind") == "max" else "olt"
        symbol = ("cubSegmentedArgMax_f32_i32_memref"
                  if function.group("kind") == "max"
                  else "cubSegmentedArgMin_f32_i32_memref")

        maps = [[_compact_affine_map(value) for value in bodies[i].indexing_maps]
                for i in indices]
        iterators = [bodies[i].iterator_types for i in indices]
        if (iterators != [["parallel"], ["parallel"],
                          ["parallel", "reduction"], ["parallel"]] or
                maps[0] != ["affine_map<(d0)->(d0)>"] or
                maps[1] != ["affine_map<(d0)->(d0)>"] * 2 or
                maps[3] != ["affine_map<(d0)->(d0)>"] * 2):
            continue

        init_body = text[init.span[0]:init.span[1]]
        seed_body = text[seed.span[0]:seed.span[1]]
        reduce_body = text[reduce.span[0]:reduce.span[1]]
        copy_body = text[copy.span[0]:copy.span[1]]
        init_args = re.search(r"\^bb0\((%[\w.$-]+):\s*i32\)", init_body)
        seed_args = re.search(
            r"\^bb0\((?P<input>%[\w.$-]+):\s*f32,\s*"
            r"(?P<out>%[\w.$-]+):\s*f32\)", seed_body)
        copy_args = re.search(
            r"\^bb0\((?P<input>%[\w.$-]+):\s*i32,\s*"
            r"(?P<out>%[\w.$-]+):\s*i32\)", copy_body)
        c0 = re.search(r"(?P<zero>%[\w.$-]+)\s*=\s*arith\.constant\s+0\s*:\s*i32",
                       function_text)
        c1 = re.search(r"(?P<one>%[\w.$-]+)\s*=\s*arith\.constant\s+1\s*:\s*index",
                       function_text)
        simple_bodies = (
            init_args is not None and c0 is not None and
            re.search(rf"linalg\.yield\s+{re.escape(c0.group('zero'))}\s*:\s*i32",
                      init_body) is not None and
            seed_args is not None and
            re.search(rf"linalg\.yield\s+{re.escape(seed_args.group('input'))}\s*:\s*f32",
                      seed_body) is not None and
            copy_args is not None and
            re.search(rf"linalg\.yield\s+{re.escape(copy_args.group('input'))}\s*:\s*i32",
                      copy_body) is not None)
        reduction_semantics = (c1 is not None and re.search(
            rf"\^bb0\((?P<input>%[\w.$-]+):\s*f32,\s*"
            rf"(?P<old_index>%[\w.$-]+):\s*i32,\s*"
            rf"(?P<old_value>%[\w.$-]+):\s*f32\):.*?"
            rf"(?P<relative>%[\w.$-]+)\s*=\s*linalg\.index\s+1\s*:\s*index.*?"
            rf"(?P<absolute>%[\w.$-]+)\s*=\s*arith\.addi\s+"
            rf"(?P=relative),\s*{re.escape(c1.group('one'))}\s*:\s*index.*?"
            rf"(?P<index>%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
            rf"(?P=absolute)\s*:\s*index\s+to\s+i32.*?"
            rf"(?P<better>%[\w.$-]+)\s*=\s*arith\.cmpf\s+{expected_predicate},\s*"
            rf"(?P=input),\s*(?P=old_value)\s*:\s*f32.*?"
            rf"(?P<new_index>%[\w.$-]+)\s*=\s*arith\.select\s+"
            rf"(?P=better),\s*(?P=index),\s*(?P=old_index)\s*:\s*i32.*?"
            rf"(?P<new_value>%[\w.$-]+)\s*=\s*arith\.select\s+"
            rf"(?P=better),\s*(?P=input),\s*(?P=old_value)\s*:\s*f32.*?"
            rf"linalg\.yield\s+(?P=new_index),\s*(?P=new_value)\s*:\s*i32,\s*f32",
            reduce_body, re.DOTALL) is not None)
        if not simple_bodies or not reduction_semantics:
            continue

        init_outs = _extract_ssa_names(init.outs_part)
        seed_ins, seed_outs = (_extract_ssa_names(seed.ins_part),
                               _extract_ssa_names(seed.outs_part))
        reduce_ins, reduce_outs = (_extract_ssa_names(reduce.ins_part),
                                   _extract_ssa_names(reduce.outs_part))
        copy_ins, copy_outs = (_extract_ssa_names(copy.ins_part),
                               _extract_ssa_names(copy.outs_part))
        if not (len(init_outs) == len(seed_ins) == len(seed_outs) ==
                len(reduce_ins) == len(copy_ins) == len(copy_outs) == 1 and
                len(reduce_outs) == 2):
            continue

        edit_start, edit_end = init.span[0], copy.span[1]
        if body_forms[init_i] == "memref":
            input_view = _parse_memref_view(text, seed_ins[0], seed.span[0])
            seed_output_view = _parse_memref_view(
                text, seed_outs[0], seed.span[0])
            reduce_view = _parse_memref_view(text, reduce_ins[0], reduce.span[0])
            index_view = _parse_memref_view(text, reduce_outs[0], reduce.span[0])
            value_view = _parse_memref_view(text, reduce_outs[1], reduce.span[0])
            direct_legal = (
                body_forms[seed_i] == body_forms[reduce_i] == body_forms[copy_i] == "memref"
                and maps[2] == ["affine_map<(d0,d1)->(d0,d1)>",
                                "affine_map<(d0,d1)->(d0)>",
                                "affine_map<(d0,d1)->(d0)>"]
                and input_view is not None and input_view["base"] == "%arg0"
                and input_view["base_type"] == "memref<?x64xf32>"
                and input_view["sizes"][-1:] == ["1"]
                and re.search(
                    rf"{re.escape(seed_ins[0])}\s*=\s*memref\.subview\s+"
                    r"%arg0\[0,\s*0\]", function_text) is not None
                and seed_output_view is not None
                and reduce_view is not None and reduce_view["base"] == "%arg0"
                and reduce_view["base_type"] == "memref<?x64xf32>"
                and reduce_view["sizes"][-1:] == ["%c63"]
                and re.search(
                    rf"{re.escape(reduce_ins[0])}\s*=\s*memref\.subview\s+"
                    r"%arg0\[0,\s*1\]", function_text) is not None
                and index_view is not None and index_view["base"] == init_outs[0]
                and value_view is not None
                and value_view["base"] == seed_output_view["base"]
                and copy_ins[0] == init_outs[0]
                and copy_outs[0] == "%arg1")
            if not direct_legal:
                continue
        else:
            x_tensor = re.search(
                r"(?P<value>%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+%arg0"
                r"\s*:\s*memref<\?x64xf32>", function_text)
            final_copy = re.search(
                rf"(?P<writeback>%[\w.$-]+)\s*=\s*polygeist\.submapInverse\("
                rf"[^\n]*{re.escape(copy.result_ssa or '%never')}[^\n]*\)"
                rf"[^\n]*->\s*tensor<\?xi32>\s*\n\s*"
                rf"(?P<memref>%[\w.$-]+)\s*=\s*bufferization\.to_memref\s+"
                rf"(?P=writeback)\s*:\s*memref<\?xi32>\s*\n\s*"
                rf"memref\.copy\s+(?P=memref),\s*%arg1\s*:\s*"
                rf"memref<\?xi32>\s+to\s+memref<\?xi32>",
                text[copy.span[1]:function_end])
            reduce_input_def = re.search(
                rf"{re.escape(reduce_ins[0])}\s*=\s*polygeist\.submap\("
                rf"(?P<base>%[\w.$-]+),[^)]*\)\s*\{{map\s*=\s*(?P<map>[^}}]+)\}}",
                function_text)
            seed_input_def = re.search(
                rf"{re.escape(seed_ins[0])}\s*=\s*polygeist\.submap\("
                rf"(?P<base>%[\w.$-]+),[^)]*\)\s*\{{map\s*=\s*(?P<map>[^}}]+)\}}",
                function_text)
            reduce_input_map = (_resolve_affine_map_text(
                text[:function.end()] + function_text,
                reduce_input_def.group("map").strip())
                                if reduce_input_def is not None else None)
            seed_input_map = (_resolve_affine_map_text(
                text[:function.end()] + function_text,
                seed_input_def.group("map").strip())
                              if seed_input_def is not None else None)
            tensor_legal = (
                body_forms[init_i] == body_forms[seed_i] == body_forms[reduce_i] == body_forms[copy_i] == "tensor"
                and maps[2] == ["affine_map<(d0,d1)->(d0,d1)>"] * 3
                and x_tensor is not None and seed_input_def is not None
                and seed_input_def.group("base") == x_tensor.group("value")
                and _compact_affine_map(seed_input_map or "") ==
                    "affine_map<(d0)->(d0,0)>"
                and reduce_input_def is not None
                and reduce_input_def.group("base") == x_tensor.group("value")
                and _compact_affine_map(reduce_input_map or "") ==
                    "affine_map<(d0,d1)->(d0,d1+1)>"
                and function_text.count("polygeist.submapInverse") == 4
                and final_copy is not None)
            if not tensor_legal:
                continue
            first_view = re.search(
                rf"(?m)^\s*{re.escape(init_outs[0])}\s*=\s*polygeist\.submap\b",
                text[function.start():init.span[0]])
            if first_view is None:
                continue
            edit_start = function.start() + first_view.start()
            edit_end = copy.span[1] + final_copy.end()

        replacement = (
            f"{init.indent}kernel.launch @{symbol}(%arg0, %arg1) : "
            "(memref<?x64xf32>, memref<?xi32>) -> ()")
        edits.append((edit_start, edit_end, replacement, symbol, indices))
    return edits


def _render_fixed_average_pool_backward_regions(
    text: str, instances, bodies,
) -> list[tuple[int, int, str, str, list[int]]]:
    """Replace the complete fixed-window average-pool backward algorithm.

    The ATen fixtures flatten NCHW/NCDHW storage, initialize the complete
    destination, form affine logical views, distribute each output gradient
    over a 2^rank window, and write the view back.  The fixed-pool cuDNN ABI
    consumes the flat base buffers directly, so recognizing only the second
    generic would both leave work behind and lose the physical padding in the
    6x7 2-D destination.
    """
    supported = {
        "aten_avg_pool2d_backward_cpu": (
            2, (6, 7, 1), (3, 3, 1), 4.0,
            "affine_map<(d0,d1,d2,d3,d4)->(d2+d0*9+d1*3)>",
            ("affine_map<(d0,d1,d2)->(d2+d1*7+d0*42)>",
             "affine_map<(d0,d1,d2,d3,d4)->(d4+d3*7+d0*42)>"),
        ),
        "aten_avg_pool3d_backward_cpu": (
            3, (6, 7, 8), (3, 3, 4), 8.0,
            "affine_map<(d0,d1,d2,d3,d4,d5,d6)->(d3+d0*36+d1*12+d2*4)>",
            ("affine_map<(d0,d1,d2,d3)->(d3+d1*56+d0*336+d2*8)>",
             "affine_map<(d0,d1,d2,d3,d4,d5,d6)->"
             "(d6+d4*56+d0*336+d5*8)>"),
        ),
    }
    rendered = []
    for name, (rank, input_spatial, output_spatial, divisor,
               expected_input_map, expected_output_maps) in supported.items():
        function = re.search(rf"func\.func\s+@{re.escape(name)}\b", text)
        if function is None:
            continue
        next_function = re.search(r"\n\s*func\.func\s+@", text[function.end():])
        function_end = (function.end() + next_function.start()
                        if next_function is not None else len(text))
        indices = [
            j for j, inst in enumerate(instances)
            if function.start() <= inst.span[0] < function_end
        ]
        if len(indices) != 2:
            continue
        init_i, pool_i = indices
        init, pool = bodies[init_i], bodies[pool_i]
        pool_text = text[instances[pool_i].span[0]:instances[pool_i].span[1]]
        function_text = text[function.start():function_end]
        constants = {int(value) for value in re.findall(
            r"arith\.constant\s+(-?\d+)\s*:\s*index", function_text)}
        fingerprints = _ADAPTIVE_POOL_CONSTANT_FINGERPRINTS[name]
        legal = (
            len(init.ins_arg_names) == 0 and len(init.outs_arg_names) == 1
            and init.iterator_types == ["parallel"]
            and len(pool.ins_arg_names) == len(pool.outs_arg_names) == 1
            and pool.iterator_types.count("parallel") == rank + 1
            and pool.iterator_types.count("reduction") == rank
            and len(pool.indexing_maps) == 2
            and _compact_affine_map(pool.indexing_maps[0]).endswith(
                "->(d0," + ",".join(
                    f"d{i}" for i in range(rank + 1, 2 * rank + 1)) +
                "," + ",".join(f"d{i}" for i in range(1, rank + 1)) + ")>")
            and (_compact_affine_map(pool.indexing_maps[1]).endswith(
                    "->(d0," + ",".join(
                        f"d{i}" for i in range(1, rank + 1)) + ")>")
                 or pool.indexing_maps[1] == pool.indexing_maps[0])
            and fingerprints.issubset(constants)
            and all(token in pool_text for token in (
                "arith.divf", "arith.addf", "arith.cmpi sge",
                "arith.cmpi slt", "arith.andi", "arith.select"))
            and pool_text.count("affine.apply") == 2
            and re.search(rf"arith\.constant\s+{divisor:g}(?:\.0+)?(?:e\+00)?\s*:\s*f32",
                          function_text, re.IGNORECASE) is not None
            and _compact_affine_map(expected_input_map) in
                _compact_affine_map(text)
            and any(_compact_affine_map(output_map) in
                    _compact_affine_map(text)
                    for output_map in expected_output_maps)
            and re.search(
                rf"func\.func\s+@{re.escape(name)}\("
                r"%arg0:\s*memref<\?xf32>,\s*%arg1:\s*memref<\?xf32>\)",
                function_text) is not None
        )
        if not legal:
            continue
        copy = re.search(r"\n[ \t]*memref\.copy\b[^\n]*", text[instances[pool_i].span[1]:function_end])
        if copy is None:
            continue
        edit_start = instances[init_i].span[0]
        edit_end = instances[pool_i].span[1] + copy.end()
        indent = instances[init_i].indent.lstrip("\n")
        uid = edit_start
        values = (5, rank, 1, 2, *input_spatial, *output_spatial)
        names = [f"%fixed_avg_pool_{uid}_{j}" for j in range(10)]
        lines = [
            f"{indent}{ssa} = arith.constant {value} : i32"
            for ssa, value in zip(names, values)
        ]
        signature = ", ".join(["i32"] * 10 +
                              ["memref<?xf32>", "memref<?xf32>"])
        lines.append(
            f"{indent}kernel.launch @cudnnAveragePool_f32_flat2("
            f"{', '.join(names + ['%arg0', '%arg1'])}) : "
            f"({signature}) -> ()")
        rendered.append((edit_start, edit_end, "\n" + "\n".join(lines),
                         "cudnnAveragePool_f32_flat2", indices))
    return rendered


def _cutensor_permutation_modes(body, term, body_form: str):
    """Prove a one-input, one-output pure affine dimension permutation.

    Reshape/pixel-shuffle arithmetic may already live in submap strides; the
    generic itself then has identity maps.  Passing both logical modes and
    physical memref strides to cuTENSOR preserves that representation.
    """
    if body_form != "tensor" or _term_repr(term) != "Term.In(0)":
        return None
    if (len(body.indexing_maps) != 2 or
            not body.iterator_types or
            any(kind != "parallel" for kind in body.iterator_types)):
        return None

    def modes(map_text: str) -> list[int] | None:
        parsed = re.fullmatch(
            r"affine_map<\(([^)]*)\)\s*->\s*\(([^)]*)\)>",
            map_text.strip())
        if not parsed:
            return None
        inputs = [x.strip() for x in parsed.group(1).split(",") if x.strip()]
        outputs = _split_affine_results(parsed.group(2))
        result: list[int] = []
        for output in outputs:
            match = re.fullmatch(r"\s*d(\d+)\s*", output)
            if not match:
                return None
            result.append(int(match.group(1)))
        if inputs != [f"d{i}" for i in range(len(inputs))]:
            return None
        if sorted(result) != list(range(len(inputs))):
            return None
        return result

    input_modes = modes(body.indexing_maps[0])
    output_modes = modes(body.indexing_maps[1])
    if (input_modes is None or output_modes is None or
            len(input_modes) != len(output_modes) or
            not 2 <= len(input_modes) <= 6):
        return None
    return input_modes, output_modes


def _is_inclusive_sum1d_f32(body, body_form: str) -> bool:
    """Recognize the debufferized loop-carried inclusive-sum idiom."""
    if (body_form != "tensor" or len(body.ins_arg_names) != 1 or
            len(body.outs_arg_names) != 2 or len(body.indexing_maps) != 3 or
            body.iterator_types != ["parallel"] or
            len(body.yield_values) != 2 or
            body.yield_values[0] != body.yield_values[1]):
        return False
    maps = [m.replace(" ", "") for m in body.indexing_maps]
    if not (maps[0].endswith("->(d0)>") and
            maps[1].endswith("->()>") and
            maps[2].endswith("->(d0)>")):
        return False
    text = "\n".join(body.body_lines)
    add = re.search(
        rf"(%[\w.$-]+)\s*=\s*arith\.addf\s+"
        rf"{re.escape(body.outs_arg_names[0])}\s*,\s*"
        rf"{re.escape(body.ins_arg_names[0])}\s*:\s*f32", text)
    if not add:
        add = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addf\s+"
            rf"{re.escape(body.ins_arg_names[0])}\s*,\s*"
            rf"{re.escape(body.outs_arg_names[0])}\s*:\s*f32", text)
    return bool(add and body.yield_values[0] == add.group(1))


def _is_segmented_inclusive_product2d_f32(body, body_form: str) -> bool:
    if (body_form != "tensor" or len(body.ins_arg_names) != 1 or
            len(body.outs_arg_names) != 2 or len(body.indexing_maps) != 3 or
            body.iterator_types != ["parallel", "reduction"] or
            len(body.yield_values) != 2 or
            body.yield_values[0] != body.yield_values[1]):
        return False
    maps = [m.replace(" ", "") for m in body.indexing_maps]
    if not (maps[0].endswith("->(d0,d1)>") and
            maps[1].endswith("->(d0,d1)>") and
            maps[2].endswith("->(d0)>") ):
        return False
    text = "\n".join(body.body_lines)
    mul = re.search(
        rf"(%[\w.$-]+)\s*=\s*arith\.mulf\s+"
        rf"(?:{re.escape(body.outs_arg_names[1])}\s*,\s*"
        rf"{re.escape(body.ins_arg_names[0])}|"
        rf"{re.escape(body.ins_arg_names[0])}\s*,\s*"
        rf"{re.escape(body.outs_arg_names[1])})\s*:\s*f32", text)
    return bool(mul and body.yield_values[0] == mul.group(1))


def _cub_predicate_reduction_kind(body, term, body_form: str) -> str | None:
    """Classify predicate reductions directly consumable by CUB iterators."""
    if body_form != "tensor" or term is None or len(body.outs_arg_names) != 1:
        return None
    ast = _parse_term(_term_repr(term))
    zero = ("Lit", 0.0)
    nonzero = ("Cmp", "une", ("In", 0), zero)
    if ast == ("Add", ("Out", 0), nonzero):
        if (len(body.ins_arg_names) == 1 and
                body.iterator_types == ["reduction"]):
            return "count_nonzero_1d"
        if (len(body.ins_arg_names) == 1 and
                body.iterator_types == ["parallel", "reduction"]):
            return "count_nonzero_2d"
    equal = ("Cmp", "oeq", ("In", 0), ("In", 1))
    if (len(body.ins_arg_names) == 2 and
            body.iterator_types == ["reduction"] and
            ast == ("Select", ("Cmp", "ne", ("Out", 0), zero),
                    equal, zero)):
        return "equal_all_1d"
    return None


def _cub_dynamic_segmented_logical_flag(
    body, term, body_form: str,
) -> str | None:
    if (body_form != "tensor" or term is None or
            len(body.ins_arg_names) != 2 or len(body.outs_arg_names) != 1 or
            body.iterator_types != ["parallel", "reduction"]):
        return None
    zero, one, out = ("Lit", 0.0), ("Lit", 1.0), ("Out", 0)
    truth_out = ("Cmp", "ne", out, zero)
    all_expr = ("Select", truth_out,
                ("Cmp", "ne", ("In", 0), zero), zero)
    any_expr = ("Select", truth_out, one,
                ("Cmp", "ne", ("In", 1), zero))
    ast = _parse_term(_term_repr(term))
    if (isinstance(ast, tuple) and len(ast) == 4 and
            ast[0] == "Select" and ast[2] == all_expr and ast[3] == any_expr and
            isinstance(ast[1], tuple) and ast[1][0] == "Cap"):
        return ast[1][1]
    return None


def _scan_simple_memref_types(text: str, position: int | None = None) -> dict[str, str]:
    """Collect ordinary ranked memref types used by structured extraction."""
    if position is not None:
        function_start = text.rfind("func.func", 0, position)
        if function_start >= 0:
            text = text[function_start:position]
    result: dict[str, str] = {}
    for match in re.finditer(
            r"(%[\w.$-]+)\s*:\s*(memref<[^,()\n]+>)", text):
        result.setdefault(match.group(1), match.group(2))
    # Also learn result types from definitions such as
    # `%rowptr = memref.get_global @rowstr : memref<101xi32>`; fixed-size NPB
    # arrays appear this way rather than as function arguments.
    for match in re.finditer(
            r"(?m)^\s*(%[\w.$-]+)\s*=\s*[^\n]*:\s*"
            r"(memref<[^,()\n]+>)\s*$", text):
        result.setdefault(match.group(1), match.group(2))
    return result


def _render_looped_gemv_as_gemm(structured, text: str) -> tuple[int, int, str] | None:
    """Materialize the conservative loop-of-column-GEMV case as one GEMM."""
    if (structured.extracted_kind != "looped_gemv_as_gemm" or
            len(structured.region.operations) != 1):
        return None
    op = structured.region.operations[0]
    if len(op.loops) != 1 or len(op.input_roots) != 2 or len(op.output_roots) != 1:
        return None
    loop = op.loops[0]
    if not loop.bounds.startswith("0 to "):
        return None
    a, b = op.input_roots
    c = op.output_roots[0]
    types = _scan_simple_memref_types(text, loop.span[0])
    operand_types = [types.get(value) for value in (a, b, c)]
    if (any(value is None for value in operand_types) or
            any(_shaped_rank(value) != 2 for value in operand_types) or
            any(_sniff_elem_type(value) != "f64" for value in operand_types)):
        return None
    function_start = text.rfind("func.func", 0, loop.span[0])
    signature = text[function_start:loop.span[0]]
    if any(not re.search(
            rf"{re.escape(value)}\s*:\s*memref<[^>\n]+>\s*"
            rf"\{{\s*llvm\.noalias\s*\}}", signature)
           for value in (a, b, c)):
        return None

    # A is invariant. B and C must be column slices indexed by exactly the
    # parent loop IV: submap (d0)[s0] -> (d0,s0). This is the affine
    # composition that turns x[k] and y[i] into B[k,j] and C[i,j].
    normalized = [value.replace(" ", "") for value in op.accesses]
    column_map = "submap=affine_map<(d0)[s0]->(d0,s0)>"
    iv = loop.induction
    if ("submap=direct" not in normalized[0] or
            column_map not in normalized[1] or
            column_map not in normalized[2] or
            f"symbols={iv}" not in normalized[1] or
            f"symbols={iv}" not in normalized[2]):
        return None

    tensor_types = [_memref_to_tensor_type(value) for value in operand_types]
    if any(value is None for value in tensor_types):
        return None
    line_start = text.rfind("\n", 0, loop.span[0]) + 1
    indent = text[line_start:loop.span[0]]
    suffix = str(op.index)
    at, bt, ct = (f"%structured_{name}_{suffix}" for name in ("a", "b", "c"))
    result = f"%structured_gemm_{suffix}"
    result_memref = f"%structured_gemm_memref_{suffix}"
    replacement = (
        f"{at} = bufferization.to_tensor {a} restrict : {operand_types[0]}\n"
        f"{indent}{bt} = bufferization.to_tensor {b} restrict : {operand_types[1]}\n"
        f"{indent}{ct} = bufferization.to_tensor {c} restrict writable : {operand_types[2]}\n"
        f"{indent}{result} = kernel.launch @cublasDgemm_simple({at}, {bt}, {ct}) "
        f": ({tensor_types[0]}, {tensor_types[1]}, {tensor_types[2]}) "
        f"-> {tensor_types[2]}\n"
        f"{indent}{result_memref} = bufferization.to_memref {result} "
        f": {operand_types[2]}\n"
        f"{indent}memref.copy {result_memref}, {c} : "
        f"{operand_types[2]} to {operand_types[2]}")
    return loop.span[0], loop.span[1], replacement


def _render_looped_blas_as_strided_batched(
    structured, text: str
) -> tuple[int, int, str] | None:
    """Collapse a leading-dimension slice loop into one batched cuBLAS call."""
    specifications = {
        "looped_gemm_as_batched_gemm": (
            "cublasDgemm_strided_batched_subtract", (3, 3, 3)),
        "looped_gemv_as_batched_gemv": (
            "cublasDgemv_strided_batched_subtract", (3, 2, 2)),
    }
    spec = specifications.get(structured.extracted_kind)
    if spec is None or len(structured.region.operations) != 1:
        return None
    symbol, ranks = spec
    op = structured.region.operations[0]
    if len(op.loops) != 1 or len(op.input_roots) != 2 or len(op.output_roots) != 1:
        return None
    loop = op.loops[0]
    if not loop.bounds.startswith("0 to "):
        return None
    operands = [*op.input_roots, op.output_roots[0]]
    types = _scan_simple_memref_types(text, loop.span[0])
    operand_types = [types.get(value) for value in operands]
    if (any(value is None for value in operand_types) or
            tuple(_shaped_rank(value) for value in operand_types) != ranks or
            any(_sniff_elem_type(value) != "f64" for value in operand_types)):
        return None
    function_start = text.rfind("func.func", 0, loop.span[0])
    signature = text[function_start:loop.span[0]]
    if any(not re.search(
            rf"{re.escape(value)}\s*:\s*memref<[^>\n]+>\s*"
            rf"\{{\s*llvm\.noalias\s*\}}", signature)
           for value in operands):
        return None
    tensor_types = [_memref_to_tensor_type(value) for value in operand_types]
    if any(value is None for value in tensor_types):
        return None

    line_start = text.rfind("\n", 0, loop.span[0]) + 1
    indent = text[line_start:loop.span[0]]
    suffix = str(op.index)
    tensor_values = [
        f"%structured_batched_{label}_{suffix}"
        for label in ("a", "b", "c")
    ]
    result = f"%structured_batched_result_{suffix}"
    result_memref = f"%structured_batched_result_memref_{suffix}"
    lines = []
    for index, (tensor_value, operand, operand_type) in enumerate(zip(
            tensor_values, operands, operand_types)):
        writable = " writable" if index == 2 else ""
        lines.append(
            f"{tensor_value} = bufferization.to_tensor {operand} restrict"
            f"{writable} : {operand_type}")
    lines.append(
        f"{result} = kernel.launch @{symbol}({', '.join(tensor_values)}) : "
        f"({', '.join(tensor_types)}) -> {tensor_types[2]}")
    lines.append(
        f"{result_memref} = bufferization.to_memref {result} : "
        f"{operand_types[2]}")
    lines.append(
        f"memref.copy {result_memref}, {operands[2]} : "
        f"{operand_types[2]} to {operand_types[2]}")
    replacement = ("\n" + indent).join(lines)
    return loop.span[0], loop.span[1], replacement


def _render_source_faithful_sgemm(structured, text: str) -> tuple[int, int, str] | None:
    """Collapse Parboil's two loops + dot generic + alpha/beta epilogue."""
    if len(structured.region.operations) != 1:
        return None
    op = structured.region.operations[0]
    if (len(op.loops) != 2 or op.reduction_count != 1 or
            len(op.inputs) != 2 or len(op.outputs) != 1):
        return None
    canonical_memref = Term.In(2) + Term.In(0) * Term.In(1)
    canonical_tensor = Term.Out(0) + Term.In(0) * Term.In(1)
    if not (equivalent(op.term, canonical_memref,
                       include_distributivity=False) or
            equivalent(op.term, canonical_tensor,
                       include_distributivity=False)):
        return None
    outer, inner = op.loops
    outer_bound = re.fullmatch(r"0 to (%[\w.$-]+)", outer.bounds)
    inner_bound = re.fullmatch(r"0 to (%[\w.$-]+)", inner.bounds)
    if not outer_bound or not inner_bound:
        return None

    def symbols(access: str) -> list[str]:
        match = re.search(r"(?:^|;)symbols=([^;]*)(?:;|$)", access)
        return ([item for item in match.group(1).split(",") if item]
                if match else [])

    a_symbols, b_symbols = symbols(op.accesses[0]), symbols(op.accesses[1])
    if (len(a_symbols) != 3 or len(b_symbols) != 3 or
            a_symbols[0] != outer.induction or
            b_symbols[0] != inner.induction or
            a_symbols[2] != b_symbols[2]):
        return None
    lda, k = a_symbols[1], a_symbols[2]
    ldb = b_symbols[1]
    after = text[op.span[1]:inner.span[1]]
    temp = re.escape(op.output_roots[0])
    inner_iv = re.escape(inner.induction)
    dot_match = re.search(
        rf"(%[\w.$-]+)\s*=\s*affine\.load\s+{temp}\[{inner_iv}\]", after)
    if not dot_match:
        return None
    dot = dot_match.group(1)
    c_load = re.search(
        r"(%[\w.$-]+)\s*=\s*affine\.load\s+(%[\w.$-]+)\[([^]]+)\]",
        after[dot_match.end():])
    if not c_load:
        return None
    c_value, c_root, c_address = c_load.groups()
    muls = {
        result: (lhs, rhs) for result, lhs, rhs in re.findall(
            r"(%[\w.$-]+)\s*=\s*arith\.mulf\s+"
            r"(%[\w.$-]+),\s*(%[\w.$-]+)", after)
    }
    c_scaled = next((result for result, operands in muls.items()
                     if c_value in operands), None)
    dot_scaled = next((result for result, operands in muls.items()
                       if dot in operands), None)
    if not c_scaled or not dot_scaled:
        return None
    beta = next(value for value in muls[c_scaled] if value != c_value)
    alpha = next(value for value in muls[dot_scaled] if value != dot)
    sum_match = re.search(
        rf"(%[\w.$-]+)\s*=\s*arith\.addf\s+"
        rf"(?:{re.escape(c_scaled)},\s*{re.escape(dot_scaled)}|"
        rf"{re.escape(dot_scaled)},\s*{re.escape(c_scaled)})", after)
    if not sum_match:
        return None
    total = sum_match.group(1)
    store = re.search(
        rf"affine\.store\s+{re.escape(total)},\s*{re.escape(c_root)}"
        rf"\[([^]]+)\]", after)
    if not store or " ".join(store.group(1).split()) != " ".join(c_address.split()):
        return None
    types = _scan_simple_memref_types(text, outer.span[0])
    if any(types.get(root) != "memref<?xf32>"
           for root in (op.input_roots[0], op.input_roots[1], c_root)):
        return None
    scalar_types = _scan_scalar_types(text)
    if scalar_types.get(alpha) != "f32" or scalar_types.get(beta) != "f32":
        return None
    ldc_symbol = re.search(r"symbol\((%[\w.$-]+)\)", c_address)
    if not ldc_symbol:
        return None
    m, n = outer_bound.group(1), inner_bound.group(1)
    indent_start = text.rfind("\n", 0, outer.span[0]) + 1
    indent = text[indent_start:outer.span[0]]
    operands = [op.input_roots[0], op.input_roots[1], c_root,
                m, n, k, lda, ldb, ldc_symbol.group(1), beta, alpha]
    operand_types = [types[op.input_roots[0]], types[op.input_roots[1]],
                     types[c_root], "index", "index", "index", "index",
                     "index", "index", "f32", "f32"]
    replacement = (
        "kernel.launch @cublasSgemm_flat_colmajor_nt_alpha_beta("
        + ", ".join(operands) + ") : (" + ", ".join(operand_types)
        + ") -> ()")
    return outer.span[0], outer.span[1], indent + replacement


def _render_cusparse_csr_spmv(
        text: str) -> list[tuple[int, int, str, str]]:
    """Replace proven CSR reductions with NVIDIA cuSPARSE SpMV calls.

    The executable subset is intentionally the native cuSPARSE CSR ABI: i32
    row offsets/column indices, f32 or f64 values, beta=0, and one dense output
    element per enclosing row. JDS stays analysis-only because cuSPARSE has no
    JDS matrix descriptor.
    """
    loops = parse_loops(text)
    rendered: list[tuple[int, int, str, str]] = []
    seen: set[tuple[int, int]] = set()
    for candidate in analyze_residual_loops(text):
        if candidate.kind != "csr_spmv":
            continue
        parents = [loop for loop in loops
                   if loop.span[0] < candidate.loop.span[0]
                   and candidate.loop.span[1] < loop.span[1]]
        if not parents:
            continue
        row_loop = max(parents, key=lambda loop: loop.span[0])
        if row_loop.span in seen:
            continue

        # Joint multi-root debufferization keeps the ATen CSR kernel as a
        # tensor loop nest: rowPtr/column/value/x reads are tensor.extract,
        # the scalar reduction is an scf.for iter_arg, and every row is
        # inserted into a loop-carried output tensor.  Prove that complete
        # form here and recover the original memrefs for the cuSPARSE ABI.
        row_body = text[row_loop.span[0]:row_loop.span[1]]
        if "tensor.extract" in row_body:
            line_start = text.rfind("\n", 0, row_loop.span[0]) + 1
            assignment = re.match(
                r"\s*(%[\w.$-]+)\s*=\s*$",
                text[line_start:row_loop.span[0]])
            row_header = re.match(
                rf"affine\.for\s+{re.escape(row_loop.induction)}\s*=\s*"
                r"0\s+to\s+(\d+)\s+iter_args\s*\(\s*"
                r"(%[\w.$-]+)\s*=\s*(%[\w.$-]+)\s*\)\s*->\s*"
                r"\(tensor<[^>]+>\)\s*\{",
                row_body)
            if assignment is None or row_header is None:
                continue
            row_result = assignment.group(1)
            row_count = int(row_header.group(1))
            output_iter, output_init = row_header.group(2), row_header.group(3)
            if row_count <= 0:
                continue

            before_inner = text[row_loop.span[0]:candidate.loop.span[0]]
            inner = text[candidate.loop.span[0]:candidate.loop.span[1]]
            row_iv = re.escape(row_loop.induction)
            rowptr_loads = re.findall(
                r"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
                r"(%[\w.$-]+)\[([^]]+)\]\s*:\s*tensor<[^>]*xi32>",
                before_inner)
            if len(rowptr_loads) != 2 or rowptr_loads[0][1] != rowptr_loads[1][1]:
                continue
            direct_index = next((index for index, load in enumerate(rowptr_loads)
                                 if load[2].strip() == row_loop.induction), None)
            if direct_index is None:
                continue
            direct = rowptr_loads[direct_index]
            successor = rowptr_loads[1 - direct_index]
            successor_apply = re.search(
                rf"{re.escape(successor[2].strip())}\s*=\s*affine\.apply\s+"
                rf"([^\n]+)\(\s*{row_iv}\s*\)", before_inner)
            if successor_apply is None:
                continue
            successor_map = _compact_affine_map(_resolve_affine_map_text(
                text[:candidate.loop.span[0]], successor_apply.group(1).strip()))
            if successor_map != "affine_map<(d0)->(d0+1)>":
                continue

            inner_assignment_start = text.rfind(
                "\n", row_loop.span[0], candidate.loop.span[0]) + 1
            inner_assignment = re.match(
                r"\s*(%[\w.$-]+)\s*=\s*$",
                text[inner_assignment_start:candidate.loop.span[0]])
            inner_header = re.match(
                rf"scf\.for\s+{re.escape(candidate.loop.induction)}\s*=\s*"
                rf"%[\w.$-]+\s+to\s+%[\w.$-]+\s+step\s+%[\w.$-]+\s+"
                r"iter_args\s*\(\s*(%[\w.$-]+)\s*=\s*"
                r"(%[\w.$-]+)\s*\)\s*->\s*\(f32\)\s*\{",
                inner)
            if inner_assignment is None or inner_header is None:
                continue
            inner_result = inner_assignment.group(1)
            accumulator, zero = inner_header.group(1), inner_header.group(2)
            if not re.search(
                    rf"{re.escape(zero)}\s*=\s*arith\.constant\s+"
                    r"(?:0(?:\.0+)?(?:e[+-]?0+)?|0x0+)\s*:\s*f32",
                    text[:candidate.loop.span[0]], re.IGNORECASE):
                continue

            inner_iv = re.escape(candidate.loop.induction)
            values = re.search(
                rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
                rf"(%[\w.$-]+)\[{inner_iv}\]\s*:\s*tensor<[^>]*xf32>",
                inner)
            cols = re.search(
                rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
                rf"(%[\w.$-]+)\[{inner_iv}\]\s*:\s*tensor<[^>]*xi32>",
                inner)
            if values is None or cols is None:
                continue
            column_index = cols.group(1)
            cast = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
                rf"{re.escape(column_index)}\s*:\s*i32\s+to\s+index", inner)
            if cast is None:
                continue
            gather = re.search(
                rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
                rf"(%[\w.$-]+)\[{re.escape(cast.group(1))}\]\s*:\s*"
                r"tensor<[^>]*xf32>", inner)
            if gather is None:
                continue
            product = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.mulf\s+"
                rf"(?:{re.escape(values.group(1))},\s*{re.escape(gather.group(1))}|"
                rf"{re.escape(gather.group(1))},\s*{re.escape(values.group(1))})"
                r"\s*:\s*f32", inner)
            if product is None:
                continue
            total = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.addf\s+"
                rf"(?:{re.escape(accumulator)},\s*{re.escape(product.group(1))}|"
                rf"{re.escape(product.group(1))},\s*{re.escape(accumulator)})"
                r"\s*:\s*f32", inner)
            if total is None or re.search(
                    rf"scf\.yield\s+{re.escape(total.group(1))}\s*:\s*f32",
                    inner) is None:
                continue
            inserted = re.search(
                rf"(%[\w.$-]+)\s*=\s*tensor\.insert\s+"
                rf"{re.escape(inner_result)}\s+into\s+"
                rf"{re.escape(output_iter)}\[{row_iv}\]", row_body)
            if inserted is None or re.search(
                    rf"affine\.yield\s+{re.escape(inserted.group(1))}\s*:",
                    row_body) is None:
                continue

            tensor_values = [rowptr_loads[0][1], cols.group(2),
                             values.group(2), gather.group(2), output_init]
            sources = [_to_tensor_memref_source(text, value, row_loop.span[0])
                       for value in tensor_values]
            if any(source is None for source in sources):
                continue
            physical_operands = [source[0] for source in sources]
            physical_types = [source[1] for source in sources]
            if (len(set(physical_operands)) != 5 or
                    [_sniff_elem_type(ty) for ty in physical_types] !=
                    ["i32", "i32", "f32", "f32", "f32"] or
                    [_shaped_rank(ty) for ty in physical_types] != [1] * 5):
                continue

            tail = re.match(
                rf"\s*(%[\w.$-]+)\s*=\s*bufferization\.to_memref\s+"
                rf"{re.escape(row_result)}\s*:\s*memref<[^>]*xf32>\s*\n"
                rf"\s*memref\.copy\s+\1,\s*"
                rf"{re.escape(physical_operands[4])}\s*:\s*[^\n]*",
                text[row_loop.span[1]:])
            if tail is None:
                continue

            uid = row_loop.span[0]
            rows = f"%cusparse_rows_{uid}"
            target_types = ["memref<?xi32>", "memref<?xi32>",
                            "memref<?xf32>", "memref<?xf32>",
                            "memref<?xf32>"]
            indent = text[line_start:line_start + len(
                text[line_start:row_loop.span[0]])]
            indent = re.match(r"\s*", indent).group(0)
            lines = [f"{rows} = arith.constant {row_count} : index"]
            normalized = []
            for index, (operand, source_type, target_type) in enumerate(zip(
                    physical_operands, physical_types, target_types)):
                if source_type != target_type:
                    cast_name = f"%cusparse_arg_{uid}_{index}"
                    lines.append(
                        f"{cast_name} = memref.cast {operand} : "
                        f"{source_type} to {target_type}")
                    normalized.append(cast_name)
                else:
                    normalized.append(operand)
            lines.append(
                "kernel.launch @cusparseSpMV_CSR_f32_memref("
                + ", ".join([rows, *normalized]) + ") : (index, "
                + ", ".join(target_types) + ") -> ()")
            replacement = ("\n" + indent).join(lines)
            rendered.append((line_start, row_loop.span[1] + tail.end(),
                             indent + replacement,
                             "cusparseSpMV_CSR_f32_memref"))
            seen.add(row_loop.span)
            continue

        row_bound = re.fullmatch(
            r"(?:0|%c0(?:_[\w.$-]+)?) to (%[\w.$-]+)"
            r"(?: step %c1(?:_[\w.$-]+)?)?", row_loop.bounds)
        prefix_lines: list[str] = []
        if row_bound:
            rows = row_bound.group(1)
        else:
            mapped = re.fullmatch(
                r"0 to (#[\w.$-]+\(\)\[[^]]+\])", row_loop.bounds)
            if not mapped:
                continue
            rows = f"%cusparse_rows_{row_loop.span[0]}"
            prefix_lines.append(f"{rows} = affine.apply {mapped.group(1)}")

        body = text[row_loop.span[0]:row_loop.span[1]]
        before_inner = text[row_loop.span[0]:candidate.loop.span[0]]
        inner = text[candidate.loop.span[0]:candidate.loop.span[1]]
        iv = re.escape(candidate.loop.induction)
        rowptr_loads = re.findall(
            r"(?:affine|memref)\.load\s+(%[\w.$-]+)\[[^]]+\]\s*:\s*"
            r"(memref<[^>]*xi32>)", before_inner)
        values = re.search(
            rf"memref\.load\s+(%[\w.$-]+)\[{iv}\]\s*:\s*"
            r"(memref<[^>]*x(f32|f64)>)", inner)
        cols = re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+"
            rf"(%[\w.$-]+)\[{iv}\]\s*:\s*"
            r"(memref<[^>]*xi32>)", inner)
        gather = None
        if cols:
            index_value = cols.group(1)
            cast = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.(?:index_cast|ext[su]i|trunci)\s+"
                rf"{re.escape(index_value)}\b", inner)
            if cast:
                index_value = cast.group(1)
            gather = re.search(
                rf"(?:memref|affine)\.load\s+(%[\w.$-]+)"
                rf"\[{re.escape(index_value)}\]\s*:\s*"
                r"(memref<[^>]*x(f32|f64)>)", inner)
        row_iv = re.escape(row_loop.induction)
        store = re.search(
            r"(?:affine|memref)\.store\s+%[\w.$-]+,\s*"
            rf"(%[\w.$-]+)\[{row_iv}\]\s*:\s*"
            r"(memref<[^>]*x(f32|f64)>)",
            body)
        if (len(rowptr_loads) < 2 or
                rowptr_loads[0][0] != rowptr_loads[1][0] or
                not values or not cols or not gather or not store):
            continue
        elem = values.group(3)
        if gather.group(3) != elem or store.group(3) != elem:
            continue
        operands = [rows, rowptr_loads[0][0], cols.group(2), values.group(1),
                    gather.group(1), store.group(1)]
        target_types = ["memref<?xi32>", "memref<?xi32>",
                        f"memref<?x{elem}>", f"memref<?x{elem}>",
                        f"memref<?x{elem}>"]
        line_start = text.rfind("\n", 0, row_loop.span[0]) + 1
        indent = text[line_start:row_loop.span[0]]
        uid = row_loop.span[0]
        known_types = _scan_simple_memref_types(text, row_loop.span[0])
        normalized = [rows]
        for index, (operand, target) in enumerate(zip(operands[1:], target_types)):
            source = known_types.get(operand)
            if source is not None and source != target:
                cast = f"%cusparse_arg_{uid}_{index}"
                prefix_lines.append(
                    f"{cast} = memref.cast {operand} : {source} to {target}")
                normalized.append(cast)
            else:
                normalized.append(operand)
        symbol = f"cusparseSpMV_CSR_{elem}_memref"
        launch = (f"kernel.launch @{symbol}(" + ", ".join(normalized) +
                  ") : (index, " + ", ".join(target_types) + ") -> ()")
        replacement = "\n".join(
            f"{indent}{line}" for line in [*prefix_lines, launch])
        rendered.append((row_loop.span[0], row_loop.span[1], replacement,
                         symbol))
        seen.add(row_loop.span)
    return rendered


def _render_cusparse_bsr_spmv_buffer(
        text: str) -> list[tuple[int, int, str, str]]:
    """Recognize the buffer-form BSR matvec retained by large-shape raising."""
    loops = parse_loops(text)
    rendered: list[tuple[int, int, str, str]] = []
    for outer in loops:
        if outer.kind != "affine.for":
            continue
        nested = [loop for loop in loops
                  if outer.span[0] < loop.span[0] and
                  loop.span[1] < outer.span[1]]
        affine_loops = [loop for loop in nested if loop.kind == "affine.for"]
        reductions = [loop for loop in nested if loop.kind == "scf.for"]
        if len(affine_loops) != 2 or len(reductions) != 1:
            continue
        reduction = reductions[0]
        block_loop = next((loop for loop in affine_loops
                           if loop.span[0] < reduction.span[0] and
                           reduction.span[1] < loop.span[1]), None)
        scalar_loop = next((loop for loop in affine_loops
                            if reduction.span[0] < loop.span[0] and
                            loop.span[1] < reduction.span[1]), None)
        if block_loop is None or scalar_loop is None:
            continue
        outer_bound = re.fullmatch(r"0 to (\d+)", outer.bounds)
        block_bound = re.fullmatch(r"0 to (\d+)", block_loop.bounds)
        scalar_bound = re.fullmatch(r"0 to (\d+)", scalar_loop.bounds)
        if not outer_bound or not block_bound or not scalar_bound:
            continue
        block_rows = int(outer_bound.group(1))
        block_dim = int(block_bound.group(1))
        if block_rows <= 0 or block_dim <= 0 or int(scalar_bound.group(1)) != block_dim:
            continue

        outer_text = text[outer.span[0]:outer.span[1]]
        before_reduction = text[block_loop.span[0]:reduction.span[0]]
        row_iv = re.escape(outer.induction)
        rowptr = re.findall(
            rf"(%[\w.$-]+)\s*=\s*affine\.load\s+(%[\w.$-]+)"
            rf"\[({row_iv}(?:\s*\+\s*1)?)\]\s*:\s*(memref<[^>]*xi32>)",
            before_reduction)
        if len(rowptr) != 2 or rowptr[0][1] != rowptr[1][1]:
            continue
        direct = next((load for load in rowptr
                       if re.sub(r"\s+", "", load[2]) == outer.induction), None)
        successor = next((load for load in rowptr
                          if re.sub(r"\s+", "", load[2]) ==
                          outer.induction + "+1"), None)
        if direct is None or successor is None:
            continue
        casts: dict[str, str] = {}
        for loaded, _, _, _ in rowptr:
            cast = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
                rf"{re.escape(loaded)}\s*:\s*i32\s+to\s+index",
                before_reduction)
            if cast:
                casts[loaded] = cast.group(1)
        if len(casts) != 2:
            continue
        reduction_line = text.rfind("\n", block_loop.span[0], reduction.span[0]) + 1
        reduction_assignment = re.match(
            r"\s*(%[\w.$-]+)\s*=\s*$",
            text[reduction_line:reduction.span[0]])
        reduction_text = text[reduction.span[0]:reduction.span[1]]
        reduction_header = re.match(
            rf"scf\.for\s+{re.escape(reduction.induction)}\s*=\s*"
            rf"{re.escape(casts[direct[0]])}\s+to\s*"
            rf"{re.escape(casts[successor[0]])}\s+step\s+(%[\w.$-]+)\s+"
            r"iter_args\s*\(\s*(%[\w.$-]+)\s*=\s*(%[\w.$-]+)\s*\)\s*"
            r"->\s*\(f32\)\s*\{", reduction_text)
        if reduction_assignment is None or reduction_header is None:
            continue
        reduction_result = reduction_assignment.group(1)
        step, accumulator, zero = reduction_header.groups()
        prefix = text[:reduction.span[0]]
        if (re.search(rf"{re.escape(step)}\s*=\s*arith\.constant\s+1\s*:\s*index",
                      prefix) is None or
                re.search(rf"{re.escape(zero)}\s*=\s*arith\.constant\s+"
                          r"(?:0(?:\.0+)?(?:e[+-]?0+)?|0x0+)\s*:\s*f32",
                          prefix, re.IGNORECASE) is None):
            continue

        p = re.escape(reduction.induction)
        i = re.escape(block_loop.induction)
        j = re.escape(scalar_loop.induction)
        column = re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+(%[\w.$-]+)"
            rf"\[{p}\]\s*:\s*(memref<[^>]*xi32>)", reduction_text)
        column_base = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.muli\s+"
            rf"{re.escape(column.group(1))},\s*(%[\w.$-]+)\s*:\s*i32",
            reduction_text) if column else None)
        if column_base is None or re.search(
                rf"{re.escape(column_base.group(2))}\s*=\s*arith\.constant\s+"
                rf"{block_dim}\s*:\s*i32", prefix) is None:
            continue
        scalar_text = text[scalar_loop.span[0]:scalar_loop.span[1]]
        alloca = re.search(r"(%[\w.$-]+)\s*=\s*memref\.alloca\(\)\s*:\s*memref<f32>",
                           reduction_text)
        if alloca is None or re.search(
                rf"affine\.store\s+{re.escape(accumulator)},\s*"
                rf"{re.escape(alloca.group(1))}\[\]", reduction_text) is None:
            continue
        old = re.search(
            rf"(%[\w.$-]+)\s*=\s*affine\.load\s+"
            rf"{re.escape(alloca.group(1))}\[\]", scalar_text)
        j_i32 = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+{j}\s*:\s*"
            r"index\s+to\s+i32", scalar_text)
        value = re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+(%[\w.$-]+)"
            rf"\[{p},\s*{i},\s*{j}\]\s*:\s*(memref<[^>]*xf32>)",
            scalar_text)
        dense_offset = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addi\s+"
            rf"(?:{re.escape(column_base.group(1))},\s*{re.escape(j_i32.group(1))}|"
            rf"{re.escape(j_i32.group(1))},\s*{re.escape(column_base.group(1))})"
            r"\s*:\s*i32", scalar_text) if j_i32 else None)
        dense_cast = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
            rf"{re.escape(dense_offset.group(1))}\s*:\s*i32\s+to\s+index",
            scalar_text) if dense_offset else None)
        dense = (re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+(%[\w.$-]+)"
            rf"\[{re.escape(dense_cast.group(1))}\]\s*:\s*(memref<[^>]*xf32>)",
            scalar_text) if dense_cast else None)
        if old is None or value is None or dense is None or len(re.findall(
                r"(?:memref|affine)\.load\b", scalar_text)) != 3:
            continue
        product = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.mulf\s+"
            rf"(?:{re.escape(value.group(1))},\s*{re.escape(dense.group(1))}|"
            rf"{re.escape(dense.group(1))},\s*{re.escape(value.group(1))})"
            r"\s*:\s*f32", scalar_text)
        total = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addf\s+"
            rf"(?:{re.escape(old.group(1))},\s*{re.escape(product.group(1))}|"
            rf"{re.escape(product.group(1))},\s*{re.escape(old.group(1))})"
            r"\s*:\s*f32", scalar_text) if product else None)
        if (total is None or re.search(
                rf"affine\.store\s+{re.escape(total.group(1))},\s*"
                rf"{re.escape(alloca.group(1))}\[\]", scalar_text) is None):
            continue
        final_scalar = re.search(
            rf"(%[\w.$-]+)\s*=\s*affine\.load\s+"
            rf"{re.escape(alloca.group(1))}\[\]", reduction_text[scalar_loop.span[1] - reduction.span[0]:])
        if final_scalar is None or re.search(
                rf"scf\.yield\s+{re.escape(final_scalar.group(1))}\s*:\s*f32",
                reduction_text) is None:
            continue
        output = re.search(
            rf"affine\.store\s+{re.escape(reduction_result)},\s*(%[\w.$-]+)"
            rf"\[{i}\s*\+\s*{row_iv}\s*\*\s*{block_dim}\]\s*:\s*"
            r"(memref<[^>]*xf32>)", outer_text)
        if output is None:
            continue
        operands = [rowptr[0][1], column.group(2), value.group(2),
                    dense.group(2), output.group(1)]
        types = [rowptr[0][3], column.group(3), value.group(3),
                 dense.group(3), output.group(2)]
        if (len(set(operands)) != 5 or
                [_sniff_elem_type(ty) for ty in types] !=
                ["i32", "i32", "f32", "f32", "f32"] or
                [_shaped_rank(ty) for ty in types] != [1, 1, 3, 1, 1] or
                re.fullmatch(rf"memref<\?x{block_dim}x{block_dim}xf32>",
                             types[2]) is None):
            continue
        line_start = text.rfind("\n", 0, outer.span[0]) + 1
        indent = re.match(r"\s*", text[line_start:outer.span[0]]).group(0)
        uid = outer.span[0]
        rows_value = f"%cusparse_bsr_rows_{uid}"
        dim_value = f"%cusparse_bsr_dim_{uid}"
        target_types = ["memref<?xi32>", "memref<?xi32>",
                        "memref<?x?x?xf32>", "memref<?xf32>",
                        "memref<?xf32>"]
        lines = [f"{rows_value} = arith.constant {block_rows} : index",
                 f"{dim_value} = arith.constant {block_dim} : index"]
        normalized: list[str] = []
        for index, (operand, source_type, target_type) in enumerate(
                zip(operands, types, target_types)):
            if source_type != target_type:
                cast = f"%cusparse_bsr_arg_{uid}_{index}"
                lines.append(
                    f"{cast} = memref.cast {operand} : {source_type} to {target_type}")
                normalized.append(cast)
            else:
                normalized.append(operand)
        symbol = "cusparseSpMM_BSR_f32_memref"
        lines.append(
            f"kernel.launch @{symbol}(" +
            ", ".join([rows_value, dim_value, *normalized]) +
            ") : (index, index, " + ", ".join(target_types) + ") -> ()")
        replacement = ("\n" + indent).join(lines)
        rendered.append((line_start, outer.span[1], indent + replacement, symbol))
    return rendered


def _render_cusparse_csr_sddmm_buffer(
        text: str) -> list[tuple[int, int, str, str]]:
    """Recognize the memref form retained by large-shape SDDMM raising."""
    rendered: list[tuple[int, int, str, str]] = []
    function_re = re.compile(
        r"func\.func(?:\s+private)?\s+@([\w.$-]+)\s*\(([^)]*)\)",
        re.MULTILINE)
    for function in function_re.finditer(text):
        arguments = re.findall(
            r"(%[\w.$-]+)\s*:\s*(memref<[^>]+>|f32)", function.group(2))
        if len(arguments) != 8:
            continue
        names = [name for name, _ in arguments]
        types = [ty for _, ty in arguments]
        dense_a = re.fullmatch(r"memref<\?x(\d+)xf32>", types[3])
        dense_b = re.fullmatch(r"memref<\?x(\d+)xf32>", types[4])
        if (types[:3] != ["memref<?xi32>", "memref<?xi32>",
                          "memref<?xf32>"] or
                types[5:7] != ["f32", "f32"] or
                types[7] != "memref<?xf32>" or dense_a is None or
                dense_b is None or len(set(names[:5] + [names[7]])) != 6):
            continue
        signature_end = text.find("\n", function.end())
        opening = text.rfind("{", function.end(), signature_end)
        function_end = _matching_brace(text, opening) if opening >= 0 else None
        if function_end is None:
            continue
        loops = [loop for loop in parse_loops(text)
                 if opening < loop.span[0] and loop.span[1] < function_end]
        if len(loops) != 1 or loops[0].kind != "affine.for":
            continue
        row_loop = loops[0]
        row_text = text[row_loop.span[0]:row_loop.span[1]]
        row_header = re.match(
            rf"affine\.for\s+{re.escape(row_loop.induction)}\s*=\s*0\s+to\s+"
            r"(\d+)\s*[{]", row_text)
        if (row_header is None or
                len(re.findall(r"\bscf\.while\b", row_text)) != 1 or
                len(re.findall(r"\blinalg\.generic\b", row_text)) != 1):
            continue
        row_count = int(row_header.group(1))
        row_iv = re.escape(row_loop.induction)
        direct = re.search(
            rf"(%[\w.$-]+)\s*=\s*affine\.load\s+{re.escape(names[0])}"
            rf"\[{row_iv}\]\s*:\s*memref<\?xi32>", row_text)
        successor = re.search(
            rf"(%[\w.$-]+)\s*=\s*affine\.load\s+{re.escape(names[0])}"
            rf"\[{row_iv}\s*\+\s*1\]\s*:\s*memref<\?xi32>", row_text)
        if direct is None or successor is None:
            continue
        while_match = re.search(
            rf"(%[\w.$-]+)\s*=\s*scf\.while\s*\(\s*"
            rf"(%[\w.$-]+)\s*=\s*{re.escape(direct.group(1))}\s*\)\s*:\s*"
            r"\(i32\)\s*->\s*i32\s*[{]", row_text)
        if while_match is None:
            continue
        while_result, condition_p = while_match.groups()
        first_open = row_loop.span[0] + while_match.end() - 1
        first_end = _matching_brace(text, first_open)
        do_match = (re.match(r"\s*do\s*[{]", text[first_end:row_loop.span[1]])
                    if first_end is not None else None)
        if do_match is None:
            continue
        second_open = first_end + do_match.end() - 1
        while_end = _matching_brace(text, second_open)
        if while_end is None:
            continue
        condition_text = text[first_open + 1:first_end - 1]
        do_text = text[second_open + 1:while_end - 1]
        compare = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.cmpi\s+slt,\s*"
            rf"{re.escape(condition_p)},\s*{re.escape(successor.group(1))}",
            condition_text)
        if compare is None or re.search(
                rf"scf\.condition\s*\(\s*{re.escape(compare.group(1))}\s*\)"
                rf"\s*{re.escape(condition_p)}\s*:\s*i32",
                condition_text) is None:
            continue
        block = re.match(
            r"\s*\^bb0\(\s*(%[\w.$-]+)\s*:\s*i32\s*\):", do_text)
        if block is None:
            continue
        p = block.group(1)
        p_index = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+{re.escape(p)}\s*"
            r":\s*i32\s+to\s+index", do_text)
        if p_index is None:
            continue
        p_idx = p_index.group(1)
        col = re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+{re.escape(names[1])}"
            rf"\[{re.escape(p_idx)}\]\s*:\s*memref<\?xi32>", do_text)
        col_index = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
            rf"{re.escape(col.group(1))}\s*:\s*i32\s+to\s+index", do_text)
                     if col else None)
        alloca = re.search(
            r"(%[\w.$-]+)\s*=\s*memref\.alloca\(\)\s*:\s*memref<f32>",
            do_text)
        zero_store = (re.search(
            rf"affine\.store\s+(%[\w.$-]+),\s*{re.escape(alloca.group(1))}"
            r"\[\]\s*:\s*memref<f32>", do_text) if alloca else None)
        if col_index is None or zero_store is None or re.search(
                rf"{re.escape(zero_store.group(1))}\s*=\s*arith\.constant\s+"
                r"(?:0(?:\.0+)?(?:e[+-]?0+)?|0x0+)\s*:\s*f32",
                text[:row_loop.span[0]], re.IGNORECASE) is None:
            continue
        generic_match = re.search(r"linalg\.generic\b", do_text)
        generic_outs = do_text.find(" outs(", generic_match.start())
        generic_open = do_text.find("{", generic_outs)
        generic_end = _matching_brace(do_text, generic_open)
        generic_text = (do_text[generic_match.start():generic_end]
                        if generic_end is not None else "")
        if (not re.search(r'iterator_types\s*=\s*\["reduction"\]', generic_text)
                or len(re.findall(r"\bmemref\.load\b", generic_text)) != 2):
            continue
        k_match = re.search(
            r"(%[\w.$-]+)\s*=\s*linalg\.index\s+0\s*:\s*index",
            generic_text)
        block_arg = re.search(r"\^bb0\(\s*(%[\w.$-]+)\s*:\s*f32\s*\):",
                              generic_text)
        if k_match is None or block_arg is None:
            continue
        k = k_match.group(1)
        a_load = re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+{re.escape(names[3])}"
            rf"\[{row_iv},\s*{re.escape(k)}\]\s*:\s*{re.escape(types[3])}",
            generic_text)
        b_load = re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+{re.escape(names[4])}"
            rf"\[{re.escape(k)},\s*{re.escape(col_index.group(1))}\]\s*:\s*"
            rf"{re.escape(types[4])}", generic_text)
        if a_load is None or b_load is None:
            continue
        product = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.mulf\s+"
            rf"(?:{re.escape(a_load.group(1))},\s*{re.escape(b_load.group(1))}|"
            rf"{re.escape(b_load.group(1))},\s*{re.escape(a_load.group(1))})"
            r"\s*:\s*f32", generic_text)
        total = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addf\s+"
            rf"(?:{re.escape(block_arg.group(1))},\s*{re.escape(product.group(1))}|"
            rf"{re.escape(product.group(1))},\s*{re.escape(block_arg.group(1))})"
            r"\s*:\s*f32", generic_text) if product else None)
        if total is None or re.search(
                rf"linalg\.yield\s+{re.escape(total.group(1))}\s*:\s*f32",
                generic_text) is None:
            continue
        dot = re.search(
            rf"(%[\w.$-]+)\s*=\s*affine\.load\s+{re.escape(alloca.group(1))}"
            r"\[\]\s*:\s*memref<f32>", do_text[generic_end:])
        self_value = re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+{re.escape(names[2])}"
            rf"\[{re.escape(p_idx)}\]\s*:\s*memref<\?xf32>", do_text)
        if dot is None or self_value is None:
            continue
        beta_product = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.mulf\s+"
            rf"{re.escape(names[6])},\s*{re.escape(self_value.group(1))}\s*:\s*f32",
            do_text)
        alpha_product = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.mulf\s+"
            rf"{re.escape(names[5])},\s*{re.escape(dot.group(1))}\s*:\s*f32",
            do_text)
        summed = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addf\s+"
            rf"{re.escape(beta_product.group(1))},\s*"
            rf"{re.escape(alpha_product.group(1))}\s*:\s*f32", do_text)
                  if beta_product and alpha_product else None)
        if summed is None or re.search(
                rf"memref\.store\s+{re.escape(summed.group(1))},\s*"
                rf"{re.escape(names[7])}\[{re.escape(p_idx)}\]",
                do_text) is None:
            continue
        increment = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addi\s+{re.escape(p)},\s*"
            r"(%[\w.$-]+)\s*:\s*i32", do_text)
        if increment is None or re.search(
                rf"{re.escape(increment.group(2))}\s*=\s*arith\.constant\s+1"
                r"\s*:\s*i32", text[:row_loop.span[0]]) is None or re.search(
                rf"scf\.yield\s+{re.escape(increment.group(1))}\s*:\s*i32",
                do_text) is None:
            continue
        line_start = text.rfind("\n", 0, row_loop.span[0]) + 1
        indent = re.match(r"\s*", text[line_start:row_loop.span[0]]).group(0)
        uid = row_loop.span[0]
        rows_value = f"%cusparse_sddmm_rows_{uid}"
        a_cast = f"%cusparse_sddmm_a_{uid}"
        b_cast = f"%cusparse_sddmm_b_{uid}"
        lines = [
            f"{rows_value} = arith.constant {row_count} : index",
            f"{a_cast} = memref.cast {names[3]} : {types[3]} to memref<?x?xf32>",
            f"{b_cast} = memref.cast {names[4]} : {types[4]} to memref<?x?xf32>",
            "kernel.launch @cusparseSDDMM_CSR_f32_memref(" +
            ", ".join([rows_value, names[0], names[1], names[2], a_cast,
                       b_cast, names[5], names[6], names[7]]) +
            ") : (index, memref<?xi32>, memref<?xi32>, memref<?xf32>, "
            "memref<?x?xf32>, memref<?x?xf32>, f32, f32, memref<?xf32>) -> ()"
        ]
        replacement = ("\n" + indent).join(lines)
        rendered.append((line_start, row_loop.span[1], indent + replacement,
                         "cusparseSDDMM_CSR_f32_memref"))
    return rendered


def _render_cusparse_csr_sddmm(
        text: str) -> list[tuple[int, int, str, str]]:
    """Recognize complete tensorized CSR sampled-addmm computations.

    This is the cuSPARSE SDDMM contract
      out[p] = beta * self[p] + alpha * sum_k A[row, k] * B[k, col[p]].
    The rewrite consumes the CSR row traversal, reduction, scalar epilogue,
    loop-carried output, and final copy as one operation.  Fixed inner matrix
    extents prove row-major dense layouts and the shared reduction dimension.
    """
    rendered = _render_cusparse_csr_sddmm_buffer(text)
    function_re = re.compile(
        r"func\.func(?:\s+private)?\s+@([\w.$-]+)\s*\(([^)]*)\)",
        re.MULTILINE)
    for function in function_re.finditer(text):
        arguments = re.findall(
            r"(%[\w.$-]+)\s*:\s*(memref<[^>]+>|f32)",
            function.group(2))
        if len(arguments) != 8:
            continue
        names = [name for name, _ in arguments]
        types = [ty for _, ty in arguments]
        dense_a = re.fullmatch(r"memref<\?x(\d+)xf32>", types[3])
        dense_b = re.fullmatch(r"memref<\?x(\d+)xf32>", types[4])
        if (types[:3] != ["memref<?xi32>", "memref<?xi32>",
                          "memref<?xf32>"] or
                types[5:7] != ["f32", "f32"] or
                types[7] != "memref<?xf32>" or
                dense_a is None or dense_b is None or
                len(set(names[:5] + [names[7]])) != 6):
            continue
        reduction_size = int(dense_a.group(1))
        column_count = int(dense_b.group(1))
        if reduction_size <= 0 or column_count <= 0:
            continue

        signature_end = text.find("\n", function.end())
        opening = text.rfind("{", function.end(), signature_end)
        if opening < 0:
            continue
        function_end = _matching_brace(text, opening)
        if function_end is None:
            continue
        body = text[opening + 1:function_end - 1]
        body_offset = opening + 1
        if (len(re.findall(r"\blinalg\.generic\b", body)) != 1 or
                len(re.findall(r"\baffine\.for\b", body)) != 1 or
                len(re.findall(r"\bscf\.while\b", body)) != 1 or
                re.search(r"\bscf\.for\b", body)):
            continue

        row_tensor = re.search(
            rf"(%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+"
            rf"{re.escape(names[0])}\s*:\s*memref<\?xi32>", body)
        col_tensor = re.search(
            rf"(%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+"
            rf"{re.escape(names[1])}\s*:\s*memref<\?xi32>", body)
        self_tensor = re.search(
            rf"(%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+"
            rf"{re.escape(names[2])}\s*:\s*memref<\?xf32>", body)
        out_tensor = re.search(
            rf"(%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+"
            rf"{re.escape(names[7])}\s*:\s*memref<\?xf32>", body)
        if not all((row_tensor, col_tensor, self_tensor, out_tensor)):
            continue

        loops = [loop for loop in parse_loops(text)
                 if body_offset <= loop.span[0] and loop.span[1] < function_end]
        if len(loops) != 1 or loops[0].kind != "affine.for":
            continue
        row_loop = loops[0]
        row_line = text.rfind("\n", body_offset, row_loop.span[0]) + 1
        row_assignment = re.match(
            r"\s*(%[\w.$-]+)\s*=\s*$", text[row_line:row_loop.span[0]])
        row_text = text[row_loop.span[0]:row_loop.span[1]]
        row_header = re.match(
            rf"affine\.for\s+{re.escape(row_loop.induction)}\s*=\s*0\s+to\s+"
            r"(\d+)\s+iter_args\s*\(\s*(%[\w.$-]+)\s*=\s*"
            rf"{re.escape(out_tensor.group(1))}\s*\)\s*->\s*"
            r"\(tensor<\?xf32>\)\s*[{]", row_text)
        if row_assignment is None or row_header is None:
            continue
        row_result = row_assignment.group(1)
        row_count = int(row_header.group(1))
        row_output = row_header.group(2)
        if row_count <= 0:
            continue

        row_iv = re.escape(row_loop.induction)
        direct = re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            rf"{re.escape(row_tensor.group(1))}\[{row_iv}\]\s*:\s*"
            r"tensor<\?xi32>", row_text)
        successor_index = re.search(
            rf"(%[\w.$-]+)\s*=\s*affine\.apply\s+([^\n]+)"
            rf"\(\s*{row_iv}\s*\)", row_text)
        if direct is None or successor_index is None:
            continue
        successor_map = _compact_affine_map(_resolve_affine_map_text(
            text[:row_loop.span[1]], successor_index.group(2).strip()))
        successor = re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            rf"{re.escape(row_tensor.group(1))}"
            rf"\[{re.escape(successor_index.group(1))}\]\s*:\s*"
            r"tensor<\?xi32>", row_text)
        if successor_map != "affine_map<(d0)->(d0+1)>" or successor is None:
            continue

        while_start_match = re.search(
            rf"(%[\w.$-]+):2\s*=\s*scf\.while\s*\(\s*"
            rf"(%[\w.$-]+)\s*=\s*{re.escape(direct.group(1))}\s*,\s*"
            rf"(%[\w.$-]+)\s*=\s*{re.escape(row_output)}\s*\)\s*:\s*"
            r"\(i32,\s*tensor<\?xf32>\)\s*->\s*"
            r"\(i32,\s*tensor<\?xf32>\)\s*\{", row_text)
        if while_start_match is None:
            continue
        while_result, condition_p, condition_out = while_start_match.groups()
        first_open = row_loop.span[0] + while_start_match.end() - 1
        first_end = _matching_brace(text, first_open)
        if first_end is None:
            continue
        do_match = re.match(r"\s*do\s*\{", text[first_end:row_loop.span[1]])
        if do_match is None:
            continue
        second_open = first_end + do_match.end() - 1
        while_end = _matching_brace(text, second_open)
        if while_end is None:
            continue
        condition_text = text[first_open + 1:first_end - 1]
        do_text = text[second_open + 1:while_end - 1]
        compare = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.cmpi\s+slt,\s*"
            rf"{re.escape(condition_p)},\s*{re.escape(successor.group(1))}"
            r"\s*:\s*i32", condition_text)
        if compare is None or re.search(
                rf"scf\.condition\s*\(\s*{re.escape(compare.group(1))}\s*\)"
                rf"\s*{re.escape(condition_p)},\s*{re.escape(condition_out)}",
                condition_text) is None:
            continue

        block = re.match(
            r"\s*\^bb0\(\s*(%[\w.$-]+)\s*:\s*i32,\s*"
            r"(%[\w.$-]+)\s*:\s*tensor<\?xf32>\s*\):", do_text)
        if block is None:
            continue
        p, carried_out = block.groups()
        p_index = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+{re.escape(p)}\s*"
            r":\s*i32\s+to\s+index", do_text)
        if p_index is None:
            continue
        p_idx = p_index.group(1)
        col_value = re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            rf"{re.escape(col_tensor.group(1))}\[{re.escape(p_idx)}\]"
            r"\s*:\s*tensor<\?xi32>", do_text)
        col_index = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
            rf"{re.escape(col_value.group(1))}\s*:\s*i32\s+to\s+index",
            do_text) if col_value else None)
        if col_index is None:
            continue

        generic_match = re.search(r"linalg\.generic\b", do_text)
        generic_outs = (do_text.find(" outs(", generic_match.start())
                        if generic_match else -1)
        generic_open = do_text.find("{", generic_outs) if generic_outs >= 0 else -1
        if generic_open < 0:
            continue
        generic_end = _matching_brace(do_text, generic_open)
        if generic_end is None:
            continue
        generic_text = do_text[generic_match.start():generic_end]
        if (not re.search(r'iterator_types\s*=\s*\["reduction"\]', generic_text) or
                len(re.findall(r"\bmemref\.load\b", generic_text)) != 2):
            continue
        reduction_index = re.search(
            r"(%[\w.$-]+)\s*=\s*linalg\.index\s+0\s*:\s*index",
            generic_text)
        block_arg = re.search(r"\^bb0\(\s*(%[\w.$-]+)\s*:\s*f32\s*\):",
                              generic_text)
        if reduction_index is None or block_arg is None:
            continue
        k = reduction_index.group(1)
        a_load = re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+{re.escape(names[3])}"
            rf"\[{row_iv},\s*{re.escape(k)}\]\s*:\s*{re.escape(types[3])}",
            generic_text)
        b_load = re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+{re.escape(names[4])}"
            rf"\[{re.escape(k)},\s*{re.escape(col_index.group(1))}\]\s*:\s*"
            rf"{re.escape(types[4])}", generic_text)
        if a_load is None or b_load is None:
            continue
        product = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.mulf\s+"
            rf"(?:{re.escape(a_load.group(1))},\s*{re.escape(b_load.group(1))}|"
            rf"{re.escape(b_load.group(1))},\s*{re.escape(a_load.group(1))})"
            r"\s*:\s*f32", generic_text)
        total = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addf\s+"
            rf"(?:{re.escape(block_arg.group(1))},\s*{re.escape(product.group(1))}|"
            rf"{re.escape(product.group(1))},\s*{re.escape(block_arg.group(1))})"
            r"\s*:\s*f32", generic_text) if product else None)
        if total is None or re.search(
                rf"linalg\.yield\s+{re.escape(total.group(1))}\s*:\s*f32",
                generic_text) is None:
            continue

        generic_assignment_line = do_text.rfind("\n", 0, generic_match.start()) + 1
        generic_assignment = re.match(
            r"\s*(%[\w.$-]+)\s*=\s*$",
            do_text[generic_assignment_line:generic_match.start()])
        if generic_assignment is None:
            continue
        inverse = re.search(
            rf"(%[\w.$-]+)\s*=\s*polygeist\.submapInverse\([^\n]*"
            rf"{re.escape(generic_assignment.group(1))}[^\n]*\)", do_text)
        dot = (re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            rf"{re.escape(inverse.group(1))}\[\]", do_text) if inverse else None)
        self_value = re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            rf"{re.escape(self_tensor.group(1))}\[{re.escape(p_idx)}\]",
            do_text)
        if dot is None or self_value is None:
            continue
        beta_product = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.mulf\s+"
            rf"(?:{re.escape(names[6])},\s*{re.escape(self_value.group(1))}|"
            rf"{re.escape(self_value.group(1))},\s*{re.escape(names[6])})"
            r"\s*:\s*f32", do_text)
        alpha_product = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.mulf\s+"
            rf"(?:{re.escape(names[5])},\s*{re.escape(dot.group(1))}|"
            rf"{re.escape(dot.group(1))},\s*{re.escape(names[5])})"
            r"\s*:\s*f32", do_text)
        sum_value = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addf\s+"
            rf"(?:{re.escape(beta_product.group(1))},\s*{re.escape(alpha_product.group(1))}|"
            rf"{re.escape(alpha_product.group(1))},\s*{re.escape(beta_product.group(1))})"
            r"\s*:\s*f32", do_text)
                     if beta_product and alpha_product else None)
        inserted = (re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.insert\s+"
            rf"{re.escape(sum_value.group(1))}\s+into\s+"
            rf"{re.escape(carried_out)}\[{re.escape(p_idx)}\]",
            do_text) if sum_value else None)
        increment = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addi\s+{re.escape(p)},\s*"
            r"(%[\w.$-]+)\s*:\s*i32", do_text)
        if inserted is None or increment is None or re.search(
                rf"{re.escape(increment.group(2))}\s*=\s*arith\.constant\s+1"
                r"\s*:\s*i32", text[:row_loop.span[0]]) is None or re.search(
                rf"scf\.yield\s+{re.escape(increment.group(1))},\s*"
                rf"{re.escape(inserted.group(1))}", do_text) is None:
            continue
        if re.search(
                rf"affine\.yield\s+{re.escape(while_result)}#1\s*:\s*"
                r"tensor<\?xf32>", row_text[while_end - row_loop.span[0]:]) is None:
            continue
        tail = re.match(
            rf"\s*(%[\w.$-]+)\s*=\s*bufferization\.to_memref\s+"
            rf"{re.escape(row_result)}\s*:\s*memref<\?xf32>\s*\n"
            rf"\s*memref\.copy\s+\1,\s*{re.escape(names[7])}\s*:"
            r"[^\n]*",
            text[row_loop.span[1]:])
        if tail is None:
            continue

        line_start = text.rfind("\n", 0, row_loop.span[0]) + 1
        indent = re.match(r"\s*", text[line_start:row_loop.span[0]]).group(0)
        uid = row_loop.span[0]
        rows_value = f"%cusparse_sddmm_rows_{uid}"
        a_cast = f"%cusparse_sddmm_a_{uid}"
        b_cast = f"%cusparse_sddmm_b_{uid}"
        lines = [
            f"{rows_value} = arith.constant {row_count} : index",
            f"{a_cast} = memref.cast {names[3]} : {types[3]} to memref<?x?xf32>",
            f"{b_cast} = memref.cast {names[4]} : {types[4]} to memref<?x?xf32>",
            "kernel.launch @cusparseSDDMM_CSR_f32_memref(" +
            ", ".join([rows_value, names[0], names[1], names[2], a_cast,
                       b_cast, names[5], names[6], names[7]]) +
            ") : (index, memref<?xi32>, memref<?xi32>, memref<?xf32>, "
            "memref<?x?xf32>, memref<?x?xf32>, f32, f32, memref<?xf32>) -> ()"
        ]
        replacement = ("\n" + indent).join(lines)
        symbol = "cusparseSDDMM_CSR_f32_memref"
        rendered.append((line_start, row_loop.span[1] + tail.end(),
                         indent + replacement, symbol))
    return rendered


def _render_cusparse_index_conversions_buffer(
        text: str) -> list[tuple[int, int, str, str]]:
    """Recognize large-shape memref forms of ATen COO/CSR conversion."""
    coo_to_csr = {"aten_convert_coo_to_csr_cpu",
                  "aten_sparse_coo_to_csr_cpu"}
    csr_to_coo = {"aten_convert_csr_to_coo_cpu",
                  "aten_sparse_matmul_csr_to_coo_cpu"}
    rendered: list[tuple[int, int, str, str]] = []
    function_re = re.compile(
        r"func\.func(?:\s+private)?\s+@([\w.$-]+)\s*\(([^)]*)\)",
        re.MULTILINE)
    for function in function_re.finditer(text):
        name = function.group(1)
        if name not in coo_to_csr | csr_to_coo:
            continue
        arguments = re.findall(
            r"(%[\w.$-]+)\s*:\s*(memref<[^>]+>)", function.group(2))
        expected = 3 if name == "aten_convert_csr_to_coo_cpu" else 2
        if (len(arguments) != expected or
                any(ty != "memref<?xi32>" for _, ty in arguments)):
            continue
        names = [value for value, _ in arguments]
        source = names[0]
        output = names[2] if expected == 3 else names[1]
        if source == output:
            continue
        signature_end = text.find("\n", function.end())
        opening = text.rfind("{", function.end(), signature_end)
        function_end = _matching_brace(text, opening) if opening >= 0 else None
        if function_end is None:
            continue
        loops = [loop for loop in parse_loops(text)
                 if opening < loop.span[0] and loop.span[1] < function_end]
        if len(loops) != 1 or loops[0].kind != "affine.for":
            continue
        outer = loops[0]
        body = text[outer.span[0]:outer.span[1]]
        header = re.match(
            rf"affine\.for\s+{re.escape(outer.induction)}\s*=\s*0\s+to\s+"
            r"(\d+)\s*[{]", body)
        if header is None or len(re.findall(r"\bscf\.while\b", body)) != 1:
            continue
        bound = int(header.group(1))
        iv = re.escape(outer.induction)

        if name in coo_to_csr:
            if bound <= 1:
                continue
            rows = bound - 1
            alloca = re.search(
                r"(%[\w.$-]+)\s*=\s*memref\.alloca\(\)\s*:\s*memref<i32>",
                text[opening:outer.span[0]])
            carried = (re.search(
                rf"(%[\w.$-]+)\s*=\s*affine\.load\s+"
                rf"{re.escape(alloca.group(1))}\[\]\s*:\s*memref<i32>", body)
                       if alloca else None)
            row_i = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+{iv}\s*:\s*"
                r"index\s+to\s+i32", body)
            while_match = (re.search(
                rf"(%[\w.$-]+)\s*=\s*scf\.while\s*\(\s*"
                rf"(%[\w.$-]+)\s*=\s*{re.escape(carried.group(1))}\s*\)",
                body) if carried else None)
            if alloca is None or row_i is None or while_match is None:
                continue
            result, p = while_match.groups()
            p_index = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+{re.escape(p)}"
                r"\s*:\s*i32\s+to\s+index", body)
            row_value = (re.search(
                rf"(%[\w.$-]+)\s*=\s*memref\.load\s+{re.escape(source)}"
                rf"\[{re.escape(p_index.group(1))}\]", body)
                         if p_index else None)
            less_row = (re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.cmpi\s+slt,\s*"
                rf"{re.escape(row_value.group(1))},\s*"
                rf"{re.escape(row_i.group(1))}", body) if row_value else None)
            nnz_cmp = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.cmpi\s+slt,\s*"
                rf"{re.escape(p)},\s*(%[\w.$-]+)"
                r"\s*:\s*i32", body)
            stores = (re.search(
                rf"affine\.store\s+{re.escape(result)},\s*{re.escape(output)}"
                rf"\[{iv}\].*?affine\.store\s+{re.escape(result)},\s*"
                rf"{re.escape(alloca.group(1))}\[\]", body, re.DOTALL))
            if less_row is None or nnz_cmp is None or stores is None:
                continue
            nnz_constant = re.search(
                rf"{re.escape(nnz_cmp.group(2))}\s*=\s*arith\.constant\s+"
                r"(\d+)\s*:\s*i32", text[opening:outer.span[0]])
            zero_store = re.search(
                rf"affine\.store\s+(%[\w.$-]+),\s*"
                rf"{re.escape(alloca.group(1))}\[\]", text[opening:outer.span[0]])
            if nnz_constant is None or zero_store is None or re.search(
                    rf"{re.escape(zero_store.group(1))}\s*=\s*"
                    r"arith\.constant\s+0\s*:\s*i32",
                    text[opening:outer.span[0]]) is None:
                continue
            increment = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.addi\s+{re.escape(p)},\s*"
                r"(%[\w.$-]+)\s*:\s*i32", body)
            next_if_row = (re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.select\s+"
                rf"{re.escape(less_row.group(1))},\s*"
                rf"{re.escape(increment.group(1))},\s*{re.escape(p)}\s*:\s*i32",
                body) if increment else None)
            condition_value = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.select\s+"
                rf"{re.escape(nnz_cmp.group(1))},\s*"
                rf"{re.escape(less_row.group(1))},\s*(%[\w.$-]+)\s*:\s*i1",
                body)
            next_p = (re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.select\s+"
                rf"{re.escape(nnz_cmp.group(1))},\s*"
                rf"{re.escape(next_if_row.group(1))},\s*{re.escape(p)}\s*:\s*i32",
                body) if next_if_row else None)
            if (increment is None or condition_value is None or next_p is None or
                    re.search(rf"{re.escape(increment.group(2))}\s*=\s*"
                              r"arith\.constant\s+1\s*:\s*i32",
                              text[opening:outer.span[0]]) is None or
                    re.search(rf"{re.escape(condition_value.group(2))}\s*=\s*"
                              r"arith\.constant\s+false",
                              text[opening:outer.span[0]]) is None or
                    re.search(rf"scf\.condition\s*\(\s*"
                              rf"{re.escape(condition_value.group(1))}\s*\)\s*"
                              rf"{re.escape(next_p.group(1))}\s*:\s*i32",
                              body) is None):
                continue
            symbol = "cusparseXcoo2csr_i32_memref"
        else:
            if bound <= 0:
                continue
            rows = bound
            row_i = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+{iv}\s*:\s*"
                r"index\s+to\s+i32", body)
            direct = re.search(
                rf"(%[\w.$-]+)\s*=\s*affine\.load\s+{re.escape(source)}"
                rf"\[{iv}\]", body)
            successor = re.search(
                rf"(%[\w.$-]+)\s*=\s*affine\.load\s+{re.escape(source)}"
                rf"\[{iv}\s*\+\s*1\]", body)
            while_match = (re.search(
                rf"(%[\w.$-]+)\s*=\s*scf\.while\s*\(\s*"
                rf"(%[\w.$-]+)\s*=\s*{re.escape(direct.group(1))}\s*\)",
                body) if direct else None)
            if not all((row_i, direct, successor, while_match)):
                continue
            _, p = while_match.groups()
            condition = re.search(
                rf"arith\.cmpi\s+slt,\s*{re.escape(p)},\s*"
                rf"{re.escape(successor.group(1))}", body)
            p_index = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+{re.escape(p)}"
                r"\s*:\s*i32\s+to\s+index", body)
            store = (re.search(
                rf"memref\.store\s+{re.escape(row_i.group(1))},\s*"
                rf"{re.escape(output)}\[{re.escape(p_index.group(1))}\]",
                body) if p_index else None)
            increment = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.addi\s+{re.escape(p)},\s*"
                r"(%[\w.$-]+)\s*:\s*i32", body)
            if condition is None or store is None or increment is None or re.search(
                    rf"{re.escape(increment.group(2))}\s*=\s*"
                    r"arith\.constant\s+1\s*:\s*i32",
                    text[opening:outer.span[0]]) is None or re.search(
                    rf"scf\.yield\s+{re.escape(increment.group(1))}\s*:\s*i32",
                    body) is None:
                continue
            symbol = "cusparseXcsr2coo_i32_memref"

        line_start = text.rfind("\n", 0, outer.span[0]) + 1
        indent = re.match(r"\s*", text[line_start:outer.span[0]]).group(0)
        rows_value = f"%cusparse_convert_rows_{outer.span[0]}"
        replacement = (f"{indent}{rows_value} = arith.constant {rows} : index\n"
                       f"{indent}kernel.launch @{symbol}({rows_value}, "
                       f"{source}, {output}) : (index, memref<?xi32>, "
                       "memref<?xi32>) -> ()")
        rendered.append((line_start, outer.span[1], replacement, symbol))
    return rendered


def _render_cusparse_index_conversions(
        text: str) -> list[tuple[int, int, str, str]]:
    """Recognize complete ATen COO/CSR row-index conversion algorithms."""
    coo_to_csr = {"aten_convert_coo_to_csr_cpu",
                  "aten_sparse_coo_to_csr_cpu"}
    csr_to_coo = {"aten_convert_csr_to_coo_cpu",
                  "aten_sparse_matmul_csr_to_coo_cpu"}
    wanted = coo_to_csr | csr_to_coo
    rendered = _render_cusparse_index_conversions_buffer(text)
    function_re = re.compile(
        r"func\.func(?:\s+private)?\s+@([\w.$-]+)\s*\(([^)]*)\)",
        re.MULTILINE)
    for function in function_re.finditer(text):
        function_name = function.group(1)
        if function_name not in wanted:
            continue
        arguments = re.findall(
            r"(%[\w.$-]+)\s*:\s*(memref<[^>]+>)", function.group(2))
        expected_args = 2 if function_name != "aten_convert_csr_to_coo_cpu" else 3
        if (len(arguments) != expected_args or
                any(ty != "memref<?xi32>" for _, ty in arguments)):
            continue
        names = [name for name, _ in arguments]
        source = names[0]
        output = names[1] if expected_args == 2 else names[2]
        if source == output:
            continue
        signature_end = text.find("\n", function.end())
        opening = text.rfind("{", function.end(), signature_end)
        function_end = _matching_brace(text, opening) if opening >= 0 else None
        if function_end is None:
            continue
        loops = [loop for loop in parse_loops(text)
                 if opening < loop.span[0] and loop.span[1] < function_end]
        if len(loops) != 1 or loops[0].kind != "affine.for":
            continue
        outer = loops[0]
        outer_text = text[outer.span[0]:outer.span[1]]
        if (len(re.findall(r"\bscf\.while\b", outer_text)) != 1 or
                re.search(r"\blinalg\.", outer_text)):
            continue
        source_tensor = re.search(
            rf"(%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+"
            rf"{re.escape(source)}\s*:\s*memref<\?xi32>",
            text[opening:outer.span[0]])
        output_tensor = re.search(
            rf"(%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+"
            rf"{re.escape(output)}\s*:\s*memref<\?xi32>",
            text[opening:outer.span[0]])
        if source_tensor is None or output_tensor is None:
            continue
        line_start = text.rfind("\n", 0, outer.span[0]) + 1

        if function_name in coo_to_csr:
            assignment = re.match(
                r"\s*(%[\w.$-]+):2\s*=\s*$", text[line_start:outer.span[0]])
            header = re.match(
                rf"affine\.for\s+{re.escape(outer.induction)}\s*=\s*0\s+to\s+"
                r"(\d+)\s+iter_args\s*\(\s*(%[\w.$-]+)\s*=\s*"
                r"(%[\w.$-]+),\s*(%[\w.$-]+)\s*=\s*"
                rf"{re.escape(output_tensor.group(1))}\s*\)\s*->\s*"
                r"\(tensor<i32>,\s*tensor<\?xi32>\)\s*[{]", outer_text)
            if assignment is None or header is None or int(header.group(1)) <= 1:
                continue
            rows = int(header.group(1)) - 1
            scalar_iter, scalar_init, output_iter = header.groups()[1:]
            initial_zero = re.search(
                rf"{re.escape(scalar_init)}\s*=\s*tensor\.insert\s+"
                r"(%[\w.$-]+)\s+into\s+%[\w.$-]+\[\]",
                text[opening:outer.span[0]])
            row_i = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
                rf"{re.escape(outer.induction)}\s*:\s*index\s+to\s+i32",
                outer_text)
            scalar_extract = re.search(
                rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
                rf"{re.escape(scalar_iter)}\[\]", outer_text)
            if initial_zero is None or row_i is None or scalar_extract is None:
                continue
            if re.search(
                    rf"{re.escape(initial_zero.group(1))}\s*=\s*"
                    r"arith\.constant\s+0\s*:\s*i32",
                    text[opening:outer.span[0]]) is None:
                continue
            while_match = re.search(
                rf"(%[\w.$-]+)\s*=\s*scf\.while\s*\(\s*"
                rf"(%[\w.$-]+)\s*=\s*{re.escape(scalar_extract.group(1))}"
                r"\s*\)\s*:\s*\(i32\)\s*->\s*i32\s*[{]",
                outer_text)
            if while_match is None:
                continue
            while_result, p = while_match.groups()
            p_index = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+{re.escape(p)}"
                r"\s*:\s*i32\s+to\s+index", outer_text)
            row_value = (re.search(
                rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
                rf"{re.escape(source_tensor.group(1))}"
                rf"\[{re.escape(p_index.group(1))}\]", outer_text)
                         if p_index else None)
            less_row = (re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.cmpi\s+slt,\s*"
                rf"{re.escape(row_value.group(1))},\s*{re.escape(row_i.group(1))}",
                outer_text) if row_value else None)
            nnz_compare = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.cmpi\s+slt,\s*{re.escape(p)},\s*"
                r"(%[\w.$-]+)\s*:\s*i32", outer_text)
            if less_row is None or nnz_compare is None or re.search(
                    rf"{re.escape(nnz_compare.group(2))}\s*=\s*"
                    r"arith\.constant\s+(\d+)\s*:\s*i32",
                    text[opening:outer.span[0]]) is None:
                continue
            increment = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.addi\s+{re.escape(p)},\s*"
                r"(%[\w.$-]+)\s*:\s*i32", outer_text)
            next_if_row = (re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.select\s+"
                rf"{re.escape(less_row.group(1))},\s*"
                rf"{re.escape(increment.group(1))},\s*{re.escape(p)}\s*:\s*i32",
                outer_text) if increment else None)
            condition_value = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.select\s+"
                rf"{re.escape(nnz_compare.group(1))},\s*"
                rf"{re.escape(less_row.group(1))},\s*(%[\w.$-]+)\s*:\s*i1",
                outer_text)
            next_p = (re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.select\s+"
                rf"{re.escape(nnz_compare.group(1))},\s*"
                rf"{re.escape(next_if_row.group(1))},\s*{re.escape(p)}\s*:\s*i32",
                outer_text) if next_if_row else None)
            if (increment is None or condition_value is None or next_p is None or
                    re.search(rf"{re.escape(increment.group(2))}\s*=\s*"
                              r"arith\.constant\s+1\s*:\s*i32",
                              text[opening:outer.span[0]]) is None or
                    re.search(rf"{re.escape(condition_value.group(2))}\s*=\s*"
                              r"arith\.constant\s+false",
                              text[opening:outer.span[0]]) is None or
                    re.search(rf"scf\.condition\s*\(\s*"
                              rf"{re.escape(condition_value.group(1))}\s*\)\s*"
                              rf"{re.escape(next_p.group(1))}\s*:\s*i32",
                              outer_text) is None):
                continue
            inserted_out = re.search(
                rf"(%[\w.$-]+)\s*=\s*tensor\.insert\s+"
                rf"{re.escape(while_result)}\s+into\s+{re.escape(output_iter)}"
                rf"\[{re.escape(outer.induction)}\]", outer_text)
            inserted_scalar = re.search(
                rf"(%[\w.$-]+)\s*=\s*tensor\.insert\s+"
                rf"{re.escape(while_result)}\s+into\s+{re.escape(scalar_iter)}"
                r"\[\]", outer_text)
            if inserted_out is None or inserted_scalar is None or re.search(
                    rf"affine\.yield\s+{re.escape(inserted_scalar.group(1))},\s*"
                    rf"{re.escape(inserted_out.group(1))}", outer_text) is None:
                continue
            result_ref = assignment.group(1) + "#1"
            symbol = "cusparseXcoo2csr_i32_memref"
        else:
            assignment = re.match(
                r"\s*(%[\w.$-]+)\s*=\s*$", text[line_start:outer.span[0]])
            header = re.match(
                rf"affine\.for\s+{re.escape(outer.induction)}\s*=\s*0\s+to\s+"
                r"(\d+)\s+iter_args\s*\(\s*(%[\w.$-]+)\s*=\s*"
                rf"{re.escape(output_tensor.group(1))}\s*\)\s*->\s*"
                r"\(tensor<\?xi32>\)\s*[{]", outer_text)
            if assignment is None or header is None or int(header.group(1)) <= 0:
                continue
            rows = int(header.group(1))
            output_iter = header.group(2)
            row_i = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
                rf"{re.escape(outer.induction)}\s*:\s*index\s+to\s+i32",
                outer_text)
            direct = re.search(
                rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
                rf"{re.escape(source_tensor.group(1))}"
                rf"\[{re.escape(outer.induction)}\]", outer_text)
            successor_index = re.search(
                rf"(%[\w.$-]+)\s*=\s*affine\.apply\s+([^\n]+)"
                rf"\(\s*{re.escape(outer.induction)}\s*\)", outer_text)
            successor = (re.search(
                rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
                rf"{re.escape(source_tensor.group(1))}"
                rf"\[{re.escape(successor_index.group(1))}\]", outer_text)
                         if successor_index else None)
            if not all((row_i, direct, successor_index, successor)):
                continue
            successor_map = _compact_affine_map(_resolve_affine_map_text(
                text[:outer.span[1]], successor_index.group(2).strip()))
            if successor_map != "affine_map<(d0)->(d0+1)>":
                continue
            while_match = re.search(
                rf"(%[\w.$-]+):2\s*=\s*scf\.while\s*\(\s*"
                rf"(%[\w.$-]+)\s*=\s*{re.escape(direct.group(1))},\s*"
                rf"(%[\w.$-]+)\s*=\s*{re.escape(output_iter)}\s*\)",
                outer_text)
            if while_match is None:
                continue
            while_result, p, carried = while_match.groups()
            condition = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.cmpi\s+slt,\s*{re.escape(p)},\s*"
                rf"{re.escape(successor.group(1))}", outer_text)
            p_index = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+{re.escape(p)}"
                r"\s*:\s*i32\s+to\s+index", outer_text)
            inserted = (re.search(
                rf"(%[\w.$-]+)\s*=\s*tensor\.insert\s+"
                rf"{re.escape(row_i.group(1))}\s+into\s+{re.escape(carried)}"
                rf"\[{re.escape(p_index.group(1))}\]", outer_text)
                        if p_index else None)
            if condition is None or inserted is None or re.search(
                    rf"affine\.yield\s+{re.escape(while_result)}#1",
                    outer_text) is None:
                continue
            result_ref = assignment.group(1)
            symbol = "cusparseXcsr2coo_i32_memref"

        tail = re.match(
            rf"\s*(%[\w.$-]+)\s*=\s*bufferization\.to_memref\s+"
            rf"{re.escape(result_ref)}\s*:\s*memref<\?xi32>\s*\n"
            rf"\s*memref\.copy\s+\1,\s*{re.escape(output)}\s*:[^\n]*",
            text[outer.span[1]:])
        if tail is None:
            continue
        indent = re.match(r"\s*", text[line_start:outer.span[0]]).group(0)
        rows_value = f"%cusparse_convert_rows_{outer.span[0]}"
        lines = [f"{rows_value} = arith.constant {rows} : index",
                 f"kernel.launch @{symbol}({rows_value}, {source}, {output}) : "
                 "(index, memref<?xi32>, memref<?xi32>) -> ()"]
        replacement = ("\n" + indent).join(lines)
        rendered.append((line_start, outer.span[1] + tail.end(),
                         indent + replacement, symbol))
    return rendered


def _render_cub_quant_col_offsets(
        text: str, instances: list, bodies: list
        ) -> list[tuple[int, int, str, str, list[int]]]:
    """Fuse the complete ATen quantized column-offset composition into CUB."""
    rendered: list[tuple[int, int, str, str, list[int]]] = []
    function_re = re.compile(
        r"func\.func(?:\s+private)?\s+@aten_quant_col_offsets_cpu\s*"
        r"\(\s*(%[\w.$-]+)\s*:\s*memref<\?x(\d+)xi8>\s*,\s*"
        r"(%[\w.$-]+)\s*:\s*i32\s*,\s*"
        r"(%[\w.$-]+)\s*:\s*memref<\?xi32>\s*\)", re.MULTILINE)
    constants = parse_constants(text)
    for function in function_re.finditer(text):
        weights, physical_cols, zero_point, output = function.groups()
        signature_end = text.find("\n", function.end())
        opening = text.rfind("{", function.end(), signature_end)
        function_end = _matching_brace(text, opening) if opening >= 0 else None
        if function_end is None:
            continue
        owned = [i for i, inst in enumerate(instances)
                 if opening < inst.span[0] and inst.span[1] < function_end]
        if len(owned) != 3:
            continue
        i0, i1, i2 = owned
        init, reduction, subtract = (bodies[i0], bodies[i1], bodies[i2])
        if (init.iterator_types != ["parallel"] or
                len(init.ins_arg_names) != 0 or
                len(init.outs_arg_names) != 1 or
                len(init.yield_values) != 1 or
                constants.get(init.yield_values[0]) != 0.0):
            continue
        reduction_maps = [
            _compact_affine_map(m) for m in reduction.indexing_maps]
        direct_maps = ["affine_map<(d0,d1)->(d1,d0)>",
                       "affine_map<(d0,d1)->(d0)>"]
        submap_maps = ["affine_map<(d0,d1)->(d0,d1)>",
                       "affine_map<(d0,d1)->(d0,d1)>"]
        if (reduction.iterator_types != ["parallel", "reduction"] or
                len(reduction.ins_arg_names) != 1 or
                len(reduction.outs_arg_names) != 1 or
                reduction_maps not in (direct_maps, submap_maps)):
            continue
        uses_submaps = reduction_maps == submap_maps
        reduction_text = "\n".join(reduction.body_lines)
        value = re.escape(reduction.ins_arg_names[0])
        accumulator = re.escape(reduction.outs_arg_names[0])
        widened = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.extsi\s+{value}\s*:\s*i8\s+to\s+i32",
            reduction_text)
        total = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addi\s+"
            rf"(?:{accumulator},\s*{re.escape(widened.group(1))}|"
            rf"{re.escape(widened.group(1))},\s*{accumulator})\s*:\s*i32",
            reduction_text) if widened else None)
        if (total is None or reduction.yield_values != [total.group(1)]):
            continue
        if (subtract.iterator_types != ["parallel"] or
                len(subtract.ins_arg_names) != 1 or
                len(subtract.outs_arg_names) != 1 or
                [_compact_affine_map(m) for m in subtract.indexing_maps] != [
                    "affine_map<(d0)->(d0)>",
                    "affine_map<(d0)->(d0)>"]):
            continue
        subtract_text = "\n".join(subtract.body_lines)
        sub_input = re.escape(subtract.ins_arg_names[0])
        sub = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.subi\s+{sub_input},\s*"
            r"(%[\w.$-]+)\s*:\s*i32", subtract_text)
        if (sub is None or subtract.yield_values != [sub.group(1)]):
            continue
        offset = sub.group(2)

        inst0, inst1, inst2 = instances[i0], instances[i1], instances[i2]
        input_names = _extract_ssa_names(inst1.ins_part)
        reduction_outs = _extract_ssa_names(inst1.outs_part)
        subtract_ins = _extract_ssa_names(inst2.ins_part)
        subtract_outs = _extract_ssa_names(inst2.outs_part)
        if (len(input_names) != 1 or len(reduction_outs) != 1 or
                len(subtract_ins) != 1 or len(subtract_outs) != 1 or
                inst0.result_ssa is None or inst1.result_ssa is None or
                inst2.result_ssa is None):
            continue
        prefix = text[opening:inst0.span[0]]
        between01 = text[inst0.span[1]:inst1.span[0]]
        between12 = text[inst1.span[1]:inst2.span[0]]
        tail = text[inst2.span[1]:function_end]
        weights_tensor = re.search(
            rf"(%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+"
            rf"{re.escape(weights)}\s*:\s*memref<\?x{physical_cols}xi8>",
            prefix)
        output_tensor = re.search(
            rf"(%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+"
            rf"{re.escape(output)}\s*:\s*memref<\?xi32>", prefix)
        if weights_tensor is None or output_tensor is None:
            continue
        final_tensor = inst2.result_ssa
        if not uses_submaps:
            input_slice = re.search(
                rf"{re.escape(input_names[0])}\s*=\s*tensor\.extract_slice\s+"
                rf"{re.escape(weights_tensor.group(1))}\[0,\s*0\]\s*"
                r"\[(%[\w.$-]+),\s*(%[\w.$-]+)\]\s*\[1,\s*1\]",
                between01)
            if input_slice is None:
                continue
            rows_ssa, cols_ssa = input_slice.groups()
            init_slice = re.search(
                rf"{re.escape(reduction_outs[0])}\s*=\s*"
                rf"tensor\.extract_slice\s+{re.escape(inst0.result_ssa)}"
                rf"\[0\]\s*\[{re.escape(cols_ssa)}\]\s*\[1\]", between01)
            inserted = re.search(
                rf"(%[\w.$-]+)\s*=\s*tensor\.insert_slice\s+"
                rf"{re.escape(inst1.result_ssa)}\s+into\s+"
                rf"{re.escape(inst0.result_ssa)}\[0\]\s*"
                rf"\[{re.escape(cols_ssa)}\]\s*\[1\]", between12)
            if (init_slice is None or inserted is None or
                    subtract_ins[0] != inserted.group(1) or
                    subtract_outs[0] != output_tensor.group(1)):
                continue
        else:
            init_out = _extract_ssa_names(inst0.outs_part)
            if len(init_out) != 1:
                continue
            init_submap = re.search(
                rf"{re.escape(init_out[0])}\s*=\s*polygeist\.submap\("
                r"(%[\w.$-]+),\s*(%[\w.$-]+)\)\s*"
                r"[{]map\s*=\s*([^}]+)[}]", prefix)
            init_inverse = re.search(
                rf"(%[\w.$-]+)\s*=\s*polygeist\.submapInverse\("
                rf"{re.escape(init_submap.group(1))},\s*"
                rf"{re.escape(inst0.result_ssa)},\s*"
                rf"{re.escape(init_submap.group(2))}\)\s*"
                r"[{]map\s*=\s*([^}]+)[}]", between01) if init_submap else None
            input_submap = re.search(
                rf"{re.escape(input_names[0])}\s*=\s*polygeist\.submap\("
                rf"{re.escape(weights_tensor.group(1))},\s*"
                r"(%[\w.$-]+),\s*(%[\w.$-]+)\)\s*"
                r"[{]map\s*=\s*([^}]+)[}]", between01)
            reduction_submap = (re.search(
                rf"{re.escape(reduction_outs[0])}\s*=\s*polygeist\.submap\("
                rf"{re.escape(init_inverse.group(1))},\s*"
                rf"{re.escape(input_submap.group(1))},\s*"
                rf"{re.escape(input_submap.group(2))}\)\s*"
                r"[{]map\s*=\s*([^}]+)[}]", between01)
                if init_inverse and input_submap else None)
            if not all((init_submap, init_inverse, input_submap,
                        reduction_submap)):
                continue
            cols_ssa, rows_ssa = input_submap.group(1), input_submap.group(2)
            resolved_input_map = _compact_affine_map(_resolve_affine_map_text(
                text[:inst1.span[0]], input_submap.group(3).strip()))
            resolved_reduce_map = _compact_affine_map(_resolve_affine_map_text(
                text[:inst1.span[0]], reduction_submap.group(1).strip()))
            if (resolved_input_map != "affine_map<(d0,d1)->(d1,d0)>" or
                    resolved_reduce_map != "affine_map<(d0,d1)->(d0)>" or
                    init_submap.group(2) != cols_ssa):
                continue
            reduction_inverse = re.search(
                rf"(%[\w.$-]+)\s*=\s*polygeist\.submapInverse\("
                rf"{re.escape(init_inverse.group(1))},\s*"
                rf"{re.escape(inst1.result_ssa)},\s*"
                rf"{re.escape(cols_ssa)},\s*{re.escape(rows_ssa)}\)\s*"
                r"[{]map\s*=\s*([^}]+)[}]", between12)
            subtract_input_view = (re.search(
                rf"{re.escape(subtract_ins[0])}\s*=\s*polygeist\.submap\("
                rf"{re.escape(reduction_inverse.group(1))},\s*"
                rf"{re.escape(cols_ssa)}\)\s*"
                r"[{]map\s*=\s*([^}]+)[}]", between12)
                if reduction_inverse else None)
            subtract_output_view = re.search(
                rf"{re.escape(subtract_outs[0])}\s*=\s*polygeist\.submap\("
                rf"{re.escape(output_tensor.group(1))},\s*"
                rf"{re.escape(cols_ssa)}\)\s*"
                r"[{]map\s*=\s*([^}]+)[}]", between12)
            final_inverse = (re.search(
                rf"(%[\w.$-]+)\s*=\s*polygeist\.submapInverse\("
                rf"{re.escape(output_tensor.group(1))},\s*"
                rf"{re.escape(inst2.result_ssa)},\s*"
                rf"{re.escape(cols_ssa)}\)\s*"
                r"[{]map\s*=\s*([^}]+)[}]", tail)
                if subtract_input_view and subtract_output_view else None)
            if final_inverse is None:
                continue
            one_dim_identity = "affine_map<(d0)->(d0)>"
            one_dim_projection = "affine_map<(d0,d1)->(d0)>"
            map_operands = [
                (init_submap.group(3), one_dim_identity),
                (init_inverse.group(2), one_dim_identity),
                (reduction_inverse.group(2), one_dim_projection),
                (subtract_input_view.group(1), one_dim_identity),
                (subtract_output_view.group(1), one_dim_identity),
                (final_inverse.group(2), one_dim_identity),
            ]
            if any(_compact_affine_map(_resolve_affine_map_text(
                    text[:function_end], actual.strip())) != expected
                   for actual, expected in map_operands):
                continue
            final_tensor = final_inverse.group(1)

        rows = constants.get(rows_ssa)
        cols = constants.get(cols_ssa)
        if (rows is None or cols is None or int(rows) != rows or
                int(cols) != cols or rows <= 0 or cols <= 0 or
                int(cols) != int(physical_cols)):
            continue
        offset_def = re.search(
            rf"{re.escape(offset)}\s*=\s*arith\.muli\s+"
            rf"(?:{re.escape(zero_point)},\s*(%[\w.$-]+)|"
            rf"(%[\w.$-]+),\s*{re.escape(zero_point)})\s*:\s*i32", prefix)
        if offset_def is None:
            continue
        row_i32 = offset_def.group(1) or offset_def.group(2)
        if constants.get(row_i32) != rows:
            continue
        to_memref = re.search(
            rf"(%[\w.$-]+)\s*=\s*bufferization\.to_memref\s+"
            rf"{re.escape(final_tensor)}\s*:\s*memref<\?xi32>", tail)
        copy = (re.search(
            rf"memref\.copy\s+{re.escape(to_memref.group(1))},\s*"
            rf"{re.escape(output)}\s*:\s*memref<\?xi32>\s+to\s+"
            r"memref<\?xi32>", tail) if to_memref else None)
        if copy is None:
            continue

        start = text.rfind("\n", 0, inst0.span[0]) + 1
        indent = re.match(r"\s*", text[start:inst0.span[0]]).group(0)
        attrs = (" {polygeist.fixed_extents = array<i64: "
                 f"{int(rows)}, {int(cols)}>}}")
        launch = (
            f"{indent}kernel.launch @cubQuantColOffsets_i8_i32_memref("
            f"{weights}, {offset}, {output}){attrs} : "
            f"(memref<?x{physical_cols}xi8>, i32, memref<?xi32>) -> ()")
        rendered.append((start, inst2.span[1] + copy.end(), launch,
                         "cubQuantColOffsets_i8_i32_memref", owned))
    return rendered


def _render_cub_adjacent_difference(
        text: str, instances: list, bodies: list
        ) -> list[tuple[int, int, str, str, list[int]]]:
    """Recognize ATen's complete forward adjacent-difference operation."""
    rendered: list[tuple[int, int, str, str, list[int]]] = []
    function_re = re.compile(
        r"func\.func(?:\s+private)?\s+@aten_diff_cpu\s*"
        r"\(\s*(%[\w.$-]+)\s*:\s*memref<\?xf32>\s*,\s*"
        r"(%[\w.$-]+)\s*:\s*memref<\?xf32>\s*\)", re.MULTILINE)
    constants = parse_constants(text)
    for function in function_re.finditer(text):
        input_memref, output_memref = function.groups()
        signature_end = text.find("\n", function.end())
        opening = text.rfind("{", function.end(), signature_end)
        function_end = _matching_brace(text, opening) if opening >= 0 else None
        if function_end is None:
            continue
        owned = [i for i, instance in enumerate(instances)
                 if opening < instance.span[0] and
                 instance.span[1] < function_end]
        if len(owned) != 1:
            continue
        body = bodies[owned[0]]
        if (body.iterator_types != ["parallel"] or
                len(body.ins_arg_names) != 2 or
                len(body.outs_arg_names) != 1 or
                [_compact_affine_map(m) for m in body.indexing_maps] != [
                    "affine_map<(d0)->(d0)>",
                    "affine_map<(d0)->(d0)>",
                    "affine_map<(d0)->(d0)>"]):
            continue
        body_text = "\n".join(body.body_lines)
        difference = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.subf\s+"
            rf"{re.escape(body.ins_arg_names[0])},\s*"
            rf"{re.escape(body.ins_arg_names[1])}\s*:\s*f32", body_text)
        if difference is None or body.yield_values != [difference.group(1)]:
            continue

        instance = instances[owned[0]]
        inputs = _extract_ssa_names(instance.ins_part)
        outputs = _extract_ssa_names(instance.outs_part)
        if (len(inputs) != 2 or len(outputs) != 1 or
                instance.result_ssa is None):
            continue
        prefix = text[opening:instance.span[0]]
        input_tensor = re.search(
            rf"(%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+"
            rf"{re.escape(input_memref)}\s*:\s*memref<\?xf32>", prefix)
        output_tensor = re.search(
            rf"(%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+"
            rf"{re.escape(output_memref)}\s*:\s*memref<\?xf32>", prefix)
        if input_tensor is None or output_tensor is None:
            continue

        def exact_slice(value: str, source: str, offset: int):
            return re.search(
                rf"{re.escape(value)}\s*=\s*tensor\.extract_slice\s+"
                rf"{re.escape(source)}\[{offset}\]\s*"
                r"\[(%[\w.$-]+)\]\s*\[1\]\s*:\s*"
                r"tensor<\?xf32>\s+to\s+tensor<\?xf32>", prefix)

        right = exact_slice(inputs[0], input_tensor.group(1), 1)
        left = exact_slice(inputs[1], input_tensor.group(1), 0)
        destination = exact_slice(outputs[0], output_tensor.group(1), 0)
        uses_submaps = not all((right, left, destination))
        if uses_submaps:
            def exact_submap(value: str, source: str):
                return re.search(
                    rf"{re.escape(value)}\s*=\s*polygeist\.submap\("
                    rf"{re.escape(source)},\s*(%[\w.$-]+)\)\s*"
                    r"[{]map\s*=\s*([^}]+)[}]", prefix)

            right = exact_submap(inputs[0], input_tensor.group(1))
            left = exact_submap(inputs[1], input_tensor.group(1))
            destination = exact_submap(outputs[0], output_tensor.group(1))
            if right is None or left is None or destination is None:
                continue
            resolved_maps = [
                _compact_affine_map(_resolve_affine_map_text(
                    text[:instance.span[0]], match.group(2).strip()))
                for match in (right, left, destination)]
            if resolved_maps != ["affine_map<(d0)->(d0+1)>",
                                 "affine_map<(d0)->(d0)>",
                                 "affine_map<(d0)->(d0)>"]:
                continue
        extent_ssa = right.group(1)
        if left.group(1) != extent_ssa or destination.group(1) != extent_ssa:
            continue
        output_count = constants.get(extent_ssa)
        if (output_count is None or int(output_count) != output_count or
                output_count < 1 or output_count >= 2147483647):
            continue

        tail = text[instance.span[1]:function_end]
        if uses_submaps:
            inserted = re.search(
                rf"(%[\w.$-]+)\s*=\s*polygeist\.submapInverse\("
                rf"{re.escape(output_tensor.group(1))},\s*"
                rf"{re.escape(instance.result_ssa)},\s*"
                rf"{re.escape(extent_ssa)}\)\s*"
                r"[{]map\s*=\s*([^}]+)[}]", tail)
            if (inserted is None or
                    _compact_affine_map(_resolve_affine_map_text(
                        text[:function_end], inserted.group(2).strip())) !=
                    "affine_map<(d0)->(d0)>"):
                continue
        else:
            inserted = re.search(
                rf"(%[\w.$-]+)\s*=\s*tensor\.insert_slice\s+"
                rf"{re.escape(instance.result_ssa)}\s+into\s+"
                rf"{re.escape(output_tensor.group(1))}\[0\]\s*"
                rf"\[{re.escape(extent_ssa)}\]\s*\[1\]", tail)
        to_memref = (re.search(
            rf"(%[\w.$-]+)\s*=\s*bufferization\.to_memref\s+"
            rf"{re.escape(inserted.group(1))}\s*:\s*memref<\?xf32>", tail)
                     if inserted else None)
        copy = (re.search(
            rf"memref\.copy\s+{re.escape(to_memref.group(1))},\s*"
            rf"{re.escape(output_memref)}\s*:\s*memref<\?xf32>\s+to\s+"
            r"memref<\?xf32>", tail) if to_memref else None)
        if copy is None:
            continue

        slice_starts = [prefix.rfind(value + " =") for value in inputs + outputs]
        if any(position < 0 for position in slice_starts):
            continue
        start = opening + min(slice_starts)
        start = text.rfind("\n", 0, start) + 1
        indent = re.match(r"\s*", text[start:instance.span[0]]).group(0)
        input_count = int(output_count) + 1
        launch = (
            f"{indent}kernel.launch @cubAdjacentDifference_f32_memref("
            f"{input_memref}, {output_memref}) "
            f"{{polygeist.fixed_extents = array<i64: {input_count}>}} : "
            f"(memref<?xf32>, memref<?xf32>) -> ()")
        rendered.append((start, instance.span[1] + copy.end(), launch,
                         "cubAdjacentDifference_f32_memref", owned))
    return rendered


def _render_cusparse_bsr_spmv(
        text: str) -> list[tuple[int, int, str, str]]:
    """Fuse the tensorized ATen square-BSR matvec into cuSPARSE SpMM.

    CUDA 12.6's generic SpMV does not execute BSR, while its SpMM BSR
    algorithm does.  Treat the vector as a one-column dense matrix.  This
    recognizer deliberately requires square row-major blocks and consumes the
    complete row/block-row/block-column reduction plus final writeback.
    """
    loops = parse_loops(text)
    rendered = _render_cusparse_bsr_spmv_buffer(text)
    for outer in loops:
        if outer.kind != "affine.for":
            continue
        nested = [loop for loop in loops
                  if outer.span[0] < loop.span[0] and
                  loop.span[1] < outer.span[1]]
        affine_children = [loop for loop in nested if loop.kind == "affine.for"]
        reductions = [loop for loop in nested if loop.kind == "scf.for"]
        if len(affine_children) != 1 or len(reductions) != 1:
            continue
        inner_loop, reduction = affine_children[0], reductions[0]
        if not (inner_loop.span[0] < reduction.span[0] and
                reduction.span[1] < inner_loop.span[1]):
            continue

        outer_line = text.rfind("\n", 0, outer.span[0]) + 1
        outer_assignment = re.match(
            r"\s*(%[\w.$-]+)\s*=\s*$", text[outer_line:outer.span[0]])
        outer_body = text[outer.span[0]:outer.span[1]]
        outer_header = re.match(
            rf"affine\.for\s+{re.escape(outer.induction)}\s*=\s*"
            r"0\s+to\s+(\d+)\s+iter_args\s*\(\s*"
            r"(%[\w.$-]+)\s*=\s*(%[\w.$-]+)\s*\)\s*->\s*"
            r"\(tensor<[^>]+>\)\s*\{", outer_body)
        inner_line = text.rfind("\n", outer.span[0], inner_loop.span[0]) + 1
        inner_assignment = re.match(
            r"\s*(%[\w.$-]+)\s*=\s*$", text[inner_line:inner_loop.span[0]])
        inner_body = text[inner_loop.span[0]:inner_loop.span[1]]
        inner_header = re.match(
            rf"affine\.for\s+{re.escape(inner_loop.induction)}\s*=\s*"
            r"0\s+to\s+(\d+)\s+iter_args\s*\(\s*"
            r"(%[\w.$-]+)\s*=\s*(%[\w.$-]+)\s*\)\s*->\s*"
            r"\(tensor<[^>]+>\)\s*\{", inner_body)
        if not all((outer_assignment, outer_header,
                    inner_assignment, inner_header)):
            continue
        outer_result = outer_assignment.group(1)
        block_rows = int(outer_header.group(1))
        outer_iter, output_init = outer_header.group(2), outer_header.group(3)
        inner_result = inner_assignment.group(1)
        block_dim = int(inner_header.group(1))
        inner_iter, inner_init = inner_header.group(2), inner_header.group(3)
        if (block_rows <= 0 or block_dim <= 0 or inner_init != outer_iter or
                re.search(rf"affine\.yield\s+{re.escape(inner_result)}\s*:",
                          outer_body) is None):
            continue

        before_reduction = text[inner_loop.span[0]:reduction.span[0]]
        reduction_line = text.rfind(
            "\n", inner_loop.span[0], reduction.span[0]) + 1
        reduction_assignment = re.match(
            r"\s*(%[\w.$-]+)\s*=\s*$",
            text[reduction_line:reduction.span[0]])
        if reduction_assignment is None:
            continue
        reduction_result = reduction_assignment.group(1)
        row_iv = re.escape(outer.induction)
        rowptr_loads = re.findall(
            r"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            r"(%[\w.$-]+)\[([^]]+)\]\s*:\s*tensor<[^>]*xi32>",
            before_reduction)
        if len(rowptr_loads) != 2 or rowptr_loads[0][1] != rowptr_loads[1][1]:
            continue
        direct = next((load for load in rowptr_loads
                       if load[2].strip() == outer.induction), None)
        successor = next((load for load in rowptr_loads if load != direct), None)
        if direct is None or successor is None:
            continue
        successor_apply = re.search(
            rf"{re.escape(successor[2].strip())}\s*=\s*affine\.apply\s+"
            rf"([^\n]+)\(\s*{row_iv}\s*\)", before_reduction)
        if successor_apply is None or _compact_affine_map(
                _resolve_affine_map_text(
                    text[:reduction.span[0]], successor_apply.group(1).strip())) != \
                "affine_map<(d0)->(d0+1)>":
            continue
        casts: dict[str, str] = {}
        for loaded, _, _ in rowptr_loads:
            cast = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
                rf"{re.escape(loaded)}\s*:\s*i32\s+to\s+index",
                before_reduction)
            if cast:
                casts[loaded] = cast.group(1)
        if len(casts) != 2:
            continue

        reduction_text = text[reduction.span[0]:reduction.span[1]]
        reduction_header = re.match(
            rf"scf\.for\s+{re.escape(reduction.induction)}\s*=\s*"
            rf"{re.escape(casts[direct[0]])}\s+to\s+"
            rf"{re.escape(casts[successor[0]])}\s+step\s+(%[\w.$-]+)\s+"
            r"iter_args\s*\(\s*(%[\w.$-]+)\s*=\s*(%[\w.$-]+)\s*\)\s*"
            r"->\s*\(f32\)\s*\{", reduction_text)
        if reduction_header is None:
            continue
        step, accumulator, zero = reduction_header.groups()
        prefix = text[:reduction.span[0]]
        if (re.search(rf"{re.escape(step)}\s*=\s*arith\.constant\s+1\s*:\s*index",
                      prefix) is None or
                re.search(rf"{re.escape(zero)}\s*=\s*arith\.constant\s+"
                          r"(?:0(?:\.0+)?(?:e[+-]?0+)?|0x0+)\s*:\s*f32",
                          prefix, re.IGNORECASE) is None):
            continue

        p = re.escape(reduction.induction)
        i = re.escape(inner_loop.induction)
        column = re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            rf"(%[\w.$-]+)\[{p}\]\s*:\s*tensor<[^>]*xi32>",
            reduction_text)
        if column is None:
            continue
        column_base = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.muli\s+{re.escape(column.group(1))},\s*"
            rf"(%[\w.$-]+)\s*:\s*i32", reduction_text)
        if column_base is None or re.search(
                rf"{re.escape(column_base.group(2))}\s*=\s*arith\.constant\s+"
                rf"{block_dim}\s*:\s*i32", prefix) is None:
            continue
        generic = re.search(
            r"linalg\.generic\s*\{[^}]*iterator_types\s*=\s*"
            r"\[\"reduction\"\][^}]*\}.*?\}\s*->\s*tensor<[^>]+>",
            reduction_text, re.DOTALL)
        if generic is None:
            continue
        generic_text = generic.group(0)
        block_arg = re.search(r"\^bb0\((%[\w.$-]+):\s*f32\):", generic_text)
        reduction_index = re.search(
            r"(%[\w.$-]+)\s*=\s*linalg\.index\s+0\s*:\s*index",
            generic_text)
        if block_arg is None or reduction_index is None:
            continue
        j = re.escape(reduction_index.group(1))
        j_i32 = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+{j}\s*:\s*"
            r"index\s+to\s+i32", generic_text)
        value = re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+(%[\w.$-]+)"
            rf"\[{p},\s*{i},\s*{j}\]\s*:\s*(memref<[^>]*xf32>)",
            generic_text)
        dense_offset = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addi\s+"
            rf"(?:{re.escape(column_base.group(1))},\s*{re.escape(j_i32.group(1))}|"
            rf"{re.escape(j_i32.group(1))},\s*{re.escape(column_base.group(1))})"
            r"\s*:\s*i32", generic_text) if j_i32 else None)
        dense_cast = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
            rf"{re.escape(dense_offset.group(1))}\s*:\s*i32\s+to\s+index",
            generic_text) if dense_offset else None)
        dense = (re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+(%[\w.$-]+)"
            rf"\[{re.escape(dense_cast.group(1))}\]\s*:\s*(memref<[^>]*xf32>)",
            generic_text) if dense_cast else None)
        if value is None or dense is None or len(re.findall(
                r"memref\.load\b", generic_text)) != 2:
            continue
        product = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.mulf\s+"
            rf"(?:{re.escape(value.group(1))},\s*{re.escape(dense.group(1))}|"
            rf"{re.escape(dense.group(1))},\s*{re.escape(value.group(1))})"
            r"\s*:\s*f32", generic_text)
        total = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addf\s+"
            rf"(?:{re.escape(block_arg.group(1))},\s*{re.escape(product.group(1))}|"
            rf"{re.escape(product.group(1))},\s*{re.escape(block_arg.group(1))})"
            r"\s*:\s*f32", generic_text) if product else None)
        if total is None or re.search(
                rf"linalg\.yield\s+{re.escape(total.group(1))}\s*:\s*f32",
                generic_text) is None:
            continue
        reduction_result_line = text.rfind("\n", reduction.span[0], generic.start() + reduction.span[0]) + 1
        generic_result = re.match(
            r"\s*(%[\w.$-]+)\s*=\s*$",
            text[reduction_result_line:reduction.span[0] + generic.start()])
        # The generic result must be converted back to the scalar reduction
        # and yielded; do not accept a merely similar inner computation.
        if generic_result is None:
            continue
        inverse = re.search(
            rf"(%[\w.$-]+)\s*=\s*polygeist\.submapInverse\([^\n]*"
            rf"{re.escape(generic_result.group(1))}[^\n]*\)", reduction_text)
        scalar = (re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            rf"{re.escape(inverse.group(1))}\[\]", reduction_text)
                  if inverse else None)
        if scalar is None or re.search(
                rf"scf\.yield\s+{re.escape(scalar.group(1))}\s*:\s*f32",
                reduction_text) is None:
            continue

        output_apply = re.search(
            rf"(%[\w.$-]+)\s*=\s*affine\.apply\s+([^\n]+)"
            rf"\(\s*{i},\s*{row_iv}\s*\)", inner_body)
        if output_apply is None:
            continue
        output_map = _compact_affine_map(_resolve_affine_map_text(
            text[:inner_loop.span[1]], output_apply.group(2).strip()))
        if output_map != f"affine_map<(d0,d1)->(d0+d1*{block_dim})>":
            continue
        inserted = re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.insert\s+"
            rf"{re.escape(reduction_result)}"
            rf"\s+into\s+{re.escape(inner_iter)}"
            rf"\[{re.escape(output_apply.group(1))}\]", inner_body)
        if inserted is None or re.search(
                rf"affine\.yield\s+{re.escape(inserted.group(1))}\s*:",
                inner_body) is None:
            continue

        row_source = _to_tensor_memref_source(
            text, rowptr_loads[0][1], outer.span[0])
        column_source = _to_tensor_memref_source(
            text, column.group(2), outer.span[0])
        output_source = _to_tensor_memref_source(text, output_init, outer.span[0])
        if row_source is None or column_source is None or output_source is None:
            continue
        operands = [row_source[0], column_source[0], value.group(2),
                    dense.group(2), output_source[0]]
        types = [row_source[1], column_source[1], value.group(3),
                 dense.group(3), output_source[1]]
        if (len(set(operands)) != 5 or
                [_sniff_elem_type(ty) for ty in types] !=
                ["i32", "i32", "f32", "f32", "f32"] or
                [_shaped_rank(ty) for ty in types] != [1, 1, 3, 1, 1] or
                re.fullmatch(rf"memref<\?x{block_dim}x{block_dim}xf32>",
                             types[2]) is None):
            continue
        tail = re.match(
            rf"\s*(%[\w.$-]+)\s*=\s*bufferization\.to_memref\s+"
            rf"{re.escape(outer_result)}\s*:\s*memref<[^>]*xf32>\s*\n"
            rf"\s*memref\.copy\s+\1,\s*{re.escape(operands[4])}\s*:\s*[^\n]*",
            text[outer.span[1]:])
        if tail is None:
            continue

        uid = outer.span[0]
        rows_value = f"%cusparse_bsr_rows_{uid}"
        dim_value = f"%cusparse_bsr_dim_{uid}"
        target_types = ["memref<?xi32>", "memref<?xi32>",
                        "memref<?x?x?xf32>", "memref<?xf32>",
                        "memref<?xf32>"]
        indent = re.match(r"\s*", text[outer_line:outer.span[0]]).group(0)
        lines = [f"{rows_value} = arith.constant {block_rows} : index",
                 f"{dim_value} = arith.constant {block_dim} : index"]
        normalized: list[str] = []
        for index, (operand, source_type, target_type) in enumerate(
                zip(operands, types, target_types)):
            if source_type != target_type:
                cast = f"%cusparse_bsr_arg_{uid}_{index}"
                lines.append(
                    f"{cast} = memref.cast {operand} : {source_type} to {target_type}")
                normalized.append(cast)
            else:
                normalized.append(operand)
        symbol = "cusparseSpMM_BSR_f32_memref"
        lines.append(
            f"kernel.launch @{symbol}(" +
            ", ".join([rows_value, dim_value, *normalized]) +
            ") : (index, index, " + ", ".join(target_types) + ") -> ()")
        replacement = ("\n" + indent).join(lines)
        rendered.append((outer_line, outer.span[1] + tail.end(),
                         indent + replacement, symbol))
    return rendered


def _render_cusparse_csr_spmm(
        text: str) -> list[tuple[int, int, str, str]]:
    """Replace a complete tensorized CSR-times-dense loop nest with SpMM."""
    loops = parse_loops(text)
    rendered: list[tuple[int, int, str, str]] = []
    for candidate in analyze_residual_loops(text):
        if candidate.kind != "csr_spmm":
            continue
        parents = [loop for loop in loops
                   if loop.span[0] < candidate.loop.span[0] and
                   candidate.loop.span[1] < loop.span[1]]
        if len(parents) != 2:
            continue
        row_loop = min(parents, key=lambda loop: loop.span[0])
        column_loop = max(parents, key=lambda loop: loop.span[0])
        if not (row_loop.span[0] < column_loop.span[0] and
                column_loop.span[1] < row_loop.span[1]):
            continue

        row_line_start = text.rfind("\n", 0, row_loop.span[0]) + 1
        row_assignment = re.match(
            r"\s*(%[\w.$-]+)\s*=\s*$",
            text[row_line_start:row_loop.span[0]])
        row_body = text[row_loop.span[0]:row_loop.span[1]]
        row_header = re.match(
            rf"affine\.for\s+{re.escape(row_loop.induction)}\s*=\s*"
            r"0\s+to\s+(\d+)\s+iter_args\s*\(\s*"
            r"(%[\w.$-]+)\s*=\s*(%[\w.$-]+)\s*\)\s*->\s*"
            r"\(tensor<[^>]+>\)\s*\{", row_body)
        column_line_start = text.rfind(
            "\n", row_loop.span[0], column_loop.span[0]) + 1
        column_assignment = re.match(
            r"\s*(%[\w.$-]+)\s*=\s*$",
            text[column_line_start:column_loop.span[0]])
        column_body = text[column_loop.span[0]:column_loop.span[1]]
        column_header = re.match(
            rf"affine\.for\s+{re.escape(column_loop.induction)}\s*=\s*"
            r"0\s+to\s+(\d+)\s+iter_args\s*\(\s*"
            r"(%[\w.$-]+)\s*=\s*(%[\w.$-]+)\s*\)\s*->\s*"
            r"\(tensor<[^>]+>\)\s*\{", column_body)
        if not all((row_assignment, row_header, column_assignment, column_header)):
            continue
        row_result = row_assignment.group(1)
        row_count = int(row_header.group(1))
        row_iter, output_init = row_header.group(2), row_header.group(3)
        column_result = column_assignment.group(1)
        column_count = int(column_header.group(1))
        column_iter, column_init = column_header.group(2), column_header.group(3)
        if (row_count <= 0 or column_count <= 0 or column_init != row_iter or
                re.search(rf"affine\.yield\s+{re.escape(column_result)}\s*:",
                          row_body) is None):
            continue

        before_inner = text[column_loop.span[0]:candidate.loop.span[0]]
        inner = text[candidate.loop.span[0]:candidate.loop.span[1]]
        rowptr_loads = re.findall(
            r"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            r"(%[\w.$-]+)\[([^]]+)\]\s*:\s*tensor<[^>]*xi32>",
            before_inner)
        if len(rowptr_loads) != 2 or rowptr_loads[0][1] != rowptr_loads[1][1]:
            continue
        direct_index = next((i for i, load in enumerate(rowptr_loads)
                             if load[2].strip() == row_loop.induction), None)
        if direct_index is None:
            continue
        direct = rowptr_loads[direct_index]
        successor = rowptr_loads[1 - direct_index]
        successor_apply = re.search(
            rf"{re.escape(successor[2].strip())}\s*=\s*affine\.apply\s+"
            rf"([^\n]+)\(\s*{re.escape(row_loop.induction)}\s*\)",
            before_inner)
        if successor_apply is None or _compact_affine_map(
                _resolve_affine_map_text(
                    text[:candidate.loop.span[0]],
                    successor_apply.group(1).strip())) != \
                "affine_map<(d0)->(d0+1)>":
            continue
        direct_cast = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
            rf"{re.escape(direct[0])}\s*:\s*i32\s+to\s+index", before_inner)
        successor_cast = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
            rf"{re.escape(successor[0])}\s*:\s*i32\s+to\s+index", before_inner)
        if direct_cast is None or successor_cast is None:
            continue

        inner_line_start = text.rfind(
            "\n", column_loop.span[0], candidate.loop.span[0]) + 1
        inner_assignment = re.match(
            r"\s*(%[\w.$-]+)\s*=\s*$",
            text[inner_line_start:candidate.loop.span[0]])
        inner_header = re.match(
            rf"scf\.for\s+{re.escape(candidate.loop.induction)}\s*=\s*"
            rf"{re.escape(direct_cast.group(1))}\s+to\s+"
            rf"{re.escape(successor_cast.group(1))}\s+step\s+%[\w.$-]+\s+"
            r"iter_args\s*\(\s*(%[\w.$-]+)\s*=\s*"
            r"(%[\w.$-]+)\s*\)\s*->\s*\(f32\)\s*\{", inner)
        if inner_assignment is None or inner_header is None:
            continue
        inner_result = inner_assignment.group(1)
        accumulator, zero = inner_header.group(1), inner_header.group(2)
        if re.search(
                rf"{re.escape(zero)}\s*=\s*arith\.constant\s+"
                r"(?:0(?:\.0+)?(?:e[+-]?0+)?|0x0+)\s*:\s*f32",
                text[:candidate.loop.span[0]], re.IGNORECASE) is None:
            continue

        p = re.escape(candidate.loop.induction)
        value = re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            rf"(%[\w.$-]+)\[{p}\]\s*:\s*tensor<[^>]*xf32>", inner)
        index = re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            rf"(%[\w.$-]+)\[{p}\]\s*:\s*tensor<[^>]*xi32>", inner)
        if value is None or index is None:
            continue
        index_cast = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
            rf"{re.escape(index.group(1))}\s*:\s*i32\s+to\s+index", inner)
        if index_cast is None:
            continue
        dense = re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            rf"(%[\w.$-]+)\[{re.escape(index_cast.group(1))},\s*"
            rf"{re.escape(column_loop.induction)}\]\s*:\s*"
            r"tensor<[^>]*xf32>", inner)
        if dense is None:
            continue
        product = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.mulf\s+"
            rf"(?:{re.escape(value.group(1))},\s*{re.escape(dense.group(1))}|"
            rf"{re.escape(dense.group(1))},\s*{re.escape(value.group(1))})"
            r"\s*:\s*f32", inner)
        total = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addf\s+"
            rf"(?:{re.escape(accumulator)},\s*{re.escape(product.group(1))}|"
            rf"{re.escape(product.group(1))},\s*{re.escape(accumulator)})"
            r"\s*:\s*f32", inner) if product else None)
        if total is None or re.search(
                rf"scf\.yield\s+{re.escape(total.group(1))}\s*:\s*f32",
                inner) is None:
            continue
        inserted = re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.insert\s+"
            rf"{re.escape(inner_result)}\s+into\s+{re.escape(column_iter)}"
            rf"\[{re.escape(row_loop.induction)},\s*"
            rf"{re.escape(column_loop.induction)}\]", column_body)
        if inserted is None or re.search(
                rf"affine\.yield\s+{re.escape(inserted.group(1))}\s*:",
                column_body) is None:
            continue

        tensors = [rowptr_loads[0][1], index.group(2), value.group(2),
                   dense.group(2), output_init]
        sources = [_to_tensor_memref_source(text, tensor, row_loop.span[0])
                   for tensor in tensors]
        if any(source is None for source in sources):
            continue
        operands = [source[0] for source in sources]
        types = [source[1] for source in sources]
        if (len(set(operands)) != 5 or
                [_sniff_elem_type(ty) for ty in types] !=
                ["i32", "i32", "f32", "f32", "f32"] or
                [_shaped_rank(ty) for ty in types] != [1, 1, 1, 2, 2] or
                any("," in ty for ty in types)):
            continue
        dense_type = re.fullmatch(r"memref<\?x(\d+)xf32>", types[3])
        output_type = re.fullmatch(r"memref<\?x(\d+)xf32>", types[4])
        if (dense_type is None or output_type is None or
                int(dense_type.group(1)) != column_count or
                int(output_type.group(1)) != column_count):
            continue
        tail = re.match(
            rf"\s*(%[\w.$-]+)\s*=\s*bufferization\.to_memref\s+"
            rf"{re.escape(row_result)}\s*:\s*memref<[^>]*xf32>\s*\n"
            rf"\s*memref\.copy\s+\1,\s*{re.escape(operands[4])}\s*:\s*[^\n]*",
            text[row_loop.span[1]:])
        if tail is None:
            continue

        uid = row_loop.span[0]
        rows = f"%cusparse_spmm_rows_{uid}"
        target_types = ["memref<?xi32>", "memref<?xi32>",
                        "memref<?xf32>", "memref<?x?xf32>",
                        "memref<?x?xf32>"]
        indent = re.match(r"\s*", text[row_line_start:row_loop.span[0]]).group(0)
        lines = [f"{rows} = arith.constant {row_count} : index"]
        normalized: list[str] = []
        for i, (operand, source_type, target_type) in enumerate(
                zip(operands, types, target_types)):
            if source_type != target_type:
                cast = f"%cusparse_spmm_arg_{uid}_{i}"
                lines.append(
                    f"{cast} = memref.cast {operand} : {source_type} to {target_type}")
                normalized.append(cast)
            else:
                normalized.append(operand)
        symbol = "cusparseSpMM_CSR_f32_memref"
        lines.append(
            f"kernel.launch @{symbol}(" + ", ".join([rows, *normalized]) +
            ") : (index, " + ", ".join(target_types) + ") -> ()")
        replacement = ("\n" + indent).join(lines)
        rendered.append((row_line_start, row_loop.span[1] + tail.end(),
                         indent + replacement, symbol))
    return rendered


def _render_cusparse_coo_spmm(
        text: str) -> list[tuple[int, int, str, str]]:
    """Fuse zero-fill plus an indexed COO update nest into cuSPARSE SpMM."""
    loops = parse_loops(text)
    rendered: list[tuple[int, int, str, str]] = []
    for outer in loops:
        children = [loop for loop in loops
                    if outer.span[0] < loop.span[0] and
                    loop.span[1] < outer.span[1]]
        if len(children) != 1:
            continue
        inner_loop = children[0]
        outer_line = text.rfind("\n", 0, outer.span[0]) + 1
        outer_assignment = re.match(
            r"\s*(%[\w.$-]+)\s*=\s*$", text[outer_line:outer.span[0]])
        outer_body = text[outer.span[0]:outer.span[1]]
        outer_header = re.match(
            rf"affine\.for\s+{re.escape(outer.induction)}\s*=\s*"
            r"0\s+to\s+(\d+)\s+iter_args\s*\(\s*"
            r"(%[\w.$-]+)\s*=\s*(%[\w.$-]+)\s*\)\s*->\s*"
            r"\(tensor<[^>]+>\)\s*\{", outer_body)
        inner_line = text.rfind("\n", outer.span[0], inner_loop.span[0]) + 1
        inner_assignment = re.match(
            r"\s*(%[\w.$-]+)\s*=\s*$", text[inner_line:inner_loop.span[0]])
        inner_body = text[inner_loop.span[0]:inner_loop.span[1]]
        inner_header = re.match(
            rf"affine\.for\s+{re.escape(inner_loop.induction)}\s*=\s*"
            r"0\s+to\s+(\d+)\s+iter_args\s*\(\s*"
            r"(%[\w.$-]+)\s*=\s*(%[\w.$-]+)\s*\)\s*->\s*"
            r"\(tensor<[^>]+>\)\s*\{", inner_body)
        if not all((outer_assignment, outer_header,
                    inner_assignment, inner_header)):
            continue
        outer_result = outer_assignment.group(1)
        nnz = int(outer_header.group(1))
        outer_iter, initialized_output = outer_header.group(2), outer_header.group(3)
        inner_result = inner_assignment.group(1)
        columns = int(inner_header.group(1))
        inner_iter, inner_init = inner_header.group(2), inner_header.group(3)
        if (nnz <= 0 or columns <= 0 or inner_init != outer_iter or
                re.search(rf"affine\.yield\s+{re.escape(inner_result)}\s*:",
                          outer_body) is None):
            continue

        function_start = text.rfind("func.func", 0, outer.span[0])
        preamble = text[function_start:outer.span[0]]
        inserted_slice = re.search(
            rf"{re.escape(initialized_output)}\s*=\s*tensor\.insert_slice\s+"
            r"(%[\w.$-]+)\s+into\s+(%[\w.$-]+)"
            r"\[0,\s*0\]\s*\[([^,\]]+),\s*([^\]]+)\]\s*"
            r"\[1,\s*1\]", preamble)
        region_start = None
        if inserted_slice is not None:
            fill_result = inserted_slice.group(1)
            output_tensor = inserted_slice.group(2)
            row_size, column_size = (inserted_slice.group(3).strip(),
                                     inserted_slice.group(4).strip())
            extracted_slice = re.search(
                rf"(%[\w.$-]+)\s*=\s*tensor\.extract_slice\s+"
                rf"{re.escape(output_tensor)}\[0,\s*0\]\s*"
                rf"\[{re.escape(row_size)},\s*{re.escape(column_size)}\]\s*"
                r"\[1,\s*1\]", preamble)
            if extracted_slice is None:
                continue
            extracted_tensor = extracted_slice.group(1)
            region_start = extracted_slice.start()
        else:
            # The production raising pipeline preserves Polygeist submaps.
            # Accept the equivalent full-prefix view / inverse pair, but only
            # with the same base tensor, extents, and identity access map.
            inverse = re.search(
                rf"{re.escape(initialized_output)}\s*=\s*"
                rf"polygeist\.submapInverse\(\s*(%[\w.$-]+),\s*"
                rf"(%[\w.$-]+),\s*([^,]+),\s*([^\)]+)\)\s*"
                r"\{\s*map\s*=\s*([^}]+)\}", preamble)
            if inverse is None:
                continue
            output_tensor, fill_result = inverse.group(1), inverse.group(2)
            row_size, column_size = (inverse.group(3).strip(),
                                     inverse.group(4).strip())
            submap = re.search(
                rf"(%[\w.$-]+)\s*=\s*polygeist\.submap\(\s*"
                rf"{re.escape(output_tensor)},\s*{re.escape(row_size)},\s*"
                rf"{re.escape(column_size)}\s*\)\s*"
                rf"\{{\s*map\s*=\s*{re.escape(inverse.group(5).strip())}\s*\}}",
                preamble)
            if submap is None:
                continue
            extracted_tensor = submap.group(1)
            region_start = submap.start()
        fill = re.search(
            rf"{re.escape(fill_result)}\s*=\s*linalg\.generic\s*"
            r"\{[^}]*iterator_types\s*=\s*\[\"parallel\",\s*\"parallel\"\]"
            rf"[^}}]*\}}\s*outs\(\s*{re.escape(extracted_tensor)}\s*:"
            r"\s*tensor<[^>]+>\)\s*\{.*?linalg\.yield\s+(%[\w.$-]+)\s*:"
            r"\s*f32\s*\}\s*->\s*tensor<[^>]+>", preamble, re.DOTALL)
        if fill is None or re.search(
                rf"{re.escape(fill.group(1))}\s*=\s*arith\.constant\s+"
                r"(?:0(?:\.0+)?(?:e[+-]?0+)?|0x0+)\s*:\s*f32",
                preamble, re.IGNORECASE) is None:
            continue

        def constant_extent(value: str) -> int | None:
            if value.isdigit():
                return int(value)
            match = re.search(
                rf"{re.escape(value)}\s*=\s*arith\.constant\s+(\d+)\s*:\s*index",
                preamble)
            return int(match.group(1)) if match else None

        rows = constant_extent(row_size)
        if rows is None or rows <= 0 or constant_extent(column_size) != columns:
            continue

        p = re.escape(outer.induction)
        c = re.escape(inner_loop.induction)
        index_loads = re.findall(
            rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            rf"(%[\w.$-]+)\[{p}\]\s*:\s*tensor<[^>]*xi32>", inner_body)
        if len(index_loads) != 2 or index_loads[0][1] == index_loads[1][1]:
            continue
        casts = {}
        for loaded, tensor in index_loads:
            cast = re.search(
                rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
                rf"{re.escape(loaded)}\s*:\s*i32\s+to\s+index", inner_body)
            if cast:
                casts[cast.group(1)] = tensor
        if len(casts) != 2:
            continue
        dense = None
        dense_index = None
        for cast, tensor in casts.items():
            match = re.search(
                rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
                rf"(%[\w.$-]+)\[{re.escape(cast)},\s*{c}\]\s*:"
                r"\s*tensor<[^>]*xf32>", inner_body)
            if match and match.group(2) != inner_iter:
                dense, dense_index = match, cast
                break
        if dense is None:
            continue
        row_index = next((cast for cast in casts if cast != dense_index), None)
        value = re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            rf"(%[\w.$-]+)\[{p}\]\s*:\s*tensor<[^>]*xf32>", inner_body)
        old_output = re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            rf"{re.escape(inner_iter)}\[{re.escape(row_index)},\s*{c}\]",
            inner_body) if row_index else None
        if value is None or old_output is None:
            continue
        product = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.mulf\s+"
            rf"(?:{re.escape(value.group(1))},\s*{re.escape(dense.group(1))}|"
            rf"{re.escape(dense.group(1))},\s*{re.escape(value.group(1))})"
            r"\s*:\s*f32", inner_body)
        total = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addf\s+"
            rf"(?:{re.escape(old_output.group(1))},\s*{re.escape(product.group(1))}|"
            rf"{re.escape(product.group(1))},\s*{re.escape(old_output.group(1))})"
            r"\s*:\s*f32", inner_body) if product else None)
        inserted = (re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.insert\s+"
            rf"{re.escape(total.group(1))}\s+into\s+{re.escape(inner_iter)}"
            rf"\[{re.escape(row_index)},\s*{c}\]", inner_body)
                    if total and row_index else None)
        if inserted is None or re.search(
                rf"affine\.yield\s+{re.escape(inserted.group(1))}\s*:",
                inner_body) is None:
            continue

        row_tensor = casts[row_index]
        column_tensor = casts[dense_index]
        tensors = [row_tensor, column_tensor, value.group(2),
                   dense.group(2), output_tensor]
        sources = [_to_tensor_memref_source(text, tensor, outer.span[0])
                   for tensor in tensors]
        if any(source is None for source in sources):
            continue
        operands = [source[0] for source in sources]
        types = [source[1] for source in sources]
        if (len(set(operands)) != 5 or
                [_sniff_elem_type(ty) for ty in types] !=
                ["i32", "i32", "f32", "f32", "f32"] or
                [_shaped_rank(ty) for ty in types] != [1, 1, 1, 2, 2] or
                any("," in ty for ty in types)):
            continue
        dense_type = re.fullmatch(r"memref<\?x(\d+)xf32>", types[3])
        output_type = re.fullmatch(r"memref<\?x(\d+)xf32>", types[4])
        if (dense_type is None or output_type is None or
                int(dense_type.group(1)) != columns or
                int(output_type.group(1)) != columns):
            continue
        tail = re.match(
            rf"\s*(%[\w.$-]+)\s*=\s*bufferization\.to_memref\s+"
            rf"{re.escape(outer_result)}\s*:\s*memref<[^>]*xf32>\s*\n"
            rf"\s*memref\.copy\s+\1,\s*{re.escape(operands[4])}\s*:\s*[^\n]*",
            text[outer.span[1]:])
        if tail is None:
            continue

        slice_line = text.rfind("\n", 0, function_start + region_start) + 1
        uid = outer.span[0]
        row_value = f"%cusparse_coo_rows_{uid}"
        nnz_value = f"%cusparse_coo_nnz_{uid}"
        target_types = ["memref<?xi32>", "memref<?xi32>",
                        "memref<?xf32>", "memref<?x?xf32>",
                        "memref<?x?xf32>"]
        indent = re.match(r"\s*", text[slice_line:function_start +
                                         region_start]).group(0)
        lines = [f"{row_value} = arith.constant {rows} : index",
                 f"{nnz_value} = arith.constant {nnz} : index"]
        normalized: list[str] = []
        for i, (operand, source_type, target_type) in enumerate(
                zip(operands, types, target_types)):
            if source_type != target_type:
                cast = f"%cusparse_coo_arg_{uid}_{i}"
                lines.append(
                    f"{cast} = memref.cast {operand} : {source_type} to {target_type}")
                normalized.append(cast)
            else:
                normalized.append(operand)
        symbol = "cusparseSpMM_COO_f32_memref"
        lines.append(
            f"kernel.launch @{symbol}(" +
            ", ".join([row_value, nnz_value, *normalized]) +
            ") : (index, index, " + ", ".join(target_types) + ") -> ()")
        replacement = ("\n" + indent).join(lines)
        rendered.append((slice_line, outer.span[1] + tail.end(),
                         indent + replacement, symbol))
    return rendered


def _render_cusparse_jds_spmv(
        text: str) -> list[tuple[int, int, str, str]]:
    """Replace a proven repeated JDS SpMV with a cuSPARSE adapter call.

    cuSPARSE has no JDS descriptor.  The runtime ABI therefore receives the
    source JDS arrays, converts their storage metadata to CSR, and performs
    the numerical operation with cuSPARSE.  Keep the recognition deliberately
    strict: the Parboil loop must load one row count, traverse
    ``offset[k] + row``, gather ``x[column]``, accumulate value*x, and scatter
    through the JDS row permutation.
    """
    loops = parse_loops(text)
    rendered: list[tuple[int, int, str, str]] = []
    seen: set[tuple[int, int]] = set()
    for candidate in analyze_residual_loops(text):
        if candidate.kind != "jds_spmv":
            continue
        parents = [loop for loop in loops
                   if loop.span[0] < candidate.loop.span[0]
                   and candidate.loop.span[1] < loop.span[1]]
        if len(parents) < 2:
            continue
        row_loop = max(parents, key=lambda loop: loop.span[0])
        repeat_parents = [loop for loop in parents if loop is not row_loop]
        repeat_loop = max(repeat_parents, key=lambda loop: loop.span[0])
        if repeat_loop.span in seen:
            continue
        row_bound = re.fullmatch(
            r"(?:0|%c0(?:_[\w.$-]+)?) to (%[\w.$-]+)"
            r"(?: step %c1(?:_[\w.$-]+)?)?", row_loop.bounds)
        repeat_bound = re.fullmatch(r"0 to ([1-9][0-9]*)", repeat_loop.bounds)
        if not row_bound or not repeat_bound:
            continue
        rows = row_bound.group(1)
        row_iv = re.escape(row_loop.induction)
        k_iv = re.escape(candidate.loop.induction)
        before_inner = text[row_loop.span[0]:candidate.loop.span[0]]
        inner = text[candidate.loop.span[0]:candidate.loop.span[1]]
        after_inner = text[candidate.loop.span[1]:row_loop.span[1]]

        count = re.search(
            rf"(%[\w.$-]+)\s*=\s*(?:affine|memref)\.load\s+"
            rf"(%[\w.$-]+)\[{row_iv}\]\s*:\s*(memref<[^>]*xi32>)",
            before_inner)
        if not count:
            continue
        upper = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
            rf"{re.escape(count.group(1))}\s*:\s*i32\s+to\s+index",
            before_inner)
        if not upper or upper.group(1) not in candidate.loop.bounds:
            continue
        row32 = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+{row_iv}\s*:\s*"
            r"index\s+to\s+i32", before_inner + inner)
        offset = re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+(%[\w.$-]+)\[{k_iv}\]"
            r"\s*:\s*(memref<[^>]*xi32>)", inner)
        if not row32 or not offset:
            continue
        position32 = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addi\s+"
            rf"(?:{re.escape(offset.group(1))},\s*{re.escape(row32.group(1))}|"
            rf"{re.escape(row32.group(1))},\s*{re.escape(offset.group(1))})"
            r"\s*:\s*i32", inner)
        if not position32:
            continue
        position = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
            rf"{re.escape(position32.group(1))}\s*:\s*i32\s+to\s+index", inner)
        if not position:
            continue
        pos = re.escape(position.group(1))
        column = re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+(%[\w.$-]+)\[{pos}\]"
            r"\s*:\s*(memref<[^>]*xi32>)", inner)
        value = re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+(%[\w.$-]+)\[{pos}\]"
            r"\s*:\s*(memref<[^>]*x(f32|f64)>)", inner)
        if not column or not value or value.group(4) != "f32":
            continue
        column_index = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
            rf"{re.escape(column.group(1))}\s*:\s*i32\s+to\s+index", inner)
        if not column_index:
            continue
        xvalue = re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+(%[\w.$-]+)"
            rf"\[{re.escape(column_index.group(1))}\]\s*:\s*"
            r"(memref<[^>]*xf32>)", inner)
        if not xvalue:
            continue
        product = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.mulf\s+"
            rf"(?:{re.escape(value.group(1))},\s*{re.escape(xvalue.group(1))}|"
            rf"{re.escape(xvalue.group(1))},\s*{re.escape(value.group(1))})"
            r"\s*:\s*f32", inner)
        if not product or not re.search(
                rf"arith\.addf\s+[^\n]*{re.escape(product.group(1))}[^\n]*"
                r":\s*f32", inner):
            continue
        loop_prefix = text[max(row_loop.span[0], candidate.loop.span[0] - 96):
                           candidate.loop.span[0]]
        result_match = re.search(r"(%[\w.$-]+)\s*=\s*$", loop_prefix)
        if not result_match:
            continue
        result = result_match.group(1)
        permutation = re.search(
            rf"(%[\w.$-]+)\s*=\s*(?:affine|memref)\.load\s+"
            rf"(%[\w.$-]+)\[{row_iv}\]\s*:\s*(memref<[^>]*xi32>)",
            after_inner)
        if not permutation:
            continue
        perm_index = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
            rf"{re.escape(permutation.group(1))}\s*:\s*i32\s+to\s+index",
            after_inner)
        if not perm_index:
            continue
        output = re.search(
            rf"memref\.store\s+{re.escape(result)},\s*(%[\w.$-]+)"
            rf"\[{re.escape(perm_index.group(1))}\]\s*:\s*(memref<[^>]*xf32>)",
            after_inner)
        if not output:
            continue

        roots = [count.group(2), offset.group(2), column.group(2),
                 value.group(2), permutation.group(2), xvalue.group(2),
                 output.group(1)]
        source_types = [count.group(3), offset.group(3), column.group(3),
                        value.group(3), permutation.group(3), xvalue.group(3),
                        output.group(2)]
        targets = ["memref<?xi32>", "memref<?xi32>", "memref<?xi32>",
                   "memref<?xf32>", "memref<?xi32>", "memref<?xf32>",
                   "memref<?xf32>"]
        uid = repeat_loop.span[0]
        prefix_lines = [
            f"%cusparse_repeats_{uid} = arith.constant {repeat_bound.group(1)} : index"]
        normalized: list[str] = []
        for index, (operand, source, target) in enumerate(
                zip(roots, source_types, targets)):
            if source != target:
                cast = f"%cusparse_jds_arg_{uid}_{index}"
                prefix_lines.append(
                    f"{cast} = memref.cast {operand} : {source} to {target}")
                normalized.append(cast)
            else:
                normalized.append(operand)
        symbol = "cusparseSpMV_JDS_f32_memref"
        operands = [rows, f"%cusparse_repeats_{uid}", *normalized]
        operand_types = ["index", "index", *targets]
        launch = (f"kernel.launch @{symbol}(" + ", ".join(operands) +
                  ") : (" + ", ".join(operand_types) + ") -> ()")
        line_start = text.rfind("\n", 0, repeat_loop.span[0]) + 1
        indent = text[line_start:repeat_loop.span[0]]
        replacement = "\n".join(
            f"{indent}{line}" for line in [*prefix_lines, launch])
        rendered.append((repeat_loop.span[0], repeat_loop.span[1], replacement,
                         symbol))
        seen.add(repeat_loop.span)
    return rendered


def _render_dense_factorization_regions(
        text: str, instances) -> list[tuple[int, int, str, str, list[int]]]:
    """Recognize whole sequential algorithms hidden around Linalg reductions.

    This intentionally starts with the two unambiguous PolyBench shapes.  A
    library launch is emitted only after checking the surrounding recurrence,
    diagonal access, update arithmetic, element type, and function operands;
    recognizing an isolated dot-product generic would not be sufficient.
    """
    rendered: list[tuple[int, int, str, str, list[int]]] = []
    loops = sorted(parse_loops(text),
                   key=lambda loop: loop.span[1] - loop.span[0], reverse=True)
    claimed: list[tuple[int, int]] = []
    for loop in loops:
        if any(start <= loop.span[0] and loop.span[1] <= end
               for start, end in claimed):
            continue
        args = _enclosing_func_args(text, loop.span[0])
        body = text[loop.span[0]:loop.span[1]]
        prefix_start = text.rfind("func.func", 0, loop.span[0])
        prefix = text[prefix_start:loop.span[0]] if prefix_start >= 0 else ""
        if not args or not re.fullmatch(r"0 to (%[\w.$-]+)", loop.bounds):
            continue
        upper = re.fullmatch(r"0 to (%[\w.$-]+)", loop.bounds).group(1)
        n_arg = args[0][0]
        if args[0][1] != "i32" or not re.search(
                rf"{re.escape(upper)}\s*=\s*arith\.index_cast\s+"
                rf"{re.escape(n_arg)}\s*:\s*i32\s+to\s+index", prefix):
            continue
        iv = re.escape(loop.induction)
        line_start = text.rfind("\n", 0, loop.span[0]) + 1
        indent = text[line_start:loop.span[0]]
        consumed = [i for i, inst in enumerate(instances)
                    if loop.span[0] <= inst.span[0] and
                    inst.span[1] <= loop.span[1]]

        rhs_direct_read = bool(re.search(
            rf"(?:affine|memref)\.load\s+{re.escape(args[3][0])}"
            rf"\[{iv}\]", body)) if len(args) == 4 else False
        rhs_tensor_read = False
        if len(args) == 4:
            rhs_tensor = re.search(
                rf"(%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+"
                rf"{re.escape(args[3][0])}\b", prefix)
            rhs_tensor_read = bool(
                rhs_tensor and re.search(
                    rf"tensor\.extract\s+{re.escape(rhs_tensor.group(1))}"
                    rf"\[{iv}\]", body))

        # Forward substitution: x[i]=b[i]; x[i]-=A[i,j]*x[j];
        # x[i]/=A[i,i].  The loop-carried recurrence is the evidence for
        # DTRSV, not a reason to leave the algorithm unmatched.
        matrix_is_dynamic_f64 = bool(re.fullmatch(
            r"memref<\?x(?:\?|[1-9][0-9]*)xf64>", args[1][1])) \
            if len(args) >= 2 else False
        matrix_operand = args[1][0] if len(args) >= 2 else ""
        matrix_prefix: list[str] = []
        if matrix_is_dynamic_f64 and args[1][1] != "memref<?x?xf64>":
            matrix_operand = f"%factor_matrix_{loop.span[0]}"
            matrix_prefix.append(
                f"{indent}{matrix_operand} = memref.cast {args[1][0]} : "
                f"{args[1][1]} to memref<?x?xf64>")

        if (len(args) == 4 and matrix_is_dynamic_f64 and
                args[2][1] == "memref<?xf64>" and
                args[3][1] == "memref<?xf64>" and
                body.count("linalg.generic") == 1 and
                'iterator_types = ["reduction"]' in body and
                "arith.mulf" in body and "arith.subf" in body and
                "arith.divf" in body and
                re.search(rf"arith\.cmpi\s+slt,\s*%[\w.$-]+,\s*{iv}", body) and
                re.search(rf"(?:affine|memref)\.load\s+{re.escape(args[1][0])}"
                          rf"\[{iv},\s*{iv}\]", body) and
                (rhs_direct_read or rhs_tensor_read) and
                re.search(rf"(?:affine|memref)\.store\s+%[\w.$-]+,\s*"
                          rf"{re.escape(args[2][0])}\[{iv}\]", body)):
            symbol = "cublasDtrsvLowerRowMajor_memref"
            launch = "\n".join(matrix_prefix + [
                f"{indent}kernel.launch @{symbol}("
                f"{matrix_operand}, {args[3][0]}, {args[2][0]}) : "
                "(memref<?x?xf64>, memref<?xf64>, memref<?xf64>) -> ()"])
            rendered.append((loop.span[0], loop.span[1], launch, symbol,
                             consumed))
            claimed.append(loop.span)
            continue

        # Unblocked lower Cholesky: an off-diagonal dot/subtract/divide
        # recurrence followed by a diagonal sum-of-squares and sqrt.
        if (len(args) == 2 and matrix_is_dynamic_f64 and
                body.count("linalg.generic") == 2 and
                body.count('iterator_types = ["reduction"]') == 2 and
                body.count("arith.mulf") >= 2 and
                body.count("arith.subf") >= 2 and
                "arith.divf" in body and "math.sqrt" in body and
                re.search(rf"(?:affine|memref)\.load\s+{re.escape(args[1][0])}"
                          rf"\[{iv},\s*{iv}\]", body) and
                re.search(rf"(?:affine|memref)\.store\s+%[\w.$-]+,\s*"
                          rf"{re.escape(args[1][0])}\[{iv},\s*{iv}\]", body)):
            symbol = "cusolverDnDpotrfLowerRowMajor_memref"
            launch = (
                "\n".join(matrix_prefix + [
                f"{indent}kernel.launch @{symbol}({matrix_operand}) : "
                "(memref<?x?xf64>) -> ()"]))
            rendered.append((loop.span[0], loop.span[1], launch, symbol,
                             consumed))
            claimed.append(loop.span)
    return rendered


def _render_zeroed_i32_histograms(
        text: str) -> list[tuple[list[tuple[int, int, str]], str]]:
    """Lower a zero-fill followed by a direct integer-bin count to CUB.

    The zero-fill proof matters because DeviceHistogram overwrites its output,
    whereas a general imperative read/add/write histogram may accumulate into
    nonzero state.  The initial implementation also requires both loops to
    cover their complete statically sized buffers.
    """
    loops = parse_loops(text)
    generics = collect_generics_with_spans(text)
    generic_bodies = parse_generics(text, parse_constants(text))
    results: list[tuple[list[tuple[int, int, str]], str]] = []

    # Joint-root debufferization represents ATen embedding-bag counts as a
    # zeroing generic followed by a tensor-carried indirect increment loop.
    # Recognize that complete composition before the legacy memref form below.
    for count_loop in loops:
        if count_loop.kind != "affine.for":
            continue
        function_start = text.rfind("func.func", 0, count_loop.span[0])
        function_header = text[function_start:text.find("\n", function_start)]
        signature = re.search(
            r"@aten_embedding_bag_counts_cpu\s*\(\s*"
            r"(%[\w.$-]+)\s*:\s*memref<\?xi32>\s*,\s*"
            r"(%[\w.$-]+)\s*:\s*memref<\?xi32>\s*\)", function_header)
        bound = re.fullmatch(
            r"0 to (\d+)\s+iter_args\([^\)]*\)\s*->\s*"
            r"\(tensor<\?xi32>\)", count_loop.bounds)
        if signature is None or bound is None or int(bound.group(1)) <= 0:
            continue
        samples, histogram = signature.groups()
        count = int(bound.group(1))
        prefix = text[function_start:count_loop.span[0]]
        sample_tensor = re.search(
            rf"(%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+"
            rf"{re.escape(samples)}\s*:\s*memref<\?xi32>", prefix)
        histogram_tensor = re.search(
            rf"(%[\w.$-]+)\s*=\s*bufferization\.to_tensor\s+"
            rf"{re.escape(histogram)}\s*:\s*memref<\?xi32>", prefix)
        header = re.match(
            rf"affine\.for\s+{re.escape(count_loop.induction)}\s*=\s*"
            rf"0\s+to\s+{count}\s+iter_args\("
            r"(%[\w.$-]+)\s*=\s*(%[\w.$-]+)\)\s*->\s*"
            r"\(tensor<\?xi32>\)",
            text[count_loop.span[0]:])
        if sample_tensor is None or histogram_tensor is None or header is None:
            continue
        carried, loop_initial = header.groups()
        zero_result = loop_initial
        inverse = re.search(
            rf"{re.escape(loop_initial)}\s*=\s*"
            rf"polygeist\.submapInverse\("
            rf"{re.escape(histogram_tensor.group(1))},\s*"
            r"(%[\w.$-]+),\s*(%[\w.$-]+)\)\s*"
            r"[{]map\s*=\s*([^}]+)[}]", prefix)
        bin_extent_ssa = None
        if inverse:
            zero_result = inverse.group(1)
            bin_extent_ssa = inverse.group(2)
            if (_compact_affine_map(_resolve_affine_map_text(
                    text[:count_loop.span[0]], inverse.group(3).strip())) !=
                    "affine_map<(d0)->(d0)>"):
                continue
        zero_index = next((index for index, generic in enumerate(generics)
                           if generic.result_ssa == zero_result and
                           generic.span[1] < count_loop.span[0]), None)
        if zero_index is None or zero_index >= len(generic_bodies):
            continue
        zero_generic = generics[zero_index]
        zero_body = generic_bodies[zero_index]
        zero_outs = _extract_ssa_names(zero_generic.outs_part)
        constants = parse_constants(text[:zero_generic.span[0]])
        zero_output_is_complete = zero_outs == [histogram_tensor.group(1)]
        if not zero_output_is_complete and len(zero_outs) == 1:
            zero_view = re.search(
                rf"{re.escape(zero_outs[0])}\s*=\s*polygeist\.submap\("
                rf"{re.escape(histogram_tensor.group(1))},\s*"
                r"(%[\w.$-]+)\)\s*[{]map\s*=\s*([^}]+)[}]", prefix)
            if (zero_view and
                    _compact_affine_map(_resolve_affine_map_text(
                        text[:zero_generic.span[0]],
                        zero_view.group(2).strip())) ==
                    "affine_map<(d0)->(d0)>" and
                    (bin_extent_ssa is None or
                     bin_extent_ssa == zero_view.group(1))):
                zero_output_is_complete = True
                bin_extent_ssa = zero_view.group(1)
        if (not zero_output_is_complete or
                _extract_ssa_names(zero_generic.ins_part) or
                zero_body.iterator_types != ["parallel"] or
                len(zero_body.yield_values) != 1 or
                constants.get(zero_body.yield_values[0]) != 0.0):
            continue

        loop_text = text[count_loop.span[0]:count_loop.span[1]]
        iv = re.escape(count_loop.induction)
        sample = re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            rf"{re.escape(sample_tensor.group(1))}\[{iv}\]\s*:\s*"
            r"tensor<\?xi32>", loop_text)
        bin_index = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
            rf"{re.escape(sample.group(1))}\s*:\s*i32\s+to\s+index",
            loop_text) if sample else None)
        old = (re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.extract\s+"
            rf"{re.escape(carried)}\[{re.escape(bin_index.group(1))}\]"
            r"\s*:\s*tensor<\?xi32>", loop_text) if bin_index else None)
        add = (re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addi\s+"
            rf"{re.escape(old.group(1))},\s*(%[\w.$-]+)\s*:\s*i32",
            loop_text) if old else None)
        one = add.group(2) if add else None
        inserted = (re.search(
            rf"(%[\w.$-]+)\s*=\s*tensor\.insert\s+"
            rf"{re.escape(add.group(1))}\s+into\s+{re.escape(carried)}"
            rf"\[{re.escape(bin_index.group(1))}\]\s*:\s*tensor<\?xi32>",
            loop_text) if add else None)
        if (inserted is None or one is None or not re.search(
                rf"{re.escape(one)}\s*=\s*arith\.constant\s+1\s*:\s*i32",
                prefix) or not re.search(
                    rf"affine\.yield\s+{re.escape(inserted.group(1))}\s*:\s*"
                    r"tensor<\?xi32>", loop_text)):
            continue
        writes = re.findall(r"tensor\.insert\b", loop_text)
        if len(writes) != 1:
            continue

        function_line_end = text.find("\n", function_start)
        function_opening = text.rfind("{", function_start, function_line_end)
        function_end = _matching_brace(text, function_opening)
        if function_end is None:
            continue
        tail = text[count_loop.span[1]:function_end]
        loop_line_start = text.rfind("\n", 0, count_loop.span[0]) + 1
        loop_result_match = re.search(
            r"(%[\w.$-]+)\s*=\s*$",
            text[loop_line_start:count_loop.span[0]])
        to_memref = re.search(
            rf"(%[\w.$-]+)\s*=\s*bufferization\.to_memref\s+"
            rf"{re.escape(loop_result_match.group(1))}\s*:\s*"
            r"memref<\?xi32>", tail) if loop_result_match else None
        copy = (re.search(
            rf"memref\.copy\s+{re.escape(to_memref.group(1))},\s*"
            rf"{re.escape(histogram)}\s*:\s*memref<\?xi32>\s+to\s+"
            r"memref<\?xi32>", tail) if to_memref else None)
        if copy is None:
            continue
        bin_count = None
        if bin_extent_ssa is not None:
            value = constants.get(bin_extent_ssa)
            if (value is None or int(value) != value or value <= 0 or
                    value > 2147483647):
                continue
            bin_count = int(value)
        start = text.rfind("\n", 0, zero_generic.span[0]) + 1
        indent = re.match(r"\s*", text[start:zero_generic.span[0]]).group(0)
        zero_shift = f"%histogram_shift_{count_loop.span[0]}"
        fixed_extents = (f"{count}, {bin_count}" if bin_count is not None
                         else f"{count}")
        replacement = (
            f"{indent}{zero_shift} = arith.constant 0 : i32\n"
            f"{indent}kernel.launch @cubHistogramEvenI32ShiftZero_memref("
            f"{samples}, {histogram}, {zero_shift}) "
            f"{{polygeist.fixed_extents = array<i64: {fixed_extents}>}} : "
            f"(memref<?xi32>, memref<?xi32>, i32) -> ()")
        results.append(([(start, count_loop.span[1] + copy.end(), replacement)],
                        "cubHistogramEvenI32ShiftZero_memref"))

    for count_loop in loops:
        count_body = text[count_loop.span[0]:count_loop.span[1]]
        # Replacing the entire loop is valid only for a pure histogram update.
        # In particular, reject fused scatter/output writes that merely happen
        # to contain the same load-add-store subsequence.
        if (count_body.count("affine.for") + count_body.count("scf.for") != 1 or
                re.search(r"\b(?:func\.call|kernel\.launch|linalg\.)\b",
                          count_body)):
            continue
        iv = re.escape(count_loop.induction)
        sample = re.search(
            rf"(%[\w.$-]+)\s*=\s*(?:affine|memref)\.load\s+"
            rf"(%[\w.$-]+)\[{iv}\]\s*:\s*(memref<([^>]+)>)",
            count_body)
        if not sample or not sample.group(4).endswith("xi32"):
            continue
        sample_value, samples, samples_type = sample.group(1, 2, 3)
        binned_value = sample_value
        right_shift = None
        shift = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.shrsi\s+"
            rf"{re.escape(sample_value)},\s*(%[\w.$-]+)\s*:\s*i32",
            count_body)
        if shift:
            binned_value, right_shift = shift.group(1), shift.group(2)
        cast = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.index_cast\s+"
            rf"{re.escape(binned_value)}\s*:\s*i32\s+to\s+index",
            count_body)
        if not cast:
            continue
        bin_index = cast.group(1)
        old = re.search(
            rf"(%[\w.$-]+)\s*=\s*memref\.load\s+(%[\w.$-]+)"
            rf"\[{re.escape(bin_index)}\]\s*:\s*(memref<([^>]+)>)",
            count_body)
        if not old or not old.group(4).endswith("xi32"):
            continue
        old_value, histogram, histogram_type = old.group(1, 2, 3)
        add = re.search(
            rf"(%[\w.$-]+)\s*=\s*arith\.addi\s+"
            rf"(?:{re.escape(old_value)},\s*(%[\w.$-]+)|"
            rf"(%[\w.$-]+),\s*{re.escape(old_value)})\s*:\s*i32",
            count_body)
        if not add:
            continue
        increment = add.group(2) or add.group(3)
        prefix = text[text.rfind("func.func", 0, count_loop.span[0]):
                      count_loop.span[0]]
        if not re.search(
                rf"{re.escape(increment)}\s*=\s*arith\.constant\s+1\s*:\s*i32",
                prefix):
            continue
        direct_shift = right_shift is None
        if direct_shift:
            # Materialize the direct-histogram case with a zero shift.
            right_shift = f"%histogram_shift_{count_loop.span[0]}"
        elif not re.search(
                rf"{re.escape(right_shift)}\s*=\s*arith\.constant\s+"
                r"(?:[0-9]|[12][0-9]|30)\s*:\s*i32", prefix):
            continue
        next_value = add.group(1)
        if not re.search(
                rf"memref\.store\s+{re.escape(next_value)},\s*"
                rf"{re.escape(histogram)}\[{re.escape(bin_index)}\]",
                count_body):
            continue
        writes = re.findall(
            r"(?:affine|memref)\.store\s+%[\w.$-]+,\s*(%[\w.$-]+)\[",
            count_body)
        if writes != [histogram]:
            continue
        reads = re.findall(
            r"(?:affine|memref)\.load\s+(%[\w.$-]+)\[", count_body)
        if len(reads) != 2 or set(reads) != {samples, histogram}:
            continue

        def static_extent(memref_type: str) -> int | None:
            match = re.fullmatch(r"memref<(\d+)xi32>", memref_type)
            return int(match.group(1)) if match else None

        count_extent = static_extent(samples_type)
        bin_extent = static_extent(histogram_type)
        # Pointer arguments arrive as dynamic memrefs even when the complete
        # source loop supplies a constant extent.  Recover that proof from the
        # loop bound and from the complete submap used by the preceding zero
        # fill.  This keeps the rule shape-driven while allowing extracted
        # application routines to use ordinary pointer ABIs.
        if count_extent is None:
            constant_bound = re.fullmatch(r"0 to (\d+)", count_loop.bounds)
            if constant_bound:
                count_extent = int(constant_bound.group(1))
        function_start = text.rfind("func.func", 0, count_loop.span[0])
        before_count = text[function_start:count_loop.span[0]]
        if bin_extent is None:
            extent_uses = list(re.finditer(
                rf"%[\w.$-]+\s*=\s*polygeist\.submap\("
                rf"{re.escape(histogram)},\s*(%[\w.$-]+)\)",
                before_count))
            if extent_uses:
                bin_extent = _constant_index_value(
                    before_count, extent_uses[-1].group(1))
        if count_extent is None or bin_extent is None:
            continue
        if count_loop.bounds != f"0 to {count_extent}":
            continue
        zero_span = None
        for loop in loops:
            if loop.span[1] >= count_loop.span[0]:
                continue
            if text.rfind("func.func", 0, loop.span[0]) != \
                    text.rfind("func.func", 0, count_loop.span[0]):
                continue
            if loop.bounds != f"0 to {bin_extent}":
                continue
            zero_body = text[loop.span[0]:loop.span[1]]
            ziv = re.escape(loop.induction)
            if (zero_body.count("affine.for") + zero_body.count("scf.for") != 1 or
                    re.search(r"\b(?:func\.call|kernel\.launch|linalg\.)\b",
                              zero_body)):
                continue
            zero_writes = re.findall(
                r"(?:affine|memref)\.store\s+%[\w.$-]+,\s*"
                r"(%[\w.$-]+)\[", zero_body)
            if zero_writes != [histogram]:
                continue
            if re.search(
                    rf"(?:affine|memref)\.store\s+(%[\w.$-]+),\s*"
                    rf"{re.escape(histogram)}\[{ziv}\]", zero_body):
                zero_value = re.search(
                    rf"(?:affine|memref)\.store\s+(%[\w.$-]+),\s*"
                    rf"{re.escape(histogram)}\[{ziv}\]", zero_body).group(1)
                before_zero = text[text.rfind("func.func", 0, loop.span[0]):
                                   loop.span[0]]
                if re.search(
                        rf"{re.escape(zero_value)}\s*=\s*arith\.constant\s+0\s*:\s*i32",
                        before_zero):
                    zero_span = loop.span

        # Debufferization commonly raises the full zero-fill loop into a
        # one-output parallel linalg.generic immediately before leaving the
        # data-dependent count loop imperative. Accept that equivalent form
        # after proving the output is a complete identity view of the same
        # static histogram and the body yields only integer zero.
        if zero_span is None:
            function_start = text.rfind("func.func", 0, count_loop.span[0])
            for generic in generics:
                if (generic.span[1] >= count_loop.span[0] or
                        generic.span[0] <= function_start):
                    continue
                generic_text = text[generic.span[0]:generic.span[1]]
                outs = _extract_ssa_names(generic.outs_part)
                if (len(outs) != 1 or
                        _extract_ssa_names(generic.ins_part) or
                        generic_text.count("linalg.generic") != 1 or
                        'iterator_types = ["parallel"]' not in generic_text):
                    continue
                before_generic = text[function_start:generic.span[0]]
                view = outs[0]
                extent_value = None
                if view == histogram:
                    extent_value = str(bin_extent)
                else:
                    view_match = re.search(
                        rf"{re.escape(view)}\s*=\s*polygeist\.submap\("
                        rf"{re.escape(histogram)},\s*(%[\w.$-]+)\)",
                        before_generic)
                    if view_match:
                        extent_value = view_match.group(1)
                if extent_value is None:
                    continue
                if extent_value != str(bin_extent) and not re.search(
                        rf"{re.escape(extent_value)}\s*=\s*arith\.constant\s+"
                        rf"{bin_extent}\s*:\s*index", before_generic):
                    continue
                yielded = re.fullmatch(
                    r"\s*linalg\.generic[^\{]*\{[^\}]*\}\s*outs\([^\)]*\)\s*\{"
                    r"\s*\^bb0\([^\)]*\):\s*linalg\.yield\s+(%[\w.$-]+)\s*"
                    r":\s*i32\s*\}\s*",
                    generic_text, flags=re.DOTALL)
                if not yielded or not re.search(
                        rf"{re.escape(yielded.group(1))}\s*=\s*"
                        r"arith\.constant\s+0\s*:\s*i32", before_generic):
                    continue
                zero_span = generic.span
        if zero_span is None:
            continue
        between = text[zero_span[1]:count_loop.span[0]]
        if re.search(
                rf"(?:affine|memref)\.(?:load|store)\s+[^\n]*"
                rf"{re.escape(histogram)}\[", between):
            continue
        line_start = text.rfind("\n", 0, count_loop.span[0]) + 1
        indent = text[line_start:count_loop.span[0]]
        shift_line = (f"{right_shift} = arith.constant 0 : i32\n{indent}"
                      if direct_shift else "")
        uid = count_loop.span[0]
        samples_cast = f"%histogram_samples_{uid}"
        output_cast = f"%histogram_output_{uid}"
        replacement = (
            f"{indent}{shift_line}{samples_cast} = memref.cast {samples} : "
            f"{samples_type} to memref<?xi32>\n"
            f"{indent}{output_cast} = memref.cast {histogram} : "
            f"{histogram_type} to memref<?xi32>\n"
            f"{indent}kernel.launch @cubHistogramEvenI32ShiftZero_memref("
            f"{samples_cast}, {output_cast}, {right_shift}) : "
            "(memref<?xi32>, memref<?xi32>, i32) -> ()")
        results.append(([(zero_span[0], zero_span[1], ""),
                         (count_loop.span[0], count_loop.span[1], replacement)],
                        "cubHistogramEvenI32ShiftZero_memref"))
    return results


def rewrite_mlir(
    text: str,
    dry_run: bool = False,
    roundtrip_markers: bool = False,
    show_candidates: bool = False,
    show_semantic_only: bool = False,
    max_launches: int | None = None,
    disable_pointwise_matching: bool = False,
    show_structured_regions: bool = False,
    enable_structured_rewrite: bool = False,
    stencil_backend: str = "cudnn",
    disabled_kernels: set[str] | None = None,
    only_kernels: set[str] | None = None,
) -> tuple[str, list[tuple]]:
    """Run the matcher on `text` and return (rewritten_text, match_report).

    match_report: list of (kernel_name_or_None, body_indices, launch_name).

    When `roundtrip_markers` is set, each emitted `kernel.launch` is preceded
    by a comment block holding the original linalg.generic span verbatim,
    bounded by ``// POLYGEIST-MATCH-BEGIN-<name>`` / ``// POLYGEIST-MATCH-END``
    markers. This lets `kernel_launch_lower.py` undo the rewrite for e2e
    correctness testing — see notes/raise_correctness_testing.md.
    """





    consts = parse_constants(text)
    bodies = parse_generics(text, consts)
    instances = collect_generics_with_spans(text)
    scalar_types = _scan_scalar_types(text)
    if len(bodies) != len(instances):
        # Re-parser disagrees with our regex span scanner; bail clean.
        return text, [("warning", None, f"parser drift: {len(bodies)} vs {len(instances)}")]

    body_terms = []
    for b in bodies:
        try:
            body_terms.append(encode_body(b))
        except Exception:
            body_terms.append(None)

    # Per-body form ("tensor" / "memref"), aligned with `instances`.
    # Multi-result tensor generics print as `%x:2 = linalg.generic ...`; the
    # lightweight block regex intentionally starts at `linalg.generic`, so
    # `result_ssa` is absent for that form. Use the trailing result type to
    # classify tensor-vs-memref instead.
    body_forms = [
        "tensor" if (inst.result_type and "tensor<" in inst.result_type)
        else "memref"
        for inst in instances
    ]

    disabled_kernels = disabled_kernels or set()
    semantic_comps = composition_library()
    comps = [
        entry for entry in semantic_comps
        if entry.name not in disabled_kernels and
           (only_kernels is None or entry.name in only_kernels) and
           (entry.name in ABI_LOWERABLE_KERNELS or
            entry.name in COMPOSITION_LOWERING_ADAPTERS)
    ]

    # Walk bodies front-to-back, greedy-match compositions.
    report: list[tuple] = []
    structured_results = (
        analyze_structured_regions(text, instances, bodies, body_terms)
        if show_structured_regions or enable_structured_rewrite else [])
    if show_structured_regions:
        for structured in structured_results:
            report.append((
                "structured_fusion" if structured.fused is not None
                else "structured_reject",
                [op.index for op in structured.region.operations],
                format_result(structured),
            ))
        for candidate in analyze_residual_loops(text):
            report.append((
                "residual_idiom_candidate",
                [],
                format_residual_candidate(candidate),
            ))
    if dry_run and show_candidates:
        for cand_i in range(len(body_terms)):
            for cand in enumerate_semantic_candidates(
                bodies, body_terms, semantic_comps, start=cand_i,
                body_forms=body_forms
            ):
                has_backend = _candidate_backend(cand) is not None
                if not has_backend and not show_semantic_only:
                    continue
                report.append((
                    "kernel_candidate" if has_backend else "semantic_debug",
                    list(cand.body_indices),
                    _format_candidate_for_report(
                        cand, include_semantic_only=show_semantic_only
                    ),
                ))

    edits: list[tuple[int, int, str]] = []   # (start, end, replacement)
    consumed_structured_bodies: set[int] = set()
    if enable_structured_rewrite:
        for structured in structured_results:
            rendered = _render_looped_blas_as_strided_batched(structured, text)
            launch_name = (
                "cublasDgemm_strided_batched_subtract[loop-lifted]"
                if structured.extracted_kind == "looped_gemm_as_batched_gemm"
                else "cublasDgemv_strided_batched_subtract[loop-lifted]"
            )
            if rendered is None:
                rendered = _render_looped_gemv_as_gemm(structured, text)
                launch_name = "cublasDgemm_simple[loop-lifted]"
            if rendered is None:
                rendered = _render_source_faithful_sgemm(structured, text)
                launch_name = "cublasSgemm_flat_colmajor_nt_alpha_beta[loop+epilogue]"
            if rendered is None:
                continue
            edits.append(rendered)
            consumed = [op.index for op in structured.region.operations]
            consumed_structured_bodies.update(consumed)
            report.append(("match", consumed, launch_name))
    emitted_launches = 0
    for start, end, replacement, symbol, consumed in \
            _render_compressed_block_permutation(text):
        if max_launches is not None and emitted_launches >= max_launches:
            report.append(("launch_limit", consumed, symbol))
            continue
        if (symbol in disabled_kernels or
                (only_kernels is not None and symbol not in only_kernels)):
            continue
        edits.append((start, end, replacement))
        report.append(("match", consumed, symbol + "[block-layout]"))
        emitted_launches += 1
    for start, end, replacement, symbol, consumed in \
            _render_dynamic_allany_reduction(
                text, instances, bodies, body_terms, body_forms):
        if max_launches is not None and emitted_launches >= max_launches:
            report.append(("launch_limit", consumed, symbol))
            continue
        if (symbol in disabled_kernels or
                (only_kernels is not None and symbol not in only_kernels)):
            continue
        edits.append((start, end, replacement))
        consumed_structured_bodies.update(consumed)
        report.append(("match", consumed, symbol + "[dynamic-allany]"))
        emitted_launches += 1
    for start, end, replacement, symbol, consumed in \
            _render_rowwise_argreduce(text, instances, bodies, body_forms):
        if max_launches is not None and emitted_launches >= max_launches:
            report.append(("launch_limit", consumed, symbol))
            continue
        if (symbol in disabled_kernels or
                (only_kernels is not None and symbol not in only_kernels)):
            continue
        edits.append((start, end, replacement))
        consumed_structured_bodies.update(consumed)
        report.append(("match", consumed, symbol + "[seeded-argreduce]"))
        emitted_launches += 1
    for start, end, replacement, symbol, consumed in \
            _render_bilinear_upsample2x(text, instances, bodies):
        if max_launches is not None and emitted_launches >= max_launches:
            report.append(("launch_limit", consumed, symbol))
            continue
        if (symbol in disabled_kernels or
                (only_kernels is not None and symbol not in only_kernels)):
            continue
        edits.append((start, end, replacement))
        consumed_structured_bodies.update(consumed)
        report.append(("match", consumed, symbol + "[bilinear-2x]"))
        emitted_launches += 1
    for start, end, replacement, symbol, consumed in \
            _render_fixed_average_pool_backward_regions(
                text, instances, bodies):
        if max_launches is not None and emitted_launches >= max_launches:
            report.append(("launch_limit", consumed, symbol))
            continue
        if (symbol in disabled_kernels or
                (only_kernels is not None and symbol not in only_kernels)):
            continue
        edits.append((start, end, replacement))
        consumed_structured_bodies.update(consumed)
        report.append(("match", consumed, symbol + "[whole-algorithm]"))
        emitted_launches += 1
    for start, end, replacement, symbol, consumed in \
            _render_cub_quant_col_offsets(text, instances, bodies):
        if max_launches is not None and emitted_launches >= max_launches:
            report.append(("launch_limit", consumed, symbol))
            continue
        if (symbol in disabled_kernels or
                (only_kernels is not None and symbol not in only_kernels)):
            continue
        edits.append((start, end, replacement))
        consumed_structured_bodies.update(consumed)
        report.append(("match", consumed, symbol + "[column-reduce+offset]"))
        emitted_launches += 1
    for start, end, replacement, symbol, consumed in \
            _render_cub_adjacent_difference(text, instances, bodies):
        if max_launches is not None and emitted_launches >= max_launches:
            report.append(("launch_limit", consumed, symbol))
            continue
        if (symbol in disabled_kernels or
                (only_kernels is not None and symbol not in only_kernels)):
            continue
        edits.append((start, end, replacement))
        consumed_structured_bodies.update(consumed)
        report.append(("match", consumed, symbol + "[whole-adjacent-difference]"))
        emitted_launches += 1
    if enable_structured_rewrite:
        for histogram_edits, symbol in _render_zeroed_i32_histograms(text):
            if max_launches is not None and emitted_launches >= max_launches:
                report.append(("launch_limit", [], symbol))
                continue
            edits.extend(histogram_edits)
            report.append(("match", [], symbol + "[zero+indirect-count]"))
            emitted_launches += 1
        for start, end, replacement, symbol, consumed in \
                _render_dense_factorization_regions(text, instances):
            if max_launches is not None and emitted_launches >= max_launches:
                report.append(("launch_limit", consumed, symbol))
                continue
            edits.append((start, end, replacement))
            consumed_structured_bodies.update(consumed)
            report.append(("match", consumed, symbol + "[whole-algorithm]"))
            emitted_launches += 1
        for start, end, replacement, symbol in _render_cusparse_csr_spmv(text):
            if max_launches is not None and emitted_launches >= max_launches:
                report.append(("launch_limit", [], symbol))
                continue
            edits.append((start, end, replacement))
            report.append(("match", [], symbol + "[indirect-row-reduction]"))
            emitted_launches += 1
        for start, end, replacement, symbol in _render_cusparse_csr_sddmm(text):
            if max_launches is not None and emitted_launches >= max_launches:
                report.append(("launch_limit", [], symbol))
                continue
            edits.append((start, end, replacement))
            consumed_structured_bodies.update(
                index for index, instance in enumerate(instances)
                if start <= instance.span[0] and instance.span[1] <= end)
            report.append(("match", [], symbol + "[sampled-dense-product]"))
            emitted_launches += 1
        for start, end, replacement, symbol in \
                _render_cusparse_index_conversions(text):
            if max_launches is not None and emitted_launches >= max_launches:
                report.append(("launch_limit", [], symbol))
                continue
            edits.append((start, end, replacement))
            report.append(("match", [], symbol + "[sparse-index-conversion]"))
            emitted_launches += 1
        for start, end, replacement, symbol in _render_cusparse_bsr_spmv(text):
            if max_launches is not None and emitted_launches >= max_launches:
                report.append(("launch_limit", [], symbol))
                continue
            edits.append((start, end, replacement))
            consumed_structured_bodies.update(
                index for index, instance in enumerate(instances)
                if start <= instance.span[0] and instance.span[1] <= end)
            report.append(("match", [], symbol + "[block-row-reduction]"))
            emitted_launches += 1
        for start, end, replacement, symbol in _render_cusparse_csr_spmm(text):
            if max_launches is not None and emitted_launches >= max_launches:
                report.append(("launch_limit", [], symbol))
                continue
            edits.append((start, end, replacement))
            report.append(("match", [], symbol + "[indirect-row-matrix-reduction]"))
            emitted_launches += 1
        for start, end, replacement, symbol in _render_cusparse_coo_spmm(text):
            if max_launches is not None and emitted_launches >= max_launches:
                report.append(("launch_limit", [], symbol))
                continue
            edits.append((start, end, replacement))
            consumed_structured_bodies.update(
                index for index, instance in enumerate(instances)
                if start <= instance.span[0] and instance.span[1] <= end)
            report.append(("match", [], symbol + "[zero+indexed-matrix-update]"))
            emitted_launches += 1
        for start, end, replacement, symbol in _render_cusparse_jds_spmv(text):
            if max_launches is not None and emitted_launches >= max_launches:
                report.append(("launch_limit", [], symbol))
                continue
            edits.append((start, end, replacement))
            report.append(("match", [], symbol + "[jds-to-csr+indirect-row-reduction]"))
            emitted_launches += 1
    i = 0
    while i < len(body_terms):
        if i in consumed_structured_bodies:
            i += 1
            continue
        generic_graph_spec: dict | None = None
        generic_graph_partition: tuple[dict, dict] | None = None
        feature_scale = _feature_mask_scale_capture(bodies[i])
        if feature_scale is not None:
            inst = instances[i]
            ins = _extract_ssa_names(inst.ins_part)
            in_types = _extract_ssa_types(inst.ins_part)
            outs = _extract_ssa_names(inst.outs_part)
            out_types = _extract_ssa_types(inst.outs_part)
            if (body_forms[i] == "tensor" and len(ins) == 2 and
                    len(outs) == 1 and inst.result_ssa is not None and
                    inst.result_type is not None and
                    [_shaped_rank(t) for t in in_types + out_types] ==
                        [4, 2, 4] and
                    all(_sniff_elem_type(t) == "f32"
                        for t in in_types + out_types) and
                    scalar_types.get(feature_scale) == "f32"):
                symbol = "cudnnFeatureMaskScale_f32_tensor"
                launch = render_launch(
                    symbol, inst.result_ssa, inst.result_type,
                    ins + [feature_scale] + outs, inst.indent, {}, [],
                    operand_types=in_types + ["f32"] + out_types,
                    scalar_type_map=scalar_types,
                    result_count=inst.result_count)
                edits.append((inst.span[0], inst.span[1], launch))
                report.append(("match", [i], symbol))
                emitted_launches += 1
                i += 1
                continue
        batchnorm_order = _batchnorm_inference_operand_order(bodies[i])
        if batchnorm_order is not None:
            inst = instances[i]
            ins = _extract_ssa_names(inst.ins_part)
            in_types = _extract_ssa_types(inst.ins_part)
            outs = _extract_ssa_names(inst.outs_part)
            out_types = _extract_ssa_types(inst.outs_part)
            legal = (
                body_forms[i] == "tensor" and len(ins) == 5 and
                len(outs) == 1 and inst.result_ssa is not None and
                inst.result_type is not None and
                [_shaped_rank(in_types[j]) for j in batchnorm_order] ==
                    [4, 1, 1, 1, 1] and
                _shaped_rank(out_types[0]) == 4 and
                all(_sniff_elem_type(t) == "f32"
                    for t in in_types + out_types))
            if legal:
                symbol = "cudnnBatchNormalizationForwardInference"
                operands = [ins[j] for j in batchnorm_order] + outs
                operand_types = [in_types[j] for j in batchnorm_order] + out_types
                launch = render_launch(
                    symbol, inst.result_ssa, inst.result_type, operands,
                    inst.indent, {}, [], operand_types=operand_types,
                    scalar_type_map=scalar_types,
                    result_count=inst.result_count)
                edits.append((inst.span[0], inst.span[1], launch))
                report.append(("match", [i], symbol))
                emitted_launches += 1
                i += 1
                continue
        # Bias initialization followed by the canonical 3D convolution
        # contraction is one cuDNN operation.  Egglog intentionally reasons
        # about scalar bodies and therefore sees the first stage as a copy;
        # the iterator/access metadata below supplies the missing tensor-level
        # proof that it is an OC bias broadcast into an NCDHW result.
        if i + 1 < len(bodies):
            init_body = bodies[i]
            conv_body = bodies[i + 1]
            init_inst = instances[i]
            conv_inst = instances[i + 1]
            init_ins = _extract_ssa_names(init_inst.ins_part)
            init_in_types = _extract_ssa_types(init_inst.ins_part)
            init_outs = _extract_ssa_names(init_inst.outs_part)
            init_out_types = _extract_ssa_types(init_inst.outs_part)
            conv_ins = _extract_ssa_names(conv_inst.ins_part)
            conv_in_types = _extract_ssa_types(conv_inst.ins_part)
            conv_outs = _extract_ssa_names(conv_inst.outs_part)
            conv_text = "\n".join(conv_body.body_lines)
            is_i8_i32_gemm = (
                not init_ins and len(init_outs) == 1
                and init_body.iterator_types == ["parallel"] * 2
                and _term_repr(body_terms[i]) == "Term.Lit(0.0)"
                and len(conv_ins) == 2 and len(conv_outs) == 1
                and [_shaped_rank(t) for t in conv_in_types] == [2, 2]
                and [_sniff_elem_type(t) for t in conv_in_types] == ["i8", "i8"]
                and [_shaped_rank(t) for t in
                     _extract_ssa_types(conv_inst.outs_part)] == [2]
                and [_sniff_elem_type(t) for t in
                     _extract_ssa_types(conv_inst.outs_part)] == ["i32"]
                and conv_body.iterator_types ==
                    ["parallel", "parallel", "reduction"]
                and conv_text.count("arith.extsi") == 2
                and "arith.muli" in conv_text and "arith.addi" in conv_text
                and init_inst.result_ssa is not None
                and conv_outs == [init_inst.result_ssa]
                and conv_inst.result_ssa is not None
                and conv_inst.result_type is not None)
            if is_i8_i32_gemm:
                emit_name = "cublasGemmEx_i8_i32_tensor"
                launch_line = render_launch(
                    emit_name, conv_inst.result_ssa, conv_inst.result_type,
                    conv_ins + init_outs, conv_inst.indent, {}, [],
                    operand_types=conv_in_types + init_out_types,
                    scalar_type_map=scalar_types,
                    result_count=conv_inst.result_count)
                edits.append((init_inst.span[0], init_inst.span[1], ""))
                edits.append((conv_inst.span[0], conv_inst.span[1], launch_line))
                report.append(("match", [i, i + 1], emit_name))
                emitted_launches += 1
                i += 2
                continue
            dilation = (_dilated_conv2d_factors(text, conv_ins[0])
                        if conv_ins else None)
            is_zero_dilated_conv2d = (
                not init_ins and len(init_outs) == 1
                and len(init_body.outs_arg_names) == 1
                and init_body.iterator_types == ["parallel"] * 3
                and _term_repr(body_terms[i]) == "Term.Lit(0.0)"
                and len(conv_ins) == 2 and len(conv_outs) == 1
                and [_shaped_rank(t) for t in conv_in_types] == [6, 4]
                and [_shaped_rank(t) for t in
                     _extract_ssa_types(conv_inst.outs_part)] == [3]
                and conv_body.iterator_types ==
                    ["parallel"] * 3 + ["reduction"] * 3
                and "arith.mulf" in conv_text
                and "arith.addf" in conv_text
                and init_inst.result_ssa is not None
                and conv_outs == [init_inst.result_ssa]
                and conv_inst.result_ssa is not None
                and conv_inst.result_type is not None
                and dilation is not None
                and all(_sniff_elem_type(t) == "f32" for t in
                        init_out_types + conv_in_types)
            )
            if is_zero_dilated_conv2d:
                emit_name = "cudnnConvolution2D_f32_dilated"
                if max_launches is not None and emitted_launches >= max_launches:
                    report.append(("launch_limit", [i, i + 1], emit_name))
                    i += 2
                    continue
                attrs = (f" {{dilation_h = {dilation[0]} : i64, "
                         f"dilation_w = {dilation[1]} : i64}}")
                launch_line = render_launch(
                    emit_name, conv_inst.result_ssa, conv_inst.result_type,
                    conv_ins + init_outs, conv_inst.indent, {}, [],
                    operand_types=conv_in_types + init_out_types,
                    scalar_type_map=scalar_types,
                    result_count=conv_inst.result_count,
                    launch_attrs=attrs)
                edits.append((init_inst.span[0], init_inst.span[1], ""))
                edits.append((conv_inst.span[0], conv_inst.span[1],
                              launch_line))
                report.append(("match", [i, i + 1], emit_name))
                emitted_launches += 1
                i += 2
                continue
            is_bias_conv1d = (
                len(init_ins) == len(init_outs) == 1
                and len(init_body.ins_arg_names) == 1
                and init_body.yield_values == init_body.ins_arg_names
                and init_body.iterator_types == ["parallel"] * 3
                and [_shaped_rank(t) for t in init_in_types] == [1]
                and [_shaped_rank(t) for t in init_out_types] == [3]
                and len(conv_ins) == 2 and len(conv_outs) == 1
                and [_shaped_rank(t) for t in conv_in_types] == [5, 3]
                and [_shaped_rank(t) for t in
                     _extract_ssa_types(conv_inst.outs_part)] == [3]
                and conv_body.iterator_types ==
                    ["parallel"] * 3 + ["reduction"] * 2
                and "arith.mulf" in conv_text
                and "arith.addf" in conv_text
                and "linalg.index" not in conv_text
                and "arith.cmpi" not in conv_text
                and init_inst.result_ssa is not None
                and conv_outs == [init_inst.result_ssa]
                and conv_inst.result_ssa is not None
                and conv_inst.result_type is not None
                and all(_sniff_elem_type(t) == "f32" for t in
                        init_in_types + init_out_types + conv_in_types)
                and re.search(
                    rf"{re.escape(conv_ins[0])}\s*=\s*polygeist\.submap\("
                    rf"[^\n]*\).*:\s*\(tensor<[^>]*x[^>]*x[^>]*xf32>",
                    text[:conv_inst.span[0]]) is not None
            )
            if is_bias_conv1d:
                emit_name = "cudnnConvolution1D_f32_bias"
                if max_launches is not None and emitted_launches >= max_launches:
                    report.append(("launch_limit", [i, i + 1], emit_name))
                    i += 2
                    continue
                launch_line = render_launch(
                    emit_name, conv_inst.result_ssa, conv_inst.result_type,
                    conv_ins + init_ins + init_outs, conv_inst.indent, {}, [],
                    operand_types=(conv_in_types + init_in_types +
                                   init_out_types),
                    scalar_type_map=scalar_types,
                    result_count=conv_inst.result_count,
                )
                edits.append((init_inst.span[0], init_inst.span[1], ""))
                edits.append((conv_inst.span[0], conv_inst.span[1],
                              launch_line))
                report.append(("match", [i, i + 1], emit_name))
                emitted_launches += 1
                i += 2
                continue
            is_bias_conv3d = (
                len(init_ins) == len(init_outs) == 1
                and len(init_body.ins_arg_names) == 1
                and init_body.yield_values == init_body.ins_arg_names
                and init_body.iterator_types == ["parallel"] * 4
                and [_shaped_rank(t) for t in init_in_types] == [1]
                and [_shaped_rank(t) for t in init_out_types] == [4]
                and len(conv_ins) == 2 and len(conv_outs) == 1
                and [_shaped_rank(t) for t in conv_in_types] == [8, 5]
                and [_shaped_rank(t) for t in
                     _extract_ssa_types(conv_inst.outs_part)] == [4]
                and conv_body.iterator_types ==
                    ["parallel"] * 4 + ["reduction"] * 4
                and _is_forward_conv3d_window(text, conv_ins[0])
                and "arith.mulf" in conv_text
                and "arith.addf" in conv_text
                and init_inst.result_ssa is not None
                and conv_outs == [init_inst.result_ssa]
                and conv_inst.result_ssa is not None
                and conv_inst.result_type is not None
                and all(_sniff_elem_type(t) == "f32" for t in
                        init_in_types + init_out_types + conv_in_types)
            )
            if is_bias_conv3d:
                emit_name = "cudnnConvolution3D_f32_bias"
                if max_launches is not None and emitted_launches >= max_launches:
                    report.append(("launch_limit", [i, i + 1], emit_name))
                    i += 2
                    continue
                # The runtime overwrites the original output slice, so pass
                # the init destination, not the bias-filled SSA result.
                launch_operands = conv_ins + init_ins + init_outs
                launch_types = conv_in_types + init_in_types + init_out_types
                launch_line = render_launch(
                    emit_name, conv_inst.result_ssa, conv_inst.result_type,
                    launch_operands, conv_inst.indent, {}, [],
                    operand_types=launch_types,
                    scalar_type_map=scalar_types,
                    result_count=conv_inst.result_count,
                )
                edits.append((init_inst.span[0], init_inst.span[1], ""))
                edits.append((conv_inst.span[0], conv_inst.span[1],
                              launch_line))
                report.append(("match", [i, i + 1], emit_name))
                emitted_launches += 1
                i += 2
                continue
        if body_terms[i] is None:
            report.append(("encoder_fail", i, "?"))
            i += 1
            continue
        if _is_inclusive_sum1d_f32(bodies[i], body_forms[i]):
            inst = instances[i]
            ins = _extract_ssa_names(inst.ins_part)
            in_types = _extract_ssa_types(inst.ins_part)
            outs = _extract_ssa_names(inst.outs_part)
            out_types = _extract_ssa_types(inst.outs_part)
            legal = (
                len(ins) == len(in_types) == 1 and
                len(outs) == len(out_types) == 2 and
                _sniff_elem_type(in_types[0]) == "f32" and
                [_shaped_rank(t) for t in in_types + out_types] == [1, 0, 1]
                and inst.result_ssa is not None and
                inst.result_type is not None and inst.result_count == 2)
            if legal:
                symbol = "cubInclusiveSum1D_f32_tensor"
                launch = render_launch(
                    symbol, inst.result_ssa, inst.result_type,
                    ins + outs, inst.indent, {}, [],
                    operand_types=in_types + out_types,
                    scalar_type_map=scalar_types,
                    result_count=inst.result_count)
                edits.append((inst.span[0], inst.span[1], launch))
                report.append(("match", [i], symbol))
                emitted_launches += 1
                i += 1
                continue
        if _is_segmented_inclusive_product2d_f32(
                bodies[i], body_forms[i]):
            inst = instances[i]
            ins = _extract_ssa_names(inst.ins_part)
            in_types = _extract_ssa_types(inst.ins_part)
            outs = _extract_ssa_names(inst.outs_part)
            out_types = _extract_ssa_types(inst.outs_part)
            legal = (
                len(ins) == len(in_types) == 1 and
                len(outs) == len(out_types) == 2 and
                all(_sniff_elem_type(t) == "f32"
                    for t in in_types + out_types) and
                [_shaped_rank(t) for t in in_types + out_types] == [2, 2, 1]
                and inst.result_ssa is not None and
                inst.result_type is not None and inst.result_count == 2)
            if legal:
                symbol = "cubSegmentedInclusiveProduct2D_f32_tensor"
                launch = render_launch(
                    symbol, inst.result_ssa, inst.result_type,
                    ins + outs, inst.indent, {}, [],
                    operand_types=in_types + out_types,
                    scalar_type_map=scalar_types,
                    result_count=inst.result_count)
                edits.append((inst.span[0], inst.span[1], launch))
                report.append(("match", [i], symbol))
                emitted_launches += 1
                i += 1
                continue
        predicate_reduction = _cub_predicate_reduction_kind(
            bodies[i], body_terms[i], body_forms[i])
        if predicate_reduction is not None:
            inst = instances[i]
            ins = _extract_ssa_names(inst.ins_part)
            in_types = _extract_ssa_types(inst.ins_part)
            outs = _extract_ssa_names(inst.outs_part)
            out_types = _extract_ssa_types(inst.outs_part)
            expected = {
                "count_nonzero_1d": ([1], [0], "cubCountNonzero1D_f32_tensor"),
                "count_nonzero_2d": ([2], [1], "cubSegmentedCountNonzero2D_f32_tensor"),
                "equal_all_1d": ([1, 1], [0], "cubEqualAll1D_f32_tensor"),
            }[predicate_reduction]
            in_ranks, out_ranks, symbol = expected
            legal = (
                len(ins) == len(in_types) == len(in_ranks) and
                len(outs) == len(out_types) == len(out_ranks) == 1 and
                [_shaped_rank(t) for t in in_types] == in_ranks and
                [_shaped_rank(t) for t in out_types] == out_ranks and
                all(_sniff_elem_type(t) == "f32" for t in in_types) and
                _sniff_elem_type(out_types[0]) == "i32" and
                inst.result_ssa is not None and
                inst.result_type is not None and inst.result_count == 1)
            if legal:
                launch = render_launch(
                    symbol, inst.result_ssa, inst.result_type,
                    ins + outs, inst.indent, {}, [],
                    operand_types=in_types + out_types,
                    scalar_type_map=scalar_types,
                    result_count=inst.result_count)
                edits.append((inst.span[0], inst.span[1], launch))
                report.append(("match", [i], symbol))
                emitted_launches += 1
                i += 1
                continue
        logical_flag = _cub_dynamic_segmented_logical_flag(
            bodies[i], body_terms[i], body_forms[i])
        if logical_flag is not None:
            inst = instances[i]
            ins = _extract_ssa_names(inst.ins_part)
            in_types = _extract_ssa_types(inst.ins_part)
            outs = _extract_ssa_names(inst.outs_part)
            out_types = _extract_ssa_types(inst.outs_part)
            legal = (
                len(ins) == len(in_types) == 2 and
                len(outs) == len(out_types) == 1 and
                [_shaped_rank(t) for t in in_types] == [2, 2] and
                _shaped_rank(out_types[0]) == 1 and
                all(_sniff_elem_type(t) == "i32"
                    for t in in_types + out_types) and
                scalar_types.get(logical_flag) in ("i1", "i32") and
                inst.result_ssa is not None and
                inst.result_type is not None and inst.result_count == 1)
            if legal:
                symbol = "cubSegmentedLogicalSelect_i32_tensor"
                launch = render_launch(
                    symbol, inst.result_ssa, inst.result_type,
                    ins + [logical_flag] + outs, inst.indent, {}, [],
                    operand_types=in_types + ["i1"] + out_types,
                    scalar_type_map=scalar_types,
                    result_count=inst.result_count)
                edits.append((inst.span[0], inst.span[1], launch))
                report.append(("match", [i], symbol))
                emitted_launches += 1
                i += 1
                continue
        pending_composition = match_composition(
            bodies, body_terms, comps, start=i, body_forms=body_forms)
        permutation = (
            None
            if (pending_composition is not None and
                len(pending_composition[0].steps) > 1)
            else _cutensor_permutation_modes(
                bodies[i], body_terms[i], body_forms[i])
        )
        if permutation is not None:
            inst = instances[i]
            ins = _extract_ssa_names(inst.ins_part)
            in_types = _extract_ssa_types(inst.ins_part)
            outs = _extract_ssa_names(inst.outs_part)
            out_types = _extract_ssa_types(inst.outs_part)
            input_modes, output_modes = permutation
            rank = len(input_modes)
            legal = (
                len(ins) == len(in_types) == len(outs) == len(out_types) == 1
                and inst.result_ssa is not None
                and inst.result_type is not None
                and [_shaped_rank(in_types[0]), _shaped_rank(out_types[0])] ==
                    [rank, rank]
                and _sniff_elem_type(in_types[0]) == "f32"
                and _sniff_elem_type(out_types[0]) == "f32")
            if legal and input_modes == output_modes:
                prefix = text[:inst.span[0]]
                view_defined = any(re.search(
                    rf"^\s*{re.escape(value)}\s*=\s*polygeist\.submap\b",
                    prefix, re.MULTILINE) for value in ins + outs)
                # Preserve the cheaper CUDA copy route for ordinary identity
                # copies. Identity modes are a permutation only when a
                # reshape/shuffle is encoded in a submap's physical strides.
                legal = view_defined
            if legal:
                symbol = f"cutensorPermute_f32_r{rank}_tensor"
                if (symbol in disabled_kernels or
                        (only_kernels is not None and
                         symbol not in only_kernels)):
                    legal = False
            if legal:
                symbol = f"cutensorPermute_f32_r{rank}_tensor"
                if max_launches is not None and emitted_launches >= max_launches:
                    report.append(("launch_limit", i, symbol))
                    i += 1
                    continue
                attrs = (
                    " {cutensor_input_modes = array<i64: " +
                    ", ".join(map(str, input_modes)) +
                    ">, cutensor_output_modes = array<i64: " +
                    ", ".join(map(str, output_modes)) + ">}")
                launch = render_launch(
                    symbol, inst.result_ssa, inst.result_type,
                    ins + outs, inst.indent, {}, [],
                    operand_types=in_types + out_types,
                    scalar_type_map=scalar_types,
                    result_count=inst.result_count, launch_attrs=attrs)
                edits.append((inst.span[0], inst.span[1], launch))
                report.append(("match", [i], symbol))
                emitted_launches += 1
                i += 1
                continue
        m = match_composition(bodies, body_terms, comps, start=i,
                              body_forms=body_forms)
        # Rank-2 GEMM subtraction and strided-batched GEMV subtraction both
        # have two parallel iterators, one reduction iterator, and the scalar
        # body `out - a*x`.  Scalar equality therefore produces both semantic
        # candidates.  Resolve that ambiguity with operand ranks before the
        # greedy choice reaches the ABI legality checks; otherwise the first
        # batched-GEMV candidate can shadow a valid ordinary GEMM.
        if (m is not None and
                m[0].name == "cublasDgemv_strided_batched_subtract"):
            ambiguous_inst = instances[i]
            ambiguous_types = (
                _extract_ssa_types(ambiguous_inst.ins_part) +
                _extract_ssa_types(ambiguous_inst.outs_part))
            if [_shaped_rank(t) for t in ambiguous_types] != [3, 2, 2]:
                m = match_composition(
                    bodies, body_terms,
                    [entry for entry in comps
                     if entry.name != "cublasDgemv_strided_batched_subtract"],
                    start=i, body_forms=body_forms)
        # The scratch-eliding overwrite ABI is currently FP64-only.  Do not
        # let its longer scalar pattern shadow the established FP32 sequence
        # (zero + SGEMV + copy) when the legality check cannot select DGEMV.
        if (m is not None and m[0].name == "cublasDgemv_T_zero" and
                i + 1 < len(instances)):
            contraction_types = (
                _extract_ssa_types(instances[i + 1].ins_part) +
                _extract_ssa_types(instances[i + 1].outs_part))
            if any(_sniff_elem_type(ty) != "f64"
                   for ty in contraction_types):
                m = match_composition(
                    bodies, body_terms,
                    [entry for entry in comps
                     if entry.name != "cublasDgemv_T_zero"],
                    start=i, body_forms=body_forms)
        if m is None:
            entry = match_elementwise_semantic(
                bodies[i], body_terms[i], body_forms[i]
            )
            if (entry is not None and
                    (entry.name in disabled_kernels or
                     (only_kernels is not None and
                      entry.name not in only_kernels))):
                entry = None
            if entry is None:
                graph = (None if disable_pointwise_matching else
                         _compile_cudnn_pointwise_graph(
                             bodies[i], body_terms[i]))
                inst = instances[i]
                in_types = _extract_ssa_types(inst.ins_part)
                out_types = _extract_ssa_types(inst.outs_part)
                maps = bodies[i].indexing_maps
                input_names = _extract_ssa_names(inst.ins_part)
                source_is_submap = any(re.search(
                    rf"^\s*{re.escape(source)}\s*=\s*polygeist\.submap\b",
                    text[:inst.span[0]], re.MULTILINE) for source in input_names)
                graph_legal = (
                    graph is not None
                    and graph.get("device_legal", False)
                    and body_forms[i] == "tensor"
                    and 1 <= len(in_types) <= 4
                    and len(out_types) == 1
                    and all(_sniff_elem_type(t) == "f32"
                            for t in in_types + out_types)
                    and all(_shaped_rank(t) == 1
                            for t in in_types + out_types)
                    and len(maps) == len(in_types) + 1
                    and all(m == maps[-1] for m in maps[:-1])
                    and bodies[i].iterator_types == ["parallel"]
                    and not source_is_submap
                    and all(key[0] == "Lit" or
                            scalar_types.get(key[1]) == "f32"
                            for key in (graph["scalars"] if graph else []))
                )
                partition = _partition_cudnn_pointwise_graph(
                    bodies[i], body_terms[i], len(in_types))
                partition_legal = (
                    partition is not None
                    and body_forms[i] == "tensor"
                    and 1 <= len(in_types) <= 3
                    and len(out_types) == 1
                    and all(_sniff_elem_type(t) == "f32"
                            for t in in_types + out_types)
                    and all(_shaped_rank(t) == 1
                            for t in in_types + out_types)
                    and len(maps) == len(in_types) + 1
                    and all(m == maps[-1] for m in maps[:-1])
                    and bodies[i].iterator_types == ["parallel"]
                    and not source_is_submap
                    and all(key[0] == "Lit" or
                            scalar_types.get(key[1]) == "f32"
                            for spec in (partition or ())
                            for key in spec["scalars"])
                )
                if not graph_legal:
                    if not partition_legal:
                        report.append(("no_match", i, "?"))
                        i += 1
                        continue
                    generic_graph_partition = partition
                    graph = partition[1]
                generic_graph_spec = graph
                entry = CompositionEntry(
                    name="cudnnPointwiseGraph_f32",
                    steps=[CompositionStep(
                        body=body_terms[i], num_ins=len(in_types), num_outs=1,
                        parallel_dim_count=1, reduction_dim_count=0)],
                    form="tensor", element_type="f32")
            binds = {}
            n = 1
        else:
            entry, _, binds = m
            n = len(entry.steps)
            # Algebraic templates also contain semantic-only entries whose
            # names have no executable ABI.  Do not let one of those shadow
            # the generic cuDNN graph route for a single legal pointwise DAG.
            if (n == 1 and entry.name not in ABI_LOWERABLE_KERNELS and
                    not disable_pointwise_matching):
                graph = _compile_cudnn_pointwise_graph(
                    bodies[i], body_terms[i])
                inst = instances[i]
                in_types = _extract_ssa_types(inst.ins_part)
                out_types = _extract_ssa_types(inst.outs_part)
                maps = bodies[i].indexing_maps
                input_names = _extract_ssa_names(inst.ins_part)
                source_is_submap = any(re.search(
                    rf"^\s*{re.escape(source)}\s*=\s*polygeist\.submap\b",
                    text[:inst.span[0]], re.MULTILINE)
                    for source in input_names)
                graph_legal = (
                    graph is not None
                    and graph.get("device_legal", False)
                    and body_forms[i] == "tensor"
                    and 1 <= len(in_types) <= 4
                    and len(out_types) == 1
                    and all(_sniff_elem_type(t) == "f32"
                            for t in in_types + out_types)
                    and all(_shaped_rank(t) == 1
                            for t in in_types + out_types)
                    and len(maps) == len(in_types) + 1
                    and all(indexing_map == maps[-1]
                            for indexing_map in maps[:-1])
                    and bodies[i].iterator_types == ["parallel"]
                    and not source_is_submap
                    and all(key[0] == "Lit" or
                            scalar_types.get(key[1]) == "f32"
                            for key in (graph["scalars"] if graph else []))
                )
                partition = _partition_cudnn_pointwise_graph(
                    bodies[i], body_terms[i], len(in_types))
                partition_legal = (
                    partition is not None
                    and body_forms[i] == "tensor"
                    and 1 <= len(in_types) <= 3
                    and len(out_types) == 1
                    and all(_sniff_elem_type(t) == "f32"
                            for t in in_types + out_types)
                    and all(_shaped_rank(t) == 1
                            for t in in_types + out_types)
                    and len(maps) == len(in_types) + 1
                    and all(indexing_map == maps[-1]
                            for indexing_map in maps[:-1])
                    and bodies[i].iterator_types == ["parallel"]
                    and not source_is_submap
                    and all(key[0] == "Lit" or
                            scalar_types.get(key[1]) == "f32"
                            for spec in (partition or ())
                            for key in spec["scalars"])
                )
                if graph_legal:
                    generic_graph_spec = graph
                    entry = CompositionEntry(
                        name="cudnnPointwiseGraph_f32",
                        steps=[CompositionStep(
                            body=body_terms[i], num_ins=len(in_types),
                            num_outs=1, parallel_dim_count=1,
                            reduction_dim_count=0)],
                        form="tensor", element_type="f32")
                    binds = {}
                elif partition_legal:
                    generic_graph_partition = partition
                    generic_graph_spec = partition[1]
                    entry = CompositionEntry(
                        name="cudnnPointwiseGraph_f32",
                        steps=[CompositionStep(
                            body=body_terms[i], num_ins=len(in_types),
                            num_outs=1, parallel_dim_count=1,
                            reduction_dim_count=0)],
                        form="tensor", element_type="f32")
                    binds = {}

        if _is_standalone_blas_candidate(entry.name):
            work = _static_generic_iteration_count(
                text, instances[i + n - 1], bodies[i + n - 1])
            if (work is not None and
                    work < _MIN_STANDALONE_BLAS_ITERATIONS):
                report.append((
                    "profitability_reject", list(range(i, i + n)),
                    f"{entry.name}[iterations={work},minimum="
                    f"{_MIN_STANDALONE_BLAS_ITERATIONS}]",
                ))
                i += n
                continue
        # Build a single kernel.launch covering instances[i..i+n-1].
        # We emit the launch *in place of the last generic* and delete the
        # earlier generics individually — that way any ops sitting BETWEEN
        # the matched generics (e.g. a `polygeist.submap` that the
        # contraction generic reads as an operand) are preserved
        # verbatim. Replacing the whole span [first.start, last.end]
        # with one launch would drop those intervening defs and leave
        # the launch referring to undefined SSA values.
        start = instances[i].span[0]
        end = instances[i + n - 1].span[1]
        # Operands: gather all tensor ins + the *first* outs (the chain root).
        all_tensor_ins: list[str] = []
        all_tensor_in_types: list[str] = []
        for j in range(n):
            inst = instances[i + j]
            all_tensor_ins.extend(_extract_ssa_names(inst.ins_part))
            all_tensor_in_types.extend(_extract_ssa_types(inst.ins_part))
        outs0 = _extract_ssa_names(instances[i].outs_part)
        outs0_types = _extract_ssa_types(instances[i].outs_part)
        operands = all_tensor_ins + outs0
        operand_types = all_tensor_in_types + outs0_types
        # Canonicalize input-operand order: higher-rank tensors first. For
        # bodies that are commutative in their two ins (e.g. gemv = out +
        # In(0)*In(1)), the matcher binds In(0)/In(1) in source-text order,
        # which produces (1D, 2D) for some callers and (2D, 1D) for others.
        # Reordering by rank gives a single canonical operand layout per
        # library entry so one kernel.defn suffices. Only sort the *inputs*
        # (`all_tensor_ins`); the launch's `outs0` is the chain root and
        # stays at its position. Safe only because library bodies treat the
        # two inputs symmetrically — the entries we ship in
        # kernel_library_phase2.mlir all do.
        def _tensor_rank(t: str) -> int:
            # Keep scalar tensors rank-0.  The former string heuristic treated
            # `tensor<f32>` as rank-1 because it mistook the element type for
            # a shape component, rejecting otherwise legal BLAS dot results.
            return _shaped_rank(t)
        if len(all_tensor_ins) >= 2:
            paired = sorted(
                zip(all_tensor_in_types, all_tensor_ins),
                key=lambda p: -_tensor_rank(p[0]),
            )
            sorted_types, sorted_names = zip(*paired)
            operands = list(sorted_names) + outs0
            operand_types = list(sorted_types) + outs0_types
        # The launch's result is the LAST generic's result SSA + type.
        last = instances[i + n - 1]

        # Symbol-name override: same body shape can come from different
        # operand-rank patterns that need different canonical defns. The
        # only case today: `cublasDcopy` body = In(0) fires on both
        #   - 1D-to-1D identity copy (doitgen)
        #   - scalar broadcast to 1D (fdtd-2d source-inject)
        # Distinguish by the input operand type: if it's a 0-D memref
        # (rank-0, written as `memref<<elem-type>, strided<...>>`), emit
        # `@broadcast_scalar_to_vec` instead. We use the operand type
        # rather than the indexing_map because parse_generics doesn't
        # resolve `#map` symbol references (only inline affine_map).
        emit_name = entry.name
        replace_full_span = False
        custom_launch_line: str | None = None
        custom_first_launch_line: str | None = None
        custom_edit_span: tuple[int, int] | None = None
        composition_root_rewires: list[tuple[str, str]] = []
        tail_only_rewires: list[tuple[str, str]] = []
        suppress_composition_tail_rewire = False
        pre_launch_lines: list[str] = []
        redundant_zero_fill_span: tuple[int, int] | None = None

        # These CUB operations overwrite their destinations.  Their raised
        # forms nevertheless contain an explicit initializer because that is
        # how the source loop expresses the reduction identity.  Once the
        # two bodies match as a composition, pass the real reduction outputs
        # to CUB and rewire any slice/writeback uses of the removed identity
        # tensor to its original destination.
        if (entry.name == "cubSegmentedCountNonzero2D_f32_tensor" and
                n == 2):
            init_result = instances[i].result_ssa
            init_out_names = _extract_ssa_names(instances[i].outs_part)
            if init_result is not None and len(init_out_names) == 1:
                composition_root_rewires.append(
                    (init_result, init_out_names[0]))

        if (entry.name in (
                "cubSegmentedLogicalAnd_i32",
                "cubSegmentedLogicalOr_i32",
                "cubSegmentedBitXor_i32",
                "cudnnMaxPoolFwd_batched") and n == 2):
            init_result = instances[i].result_ssa
            init_out_names = _extract_ssa_names(instances[i].outs_part)
            if init_result is not None and len(init_out_names) == 1:
                composition_root_rewires.append(
                    (init_result, init_out_names[0]))

        if (entry.name == "cubSegmentedInclusiveProduct2D_f32_tensor" and
                n == 2):
            product_inst = instances[i + 1]
            product_ins = _extract_ssa_names(product_inst.ins_part)
            product_in_types = _extract_ssa_types(product_inst.ins_part)
            product_outs = _extract_ssa_names(product_inst.outs_part)
            product_out_types = _extract_ssa_types(product_inst.outs_part)
            init_result = instances[i].result_ssa
            init_out_names = _extract_ssa_names(instances[i].outs_part)
            init_out_types = _extract_ssa_types(instances[i].outs_part)
            if (len(product_ins) == len(product_in_types) == 1 and
                    len(product_outs) == len(product_out_types) == 2 and
                    init_result is not None and
                    len(init_out_names) == len(init_out_types) == 1):
                # Some raised forms expand the row-final tensor to the full
                # [R,K] iteration domain and fold it back afterward.  CUB's
                # ABI writes one final value per row, so bind that operand and
                # result to the initializer's true [R] destination.
                operands = product_ins + [product_outs[0], init_out_names[0]]
                operand_types = (product_in_types + [product_out_types[0],
                                                      init_out_types[0]])
                composition_root_rewires.append(
                    (init_result, init_out_names[0]))
                if (product_inst.result_ssa is not None and
                        _shaped_rank(init_out_types[0]) == 1):
                    result_type = (f"({product_out_types[0]}, "
                                   f"{init_out_types[0]})")
                    custom_launch_line = render_launch(
                        entry.name, product_inst.result_ssa, result_type,
                        operands, product_inst.indent, binds, [],
                        operand_types=operand_types,
                        scalar_type_map=scalar_types,
                        result_count=2,
                    )

        destination_shaped_compositions = {
            "cubSegmentedCountNonzero2D_f32_tensor",
            "cubSegmentedLogicalAnd_i32",
            "cubSegmentedLogicalOr_i32",
            "cubSegmentedBitXor_i32",
            "cudnnMaxPoolFwd_batched",
        }
        if (entry.name in destination_shaped_compositions and n == 2 and
                last.result_ssa is not None):
            writeback = _find_tensor_submap_inverse(
                text, last.span[1], last.result_ssa)
            if writeback is not None:
                writeback_result, writeback_type, writeback_span = writeback
                custom_launch_line = render_launch(
                    entry.name, writeback_result, writeback_type,
                    operands, last.indent, binds, [],
                    operand_types=operand_types,
                    scalar_type_map=scalar_types,
                    result_count=1,
                )
                # The library result already has the enclosing destination
                # shape and defines writeback_result directly.
                edits.append((*writeback_span, ""))
                suppress_composition_tail_rewire = True

        fixed_conv_symbols = {
            "cudnnConvolutionTranspose3D_f32_memref",
            "cudnnConvolutionBackwardFilter3D_f32_memref",
            "cudnnConvolutionTBCBackward_f32_memref",
        }
        if entry.name == "cublasDgemv_T_zero":
            init_inst = instances[i]
            contraction_inst = instances[i + 1]
            copy_inst = instances[i + 2]
            contraction_ins = _extract_ssa_names(contraction_inst.ins_part)
            contraction_in_types = _extract_ssa_types(
                contraction_inst.ins_part)
            contraction_outs = _extract_ssa_names(
                contraction_inst.outs_part)
            contraction_out_types = _extract_ssa_types(
                contraction_inst.outs_part)
            copy_ins = _extract_ssa_names(copy_inst.ins_part)
            copy_outs = _extract_ssa_names(copy_inst.outs_part)
            copy_out_types = _extract_ssa_types(copy_inst.outs_part)
            maps = bodies[i + 1].indexing_maps

            def _map_outputs(txt: str) -> list[int]:
                resolved = _resolve_affine_map_text(text, txt)
                match = (re.search(r"->\s*\(([^)]*)\)>", resolved)
                         if resolved else None)
                if not match:
                    return []
                dims: list[int] = []
                for part in match.group(1).split(","):
                    dim = re.fullmatch(r"\s*d(\d+)\s*", part)
                    if not dim:
                        return []
                    dims.append(int(dim.group(1)))
                return dims

            map_dims = [_map_outputs(mapping) for mapping in maps]
            ranks = [_tensor_rank(ty) for ty in
                     contraction_in_types + contraction_out_types]
            elems = [_sniff_elem_type(ty) for ty in
                     contraction_in_types + contraction_out_types]
            iterator_roles = bodies[i + 1].iterator_types
            parallel_dims = {dim for dim, role in enumerate(iterator_roles)
                             if role == "parallel"}
            reduction_dims = {dim for dim, role in enumerate(iterator_roles)
                              if role == "reduction"}
            matrix_index = (ranks[:2].index(2)
                            if sorted(ranks[:2]) == [1, 2] else -1)
            vector_index = 1 - matrix_index if matrix_index >= 0 else -1
            legal = (
                len(contraction_ins) == 2
                and len(contraction_outs) == 1
                and sorted(ranks[:2]) == [1, 2]
                and ranks[2:] == [1]
                and elems == ["f64", "f64", "f64"]
                and len(parallel_dims) == 1
                and len(reduction_dims) == 1
                and len(map_dims) == 3
                and all(map_dims)
                and map_dims[matrix_index][0] in reduction_dims
                and map_dims[matrix_index][1] in parallel_dims
                and map_dims[vector_index] == [map_dims[matrix_index][0]]
                and map_dims[2] == [map_dims[matrix_index][1]]
                and copy_ins == [contraction_inst.result_ssa]
                and len(copy_outs) == len(copy_out_types) == 1
                and _tensor_rank(copy_out_types[0]) == 1
                and _sniff_elem_type(copy_out_types[0]) == "f64"
                and init_inst.result_ssa is not None
                and contraction_inst.result_ssa is not None
                and copy_inst.result_ssa is not None
                and copy_inst.result_type is not None
            )
            if not legal:
                report.append(("gemv_overwrite_abi_reject", i, entry.name))
                i += n
                continue
            emit_name = entry.name
            operands = [contraction_ins[matrix_index],
                        contraction_ins[vector_index]] + copy_outs
            operand_types = [contraction_in_types[matrix_index],
                             contraction_in_types[vector_index]] + \
                copy_out_types
            custom_launch_line = render_launch(
                emit_name, copy_inst.result_ssa, copy_inst.result_type,
                operands, copy_inst.indent, {}, [],
                operand_types=operand_types,
                scalar_type_map=scalar_types,
                result_count=copy_inst.result_count,
            )
            if outs0:
                composition_root_rewires.append(
                    (init_inst.result_ssa, outs0[0]))
            composition_root_rewires.append(
                (contraction_inst.result_ssa, contraction_outs[0]))
            # Both insert_slice operations become self-updates after the
            # scratch contraction is redirected to the true destination.
            # Rewire their loop-carried results to the original destination
            # tensors; leaving a functional "new tensor" yield would make
            # one-shot bufferization allocate and copy an entire parent tensor
            # for each loop iteration.
            between = text[contraction_inst.span[1]:copy_inst.span[0]]
            scratch_insert = re.search(
                rf"(?m)^\s*(%[\w_]+)\s*=\s*tensor\.insert_slice\s+"
                rf"{re.escape(contraction_inst.result_ssa)}\s+into\s+"
                rf"(%[\w_]+)[^\n]*(?:\n|$)", between)
            if scratch_insert:
                scratch_destination = scratch_insert.group(2)
                if (outs0 and
                        scratch_destination == init_inst.result_ssa):
                    scratch_destination = outs0[0]
                tail_only_rewires.append(
                    (scratch_insert.group(1), scratch_destination))
            after_copy = text[copy_inst.span[1]:]
            destination_insert = re.search(
                rf"(?m)^\s*(%[\w_]+)\s*=\s*tensor\.insert_slice\s+"
                rf"{re.escape(copy_inst.result_ssa)}\s+into\s+"
                rf"(%[\w_]+)[^\n]*(?:\n|$)",
                after_copy)
            if destination_insert:
                tail_only_rewires.append(
                    (destination_insert.group(1),
                     destination_insert.group(2)))
        elif entry.name in fixed_conv_symbols:
            conv_inst = instances[i]
            conv_ins = _extract_ssa_names(conv_inst.ins_part)
            conv_outs = _extract_ssa_names(conv_inst.outs_part)
            views = [
                _parse_memref_view(text, value, conv_inst.span[0])
                for value in conv_ins + conv_outs
            ]
            maps = [_compact_affine_map(m) for m in bodies[i].indexing_maps]
            legal = (
                len(conv_ins) == 2
                and len(conv_outs) == 1
                and all(view is not None for view in views)
            )
            physical_operands: list[str] = []
            physical_types: list[str] = []

            expanded_fixed_conv = (
                legal
                and all(view["kind"] == "submap" for view in views)
                and all(view["sizes"] == views[0]["sizes"] for view in views)
                and len(views[0]["sizes"]) in (5, 8)
            )
            if expanded_fixed_conv:
                logical_rank = len(views[0]["sizes"])
                # The parser canonicalizes iterator order, so an identity
                # generic may print as the same bijective dimension
                # permutation on all operands.  Accept only a complete
                # permutation—no projection, repetition, or arithmetic.
                map_match = re.fullmatch(
                    r"affine_map<\(([^)]*)\)->\(([^)]*)\)>", maps[0])
                map_inputs = (map_match.group(1).split(",")
                              if map_match else [])
                map_outputs = (map_match.group(2).split(",")
                               if map_match else [])
                expected_dims = [f"d{j}" for j in range(logical_rank)]
                expanded_fixed_conv = (
                    len(set(maps)) == 1
                    and map_inputs == expected_dims
                    and sorted(map_outputs) == expected_dims
                )
            if expanded_fixed_conv:
                logical_values = [_constant_index_value(text, size)
                                  for size in views[0]["sizes"]]
                physical_types = [view["base_type"] for view in views]
                expanded_fixed_conv = all(
                    value is not None and value > 0
                    for value in logical_values)

            if (expanded_fixed_conv and entry.name ==
                    "cudnnConvolutionTBCBackward_f32_memref"):
                grad, filt, output = views
                expected_view_maps = [
                    "affine_map<(d0,d1,d2,d3,d4)->(d0,d1,d2)>",
                    "affine_map<(d0,d1,d2,d3,d4)->(d3,d4,d2)>",
                    "affine_map<(d0,d1,d2,d3,d4)->(d3+d0,d1,d4)>",
                ]
                time, batch, out_channels, kernel, in_channels = logical_values
                physical_operands = [grad["base"], filt["base"], output["base"]]
                physical_types = [grad["base_type"], filt["base_type"],
                                  output["base_type"]]
                expanded_fixed_conv = (
                    [_compact_affine_map(view["map"]) for view in views]
                    == expected_view_maps
                    and _plain_f32_memrefs(physical_types, [3, 3, 3])
                    and _plain_shape_compatible(
                        physical_types[0], "f32", [time, batch, out_channels])
                    and _plain_shape_compatible(
                        physical_types[1], "f32",
                        [kernel, in_channels, out_channels])
                    and _plain_shape_compatible(
                        physical_types[2], "f32",
                        [time + kernel - 1, batch, in_channels])
                )
            elif (expanded_fixed_conv and entry.name ==
                  "cudnnConvolutionTranspose3D_f32_memref"):
                input_view, filter_view, output = views
                expected_view_maps = [
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->(d0,d2,d3,d4)>",
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->"
                    "(d0,d1,d5,d6,d7)>",
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->"
                    "(d1,d5+d2,d6+d3,d7+d4)>",
                ]
                (in_channels, out_channels, depth, height, width,
                 kd, kh, kw) = logical_values
                physical_operands = [input_view["base"], filter_view["base"],
                                     output["base"]]
                physical_types = [input_view["base_type"],
                                  filter_view["base_type"], output["base_type"]]
                expanded_fixed_conv = (
                    [_compact_affine_map(view["map"]) for view in views]
                    == expected_view_maps
                    and _plain_f32_memrefs(physical_types, [4, 5, 4])
                    and _plain_shape_compatible(
                        physical_types[0], "f32",
                        [in_channels, depth, height, width])
                    and _plain_shape_compatible(
                        physical_types[1], "f32",
                        [in_channels, out_channels, kd, kh, kw])
                    and _plain_shape_compatible(
                        physical_types[2], "f32",
                        [out_channels, depth + kd - 1,
                         height + kh - 1, width + kw - 1])
                )
            elif (expanded_fixed_conv and entry.name ==
                  "cudnnConvolutionBackwardFilter3D_f32_memref"):
                first, second, output = views
                small_map = (
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->(d0,d5,d6,d7)>"
                )
                window_map = (
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->"
                    "(d1,d5+d2,d6+d3,d7+d4)>"
                )
                output_map = (
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->"
                    "(d0,d1,d2,d3,d4)>"
                )
                first_map = _compact_affine_map(first["map"])
                second_map = _compact_affine_map(second["map"])
                out_channels, in_channels, kd, kh, kw, depth, height, width = (
                    logical_values)
                window = first if first_map == window_map else second
                small = second if first_map == window_map else first
                physical_operands = [window["base"], small["base"], output["base"]]
                physical_types = [window["base_type"], small["base_type"],
                                  output["base_type"]]
                expanded_fixed_conv = (
                    {first_map, second_map} == {small_map, window_map}
                    and _compact_affine_map(output["map"]) == output_map
                    and _plain_f32_memrefs(physical_types, [4, 4, 5])
                    and _plain_shape_compatible(
                        physical_types[0], "f32",
                        [in_channels, depth + kd - 1,
                         height + kh - 1, width + kw - 1])
                    and _plain_shape_compatible(
                        physical_types[1], "f32",
                        [out_channels, depth, height, width])
                    and _plain_shape_compatible(
                        physical_types[2], "f32",
                        [out_channels, in_channels, kd, kh, kw])
                )
            elif expanded_fixed_conv:
                expanded_fixed_conv = False

            if legal and all(view["kind"] == "submap" for view in views):
                legal = expanded_fixed_conv

            if (legal and not expanded_fixed_conv and
                    entry.name == "cudnnConvolutionTBCBackward_f32_memref"):
                grad, filt, output = views
                expected_maps = [
                    "affine_map<(d0,d1,d2,d3,d4)->(d0,d1,d4)>",
                    "affine_map<(d0,d1,d2,d3,d4)->(d2,d3,d4)>",
                    "affine_map<(d0,d1,d2,d3,d4)->(d0,d1,d2,d3)>",
                ]
                expected_output_map = (
                    "affine_map<(d0,d1,d2,d3)->(d2+d0,d1,d3)>"
                )
                legal = (
                    [grad["kind"], filt["kind"], output["kind"]]
                    == ["subview", "subview", "submap"]
                    and maps == expected_maps
                    and _compact_affine_map(output["map"]) == expected_output_map
                    and len(grad["sizes"]) == 3
                    and len(filt["sizes"]) == 3
                    and output["sizes"] == [
                        grad["sizes"][0], grad["sizes"][1],
                        filt["sizes"][0], filt["sizes"][1]
                    ]
                    and grad["sizes"][2] == filt["sizes"][2]
                )
                if legal:
                    physical_operands = [
                        grad["base"], filt["base"], output["base"]
                    ]
                    physical_types = [
                        grad["base_type"], filt["base_type"],
                        output["base_type"]
                    ]
                    legal = _plain_f32_memrefs(physical_types, [3, 3, 3])

            elif (legal and not expanded_fixed_conv and
                  entry.name == "cudnnConvolutionTranspose3D_f32_memref"):
                input_view, filter_view, output = views
                expected_maps = [
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->(d7,d0,d1,d2)>",
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->(d7,d3,d4,d5,d6)>",
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->(d3,d0,d1,d2,d4,d5,d6)>",
                ]
                expected_output_map = (
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6)->"
                    "(d0,d4+d1,d5+d2,d6+d3)>"
                )
                legal = (
                    [input_view["kind"], filter_view["kind"], output["kind"]]
                    == ["subview", "subview", "submap"]
                    and maps == expected_maps
                    and _compact_affine_map(output["map"]) == expected_output_map
                    and len(input_view["sizes"]) == 4
                    and len(filter_view["sizes"]) == 5
                    and filter_view["sizes"][0] == input_view["sizes"][0]
                    and output["sizes"] == [
                        filter_view["sizes"][1],
                        *input_view["sizes"][1:],
                        *filter_view["sizes"][2:],
                    ]
                )
                if legal:
                    physical_operands = [
                        input_view["base"], filter_view["base"], output["base"]
                    ]
                    physical_types = [
                        input_view["base_type"], filter_view["base_type"],
                        output["base_type"]
                    ]
                    legal = _plain_f32_memrefs(physical_types, [4, 5, 4])

            elif legal and not expanded_fixed_conv:
                # Both ATen weight-gradient forms use the same cuDNN
                # backward-filter call. One input is an expanded sliding
                # window; order that physical base first. This deliberately
                # reverses the transposed-convolution fixture's source inputs.
                first, second, output = views
                window = first if first["kind"] == "submap" else second
                small = second if first["kind"] == "submap" else first
                expected_output_map = (
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->"
                    "(d0,d1,d2,d3,d4)>"
                )
                expected_window_map = (
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->"
                    "(d1,d5+d2,d6+d3,d7+d4)>"
                )
                legal = (
                    {first["kind"], second["kind"]} == {"submap", "subview"}
                    and output["kind"] == "subview"
                    and maps[2] == expected_output_map
                    and _compact_affine_map(window["map"]) == expected_window_map
                    and len(window["sizes"]) == 8
                    and len(small["sizes"]) == 4
                    and len(output["sizes"]) == 5
                    and small["sizes"][1:] == window["sizes"][5:]
                    and output["sizes"] == [
                        small["sizes"][0], window["sizes"][1],
                        *window["sizes"][2:5]
                    ]
                )
                if legal:
                    physical_operands = [
                        window["base"], small["base"], output["base"]
                    ]
                    physical_types = [
                        window["base_type"], small["base_type"],
                        output["base_type"]
                    ]
                    legal = _plain_f32_memrefs(physical_types, [4, 4, 5])

            if legal:
                redundant_zero_fill_span = _zero_fill_span_for_base(
                    text, conv_inst.span[0], physical_operands[2]
                )
                legal = (
                    len(set(physical_operands)) == 3
                    and redundant_zero_fill_span is not None
                )
            if not legal:
                report.append(("fixed_convolution_layout_reject", i, entry.name))
                i += 1
                continue
            dynamic_types = {
                "cudnnConvolutionTBCBackward_f32_memref":
                    ["memref<?x?x?xf32>"] * 3,
                "cudnnConvolutionTranspose3D_f32_memref": [
                    "memref<?x?x?x?xf32>", "memref<?x?x?x?x?xf32>",
                    "memref<?x?x?x?xf32>"],
                "cudnnConvolutionBackwardFilter3D_f32_memref": [
                    "memref<?x?x?x?xf32>", "memref<?x?x?x?xf32>",
                    "memref<?x?x?x?x?xf32>"],
            }[entry.name]
            cast_names = [f"%fixed_conv_{i}_{j}" for j in range(3)]
            launch_lines = [
                f"{conv_inst.indent}{cast} = memref.cast {operand} : "
                f"{operand_type} to {dynamic_type}"
                for cast, operand, operand_type, dynamic_type in zip(
                    cast_names, physical_operands, physical_types,
                    dynamic_types)
            ]
            launch_lines.append(
                f"{conv_inst.indent}kernel.launch @{entry.name}("
                f"{', '.join(cast_names)}) : "
                f"({', '.join(dynamic_types)}) -> ()")
            custom_launch_line = "\n".join(launch_lines)
            operands = []
            operand_types = []
            binds = {}

        # The scaled TBC fixture is fully expanded by joint multi-root
        # debufferization: input, filter, and destination are rank-5 submaps
        # and the contraction itself consequently has identity maps.  Recover
        # the physical rank-3 buffers and prove the maps carried by those
        # submaps before selecting the same cuDNN TBC implementation.
        if (entry.name == "cudnnConvolutionTBC_f32_memref" and n == 2 and
                custom_launch_line is None):
            init_inst, conv_inst = instances[i:i + 2]
            init_outs = _extract_ssa_names(init_inst.outs_part)
            conv_ins = _extract_ssa_names(conv_inst.ins_part)
            conv_outs = _extract_ssa_names(conv_inst.outs_part)
            init_maps = [_compact_affine_map(mapping)
                         for mapping in bodies[i].indexing_maps]
            conv_maps = [_compact_affine_map(mapping)
                         for mapping in bodies[i + 1].indexing_maps]
            views = (
                [_tensor_submap_info(text, value, conv_inst.span[0])
                 for value in init_outs + conv_ins + conv_outs]
                if len(init_outs) == 1 and len(conv_ins) == 2 and
                len(conv_outs) == 1 else []
            )
            expanded_tbc = (
                len(views) == 4 and all(view is not None for view in views)
                and init_maps == [
                    "affine_map<(d0,d1,d2)->(d0,d1,d2)>"]
                and conv_maps == [
                    "affine_map<(d0,d1,d2,d3,d4)->(d0,d1,d2,d3,d4)>"] * 3
            )
            init_view = input_view = filter_view = output_view = None
            if expanded_tbc:
                init_view, input_view, filter_view, output_view = views
                init_writeback = _find_tensor_submap_inverse(
                    text, init_inst.span[1], init_inst.result_ssa)
                expected_view_maps = [
                    "affine_map<(d0,d1,d2)->(d0,d1,d2)>",
                    "affine_map<(d0,d1,d2,d3,d4)->(d3+d0,d1,d4)>",
                    "affine_map<(d0,d1,d2,d3,d4)->(d3,d4,d2)>",
                    "affine_map<(d0,d1,d2,d3,d4)->(d0,d1,d2)>",
                ]
                expanded_tbc = (
                    [_compact_affine_map(view["map"]) for view in views]
                    == expected_view_maps
                    and len(init_view["sizes"]) == 3
                    and len(input_view["sizes"]) == 5
                    and input_view["sizes"] == filter_view["sizes"]
                    and input_view["sizes"] == output_view["sizes"]
                    and output_view["sizes"][:3] == init_view["sizes"]
                    and init_writeback is not None
                    and init_writeback[0] == output_view["source"]
                    and init_writeback[2][1] <= conv_inst.span[0]
                )
            sources = []
            values = []
            if expanded_tbc:
                sources = [
                    _to_tensor_memref_source(
                        text, view["source"], conv_inst.span[0])
                    for view in (input_view, filter_view, init_view)
                ]
                values = [_constant_index_value(text, size)
                          for size in input_view["sizes"]]
                expanded_tbc = (
                    all(source is not None for source in sources)
                    and all(value is not None and value > 0 for value in values)
                )
            if expanded_tbc:
                to, batch, out_channels, kernel, in_channels = values
                physical_operands = [source[0] for source in sources]
                physical_types = [source[1] for source in sources]
                expanded_tbc = (
                    len(set(physical_operands)) == 3
                    and _plain_f32_memrefs(physical_types, [3, 3, 3])
                    and _plain_shape_compatible(
                        physical_types[0], "f32",
                        [to + kernel - 1, batch, in_channels])
                    and _plain_shape_compatible(
                        physical_types[1], "f32",
                        [kernel, in_channels, out_channels])
                    and _plain_shape_compatible(
                        physical_types[2], "f32", [to, batch, out_channels])
                )
            tail = None
            if expanded_tbc:
                size_pattern = r",\s*".join(
                    map(re.escape, input_view["sizes"]))
                tail = re.match(
                    rf"\s*(%[\w_\-]+)\s*=\s*polygeist\.submapInverse\s*"
                    rf"\(\s*{re.escape(output_view['source'])}\s*,\s*"
                    rf"{re.escape(conv_inst.result_ssa)}\s*,\s*{size_pattern}\)"
                    rf"\s*\{{\s*map\s*=\s*([^}}]+)\}}\s*:[^\n]+\n"
                    rf"\s*(%[\w_\-]+)\s*=\s*bufferization\.to_memref\s+\1"
                    rf"\s*:\s*[^\n]+\n\s*memref\.copy\s+\3\s*,\s*"
                    rf"{re.escape(physical_operands[2])}\s*:\s*[^\n]+",
                    text[conv_inst.span[1]:])
                expanded_tbc = tail is not None
            if expanded_tbc:
                inverse_map = _compact_affine_map(_resolve_affine_map_text(
                    text[:conv_inst.span[1] + tail.end()],
                    tail.group(2).strip()))
                expanded_tbc = inverse_map == expected_view_maps[3]
            if expanded_tbc:
                uid = conv_inst.result_ssa.lstrip("%").replace(".", "_")
                dynamic_type = "memref<?x?x?xf32>"
                cast_names = [f"%fixed_conv_{uid}_{j}" for j in range(3)]
                lines = [
                    f"{conv_inst.indent}{cast} = memref.cast {operand} : "
                    f"{operand_type} to {dynamic_type}"
                    for cast, operand, operand_type in zip(
                        cast_names, physical_operands, physical_types)
                ]
                lines.append(
                    f"{conv_inst.indent}kernel.launch @{entry.name}("
                    f"{', '.join(cast_names)}) : "
                    f"({', '.join([dynamic_type] * 3)}) -> ()")
                custom_launch_line = "\n".join(lines)
                replace_full_span = True
                custom_edit_span = (
                    init_inst.span[0], conv_inst.span[1] + tail.end())
                operands = []
                operand_types = []
                binds = {}

        if (entry.name == "cudnnConvolutionTranspose2D_f32_memref" and
                n == 2 and custom_launch_line is None):
            init_inst, conv_inst = instances[i:i + 2]
            init_outs = _extract_ssa_names(init_inst.outs_part)
            conv_ins = _extract_ssa_names(conv_inst.ins_part)
            conv_outs = _extract_ssa_names(conv_inst.outs_part)
            init_maps = [_compact_affine_map(mapping)
                         for mapping in bodies[i].indexing_maps]
            conv_maps = [_compact_affine_map(mapping)
                         for mapping in bodies[i + 1].indexing_maps]
            views = (
                [_tensor_submap_info(text, value, conv_inst.span[0])
                 for value in init_outs + conv_ins + conv_outs]
                if len(init_outs) == 1 and len(conv_ins) == 2 and
                len(conv_outs) == 1 else []
            )
            expanded_transpose2d = (
                len(views) == 4 and all(view is not None for view in views)
                and init_maps == [
                    "affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>"]
                and conv_maps == [
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6)->"
                    "(d0,d6,d1,d2,d3,d4,d5)>"] * 3
            )
            init_view = input_view = filter_view = output_view = None
            if expanded_transpose2d:
                init_view, input_view, filter_view, output_view = views
                init_writeback = _find_tensor_submap_inverse(
                    text, init_inst.span[1], init_inst.result_ssa)
                expected_view_maps = [
                    "affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>",
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6)->(d0,d1,d2,d3)>",
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6)->(d1,d4,d5,d6)>",
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6)->"
                    "(d0,d4,d5+d2,d6+d3)>",
                ]
                expanded_transpose2d = (
                    [_compact_affine_map(view["map"]) for view in views]
                    == expected_view_maps
                    and len(init_view["sizes"]) == 4
                    and len(input_view["sizes"]) == 7
                    and input_view["sizes"] == filter_view["sizes"]
                    and input_view["sizes"] == output_view["sizes"]
                    and init_writeback is not None
                    and init_writeback[0] == output_view["source"]
                    and init_writeback[2][1] <= conv_inst.span[0]
                )
            sources = []
            values = []
            if expanded_transpose2d:
                sources = [
                    _to_tensor_memref_source(
                        text, view["source"], conv_inst.span[0])
                    for view in (input_view, filter_view, init_view)
                ]
                values = [_constant_index_value(text, size)
                          for size in input_view["sizes"]]
                expanded_transpose2d = (
                    all(source is not None for source in sources)
                    and all(value is not None and value > 0 for value in values)
                )
            if expanded_transpose2d:
                batch, in_channels, ih, iw, out_channels, kh, kw = values
                output_values = [_constant_index_value(text, size)
                                 for size in init_view["sizes"]]
                physical_operands = [source[0] for source in sources]
                physical_types = [source[1] for source in sources]
                expanded_transpose2d = (
                    output_values == [batch, out_channels,
                                      ih + kh - 1, iw + kw - 1]
                    and len(set(physical_operands)) == 3
                    and _plain_f32_memrefs(physical_types, [4, 4, 4])
                    and _plain_shape_compatible(
                        physical_types[0], "f32", [batch, in_channels, ih, iw])
                    and _plain_shape_compatible(
                        physical_types[1], "f32",
                        [in_channels, out_channels, kh, kw])
                    and _plain_shape_compatible(
                        physical_types[2], "f32", output_values)
                )
            tail = None
            if expanded_transpose2d:
                size_pattern = r",\s*".join(
                    map(re.escape, input_view["sizes"]))
                tail = re.match(
                    rf"\s*(%[\w_\-]+)\s*=\s*polygeist\.submapInverse\s*"
                    rf"\(\s*{re.escape(output_view['source'])}\s*,\s*"
                    rf"{re.escape(conv_inst.result_ssa)}\s*,\s*{size_pattern}\)"
                    rf"\s*\{{\s*map\s*=\s*([^}}]+)\}}\s*:[^\n]+\n"
                    rf"\s*(%[\w_\-]+)\s*=\s*bufferization\.to_memref\s+\1"
                    rf"\s*:\s*[^\n]+\n\s*memref\.copy\s+\3\s*,\s*"
                    rf"{re.escape(physical_operands[2])}\s*:\s*[^\n]+",
                    text[conv_inst.span[1]:])
                expanded_transpose2d = tail is not None
            if expanded_transpose2d:
                inverse_map = _compact_affine_map(_resolve_affine_map_text(
                    text[:conv_inst.span[1] + tail.end()],
                    tail.group(2).strip()))
                expanded_transpose2d = inverse_map == expected_view_maps[3]
            if expanded_transpose2d:
                uid = conv_inst.result_ssa.lstrip("%").replace(".", "_")
                dynamic_type = "memref<?x?x?x?xf32>"
                cast_names = [f"%fixed_conv_{uid}_{j}" for j in range(3)]
                lines = [
                    f"{conv_inst.indent}{cast} = memref.cast {operand} : "
                    f"{operand_type} to {dynamic_type}"
                    for cast, operand, operand_type in zip(
                        cast_names, physical_operands, physical_types)
                ]
                lines.append(
                    f"{conv_inst.indent}kernel.launch @{entry.name}("
                    f"{', '.join(cast_names)}) : "
                    f"({', '.join([dynamic_type] * 3)}) -> ()")
                custom_launch_line = "\n".join(lines)
                replace_full_span = True
                custom_edit_span = (
                    init_inst.span[0], conv_inst.span[1] + tail.end())
                operands = []
                operand_types = []
                binds = {}

        if (entry.name == "cudnnDepthwiseConvolution2D_f32_memref" and
                n == 2 and custom_launch_line is None):
            init_inst, conv_inst = instances[i:i + 2]
            init_ins = _extract_ssa_names(init_inst.ins_part)
            init_outs = _extract_ssa_names(init_inst.outs_part)
            conv_ins = _extract_ssa_names(conv_inst.ins_part)
            conv_outs = _extract_ssa_names(conv_inst.outs_part)
            init_maps = [_compact_affine_map(mapping)
                         for mapping in bodies[i].indexing_maps]
            conv_maps = [_compact_affine_map(mapping)
                         for mapping in bodies[i + 1].indexing_maps]
            values_to_parse = init_ins + init_outs + conv_ins + conv_outs
            views = (
                [_tensor_submap_info(text, value, conv_inst.span[0])
                 for value in values_to_parse]
                if (len(init_ins) == len(init_outs) == len(conv_outs) == 1
                    and len(conv_ins) == 2) else []
            )
            expanded_depthwise = (
                len(views) == 5 and all(view is not None for view in views)
                and init_maps == [
                    "affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>"] * 2
                and conv_maps == [
                    "affine_map<(d0,d1,d2,d3,d4,d5)->"
                    "(d0,d1,d2,d3,d4,d5)>"] * 3
            )
            bias_view = init_output_view = input_view = None
            filter_view = conv_output_view = None
            if expanded_depthwise:
                (bias_view, init_output_view, input_view,
                 filter_view, conv_output_view) = views
                init_writeback = _find_tensor_submap_inverse(
                    text, init_inst.span[1], init_inst.result_ssa)
                expected_view_maps = [
                    "affine_map<(d0,d1,d2,d3)->(d1)>",
                    "affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>",
                    "affine_map<(d0,d1,d2,d3,d4,d5)->"
                    "(d0,d1,d4+d2-1,d5+d3-1)>",
                    "affine_map<(d0,d1,d2,d3,d4,d5)->(d1,d4,d5)>",
                    "affine_map<(d0,d1,d2,d3,d4,d5)->(d0,d1,d2,d3)>",
                ]
                expanded_depthwise = (
                    [_compact_affine_map(view["map"]) for view in views]
                    == expected_view_maps
                    and len(init_output_view["sizes"]) == 4
                    and bias_view["sizes"] == init_output_view["sizes"]
                    and len(input_view["sizes"]) == 6
                    and input_view["sizes"] == filter_view["sizes"]
                    and input_view["sizes"] == conv_output_view["sizes"]
                    and input_view["sizes"][:4] == init_output_view["sizes"]
                    and init_writeback is not None
                    and init_writeback[0] == conv_output_view["source"]
                    and init_writeback[2][1] <= conv_inst.span[0]
                )
            sources = []
            logical_values = []
            if expanded_depthwise:
                sources = [
                    _to_tensor_memref_source(
                        text, view["source"], conv_inst.span[0])
                    for view in (input_view, filter_view, bias_view,
                                 init_output_view)
                ]
                logical_values = [_constant_index_value(text, size)
                                  for size in input_view["sizes"]]
                expanded_depthwise = (
                    all(source is not None for source in sources)
                    and all(value is not None and value > 0
                            for value in logical_values)
                )
            if expanded_depthwise:
                batch, channels, height, width, kh, kw = logical_values
                physical_operands = [source[0] for source in sources]
                physical_types = [source[1] for source in sources]
                expanded_depthwise = (
                    kh == kw == 3
                    and len(set(physical_operands)) == 4
                    and _plain_f32_memrefs(physical_types, [4, 3, 1, 4])
                    and _plain_shape_compatible(
                        physical_types[0], "f32",
                        [batch, channels, height, width])
                    and _plain_shape_compatible(
                        physical_types[1], "f32", [channels, kh, kw])
                    and _plain_shape_compatible(
                        physical_types[2], "f32", [channels])
                    and _plain_shape_compatible(
                        physical_types[3], "f32",
                        [batch, channels, height, width])
                    and _fixed_depthwise_padding_body_legal(
                        text, bodies[i + 1], height, width)
                )
            tail = None
            if expanded_depthwise:
                size_pattern = r",\s*".join(
                    map(re.escape, input_view["sizes"]))
                tail = re.match(
                    rf"\s*(%[\w_\-]+)\s*=\s*polygeist\.submapInverse\s*"
                    rf"\(\s*{re.escape(conv_output_view['source'])}\s*,\s*"
                    rf"{re.escape(conv_inst.result_ssa)}\s*,\s*{size_pattern}\)"
                    rf"\s*\{{\s*map\s*=\s*([^}}]+)\}}\s*:[^\n]+\n"
                    rf"\s*(%[\w_\-]+)\s*=\s*bufferization\.to_memref\s+\1"
                    rf"\s*:\s*[^\n]+\n\s*memref\.copy\s+\3\s*,\s*"
                    rf"{re.escape(physical_operands[3])}\s*:\s*[^\n]+",
                    text[conv_inst.span[1]:])
                expanded_depthwise = tail is not None
            if expanded_depthwise:
                inverse_map = _compact_affine_map(_resolve_affine_map_text(
                    text[:conv_inst.span[1] + tail.end()],
                    tail.group(2).strip()))
                expanded_depthwise = inverse_map == expected_view_maps[4]
            if expanded_depthwise:
                uid = conv_inst.result_ssa.lstrip("%").replace(".", "_")
                dynamic_types = [
                    "memref<?x?x?x?xf32>", "memref<?x?x?xf32>",
                    "memref<?xf32>", "memref<?x?x?x?xf32>"]
                cast_names = [f"%fixed_conv_{uid}_{j}" for j in range(4)]
                lines = [
                    f"{conv_inst.indent}{cast} = memref.cast {operand} : "
                    f"{operand_type} to {dynamic_type}"
                    for cast, operand, operand_type, dynamic_type in zip(
                        cast_names, physical_operands, physical_types,
                        dynamic_types)
                ]
                lines.append(
                    f"{conv_inst.indent}kernel.launch @{entry.name}("
                    f"{', '.join(cast_names)}) : "
                    f"({', '.join(dynamic_types)}) -> ()")
                custom_launch_line = "\n".join(lines)
                replace_full_span = True
                custom_edit_span = (
                    init_inst.span[0], conv_inst.span[1] + tail.end())
                operands = []
                operand_types = []
                binds = {}

        tensor_fixed_conv_symbols = {
            "cudnnConvolutionTranspose2D_f32_memref",
            "cudnnConvolutionTBC_f32_memref",
            "cudnnDepthwiseConvolution2D_f32_memref",
        }
        if (entry.name in tensor_fixed_conv_symbols and
                custom_launch_line is None):
            init_inst = instances[i]
            conv_inst = instances[i + 1]
            init_ins = _extract_ssa_names(init_inst.ins_part)
            init_outs = _extract_ssa_names(init_inst.outs_part)
            conv_ins = _extract_ssa_names(conv_inst.ins_part)
            conv_outs = _extract_ssa_names(conv_inst.outs_part)
            init_maps = [
                _compact_affine_map(m) for m in bodies[i].indexing_maps]
            conv_maps = [
                _compact_affine_map(m) for m in bodies[i + 1].indexing_maps]
            legal = (
                n == 2
                and len(init_outs) == 1
                and init_inst.result_ssa is not None
                and len(conv_ins) == 2
                and len(conv_outs) == 1
                and conv_inst.result_ssa is not None
            )
            physical_operands: list[str] = []
            physical_types: list[str] = []
            custom_tail = None

            if legal and entry.name == "cudnnConvolutionTBC_f32_memref":
                output_slice = _tensor_extract_slice_info(
                    text, init_outs[0], init_inst.span[0])
                input_window = _tensor_submap_info(
                    text, conv_ins[0], conv_inst.span[0])
                filter_slice = _tensor_extract_slice_info(
                    text, conv_ins[1], conv_inst.span[0])
                legal = all(item is not None for item in (
                    output_slice, input_window, filter_slice))
                if legal:
                    output_sizes = output_slice["sizes"]
                    window_sizes = input_window["sizes"]
                    filter_sizes = filter_slice["sizes"]
                    expected_maps = [
                        "affine_map<(d0,d1,d2,d3,d4)->(d0,d1,d2,d3,d4)>",
                        "affine_map<(d0,d1,d2,d3,d4)->(d3,d4,d2)>",
                        "affine_map<(d0,d1,d2,d3,d4)->(d0,d1,d2)>",
                    ]
                    expected_window_map = (
                        "affine_map<(d0,d1,d2,d3,d4)->(d3+d0,d1,d4)>"
                    )
                    legal = (
                        conv_outs == [init_inst.result_ssa]
                        and
                        not init_ins
                        and init_maps == [
                            "affine_map<(d0,d1,d2)->(d0,d1,d2)>"]
                        and conv_maps == expected_maps
                        and output_slice["offsets"] == ["0"] * 3
                        and output_slice["strides"] == ["1"] * 3
                        and len(output_sizes) == 3
                        and len(window_sizes) == 5
                        and filter_slice["offsets"] == ["0"] * 3
                        and filter_slice["strides"] == ["1"] * 3
                        and len(filter_sizes) == 3
                        and window_sizes == [
                            *output_sizes, filter_sizes[0], filter_sizes[1]]
                        and filter_sizes[2] == output_sizes[2]
                        and _compact_affine_map(input_window["map"])
                        == expected_window_map
                    )
                if legal:
                    input_source = _to_tensor_memref_source(
                        text, input_window["source"], conv_inst.span[0])
                    filter_source = _to_tensor_memref_source(
                        text, filter_slice["source"], conv_inst.span[0])
                    output_source = _to_tensor_memref_source(
                        text, output_slice["source"], conv_inst.span[0])
                    legal = all(source is not None for source in (
                        input_source, filter_source, output_source))
                if legal:
                    physical_operands = [
                        input_source[0], filter_source[0], output_source[0]]
                    physical_types = [
                        input_source[1], filter_source[1], output_source[1]]
                    values = [
                        _constant_index_value(text, size)
                        for size in output_sizes + filter_sizes]
                    to, batch, out_channels, kernel, in_channels, filter_out = values
                    legal = (
                        all(value is not None and value > 0 for value in values)
                        and len(set(physical_operands)) == 3
                        and _plain_f32_memrefs(physical_types, [3, 3, 3])
                        and _plain_shape_compatible(
                            physical_types[0], "f32",
                            [to + kernel - 1, batch, in_channels])
                        and _plain_shape_compatible(
                            physical_types[1], "f32",
                            [kernel, in_channels, filter_out])
                        and _plain_shape_compatible(
                            physical_types[2], "f32",
                            [to, batch, out_channels])
                    )
                if legal:
                    sizes_pattern = r",\s*".join(map(re.escape, output_sizes))
                    custom_tail = re.match(
                        rf"\s*(%[\w_\-]+)\s*=\s*tensor\.insert_slice\s+"
                        rf"{re.escape(conv_inst.result_ssa)}\s+into\s+"
                        rf"{re.escape(output_slice['source'])}"
                        rf"\[0,\s*0,\s*0\]\s*\[{sizes_pattern}\]\s*"
                        rf"\[1,\s*1,\s*1\][^\n]*\n"
                        rf"\s*(%[\w_\-]+)\s*=\s*bufferization\.to_memref\s+\1"
                        rf"\s*:\s*[^\n]+\n\s*memref\.copy\s+\2\s*,\s*"
                        rf"{re.escape(physical_operands[2])}\s*:\s*[^\n]+",
                        text[conv_inst.span[1]:])
                    legal = custom_tail is not None

            elif legal and entry.name == "cudnnConvolutionTranspose2D_f32_memref":
                output_slice = _tensor_extract_slice_info(
                    text, init_outs[0], init_inst.span[0])
                input_slice = _tensor_extract_slice_info(
                    text, conv_ins[0], conv_inst.span[0])
                filter_slice = _tensor_extract_slice_info(
                    text, conv_ins[1], conv_inst.span[0])
                output_view = _tensor_submap_info(
                    text, conv_outs[0], conv_inst.span[0])
                legal = all(item is not None for item in (
                    output_slice, input_slice, filter_slice, output_view))
                inserted_match = None
                if legal:
                    output_sizes = output_slice["sizes"]
                    output_size_pattern = r",\s*".join(
                        map(re.escape, output_sizes))
                    inserted_match = re.search(
                        rf"^\s*(%[\w_\-]+)\s*=\s*tensor\.insert_slice\s+"
                        rf"{re.escape(init_inst.result_ssa)}\s+into\s+"
                        rf"{re.escape(output_slice['source'])}"
                        rf"\[0,\s*0,\s*0,\s*0\]\s*"
                        rf"\[{output_size_pattern}\]\s*"
                        rf"\[1,\s*1,\s*1,\s*1\]",
                        text[init_inst.span[1]:conv_inst.span[0]], re.MULTILINE)
                    legal = inserted_match is not None
                if legal:
                    input_sizes = input_slice["sizes"]
                    filter_sizes = filter_slice["sizes"]
                    view_sizes = output_view["sizes"]
                    expected_maps = [
                        "affine_map<(d0,d1,d2,d3,d4,d5)->(d5,d0,d1)>",
                        "affine_map<(d0,d1,d2,d3,d4,d5)->(d5,d2,d3,d4)>",
                        "affine_map<(d0,d1,d2,d3,d4,d5)->(d0,d1,d2,d3,d4)>",
                    ]
                    expected_output_map = (
                        "affine_map<(d0,d1,d2,d3,d4)->"
                        "(0,d2,d3+d0,d4+d1)>"
                    )
                    legal = (
                        not init_ins
                        and init_maps == [
                            "affine_map<(d0,d1,d2)->(d0,d1,d2)>"]
                        and conv_maps == expected_maps
                        and output_slice["offsets"] == ["0"] * 4
                        and output_slice["strides"] == ["1"] * 4
                        and input_slice["offsets"] == ["0"] * 4
                        and input_slice["strides"] == ["1"] * 4
                        and filter_slice["offsets"] == ["0"] * 4
                        and filter_slice["strides"] == ["1"] * 4
                        and len(output_sizes) == len(input_sizes) == 4
                        and len(filter_sizes) == 4
                        and len(view_sizes) == 5
                        and output_view["source"] == inserted_match.group(1)
                        and view_sizes == [
                            input_sizes[2], input_sizes[3],
                            filter_sizes[1], filter_sizes[2], filter_sizes[3]]
                        and input_sizes[1] == filter_sizes[0]
                        and _compact_affine_map(output_view["map"])
                        == expected_output_map
                    )
                if legal:
                    input_source = _to_tensor_memref_source(
                        text, input_slice["source"], conv_inst.span[0])
                    filter_source = _to_tensor_memref_source(
                        text, filter_slice["source"], conv_inst.span[0])
                    output_source = _to_tensor_memref_source(
                        text, output_slice["source"], conv_inst.span[0])
                    legal = all(source is not None for source in (
                        input_source, filter_source, output_source))
                if legal:
                    physical_operands = [
                        input_source[0], filter_source[0], output_source[0]]
                    physical_types = [
                        input_source[1], filter_source[1], output_source[1]]
                    values = [
                        _constant_index_value(text, size) if size != "1" else 1
                        for size in input_sizes + filter_sizes + output_sizes]
                    (n_batch, in_channels, ih, iw, filter_in, out_channels,
                     kh, kw, out_batch, output_channels, oh, ow) = values
                    legal = (
                        all(value is not None and value > 0 for value in values)
                        and n_batch == out_batch == 1
                        and in_channels == filter_in
                        and out_channels == output_channels
                        and oh == ih + kh - 1
                        and ow == iw + kw - 1
                        and len(set(physical_operands)) == 3
                        and _plain_f32_memrefs(physical_types, [4, 4, 4])
                        and _plain_shape_compatible(
                            physical_types[0], "f32",
                            [n_batch, in_channels, ih, iw])
                        and _plain_shape_compatible(
                            physical_types[1], "f32",
                            [filter_in, out_channels, kh, kw])
                        and _plain_shape_compatible(
                            physical_types[2], "f32",
                            [out_batch, output_channels, oh, ow])
                    )
                if legal:
                    view_pattern = r",\s*".join(map(re.escape, view_sizes))
                    custom_tail = re.match(
                        rf"\s*(%[\w_\-]+)\s*=\s*polygeist\.submapInverse\s*"
                        rf"\(\s*{re.escape(inserted_match.group(1))}\s*,\s*"
                        rf"{re.escape(conv_inst.result_ssa)}\s*,\s*{view_pattern}\)"
                        rf"\s*\{{\s*map\s*=\s*([^}}]+)\}}\s*:[^\n]+\n"
                        rf"\s*(%[\w_\-]+)\s*=\s*bufferization\.to_memref\s+\1"
                        rf"\s*:\s*[^\n]+\n\s*memref\.copy\s+\3\s*,\s*"
                        rf"{re.escape(physical_operands[2])}\s*:\s*[^\n]+",
                        text[conv_inst.span[1]:])
                    legal = custom_tail is not None
                if legal:
                    inverse_map = _compact_affine_map(_resolve_affine_map_text(
                        text[:conv_inst.span[1] + custom_tail.end()],
                        custom_tail.group(2).strip()))
                    legal = inverse_map == expected_output_map

            elif legal:
                bias_slice = _tensor_extract_slice_info(
                    text, init_ins[0], init_inst.span[0]) if len(init_ins) == 1 else None
                output_slice = _tensor_extract_slice_info(
                    text, init_outs[0], init_inst.span[0])
                input_window = _tensor_submap_info(
                    text, conv_ins[0], conv_inst.span[0])
                filter_slice = _tensor_extract_slice_info(
                    text, conv_ins[1], conv_inst.span[0])
                legal = all(item is not None for item in (
                    bias_slice, output_slice, input_window, filter_slice))
                if legal:
                    bias_sizes = bias_slice["sizes"]
                    output_sizes = output_slice["sizes"]
                    window_sizes = input_window["sizes"]
                    filter_sizes = filter_slice["sizes"]
                    expected_init_maps = [
                        "affine_map<(d0,d1,d2)->(d0)>",
                        "affine_map<(d0,d1,d2)->(d0,d1,d2)>",
                    ]
                    expected_conv_maps = [
                        "affine_map<(d0,d1,d2,d3,d4)->(d0,d1,d2,d3,d4)>",
                        "affine_map<(d0,d1,d2,d3,d4)->(d0,d3,d4)>",
                        "affine_map<(d0,d1,d2,d3,d4)->(d0,d1,d2)>",
                    ]
                    expected_window_map = (
                        "affine_map<(d0,d1,d2,d3,d4)->"
                        "(0,d0,d3+d1-1,d4+d2-1)>"
                    )
                    legal = (
                        conv_outs == [init_inst.result_ssa]
                        and init_maps == expected_init_maps
                        and conv_maps == expected_conv_maps
                        and bias_slice["offsets"] == ["0"]
                        and bias_slice["strides"] == ["1"]
                        and output_slice["offsets"] == ["0"] * 4
                        and output_slice["strides"] == ["1"] * 4
                        and filter_slice["offsets"] == ["0"] * 3
                        and filter_slice["strides"] == ["1"] * 3
                        and len(bias_sizes) == 1
                        and len(output_sizes) == 4
                        and len(window_sizes) == 5
                        and len(filter_sizes) == 3
                        and output_sizes[0] == "1"
                        and bias_sizes[0] == output_sizes[1]
                        and window_sizes == [
                            output_sizes[1], output_sizes[2], output_sizes[3],
                            filter_sizes[1], filter_sizes[2]]
                        and filter_sizes[0] == output_sizes[1]
                        and _compact_affine_map(input_window["map"])
                        == expected_window_map
                    )
                if legal:
                    input_source = _to_tensor_memref_source(
                        text, input_window["source"], conv_inst.span[0])
                    filter_source = _to_tensor_memref_source(
                        text, filter_slice["source"], conv_inst.span[0])
                    bias_source = _to_tensor_memref_source(
                        text, bias_slice["source"], conv_inst.span[0])
                    output_source = _to_tensor_memref_source(
                        text, output_slice["source"], conv_inst.span[0])
                    legal = all(source is not None for source in (
                        input_source, filter_source, bias_source, output_source))
                if legal:
                    physical_operands = [
                        input_source[0], filter_source[0], bias_source[0],
                        output_source[0]]
                    physical_types = [
                        input_source[1], filter_source[1], bias_source[1],
                        output_source[1]]
                    values = [_constant_index_value(text, size) if size != "1" else 1
                              for size in output_sizes + filter_sizes]
                    batch, channels, height, width, filter_channels, kh, kw = values
                    legal = (
                        all(value is not None and value > 0 for value in values)
                        and batch == 1
                        and channels == filter_channels
                        and kh == kw == 3
                        and len(set(physical_operands)) == 4
                        and _plain_f32_memrefs(physical_types, [4, 3, 1, 4])
                        and _plain_shape_compatible(
                            physical_types[0], "f32",
                            [batch, channels, height, width])
                        and _plain_shape_compatible(
                            physical_types[1], "f32", [channels, kh, kw])
                        and _plain_shape_compatible(
                            physical_types[2], "f32", [channels])
                        and _plain_shape_compatible(
                            physical_types[3], "f32",
                            [batch, channels, height, width])
                    )
                if legal:
                    legal = _fixed_depthwise_padding_body_legal(
                        text, bodies[i + 1], height, width)
                if legal:
                    output_pattern = r",\s*".join(map(re.escape, output_sizes))
                    custom_tail = re.match(
                        rf"\s*(%[\w_\-]+)\s*=\s*tensor\.insert_slice\s+"
                        rf"{re.escape(conv_inst.result_ssa)}\s+into\s+"
                        rf"{re.escape(output_slice['source'])}"
                        rf"\[0,\s*0,\s*0,\s*0\]\s*\[{output_pattern}\]\s*"
                        rf"\[1,\s*1,\s*1,\s*1\][^\n]*\n"
                        rf"\s*(%[\w_\-]+)\s*=\s*bufferization\.to_memref\s+\1"
                        rf"\s*:\s*[^\n]+\n\s*memref\.copy\s+\2\s*,\s*"
                        rf"{re.escape(physical_operands[3])}\s*:\s*[^\n]+",
                        text[conv_inst.span[1]:])
                    legal = custom_tail is not None

            if not legal:
                report.append(("fixed_tensor_convolution_layout_reject", i,
                               entry.name))
                i += 1
                continue

            uid = conv_inst.result_ssa.lstrip("%").replace(".", "_")
            dynamic_types = [
                "memref<?x?x?x?xf32>", "memref<?x?x?x?xf32>",
                "memref<?x?x?x?xf32>"
            ]
            if entry.name == "cudnnDepthwiseConvolution2D_f32_memref":
                dynamic_types = [
                    "memref<?x?x?x?xf32>", "memref<?x?x?xf32>",
                    "memref<?xf32>", "memref<?x?x?x?xf32>"]
            cast_names = [
                f"%fixed_conv_{uid}_{operand_index}"
                for operand_index in range(len(physical_operands))]
            lines = [
                f"{conv_inst.indent}{cast_name} = memref.cast {operand} : "
                f"{operand_type} to {dynamic_type}"
                for cast_name, operand, operand_type, dynamic_type in zip(
                    cast_names, physical_operands, physical_types, dynamic_types)
            ]
            lines.append(
                f"{conv_inst.indent}kernel.launch @{entry.name}("
                f"{', '.join(cast_names)}) : "
                f"({', '.join(dynamic_types)}) -> ()")
            custom_launch_line = "\n".join(lines)
            replace_full_span = True
            custom_edit_span = (
                init_inst.span[0], conv_inst.span[1] + custom_tail.end())
            operands = []
            operand_types = []
            binds = {}

        if entry.name == "cudnnAveragePool_f32_r5":
            init_inst = instances[i]
            pool_inst = instances[i + 1]
            init_outs = _extract_ssa_names(init_inst.outs_part)
            pool_ins = _extract_ssa_names(pool_inst.ins_part)
            pool_outs = _extract_ssa_names(pool_inst.outs_part)
            maps = [_compact_affine_map(m) for m in bodies[i + 1].indexing_maps]
            expected_maps = [
                "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->"
                "(d0,d1,d2,d3,d4,d5,d6,d7)>",
                "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->"
                "(d0,d1,d2,d3,d4)>",
            ]
            legal = (
                n == 2
                and len(init_outs) == 1
                and init_inst.result_ssa is not None
                and len(pool_ins) == 1
                and pool_outs == [init_inst.result_ssa]
                and pool_inst.result_ssa is not None
                and maps == expected_maps
            )

            prefix = text[:pool_inst.span[0]]
            window_match = None
            if legal:
                window_match = re.search(
                    rf"^\s*{re.escape(pool_ins[0])}\s*=\s*polygeist\.submap\s*"
                    rf"\(\s*(%[\w_\-]+)\s*,\s*([^)]+)\)\s*"
                    rf"\{{\s*map\s*=\s*([^}}]+)\}}\s*:",
                    prefix, re.MULTILINE)
                legal = window_match is not None

            input_tensor = output_tensor = input_base = output_base = None
            input_type = output_type = None
            sizes: list[str] = []
            if legal:
                input_tensor = window_match.group(1)
                sizes = [part.strip() for part in window_match.group(2).split(",")]
                window_map = _compact_affine_map(_resolve_affine_map_text(
                    prefix, window_match.group(3).strip()))
                expected_window_map = (
                    "affine_map<(d0,d1,d2,d3,d4,d5,d6,d7)->"
                    "(d0,d1,d5+d2*2,d6+d3*2,d7+d4*2)>"
                )
                legal = len(sizes) == 8 and window_map == expected_window_map

            slice_match = None
            if legal:
                slice_match = re.search(
                    rf"^\s*{re.escape(init_outs[0])}\s*=\s*"
                    rf"tensor\.extract_slice\s+(%[\w_\-]+)"
                    rf"\[0,\s*0,\s*0,\s*0,\s*0\]\s*"
                    rf"\[([^]]+)\]\s*\[1,\s*1,\s*1,\s*1,\s*1\]",
                    text[:init_inst.span[0]], re.MULTILINE)
                legal = slice_match is not None
            if legal:
                output_tensor = slice_match.group(1)
                output_sizes = [
                    part.strip() for part in slice_match.group(2).split(",")
                ]
                legal = output_sizes == sizes[:5]

            def _to_tensor_source(value: str | None):
                if value is None:
                    return None
                match = re.search(
                    rf"^\s*{re.escape(value)}\s*=\s*bufferization\.to_tensor\s+"
                    rf"(%[\w_\-]+)(?:\s+[^:]*)?\s*:\s*(memref<[^\n]+>)$",
                    text[:init_inst.span[0]], re.MULTILINE)
                return (match.group(1), match.group(2).strip()) if match else None

            input_source = _to_tensor_source(input_tensor) if legal else None
            output_source = _to_tensor_source(output_tensor) if legal else None
            legal = legal and input_source is not None and output_source is not None
            if legal:
                input_base, input_type = input_source
                output_base, output_type = output_source
                size_values = [_constant_index_value(text, value) for value in sizes]

                def _shape_compatible(ty: str, expected: list[int]) -> bool:
                    payload = _type_payload(ty, "memref")
                    if payload is None:
                        return False
                    shaped = _top_level_first_type_piece(payload)
                    suffix = "xf32"
                    if not shaped.endswith(suffix):
                        return False
                    dims = shaped[:-len(suffix)].split("x")
                    return len(dims) == len(expected) and all(
                        dim == "?" or (dim.isdigit() and int(dim) == want)
                        for dim, want in zip(dims, expected)
                    )

                legal = (
                    size_values == [2, 3, 4, 4, 4, 2, 2, 2]
                    and input_base != output_base
                    and _plain_f32_memrefs(
                        [input_type, output_type], [5, 5])
                    and _shape_compatible(input_type, [2, 3, 8, 8, 8])
                    and _shape_compatible(output_type, [2, 3, 4, 4, 4])
                )

            tail_match = None
            if legal:
                output_size_pattern = r",\s*".join(
                    map(re.escape, sizes[:5]))
                tail_match = re.match(
                    rf"\s*(%[\w_\-]+)\s*=\s*tensor\.insert_slice\s+"
                    rf"{re.escape(pool_inst.result_ssa)}\s+into\s+"
                    rf"{re.escape(output_tensor)}\[0,\s*0,\s*0,\s*0,\s*0\]"
                    rf"\s*\[{output_size_pattern}\]"
                    rf"\s*\[1,\s*1,\s*1,\s*1,\s*1\][^\n]*\n"
                    rf"\s*(%[\w_\-]+)\s*=\s*bufferization\.to_memref\s+\1"
                    rf"\s*:\s*[^\n]+\n\s*memref\.copy\s+\2,\s*"
                    rf"{re.escape(output_base)}\s*:[^\n]+",
                    text[pool_inst.span[1]:])
                legal = tail_match is not None

            if not legal:
                report.append(("fixed_average_pool3d_layout_reject", i,
                               entry.name))
                i += 1
                continue

            uid = pool_inst.result_ssa.lstrip("%").replace(".", "_")
            input_cast = f"%avgpool3d_{uid}_input"
            output_cast = f"%avgpool3d_{uid}_output"
            dynamic_type = "memref<?x?x?x?x?xf32>"
            constants = [4, 3, 2, 3, 8, 8, 8, 4, 4, 4]
            labels = ("op", "rank", "n", "c", "i0", "i1", "i2",
                      "o0", "o1", "o2")
            constant_ssas = [f"%avgpool3d_{uid}_{label}" for label in labels]
            lines = [
                f"{pool_inst.indent}{input_cast} = memref.cast {input_base} : "
                f"{input_type} to {dynamic_type}",
                f"{pool_inst.indent}{output_cast} = memref.cast {output_base} : "
                f"{output_type} to {dynamic_type}",
            ]
            lines.extend(
                f"{pool_inst.indent}{name} = arith.constant {value} : i32"
                for name, value in zip(constant_ssas, constants)
            )
            lines.append(
                f"{pool_inst.indent}kernel.launch @cudnnAveragePool_f32_r5("
                f"{', '.join(constant_ssas + [input_cast, output_cast])}) : "
                f"({', '.join(['i32'] * 10 + [dynamic_type, dynamic_type])}) "
                f"-> ()")
            custom_launch_line = "\n".join(lines)
            replace_full_span = True
            custom_edit_span = (
                init_inst.span[0], pool_inst.span[1] + tail_match.end())
            operands = []
            operand_types = []
            binds = {}

        if entry.name == "cutensorKroneckerProduct2D_f32_memref":
            kron_inst = instances[i]
            kron_ins = _extract_ssa_names(kron_inst.ins_part)
            kron_outs = _extract_ssa_names(kron_inst.outs_part)
            maps = [_compact_affine_map(m) for m in bodies[i].indexing_maps]
            expected_maps = [
                "affine_map<(d0,d1,d2,d3)->(d0,d1)>",
                "affine_map<(d0,d1,d2,d3)->(d2,d3)>",
                "affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>",
            ]
            canonical_layout = (
                n == 1
                and len(kron_ins) == 2
                and len(kron_outs) == 1
                and kron_inst.result_ssa is not None
                and maps == expected_maps
            )
            # Joint multi-root debufferization can expand all three operands
            # to the common rank-4 iteration space.  In that spelling the
            # generic maps are identities and the meaningful Kronecker maps
            # live on the three polygeist.submap producers instead.
            expanded_views = (
                [_tensor_submap_info(text, value, kron_inst.span[0])
                 for value in kron_ins + kron_outs]
                if (n == 1 and len(kron_ins) == 2 and len(kron_outs) == 1
                    and kron_inst.result_ssa is not None
                    and maps == [expected_maps[2]] * 3)
                else []
            )
            expanded_layout = (
                len(expanded_views) == 3
                and all(view is not None for view in expanded_views)
                and [_compact_affine_map(view["map"])
                     for view in expanded_views[:2]] == expected_maps[:2]
                and expanded_views[0]["sizes"] == expanded_views[1]["sizes"]
                and expanded_views[0]["sizes"] == expanded_views[2]["sizes"]
            )
            legal = canonical_layout or expanded_layout

            slices: list[tuple[str, list[str]]] = []
            if legal and not expanded_layout:
                prefix = text[:kron_inst.span[0]]
                for value in kron_ins:
                    slice_match = re.search(
                        rf"^\s*{re.escape(value)}\s*=\s*"
                        rf"tensor\.extract_slice\s+(%[\w_\-]+)"
                        rf"\[0,\s*0\]\s*\[([^]]+)\]\s*\[1,\s*1\]",
                        prefix,
                        re.MULTILINE,
                    )
                    if slice_match is None:
                        legal = False
                        break
                    slice_sizes = [
                        part.strip() for part in slice_match.group(2).split(",")
                    ]
                    if len(slice_sizes) != 2:
                        legal = False
                        break
                    slices.append((slice_match.group(1), slice_sizes))

            output_tensor = output_map_ref = None
            logical_sizes: list[str] = []
            prefix = text[:kron_inst.span[0]]
            if legal and expanded_layout:
                output_tensor = expanded_views[2]["source"]
                logical_sizes = expanded_views[2]["sizes"]
                output_map_ref = expanded_views[2]["map"]
                legal = len(logical_sizes) == 4
            elif legal:
                output_match = re.search(
                    rf"^\s*{re.escape(kron_outs[0])}\s*=\s*"
                    rf"polygeist\.submap\s*\(\s*(%[\w_\-]+)\s*,\s*"
                    rf"([^)]+)\)\s*\{{\s*map\s*=\s*([^}}]+)\}}\s*:",
                    prefix,
                    re.MULTILINE,
                )
                legal = output_match is not None
            if legal and not expanded_layout:
                output_tensor = output_match.group(1)
                logical_sizes = [
                    part.strip() for part in output_match.group(2).split(",")
                ]
                output_map_ref = output_match.group(3).strip()
                legal = len(logical_sizes) == 4

            input_sources = []
            output_source = None
            size_values: list[int | None] = []
            if legal and expanded_layout:
                input_sources = [
                    _to_tensor_memref_source(
                        text, view["source"], kron_inst.span[0])
                    for view in expanded_views[:2]
                ]
                output_source = _to_tensor_memref_source(
                    text, expanded_views[2]["source"], kron_inst.span[0])
                legal = all(source is not None for source in input_sources) and (
                    output_source is not None
                )
            elif legal:
                input_sources = [
                    _to_tensor_memref_source(text, tensor, kron_inst.span[0])
                    for tensor, _ in slices
                ]
                output_source = _to_tensor_memref_source(
                    text, output_tensor, kron_inst.span[0])
                legal = all(source is not None for source in input_sources) and (
                    output_source is not None
                )
            if legal:
                size_values = [
                    _constant_index_value(text, value) for value in logical_sizes
                ]
                a, b, c, d = size_values
                expected_output_map = (
                    f"affine_map<(d0,d1,d2,d3)->"
                    f"(d2+d0*{c},d3+d1*{d})>"
                )
                resolved_output_map = _compact_affine_map(
                    _resolve_affine_map_text(prefix, output_map_ref))
                physical_operands = [
                    input_sources[0][0], input_sources[1][0], output_source[0]
                ]
                physical_types = [
                    input_sources[0][1], input_sources[1][1], output_source[1]
                ]
                input_sizes_match = (
                    expanded_views[0]["sizes"] == logical_sizes
                    and expanded_views[1]["sizes"] == logical_sizes
                    if expanded_layout else
                    slices[0][1] == logical_sizes[:2]
                    and slices[1][1] == logical_sizes[2:]
                )
                legal = (
                    all(value is not None and value > 0 for value in size_values)
                    and input_sizes_match
                    and resolved_output_map == expected_output_map
                    and len(set(physical_operands)) == 3
                    and _plain_f32_memrefs(physical_types, [2, 2, 2])
                    and _plain_shape_compatible(physical_types[0], "f32", [a, b])
                    and _plain_shape_compatible(physical_types[1], "f32", [c, d])
                    and _plain_shape_compatible(
                        physical_types[2], "f32", [a * c, b * d])
                )

            tail_match = None
            if legal:
                size_pattern = r",\s*".join(map(re.escape, logical_sizes))
                tail_match = re.match(
                    rf"\s*(%[\w_\-]+)\s*=\s*polygeist\.submapInverse\s*"
                    rf"\(\s*{re.escape(output_tensor)}\s*,\s*"
                    rf"{re.escape(kron_inst.result_ssa)}\s*,\s*{size_pattern}\)"
                    rf"\s*\{{\s*map\s*=\s*([^}}]+)\}}\s*:[^\n]+\n"
                    rf"\s*(%[\w_\-]+)\s*=\s*bufferization\.to_memref\s+\1"
                    rf"\s*:\s*[^\n]+\n\s*memref\.copy\s+\3\s*,\s*"
                    rf"{re.escape(physical_operands[2])}\s*:\s*[^\n]+",
                    text[kron_inst.span[1]:],
                )
                legal = tail_match is not None
            if legal:
                inverse_map = _compact_affine_map(_resolve_affine_map_text(
                    text[:kron_inst.span[1] + tail_match.end()],
                    tail_match.group(2).strip(),
                ))
                legal = inverse_map == expected_output_map

            if not legal:
                report.append(("kronecker_layout_reject", i, entry.name))
                i += 1
                continue

            uid = kron_inst.result_ssa.lstrip("%").replace(".", "_")
            dynamic_type = "memref<?x?xf32>"
            cast_names = [f"%kron_{uid}_{label}" for label in ("x", "y", "out")]
            lines = [
                f"{kron_inst.indent}{cast_name} = memref.cast {operand} : "
                f"{operand_type} to {dynamic_type}"
                for cast_name, operand, operand_type in zip(
                    cast_names, physical_operands, physical_types)
            ]
            lines.append(
                f"{kron_inst.indent}kernel.launch "
                f"@cutensorKroneckerProduct2D_f32_memref("
                f"{', '.join(cast_names)}) : "
                f"({', '.join([dynamic_type] * 3)}) -> ()"
            )
            custom_launch_line = "\n".join(lines)
            replace_full_span = True
            custom_edit_span = (
                kron_inst.span[0], kron_inst.span[1] + tail_match.end())
            operands = []
            operand_types = []
            binds = {}

        if entry.name == "cudnnAddrElementwise_f32_memref":
            addr_inst = instances[i]
            addr_ins = _extract_ssa_names(addr_inst.ins_part)
            addr_outs = _extract_ssa_names(addr_inst.outs_part)
            sources = [
                _to_tensor_memref_source(text, value, addr_inst.span[0])
                for value in addr_ins + addr_outs
            ]
            beta = binds.get("%beta")
            alpha = binds.get("%alpha")
            predicate = binds.get("%predicate")
            legal = (
                n == 1 and len(addr_ins) == 5 and len(addr_outs) == 1
                and addr_ins[0] == addr_ins[3]
                and addr_ins[1] == addr_ins[4]
                and all(source is not None for source in sources)
                and beta is not None and beta[0] == "Cap"
                and alpha is not None and alpha[0] == "Cap"
                and predicate is not None and predicate[0] == "Cap"
            )
            if legal:
                physical_operands = [
                    sources[2][0], sources[0][0], sources[1][0],
                    beta[1], alpha[1], sources[5][0]]
                physical_types = [
                    sources[2][1], sources[0][1], sources[1][1],
                    "f32", "f32", sources[5][1]]
                zero_cmp = re.search(
                    rf"^\s*{re.escape(predicate[1])}\s*=\s*arith\.cmpf\s+oeq,\s*"
                    rf"{re.escape(beta[1])},\s*(%[\w_\-]+)\s*:\s*f32\s*$",
                    text[:addr_inst.span[0]], re.MULTILINE)
                legal = (
                    zero_cmp is not None
                    and parse_constants(text[:addr_inst.span[0]]).get(
                        zero_cmp.group(1)) == 0.0
                    and len(set(physical_operands[:3] + physical_operands[5:])) == 4
                    and _plain_f32_memrefs(
                        physical_types[:3] + physical_types[5:], [1, 1, 1, 1])
                )
            tail = None
            if legal:
                tail = re.match(
                    rf"\s*(%[\w_\-]+)\s*=\s*bufferization\.to_memref\s+"
                    rf"{re.escape(addr_inst.result_ssa)}\s*:\s*memref<\?xf32>\s*\n"
                    rf"\s*memref\.copy\s+\1,\s*{re.escape(physical_operands[5])}\s*:[^\n]*",
                    text[addr_inst.span[1]:])
                legal = tail is not None
            if not legal:
                report.append(("aten_addr_region_reject", i, entry.name))
                i += n
                continue
            custom_launch_line = (
                f"{addr_inst.indent}kernel.launch @{entry.name}("
                f"{', '.join(physical_operands)}) : "
                f"({', '.join(physical_types)}) -> ()")
            replace_full_span = True
            custom_edit_span = (addr_inst.span[0], addr_inst.span[1] + tail.end())
            operands = []
            operand_types = []
            binds = {}

        if entry.name == "cudnnBinaryCrossEntropyMean_f32_memref":
            bce_inst = instances[i]
            bce_ins = _extract_ssa_names(bce_inst.ins_part)
            bce_outs = _extract_ssa_names(bce_inst.outs_part)
            view = (_parse_memref_view(text, bce_outs[0], bce_inst.span[0])
                    if len(bce_outs) == 1 else None)
            legal = (
                n == 1 and len(bce_ins) == 4
                and bce_ins[0] == bce_ins[2]
                and bce_ins[1] == bce_ins[3]
                and view is not None and view["kind"] == "reinterpret_cast"
                and _plain_f32_memrefs(
                    [_extract_ssa_types(bce_inst.ins_part)[0],
                     _extract_ssa_types(bce_inst.ins_part)[1],
                     view["base_type"]], [1, 1, 1])
                and len({bce_ins[0], bce_ins[1], view["base"]}) == 3
            )
            prefix = text[:bce_inst.span[0]]
            init = None
            if legal:
                init = list(re.finditer(
                    rf"^\s*affine\.store\s+(%[\w_\-]+),\s*"
                    rf"{re.escape(view['base'])}\[0\]\s*:\s*[^\n]+$",
                    prefix, re.MULTILINE))
                legal = bool(init) and parse_constants(prefix).get(
                    init[-1].group(1)) == 0.0
                if legal:
                    legal = not re.search(
                        r"\b(?:affine|memref)\.store\b|\blinalg\.",
                        text[init[-1].end():bce_inst.span[0]])
            tail = None
            if legal:
                tail = re.match(
                    rf"\s*(%[\w_\-]+)\s*=\s*affine\.load\s+"
                    rf"{re.escape(view['base'])}\[0\]\s*:[^\n]+\n"
                    rf"\s*(%[\w_\-]+)\s*=\s*arith\.divf\s+\1,\s*"
                    rf"(%[\w_\-]+)\s*:\s*f32\s*\n"
                    rf"\s*affine\.store\s+\2,\s*{re.escape(view['base'])}\[0\][^\n]*",
                    text[bce_inst.span[1]:])
                legal = tail is not None and parse_constants(text).get(
                    tail.group(3)) == 256.0
            if not legal:
                report.append(("aten_bce_region_reject", i, entry.name))
                i += n
                continue
            physical_operands = [bce_ins[1], bce_ins[0], view["base"]]
            physical_types = [_extract_ssa_types(bce_inst.ins_part)[1],
                              _extract_ssa_types(bce_inst.ins_part)[0],
                              view["base_type"]]
            custom_launch_line = (
                f"{bce_inst.indent}kernel.launch @{entry.name}("
                f"{', '.join(physical_operands)}) : "
                f"({', '.join(physical_types)}) -> ()")
            replace_full_span = True
            custom_edit_span = (init[-1].start(), bce_inst.span[1] + tail.end())
            operands = []
            operand_types = []
            binds = {}

        if entry.name == "cudnnLogSigmoid_f32_memref":
            log_inst = instances[i]
            log_ins = _extract_ssa_names(log_inst.ins_part)
            log_outs = _extract_ssa_names(log_inst.outs_part)
            sources = [
                _to_tensor_memref_source(text, value, log_inst.span[0])
                for value in log_ins + log_outs
            ]
            legal = (
                n == 1 and len(log_ins) == len(log_outs) == 2
                and log_ins[0] == log_ins[1]
                and log_inst.result_count == 2
                and all(source is not None for source in sources)
            )
            if legal:
                physical_operands = [sources[0][0], sources[3][0], sources[2][0]]
                physical_types = [sources[0][1], sources[3][1], sources[2][1]]
                legal = (len(set(physical_operands)) == 3 and
                         _plain_f32_memrefs(physical_types, [1, 1, 1]))
            tail = None
            if legal:
                tail = re.match(
                    rf"\s*(%[\w_\-]+)\s*=\s*bufferization\.to_memref\s+"
                    rf"{re.escape(log_inst.result_ssa)}#1\s*:[^\n]+\n"
                    rf"\s*memref\.copy\s+\1,\s*{re.escape(physical_operands[1])}\s*:[^\n]+\n"
                    rf"\s*(%[\w_\-]+)\s*=\s*bufferization\.to_memref\s+"
                    rf"{re.escape(log_inst.result_ssa)}#0\s*:[^\n]+\n"
                    rf"\s*memref\.copy\s+\2,\s*{re.escape(physical_operands[2])}\s*:[^\n]*",
                    text[log_inst.span[1]:])
                legal = tail is not None
            if not legal:
                report.append(("aten_log_sigmoid_region_reject", i, entry.name))
                i += n
                continue
            custom_launch_line = (
                f"{log_inst.indent}kernel.launch @{entry.name}("
                f"{', '.join(physical_operands)}) : "
                f"({', '.join(physical_types)}) -> ()")
            replace_full_span = True
            custom_edit_span = (log_inst.span[0], log_inst.span[1] + tail.end())
            operands = []
            operand_types = []
            binds = {}

        if entry.name == "cubExclusiveSum1D_i32_memref":
            scan_inst = instances[i]
            scan_ins = _extract_ssa_names(scan_inst.ins_part)
            scan_outs = _extract_ssa_names(scan_inst.outs_part)
            slices = [
                _tensor_extract_slice_info(text, value, scan_inst.span[0])
                for value in scan_ins + scan_outs
            ]
            legal = n == 1 and len(scan_ins) == 2 and len(scan_outs) == 1 and all(
                item is not None for item in slices)
            inserted = None
            if legal:
                inserted = re.search(
                    rf"^\s*({re.escape(slices[0]['source'])})\s*=\s*"
                    rf"tensor\.insert\s+(%[\w_\-]+)\s+into\s+"
                    rf"(%[\w_\-]+)\[(%[\w_\-]+)\]",
                    text[:scan_inst.span[0]], re.MULTILINE)
                count = _constant_index_value(text, slices[0]["sizes"][0])
                zero_index = (_constant_index_value(text, inserted.group(4))
                              if inserted else None)
                zero_value = (parse_constants(text).get(inserted.group(2))
                              if inserted else None)
                legal = (
                    inserted is not None and count == 64
                    and zero_index == 0 and zero_value == 0
                    and slices[0]["source"] == slices[2]["source"]
                    and slices[0]["offsets"] == ["0"]
                    and slices[1]["offsets"] == ["0"]
                    and slices[2]["offsets"] == ["1"]
                    and slices[0]["sizes"] == slices[1]["sizes"] == slices[2]["sizes"]
                    and slices[0]["strides"] == slices[1]["strides"] == slices[2]["strides"] == ["1"]
                )
            if legal:
                input_source = _to_tensor_memref_source(
                    text, slices[1]["source"], scan_inst.span[0])
                output_source = _to_tensor_memref_source(
                    text, inserted.group(3), scan_inst.span[0])
                legal = input_source is not None and output_source is not None
            if legal:
                physical_operands = [input_source[0], output_source[0]]
                physical_types = [input_source[1], output_source[1]]
                legal = (
                    physical_operands[0] != physical_operands[1]
                    and [_shaped_rank(t) for t in physical_types] == [1, 1]
                    and [_sniff_elem_type(t) for t in physical_types] == ["i32", "i32"]
                    and all("," not in t for t in physical_types)
                )
            tail = None
            if legal:
                size = re.escape(slices[0]["sizes"][0])
                tail = re.match(
                    rf"\s*(%[\w_\-]+)\s*=\s*tensor\.insert_slice\s+"
                    rf"{re.escape(scan_inst.result_ssa)}\s+into\s+"
                    rf"{re.escape(slices[0]['source'])}\[1\]\s*\[{size}\]\s*\[1\][^\n]*\n"
                    rf"\s*(%[\w_\-]+)\s*=\s*bufferization\.to_memref\s+\1\s*:[^\n]+\n"
                    rf"\s*memref\.copy\s+\2,\s*{re.escape(physical_operands[1])}\s*:[^\n]*",
                    text[scan_inst.span[1]:])
                legal = tail is not None
            if not legal:
                report.append(("aten_nested_offsets_region_reject", i, entry.name))
                i += n
                continue
            custom_launch_line = (
                f"{scan_inst.indent}kernel.launch @{entry.name}("
                f"{', '.join(physical_operands)}) : "
                f"({', '.join(physical_types)}) -> ()")
            replace_full_span = True
            custom_edit_span = (inserted.start(), scan_inst.span[1] + tail.end())
            operands = []
            operand_types = []
            binds = {}

        if entry.name == "cudnnTransformBiasRescaleQKV_f32_memref":
            qkv_insts = instances[i:i + n]
            legal = n == 3 and all(inst.result_ssa is not None for inst in qkv_insts)
            qkv_slices = []
            bias_slices = []
            output_slices = []
            expanded_qkv = False
            if legal:
                operand_names = []
                for inst in qkv_insts:
                    names_in = _extract_ssa_names(inst.ins_part)
                    names_out = _extract_ssa_names(inst.outs_part)
                    if len(names_in) != 2 or len(names_out) != 1:
                        legal = False
                        break
                    operand_names.append((names_in[0], names_in[1],
                                          names_out[0]))
            if legal:
                submap_groups = [
                    tuple(_tensor_submap_info(text, value, inst.span[0])
                          for value in names)
                    for inst, names in zip(qkv_insts, operand_names)
                ]
                expanded_qkv = all(
                    item is not None for group in submap_groups
                    for item in group)
                if expanded_qkv:
                    qkv_slices = [group[0] for group in submap_groups]
                    bias_slices = [group[1] for group in submap_groups]
                    output_slices = [group[2] for group in submap_groups]
                else:
                    qkv_slices = [_tensor_extract_slice_info(
                        text, names[0], inst.span[0])
                        for inst, names in zip(qkv_insts, operand_names)]
                    bias_slices = [_tensor_extract_slice_info(
                        text, names[1], inst.span[0])
                        for inst, names in zip(qkv_insts, operand_names)]
                    output_slices = [_tensor_extract_slice_info(
                        text, names[2], inst.span[0])
                        for inst, names in zip(qkv_insts, operand_names)]
                legal = legal and all(item is not None for item in
                                      qkv_slices + bias_slices + output_slices)
            if legal:
                qkv_tensor = qkv_slices[0]["source"]
                bias_tensor = bias_slices[0]["source"]
                output_tensors = [item["source"] for item in output_slices]
                common_legal = (
                    [item["source"] for item in qkv_slices] ==
                    [qkv_tensor] * 3
                    and [item["source"] for item in bias_slices] ==
                    [bias_tensor] * 3
                    and len(set(output_tensors)) == 3)
                if expanded_qkv:
                    expected_qkv_maps = [
                        "affine_map<(d0,d1,d2,d3)->"
                        f"(d0,d1,{j},d2,d3)>" for j in range(3)]
                    expected_bias_maps = [
                        "affine_map<(d0,d1,d2,d3)->"
                        f"({j},d2,d3)>" for j in range(3)]
                    expected_output_map = (
                        "affine_map<(d0,d1,d2,d3)->(d0,d2,d1,d3)>")
                    body_maps = [
                        [_compact_affine_map(mapping)
                         for mapping in body.indexing_maps]
                        for body in bodies[i:i + 3]]
                    legal = (
                        common_legal
                        and [_compact_affine_map(item["map"])
                             for item in qkv_slices] == expected_qkv_maps
                        and [_compact_affine_map(item["map"])
                             for item in bias_slices] == expected_bias_maps
                        and all(_compact_affine_map(item["map"]) ==
                                expected_output_map
                                for item in output_slices)
                        and all(item["sizes"] == qkv_slices[0]["sizes"]
                                for item in qkv_slices + bias_slices +
                                output_slices)
                        and len(qkv_slices[0]["sizes"]) == 4
                        and all(len(set(maps)) == 1 for maps in body_maps)
                        and all(maps[0] ==
                                "affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>"
                                for maps in body_maps))
                else:
                    expected_maps = [
                        "affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>",
                        "affine_map<(d0,d1,d2,d3)->(d2,d3)>",
                        "affine_map<(d0,d1,d2,d3)->(d0,d2,d1,d3)>",
                    ]
                    legal = (
                        common_legal
                        and [item["offsets"] for item in qkv_slices] == [
                            ["0", "0", str(j), "0", "0"] for j in range(3)]
                        and [item["offsets"] for item in bias_slices] == [
                            [str(j), "0", "0"] for j in range(3)]
                        and all(item["offsets"] == ["0"] * 4
                                for item in output_slices)
                        and all(item["strides"] == ["1"] * 5
                                for item in qkv_slices)
                        and all(item["strides"] == ["1"] * 3
                                for item in bias_slices)
                        and all(item["strides"] == ["1"] * 4
                                for item in output_slices)
                        and all([_compact_affine_map(m)
                                 for m in body.indexing_maps] == expected_maps
                                for body in bodies[i:i + 3]))
            if legal:
                if expanded_qkv:
                    logical_values = [
                        _constant_index_value(text, value)
                        for value in qkv_slices[0]["sizes"]]
                    legal = all(value is not None and value > 0
                                for value in logical_values)
                else:
                    qkv_values = [_constant_index_value(text, value)
                                  if value != "1" else 1
                                  for value in qkv_slices[0]["sizes"]]
                    bias_values = [_constant_index_value(text, value)
                                   if value != "1" else 1
                                   for value in bias_slices[0]["sizes"]]
                    output_values = [_constant_index_value(text, value)
                                     if value != "1" else 1
                                     for value in output_slices[0]["sizes"]]
                    legal = (
                        qkv_values == [2, 16, 1, 4, 8]
                        and bias_values == [1, 4, 8]
                        and output_values == [2, 4, 16, 8]
                        and all(item["sizes"] == qkv_slices[0]["sizes"]
                                for item in qkv_slices)
                        and all(item["sizes"] == bias_slices[0]["sizes"]
                                for item in bias_slices)
                        and all(item["sizes"] == output_slices[0]["sizes"]
                                for item in output_slices))
            if legal:
                sources = [_to_tensor_memref_source(text, value, qkv_insts[0].span[0])
                           for value in [qkv_tensor, bias_tensor] + output_tensors]
                scale = binds.get("%scale")
                legal = (all(source is not None for source in sources)
                         and scale is not None and scale[0] == "Cap"
                         and scalar_types.get(scale[1]) == "f32")
            if legal:
                physical_operands = [sources[0][0], sources[1][0], scale[1]] + [
                    source[0] for source in sources[2:]]
                physical_types = [sources[0][1], sources[1][1], "f32"] + [
                    source[1] for source in sources[2:]]
                memref_operands = physical_operands[:2] + physical_operands[3:]
                memref_types = physical_types[:2] + physical_types[3:]
                legal = (len(set(memref_operands)) == 5 and
                         _plain_f32_memrefs(memref_types, [5, 3, 4, 4, 4]))
                if legal and expanded_qkv:
                    batch, sequence, heads, width = logical_values
                    legal = (
                        _plain_shape_compatible(
                            memref_types[0], "f32",
                            [batch, sequence, 3, heads, width])
                        and _plain_shape_compatible(
                            memref_types[1], "f32", [3, heads, width])
                        and all(_plain_shape_compatible(
                            output_type, "f32",
                            [batch, heads, sequence, width])
                            for output_type in memref_types[2:]))
            tails = []
            if legal:
                for inst, output_slice, physical_output in zip(
                        qkv_insts, output_slices, physical_operands[3:]):
                    sizes = r",\s*".join(map(re.escape, output_slice["sizes"]))
                    if expanded_qkv:
                        tail = re.match(
                            rf"\s*(%[\w_\-]+)\s*=\s*"
                            rf"polygeist\.submapInverse\s*\(\s*"
                            rf"{re.escape(output_slice['source'])}\s*,\s*"
                            rf"{re.escape(inst.result_ssa)}\s*,\s*{sizes}\)"
                            rf"\s*\{{\s*map\s*=\s*([^}}]+)\}}\s*:[^\n]+\n"
                            rf"\s*(%[\w_\-]+)\s*=\s*"
                            rf"bufferization\.to_memref\s+\1\s*:[^\n]+\n"
                            rf"\s*memref\.copy\s+\3,\s*"
                            rf"{re.escape(physical_output)}\s*:[^\n]*",
                            text[inst.span[1]:])
                        if tail is not None:
                            inverse_map = _compact_affine_map(
                                _resolve_affine_map_text(
                                    text[:inst.span[1] + tail.end()],
                                    tail.group(2).strip()))
                            if inverse_map != _compact_affine_map(
                                    output_slice["map"]):
                                tail = None
                    else:
                        tail = re.match(
                            rf"\s*(%[\w_\-]+)\s*=\s*tensor\.insert_slice\s+"
                            rf"{re.escape(inst.result_ssa)}\s+into\s+"
                            rf"{re.escape(output_slice['source'])}"
                            rf"\[0,\s*0,\s*0,\s*0\]"
                            rf"\s*\[{sizes}\]\s*"
                            rf"\[1,\s*1,\s*1,\s*1\][^\n]*\n"
                            rf"\s*(%[\w_\-]+)\s*=\s*"
                            rf"bufferization\.to_memref\s+\1\s*:[^\n]+\n"
                            rf"\s*memref\.copy\s+\2,\s*"
                            rf"{re.escape(physical_output)}\s*:[^\n]*",
                            text[inst.span[1]:])
                    if tail is None:
                        legal = False
                        break
                    tails.append(tail)
            if not legal:
                report.append(("aten_qkv_region_reject", list(range(i, i + n)), entry.name))
                i += n
                continue
            uid = qkv_insts[0].result_ssa.lstrip("%").replace(".", "_")
            dynamic_memref_types = [
                "memref<?x?x?x?x?xf32>", "memref<?x?x?xf32>",
                "memref<?x?x?x?xf32>", "memref<?x?x?x?xf32>",
                "memref<?x?x?x?xf32>",
            ]
            cast_names = [f"%aten_qkv_{uid}_{j}" for j in range(5)]
            cast_sources = physical_operands[:2] + physical_operands[3:]
            cast_source_types = physical_types[:2] + physical_types[3:]
            lines = [
                f"{qkv_insts[0].indent}{cast} = memref.cast {source} : "
                f"{source_type} to {dynamic_type}"
                for cast, source, source_type, dynamic_type in zip(
                    cast_names, cast_sources, cast_source_types,
                    dynamic_memref_types)
            ]
            launch_operands = cast_names[:2] + [physical_operands[2]] + cast_names[2:]
            launch_types = dynamic_memref_types[:2] + ["f32"] + dynamic_memref_types[2:]
            lines.append(
                f"{qkv_insts[0].indent}kernel.launch @{entry.name}("
                f"{', '.join(launch_operands)}) : "
                f"({', '.join(launch_types)}) -> ()")
            custom_launch_line = "\n".join(lines)
            replace_full_span = True
            custom_edit_span = (qkv_insts[0].span[0],
                                qkv_insts[-1].span[1] + tails[-1].end())
            operands = []
            operand_types = []
            binds = {}

        if entry.name == "cublasSgemvTZero_memref":
            init_inst, gemv_inst = instances[i:i + 2]
            init_outs = _extract_ssa_names(init_inst.outs_part)
            gemv_ins = _extract_ssa_names(gemv_inst.ins_part)
            gemv_outs = _extract_ssa_names(gemv_inst.outs_part)
            views = [
                _parse_memref_view(text, value, gemv_inst.span[0])
                for value in gemv_ins + gemv_outs
            ]
            maps = [_compact_affine_map(m) for m in bodies[i + 1].indexing_maps]
            expected_maps = [
                "affine_map<(d0,d1)->(d1,d0)>",
                "affine_map<(d0,d1)->(d1)>",
                "affine_map<(d0,d1)->(d0)>",
            ]
            legal = (
                n == 2 and len(init_outs) == 1
                and len(gemv_ins) == 2 and len(gemv_outs) == 1
                and all(view is not None for view in views)
                and views[0]["kind"] == views[1]["kind"] == "subview"
                and views[2]["kind"] == "reinterpret_cast"
                and init_outs[0] == views[2]["base"]
                and maps == expected_maps
                and views[0]["sizes"] == ["%c64", "%c128"]
                and views[1]["sizes"] == ["%c64"]
            )
            if legal:
                reinterpret = re.search(
                    rf"^\s*{re.escape(gemv_outs[0])}\s*=\s*"
                    rf"memref\.reinterpret_cast\s+{re.escape(views[2]['base'])}"
                    rf"\s+to\s+offset:\s*\[0\],\s*sizes:\s*\[(%[\w_\-]+)\],"
                    rf"\s*strides:\s*\[1\]",
                    text[:gemv_inst.span[0]], re.MULTILINE)
                legal = (reinterpret is not None and
                         _constant_index_value(text, reinterpret.group(1)) == 128)
            if legal:
                physical_operands = [
                    views[0]["base"], views[1]["base"], views[2]["base"]]
                physical_types = [
                    views[0]["base_type"], views[1]["base_type"],
                    views[2]["base_type"]]
                legal = (
                    len(set(physical_operands)) == 3
                    and _plain_f32_memrefs(physical_types, [2, 1, 1])
                    and _plain_shape_compatible(
                        physical_types[0], "f32", [64, 128])
                )
            if not legal:
                report.append(("aten_gemv_transpose_region_reject",
                               [i, i + 1], entry.name))
                i += n
                continue
            uid = gemv_inst.result_ssa.lstrip("%") if gemv_inst.result_ssa else str(i)
            matrix_cast = f"%aten_gemvt_{uid}_matrix"
            lines = [
                f"{gemv_inst.indent}{matrix_cast} = memref.cast "
                f"{physical_operands[0]} : {physical_types[0]} to memref<?x?xf32>",
                f"{gemv_inst.indent}kernel.launch @{entry.name}("
                f"{matrix_cast}, {physical_operands[1]}, {physical_operands[2]}) "
                "{polygeist.fixed_extents = array<i64: 64, 128>} : "
                f"(memref<?x?xf32>, {physical_types[1]}, {physical_types[2]}) -> ()",
            ]
            custom_launch_line = "\n".join(lines)
            replace_full_span = True
            custom_edit_span = (init_inst.span[0], gemv_inst.span[1])
            operands = []
            operand_types = []
            binds = {}

        if entry.name == "cubSegmentedLogicalAnd_i32_memref":
            init_inst, reduce_inst = instances[i:i + 2]
            init_outs = _extract_ssa_names(init_inst.outs_part)
            reduce_ins = _extract_ssa_names(reduce_inst.ins_part)
            reduce_outs = _extract_ssa_names(reduce_inst.outs_part)
            input_view = (_parse_memref_view(
                text, reduce_ins[0], reduce_inst.span[0])
                if len(reduce_ins) == 1 else None)
            output_view = (_parse_memref_view(
                text, reduce_outs[0], reduce_inst.span[0])
                if len(reduce_outs) == 1 else None)
            maps = [_compact_affine_map(m)
                    for m in bodies[i + 1].indexing_maps]
            legal = (
                n == 2 and len(init_outs) == 1
                and input_view is not None and output_view is not None
                and input_view["kind"] == "subview"
                and output_view["kind"] == "reinterpret_cast"
                and input_view["sizes"] == ["%c32", "%c64"]
                and output_view["base"] == init_outs[0]
                and maps == [
                    "affine_map<(d0,d1)->(d0,d1)>",
                    "affine_map<(d0,d1)->(d0)>",
                ]
                and _constant_index_value(text, "%c32") == 32
                and _constant_index_value(text, "%c64") == 64
                and _shaped_rank(input_view["base_type"]) == 2
                and _shaped_rank(output_view["base_type"]) == 1
                and _sniff_elem_type(input_view["base_type"]) == "i32"
                and _sniff_elem_type(output_view["base_type"]) == "i32"
                and input_view["base"] != output_view["base"]
            )
            if not legal:
                report.append(("aten_segmented_logical_region_reject",
                               [i, i + 1], entry.name))
                i += n
                continue
            custom_launch_line = (
                f"{reduce_inst.indent}kernel.launch @{entry.name}("
                f"{input_view['base']}, {output_view['base']}) "
                "{polygeist.fixed_extents = array<i64: 32, 64>} : "
                f"({input_view['base_type']}, {output_view['base_type']}) -> ()"
            )
            replace_full_span = True
            custom_edit_span = (init_inst.span[0], reduce_inst.span[1])
            operands = []
            operand_types = []
            binds = {}

        if entry.name == "atenSegmentedSum" and n == 2:
            init_inst, reduce_inst = instances[i:i + 2]
            init_outs = _extract_ssa_names(init_inst.outs_part)
            reduce_ins = _extract_ssa_names(reduce_inst.ins_part)
            reduce_outs = _extract_ssa_names(reduce_inst.outs_part)
            input_view = (_tensor_submap_info(
                text, reduce_ins[0], reduce_inst.span[0])
                if len(reduce_ins) == 1 else None)
            output_view = (_tensor_submap_info(
                text, reduce_outs[0], reduce_inst.span[0])
                if len(reduce_outs) == 1 else None)
            init_view = (_tensor_submap_info(
                text, init_outs[0], init_inst.span[0])
                if len(init_outs) == 1 else None)
            expanded_legal = (
                input_view is not None and output_view is not None
                and init_view is not None
                and input_view["sizes"] == output_view["sizes"]
                and len(input_view["sizes"]) == 2
                and _compact_affine_map(input_view["map"]) ==
                    "affine_map<(d0,d1)->(d0,d1)>"
                and _compact_affine_map(output_view["map"]) ==
                    "affine_map<(d0,d1)->(d0)>"
            )
            if expanded_legal:
                input_source = _to_tensor_memref_source(
                    text, input_view["source"], init_inst.span[0])
                output_source = _to_tensor_memref_source(
                    text, init_view["source"], init_inst.span[0])
                sizes = [_constant_index_value(text, value)
                         for value in input_view["sizes"]]
                expanded_legal = (
                    input_source is not None and output_source is not None
                    and sizes[0] is not None and sizes[0] > 0
                    and sizes[1] == 64
                    and [_shaped_rank(input_source[1]),
                         _shaped_rank(output_source[1])] == [2, 1]
                    and _sniff_elem_type(input_source[1]) in ("f32", "f64")
                    and _sniff_elem_type(output_source[1]) ==
                        _sniff_elem_type(input_source[1])
                    and "," not in input_source[1]
                    and "," not in output_source[1]
                    and input_source[0] != output_source[0]
                )
            expanded_tail = None
            if expanded_legal and reduce_inst.result_ssa is not None:
                expanded_tail = re.match(
                    rf"\s*(%[\w_\-]+)\s*=\s*polygeist\.submapInverse\s*"
                    rf"\([^,\n]+,\s*{re.escape(reduce_inst.result_ssa)}\s*,"
                    rf"[^\n]*\)[^\n]*\n\s*(%[\w_\-]+)\s*=\s*"
                    rf"bufferization\.to_memref\s+\1\s*:[^\n]+\n"
                    rf"\s*memref\.copy\s+\2,\s*"
                    rf"{re.escape(output_source[0])}\s*:[^\n]*",
                    text[reduce_inst.span[1]:])
                expanded_legal = expanded_tail is not None
            if expanded_legal:
                elem = _sniff_elem_type(input_source[1])
                emit_name = f"cubSegmentedSum_{elem}_memref"
                uid = reduce_inst.result_ssa.lstrip("%")
                input_cast = f"%aten_sum_{uid}_input"
                dynamic_input = f"memref<?x?x{elem}>"
                fixed = f"{sizes[0]}, {sizes[1]}"
                custom_launch_line = "\n".join([
                    f"{reduce_inst.indent}{input_cast} = memref.cast "
                    f"{input_source[0]} : {input_source[1]} to {dynamic_input}",
                    f"{reduce_inst.indent}kernel.launch @{emit_name}("
                    f"{input_cast}, {output_source[0]}) "
                    f"{{polygeist.fixed_extents = array<i64: {fixed}>}} : "
                    f"({dynamic_input}, {output_source[1]}) -> ()",
                ])
                replace_full_span = True
                custom_edit_span = (
                    init_inst.span[0],
                    reduce_inst.span[1] + expanded_tail.end())
                operands = []
                operand_types = []
                binds = {}

        if entry.name == "atenSegmentedSum" and custom_launch_line is None:
            init_inst, reduce_inst = instances[i:i + 2]
            init_outs = _extract_ssa_names(init_inst.outs_part)
            reduce_ins = _extract_ssa_names(reduce_inst.ins_part)
            reduce_outs = _extract_ssa_names(reduce_inst.outs_part)
            input_slice = (_tensor_extract_slice_info(
                text, reduce_ins[0], reduce_inst.span[0])
                if len(reduce_ins) == 1 else None)
            output_slice = (_tensor_extract_slice_info(
                text, reduce_outs[0], reduce_inst.span[0])
                if len(reduce_outs) == 1 else None)
            legal = (
                n == 2 and len(init_outs) == 1
                and input_slice is not None and output_slice is not None
                and output_slice["source"] == init_inst.result_ssa
                and input_slice["offsets"] == ["0", "0"]
                and output_slice["offsets"] == ["0"]
                and input_slice["strides"] == ["1", "1"]
                and output_slice["strides"] == ["1"]
                and len(input_slice["sizes"]) == 2
                and output_slice["sizes"] == [input_slice["sizes"][0]]
            )
            if legal:
                input_source = _to_tensor_memref_source(
                    text, input_slice["source"], init_inst.span[0])
                output_source = _to_tensor_memref_source(
                    text, init_outs[0], init_inst.span[0])
                legal = input_source is not None and output_source is not None
            if legal:
                elem = _sniff_elem_type(input_source[1])
                physical_operands = [input_source[0], output_source[0]]
                physical_types = [input_source[1], output_source[1]]
                sizes = [_constant_index_value(text, value)
                         for value in input_slice["sizes"]]
                legal = (
                    elem in ("f32", "f64") and sizes == [16, 64]
                    and [_shaped_rank(t) for t in physical_types] == [2, 1]
                    and [_sniff_elem_type(t) for t in physical_types] == [elem, elem]
                    and all("," not in t for t in physical_types)
                    and len(set(physical_operands)) == 2
                )
            tail = None
            if legal:
                row_size = re.escape(input_slice["sizes"][0])
                tail = re.match(
                    rf"\s*(%[\w_\-]+)\s*=\s*tensor\.insert_slice\s+"
                    rf"{re.escape(reduce_inst.result_ssa)}\s+into\s+"
                    rf"{re.escape(init_inst.result_ssa)}\[0\]\s*\[{row_size}\]\s*\[1\][^\n]*\n"
                    rf"\s*(%[\w_\-]+)\s*=\s*bufferization\.to_memref\s+\1\s*:[^\n]+\n"
                    rf"\s*memref\.copy\s+\2,\s*{re.escape(physical_operands[1])}\s*:[^\n]*",
                    text[reduce_inst.span[1]:])
                legal = tail is not None
            if not legal:
                report.append(("aten_segmented_sum_region_reject",
                               [i, i + 1], entry.name))
                i += n
                continue
            emit_name = f"cubSegmentedSum_{elem}_memref"
            uid = reduce_inst.result_ssa.lstrip("%")
            input_cast = f"%aten_sum_{uid}_input"
            dynamic_input = f"memref<?x?x{elem}>"
            lines = [
                f"{reduce_inst.indent}{input_cast} = memref.cast "
                f"{physical_operands[0]} : {physical_types[0]} to {dynamic_input}",
                f"{reduce_inst.indent}kernel.launch @{emit_name}("
                f"{input_cast}, {physical_operands[1]}) "
                "{polygeist.fixed_extents = array<i64: 16, 64>} : "
                f"({dynamic_input}, {physical_types[1]}) -> ()",
            ]
            custom_launch_line = "\n".join(lines)
            replace_full_span = True
            custom_edit_span = (
                init_inst.span[0], reduce_inst.span[1] + tail.end())
            operands = []
            operand_types = []
            binds = {}

        if entry.name in ("cubSegmentedMin_f32_memref",
                           "cubSegmentedMax_f32_memref"):
            init_inst, reduce_inst = instances[i:i + 2]
            init_ins = _extract_ssa_names(init_inst.ins_part)
            init_outs = _extract_ssa_names(init_inst.outs_part)
            reduce_ins = _extract_ssa_names(reduce_inst.ins_part)
            reduce_outs = _extract_ssa_names(reduce_inst.outs_part)
            first_slice = (_tensor_extract_slice_info(
                text, init_ins[0], init_inst.span[0]) if len(init_ins) == 1 else None)
            init_output_slice = (_tensor_extract_slice_info(
                text, init_outs[0], init_inst.span[0]) if len(init_outs) == 1 else None)
            rest_slice = (_tensor_extract_slice_info(
                text, reduce_ins[0], reduce_inst.span[0])
                if len(reduce_ins) == 1 else None)
            legal = (
                n == 2 and first_slice is not None
                and init_output_slice is not None and rest_slice is not None
                and reduce_outs == [init_inst.result_ssa]
                and first_slice["source"] == rest_slice["source"]
                and first_slice["offsets"] == ["0", "0"]
                and rest_slice["offsets"] == ["0", "1"]
                and first_slice["strides"] == rest_slice["strides"] == ["1", "1"]
                and first_slice["sizes"][0] == rest_slice["sizes"][0]
                and (first_slice["sizes"][1] == "1" or
                     _constant_index_value(
                         text, first_slice["sizes"][1]) == 1)
                and _constant_index_value(text, rest_slice["sizes"][1]) == 63
                and init_output_slice["offsets"] == ["0"]
                and init_output_slice["sizes"] == [first_slice["sizes"][0]]
                and init_output_slice["strides"] == ["1"]
            )
            if legal:
                input_source = _to_tensor_memref_source(
                    text, first_slice["source"], init_inst.span[0])
                output_source = _to_tensor_memref_source(
                    text, init_output_slice["source"], init_inst.span[0])
                legal = input_source is not None and output_source is not None
            if legal:
                physical_operands = [input_source[0], output_source[0]]
                physical_types = [input_source[1], output_source[1]]
                legal = (
                    _plain_f32_memrefs(physical_types, [2, 1])
                    and _plain_shape_compatible(
                        physical_types[0], "f32", [32, 64])
                    and physical_operands[0] != physical_operands[1]
                )
            tail = None
            if legal:
                rows = re.escape(first_slice["sizes"][0])
                tail = re.match(
                    rf"\s*(%[\w_\-]+)\s*=\s*tensor\.insert_slice\s+"
                    rf"{re.escape(reduce_inst.result_ssa)}\s+into\s+"
                    rf"{re.escape(init_output_slice['source'])}\[0\]\s*\[{rows}\]\s*\[1\][^\n]*\n"
                    rf"\s*(%[\w_\-]+)\s*=\s*bufferization\.to_memref\s+\1\s*:[^\n]+\n"
                    rf"\s*memref\.copy\s+\2,\s*{re.escape(physical_operands[1])}\s*:[^\n]*",
                    text[reduce_inst.span[1]:])
                legal = tail is not None
            if not legal:
                report.append(("aten_segmented_extreme_region_reject",
                               [i, i + 1], entry.name))
                i += n
                continue
            uid = reduce_inst.result_ssa.lstrip("%")
            input_cast = f"%aten_extreme_{uid}_input"
            lines = [
                f"{reduce_inst.indent}{input_cast} = memref.cast "
                f"{physical_operands[0]} : {physical_types[0]} to memref<?x?xf32>",
                f"{reduce_inst.indent}kernel.launch @{entry.name}("
                f"{input_cast}, {physical_operands[1]}) "
                "{polygeist.fixed_extents = array<i64: 32, 64>} : "
                f"(memref<?x?xf32>, {physical_types[1]}) -> ()",
            ]
            custom_launch_line = "\n".join(lines)
            replace_full_span = True
            custom_edit_span = (
                init_inst.span[0], reduce_inst.span[1] + tail.end())
            operands = []
            operand_types = []
            binds = {}

        if entry.name == "cutensornetNetwork_f32_n3_aten":
            init_inst, network_inst = instances[i:i + 2]
            init_outs = _extract_ssa_names(init_inst.outs_part)
            network_ins = _extract_ssa_names(network_inst.ins_part)
            network_outs = _extract_ssa_names(network_inst.outs_part)
            input_slices = [
                _tensor_extract_slice_info(text, value, network_inst.span[0])
                for value in network_ins
            ]
            output_slice = (_tensor_extract_slice_info(
                text, init_outs[0], init_inst.span[0])
                if len(init_outs) == 1 else None)
            maps = [_compact_affine_map(m)
                    for m in bodies[i + 1].indexing_maps]
            bilinear_maps = [
                "affine_map<(d0,d1,d2,d3)->(d0,d2)>",
                "affine_map<(d0,d1,d2,d3)->(d1,d2,d3)>",
                "affine_map<(d0,d1,d2,d3)->(d0,d3)>",
                "affine_map<(d0,d1,d2,d3)->(d0,d1)>",
            ]
            trilinear_maps = [
                "affine_map<(d0,d1,d2,d3)->(d0,d2)>",
                "affine_map<(d0,d1,d2,d3)->(d2,d3,d1)>",
                "affine_map<(d0,d1,d2,d3)->(d0,d3)>",
                "affine_map<(d0,d1,d2,d3)->(d0,d1)>",
            ]
            legal = (
                n == 2 and len(network_ins) == 3
                and network_outs == [init_inst.result_ssa]
                and output_slice is not None
                and all(item is not None for item in input_slices)
                and maps in (bilinear_maps, trilinear_maps)
                and output_slice["offsets"] == ["0", "0"]
                and output_slice["strides"] == ["1", "1"]
            )
            if legal:
                sources = [
                    _to_tensor_memref_source(
                        text, item["source"], init_inst.span[0])
                    for item in input_slices
                ]
                output_source = _to_tensor_memref_source(
                    text, output_slice["source"], init_inst.span[0])
                legal = all(source is not None for source in sources) and (
                    output_source is not None)
            if legal:
                physical_operands = [source[0] for source in sources] + [
                    output_source[0]]
                physical_types = [source[1] for source in sources] + [
                    output_source[1]]
                size_values = [[
                    _constant_index_value(text, value) if value != "1" else 1
                    for value in item["sizes"]] for item in input_slices]
                output_sizes = [
                    _constant_index_value(text, value)
                    for value in output_slice["sizes"]]
                expected_sizes = (
                    [[8, 16], [24, 16, 20], [8, 20]]
                    if maps == bilinear_maps
                    else [[8, 16], [16, 20, 24], [8, 20]])
                legal = (
                    size_values == expected_sizes
                    and output_sizes == [8, 24]
                    and len(set(physical_operands)) == 4
                    and _plain_f32_memrefs(physical_types, [2, 3, 2, 2])
                )
            tail = None
            if legal:
                sizes = r",\s*".join(map(re.escape, output_slice["sizes"]))
                tail = re.match(
                    rf"\s*(%[\w_\-]+)\s*=\s*tensor\.insert_slice\s+"
                    rf"{re.escape(network_inst.result_ssa)}\s+into\s+"
                    rf"{re.escape(output_slice['source'])}\[0,\s*0\]\s*"
                    rf"\[{sizes}\]\s*\[1,\s*1\][^\n]*\n"
                    rf"\s*(%[\w_\-]+)\s*=\s*bufferization\.to_memref\s+\1\s*:[^\n]+\n"
                    rf"\s*memref\.copy\s+\2,\s*{re.escape(physical_operands[3])}\s*:[^\n]*",
                    text[network_inst.span[1]:])
                legal = tail is not None
            if not legal:
                report.append(("aten_three_input_network_region_reject",
                               [i, i + 1], entry.name))
                i += n
                continue
            uid = network_inst.result_ssa.lstrip("%")
            dynamic_types = ["memref<?x?xf32>", "memref<?x?x?xf32>",
                             "memref<?x?xf32>", "memref<?x?xf32>"]
            cast_names = [f"%aten_network_{uid}_{j}" for j in range(4)]
            lines = [
                f"{network_inst.indent}{cast} = memref.cast {operand} : "
                f"{operand_type} to {dynamic_type}"
                for cast, operand, operand_type, dynamic_type in zip(
                    cast_names, physical_operands, physical_types,
                    dynamic_types)
            ]
            map_attrs = [
                "affine_map<(d0, d1, d2, d3) -> (d0, d2)>",
                ("affine_map<(d0, d1, d2, d3) -> (d1, d2, d3)>"
                 if maps == bilinear_maps else
                 "affine_map<(d0, d1, d2, d3) -> (d2, d3, d1)>") ,
                "affine_map<(d0, d1, d2, d3) -> (d0, d3)>",
                "affine_map<(d0, d1, d2, d3) -> (d0, d1)>",
            ]
            attrs = (
                " {network_maps = [" + ", ".join(map_attrs) +
                "], polygeist.fixed_operand_extents = array<i64: " +
                ("8, 16, 24, 16, 20, 8, 20, 8, 24"
                 if maps == bilinear_maps else
                 "8, 16, 16, 20, 24, 8, 20, 8, 24") +
                ">, polygeist.result_destinations = array<i64: 3>}"
            )
            lines.append(
                f"{network_inst.indent}kernel.launch @{entry.name}("
                f"{', '.join(cast_names)}){attrs} : "
                f"({', '.join(dynamic_types)}) -> ()")
            custom_launch_line = "\n".join(lines)
            replace_full_span = True
            custom_edit_span = (
                init_inst.span[0], network_inst.span[1] + tail.end())
            operands = []
            operand_types = []
            binds = {}

        if entry.name.startswith("cubSegmented") and n == 2:
            # The first generic only writes the reduction identity.  The CUB
            # primitive receives that identity as part of its configured
            # operation and overwrites every output row, so retain an SSA
            # alias for intervening extract_slice users without executing the
            # redundant initializer generic.
            init_inst = instances[i]
            if (init_inst.result_ssa is not None and
                    init_inst.result_type is not None and len(outs0) == 1 and
                    len(outs0_types) == 1):
                custom_first_launch_line = (
                    f"{init_inst.indent}{init_inst.result_ssa} = tensor.cast "
                    f"{outs0[0]} : {outs0_types[0]} to {init_inst.result_type}")

        if entry.name == "cudnnConvolution2DWindow_f32":
            init_inst = instances[i]
            reduce_inst = instances[i + 1]
            reduce_inputs = _extract_ssa_names(reduce_inst.ins_part)
            reduce_input_types = _extract_ssa_types(reduce_inst.ins_part)
            reduce_outputs = _extract_ssa_names(reduce_inst.outs_part)
            init_outputs = _extract_ssa_names(init_inst.outs_part)
            init_output_types = _extract_ssa_types(init_inst.outs_part)
            geometry = (
                _regular_window_conv2d_info(text, reduce_inputs[0])
                if len(reduce_inputs) == 1 else None
            )
            init_result = init_inst.result_ssa
            if (geometry is None or len(init_outputs) != 1 or
                    len(init_output_types) != 1 or len(reduce_outputs) != 1 or
                    reduce_outputs != [init_result] or
                    reduce_inst.result_ssa is None or
                    reduce_inst.result_type is None or
                    _shaped_rank(init_output_types[0]) != 4 or
                    _sniff_elem_type(init_output_types[0]) != "f32" or
                    not reduce_input_types or
                    _shaped_rank(reduce_input_types[0]) != 6 or
                    _sniff_elem_type(reduce_input_types[0]) != "f32"):
                report.append(("window_conv2d_reject", [i, i + 1], entry.name))
                i += n
                continue
            (base, base_type, kh, kw, sh, sw, dh, dw, ph, pw) = geometry
            bound_weight = binds.get("%weight")
            weight_ssa: str | None = None
            weight_value: float | None = None
            if (isinstance(bound_weight, tuple) and len(bound_weight) == 2 and
                    bound_weight[0] == "Cap"):
                weight_ssa = bound_weight[1]
                if scalar_types.get(weight_ssa) != "f32":
                    report.append(("window_conv2d_weight_reject", [i, i + 1],
                                   entry.name))
                    i += n
                    continue
            elif (isinstance(bound_weight, tuple) and len(bound_weight) == 2 and
                  bound_weight[0] == "Lit"):
                weight_value = float(bound_weight[1])
            else:
                report.append(("window_conv2d_weight_reject", [i, i + 1],
                               entry.name))
                i += n
                continue
            custom_launch_line = _render_window_conv2d_launch(
                reduce_inst.result_ssa,
                reduce_inst.result_type,
                base,
                base_type,
                init_outputs[0],
                init_output_types[0],
                weight_ssa,
                weight_value,
                (kh, kw, sh, sw, dh, dw, ph, pw),
                reduce_inst.indent,
                i,
            )
            # The window submap sits between the two generics. Keep it (it may
            # still have debug/round-trip users), remove the initializer, and
            # replace only the reduction with the launch.
            binds = {}

        if generic_graph_spec is not None:
            def signed_i64(value: int) -> int:
                return value if value < (1 << 63) else value - (1 << 64)

            def render_graph(spec, graph_inputs, graph_input_types,
                             graph_outs, graph_out_types, result_ssa,
                             result_type, tag):
                # The generic graph ABI has four tensor and eight scalar slots.
                # Unused slots are duplicates/zeros and bytecode cannot refer
                # to them accidentally.
                graph_inputs = list(graph_inputs)
                graph_input_types = list(graph_input_types)
                while len(graph_inputs) < 4:
                    graph_inputs.append(graph_inputs[0])
                    graph_input_types.append(graph_input_types[0])
                scalar_names: list[str] = []
                scalar_lines: list[str] = []
                for scalar_i, key in enumerate(spec["scalars"]):
                    if key[0] == "Cap":
                        scalar_names.append(key[1])
                    else:
                        ssa = _derived_ssa_name(
                            last.result_ssa, f"pw_{tag}_scalar_{scalar_i}")
                        value = repr(float(key[1]))
                        if ("." not in value and "e" not in value and
                                "E" not in value):
                            value += ".0"
                        scalar_lines.append(
                            f"{last.indent}{ssa} = arith.constant {value} : f32")
                        scalar_names.append(ssa)
                while len(scalar_names) < 8:
                    ssa = _derived_ssa_name(
                        last.result_ssa, f"pw_{tag}_pad_{len(scalar_names)}")
                    scalar_lines.append(
                        f"{last.indent}{ssa} = arith.constant 0.0 : f32")
                    scalar_names.append(ssa)
                encoded_words = [signed_i64(v) for v in spec["words"]]
                attrs = (
                    " {pointwise_graph = array<i64: " +
                    ", ".join(str(v) for v in encoded_words) +
                    ">, pointwise_num_nodes = " +
                    str(spec["nodes"]) + " : i64}")
                rendered = render_launch(
                    "cudnnPointwiseGraph_f32", result_ssa, result_type,
                    graph_inputs + graph_outs + scalar_names,
                    last.indent, {}, [],
                    operand_types=(graph_input_types + graph_out_types +
                                   ["f32"] * 8),
                    scalar_type_map=scalar_types, result_count=1,
                    launch_attrs=attrs)
                return scalar_lines + [rendered]

            if generic_graph_partition is None:
                custom_launch_line = "\n".join(render_graph(
                    generic_graph_spec, all_tensor_ins, all_tensor_in_types,
                    outs0, outs0_types, last.result_ssa, last.result_type,
                    "single"))
            else:
                first_spec, second_spec = generic_graph_partition
                axis = _derived_ssa_name(last.result_ssa, "pw_axis")
                extent = _derived_ssa_name(last.result_ssa, "pw_extent")
                empty = _derived_ssa_name(last.result_ssa, "pw_empty")
                middle = _derived_ssa_name(last.result_ssa, "pw_middle")
                intermediate_type = "tensor<?xf32>"
                lines = [
                    f"{last.indent}{axis} = arith.constant 0 : index",
                    f"{last.indent}{extent} = tensor.dim "
                    f"{all_tensor_ins[0]}, {axis} : {all_tensor_in_types[0]}",
                    f"{last.indent}{empty} = bufferization.alloc_tensor({extent}) : "
                    f"{intermediate_type}",
                ]
                lines.extend(render_graph(
                    first_spec, all_tensor_ins, all_tensor_in_types,
                    [empty], [intermediate_type], middle, intermediate_type,
                    "first"))
                lines.extend(render_graph(
                    second_spec, all_tensor_ins + [middle],
                    all_tensor_in_types + [intermediate_type], outs0,
                    outs0_types, last.result_ssa, last.result_type, "second"))
                custom_launch_line = "\n".join(lines)

        def _tensor_copy_layout_is_legal() -> bool:
            """Conservatively prove that a semantic `yield %in` is memcpy.

            Body equivalence alone also matches transpose, pixel-shuffle, and
            view-gather operations.  The CUDA copy shims are flat contiguous
            copies, so require identical indexing maps and identical shaped
            tensor types, and reject sources produced by polygeist.submap.
            """
            copy_body = bodies[i]
            if (len(copy_body.indexing_maps) != 2 or
                    copy_body.indexing_maps[0] != copy_body.indexing_maps[1]):
                return False
            if (len(all_tensor_in_types) != 1 or len(outs0_types) != 1 or
                    all_tensor_in_types[0] != outs0_types[0]):
                return False
            source = all_tensor_ins[0] if all_tensor_ins else ""
            if source:
                prefix = text[:instances[i].span[0]]
                if re.search(
                        rf"^\s*{re.escape(source)}\s*=\s*polygeist\.submap\b",
                        prefix, re.MULTILINE):
                    return False
            return True

        # Tensor-form twin of the same dispatch (multi-root debufferize).
        if entry.name == "cublasDcopy_tensor" and n == 1:
            in0_ty = all_tensor_in_types[0] if all_tensor_in_types else ""
            elem = _sniff_elem_type(in0_ty) if in0_ty else None
            ranks = [_tensor_rank(t) for t in operand_types[:2]]
            copy_body = bodies[i]
            maps = copy_body.indexing_maps
            is_broadcast = elem == "f32" and ranks == [1, 2] and len(maps) == 2
            if is_broadcast:
                input_map = re.sub(r"\s+", "", maps[0])
                output_map = re.sub(r"\s+", "", maps[1])
                if (("->(d0)" in input_map and "->(d0,d1)" in output_map) or
                        ("->(d1)" in input_map and "->(d1,d0)" in output_map)):
                    emit_name = "cublasBroadcastAxis0_f32"
                elif (("->(d1)" in input_map and "->(d0,d1)" in output_map) or
                      ("->(d0)" in input_map and "->(d1,d0)" in output_map)):
                    emit_name = "cublasBroadcastAxis1_f32"
                else:
                    report.append(("broadcast_layout_reject", i, entry.name))
                    i += 1
                    continue
            elif not _tensor_copy_layout_is_legal():
                report.append(("copy_layout_reject", i, entry.name))
                i += 1
                continue
            if in0_ty.startswith("tensor<"):
                inside = in0_ty[len("tensor<"):].split(",", 1)[0]
                if "x" not in inside and not is_broadcast:
                    emit_name = "broadcast_scalar_to_vec_tensor"
            if is_broadcast:
                pass
            elif elem == "f32" and len(ranks) == 2 and ranks[0] == ranks[1]:
                if ranks[0] == 1:
                    emit_name = "cudaCopy1D_f32_tensor"
                elif ranks[0] == 2:
                    emit_name = "cudaCopy2D_f32_tensor"
                else:
                    report.append(("rank_or_dtype_reject", i, entry.name))
                    i += 1
                    continue
            elif emit_name == "cublasDcopy_tensor":
                if not (elem == "f64" and len(ranks) == 2 and ranks == [1, 1]):
                    report.append(("rank_or_dtype_reject", i, entry.name))
                    i += 1
                    continue

        if entry.name in ("tensor_copy_2D", "tensor_copy_3D",
                          "tensor_copy_6D"):
            if not _tensor_copy_layout_is_legal():
                report.append(("copy_layout_reject", i, entry.name))
                i += 1
                continue
            expected_rank = {
                "tensor_copy_2D": 2,
                "tensor_copy_3D": 3,
                "tensor_copy_6D": 6,
            }[entry.name]
            elem = _sniff_elem_type(all_tensor_in_types[0]) if all_tensor_in_types else None
            ranks = [_tensor_rank(t) for t in operand_types[:2]]
            if elem == "f32" and len(ranks) == 2 and ranks == [
                    expected_rank, expected_rank]:
                emit_name = f"cudaCopy{expected_rank}D_f32_tensor"
            else:
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += 1
                continue

        if entry.name.startswith("cutensorUnary_"):
            ranks = [_tensor_rank(t) for t in operand_types[:2]]
            elems = [_sniff_elem_type(t) for t in operand_types[:2]]
            unary_body = bodies[i]
            source = all_tensor_ins[0] if all_tensor_ins else ""
            source_is_submap = bool(source and re.search(
                rf"^\s*{re.escape(source)}\s*=\s*polygeist\.submap\b",
                text[:instances[i].span[0]], re.MULTILINE))
            if (len(operand_types) != 2 or len(ranks) != 2 or
                    ranks != [1, 1] or
                    elems != ["f32", "f32"] or
                    len(unary_body.indexing_maps) != 2 or
                    unary_body.indexing_maps[0] != unary_body.indexing_maps[1] or
                    all_tensor_in_types[0] != outs0_types[0] or
                    source_is_submap):
                report.append(("rank_dtype_or_layout_reject", i, entry.name))
                i += n
                continue

        if entry.name == "cudnnAddTensor_batched":
            # The runtime wrapper implements cuDNN's NCHW AddTensor path for
            # rank-4 f32 only.  Elementwise-add semantics also occur in the
            # FP64 MFEM stages, but cuDNN cannot execute that signature and no
            # canonical kernel.defn exists for it.  Preserve those adds as
            # residual Linalg.
            ranks = [_tensor_rank(t) for t in operand_types[:2]]
            elems = [_sniff_elem_type(t) for t in operand_types[:2]]
            if len(operand_types) != 2 or ranks != [4, 4] or elems != ["f32", "f32"]:
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += n
                continue

        if entry.name == "cublasDaxpby":
            # The semantic template permits implicit unit coefficients, but
            # the public ABI is exactly (x, y, alpha, beta) over contiguous
            # rank-1 vectors. Do not emit the historical two-operand
            # rank-N launch: it cannot verify against the kernel definition
            # and flattening a strided tensor would be incorrect.
            ranks = [_tensor_rank(t) for t in operand_types[:2]]
            elems = [_sniff_elem_type(t) for t in operand_types[:2]]
            if (len(operand_types) != 2 or ranks != [1, 1] or
                    len(set(elems)) != 1 or elems[0] not in ("f64", "f32")):
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += n
                continue
            coefficient_type = elems[0]
            if coefficient_type == "f32":
                emit_name = "cublasSaxpby"

            scalar_names: list[str] = []
            scalar_lines: list[str] = []
            for coefficient in ("%alpha", "%beta"):
                bound = binds.get(coefficient)
                if (isinstance(bound, tuple) and len(bound) == 2 and
                        bound[0] == "Cap"):
                    scalar_names.append(bound[1])
                    continue
                if (isinstance(bound, tuple) and len(bound) == 2 and
                        bound[0] == "Lit"):
                    suffix = coefficient.lstrip("%")
                    scalar = _derived_ssa_name(last.result_ssa, suffix)
                    value = repr(float(bound[1]))
                    if "." not in value and "e" not in value and "E" not in value:
                        value += ".0"
                    scalar_lines.append(
                        f"{last.indent}{scalar} = arith.constant {value} : {coefficient_type}"
                    )
                    scalar_names.append(scalar)
                    continue
                report.append(("coefficient_reject", i, entry.name))
                break
            if len(scalar_names) != 2:
                i += n
                continue
            rendered = render_launch(
                emit_name, last.result_ssa, last.result_type,
                operands + scalar_names, last.indent, {}, [],
                operand_types=operand_types + [coefficient_type, coefficient_type],
                scalar_type_map=scalar_types,
                result_count=last.result_count,
            )
            custom_launch_line = "\n".join(scalar_lines + [rendered])

        if entry.name in ("cublasDdot", "cublasSdot",
                          "cublasDdot_memref", "cublasSdot_memref"):
            # BLAS dot writes one scalar.  A rank-1 tensor output can be a
            # non-injective submap whose every logical element aliases that
            # scalar, but emitting it as a rank-1 launch is not ABI-correct.
            # Preserve that generic until LowerPolygeistSubmap normalizes the
            # reduction to a real rank-0 tensor.
            ranks = [_tensor_rank(t) for t in operand_types[:3]]
            elems = [_sniff_elem_type(t) for t in operand_types[:3]]
            expected = "f64" if entry.name.startswith("cublasD") else "f32"
            tensor_form = (
                ranks == [1, 1, 0]
                and all(t.startswith("tensor<") for t in operand_types[:3])
            )
            memref_form = (
                [_shaped_rank(t) for t in operand_types[:3]] == [1, 1, 1]
                and all(t.startswith("memref<") for t in operand_types[:3])
                and len(operands) >= 3
                and _is_scalar_alias_submap(text, operands[2])
            )
            scalar_view = (
                _parse_memref_view(text, operands[2], last.span[0])
                if len(operands) >= 3 else None
            )
            scalar_inits = (list(re.finditer(
                rf"^\s*affine\.store\s+(%[\w_\-]+),\s*"
                rf"{re.escape(scalar_view['base'])}\[0\]\s*:\s*"
                rf"{re.escape(scalar_view['base_type'])}\s*$",
                text[:last.span[0]], re.MULTILINE))
                if scalar_view is not None else [])
            scalar_init = scalar_inits[-1] if scalar_inits else None
            reinterpret_memref_form = (
                [_shaped_rank(t) for t in operand_types[:3]] == [1, 1, 0]
                and all(t.startswith("memref<") for t in operand_types[:3])
                and scalar_view is not None
                and scalar_view["kind"] == "reinterpret_cast"
                and _shaped_rank(scalar_view["base_type"]) == 1
                and _sniff_elem_type(scalar_view["base_type"]) == expected
                and scalar_init is not None
                and parse_constants(text[:last.span[0]]).get(
                    scalar_init.group(1)) == 0.0
                and not re.search(
                    r"\b(?:affine|memref)\.store\b|\blinalg\.",
                    text[scalar_init.end():last.span[0]])
            )
            if (len(operand_types) != 3 or
                    elems != [expected, expected, expected] or
                    not (tensor_form or memref_form or
                         reinterpret_memref_form)):
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += n
                continue
            if memref_form or reinterpret_memref_form:
                emit_name = ("cublasDdot_memref" if expected == "f64"
                             else "cublasSdot_memref")
            if reinterpret_memref_form:
                operands[2] = scalar_view["base"]
                operand_types[2] = scalar_view["base_type"]

        if entry.name == "cublasDscal":
            ranks = [_tensor_rank(t) for t in operand_types[:1]]
            elems = [_sniff_elem_type(t) for t in operand_types[:1]]
            if len(operand_types) != 1 or ranks != [1] or elems != ["f32"]:
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += n
                continue
            emit_name = "cublasSscal"

        if entry.name in ("cublasSgemm_nn_zero",
                           "cublasSgemm_strided_batched_nn_zero"):
            expected_rank = 2 if entry.name == "cublasSgemm_nn_zero" else 3
            ranks = [_tensor_rank(t) for t in operand_types[:3]]
            elems = [_sniff_elem_type(t) for t in operand_types[:3]]
            zero_bound = binds.get("%zero")
            zero_ok = False
            if isinstance(zero_bound, tuple) and len(zero_bound) == 2:
                if zero_bound[0] == "Lit":
                    zero_ok = float(zero_bound[1]) == 0.0
                elif zero_bound[0] == "Cap":
                    zero_ok = bool(re.search(
                        rf"^\s*{re.escape(zero_bound[1])}\s*=\s*arith\.constant\s+"
                        rf"(?:0(?:\.0*)?|0\.0+e[+-]?0+)\s*:\s*f(?:32|64)\b",
                        text[:instances[i].span[0]], re.MULTILINE | re.IGNORECASE))
            rank_layout_ok = ranks == [expected_rank] * 3
            if (entry.name == "cublasSgemm_strided_batched_nn_zero" and
                    ranks == [3, 2, 3]):
                rank_layout_ok = True
                emit_name = "cublasSgemm_strided_batched_broadcast_rhs"
            if (entry.name == "cublasSgemm_nn_zero" and elems == ["f64"] * 3):
                emit_name = "cublasDgemm_zero"
            elif elems != ["f32"] * 3:
                rank_layout_ok = False
            if (len(operand_types) != 3 or not rank_layout_ok or not zero_ok):
                report.append(("rank_dtype_or_init_reject", i, entry.name))
                i += n
                continue
            if (entry.name == "cublasSgemm_nn_zero" and
                    elems == ["f32"] * 3):
                maps = bodies[i + n - 1].indexing_maps
                def _zero_map_outputs(txt: str) -> list[str]:
                    mm = re.search(r"->\s*\(([^)]*)\)>", txt)
                    return ([s.strip() for s in mm.group(1).split(",")]
                            if mm else [])
                if len(maps) != 3:
                    report.append(("layout_reject", i, entry.name))
                    i += n
                    continue
                a_dims = _zero_map_outputs(maps[0])
                b_dims = _zero_map_outputs(maps[1])
                c_dims = _zero_map_outputs(maps[2])
                if not all(len(dims) == 2
                           for dims in (a_dims, b_dims, c_dims)):
                    report.append(("layout_reject", i, entry.name))
                    i += n
                    continue
                m_dim, n_dim = c_dims
                a_trans = a_dims[1] == m_dim and a_dims[0] != m_dim
                b_trans = b_dims[0] == n_dim and b_dims[1] != n_dim
                a_k = a_dims[0] if a_trans else a_dims[1]
                b_k = b_dims[1] if b_trans else b_dims[0]
                if (((a_dims[1] if a_trans else a_dims[0]) != m_dim) or
                        ((b_dims[0] if b_trans else b_dims[1]) != n_dim) or
                        a_k != b_k):
                    report.append(("layout_reject", i, entry.name))
                    i += n
                    continue
                emit_name = ("cublasSgemm_" +
                             ("t" if a_trans else "n") +
                             ("t" if b_trans else "n") + "_zero")

        if entry.name in (
                "cublasGemmFor1x1Conv",
                "cutensornetContraction2_f64"):
            # Route two-input FP64 sum contractions through a layout-aware
            # cuTensorNet ABI, but only after proving their Einstein semantics.
            # The generic entry intentionally has no fixed iterator counts:
            # reduction modes and output modes are derived from the actual
            # linalg.generic rather than assuming the historical 3D d4 case.
            contraction_inst = instances[i + n - 1]
            contraction_body = bodies[i + n - 1]
            contraction_ins = _extract_ssa_names(
                contraction_inst.ins_part
            )
            contraction_in_types = _extract_ssa_types(
                contraction_inst.ins_part
            )
            actual_contraction_outs = _extract_ssa_names(
                contraction_inst.outs_part
            )

            # An opaque library call currently cannot safely consume a submap
            # whose base is itself a computed tensor (notably an MFEM
            # quadrature result assembled through submapInverse).  Converting
            # that tensor to a raw pointer before one-shot bufferization can
            # select an earlier aliased buffer.  Keep such contractions as
            # residual Linalg until the runtime call is represented by a
            # bufferizable op with explicit read/write effects.
            prefix = text[:contraction_inst.span[0]]

            def _computed_submap_base(value: str) -> bool:
                matches = list(re.finditer(
                    rf"^\s*{re.escape(value)}\s*=\s*polygeist\.submap\(\s*(%[\w.$-]+)",
                    prefix, re.MULTILINE,
                ))
                if not matches:
                    return False
                base = matches[-1].group(1)
                if base.startswith("%arg"):
                    return False
                direct_arg_view = re.search(
                    rf"^\s*{re.escape(base)}\s*=\s*bufferization\.to_tensor\s+%arg\d+\b",
                    prefix, re.MULTILINE,
                )
                return direct_arg_view is None

            if any(_computed_submap_base(value)
                   for value in contraction_ins[:2]):
                report.append(("computed_submap_base_reject", i, entry.name))
                i += n
                continue
            init_results = (
                [instances[i].result_ssa]
                if instances[i].result_ssa else []
            )
            direct_init_chain = actual_contraction_outs == init_results
            # When the contraction consumes the zero generic directly, pass
            # the zero generic's destination to the beta=0 runtime and erase
            # the redundant zero result.  Re-viewed scratch outputs must keep
            # their intermediate chain and use the contraction's actual view.
            output_inst = instances[i] if direct_init_chain else contraction_inst
            contraction_outs = _extract_ssa_names(output_inst.outs_part)
            contraction_out_types = _extract_ssa_types(output_inst.outs_part)
            maps = contraction_body.indexing_maps

            def _pure_dim_outputs(map_text: str) -> list[int] | None:
                match = re.fullmatch(
                    r"affine_map<\([^)]*\)\s*->\s*\(([^)]*)\)>",
                    map_text.strip(),
                )
                if not match:
                    return None
                outputs = []
                for expr in match.group(1).split(","):
                    dim = re.fullmatch(r"\s*d(\d+)\s*", expr)
                    if not dim:
                        return None
                    outputs.append(int(dim.group(1)))
                return outputs

            map_dims = (
                [_pure_dim_outputs(map_text) for map_text in maps]
                if len(maps) == 3 else []
            )
            reduction_dims = {
                dim for dim, role in enumerate(contraction_body.iterator_types)
                if role == "reduction"
            }

            def _resolve_affine_map(map_ref: str) -> str | None:
                map_ref = map_ref.strip()
                if map_ref.startswith("affine_map<"):
                    return map_ref
                if not map_ref.startswith("#"):
                    return None
                match = re.search(
                    rf"(?m)^\s*{re.escape(map_ref)}\s*=\s*"
                    r"(affine_map<.*?>)\s*$",
                    text,
                )
                return match.group(1) if match else None

            def _physical_broadcast_modes(
                    operand: str, access_dims: list[int] | None) -> set[int]:
                """Return access modes proved to have zero physical stride.

                A polygeist.submap flattened map that omits an operand
                dimension gives that logical dimension stride zero.  This is
                how scratch-sliced MFEM represents an output whose apparent
                Linalg rank still contains a reduction dimension.
                """
                if access_dims is None:
                    return set()
                definition = re.search(
                    rf"(?s){re.escape(operand)}\s*=\s*polygeist\.submap"
                    r"\([^)]*\)\s*\{map\s*=\s*([^}]+)\}",
                    text,
                )
                if not definition:
                    return set()
                flat_map = _resolve_affine_map(definition.group(1))
                if not flat_map:
                    return set()
                result = re.search(r"->\s*\((.*)\)\s*>", flat_map)
                if not result:
                    return set()
                flat_expr = result.group(1)
                return {
                    mode for axis, mode in enumerate(access_dims)
                    if not re.search(rf"\bd{axis}\b", flat_expr)
                }

            output_broadcast_modes = (
                _physical_broadcast_modes(
                    contraction_outs[0],
                    map_dims[2] if len(map_dims) == 3 else None,
                )
                if contraction_outs else set()
            )
            compact_output_dims = (
                [mode for mode in map_dims[2]
                 if mode not in output_broadcast_modes]
                if len(map_dims) == 3 and map_dims[2] is not None else []
            )
            elem_types = [
                _sniff_elem_type(ty)
                for ty in contraction_in_types + contraction_out_types
            ]
            ranks = [
                _tensor_rank(ty)
                for ty in contraction_in_types + contraction_out_types
            ]

            # A zero initializer followed by a rank-2 × rank-1 contraction is
            # GEMV with beta=0.  The shared contraction recognizer reaches this
            # shape before the one-step GEMV entry, so route it here instead of
            # rejecting f32 as an unsupported cuTensorNet contraction.  Emit
            # the zero and GEMV launches as a complete two-call replacement;
            # the existing GEMV ABI has beta=1 and therefore consumes the
            # explicitly initialized accumulator.
            parallel_dims = {
                dim for dim, role in enumerate(contraction_body.iterator_types)
                if role == "parallel"
            }
            is_gemv = (
                len(contraction_ins) == 2
                and len(contraction_outs) == 1
                and sorted(ranks[:2]) == [1, 2]
                and ranks[2:] == [1]
                and len(parallel_dims) == 1
                and len(reduction_dims) == 1
                and len(map_dims) == 3
                and all(dims is not None for dims in map_dims)
                and elem_types in (["f32"] * 3, ["f64"] * 3)
                and instances[i].result_ssa is not None
                and instances[i].result_type is not None
                and last.result_ssa is not None
                and last.result_type is not None
            )
            is_conv3d_f32 = (
                len(contraction_ins) == 2
                and len(contraction_outs) == 1
                and ranks == [8, 5, 4]
                and elem_types == ["f32", "f32", "f32"]
                and len(parallel_dims) == 4
                and len(reduction_dims) == 4
                and map_dims == [list(range(8)), [0, 4, 5, 6, 7],
                                 [0, 1, 2, 3]]
                and _is_forward_conv3d_window(text, contraction_ins[0])
                and last.result_ssa is not None
                and last.result_type is not None
            )
            if is_conv3d_f32:
                emit_name = "cudnnConvolution3D_f32"
                operands = contraction_ins + contraction_outs
                operand_types = contraction_in_types + contraction_out_types
                custom_launch_line = render_launch(
                    emit_name, last.result_ssa, last.result_type,
                    operands, last.indent, {}, [],
                    operand_types=operand_types,
                    scalar_type_map=scalar_types,
                    result_count=last.result_count,
                )
            elif is_gemv:
                elem = elem_types[0]
                init_outs = _extract_ssa_names(instances[i].outs_part)
                init_types = _extract_ssa_types(instances[i].outs_part)
                if len(init_outs) != 1 or len(init_types) != 1:
                    report.append(("gemv_init_reject", i, entry.name))
                    i += n
                    continue
                zero_name = ("memset_zero_1D_f32" if elem == "f32"
                             else "memset_zero_1D")
                init_line = render_launch(
                    zero_name, instances[i].result_ssa,
                    instances[i].result_type, init_outs,
                    instances[i].indent, {}, [], operand_types=init_types,
                    scalar_type_map=scalar_types,
                    result_count=instances[i].result_count,
                )
                paired_inputs = sorted(
                    zip(contraction_in_types, contraction_ins),
                    key=lambda pair: -_tensor_rank(pair[0]),
                )
                gemv_types = [pair[0] for pair in paired_inputs] + [
                    contraction_out_types[0]
                ]
                gemv_operands = [pair[1] for pair in paired_inputs] + [
                    contraction_outs[0]
                ]
                a_dims = map_dims[contraction_ins.index(gemv_operands[0])]
                y_dims = map_dims[2]
                transposed = bool(a_dims and y_dims and
                                  a_dims[0] != y_dims[0])
                gemv_name = (
                    ("cublasSgemv_T" if transposed else "cublasSgemv")
                    if elem == "f32" else
                    ("cublasDgemv_T" if transposed else "cublasDgemv")
                )
                gemv_line = render_launch(
                    gemv_name, last.result_ssa, last.result_type,
                    gemv_operands, last.indent, {}, [],
                    operand_types=gemv_types,
                    scalar_type_map=scalar_types,
                    result_count=last.result_count,
                )
                emit_name = gemv_name
                operands = gemv_operands
                operand_types = gemv_types
                custom_first_launch_line = init_line
                custom_launch_line = gemv_line
            # A canonical rank-2 matrix product is better represented by the
            # cuBLAS ABI than by the general cuTensorNet ABI.  Keep this
            # deliberately layout-strict: A[m,k] * B[k,n] -> C[m,n].  The
            # matched composition includes a zero initializer and the simple
            # GEMM shim has beta=1, so retain both external-library calls.
            elif (
                len(contraction_ins) == 2
                and len(contraction_outs) == 1
                and ranks == [2, 2, 2]
                and elem_types == ["f64", "f64", "f64"]
                and len(parallel_dims) == 2
                and len(reduction_dims) == 1
                and len(map_dims) == 3
                and all(dims is not None for dims in map_dims)
                and len(map_dims[0]) == len(map_dims[1]) == 2
                and len(map_dims[2]) == 2
                and map_dims[0][0] == map_dims[2][0]
                and map_dims[1][1] == map_dims[2][1]
                and map_dims[0][1] == map_dims[1][0]
                and map_dims[0][1] in reduction_dims
                and last.result_ssa is not None
                and last.result_type is not None
            ):
                init_outs = _extract_ssa_names(instances[i].outs_part)
                init_types = _extract_ssa_types(instances[i].outs_part)
                if len(init_outs) != 1 or len(init_types) != 1:
                    report.append(("gemm_init_reject", i, entry.name))
                    i += n
                    continue
                init_line = render_launch(
                    "memset_zero_2D", instances[i].result_ssa,
                    instances[i].result_type, init_outs,
                    instances[i].indent, {}, [], operand_types=init_types,
                    scalar_type_map=scalar_types,
                    result_count=instances[i].result_count,
                )
                emit_name = "cublasDgemm_simple"
                operands = contraction_ins + contraction_outs
                operand_types = contraction_in_types + contraction_out_types
                custom_first_launch_line = init_line
                custom_launch_line = render_launch(
                    emit_name, last.result_ssa, last.result_type,
                    operands, last.indent, {}, [],
                    operand_types=operand_types,
                    scalar_type_map=scalar_types,
                    result_count=last.result_count,
                )
            # A pair of vectors contracted without a reduction is an outer
            # product.  The composition's first generic is a zero fill, so
            # use an overwrite-mode runtime entry and replace both stages.
            # Order the vectors by the corresponding output mode rather than
            # by source operand order so the ABI is always u[M], v[N], C[M,N].
            elif (
                len(contraction_ins) == 2
                and len(contraction_outs) == 1
                and ranks == [1, 1, 2]
                and elem_types == ["f64", "f64", "f64"]
                and len(parallel_dims) == 2
                and not reduction_dims
                and len(map_dims) == 3
                and all(dims is not None for dims in map_dims)
                and len(map_dims[0]) == len(map_dims[1]) == 1
                and len(map_dims[2]) == 2
                and set(map_dims[0] + map_dims[1]) == set(map_dims[2])
                and map_dims[0][0] != map_dims[1][0]
                and last.result_ssa is not None
                and last.result_type is not None
            ):
                by_mode = {
                    map_dims[0][0]: (contraction_ins[0],
                                     contraction_in_types[0]),
                    map_dims[1][0]: (contraction_ins[1],
                                     contraction_in_types[1]),
                }
                ordered = [by_mode[mode] for mode in map_dims[2]]
                operands = [pair[0] for pair in ordered] + contraction_outs
                operand_types = [pair[1] for pair in ordered] + \
                    contraction_out_types
                emit_name = "cublasDgemm_outer_product"
                custom_launch_line = render_launch(
                    emit_name, last.result_ssa, last.result_type,
                    operands, last.indent, {}, [],
                    operand_types=operand_types,
                    scalar_type_map=scalar_types,
                    result_count=last.result_count,
                )
            # A[B,M,K] * B[K,N] -> C[B,M,N], with B shared by every
            # batch, is cublasSgemmStridedBatched with strideB=0.  Keep this
            # route deliberately strict about mode order: the runtime ABI is
            # row-major and must not silently reinterpret transposed views.
            elif (
                len(contraction_ins) == 2
                and len(contraction_outs) == 1
                and ranks == [3, 2, 3]
                and elem_types == ["f32", "f32", "f32"]
                and len(parallel_dims) == 3
                and len(reduction_dims) == 1
                and len(map_dims) == 3
                and all(dims is not None for dims in map_dims)
                and len(map_dims[0]) == 3
                and len(map_dims[1]) == 2
                and len(map_dims[2]) == 3
                and map_dims[0][:2] == map_dims[2][:2]
                and map_dims[0][2] in reduction_dims
                and map_dims[1][0] == map_dims[0][2]
                and map_dims[1][1] == map_dims[2][2]
                and last.result_ssa is not None
                and last.result_type is not None
            ):
                emit_name = "cublasSgemm_strided_batched_broadcast_rhs"
                operands = contraction_ins + contraction_outs
                operand_types = contraction_in_types + contraction_out_types
                custom_launch_line = render_launch(
                    emit_name, last.result_ssa, last.result_type,
                    operands, last.indent, {}, [],
                    operand_types=operand_types,
                    scalar_type_map=scalar_types,
                    result_count=last.result_count,
                )
            else:
                legal_maps = (
                    len(map_dims) == 3
                    and all(dims is not None for dims in map_dims)
                    and bool(reduction_dims)
                    and all(
                        red in map_dims[0] or red in map_dims[1]
                        for red in reduction_dims
                    )
                    and all(red not in compact_output_dims
                            for red in reduction_dims)
                    and all(
                        mode in map_dims[0] or mode in map_dims[1]
                        for mode in compact_output_dims
                    )
                    and all(len(dims) == len(set(dims)) for dims in map_dims)
                    and len(compact_output_dims) ==
                        len(set(compact_output_dims))
                )
                legal_types = (
                    len(contraction_ins) == 2
                    and len(contraction_outs) == 1
                    and elem_types == ["f64", "f64", "f64"]
                    and all(rank is not None and 0 < rank <= 64
                            for rank in ranks)
                    and last.result_type is not None
                    and _sniff_elem_type(last.result_type) == "f64"
                )
                if not legal_maps or not legal_types:
                    report.append(("contraction_abi_reject", i, entry.name))
                    i += n
                    continue

                legacy_names = {
                    (4, 5, 4): "cutensornetContraction2_f64_r4r5r4",
                    (5, 4, 4): "cutensornetContraction2_f64_r5r4r4",
                    (5, 5, 4): "cutensornetContraction2_f64_r5r5r4",
                }
                emit_name = (
                    legacy_names[tuple(ranks)]
                    if entry.name == "cublasGemmFor1x1Conv"
                    and tuple(ranks) in legacy_names
                    else "cutensornetContraction2_f64"
                )
                # Preserve source operand order: contraction_maps correspond
                # positionally to these operands, so the generic rank-based
                # commutative reordering is intentionally bypassed.
                operands = contraction_ins + contraction_outs
                operand_types = contraction_in_types + contraction_out_types
                custom_launch_line = _render_contraction_launch(
                    emit_name, last.result_ssa, last.result_type,
                    operands, operand_types, maps, last.indent,
                    unranked_abi=(emit_name ==
                                  "cutensornetContraction2_f64"),
                )
                # Some scratch-sliced stages re-view the zero-initialized
                # tensor before contracting into it. Keep that initialization
                # chain alive and replace only the contraction.
                if not direct_init_chain:
                    replace_full_span = True
                    custom_edit_span = contraction_inst.span

        # Dtype-suffix dispatch for cuDNN conv2d. The encoder's Term language
        # is dtype-agnostic (arith.mulf matches any float type), so one
        # template fires for f64, f32, f16, bf16 bodies. We emit a
        # dtype-specific kernel.launch symbol so the canonical defn and the
        # lowering pass can pick the right cuDNN shim per element type.
        # The default (no suffix) is f64 for backward compat with the
        # existing kernel.defn @cudnnConvolution2D_9tap declaration.
        if entry.name == "cudnnConvolutionFwd_im2col_gemm":
            im2col = _extract_guarded_im2col_input(bodies[i + 1].body_lines)
            func_args = _enclosing_func_args(text, instances[i].span[0])
            gemm_ins = _extract_ssa_names(instances[i + 2].ins_part)
            gemm_in_types = _extract_ssa_types(instances[i + 2].ins_part)
            if im2col is None or len(func_args) < 7 or len(gemm_ins) < 1:
                report.append(("im2col_gemm_reject", i, entry.name))
                i += 1
                continue
            input_ssa, input_ty = im2col
            weights_ssa = gemm_ins[0]
            weights_ty = gemm_in_types[0] if gemm_in_types else "!any"
            output_ssa = outs0[0] if outs0 else ""
            output_ty = outs0_types[0] if outs0_types else "!any"
            shape_args = func_args[:7]
            operands = [input_ssa, weights_ssa, output_ssa] + [
                name for name, _ty in shape_args
            ]
            operand_types = [input_ty, weights_ty, output_ty] + [
                ty for _name, ty in shape_args
            ]
            # The fused memref launch mutates the original flat output buffer.
            last = LinalgInstance(
                result_ssa=None,
                result_count=0,
                ins_part=last.ins_part,
                outs_part=last.outs_part,
                result_type=None,
                span=last.span,
                indent=last.indent,
            )


        if entry.name in ("cudnnSoftmaxForward", "cudnnSoftmaxForward_tensor"):
            # The raised llama2 softmax has a scalar max buffer as the first
            # generic's out, then mutates the full vector in the later two
            # generics. Emit the full vector operand, not the max scalar nor
            # the x[1:] subview used only for the initialized-max reduction.
            vector_inst = (instances[i + 1] if entry.name.endswith("_tensor")
                           else instances[i + n - 1])
            out_names = _extract_ssa_names(vector_inst.outs_part)
            out_types = _extract_ssa_types(vector_inst.outs_part)
            if len(out_names) < 1:
                report.append(("softmax_reject", i, entry.name))
                i += 1
                continue
            vector_base = _trace_tensor_storage_base(text, out_names[0])
            vector_type = _infer_tensor_type(text, vector_base)
            if not vector_type:
                report.append(("softmax_base_reject", i, entry.name))
                i += 1
                continue
            operands = [vector_base]
            operand_types = [vector_type]
            binds = {}
            if entry.name.endswith("_tensor"):
                replace_full_span = True
            else:
                last = LinalgInstance(
                    result_ssa=None,
                    result_count=0,
                    ins_part=last.ins_part,
                    outs_part=last.outs_part,
                    result_type=None,
                    span=last.span,
                    indent=last.indent,
                )

        if entry.name == "cudnnSoftmaxForwardOut_tensor":
            # Standalone attention softmax is out-of-place: step1 reads the
            # scores tensor and writes the exp-shifted values into `out`.
            vector_inst = instances[i + 1]
            score_names = _extract_ssa_names(vector_inst.ins_part)
            score_types = _extract_ssa_types(vector_inst.ins_part)
            out_names = _extract_ssa_names(vector_inst.outs_part)
            out_types = _extract_ssa_types(vector_inst.outs_part)
            if (len(score_names) < 1 or len(out_names) < 1 or
                    not score_types or not out_types or
                    _sniff_elem_type(score_types[0]) != "f32" or
                    _sniff_elem_type(out_types[0]) != "f32"):
                report.append(("softmax_out_reject", i, entry.name))
                i += 1
                continue
            operands = [score_names[0], out_names[0]]
            operand_types = [score_types[0], out_types[0]]
            binds = {}
            # The two view definitions live between the matched reduction
            # generics.  Depending on whether submap lowering ran before the
            # matcher, these are either tensor.extract_slice or
            # polygeist.submap operations.  Re-emit them when replacing the
            # full fused span; otherwise the launch retains dangling SSA
            # references.
            preserved_defs: list[str] = []
            for name in operands:
                dm = re.search(
                    rf"^\s*{re.escape(name)}\s*=\s*(?:tensor\.extract_slice|"
                    rf"polygeist\.submap\().*$",
                    text, re.MULTILINE)
                if not dm:
                    preserved_defs = []
                    break
                preserved_defs.append(dm.group(0))
            if len(preserved_defs) != 2:
                report.append(("softmax_slice_reject", i, entry.name))
                i += n
                continue
            # The normalized tensor returned by the source chain is produced
            # by a submapInverse immediately after the final divide generic.
            # Make that existing SSA value the launch result and replace the
            # entire reduction/normalization chain.  Keeping the generic's
            # result name here would leave the trailing inverses referring to
            # deleted max/sum intermediates.
            final_result = last.result_ssa
            final_type = last.result_type
            if last.result_ssa is not None:
                inverse_match = re.search(
                    rf"^\s*(%[\w.$-]+)\s*=\s*polygeist\.submapInverse\([^\n]*"
                    rf"{re.escape(last.result_ssa)}[^\n]*\)\s*\{{[^\n]*\}}\s*:"
                    rf"[^\n]*->\s*([^\n]+)$",
                    text[last.span[1]:], re.MULTILINE)
                if inverse_match:
                    final_result = inverse_match.group(1)
                    final_type = inverse_match.group(2).strip()
                    custom_edit_span = (
                        start, last.span[1] + inverse_match.end())
            last = LinalgInstance(
                result_ssa=final_result,
                result_count=1,
                ins_part=last.ins_part,
                outs_part=last.outs_part,
                result_type=final_type,
                span=last.span,
                indent=last.indent,
            )
            rendered = render_launch(
                emit_name, last.result_ssa, last.result_type,
                operands, last.indent, {}, [], operand_types=operand_types,
                scalar_type_map=scalar_types, result_count=last.result_count)
            custom_launch_line = "\n".join([*preserved_defs, rendered])
            replace_full_span = True

        if entry.name == "whisperExpShiftSum_f32_tensor":
            inst = instances[i]
            x_names = _extract_ssa_names(inst.ins_part)
            x_types = _extract_ssa_types(inst.ins_part)
            out_names = _extract_ssa_names(inst.outs_part)
            out_types = _extract_ssa_types(inst.outs_part)
            max_bound = binds.get("%max")
            max_ssa = (
                max_bound[1]
                if isinstance(max_bound, tuple) and len(max_bound) == 2 and
                max_bound[0] == "Cap"
                else None
            )
            if (len(x_names) != 1 or len(out_names) != 2 or
                    inst.result_ssa is None or inst.result_count != 2 or
                    inst.result_type is None or max_ssa is None or
                    not x_types or len(out_types) != 2 or
                    _sniff_elem_type(x_types[0]) != "f32" or
                    _sniff_elem_type(out_types[0]) != "f32" or
                    _sniff_elem_type(out_types[1]) != "f32"):
                report.append(("exp_shift_sum_reject", i, entry.name))
                i += 1
                continue
            operands = [x_names[0], out_names[0], out_names[1], max_ssa]
            operand_types = [
                x_types[0],
                out_types[0],
                out_types[1],
                scalar_types.get(max_ssa, "f32"),
            ]
            binds = {}
            custom_launch_line = _render_whisper_exp_shift_sum_launch(
                entry.name,
                inst.result_ssa,
                inst.result_count,
                inst.result_type,
                operands,
                operand_types,
                inst.indent,
            )

        if entry.name == "cudaMaskSelect_f32_tensor":
            pos = _extract_cmpi_rhs_i32(bodies[i].body_lines)
            if not pos:
                report.append(("mask_select_reject", i, entry.name))
                i += 1
                continue
            elems = [_sniff_elem_type(t) for t in operand_types[:2]]
            ranks = [_tensor_rank(t) for t in operand_types[:2]]
            if elems != ["f32", "f32"] or ranks != [1, 1]:
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += 1
                continue
            operands = operands + [pos]
            operand_types = operand_types + [scalar_types.get(pos, "i32")]
            binds = {}

        if entry.name in ("cudaAdd_f32_tensor", "cudaSwiGLU_f32_tensor"):
            elems = [_sniff_elem_type(t) for t in operand_types[:3]]
            ranks = [_tensor_rank(t) for t in operand_types[:3]]
            if elems != ["f32", "f32", "f32"] or ranks != [1, 1, 1]:
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += 1
                continue

        if entry.name in ("cudaRopeMulMulSub_f32_tensor",
                          "cudaRopeMulMulAdd_f32_tensor"):
            # Preserve the linalg operand order. The generic rank-sort above is
            # valid for commutative BLAS templates, but RoPE semantics depend
            # on [2D, 1D, 2D, 1D, out] ordering.
            in_names = _extract_ssa_names(instances[i].ins_part)
            in_types = _extract_ssa_types(instances[i].ins_part)
            out_names = _extract_ssa_names(instances[i].outs_part)
            out_types = _extract_ssa_types(instances[i].outs_part)
            operands = in_names + out_names
            operand_types = in_types + out_types
            elems = [_sniff_elem_type(t) for t in operand_types[:5]]
            ranks = [_tensor_rank(t) for t in operand_types[:5]]
            if (elems != ["f32"] * 5 or ranks != [2, 1, 2, 1, 2]):
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += 1
                continue

        if entry.name in ("elemwise_div_scalar", "elemwise_scale_input_1D",
                          "elemwise_axpby_inputs_1D"):
            # These templates are useful for algebraic recognition, but the
            # current ABI lowering path does not have a complete runtime shim
            # for them. In particular elemwise_scale_input_1D's matched launch
            # does not yet surface the scalar scale factor as an operand, so
            # lowering it would lose semantics. Keep the linalg.generic in
            # place so downstream MLIR lowering handles it as residual tensor
            # code.
            report.append(("unsupported_abi_reject", i, entry.name))
            i += 1
            continue

        if entry.name == "parboil_stencil_7pt_tensor":
            inst = instances[i]
            in_names = _extract_ssa_names(inst.ins_part)
            out_names = _extract_ssa_names(inst.outs_part)
            grid = (_parboil_7pt_flat_grid_info(text, in_names, out_names[0])
                    if len(out_names) == 1 else None)
            inverse = (_find_flat_submap_inverse(
                text, inst.span[1], inst.result_ssa)
                if inst.result_ssa is not None else None)
            center_bound = binds.get("%center_scale")
            neighbor_bound = binds.get("%neighbor_scale")
            center_scale = (center_bound[1] if isinstance(center_bound, tuple)
                            and center_bound[0] == "Cap" else None)
            neighbor_scale = (neighbor_bound[1] if isinstance(neighbor_bound, tuple)
                              and neighbor_bound[0] == "Cap" else None)
            if (grid is None or inverse is None or center_scale is None or
                    neighbor_scale is None):
                report.append(("stencil3d_layout_reject", i, entry.name))
                i += 1
                continue
            input_base, output_base, symbols, sizes = grid
            result_ssa, result_type, inverse_base, inverse_operands, inverse_span = inverse
            input_type = _infer_tensor_type(text, input_base)
            output_type = _infer_tensor_type(text, output_base)
            if (inverse_base != output_base or inverse_operands != symbols + sizes or
                    input_type is None or output_type is None or
                    _sniff_elem_type(input_type) != "f32" or
                    _sniff_elem_type(output_type) != "f32" or
                    _shaped_rank(input_type) != 1 or _shaped_rank(output_type) != 1):
                report.append(("stencil3d_layout_reject", i, entry.name))
                i += 1
                continue
            emit_name = "cudnnStencil3D7pt_f32_flat_tensor"
            replace_full_span = True
            custom_edit_span = (start, inverse_span[1])
            custom_launch_line = _render_flat_7pt_conv3d_launch(
                result_ssa, result_type, input_base, input_type,
                output_base, output_type, center_scale, neighbor_scale,
                symbols, sizes, inst.indent)
            binds = {}

        if entry.name == "miniamr_weighted_27pt_tensor":
            accum_inst = instances[i + n - 1]
            accum_ins = _extract_ssa_names(accum_inst.ins_part)
            accum_in_types = _extract_ssa_types(accum_inst.ins_part)
            out_names = _extract_ssa_names(instances[i].outs_part)
            out_types = _extract_ssa_types(instances[i].outs_part)
            elem = _sniff_elem_type(out_types[0]) if out_types else None
            if (accum_inst.result_ssa is None or accum_inst.result_type is None
                    or len(out_names) != 1 or not out_types
                    or elem not in ("f32", "f64")):
                report.append(("conv3d_ntap_reject", i, entry.name))
                i += 1
                continue
            window = _miniamr_weighted27_window_info(
                text, accum_ins, accum_in_types
            )
            if window is None:
                report.append(("conv3d_ntap_reject", i, entry.name))
                i += 1
                continue
            input_base, input_type, weight_ssa, width = window
            try:
                weight_idx = accum_ins.index(weight_ssa)
            except ValueError:
                report.append(("conv3d_ntap_reject", i, entry.name))
                i += 1
                continue
            weight_type = accum_in_types[weight_idx]
            if (_sniff_elem_type(input_type) != elem
                    or _sniff_elem_type(weight_type) != elem):
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += 1
                continue
            emit_name = (
                "cudnnConvolution3D_ntap_f32_tensor"
                if elem == "f32" else "cudnnConvolution3D_ntap_tensor"
            )
            replace_full_span = True
            binds = {}
            custom_launch_line = _render_ntap_conv3d_tensor_launch(
                emit_name,
                accum_inst.result_ssa,
                accum_inst.result_type,
                input_base,
                input_type,
                out_names[0],
                out_types[0],
                weight_ssa,
                weight_type,
                width,
                accum_inst.indent,
            )

        if entry.name == "cufftZ2Z_1D_tensor":
            dft_inst = instances[i + n - 1]
            in_names = _extract_ssa_names(dft_inst.ins_part)
            in_types = _extract_ssa_types(dft_inst.ins_part)
            zero_out_names = _extract_ssa_names(instances[i].outs_part)
            zero_out_types = _extract_ssa_types(instances[i].outs_part)
            if (dft_inst.result_ssa is None or dft_inst.result_type is None
                    or len(in_names) != 2 or len(in_types) != 2
                    or len(zero_out_names) != 1 or len(zero_out_types) != 1):
                report.append(("cufft_reject", i, entry.name))
                i += 1
                continue
            parsed_inputs = [
                _parse_static_extract_slice_offset(text, name)
                for name in in_names
            ]
            if any(p is None for p in parsed_inputs):
                report.append(("cufft_reject", i, entry.name))
                i += 1
                continue
            input_base0, off0 = parsed_inputs[0]
            input_base1, off1 = parsed_inputs[1]
            if input_base0 != input_base1 or sorted([off0, off1]) != [(0, 0), (0, 1)]:
                report.append(("cufft_reject", i, entry.name))
                i += 1
                continue
            input_type = _infer_tensor_type(text, input_base0)
            output_ssa = zero_out_names[0]
            output_type = zero_out_types[0]
            elem = _sniff_elem_type(output_type)
            inverse_flag = _dft1d_inverse_flag(bodies[i + n - 1].constants)
            inserted = _find_insert_slice_of_result(
                text, dft_inst.span[1], dft_inst.result_ssa
            )
            if (input_type is None or elem not in ("f32", "f64")
                    or _sniff_elem_type(input_type) != elem
                    or inverse_flag is None or inserted is None):
                report.append(("cufft_reject", i, entry.name))
                i += 1
                continue
            result_ssa, result_type, insert_span = inserted
            emit_name = (
                "cufftC2C_1D_tensor" if elem == "f32"
                else "cufftZ2Z_1D_tensor"
            )
            custom_launch_line = _render_cufft_1d_tensor_launch(
                emit_name,
                result_ssa,
                result_type,
                input_base0,
                input_type,
                output_ssa,
                output_type,
                inverse_flag,
                dft_inst.indent,
            )
            if custom_launch_line is None:
                report.append(("cufft_reject", i, entry.name))
                i += 1
                continue
            replace_full_span = True
            custom_edit_span = (start, insert_span[1])

        if entry.name == "cutensornetTensorProduct3D_f32_tensor":
            contract_inst = instances[i + n - 1]
            in_names = _extract_ssa_names(contract_inst.ins_part)
            in_types = _extract_ssa_types(contract_inst.ins_part)
            out_names = _extract_ssa_names(contract_inst.outs_part)
            out_types = _extract_ssa_types(contract_inst.outs_part)
            ranks = [_tensor_rank(t) for t in in_types + out_types]
            elems = [_sniff_elem_type(t) for t in in_types + out_types]
            if (contract_inst.result_ssa is None or
                    contract_inst.result_type is None or
                    len(in_names) != 4 or len(out_names) != 1 or
                    ranks != [6, 6, 6, 6, 6] or
                    elems not in (["f32"] * 5, ["f64"] * 5)):
                report.append(("cutensornet_reject", i, entry.name))
                i += 1
                continue
            emit_name = ("cutensornetTensorProduct3D_f64_tensor"
                         if elems[0] == "f64" else entry.name)
            # Keep the matched zero initializer in place: it proves that the
            # accumulator has beta=0 semantics. Replace only the contraction
            # and pass its rank-6 views; the MLIR lowering unwraps those views
            # to the original psi/u/out buffers and derives KQ/KP from dims.
            operands = in_names + out_names
            operand_types = in_types + out_types
            binds = {}
            last = contract_inst
            replace_full_span = True
            custom_edit_span = contract_inst.span
            binds = {}

        generalized_stencil_entry = entry.name in (
            "cudnnConvolution2D_ntap", "cudnnConvolution2D_ntap_tensor")
        custen_fixed_stencil_entry = (
            stencil_backend == "custen" and entry.name in (
                "cudnnConvolution2D_9tap",
                "cudnnConvolution2D_9tap_tensor",
                "cudnnConvolution2D_25tap"))
        if generalized_stencil_entry or custen_fixed_stencil_entry:
            in_names = _extract_ssa_names(instances[i].ins_part)
            in_types = _extract_ssa_types(instances[i].ins_part)
            out_names = _extract_ssa_names(instances[i].outs_part)
            out_types = _extract_ssa_types(instances[i].outs_part)
            if len(out_names) != 1 or len(in_names) == 0:
                report.append(("ntap_stencil_reject", i, entry.name))
                i += 1
                continue
            is_tensor_ntap = entry.name.endswith("_tensor")
            grid = (
                _conv2d_ntap_tensor_grid_info(text, in_names, out_names[0])
                if is_tensor_ntap
                else _conv2d_ntap_grid_info(text, in_names, out_names[0])
            )
            if grid is None:
                report.append(("ntap_stencil_reject", i, entry.name))
                i += 1
                continue
            width, top_left_ssa, ordered_indices = grid
            elem = _sniff_elem_type(in_types[0]) if in_types else None
            if elem not in ("f32", "f64"):
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += 1
                continue
            if any(_sniff_elem_type(t) != elem for t in in_types + out_types):
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += 1
                continue
            top_left_idx = in_names.index(top_left_ssa)
            inline_weights = bodies[i].inline_weights_per_in
            if not inline_weights or len(inline_weights) != len(in_names):
                report.append(("ntap_weight_reject", i, entry.name))
                i += 1
                continue
            ordered_weights = [inline_weights[idx] for idx in ordered_indices]
            if is_tensor_ntap:
                if last.result_ssa is None or last.result_type is None:
                    report.append(("ntap_stencil_reject", i, entry.name))
                    i += 1
                    continue
                emit_name = (
                    "cudnnConvolution2D_ntap_f32_tensor"
                    if elem == "f32" else "cudnnConvolution2D_ntap_tensor"
                )
                if stencil_backend == "custen":
                    if elem != "f64":
                        report.append(("custen_dtype_reject", i, entry.name))
                        i += 1
                        continue
                    emit_name = "custenStencil2DXY_f64_tensor"
                custom_launch_line = _render_ntap_conv_tensor_launch(
                    emit_name,
                    last.result_ssa,
                    last.result_type,
                    top_left_ssa,
                    in_types[top_left_idx],
                    out_names[0],
                    out_types[0],
                    width,
                    ordered_weights,
                    last.indent,
                    scalar_types,
                    bodies[i].constants,
                    elem,
                    i,
                )
            else:
                emit_name = "cudnnConvolution2D_ntap_f32" if elem == "f32" else "cudnnConvolution2D_ntap"
                if stencil_backend == "custen":
                    if elem != "f64":
                        report.append(("custen_dtype_reject", i, entry.name))
                        i += 1
                        continue
                    emit_name = "custenStencil2DXY_f64_memref"
                custom_launch_line = _render_ntap_conv_launch(
                    emit_name,
                    top_left_ssa,
                    in_types[top_left_idx],
                    out_names[0],
                    out_types[0],
                    width,
                    ordered_weights,
                    last.indent,
                    scalar_types,
                    bodies[i].constants,
                    elem,
                    i,
                )

        if entry.name in ("cudnnConvolution2D_9tap",
                          "cudnnConvolution2D_9tap_tensor"):
            elem = _sniff_elem_type(all_tensor_in_types[0]) if all_tensor_in_types else "f64"
            if elem and elem != "f64":
                emit_name = f"{entry.name}_{elem}"
        if entry.name == "cudnnConvolution2D_25tap":
            elem = _sniff_elem_type(all_tensor_in_types[0]) if all_tensor_in_types else "f64"
            if elem not in (None, "f64", "f32"):
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += 1
                continue
            if elem == "f32":
                emit_name = "cudnnConvolution2D_25tap_f32"

        # Transpose discriminator for gemv. The template `Out + In(0)*In(1)`
        # with 1 parallel + 1 reduction iter matches both `y = A·x` (no
        # transpose) and `y = Aᵀ·x` (transposed). The launch operands look
        # identical in either case — what distinguishes them is whether A's
        # first indexing-map dim matches the output's first dim (no-transpose)
        # or the other input's dim (transposed). Switch the concrete emit
        # name by both transpose and dtype so f32 tensor GEMV goes to SGEMV
        # while the shared algebraic template remains dtype-agnostic.
        # AᵀA / A·Aᵀ → cublasDsyrk operand-alias discriminator.
        # If a gemm-shape composition's two inputs resolve to the same
        # underlying tensor (after walking through polygeist.submap),
        # the math is a symmetric rank-K update — half the flops via
        # cublasDsyrk (writes only the upper triangle). Cheap check:
        # scan the matched body's ins SSA names, walk back to find the
        # defining ops, compare the submap-base SSA name.
        if entry.name in ("cublasDgemm", "cublasDgemm_simple",
                          "cublasDgemm_subtract",
                          "cublasDgemm_strided_batched_subtract",
                          "cublasDgemm_alpha_only"):
            # Multi-step GEMM compositions subsume their first producer
            # (typically output scaling or initialization). Any remaining
            # view/update uses of that producer must instead use the original
            # destination passed to the fused external-library call.
            if n > 1 and instances[i].result_ssa and outs0:
                composition_root_rewires = [(
                    instances[i].result_ssa, outs0[0])]
            gemm_inst = instances[i + n - 1]  # last (contraction) generic
            gemm_ins = _extract_ssa_names(gemm_inst.ins_part)
            gemm_in_types = _extract_ssa_types(gemm_inst.ins_part)
            gemm_outs = _extract_ssa_names(gemm_inst.outs_part)
            gemm_out_types = _extract_ssa_types(gemm_inst.outs_part)
            # A scale-and-contract composition can have extract_slice views
            # between its two generics.  The opaque GEMM must consume the
            # contraction's actual destination view, while the deleted scale
            # result is rewired to the original storage so that view remains
            # defined.
            if (n > 1 and len(gemm_ins) == 2 and len(gemm_outs) == 1 and
                    len(gemm_in_types) == 2 and len(gemm_out_types) == 1):
                if (composition_root_rewires and
                        gemm_outs[0] == composition_root_rewires[0][0]):
                    gemm_outs[0] = composition_root_rewires[0][1]
                operands = gemm_ins + gemm_outs
                operand_types = gemm_in_types + gemm_out_types
            # The full alpha/beta ABI requires both scalars even when the
            # algebraic template bound one of them to a literal (for example
            # alpha=1 in PolyBench 2mm).  Materialize such literals explicitly
            # instead of silently dropping them in render_launch.
            if entry.name == "cublasDgemm":
                for scalar_name in ("%beta", "%alpha"):
                    bound = binds.get(scalar_name)
                    if (isinstance(bound, tuple) and len(bound) == 2 and
                            bound[0] == "Lit"):
                        suffix = scalar_name.lstrip("%")
                        scalar = _derived_ssa_name(last.result_ssa, suffix)
                        value = repr(float(bound[1]))
                        if "." not in value and "e" not in value and "E" not in value:
                            value += ".0"
                        pre_launch_lines.append(
                            f"{last.indent}{scalar} = arith.constant {value} : f64")
                        binds[scalar_name] = ("Cap", scalar)
                        scalar_types[scalar] = "f64"
            if len(gemm_ins) == 2:
                # Walk each input SSA through polygeist.submap definitions
                # to find the underlying base. The submap defining-op line
                # has the form `%X = polygeist.submap(%base, ...) ...`.
                def _resolve_submap_base(ssa_name: str) -> str | None:
                    pat = re.compile(
                        rf'\s*{re.escape(ssa_name)}\s*=\s*polygeist\.submap'
                        rf'\s*\(\s*(%[\w_]+)\s*[,)]'
                    )
                    m = pat.search(text)
                    return m.group(1) if m else None
                base0 = _resolve_submap_base(gemm_ins[0]) or gemm_ins[0]
                base1 = _resolve_submap_base(gemm_ins[1]) or gemm_ins[1]
                def _map_outputs(txt: str) -> list[str]:
                    mm = re.search(r"->\s*\(([^)]*)\)>", txt)
                    return [s.strip() for s in mm.group(1).split(",")] if mm else []
                maps = bodies[i + n - 1].indexing_maps
                in0_dims = _map_outputs(maps[0]) if len(maps) >= 2 else []
                in1_dims = _map_outputs(maps[1]) if len(maps) >= 2 else []
                # Same-base GEMM is not automatically SYRK. A true symmetric
                # rank-k update has both inputs using the reduction dim in the
                # same coordinate position, e.g. A[i,k] * A[j,k] or
                # A[k,i] * A[k,j]. A dense square A[i,k] * A[k,j] is a normal
                # GEMM even though both operands resolve to the same base.
                same_base_syrk = (
                    base0 == base1 and len(in0_dims) == 2 and len(in1_dims) == 2
                    and (in0_dims[1] == in1_dims[1] or
                         in0_dims[0] == in1_dims[0])
                )
                if same_base_syrk:
                    emit_name = "cublasDsyrk_alias"
            elem = _sniff_elem_type(operand_types[0]) if operand_types else None
            operand_ranks = [_tensor_rank(t) for t in operand_types[:3]]
            if (entry.name == "cublasDgemm_strided_batched_subtract" and
                    elem == "f64" and operand_ranks == [3, 3, 3]):
                emit_name = "cublasDgemm_strided_batched_subtract"
            elif (entry.name == "cublasDgemm" and elem == "f32" and
                    operand_ranks == [3, 3, 2]):
                # Parboil basic SGEMM: the three identity-indexed operands are
                # rank-3 submaps whose affine maps encode A[m,k], B[n,k], and
                # C[m,n] over flat column-major storage.  The lowering verifies
                # those maps before accepting this ABI.
                emit_name = (
                    "cublasSgemm_broadcast3d_colmajor_nt_alpha_beta")
                # The first generic's scaled-C result feeds an intervening
                # submapInverse before the contraction.  Since the fused GEMM
                # launch performs beta scaling itself, preserve that view
                # chain but point it at the original C view after deleting the
                # first generic.
                if instances[i].result_ssa and outs0:
                    composition_root_rewires = [(
                        instances[i].result_ssa, outs0[0])]
            elif (entry.name == "cublasDgemm_simple" and elem == "f32" and
                    operand_ranks == [3, 3, 3]):
                # Darknet im2col+GEMM reaches linalg as a rank-3 broadcasted
                # view: logical (N, K, M) iteration, but the underlying buffers
                # are the usual 2D row-major A[M,K], B[K,N], C[M,N]. Emit a
                # dedicated symbol so ABI lowering can unwrap the submaps and
                # call cuBLAS SGEMM.
                emit_name = "cublasSgemm_broadcast3d_simple"
            elif (elem == "f32" and operand_ranks == [2, 2, 2]):
                maps = bodies[i + n - 1].indexing_maps
                if len(maps) != 3:
                    report.append(("layout_reject", i, entry.name))
                    i += 1
                    continue
                a_dims = _map_outputs(maps[0])
                b_dims = _map_outputs(maps[1])
                c_dims = _map_outputs(maps[2])
                if len(a_dims) != 2 or len(b_dims) != 2 or len(c_dims) != 2:
                    report.append(("layout_reject", i, entry.name))
                    i += 1
                    continue
                m_dim, n_dim = c_dims
                a_trans = a_dims[1] == m_dim and a_dims[0] != m_dim
                b_trans = b_dims[0] == n_dim and b_dims[1] != n_dim
                a_valid = ((not a_trans and a_dims[0] == m_dim) or
                           (a_trans and a_dims[1] == m_dim))
                b_valid = ((not b_trans and b_dims[1] == n_dim) or
                           (b_trans and b_dims[0] == n_dim))
                a_k = a_dims[0] if a_trans else a_dims[1]
                b_k = b_dims[1] if b_trans else b_dims[0]
                if not a_valid or not b_valid or a_k != b_k:
                    report.append(("layout_reject", i, entry.name))
                    i += 1
                    continue
                layout = (("t" if a_trans else "n") +
                          ("t" if b_trans else "n"))
                if entry.name == "cublasDgemm":
                    emit_name = f"cublasSgemm_{layout}_alpha_beta"
                elif entry.name == "cublasDgemm_alpha_only":
                    emit_name = f"cublasSgemm_{layout}_alpha"
                else:
                    emit_name = f"cublasSgemm_{layout}"
            elif elem != "f64" or operand_ranks != [2, 2, 2]:
                # Do not let generic rank-3/strided contractions masquerade as
                # the plain double GEMM ABI. The extended Llama split-Q/K
                # fixture intentionally leaves these as residual linalg until
                # we add a real batched/split projection lowering.
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += 1
                continue
        if entry.name == "memset_zero_1D":
            elem = _sniff_elem_type(outs0_types[0]) if outs0_types else None
            if elem == "f32":
                emit_name = "memset_zero_1D_f32"
            elif elem != "f64":
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += 1
                continue
        if entry.name == "memset_zero_2D":
            elem = _sniff_elem_type(outs0_types[0]) if outs0_types else None
            if elem == "f32":
                emit_name = "memset_zero_2D_f32"
            elif elem != "f64":
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += 1
                continue
        if entry.name in ("reduce_sum_1D", "cudnnReduceProduct_f32",
                          "cudnnReduceMin_f32", "cudnnReduceMax_f32",
                          "cudnnReduceMinMax_f32"):
            elems = [_sniff_elem_type(t) for t in operand_types]
            ranks = [_shaped_rank(t) for t in operand_types]
            if entry.name == "reduce_sum_1D":
                reduction_body = bodies[i]
                diagonal = (
                    len(elems) == 2 and elems == ["f32", "f32"] and
                    ranks == [2, 0] and
                    len(reduction_body.indexing_maps) == 2 and
                    re.search(r"\(d0\)\s*->\s*\(d0,\s*d0\)",
                              reduction_body.indexing_maps[0]) is not None and
                    re.search(r"\(d0\)\s*->\s*\(\s*\)",
                              reduction_body.indexing_maps[1]) is not None)
                if diagonal:
                    emit_name = "cudnnReduceTrace_f32"
                elif (len(elems) != 2 or elems[0] != elems[1] or
                      elems[0] not in ("f32", "f64") or ranks != [1, 0]):
                    report.append(("rank_dtype_or_layout_reject", i,
                                   entry.name))
                    i += n
                    continue
                else:
                    emit_name = "cudnnReduceSum_" + elems[0]
            elif entry.name == "cudnnReduceMinMax_f32":
                if (len(elems) != 3 or elems != ["f32", "f32", "f32"] or
                        ranks != [1, 0, 0]):
                    report.append(("rank_dtype_or_layout_reject", i,
                                   entry.name))
                    i += n
                    continue
            elif (len(elems) != 2 or elems != ["f32", "f32"] or
                  ranks != [1, 0]):
                report.append(("rank_dtype_or_layout_reject", i,
                               entry.name))
                i += n
                continue
        if entry.name.startswith("cubSegmented"):
            elems = [_sniff_elem_type(t) for t in operand_types]
            ranks = [_shaped_rank(t) for t in operand_types]
            if (custom_launch_line is not None and entry.name in (
                    "cubSegmentedSum_f32_memref",
                    "cubSegmentedSum_f64_memref",
                    "cubSegmentedMin_f32_memref",
                    "cubSegmentedMax_f32_memref",
                    "cubSegmentedLogicalAnd_i32_memref")):
                # The strict whole-region handlers above validate the physical
                # memref ranks and element types before clearing the original
                # tensor operands.
                legal = True
            elif entry.name == "cubSegmentedPrefixSum_f32":
                legal = elems == ["f32", "i32", "f32"] and ranks == [2, 1, 1]
            elif entry.name == "cubSegmentedPrefixLogicalAnd_i32":
                legal = elems == ["i32", "i32", "i32"] and ranks == [2, 1, 1]
            elif entry.name == "cubSegmentedCountNonzero2D_f32_tensor":
                legal = elems == ["f32", "i32"] and ranks == [2, 1]
            elif entry.name == "cubSegmentedInclusiveProduct2D_f32_tensor":
                legal = elems == ["f32", "f32", "f32"] and ranks == [2, 2, 1]
            else:
                legal = elems == ["i32", "i32"] and ranks == [2, 1]
            if not legal:
                report.append(("rank_dtype_or_layout_reject", i,
                               entry.name))
                i += n
                continue
        if entry.name == "cublasSgemm_broadcast3d_memref":
            elem = _sniff_elem_type(operand_types[0]) if operand_types else None
            operand_ranks = [_tensor_rank(t) for t in operand_types[:3]]
            if elem != "f32" or operand_ranks != [3, 3, 3]:
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += 1
                continue
        if entry.name == "cublasDgemv_strided_batched_subtract":
            elems = [_sniff_elem_type(t) for t in operand_types[:3]]
            ranks = [_tensor_rank(t) for t in operand_types[:3]]
            maps = bodies[i].indexing_maps
            def _batched_gemv_map_outputs(txt: str) -> list[str]:
                mm = re.search(r"->\s*\(([^)]*)\)>", txt)
                return ([s.strip() for s in mm.group(1).split(",")]
                        if mm and mm.group(1).strip() else [])
            map_dims = [_batched_gemv_map_outputs(m) for m in maps]
            layout_ok = False
            if len(map_dims) == 3:
                a_dims, x_dims, y_dims = map_dims
                layout_ok = (
                    len(a_dims) == 3 and len(x_dims) == 2 and
                    len(y_dims) == 2 and a_dims[0] == x_dims[0] == y_dims[0]
                    and a_dims[1] == y_dims[1] and a_dims[2] == x_dims[1]
                )
            if elems != ["f64", "f64", "f64"] or ranks != [3, 2, 2] or not layout_ok:
                report.append(("rank_dtype_or_layout_reject", i, entry.name))
                i += 1
                continue
        if entry.name in ("cublasDgemv", "cublasDgemv_subtract") and n == 1:
            elems = [_sniff_elem_type(t) for t in operand_types[:3]]
            elem = elems[0] if elems else None
            operand_ranks = [_tensor_rank(t) for t in operand_types[:3]]
            if (elem not in ("f64", "f32") or
                    len(elems) != 3 or any(e != elem for e in elems) or
                    operand_ranks != [2, 1, 1]):
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += 1
                continue
            mb = bodies[i]
            transposed = False
            if len(mb.indexing_maps) == 3:
                def _map_outputs(txt: str) -> list[str]:
                    mm = re.search(r"->\s*\(([^)]*)\)>", txt)
                    return [s.strip() for s in mm.group(1).split(",")] if mm else []
                A_dims = _map_outputs(mb.indexing_maps[0])
                y_dims = _map_outputs(mb.indexing_maps[2])
                if A_dims and y_dims and A_dims[0] != y_dims[0]:
                    transposed = True
            if elem == "f32":
                if entry.name == "cublasDgemv_subtract":
                    report.append(("rank_or_dtype_reject", i, entry.name))
                    i += 1
                    continue
                emit_name = "cublasSgemv_T" if transposed else "cublasSgemv"
            else:
                prefix = ("cublasDgemv_subtract"
                          if entry.name == "cublasDgemv_subtract"
                          else "cublasDgemv")
                emit_name = prefix + ("_T" if transposed else "")

            # A masked initializer may intentionally restrict a GEMV update
            # to a dynamic suffix (for example, preserving an already-filled
            # triangular prefix). A full-vector BLAS call is not equivalent.
            # Recognize the generic shape of that mask and reject the
            # otherwise tempting full-vector cuBLAS substitution.
            if elem == "f64" and transposed and entry.name == "cublasDgemv":
                output_name = operands[2]
                producer_index = next(
                    (p for p in range(i) if
                     instances[p].result_ssa == output_name), None)
                if producer_index is not None:
                    producer = instances[producer_index]
                    producer_text = text[producer.span[0]:producer.span[1]]
                    masked = ("linalg.index" in producer_text and
                              "arith.select" in producer_text)
                    if masked:
                        # A full-vector GEMV is not equivalent to a
                        # dynamically masked/triangular update. Leave this
                        # operation as residual Linalg until an external ABI
                        # can express the exact active slice and its
                        # loop-carried storage without stale snapshots.
                        report.append(("masked_gemv_reject", i, entry.name))
                        i += n
                        continue
        if emit_name not in ABI_LOWERABLE_KERNELS:
            report.append(("unsupported_abi_reject", list(range(i, i + n)),
                           emit_name))
            i += n
            continue

        if custom_launch_line is not None:
            launch_line = custom_launch_line
        else:
            # When the matched composition opts in to weight surfacing, hand the
            # encoder's in_arg → constant_ssa map from the FIRST matched body to
            # render_launch. (Only single-step weighted-stencil templates use
            # this today; if we ever support multi-step weighted compositions,
            # this needs to combine bodies appropriately.)
            inline_weights = (bodies[i].inline_weights_per_in
                               if getattr(entry, "surface_inline_weights", False)
                               else None)
            # Surface the weight scalars with the operand's element type
            # (f64 / f32 / f16 / bf16 / iNN), so the launch op's signature is
            # internally consistent and the cuDNN shim's scalar args match.
            weight_ty = "f64"
            if inline_weights and all_tensor_in_types:
                sniffed = _sniff_elem_type(all_tensor_in_types[0])
                if sniffed:
                    weight_ty = sniffed

            launch_line = render_launch(
                emit_name, last.result_ssa, last.result_type,
                operands, last.indent, binds, [],
                operand_types=operand_types,
                scalar_type_map=scalar_types,
                inline_weights=inline_weights,
                inline_weight_type=weight_ty,
                # Pass the body's per-SSA constant values so render_launch can
                # materialise summed-constant ops for the polybench conv3d
                # multi-coefficient case.
                body_constants=bodies[i].constants if inline_weights else None,
                result_count=last.result_count,
            )
            if pre_launch_lines:
                launch_line = "\n".join(pre_launch_lines + [launch_line])
        if max_launches is not None and emitted_launches >= max_launches:
            report.append(("launch_limit", list(range(i, i + n)), emit_name))
            i += n
            continue
        # Only report a match after it has resolved to an existing, ABI-
        # lowerable implementation and passed all operand/layout checks.
        report.append(("match", list(range(i, i + n)), emit_name))
        emitted_launches += 1
        if custom_first_launch_line is not None:
            emitted_launches += 1
        if redundant_zero_fill_span is not None:
            edits.append((*redundant_zero_fill_span, ""))
        if roundtrip_markers:
            # last.indent has a leading newline ("\n    ") because the parser
            # captures the line break before the op. Use only the spaces.
            indent_spaces = last.indent.lstrip("\n").rstrip("\n")
            # The original span starts mid-line at "\n    %X = linalg.generic..."
            # so we strip the leading newline from the captured block and
            # restore it ourselves once, before the BEGIN marker.
            original_end = (
                custom_edit_span[1]
                if replace_full_span and custom_edit_span is not None
                else end
            )
            original_block = text[start:original_end]
            stripped = original_block[1:] if original_block.startswith("\n") else original_block
            commented = "\n".join(
                f"{indent_spaces}// {ln}" if ln.strip() else f"{indent_spaces}//"
                for ln in stripped.split("\n")
            )
            replacement = (
                f"\n{indent_spaces}// POLYGEIST-MATCH-BEGIN-{entry.name}\n"
                f"{commented}\n"
                f"{indent_spaces}// POLYGEIST-MATCH-END\n"
                f"{indent_spaces}{launch_line.lstrip()}"
            )
        else:
            replacement = launch_line
        if replace_full_span:
            edit_start, edit_end = custom_edit_span or (start, end)
            edits.append((edit_start, edit_end, replacement))
        elif n == 1:
            # Single-step composition: one generic, one launch. No
            # intervening ops to preserve.
            edits.append((start, end, replacement))
        else:
            # Multi-step: emit the launch in place of the LAST generic;
            # delete the earlier generics individually so any text between
            # them (intervening defs like polygeist.submap) is preserved
            # verbatim. The earlier-generic deletions are span replacements
            # to the empty string.
            for j in range(n - 1):
                inst_j = instances[i + j]
                earlier_replacement = (
                    custom_first_launch_line
                    if j == 0 and custom_first_launch_line is not None
                    else ""
                )
                edits.append((inst_j.span[0], inst_j.span[1],
                              earlier_replacement))
            if composition_root_rewires or tail_only_rewires:
                # Rewrite only the gaps between matched generics.  A single
                # span worked for two-step compositions, but with three or
                # more steps it overlaps the generic-deletion edits and can
                # accidentally resurrect a deleted producer.
                for j in range(n - 1):
                    middle_start = instances[i + j].span[1]
                    middle_end = instances[i + j + 1].span[0]
                    middle = text[middle_start:middle_end]
                    for old_root, new_root in composition_root_rewires:
                        middle = re.sub(
                            rf"(?<![\w]){re.escape(old_root)}(?![\w])",
                            new_root, middle)
                    edits.append((middle_start, middle_end, middle))
            last_inst = instances[i + n - 1]
            edits.append((last_inst.span[0], last_inst.span[1], replacement))
            if (composition_root_rewires and
                    not suppress_composition_tail_rewire):
                # The removed producer can also feed view/update operations
                # after the final contraction (for example, an insert_slice
                # destination). Rewire those uses through the function return.
                # Limit the edit to this function because textual SSA names
                # may be reused by a later function in the module.
                tail_start = last_inst.span[1]
                return_match = re.search(
                    r"\n[ \t]*return\b", text[tail_start:])
                tail_end = (tail_start + return_match.start()
                            if return_match else tail_start)
                tail = text[tail_start:tail_end]
                for old_root, new_root in composition_root_rewires:
                    tail = re.sub(
                        rf"(?<![\w]){re.escape(old_root)}(?![\w])",
                        new_root, tail)
                for old_root, new_root in tail_only_rewires:
                    tail = re.sub(
                        rf"(affine\.yield[^\n]*)(?<![\w])"
                        rf"{re.escape(old_root)}(?![\w])",
                        rf"\1{new_root}", tail)
                edits.append((tail_start, tail_end, tail))
        i += n

    if dry_run:
        return text, report

    # Apply edits back-to-front so spans remain valid.
    out_chars = list(text)
    for start, end, repl in sorted(edits, key=lambda e: -e[0]):
        out_chars[start:end] = list(repl)
    return "".join(out_chars), report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Path to MLIR file (debuferized linalg form).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report matches; don't emit rewritten MLIR.")
    ap.add_argument("--show-candidates", action="store_true",
                    help=("With --dry-run, list backend-capable kernel "
                          "definition candidates found at each linalg body."))
    ap.add_argument("--show-semantic-only", action="store_true",
                    help=("With --dry-run --show-candidates, also list "
                          "semantic-only debug matches that do not currently "
                          "have a backend route."))
    ap.add_argument("--with-roundtrip-markers", action="store_true",
                    help=("Embed the original linalg.generic span as a "
                          "// POLYGEIST-MATCH-BEGIN/-END comment block above "
                          "each emitted kernel.launch op so the rewrite is "
                          "reversible by kernel_launch_lower.py."))
    ap.add_argument("--max-launches", type=int,
                    help=("Emit at most this many launches, preserving later "
                          "matches as residual Linalg; useful for correctness "
                          "bisection."))
    ap.add_argument("--disable-pointwise-matching", action="store_true",
                    help=("Disable the generic cuDNN scalar-expression graph "
                          "fallback while preserving named library matches."))
    ap.add_argument("--disable-kernel", action="append", default=[],
                    help=("Do not emit the named kernel; may be repeated. "
                          "Useful for provenance audits and backend bisection."))
    ap.add_argument("--only-kernel", action="append", default=None,
                    help=("Emit only the named kernel family; may be repeated. "
                          "Intended for correctness bisection."))
    ap.add_argument("--show-structured-regions", action="store_true",
                    help=("Run loop-aware Egglog analysis and report safe "
                          "producer/consumer regions and fusion proofs."))
    ap.add_argument("--enable-structured-rewrite", action="store_true",
                    help=("Enable conservative executable rewrites proven by "
                          "the loop-aware Egglog analysis."))
    ap.add_argument("--stencil-backend", choices=("cudnn", "custen"),
                    default="cudnn",
                    help=("Select the external backend for compatible packed "
                          "2D weighted stencils (default: cudnn)."))
    args = ap.parse_args()

    text = Path(args.input).read_text()
    rewritten, report = rewrite_mlir(
        text,
        dry_run=args.dry_run,
        roundtrip_markers=args.with_roundtrip_markers,
        show_candidates=args.show_candidates,
        show_semantic_only=args.show_semantic_only,
        max_launches=args.max_launches,
        disable_pointwise_matching=args.disable_pointwise_matching,
        show_structured_regions=args.show_structured_regions,
        enable_structured_rewrite=args.enable_structured_rewrite,
        stencil_backend=args.stencil_backend,
        disabled_kernels=set(args.disable_kernel),
        only_kernels=(set(args.only_kernel)
                      if args.only_kernel is not None else None),
    )
    if args.dry_run:
        print(f"== match report for {args.input} ==", file=sys.stderr)
        for kind, idx, name in report:
            print(f"  {kind:<14} body#{idx}  {name}", file=sys.stderr)
        matched = sum(1 for k, _, _ in report if k == "match")
        candidates = sum(1 for k, _, _ in report if k == "kernel_candidate")
        semantic_debug = sum(1 for k, _, _ in report if k == "semantic_debug")
        structured_fusions = sum(
            1 for k, _, _ in report if k == "structured_fusion")
        structured_rejects = sum(
            1 for k, _, _ in report if k == "structured_reject")
        total = sum(
            1 for k, _, _ in report
            if k not in ("kernel_candidate", "semantic_debug",
                         "structured_fusion", "structured_reject",
                         "residual_idiom_candidate")
        )
        print(f"  total: {matched} matched / {total} bodies", file=sys.stderr)
        if candidates:
            print(f"  kernel candidates: {candidates}", file=sys.stderr)
        if semantic_debug:
            print(f"  semantic debug matches: {semantic_debug}", file=sys.stderr)
        if structured_fusions or structured_rejects:
            print(f"  structured regions: {structured_fusions} proved / "
                  f"{structured_rejects} rejected", file=sys.stderr)
    else:
        sys.stdout.write(rewritten)


if __name__ == "__main__":
    main()
