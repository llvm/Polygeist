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
    _AFFINE_MAP_RE,
)


# Keep this in sync with lib/polygeist/Passes/LowerKernelLaunchToCuBLAS.cpp.
# The matcher may semantically recognize many more kernels than the ABI/runtime
# layer can execute today. Only emit kernel.launch for symbols that the lowering
# pass currently knows how to turn into a runtime call; leave the rest as
# residual Linalg so the normal MLIR lowering path preserves semantics.
ABI_LOWERABLE_KERNELS = {
    "cublasDgemm",
    "cublasDgemm_simple",
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
    "cublasSgemv",
    "cublasSgemv_T",
    "cublasDgemv_alpha",
    "cublasDaxpby",
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
    "customStencil3D7pt_f64_tensor",
    "customStencil3D7ptCoeff_f64_tensor",
    "customStencil3D7ptExtra_f64_tensor",
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
    "rmsnorm_f32",
    "rmsnorm_f32_tensor",
    "rmsnorm_unweighted_f32_tensor",
    "gelu_tanh_f32_tensor",
    "whisperExpShiftSum_f32_tensor",
    "cublasDdot",
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
    "cutensornetContraction2_f64_r4r5r4",
    "cutensornetContraction2_f64_r5r4r4",
    "cutensornetContraction2_f64_r5r5r4",
}


SEMANTIC_BACKEND_HINTS = {
    # The semantic node is lowered by a custom rewrite into this ABI symbol.
    "miniamr_weighted_27pt_tensor": "cudnnConvolution3D_ntap_tensor",
    # Candidate completion: not emitted yet, but this is the intended backend
    # route once the sparse filter materialization rule is implemented.
    "conv3d_sparse_3x3x3": "cudnnConvolution3D_ntap_tensor",
    "miniamr_average_7pt_tensor": "customStencil3D7pt_f64_tensor",
    "miniamr_weighted_7pt_tensor": "customStencil3D7ptCoeff_f64_tensor",
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
        if rank == 1:
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
        p = _parse_static_extract_slice_offset(text, name)
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

    out = _parse_static_extract_slice_offset(text, out_name)
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


def _render_custom_stencil3d7pt_launch(
    name: str,
    result_ssa: str,
    result_type: str,
    operands: list[str],
    operand_types: list[str],
    coeffs: list[float],
    indent: str,
) -> str:
    if len(coeffs) != 10:
        raise ValueError("custom stencil3d7pt launch expects 10 coefficients")
    cast_lines, operands, operand_types = _normalize_tensor_operands(
        operands, operand_types, indent
    )
    scalar_names = []
    scalar_lines = []
    for idx, value in enumerate(coeffs):
        ssa = _derived_ssa_name(result_ssa, f"stencil7_c{idx}")
        lit = repr(float(value))
        if "." not in lit and "e" not in lit and "E" not in lit:
            lit += ".0"
        scalar_lines.append(f"{indent}{ssa} = arith.constant {lit} : f64")
        scalar_names.append(ssa)
    all_operands = operands + scalar_names
    all_types = operand_types + ["f64"] * len(scalar_names)
    cast_prefix = "\n".join(cast_lines + scalar_lines)
    if cast_prefix:
        cast_prefix += "\n"
    operand_str = ", ".join(all_operands)
    sig = f"({', '.join(all_types)})"
    dyn_result_type = _dynamic_tensor_type(result_type)
    launch_result_ssa = result_ssa
    launch_result_type = result_type
    result_cast = ""
    if dyn_result_type is not None and dyn_result_type != result_type:
        launch_result_ssa = _derived_ssa_name(result_ssa, "tdyn")
        launch_result_type = dyn_result_type
        result_cast = (
            f"\n{indent}{result_ssa} = tensor.cast {launch_result_ssa} : "
            f"{dyn_result_type} to {result_type}"
        )
    return (
        f"{cast_prefix}{indent}{launch_result_ssa} = kernel.launch "
        f"@{name}({operand_str}) : {sig} -> {launch_result_type}"
        f"{result_cast}"
    )


def render_launch(name: str, result_ssa: str | None, result_type: str | None,
                  operands: list[str], indent: str,
                  bindings: dict, captures_per_step: list[list[str]],
                  operand_types: list[str] | None = None,
                  scalar_type_map: dict[str, str] | None = None,
                  inline_weights: list[list[str] | None] | None = None,
                  inline_weight_type: str = "f64",
                  body_constants: dict[str, float] | None = None,
                  result_count: int = 1) -> str:
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
                # This handles the polybench conv3d case where the same input
                # appears in multiple muls with different literal constants
                # (the _factor_redundant_muls normalisation in kernel_match.py
                # told the matcher this is a single conceptual weight).
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
        return f"{cast_prefix}{indent}kernel.launch @{name}({operand_str}) : {sig} -> ()"
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
        f"@{name}({operand_str}) : {sig} -> {launch_result_type}"
        f"{result_cast}"
    )


def _render_contraction_launch(
    name: str,
    result_ssa: str,
    result_type: str,
    operands: list[str],
    operand_types: list[str],
    indexing_maps: list[str],
    indent: str,
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
        target = _dynamic_tensor_type(operand_type)
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
    dynamic_result_type = _dynamic_tensor_type(result_type) or result_type
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


def rewrite_mlir(
    text: str,
    dry_run: bool = False,
    roundtrip_markers: bool = False,
    show_candidates: bool = False,
    show_semantic_only: bool = False,
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

    comps = composition_library()

    # Walk bodies front-to-back, greedy-match compositions.
    report: list[tuple] = []
    if dry_run and show_candidates:
        for cand_i in range(len(body_terms)):
            for cand in enumerate_semantic_candidates(
                bodies, body_terms, comps, start=cand_i, body_forms=body_forms
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
    i = 0
    while i < len(body_terms):
        if body_terms[i] is None:
            report.append(("encoder_fail", i, "?"))
            i += 1
            continue
        m = match_composition(bodies, body_terms, comps, start=i,
                              body_forms=body_forms)
        if m is None:
            entry = match_elementwise_semantic(
                bodies[i], body_terms[i], body_forms[i]
            )
            if entry is None:
                report.append(("no_match", i, "?"))
                i += 1
                continue
            binds = {}
            n = 1
        else:
            entry, _, binds = m
            n = len(entry.steps)
        report.append(("match", list(range(i, i + n)), entry.name))

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
            # `tensor<?x?xf64>` → 2 ; `tensor<?xf64>` → 1 ; etc.
            inside = t[t.find("<") + 1 : t.rfind(">")]
            shape = inside.rsplit("x", 1)[0]
            return shape.count("x") + 1 if shape else 0
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
        custom_edit_span: tuple[int, int] | None = None

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

        if entry.name == "cublasDcopy" and n == 1:
            in0_ty = all_tensor_in_types[0] if all_tensor_in_types else ""
            # rank-0 memref: starts with `memref<` and the chunk before the
            # outermost `,` or `>` contains no `x` (i.e. just the elem type).
            if in0_ty.startswith("memref<"):
                inside = in0_ty[len("memref<"):].split(",", 1)[0]
                if "x" not in inside:
                    emit_name = "broadcast_scalar_to_vec"
        # Tensor-form twin of the same dispatch (multi-root debufferize).
        if entry.name == "cublasDcopy_tensor" and n == 1:
            if not _tensor_copy_layout_is_legal():
                report.append(("copy_layout_reject", i, entry.name))
                i += 1
                continue
            in0_ty = all_tensor_in_types[0] if all_tensor_in_types else ""
            if in0_ty.startswith("tensor<"):
                inside = in0_ty[len("tensor<"):].split(",", 1)[0]
                if "x" not in inside:
                    emit_name = "broadcast_scalar_to_vec_tensor"
            elem = _sniff_elem_type(all_tensor_in_types[0]) if all_tensor_in_types else None
            ranks = [_tensor_rank(t) for t in operand_types[:2]]
            if elem == "f32" and len(ranks) == 2 and ranks[0] == ranks[1]:
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

        if entry.name == "cublasGemmFor1x1Conv":
            # This algebraic template is broader than a 1x1 convolution: MFEM
            # sum-factorization stages also have four parallel iterators and
            # one reduction. Route FP64 tensor contractions through a
            # layout-aware cuTensorNet ABI, but only after proving that the
            # affine maps describe a real contraction. In particular, the
            # reduction dimension must occur in both inputs and not in the
            # output. This rejects the historical false positives whose
            # "reduction" iterator actually indexes the output.
            contraction_inst = instances[i + n - 1]
            contraction_body = bodies[i + n - 1]
            contraction_ins = _extract_ssa_names(
                contraction_inst.ins_part
            )
            contraction_in_types = _extract_ssa_types(
                contraction_inst.ins_part
            )
            contraction_outs = _extract_ssa_names(
                instances[i].outs_part
            )
            contraction_out_types = _extract_ssa_types(
                instances[i].outs_part
            )
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
            elem_types = [
                _sniff_elem_type(ty)
                for ty in contraction_in_types + contraction_out_types
            ]
            ranks = [
                _tensor_rank(ty)
                for ty in contraction_in_types + contraction_out_types
            ]
            legal_maps = (
                len(map_dims) == 3
                and all(dims is not None for dims in map_dims)
                and 4 in map_dims[0] and 4 in map_dims[1]
                and 4 not in map_dims[2]
                and sorted(map_dims[2]) == [0, 1, 2, 3]
                and all(len(dims) == len(set(dims)) for dims in map_dims)
            )
            legal_types = (
                len(contraction_ins) == 2
                and len(contraction_outs) == 1
                and elem_types == ["f64", "f64", "f64"]
                and ranks in ([4, 5, 4], [5, 4, 4], [5, 5, 4])
                and last.result_type is not None
                and _tensor_rank(last.result_type) == 4
            )
            if not legal_maps or not legal_types:
                report.append(("contraction_abi_reject", i, entry.name))
                i += n
                continue

            emit_name = {
                (4, 5, 4): "cutensornetContraction2_f64_r4r5r4",
                (5, 4, 4): "cutensornetContraction2_f64_r5r4r4",
                (5, 5, 4): "cutensornetContraction2_f64_r5r5r4",
            }[tuple(ranks)]
            # Preserve source operand order: contraction_maps correspond
            # positionally to these operands, so the generic rank-based
            # commutative reordering is intentionally bypassed.
            operands = contraction_ins + contraction_outs
            operand_types = contraction_in_types + contraction_out_types
            custom_launch_line = _render_contraction_launch(
                emit_name, last.result_ssa, last.result_type,
                operands, operand_types, maps, last.indent,
            )

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

        if entry.name in ("rmsnorm_f32", "rmsnorm_unweighted_f32",
                          "rmsnorm_scaled_unweighted_f32"):
            # RMSNorm is a two-stage composition:
            #   step0: ss = sum(x[i] * x[i])
            #   step1: out[i] = weight[i] * scale * x[i]     (weighted)
            #          out[i] = scale * x[i]                 (unweighted)
            #          out[i] = gain * scale * x[i]          (scalar gain)
            # The generic operand collection above only keeps the first
            # generic's outs (the scalar ss buffer), which is not enough for
            # ABI lowering. Emit the semantic operands directly and let the
            # runtime recompute the reduction/scale in one call.
            #
            # The scalar-gain variant is recognized by the matcher factory,
            # but we do not lower it until the runtime ABI has an explicit gain
            # operand. Leaving it as residual linalg is safer than pretending
            # the weighted ABI can represent it.
            if entry.name == "rmsnorm_scaled_unweighted_f32":
                report.append(("unsupported_abi_reject", i, entry.name))
                i += n
                continue
            forms = body_forms[i : i + n]
            x_names = _extract_ssa_names(instances[i].ins_part)
            x_types = _extract_ssa_types(instances[i].ins_part)
            scale_ins = _extract_ssa_names(instances[i + 1].ins_part)
            scale_in_types = _extract_ssa_types(instances[i + 1].ins_part)
            out_names = _extract_ssa_names(instances[i + 1].outs_part)
            out_types = _extract_ssa_types(instances[i + 1].outs_part)
            min_scale_ins = 2 if entry.name == "rmsnorm_f32" else 1
            if (len(x_names) < 1 or len(scale_ins) < min_scale_ins
                    or len(out_names) < 1 or any(f != forms[0] for f in forms)):
                report.append(("rmsnorm_reject", i, entry.name))
                i += 1
                continue
            if entry.name == "rmsnorm_f32":
                operands = [x_names[0], scale_ins[0], out_names[0]]
                operand_types = [x_types[0], scale_in_types[0], out_types[0]]
            else:
                operands = [x_names[0], out_names[0]]
                operand_types = [x_types[0], out_types[0]]
            binds = {}
            if forms[0] == "tensor":
                # Tensor RMSNorm's scalar scale chain depends on the first
                # generic result. Since the shim recomputes the full RMSNorm,
                # replace the whole span, including that scalar chain, with
                # one result-producing tensor launch.
                emit_name = (
                    "rmsnorm_f32_tensor"
                    if entry.name == "rmsnorm_f32"
                    else "rmsnorm_unweighted_f32_tensor"
                )
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
            operands = [out_names[0]]
            operand_types = [out_types[0]]
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

        if entry.name in ("miniamr_average_7pt_tensor",
                          "miniamr_weighted_7pt_tensor"):
            inst = instances[i]
            in_names = _extract_ssa_names(inst.ins_part)
            in_types = _extract_ssa_types(inst.ins_part)
            out_names = _extract_ssa_names(inst.outs_part)
            out_types = _extract_ssa_types(inst.outs_part)
            expected_inputs = 7 if entry.name == "miniamr_average_7pt_tensor" else 8
            elem = _sniff_elem_type(out_types[0]) if out_types else None
            ranks = [_tensor_rank(t) for t in in_types + out_types]
            if (inst.result_ssa is None or inst.result_type is None
                    or len(in_names) != expected_inputs or len(out_names) != 1
                    or elem != "f64"
                    or len(ranks) != expected_inputs + 1
                    or any(_sniff_elem_type(t) != "f64"
                           for t in in_types + out_types)
                    or any(rank != 3 for rank in ranks)):
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += 1
                continue

            if entry.name == "miniamr_average_7pt_tensor":
                emit_name = "customStencil3D7pt_f64_tensor"
                launch_operands = in_names + out_names
                launch_types = in_types + out_types
                coeffs = [0.0, 0.0, 0.0] + [1.0 / 7.0] * 7
            else:
                emit_name = "customStencil3D7ptCoeff_f64_tensor"
                # Preserve the matched body order: first seven tensors are the
                # center/six-neighbor taps, input 7 is the cell coefficient.
                launch_operands = in_names[:7] + [in_names[7]] + out_names
                launch_types = in_types[:7] + [in_types[7]] + out_types
                coeffs = [1.0, 0.0, 0.0, -6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            binds = {}
            custom_launch_line = _render_custom_stencil3d7pt_launch(
                emit_name,
                inst.result_ssa,
                inst.result_type,
                launch_operands,
                launch_types,
                coeffs,
                inst.indent,
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

        if entry.name in ("cudnnConvolution2D_ntap",
                          "cudnnConvolution2D_ntap_tensor"):
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
                          "cublasDgemm_alpha_only"):
            gemm_inst = instances[i + n - 1]  # last (contraction) generic
            gemm_ins = _extract_ssa_names(gemm_inst.ins_part)
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
            if (entry.name == "cublasDgemm_simple" and elem == "f32" and
                    operand_ranks == [3, 3, 3]):
                # Darknet im2col+GEMM reaches linalg as a rank-3 broadcasted
                # view: logical (N, K, M) iteration, but the underlying buffers
                # are the usual 2D row-major A[M,K], B[K,N], C[M,N]. Emit a
                # dedicated symbol so ABI lowering can unwrap the submaps and
                # call cuBLAS SGEMM.
                emit_name = "cublasSgemm_broadcast3d_simple"
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
        if entry.name == "memset_zero_2D":
            elem = _sniff_elem_type(outs0_types[0]) if outs0_types else None
            if elem == "f32":
                emit_name = "memset_zero_2D_f32"
        if entry.name == "cublasSgemm_broadcast3d_memref":
            elem = _sniff_elem_type(operand_types[0]) if operand_types else None
            operand_ranks = [_tensor_rank(t) for t in operand_types[:3]]
            if elem != "f32" or operand_ranks != [3, 3, 3]:
                report.append(("rank_or_dtype_reject", i, entry.name))
                i += 1
                continue
        if entry.name == "cublasDgemv" and n == 1:
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
                emit_name = "cublasSgemv_T" if transposed else "cublasSgemv"
            else:
                emit_name = "cublasDgemv_T" if transposed else "cublasDgemv"

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
        if roundtrip_markers:
            # last.indent has a leading newline ("\n    ") because the parser
            # captures the line break before the op. Use only the spaces.
            indent_spaces = last.indent.lstrip("\n").rstrip("\n")
            # The original span starts mid-line at "\n    %X = linalg.generic..."
            # so we strip the leading newline from the captured block and
            # restore it ourselves once, before the BEGIN marker.
            original_block = text[start:end]
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
                edits.append((inst_j.span[0], inst_j.span[1], ""))
            last_inst = instances[i + n - 1]
            edits.append((last_inst.span[0], last_inst.span[1], replacement))
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
    args = ap.parse_args()

    text = Path(args.input).read_text()
    rewritten, report = rewrite_mlir(
        text,
        dry_run=args.dry_run,
        roundtrip_markers=args.with_roundtrip_markers,
        show_candidates=args.show_candidates,
        show_semantic_only=args.show_semantic_only,
    )
    if args.dry_run:
        print(f"== match report for {args.input} ==", file=sys.stderr)
        for kind, idx, name in report:
            print(f"  {kind:<14} body#{idx}  {name}", file=sys.stderr)
        matched = sum(1 for k, _, _ in report if k == "match")
        candidates = sum(1 for k, _, _ in report if k == "kernel_candidate")
        semantic_debug = sum(1 for k, _, _ in report if k == "semantic_debug")
        total = sum(
            1 for k, _, _ in report
            if k not in ("kernel_candidate", "semantic_debug")
        )
        print(f"  total: {matched} matched / {total} bodies", file=sys.stderr)
        if candidates:
            print(f"  kernel candidates: {candidates}", file=sys.stderr)
        if semantic_debug:
            print(f"  semantic debug matches: {semantic_debug}", file=sys.stderr)
    else:
        sys.stdout.write(rewritten)


if __name__ == "__main__":
    main()
