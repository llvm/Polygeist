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
import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from kernel_match import (
    parse_constants, parse_generics, encode_body,
    match_composition, composition_library,
    _AFFINE_MAP_RE,
)


# Match each linalg.generic at the IR level, capturing the full block so
# we can substitute it with a `kernel.launch`. Handles BOTH:
#   - tensor form: `%X = linalg.generic {...} ins(...) outs(...) {body} -> T`
#   - memref form: `linalg.generic {...} ins(...) outs(...) {body}`
# (no SSA prefix, no return type; the op is void and mutates `outs` in place).
# The leading SSA `%X =` and the trailing `-> type` are both optional.
_GENERIC_BLOCK_RE = re.compile(
    r"(\s*)(?:(%[\w_]+)\s*=\s*)?linalg\.generic\s*\{[^}]*\}\s*"
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
        indent, result_ssa, ins, outs, rty = m.groups()
        out.append(LinalgInstance(
            result_ssa=result_ssa,
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


def render_launch(name: str, result_ssa: str | None, result_type: str | None,
                  operands: list[str], indent: str,
                  bindings: dict, captures_per_step: list[list[str]],
                  operand_types: list[str] | None = None,
                  scalar_type_map: dict[str, str] | None = None,
                  inline_weights: list[list[str] | None] | None = None,
                  inline_weight_type: str = "f64",
                  body_constants: dict[str, float] | None = None) -> str:
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
    result_cast = ""
    dyn_result_type = _dynamic_tensor_type(result_type)
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


def rewrite_mlir(
    text: str,
    dry_run: bool = False,
    roundtrip_markers: bool = False,
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
            report.append(("no_match", i, "?"))
            i += 1
            continue
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
                ins_part=last.ins_part,
                outs_part=last.outs_part,
                result_type=None,
                span=last.span,
                indent=last.indent,
            )

        if entry.name == "rmsnorm_f32":
            # RMSNorm is a two-stage composition:
            #   step0: ss = sum(x[i] * x[i])
            #   step1: out[i] = weight[i] * scale * x[i]
            # The generic operand collection above only keeps the first
            # generic's outs (the scalar ss buffer), which is not enough for
            # ABI lowering. Emit the semantic operands directly and let the
            # runtime recompute the reduction/scale in one call.
            forms = body_forms[i : i + n]
            x_names = _extract_ssa_names(instances[i].ins_part)
            x_types = _extract_ssa_types(instances[i].ins_part)
            scale_ins = _extract_ssa_names(instances[i + 1].ins_part)
            scale_in_types = _extract_ssa_types(instances[i + 1].ins_part)
            out_names = _extract_ssa_names(instances[i + 1].outs_part)
            out_types = _extract_ssa_types(instances[i + 1].outs_part)
            if (len(x_names) < 1 or len(scale_ins) < 2 or len(out_names) < 1
                    or any(f != forms[0] for f in forms)):
                report.append(("rmsnorm_reject", i, entry.name))
                i += 1
                continue
            operands = [x_names[0], scale_ins[0], out_names[0]]
            operand_types = [x_types[0], scale_in_types[0], out_types[0]]
            binds = {}
            if forms[0] == "tensor":
                # Tensor RMSNorm's scalar scale chain depends on the first
                # generic result. Since the shim recomputes the full RMSNorm,
                # replace the whole span, including that scalar chain, with
                # one result-producing tensor launch.
                emit_name = "rmsnorm_f32_tensor"
                replace_full_span = True
            else:
                last = LinalgInstance(
                    result_ssa=None,
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

        if entry.name == "elemwise_div_scalar":
            # This template is useful for algebraic recognition, but the ABI
            # lowering path does not have a runtime shim for it. Keep the
            # linalg.generic in place so downstream MLIR lowering handles it
            # as ordinary residual tensor code.
            report.append(("unsupported_abi_reject", i, entry.name))
            i += 1
            continue

        if entry.name in ("cudnnConvolution2D_9tap",
                          "cudnnConvolution2D_9tap_tensor"):
            elem = _sniff_elem_type(all_tensor_in_types[0]) if all_tensor_in_types else "f64"
            if elem and elem != "f64":
                emit_name = f"{entry.name}_{elem}"

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
                if base0 == base1:
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
            edits.append((start, end, replacement))
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
    )
    if args.dry_run:
        print(f"== match report for {args.input} ==", file=sys.stderr)
        for kind, idx, name in report:
            print(f"  {kind:<14} body#{idx}  {name}", file=sys.stderr)
        matched = sum(1 for k, _, _ in report if k == "match")
        total = len(report)
        print(f"  total: {matched} matched / {total} bodies", file=sys.stderr)
    else:
        sys.stdout.write(rewritten)


if __name__ == "__main__":
    main()
