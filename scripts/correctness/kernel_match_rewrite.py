#!/home/arjaiswal/slacker/.venv/bin/python3
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
    r"\{\s*\^bb0\([^)]*\)\s*:.*?linalg\.yield\s+%[\w_]+\s*:[^}]*\}"
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
    return out


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
    import re
    m = re.match(r'(?:memref|tensor)<[^>]*?x(\w+)(?:,|>)', memref_or_tensor_ty)
    if not m:
        return None
    return m.group(1)


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
    return f"{cast_prefix}{indent}{result_ssa} = kernel.launch @{name}({operand_str}) : {sig} -> {result_type}"


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

    # Per-body form ("tensor" / "memref"), aligned with `instances`. The
    # form is determined by whether the linalg.generic has an SSA result —
    # tensor-form returns an SSA, memref-form is void with side effects.
    body_forms = ["tensor" if inst.result_ssa else "memref" for inst in instances]

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
        # The replacement covers the FULL span from the first generic's
        # start to the last generic's end.
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

        # Dtype-suffix dispatch for cuDNN conv2d. The encoder's Term language
        # is dtype-agnostic (arith.mulf matches any float type), so one
        # template fires for f64, f32, f16, bf16 bodies. We emit a
        # dtype-specific kernel.launch symbol so the canonical defn and the
        # lowering pass can pick the right cuDNN shim per element type.
        # The default (no suffix) is f64 for backward compat with the
        # existing kernel.defn @cudnnConvolution2D_9tap declaration.
        if entry.name in ("cudnnConvolution2D_9tap",
                          "cudnnConvolution2D_9tap_tensor"):
            elem = _sniff_elem_type(all_tensor_in_types[0]) if all_tensor_in_types else "f64"
            if elem and elem != "f64":
                emit_name = f"{entry.name}_{elem}"

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
        edits.append((start, end, replacement))
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
