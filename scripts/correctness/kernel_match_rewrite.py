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
# we can substitute it with a `kernel.launch`.
_GENERIC_BLOCK_RE = re.compile(
    r"(\s*)(%[\w_]+)\s*=\s*linalg\.generic\s*\{[^}]*\}\s*"
    r"(?:ins\(([^)]*)\)\s*)?"
    r"outs\(([^)]*)\)\s*"
    r"\{\s*\^bb0\([^)]*\)\s*:.*?linalg\.yield\s+%[\w_]+\s*:[^}]*\}\s*"
    r"->\s*([^\n]+)",
    re.DOTALL,
)


@dataclass
class LinalgInstance:
    """A single linalg.generic op extracted from the MLIR text."""
    result_ssa: str       # %12 etc.
    ins_part: str         # "%10, %11 : tensor<?x...>, tensor<...>"
    outs_part: str        # "%9 : tensor<...>"
    result_type: str      # the type after `->`
    span: tuple[int, int] # offset range in the source text
    indent: str           # leading whitespace before the SSA def


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
    # arith.constant lines: "%X = arith.constant ... : f64"
    for cm in re.finditer(r'(%[\w]+)\s*=\s*arith\.constant\s+\S+\s*:\s*(\S+)', text):
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
            result_type=rty.strip(),
            span=m.span(),
            indent=indent,
        ))
    return out


def render_launch(name: str, result_ssa: str, result_type: str,
                  operands: list[str], indent: str,
                  bindings: dict, captures_per_step: list[list[str]],
                  operand_types: list[str] | None = None,
                  scalar_type_map: dict[str, str] | None = None) -> str:
    """Build a `kernel.launch` op line in MLIR text.

    operand_types : explicit types for the tensor `operands` list (same order).
    scalar_type_map : SSA→type lookup for Cap-bound scalars.
    If types are unknown we fall back to `!any` which is unparseable — that's
    intentional, so callers see the breakage.
    """
    scalar_ssas: list[str] = []
    for tmpl_name, bound in bindings.items():
        if isinstance(bound, tuple) and len(bound) == 2 and bound[0] == "Cap":
            scalar_ssas.append(bound[1])
    all_operands = operands + scalar_ssas
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

    return (f"{indent}{result_ssa} = kernel.launch @{name}"
            f"({operand_str}) : ({', '.join(sig_types)}) "
            f"-> {result_type}")


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
        m = match_composition(bodies, body_terms, comps, start=i)
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
        launch_line = render_launch(
            entry.name, last.result_ssa, last.result_type,
            operands, last.indent, binds, [],
            operand_types=operand_types,
            scalar_type_map=scalar_types,
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
