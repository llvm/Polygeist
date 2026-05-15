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
                  bindings: dict, captures_per_step: list[list[str]]) -> str:
    """Build a `kernel.launch` op line in MLIR text."""
    # Resolve scalar capture bindings to actual SSA values. The matcher
    # returned bindings keyed by template-cap names (e.g. "%alpha" →
    # ('Cap', '%arg3')); we just want the SSA value from the second.
    scalar_ssas: list[str] = []
    for tmpl_name, bound in bindings.items():
        # bound is a parsed AST tuple. Extract the original SSA name.
        if isinstance(bound, tuple) and len(bound) == 2 and bound[0] == "Cap":
            scalar_ssas.append(bound[1])
    # Order operands: tensor operands first (in source order), then scalars.
    all_operands = operands + scalar_ssas
    operand_str = ", ".join(all_operands)
    return (f"{indent}{result_ssa} = kernel.launch @{name}"
            f"({operand_str}) : ({', '.join('!any' for _ in all_operands)}) "
            f"-> {result_type}")


def rewrite_mlir(text: str, dry_run: bool = False) -> tuple[str, list[tuple]]:
    """Run the matcher on `text` and return (rewritten_text, match_report).

    match_report: list of (kernel_name_or_None, body_indices, launch_name).
    """
    consts = parse_constants(text)
    bodies = parse_generics(text, consts)
    instances = collect_generics_with_spans(text)
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
        for j in range(n):
            all_tensor_ins.extend(_extract_ssa_names(instances[i + j].ins_part))
        outs0 = _extract_ssa_names(instances[i].outs_part)
        operands = all_tensor_ins + outs0
        # The launch's result is the LAST generic's result SSA + type.
        last = instances[i + n - 1]
        replacement = render_launch(
            entry.name, last.result_ssa, last.result_type,
            operands, last.indent, binds, [],
        )
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
    args = ap.parse_args()

    text = Path(args.input).read_text()
    rewritten, report = rewrite_mlir(text, dry_run=args.dry_run)
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
