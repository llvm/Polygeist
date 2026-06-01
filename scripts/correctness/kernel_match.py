#!/usr/bin/env python3
"""linalg.generic body matcher using egglog.

This is an iterative prototype of the "match raised linalg to a kernel
library" idea, in three layers:

  1. Regex-based parser for linalg.generic bodies (good enough for the
     debuferized PolyBench output — every body is ~6 lines of arith + yield).
  2. Encoder: linalg-body -> egglog Expr.
  3. Matcher: saturate with algebra rules, then check equivalence between
     a user body and each library pattern.

The library is built from the bodies of already-raised+debuferized PolyBench
kernels. Bodies that are *structurally equivalent under algebra* collapse to
the same library entry.
"""
from __future__ import annotations
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from egglog import EGraph, Expr, StringLike, f64, f64Like, i64Like, rewrite, ruleset, vars_


# ---------------------------------------------------------------------------
# The term language for linalg bodies.
# ---------------------------------------------------------------------------

class Term(Expr):
    """A scalar expression node inside a linalg.generic body.

    Leaves:
      - In(i)        : the i-th input operand's block arg.
      - Out(i)       : the i-th output's block arg (initial value).
      - Cap(name)    : a captured outer scalar (e.g., `%arg3` = alpha).
      - Lit(value)   : a literal constant scalar.

    Internals — one per arith op we want to recognize. Add more as kernels
    surface them.
    """
    def __init__(self, name: StringLike) -> None: ...
    @classmethod
    def In(cls, i: i64Like) -> Term: ...
    @classmethod
    def Out(cls, i: i64Like) -> Term: ...
    @classmethod
    def Cap(cls, name: StringLike) -> Term: ...
    @classmethod
    def Lit(cls, value: f64Like) -> Term: ...

    def __add__(self, other: Term) -> Term: ...
    def __mul__(self, other: Term) -> Term: ...
    def __sub__(self, other: Term) -> Term: ...
    def __truediv__(self, other: Term) -> Term: ...

    @classmethod
    def Sqrt(cls, a: Term) -> Term: ...
    @classmethod
    def Abs(cls, a: Term) -> Term: ...
    @classmethod
    def Exp(cls, a: Term) -> Term: ...
    @classmethod
    def Select(cls, pred: Term, t: Term, f: Term) -> Term: ...
    @classmethod
    def Cmp(cls, kind: StringLike, a: Term, b: Term) -> Term: ...


# ---------------------------------------------------------------------------
# Algebra rules (cosmetic variations).
# ---------------------------------------------------------------------------

a, b, c, d = vars_("a b c d", Term)


def algebra_rules():
    one = Term.Lit(1.0)
    zero = Term.Lit(0.0)
    # Numeric literal variables — required for the factoring + folding rules
    # below, where the RHS computes c1+c2 / c1*c2 via egglog's built-in f64
    # arithmetic on the captured constants. `vars_` returns a generator, so
    # single-name calls need tuple-unpack syntax.
    (x,) = vars_("x", Term)
    c1, c2 = vars_("c1 c2", f64)
    return ruleset(
        # Commutativity
        rewrite(a + b).to(b + a),
        rewrite(a * b).to(b * a),
        # Associativity
        rewrite(a + (b + c)).to((a + b) + c),
        rewrite((a + b) + c).to(a + (b + c)),
        rewrite(a * (b * c)).to((a * b) * c),
        rewrite((a * b) * c).to(a * (b * c)),
        # Distributivity (sometimes useful for kernel matching)
        rewrite(a * (b + c)).to((a * b) + (a * c)),
        rewrite((a + b) * c).to((a * c) + (b * c)),
        # Identity laws
        rewrite(a * one).to(a),
        rewrite(one * a).to(a),
        rewrite(a + zero).to(a),
        rewrite(zero + a).to(a),
        # Annihilator (mul by zero) — useful for trmm-style masks where
        # the kernel computes `mask * value + (1 - mask) * orig`.
        rewrite(a * zero).to(zero),
        rewrite(zero * a).to(zero),
        # Multi-coefficient factoring + literal folding. The first rule
        # collapses `c1*x + c2*x` into `(c1+c2)*x`; the second/third fold
        # literal arithmetic at the Term level. Together with commutativity
        # and associativity (above), they handle the polybench conv3d
        # "redundant mul" body where some inputs are multiplied by
        # multiple literal constants and summed.
        rewrite(Term.Lit(c1) * x + Term.Lit(c2) * x).to(Term.Lit(c1 + c2) * x),
        rewrite(Term.Lit(c1) + Term.Lit(c2)).to(Term.Lit(c1 + c2)),
        rewrite(Term.Lit(c1) * Term.Lit(c2)).to(Term.Lit(c1 * c2)),
    )


# ---------------------------------------------------------------------------
# Indexing-map canonicalization.
# ---------------------------------------------------------------------------

# Match affine_map<(d0, d1, ...) -> (...)> — capture the dim list and the
# result list separately.
_AFFINE_MAP_RE = re.compile(
    r"affine_map<\(([^)]*)\)\s*->\s*\(([^)]*)\)>"
)


def _rename_in_map(map_str: str, rename: dict[str, str]) -> str:
    """Apply a dim-name renaming to an affine_map's *result* expressions
    (and update the dim list to use the canonical names)."""
    m = _AFFINE_MAP_RE.match(map_str)
    if not m:
        return map_str
    dim_list, results = m.group(1), m.group(2)
    # Substitute each d<k> name with its canonical name. Do longest-first
    # to avoid d1 matching inside d10.
    keys = sorted(rename, key=lambda s: -len(s))
    new_results = results
    for k in keys:
        new_results = re.sub(rf"\b{k}\b", f"__TMP_{rename[k]}__", new_results)
    # Strip the __TMP_..._ wrapping.
    new_results = re.sub(r"__TMP_([^_]+)__", r"\1", new_results)
    # Build canonical dim list as d0, d1, ... up to max canonical index.
    used = sorted(set(rename.values()), key=lambda s: int(s[1:]))
    new_dim_list = ", ".join(used) if used else dim_list
    return f"affine_map<({new_dim_list}) -> ({new_results})>"


def canonicalize_maps_and_iters(
    maps: list[str], iters: list[str]
) -> tuple[list[str], list[str]]:
    """Canonicalize iter dim names by (a) iterator role, then (b) first-
    appearance order within each role.

    Order: all parallel dims first, then all reduction dims. Within each
    group, ordered by where they first appear across the map results.

    This makes two linalg.generic shapes that differ only by iter-dim
    naming converge to the same canonical form — *including* their
    iter_types attribute, which is permuted to match the new dim order.
    """
    if not maps or not iters:
        return maps, iters

    # First-appearance order across all maps' result expressions.
    first_seen: list[str] = []
    for map_str in maps:
        m = _AFFINE_MAP_RE.match(map_str)
        if not m:
            continue
        for tok in re.findall(r"\bd\d+\b", m.group(2)):
            if tok not in first_seen:
                first_seen.append(tok)
    if not first_seen:
        return maps, iters

    # Some dims might be in iters but not in any result expression
    # (broadcast-only iter dims). Include them too, after the seen ones.
    for i in range(len(iters)):
        name = f"d{i}"
        if name not in first_seen:
            first_seen.append(name)

    # Group by iterator role. We require every "seen" name to have an
    # iter_types entry; gracefully fall back if not.
    def role_of(old_name: str) -> str:
        idx = int(old_name[1:])
        if 0 <= idx < len(iters):
            return iters[idx]
        return "parallel"  # fallback

    parallel = [n for n in first_seen if role_of(n) == "parallel"]
    reduction = [n for n in first_seen if role_of(n) == "reduction"]
    other = [n for n in first_seen if n not in parallel and n not in reduction]
    ordered = parallel + reduction + other

    rename = {old: f"d{i}" for i, old in enumerate(ordered)}
    canon_maps = [_rename_in_map(m, rename) for m in maps]
    canon_iters = ["parallel"] * len(parallel) + \
                  ["reduction"] * len(reduction) + \
                  [role_of(n) for n in other]
    return canon_maps, canon_iters


# ---------------------------------------------------------------------------
# Parser: extract linalg.generic bodies from MLIR text.
# ---------------------------------------------------------------------------

@dataclass
class GenericBody:
    ins_arg_names: list[str]      # like ['%in', '%in_0', ...]
    outs_arg_names: list[str]     # like ['%out']
    body_lines: list[str]
    # Canonical yield list (one entry per output). Single-yield bodies have
    # len == 1; multi-yield bodies (e.g. softmax's fused exp+sum) have one
    # entry per `outs(...)` operand. Use `body.yield_value` (singular) for
    # back-compat single-yield reads — returns the first yield.
    yield_values: list[str]
    captures: list[str]           # outer SSA values referenced in body
    indexing_maps: list[str]      # raw text of each map
    iterator_types: list[str]
    constants: dict[str, float]   # captured SSA name -> Python float value
    # For each block input arg, the SSA name of the constant it's multiplied
    # with in the body — populated only if the input appears in exactly one
    # `arith.mulf %in, %cst : ...` (or `arith.mulf %cst, %in : ...`). Used by
    # render_launch to surface body-internal weight constants as launch
    # operands so the lowering pass can pass them to a generic runtime shim
    # (instead of the shim having to hardcode them). None for ins that don't
    # match the pattern. Aligned by index with ins_arg_names.
    # Each entry is either None (no constant paired with this input) or a
    # list of all constant SSAs that pair with the input. Multi-element
    # lists indicate the polybench-conv3d-style "redundant mul" pattern
    # where the same input is multiplied by several literal constants
    # and summed — the rewriter materialises a new arith.constant with
    # the summed value for the launch operand.
    inline_weights_per_in: list[list[str] | None] = None  # type: ignore[assignment]

    @property
    def yield_value(self) -> str:
        """Back-compat alias for callers written before multi-yield support
        — returns the first yield's SSA name. New code should iterate
        `yield_values` directly."""
        return self.yield_values[0] if self.yield_values else ""


_GEN_RE = re.compile(
    r"linalg\.generic\s*\{[^}]*indexing_maps\s*=\s*\[([^\]]*)\][^}]*"
    r"iterator_types\s*=\s*\[([^\]]*)\][^}]*\}[^\^]*?"
    # Yield captures one OR MORE comma-separated SSA names. Multi-yield
    # bodies (e.g. softmax's fused exp+sum) write to multiple outs in one
    # op. Single-yield bodies still match unchanged — the (?:...)*
    # group is zero-or-more. The capture is the full operand list as a
    # single string; parse_generics splits on commas to produce the
    # GenericBody.yield_values list.
    r"\^bb0\(([^)]*)\)\s*:\s*(.*?)\s*"
    r"linalg\.yield\s+(%[\w_]+(?:\s*,\s*%[\w_]+)*)\s*:",
    re.DOTALL,
)


# Recognize `%name = arith.constant <value> : <type>` at module/function scope.
# SSA names allow `-` in the body (e.g. cgeist emits `%c-8_i32` for negative
# int constants). Use a char class that includes `-` so we don't miss them.
_CONST_RE = re.compile(
    r"(%[\w_\-]+)\s*=\s*arith\.constant\s+([^\s:]+)\s*:\s*\S+"
)


def parse_constants(mlir_text: str) -> dict[str, float]:
    """Build a map from SSA name → constant literal value as a Python float.

    Floats here serve two purposes: (a) literal identity-rule matching in
    the algebra ruleset (e.g. `a*1.0 → a`), and (b) the new factoring +
    folding rules that compute on f64 constants. Both require the value
    to live in egglog's f64 sort, so we store it as a Python float here
    and let egglog auto-promote at Lit construction time.

    Integer constants (e.g. `arith.constant 5 : i32`) are coerced to
    float — this is sound because the encoder collapses int/float arith
    into the same Term operators, so int-typed constants live in the same
    Term-level numeric domain as float ones for matching purposes.

    Examples:
      `%cst = arith.constant 0.000000e+00 : f64`   →  {"%cst": 0.0}
      `%cst_0 = arith.constant 1.000000e+00 : f64` →  {"%cst_0": 1.0}
      `%c1 = arith.constant 1 : index`             →  {"%c1": 1.0}
      `%c-8_i32 = arith.constant -8 : i32`         →  {"%c-8_i32": -8.0}
    """
    out: dict[str, float] = {}
    for m in _CONST_RE.finditer(mlir_text):
        name, value = m.group(1), m.group(2)
        try:
            out[name] = float(value)
        except ValueError:
            # Non-numeric (e.g. an undef). Skip.
            pass
    return out


_MAP_ALIAS_RE = re.compile(
    # affine_map text contains `->` which has a `>`, so [^>] is wrong here.
    # Match the literal form `affine_map<(...) -> (...)>`.
    r"^\s*(#map\w*)\s*=\s*"
    r"(affine_map<\([^)]*\)\s*->\s*\([^)]*\)>)",
    re.MULTILINE
)


def _resolve_map_aliases(mlir_text: str) -> str:
    """Inline any `#mapN = affine_map<...>` top-level aliases by substituting
    each `#mapN` reference with the corresponding `affine_map<...>` literal.
    Required because parse_generics' regex only sees inline `affine_map<...>`
    text — kernels lifted via the standard pipeline carry aliased map refs,
    so without this the indexing_maps field comes back empty."""
    aliases = {name: literal for name, literal
               in _MAP_ALIAS_RE.findall(mlir_text)}
    if not aliases:
        return mlir_text
    # Sort by descending name length so #map10 substitutes before #map1.
    # No `\b` left boundary because `#` is not a word char — Python's `\b`
    # would refuse to match before it; rely on length-descending order +
    # negative lookahead on the right to disambiguate #map1 from #map10.
    for name in sorted(aliases, key=len, reverse=True):
        mlir_text = re.sub(re.escape(name) + r"(?!\w)",
                            aliases[name], mlir_text)
    return mlir_text


def parse_generics(mlir_text: str,
                   constants: dict[str, float] | None = None) -> list[GenericBody]:
    """Extract every linalg.generic with its body."""
    if constants is None:
        constants = parse_constants(mlir_text)
    mlir_text = _resolve_map_aliases(mlir_text)
    results = []
    for m in _GEN_RE.finditer(mlir_text):
        maps_str, iters_str, args_str, body_str, yield_operands_str = m.groups()
        # Split the yield's operand list on commas (multi-yield bodies have
        # multiple SSAs separated by commas). The regex preserves whitespace
        # around commas, so strip per-token.
        yield_names = [s.strip() for s in yield_operands_str.split(",") if s.strip()]
        # Back-compat for the rest of the local scope: yield_name refers to
        # the FIRST yield. Most local logic (capture detection, etc.) was
        # written assuming a single yield value — keeping it correct for
        # the single-yield case AND for the first slot of multi-yield bodies.
        yield_name = yield_names[0] if yield_names else ""

        # Parse args like "%in: f64, %in_0: f64, %out: f64"
        ins, outs = [], []
        for piece in args_str.split(","):
            piece = piece.strip()
            if not piece:
                continue
            name = piece.split(":")[0].strip()
            (outs if name.startswith("%out") else ins).append(name)

        # Tokenize indexing maps and iterator types as raw substrings.
        # Don't use `affine_map<[^>]*>` — the `->` inside contains a `>`.
        maps = [s.strip() for s in
                re.findall(r"affine_map<\([^)]*\)\s*->\s*\([^)]*\)>", maps_str)]
        iters = [s.strip().strip('"') for s in iters_str.split(",")]
        # Canonicalize: rename iter dims by their first-appearance order
        # across all maps, and permute iter_types to match.
        maps, iters = canonicalize_maps_and_iters(maps, iters)

        # Crude SSA-line extraction: each line in body is an arith op.
        body_lines = [
            ln.strip() for ln in body_str.split("\n")
            if ln.strip() and not ln.strip().startswith("//")
        ]

        # Find captures (SSA values that aren't block args and aren't defined locally).
        local_defs = set()
        captures: list[str] = []
        for ln in body_lines:
            assigned = re.match(r"(%[\w_]+)\s*=", ln)
            if assigned:
                local_defs.add(assigned.group(1))
        for ln in body_lines:
            # Find all %xxx references on the rhs.
            for tok in re.findall(r"%[\w_]+", ln):
                if (tok not in local_defs and tok not in ins and tok not in outs
                        and tok not in captures):
                    captures.append(tok)
        # Also catch yield-only captures — for every yield value, if it
        # references something defined outside the body (not a block arg,
        # not produced by any op in the body), promote it to a capture.
        for yn in yield_names:
            if (yn not in local_defs and yn not in ins
                    and yn not in outs and yn not in captures):
                captures.append(yn)

        # Build the inline-weights side-table: for each block input arg
        # %in_k, find the unique arith.mulf line that pairs it with a
        # capture-constant and record the constant's SSA name. Used by
        # the rewriter to surface body-internal weights as launch operands.
        # If an input is multiplied by more than one constant (e.g. the
        # buggy conv3d's duplicated-index pattern), record None — that
        # case needs a different matcher template anyway.
        # Build an "alias map": when the body has `%24 = arith.extsi %in : i16
        # to i32`, then `%24` is a synonym for `%in` for weight-pairing
        # purposes. C's integer-promotion rule means cgeist always inserts
        # an extsi between an i16 input and its i32-typed multiply, so the
        # mul's lhs is the extsi result, not the input itself. Same idea for
        # extui / trunci / sitofp / extf / truncf.
        alias_of: dict[str, str] = {}
        cast_re = re.compile(
            r"(%[\w_\-]+)\s*=\s*arith\."
            r"(?:extsi|extui|trunci|sitofp|uitofp|fptosi|fptoui|extf|truncf|bitcast)"
            r"\s+(%[\w_\-]+)\s*:"
        )
        for ln in body_lines:
            m_cast = cast_re.match(ln.strip())
            if m_cast:
                alias_of[m_cast.group(1)] = m_cast.group(2)

        def root_alias(ssa: str) -> str:
            # Follow the alias chain to its root (handles double casts).
            while ssa in alias_of:
                ssa = alias_of[ssa]
            return ssa

        inline_weights: list[list[str] | None] = []
        for in_arg in ins:
            constant_ssas: list[str] = []
            for ln in body_lines:
                # Match arith.mulf OR arith.muli — same surfacing logic applies
                # to integer-typed weighted stencils (the conv2d_i32 / i16
                # bodies) as to float ones.
                m_mul = re.match(
                    r"%[\w_\-]+\s*=\s*arith\.mul[fi]\s+(\S+?)\s*,\s*(\S+?)\s*:",
                    ln.strip(),
                )
                if not m_mul:
                    continue
                a, b = m_mul.group(1), m_mul.group(2)
                # Strip trailing commas (the regex's \S+? may grab one).
                a = a.rstrip(",")
                b = b.rstrip(",")
                # Resolve cast aliases so the mul's lhs (which may be an
                # extsi result) is compared to the block input arg.
                a_root = root_alias(a)
                b_root = root_alias(b)
                if a_root == in_arg and b in constants:
                    constant_ssas.append(b)
                elif b_root == in_arg and a in constants:
                    constant_ssas.append(a)
            # Empty list -> no constants paired with this input (rare); the
            # rewriter sees None and won't surface a weight for it. Single
            # or multiple -> always return the list; the rewriter decides
            # whether to use the SSA directly or materialise a summed
            # constant.
            inline_weights.append(constant_ssas if constant_ssas else None)

        results.append(GenericBody(
            ins_arg_names=ins,
            outs_arg_names=outs,
            body_lines=body_lines,
            yield_values=yield_names,
            captures=captures,
            indexing_maps=maps,
            iterator_types=iters,
            constants={
                name: constants[name]
                for name in captures
                if name in constants
            },
            inline_weights_per_in=inline_weights,
        ))
    return results


# ---------------------------------------------------------------------------
# Encoder: GenericBody -> egglog Term.
# ---------------------------------------------------------------------------

_OP_PATTERNS = {
    "arith.mulf": "mul",
    "arith.addf": "add",
    "arith.subf": "sub",
    "arith.divf": "div",
    "arith.negf": "neg",
    # Integer counterparts. The encoder collapses int and float arith into
    # the same algebraic Term (mul/add/sub/div) so one library template
    # matches both dtypes. The dtype-suffix dispatch in the rewriter picks
    # the right canonical defn and shim per element type.
    "arith.muli": "mul",
    "arith.addi": "add",
    "arith.subi": "sub",
    "arith.divsi": "div",
    "math.sqrt": "sqrt",
    "math.absf": "abs",
    "math.absi": "abs",
    # Transcendentals — used by softmax (exp), gelu (tanh), crossentropy (log).
    # Encoded as opaque unary Terms; templates can match against `Term.Exp(x)`
    # etc. so the matcher recognises the kernel without trying to fold them.
    "math.exp": "exp",
    "arith.cmpf": "cmpf",
    "arith.cmpi": "cmpi",
    "arith.select": "select",
    # Sign/zero extension and truncation cast ops. C's integer-promotion
    # rule (e.g. short * int → int) makes cgeist emit `arith.extsi %in : i16
    # to i32` before each `arith.muli`. These are semantically identity for
    # template matching — the template sees an "input × weight" product
    # regardless of how the i16/i32 widths flow underneath. Marking them
    # "transparent" makes the matcher unify both widths to the same Term.
    "arith.extsi": "transparent",
    "arith.extui": "transparent",
    "arith.trunci": "transparent",
    "arith.sitofp": "transparent",
    "arith.uitofp": "transparent",
    "arith.fptosi": "transparent",
    "arith.fptoui": "transparent",
    "arith.extf": "transparent",
    "arith.truncf": "transparent",
    "arith.bitcast": "transparent",
}


def encode_body(g: GenericBody) -> Term:
    """Build an egglog Term from a parsed body."""
    # Map SSA names to Term objects.
    env: dict[str, Term] = {}
    for i, name in enumerate(g.ins_arg_names):
        env[name] = Term.In(i)
    for i, name in enumerate(g.outs_arg_names):
        env[name] = Term.Out(i)
    for cap in g.captures:
        # Constants get a numeric Lit so identity rules can fire on them.
        if cap in g.constants:
            env[cap] = Term.Lit(g.constants[cap])
        else:
            env[cap] = Term.Cap(cap)

    def lookup(name: str) -> Term:
        """Get the Term for an SSA name; fall back to Cap/Lit for unknown values."""
        if name in env:
            return env[name]
        # Unknown — check the module-level constants map first (a yield of
        # `%cst` referring to a `arith.constant 0.0` should be Lit("0.0"),
        # not an opaque Cap).
        if name in g.constants:
            env[name] = Term.Lit(g.constants[name])
        else:
            env[name] = Term.Cap(name)
        return env[name]

    for line in g.body_lines:
        m = re.match(
            r"(%[\w_]+)\s*=\s*(\w+\.\w+)\s+(.*?)\s*:\s*\S+", line.strip()
        )
        if not m:
            continue
        result, op, args_part = m.group(1), m.group(2), m.group(3)

        # Split args by commas, ignoring those inside <...>.
        # For arith ops the args are just `%a, %b` or `%pred, %a, %b`.
        arg_toks = [s.strip() for s in args_part.split(",")]

        # Resolve each token to a Term (it's either an SSA name or a literal).
        def resolve(tok: str) -> Term:
            tok = tok.strip()
            if tok.startswith("%"):
                return lookup(tok)
            # Numeric literal. Lit is now f64-typed, so coerce. Non-numeric
            # tokens (rare — only inline-affine-attribute strings would land
            # here) get NaN as a sentinel so they still produce a valid
            # f64 Lit but won't algebraically match anything meaningful.
            try:
                return Term.Lit(float(tok))
            except ValueError:
                return Term.Lit(float("nan"))

        op_key = _OP_PATTERNS.get(op, op)
        if op_key == "transparent":
            # Cast-like op — propagate the source Term as-is.
            env[result] = resolve(arg_toks[0])
            continue
        if op_key == "mul":
            env[result] = resolve(arg_toks[0]) * resolve(arg_toks[1])
        elif op_key == "add":
            env[result] = resolve(arg_toks[0]) + resolve(arg_toks[1])
        elif op_key == "sub":
            env[result] = resolve(arg_toks[0]) - resolve(arg_toks[1])
        elif op_key == "neg":
            env[result] = Term.Lit(0.0) - resolve(arg_toks[0])
        elif op_key == "div":
            env[result] = resolve(arg_toks[0]) / resolve(arg_toks[1])
        elif op_key == "sqrt":
            env[result] = Term.Sqrt(resolve(arg_toks[0]))
        elif op_key == "abs":
            env[result] = Term.Abs(resolve(arg_toks[0]))
        elif op_key == "exp":
            env[result] = Term.Exp(resolve(arg_toks[0]))
        elif op_key == "select":
            env[result] = Term.Select(
                resolve(arg_toks[0]), resolve(arg_toks[1]), resolve(arg_toks[2])
            )
        elif op_key == "cmpf":
            # Form: "kind, %a, %b" — arg_toks[0]="kind", [1]=%a, [2]=%b.
            # Or sometimes "kind %a", "%b" if a space slipped in. Handle both.
            kind = arg_toks[0].strip()
            if " " in kind:
                kind, lhs_tok = kind.split(None, 1)
                rhs_tok = arg_toks[1]
            elif len(arg_toks) >= 3:
                lhs_tok, rhs_tok = arg_toks[1], arg_toks[2]
            else:
                # Malformed — fall back to opaque.
                env[result] = Term.Cap(result)
                continue
            env[result] = Term.Cmp(kind, resolve(lhs_tok), resolve(rhs_tok))
        else:
            # Unknown op — model as opaque Cap so matching still works elsewhere.
            env[result] = Term.Cap(result)

    return lookup(g.yield_value)


def encode_body_yields(g: GenericBody) -> list[Term]:
    """Multi-yield-aware sibling of `encode_body`. Returns one Term per
    `linalg.yield` operand, computed in the same body env so any shared
    intermediates are reflected across both yields.

    Single-yield bodies return a 1-element list (the same Term `encode_body`
    would have returned). Multi-yield bodies — like softmax's fused exp+sum
    body, which writes the elementwise exp to one output and the running
    sum to another in one iteration — return one Term per output position.
    Callers that match against multi-yield templates iterate this list in
    lockstep with the template's `body_per_yield`.
    """
    # Re-run encode_body's body walk but lookup ALL yields at the end.
    # Reuse encode_body for the env construction by calling it once (it
    # produces side-effects on a fresh env each invocation, so we re-do
    # the walk inline). For now the simplest implementation rebuilds the
    # env — duplicates encode_body's body-walking logic but extracts a
    # Term per yield position.
    env: dict[str, Term] = {}
    for i, name in enumerate(g.ins_arg_names):
        env[name] = Term.In(i)
    for i, name in enumerate(g.outs_arg_names):
        env[name] = Term.Out(i)
    for cap in g.captures:
        if cap in g.constants:
            env[cap] = Term.Lit(g.constants[cap])
        else:
            env[cap] = Term.Cap(cap)

    def lookup(name: str) -> Term:
        if name in env:
            return env[name]
        if name in g.constants:
            env[name] = Term.Lit(g.constants[name])
        else:
            env[name] = Term.Cap(name)
        return env[name]

    for line in g.body_lines:
        m = re.match(
            r"(%[\w_]+)\s*=\s*(\w+\.\w+)\s+(.*?)\s*:\s*\S+", line.strip()
        )
        if not m:
            continue
        result, op, args_part = m.group(1), m.group(2), m.group(3)
        arg_toks = [s.strip() for s in args_part.split(",")]

        def resolve(tok: str) -> Term:
            tok = tok.strip()
            if tok.startswith("%"):
                return lookup(tok)
            try:
                return Term.Lit(float(tok))
            except ValueError:
                return Term.Lit(float("nan"))

        op_key = _OP_PATTERNS.get(op, op)
        if op_key == "transparent":
            env[result] = resolve(arg_toks[0]); continue
        if op_key == "mul":
            env[result] = resolve(arg_toks[0]) * resolve(arg_toks[1])
        elif op_key == "add":
            env[result] = resolve(arg_toks[0]) + resolve(arg_toks[1])
        elif op_key == "sub":
            env[result] = resolve(arg_toks[0]) - resolve(arg_toks[1])
        elif op_key == "neg":
            env[result] = Term.Lit(0.0) - resolve(arg_toks[0])
        elif op_key == "div":
            env[result] = resolve(arg_toks[0]) / resolve(arg_toks[1])
        elif op_key == "sqrt":
            env[result] = Term.Sqrt(resolve(arg_toks[0]))
        elif op_key == "abs":
            env[result] = Term.Abs(resolve(arg_toks[0]))
        elif op_key == "exp":
            env[result] = Term.Exp(resolve(arg_toks[0]))
        elif op_key == "select":
            env[result] = Term.Select(
                resolve(arg_toks[0]), resolve(arg_toks[1]), resolve(arg_toks[2])
            )
        elif op_key == "cmpf":
            kind = arg_toks[0].strip()
            if " " in kind:
                kind, lhs_tok = kind.split(None, 1)
                rhs_tok = arg_toks[1]
            elif len(arg_toks) >= 3:
                lhs_tok, rhs_tok = arg_toks[1], arg_toks[2]
            else:
                env[result] = Term.Cap(result); continue
            env[result] = Term.Cmp(kind, resolve(lhs_tok), resolve(rhs_tok))
        else:
            env[result] = Term.Cap(result)

    return [lookup(yv) for yv in g.yield_values]


# ---------------------------------------------------------------------------
# Library + matcher.
# ---------------------------------------------------------------------------

@dataclass
class LibraryEntry:
    name: str               # e.g. "beta_scale", "gemm_accumulate"
    source_kernel: str      # which PolyBench file we extracted it from
    canonical_body: Term
    num_ins: int
    num_outs: int
    indexing_maps: list[str]
    iterator_types: list[str]


def equivalent(a: Term, b: Term) -> bool:
    """Are two Terms equivalent under the current algebra rules?"""
    eg = EGraph()
    eg.register(a, b)
    eg.run(algebra_rules() * 8)
    try:
        eg.check(a == b)
        return True
    except Exception:
        return False


def kernel_files(root: Path) -> list[Path]:
    return sorted(root.glob("*_debuf.mlir"))


def build_library_from_dir(root: Path) -> list[LibraryEntry]:
    """Walk *_debuf.mlir, extract bodies, dedupe by structural equivalence."""
    entries: list[LibraryEntry] = []
    for f in kernel_files(root):
        text = f.read_text()
        try:
            gens = parse_generics(text)
        except Exception as e:
            print(f"parse skip {f.name}: {e}")
            continue
        kernel = f.stem.replace("_debuf", "")
        for i, g in enumerate(gens):
            try:
                t = encode_body(g)
            except Exception as e:
                print(f"encode skip {f.name}#{i}: {e}")
                continue
            # Dedupe: if any existing entry matches structurally, reuse it.
            existing = next(
                (e for e in entries
                 if e.num_ins == len(g.ins_arg_names)
                 and e.num_outs == len(g.outs_arg_names)
                 and e.indexing_maps == g.indexing_maps
                 and e.iterator_types == g.iterator_types
                 and equivalent(e.canonical_body, t)),
                None,
            )
            if existing:
                continue
            entries.append(LibraryEntry(
                name=f"{kernel}_lg{i}",
                source_kernel=kernel,
                canonical_body=t,
                num_ins=len(g.ins_arg_names),
                num_outs=len(g.outs_arg_names),
                indexing_maps=g.indexing_maps,
                iterator_types=g.iterator_types,
            ))
    return entries


# ---------------------------------------------------------------------------
# Composition matcher: recognize sequences of linalg.generics as one library
# kernel (e.g. beta_scale + alpha_matmul = dgemm).
# ---------------------------------------------------------------------------

@dataclass
class CompositionStep:
    """One linalg.generic in a multi-step composition."""
    body: Term                          # template with Cap wildcards
    num_ins: Optional[int] = None       # expected ins count, or None for any
    num_outs: Optional[int] = None      # expected outs count, or None
    reduction_dim_count: Optional[int] = None  # number of "reduction" iters
    parallel_dim_count: Optional[int] = None   # number of "parallel" iters
    # For multi-yield linalg.generic bodies (e.g. softmax's fused exp+sum),
    # one template Term per yield position. The matcher walks both lists
    # in lockstep against `encode_body_yields(body)`. None falls back to
    # single-yield matching against `body` above. When set, num_outs
    # should equal len(body_per_yield).
    body_per_yield: Optional[list[Term]] = None
    # Non-scalar structural predicate for bodies whose semantics cannot be
    # represented by the scalar Term language. Used for guarded im2col:
    # the body contains scf.if + memref.load, and the value yielded from the
    # scf.if appears opaque to encode_body().
    special: Optional[str] = None


@dataclass
class CompositionEntry:
    """A named multi-linalg pattern.

    Each step's body template is matched (structural unification with
    Cap-as-wildcard) against the body of the next linalg.generic. The
    optional shape gates (num_ins, num_outs, reduction_dim_count) rule out
    same-body shapes that differ in linalg-level metadata (e.g. gemv vs
    axpy vs dot all share the body `out + a*b` but differ in iter types).

    `form` gates whether the entry fires on tensor-form linalg.generic
    (the default, what `--linalg-debufferize` produces), memref-form (used
    by stencils + other ops where debufferize doesn't lift due to outer
    time-stepping loops), or both. The canonical library defn for each
    entry only operates on one of those forms — matching the wrong form
    causes the lowering pass to fail with a type mismatch. Setting `form`
    here keeps the matcher honest.
    """
    name: str
    steps: list[CompositionStep]
    form: str = "tensor"   # "tensor" | "memref" | "any"
    # When True, the rewriter additionally appends the matched body's
    # inline weight constants (one per input block arg) as scalar operands
    # of the emitted kernel.launch op. Use for templates whose body has the
    # shape `sum_k In(k) * Cap("%wk")` where each weight is a body-internal
    # arith.constant (e.g. conv2d_9pt_weighted). The lowering pass can then
    # pass those weights to a generic runtime shim instead of hardcoding
    # them. Default False to keep behavior of every other template (gemm,
    # gemv, jacobi, ...) unchanged — they already surface scalars via
    # function-arg Caps, not body-internal Lits.
    surface_inline_weights: bool = False


# Canonical body templates. Cap names are template wildcards — they bind
# to whatever capture appears in the user's body at that position.
# Op-name targets follow real library API naming
# (cublasD<routine> / cusolverDn<routine> / cudnn<routine>...).
#
# Body shape -> library target.

def T_cap(name: str) -> Term:
    return Term.Cap(name)


def _gemm_composition() -> CompositionEntry:
    """C = β*C + α*A*B  (PolyBench gemm form)."""
    s1 = CompositionStep(
        body=Term.Out(0) * T_cap("%beta"),
        num_ins=0, num_outs=1, parallel_dim_count=2, reduction_dim_count=0,
    )
    s2 = CompositionStep(
        body=Term.Out(0) + (T_cap("%alpha") * Term.In(0)) * Term.In(1),
        num_ins=2, num_outs=1, parallel_dim_count=2, reduction_dim_count=1,
    )
    return CompositionEntry(name="cublasDgemm", steps=[s1, s2])


def _gemm_alpha_only() -> CompositionEntry:
    """C += α*A*B  (no beta — used by 2mm/3mm intermediates)."""
    body = Term.Out(0) + (T_cap("%alpha") * Term.In(0)) * Term.In(1)
    return CompositionEntry(
        name="cublasDgemm_alpha_only",
        steps=[CompositionStep(body=body, num_ins=2, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=1)],
    )


def _conv1x1_as_gemm_batched() -> CompositionEntry:
    """Batched 1×1 convolution. Mathematically a per-pixel matmul:
       (B·H·W, IC) × (IC, OC) → (B·H·W, OC)
    Because KH = KW = 1, the trivial inner loops drop out at raise
    time, leaving a 5-iter generic (4 parallel: B, OC, H, W; 1
    reduction: IC) with body `Out + In(0)*In(1)`.

    Distinguished from the standard K×K conv (`cudnnConvolutionFwd_batched`,
    which has 4 par + 3 red) purely by the reduction count.
    Routes to cublasDgemm via a reshape — much faster than cuDNN's
    generic K=1 conv path.
    """
    init_step = CompositionStep(
        body=Term.Lit(0.0),
        num_ins=0, num_outs=1,
        parallel_dim_count=4, reduction_dim_count=0,
    )
    gemm_step = CompositionStep(
        body=Term.Out(0) + Term.In(0) * Term.In(1),
        num_ins=2, num_outs=1,
        parallel_dim_count=4, reduction_dim_count=1,
    )
    return CompositionEntry(
        name="cublasGemmFor1x1Conv",
        steps=[init_step, gemm_step],
    )


def _cublaslt_gemm_bias_relu_fused() -> CompositionEntry:
    """Fused matmul + bias + relu — transformer-FFN-shape op.
    4-step composition:

      step 0 (init):  C = 0                  — 2 par, 0 ins
      step 1 (gemm):  C += A*B               — 2 par + 1 red, 2 ins
      step 2 (bias):  C += bias              — 2 par, 1 in (1D, broadcast)
      step 3 (relu):  C = max(C, 0)          — 2 par, 0 ins

    Routes to cublasLt's CUBLASLT_EPILOGUE_RELU_BIAS — natively fuses
    matmul + bias-add + relu in one kernel. Requires libcublasLt at link
    time (separate from libcublas).
    """
    init_step = CompositionStep(
        body=Term.Lit(0.0),
        num_ins=0, num_outs=1,
        parallel_dim_count=2, reduction_dim_count=0,
    )
    gemm_step = CompositionStep(
        body=Term.Out(0) + Term.In(0) * Term.In(1),
        num_ins=2, num_outs=1,
        parallel_dim_count=2, reduction_dim_count=1,
    )
    bias_step = CompositionStep(
        body=Term.Out(0) + Term.In(0),
        num_ins=1, num_outs=1,
        parallel_dim_count=2, reduction_dim_count=0,
    )
    relu_step = CompositionStep(
        body=Term.Select(
            Term.Cmp("ogt", Term.Out(0), Term.Lit(0.0)),
            Term.Out(0),
            Term.Lit(0.0),
        ),
        num_ins=0, num_outs=1,
        parallel_dim_count=2, reduction_dim_count=0,
    )
    return CompositionEntry(
        name="cublasLtMatmulBiasReluFused",
        steps=[init_step, gemm_step, bias_step, relu_step],
    )


def _cudnn_conv_bias_relu_add_fused() -> CompositionEntry:
    """Fused conv + bias + residual-add + relu — canonical ResNet output
    stage. 5-step composition:

      step 0 (init):     Bout = 0                  — 4 par, 0 ins
      step 1 (conv):     Bout += A * F             — 4 par + 3 red, 2 ins
      step 2 (bias):     Bout += bias[oc]          — 4 par, 1 in (1D)
      step 3 (residual): Bout += Z                 — 4 par, 1 in (4D)
      step 4 (relu):     Bout = max(Bout, 0)       — 4 par, 0 ins

    Steps 2 and 3 have IDENTICAL body shape (`Out + In(0)`). The matcher
    only checks the body Term-AST, so it doesn't know "this is the bias"
    vs "this is the residual" at match time. The lowering pass
    disambiguates by operand rank after submap resolution:
      - 1D operand → bias (per-channel)
      - 4D operand → residual (same shape as output)

    Routes to cudnnConvolutionBiasActivationForward, which natively
    computes y = activation(α₁·conv(x,w) + α₂·z + bias).
    """
    init_step = CompositionStep(
        body=Term.Lit(0.0),
        num_ins=0, num_outs=1,
        parallel_dim_count=4, reduction_dim_count=0,
    )
    conv_step = CompositionStep(
        body=Term.Out(0) + Term.In(0) * Term.In(1),
        num_ins=2, num_outs=1,
        parallel_dim_count=4, reduction_dim_count=3,
    )
    add_step = CompositionStep(
        body=Term.Out(0) + Term.In(0),
        num_ins=1, num_outs=1,
        parallel_dim_count=4, reduction_dim_count=0,
    )
    relu_step = CompositionStep(
        body=Term.Select(
            Term.Cmp("ogt", Term.Out(0), Term.Lit(0.0)),
            Term.Out(0),
            Term.Lit(0.0),
        ),
        num_ins=0, num_outs=1,
        parallel_dim_count=4, reduction_dim_count=0,
    )
    return CompositionEntry(
        name="cudnnConvBiasReluAddFwdFused",
        steps=[init_step, conv_step, add_step, add_step, relu_step],
    )


def _cudnn_conv_bn_relu_fused() -> CompositionEntry:
    """Fused conv + bn (inference) + relu — the inner three ops of a
    ResNet residual block. 4-step composition:

      step 1 (init):   Bout = 0           — 4 par, 0 ins
      step 2 (conv):   Bout += A * F      — 4 par + 3 red, 2 ins
      step 3 (bn):     Bout = scale*(Bout - mean)*inv_std + bias
                                          — 4 par, 4 ins (scale, mean,
                                            inv_std, bias). In-place form:
                                            Bout is BOTH read (as Out(0))
                                            AND written.
      step 4 (relu):   Bout = max(Bout, 0)
                                          — 4 par, 0 ins, in-place

    Body shapes (from cgeist + raise on conv_bn_relu_batched.c):
      step 3:  In(0) * (Out(0) - In(1)) * In(2) + In(3)
      step 4:  Select(Cmp("ogt", Out(0), Lit(0.0)), Out(0), Lit(0.0))

    Lowers to cudnnConvolutionBiasActivationForward (cuDNN's native
    fused-conv-bias-relu kernel) — needs a runtime shim that folds the
    BN parameters into a per-output-channel scaled filter + bias
    (standard "BN-folding" trick), then issues one cuDNN call instead
    of three.
    """
    init_step = CompositionStep(
        body=Term.Lit(0.0),
        num_ins=0, num_outs=1,
        parallel_dim_count=4, reduction_dim_count=0,
    )
    conv_step = CompositionStep(
        body=Term.Out(0) + Term.In(0) * Term.In(1),
        num_ins=2, num_outs=1,
        parallel_dim_count=4, reduction_dim_count=3,
    )
    bn_step = CompositionStep(
        body=(Term.In(0) * (Term.Out(0) - Term.In(1))) * Term.In(2)
             + Term.In(3),
        num_ins=4, num_outs=1,
        parallel_dim_count=4, reduction_dim_count=0,
    )
    relu_step = CompositionStep(
        body=Term.Select(
            Term.Cmp("ogt", Term.Out(0), Term.Lit(0.0)),
            Term.Out(0),
            Term.Lit(0.0),
        ),
        num_ins=0, num_outs=1,
        parallel_dim_count=4, reduction_dim_count=0,
    )
    return CompositionEntry(
        name="cudnnConvBnReluFwdFused",
        steps=[init_step, conv_step, bn_step, relu_step],
    )


def _cudnn_add_tensor_batched() -> CompositionEntry:
    """Batched 4D elementwise tensor add (ResNet residual shortcut):
       out[b,c,h,w] = in[b,c,h,w] + out[b,c,h,w]

    4-parallel, 0-reduction, 1 input, 1 output. No captures.

    The shape gates (parallel_dim_count=4, num_ins=1, body=`Out + In(0)`)
    distinguish this from axpy (which needs an α capture) and from any
    accumulating contraction (which would have reduction iters). Maps
    to cudnnAddTensor.
    """
    body = Term.Out(0) + Term.In(0)
    return CompositionEntry(
        name="cudnnAddTensor_batched",
        steps=[
            CompositionStep(
                body=body,
                num_ins=1, num_outs=1,
                parallel_dim_count=4, reduction_dim_count=0,
            ),
        ],
    )


def _cudnn_batchnorm_inference() -> CompositionEntry:
    """Batched per-channel batch normalization (inference mode):
       out[b,c,h,w] = scale[c] * (in[b,c,h,w] - mean[c]) * inv_std[c]
                      + bias[c]

    Shape: 4-parallel (B, C, H, W), zero reductions. 5 inputs (scale, A,
    mean, inv_std, bias all broadcast through `polygeist.submap` from
    their 4D / 1D shapes into the 4D iteration domain), 1 output.

    Maps to cudnnBatchNormalizationForwardInference. The runtime shim
    takes the 4D input/output + four 1D per-channel vectors and lets
    cuDNN do the fused normalize+scale+bias in one launch.

    The body order assumes the raise pass orders the ins as
    (scale, A, mean, inv_std, bias) — observed on the batchnorm_batched
    test file. If a future input reorders these (different argument
    order in the C source), the unifier sees a different shape and the
    match fails — at that point the template needs alternate input
    orderings or a more permissive structural match.
    """
    # ((scale * (A - mean)) * inv_std) + bias
    body = (
        Term.In(0) * (Term.In(1) - Term.In(2))
    ) * Term.In(3) + Term.In(4)
    return CompositionEntry(
        name="cudnnBatchNormalizationForwardInference",
        steps=[
            CompositionStep(
                body=body,
                num_ins=5, num_outs=1,
                parallel_dim_count=4, reduction_dim_count=0,
            ),
        ],
    )


def _cudnn_maxpool_batched() -> CompositionEntry:
    """Batched multi-channel 2D max pooling. Two steps:
      step1 (init): outs[b,c,oh,ow] = -INF  — 4 parallel, 0 ins.
      step2 (reduce): outs[b,c,oh,ow] = max(In(0), Out(0))
                       — 4 parallel + 2 reduction over (kh, kw).

    Body of step2 lowers from cgeist's `(v > cur) ? v : cur` ternary
    via arith.cmpf + arith.select. The matcher's algebraic encoder
    sees the select as a max op and produces a clean max-reduction
    body shape.
    """
    return CompositionEntry(
        name="cudnnMaxPoolFwd_batched",
        steps=[
            CompositionStep(
                # -FLT_MAX (≈ -3.4028235e38). cgeist canonicalises whatever
                # the C source writes (-INFINITY, -FLT_MAX, -3.4e38, etc.)
                # to the IEEE-754 float32 minimum which MLIR prints as
                # -3.40282347E+38. Matching the exact parsed value here.
                body=Term.Lit(-3.40282347e38),
                num_ins=0, num_outs=1,
                parallel_dim_count=4, reduction_dim_count=0,
            ),
            # max(In(0), Out(0)) — cgeist lowers the ternary
            # `(v > cur) ? v : cur` to `arith.cmpf ogt + arith.select`. The
            # encoder turns that into `Select(Cmp("ogt", In, Out), In, Out)`,
            # which is the same shape the softmax max-reduce step uses.
            CompositionStep(
                body=Term.Select(
                    Term.Cmp("ogt", Term.In(0), Term.Out(0)),
                    Term.In(0),
                    Term.Out(0),
                ),
                num_ins=1, num_outs=1,
                parallel_dim_count=4, reduction_dim_count=2,
            ),
        ],
    )


def _cudnn_conv2d_batched() -> CompositionEntry:
    """Batched multi-channel 2D convolution: out[b,oc,oh,ow] =
       Σ_{ic,kh,kw} in[b,ic,oh+kh,ow+kw] * filter[oc,ic,kh,kw].

    Two-step composition:
      step1 (init): outs[b,oc,oh,ow] = 0 — 4 parallel iters, 0 inputs.
      step2 (accumulate): same outs with 2 inputs (input + filter),
                          4 parallel + 3 reduction (over ic, kh, kw).

    The input tensor reaches the accumulation linalg.generic via a
    polygeist.submap that produces a 7D strided-window view of the
    original 4D input — that's the implicit im2col. The downstream
    lowering doesn't need to inspect the submap; it just maps to a
    cudnnConvolutionForward call with the standard 4D NCHW descriptors,
    and the runtime shim runs the actual convolution. The matcher only
    checks body shape + iter-type counts here.
    """
    return CompositionEntry(
        name="cudnnConvolutionFwd_batched",
        steps=[
            CompositionStep(
                body=Term.Lit(0.0),  # init body: yield 0
                num_ins=0, num_outs=1,
                parallel_dim_count=4, reduction_dim_count=0,
            ),
            CompositionStep(
                body=Term.Out(0) + Term.In(0) * Term.In(1),
                num_ins=2, num_outs=1,
                parallel_dim_count=4, reduction_dim_count=3,
            ),
        ],
    )


def _darknet_im2col_gemm_fused() -> CompositionEntry:
    """Darknet-style explicit im2col followed by GEMM.

    Raised memref IR shape:
      step0: output[:] = 0                         -- 1D flat zero-fill
      step1: workspace[k, oh, ow] = guarded load   -- im2col with zero pad
      step2: output[oc, oh*ow] += weights[oc,k] *
                                    workspace[k,oh*ow]

    The im2col body contains an scf.if and a memref.load, so the scalar Term
    encoder sees it as opaque. Match it with a structural predicate, then
    lower the whole 3-step composition as one cuDNN convolution.
    """
    init_step = CompositionStep(
        body=Term.Lit(0.0),
        num_ins=0, num_outs=1,
        parallel_dim_count=1, reduction_dim_count=0,
    )
    im2col_step = CompositionStep(
        body=T_cap("%guarded_im2col"),
        num_ins=0, num_outs=1,
        parallel_dim_count=3, reduction_dim_count=0,
        special="guarded_im2col",
    )
    gemm_step = CompositionStep(
        body=Term.Out(0) + Term.In(0) * Term.In(1),
        num_ins=2, num_outs=1,
        parallel_dim_count=2, reduction_dim_count=1,
    )
    return CompositionEntry(
        name="cudnnConvolutionFwd_im2col_gemm",
        steps=[init_step, im2col_step, gemm_step],
        form="memref",
    )


def _gemm_no_alpha() -> CompositionEntry:
    """C += A*B  (no alpha, no beta)."""
    body = Term.Out(0) + Term.In(0) * Term.In(1)
    return CompositionEntry(
        name="cublasDgemm_simple",
        steps=[CompositionStep(body=body, num_ins=2, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=1)],
    )


def _sgemm_broadcast3d_memref() -> CompositionEntry:
    """Darknet im2col GEMM in memref form after scalar-load promotion.

    The linalg view is rank-3 because A and C are broadcasted through submaps,
    but the underlying buffers are flat row-major A[M,K], B[K,N], C[M,N].
    """
    body = Term.Out(0) + Term.In(0) * Term.In(1)
    return CompositionEntry(
        name="cublasSgemm_broadcast3d_memref",
        steps=[CompositionStep(body=body, num_ins=2, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=1)],
        form="memref",
    )


def _gemv_accumulate() -> CompositionEntry:
    """y += A * x  (no alpha/beta)."""
    body = Term.Out(0) + Term.In(0) * Term.In(1)
    return CompositionEntry(
        name="cublasDgemv",
        steps=[CompositionStep(body=body, num_ins=2, num_outs=1,
                                parallel_dim_count=1, reduction_dim_count=1)],
    )


def _gemv_alpha_accumulate() -> CompositionEntry:
    """y += alpha * A * x"""
    body = Term.Out(0) + (T_cap("%alpha") * Term.In(0)) * Term.In(1)
    return CompositionEntry(
        name="cublasDgemv_alpha",
        steps=[CompositionStep(body=body, num_ins=2, num_outs=1,
                                parallel_dim_count=1, reduction_dim_count=1)],
    )


def _axpy() -> CompositionEntry:
    """y[i] += alpha * x[i]"""
    body = Term.Out(0) + T_cap("%alpha") * Term.In(0)
    return CompositionEntry(
        name="cublasDaxpy",
        steps=[CompositionStep(body=body, num_ins=1, num_outs=1,
                                reduction_dim_count=0)],
    )


def _scal_1d() -> CompositionEntry:
    """x[i] *= alpha  — 1D vector."""
    body = Term.Out(0) * T_cap("%alpha")
    return CompositionEntry(
        name="cublasDscal",
        steps=[CompositionStep(body=body, num_ins=0, num_outs=1,
                                parallel_dim_count=1, reduction_dim_count=0)],
    )


def _scal_2d() -> CompositionEntry:
    """X[i,j] *= alpha  — 2D matrix (e.g. β-scale of C)."""
    body = Term.Out(0) * T_cap("%alpha")
    return CompositionEntry(
        name="cublasDgeam_scale2D",
        steps=[CompositionStep(body=body, num_ins=0, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=0)],
    )


def _fill_zero_1d() -> CompositionEntry:
    body = Term.Lit(0.0)
    return CompositionEntry(
        name="memset_zero_1D",
        steps=[CompositionStep(body=body, num_ins=0, num_outs=1,
                                parallel_dim_count=1, reduction_dim_count=0)],
    )


def _fill_zero_2d() -> CompositionEntry:
    body = Term.Lit(0.0)
    return CompositionEntry(
        name="memset_zero_2D",
        steps=[CompositionStep(body=body, num_ins=0, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=0)],
    )


def _fill_const_1d() -> CompositionEntry:
    """x[i] = constant capture (1-d fill)."""
    body = T_cap("%const")
    return CompositionEntry(
        name="memset_const_1D",
        steps=[CompositionStep(body=body, num_ins=0, num_outs=1,
                                parallel_dim_count=1, reduction_dim_count=0)],
    )


def _fill_const_2d() -> CompositionEntry:
    body = T_cap("%const")
    return CompositionEntry(
        name="memset_const_2D",
        steps=[CompositionStep(body=body, num_ins=0, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=0)],
    )


def _dot() -> CompositionEntry:
    """s = sum_i x[i] * y[i]"""
    body = Term.Out(0) + Term.In(0) * Term.In(1)
    return CompositionEntry(
        name="cublasDdot",
        steps=[CompositionStep(body=body, num_ins=2, num_outs=1,
                                parallel_dim_count=0, reduction_dim_count=1)],
    )


def _asum() -> CompositionEntry:
    """s = sum_i |x[i]|"""
    body = Term.Out(0) + Term.Abs(Term.In(0))
    return CompositionEntry(
        name="cublasDasum",
        steps=[CompositionStep(body=body, num_ins=1, num_outs=1,
                                parallel_dim_count=0, reduction_dim_count=1)],
    )


def _divf_scalar() -> CompositionEntry:
    """out /= alpha  (e.g. mean computation)."""
    body = Term.Out(0) / T_cap("%alpha")
    return CompositionEntry(
        name="elemwise_div_scalar",
        steps=[CompositionStep(body=body, num_ins=0, num_outs=1)],
    )


def _subf_inputs() -> CompositionEntry:
    """out = in0 - in1  (e.g. centering)."""
    body = Term.In(0) - Term.In(1)
    return CompositionEntry(
        name="elemwise_sub_inputs",
        steps=[CompositionStep(body=body, num_ins=2, num_outs=1)],
    )


def _reduce_sum_axis() -> CompositionEntry:
    """out[j] = sum_i in[?, ?]  — reduce across one axis. 1 parallel + 1 reduction."""
    body = Term.Out(0) + Term.In(0)
    return CompositionEntry(
        name="reduce_sum_axis",
        steps=[CompositionStep(body=body, num_ins=1, num_outs=1,
                                parallel_dim_count=1, reduction_dim_count=1)],
    )


def _vector_add_no_alpha() -> CompositionEntry:
    """y += x  — vector add (axpy with alpha = 1, gemver third stage)."""
    body = Term.Out(0) + Term.In(0)
    return CompositionEntry(
        name="cublasDaxpy_unit",
        steps=[CompositionStep(body=body, num_ins=1, num_outs=1,
                                parallel_dim_count=1, reduction_dim_count=0)],
    )


def _centered_sum_squares() -> CompositionEntry:
    """out += (in0 - in1) * (in0 - in1)  — variance accumulation."""
    diff = Term.In(0) - Term.In(1)
    body = Term.Out(0) + diff * diff
    return CompositionEntry(
        name="centered_sum_squares",
        steps=[CompositionStep(body=body, num_ins=2, num_outs=1,
                                reduction_dim_count=1)],
    )


def _trmm_masked() -> CompositionEntry:
    """out += in0 * in1, only where mask predicate holds  — cublasDtrmm body."""
    body = Term.Select(T_cap("%mask"),
                       Term.Out(0) + Term.In(0) * Term.In(1),
                       Term.Out(0))
    return CompositionEntry(
        name="cublasDtrmm",
        steps=[CompositionStep(body=body, num_ins=2, num_outs=1,
                                parallel_dim_count=1, reduction_dim_count=1)],
    )


def _syrk_composition() -> CompositionEntry:
    """C[j<=i] = β*C[j<=i] + α*A*A^T  (symmetric rank-k update, triangular).

    Two-step: masked beta-scale then masked alpha-gemm-accumulate. The mask
    predicate is a per-step Cap because the encoder treats `arith.cmpi +
    linalg.index + affine.apply` as opaque — and each step's predicate has a
    *distinct* SSA name (e.g. %9 in step 1, %11 in step 2). Use per-step
    capture names so the cross-step binding merge in match_composition
    doesn't try to unify them.
    """
    s1 = CompositionStep(
        body=Term.Select(T_cap("%mask1"),
                         Term.Out(0) * T_cap("%beta"),
                         Term.Out(0)),
        num_ins=0, num_outs=1, parallel_dim_count=2, reduction_dim_count=0,
    )
    s2 = CompositionStep(
        body=Term.Select(T_cap("%mask2"),
                         Term.Out(0) + (T_cap("%alpha") * Term.In(0)) * Term.In(1),
                         Term.Out(0)),
        num_ins=2, num_outs=1, parallel_dim_count=2, reduction_dim_count=1,
    )
    return CompositionEntry(name="cublasDsyrk", steps=[s1, s2])


def _conv2d_9pt_weighted() -> CompositionEntry:
    """2D 9-tap weighted convolution: out = sum_{k=0..8} w_k * in_k.

    Each in_k is a strided subview of the same source tensor — one per
    3×3 neighbour position. After our `bake_polybenchgpu_extracted_mlir.sh`
    pulls the kernel out of its TU (breaking the init constant-fold chain),
    polybenchGpu's convolution-2d lifts to exactly this shape.

    Body is a left-fold sum of products, matching MLIR's natural CSE/folding
    of the polybench-style straight-line C code.
    """
    body = Term.In(0) * T_cap("%w0")
    for i in range(1, 9):
        body = body + Term.In(i) * T_cap(f"%w{i}")
    return CompositionEntry(
        name="cudnnConvolution2D_9tap",
        steps=[CompositionStep(body=body, num_ins=9, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=0)],
        form="memref",
        surface_inline_weights=True,
    )


def _conv2d_9pt_weighted_tensor() -> CompositionEntry:
    """Tensor-form sibling of _conv2d_9pt_weighted — fires after the
    multi-root debufferize on the same body."""
    body = Term.In(0) * T_cap("%w0")
    for i in range(1, 9):
        body = body + Term.In(i) * T_cap(f"%w{i}")
    return CompositionEntry(
        name="cudnnConvolution2D_9tap_tensor",
        steps=[CompositionStep(body=body, num_ins=9, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=0)],
        form="tensor",
        surface_inline_weights=True,
    )


def _conv3d_11pt_weighted() -> CompositionEntry:
    """3D 11-tap weighted convolution: out = sum_{k=0..10} w_k * in_k.

    Matches polybenchGpu's extracted conv3d body, which has 15 writes but
    only 11 unique input positions (3 positions each appear in 3 muls
    with different literal coefficients; their products are then summed).
    The factoring + literal-folding rules in `algebra_rules` collapse the
    redundant muls during egglog saturation, so the body normalises to
    one mul per unique input — exactly the shape matched here.

    The iteration nest is 3D parallel (over (i,j,k)); no reduction dims.
    """
    body = Term.In(0) * T_cap("%w0")
    for i in range(1, 11):
        body = body + Term.In(i) * T_cap(f"%w{i}")
    return CompositionEntry(
        name="cudnnConvolution3D_11tap",
        steps=[CompositionStep(body=body, num_ins=11, num_outs=1,
                                parallel_dim_count=3, reduction_dim_count=0)],
        form="memref",
        surface_inline_weights=True,
    )


def _softmax_3step() -> CompositionEntry:
    """1D softmax as 3 fused linalg.generic ops, matching what cgeist + raise
    produces for llama2.c's softmax (and the per-(B,T) row in llm.c's
    softmax_forward, after the outer affine.fors are stripped).

    Step 0 — max reduction (1 in, 1 scalar out):
        out = (in > out) ? in : out                 → Select(Cmp("ogt", In(0), Out(0)), In(0), Out(0))

    Step 1 — fused exp + sum-accumulate (0 ins, 2 outs, MULTI-YIELD):
        out_0 = exp(out_0 - max)                    → yield[0] = Exp(Out(0) - Cap("%max"))
        out_1 = out_1 + exp(out_0 - max)            → yield[1] = Out(1) + Exp(Out(0) - Cap("%max"))
      Note: both yields share the same `exp(out_0 - max)` intermediate;
      encode_body_yields produces two Terms in the same body env so the
      shared subexpression is structurally identical, letting _unify bind
      Cap("%max") consistently across both yield slots.

    Step 2 — divide-by-sum (0 ins, 1 out, parallel):
        out = out / sum                             → Out(0) / Cap("%sum")

    Lowers to a single kernel.launch @cudnnSoftmaxForward — cuDNN's
    softmax kernel implements exactly the max-shift / exp / sum-normalize
    pipeline natively, in one launch with tensor-core kernels on FP16/BF16
    inputs.
    """
    step0 = CompositionStep(
        body=Term.Select(
            Term.Cmp("ogt", Term.In(0), Term.Out(0)),
            Term.In(0),
            Term.Out(0),
        ),
        num_ins=1, num_outs=1,
        reduction_dim_count=1, parallel_dim_count=0,
    )
    exp_intermediate = Term.Exp(Term.Out(0) - T_cap("%max"))
    step1 = CompositionStep(
        body=exp_intermediate,  # back-compat placeholder; matcher uses body_per_yield
        body_per_yield=[
            exp_intermediate,                       # yield[0]: writes back to array
            Term.Out(1) + exp_intermediate,         # yield[1]: accumulates into sum scalar
        ],
        num_ins=0, num_outs=2,
        reduction_dim_count=1, parallel_dim_count=0,
    )
    step2 = CompositionStep(
        body=Term.Out(0) / T_cap("%sum"),
        num_ins=0, num_outs=1,
        reduction_dim_count=0, parallel_dim_count=1,
    )
    return CompositionEntry(
        name="cudnnSoftmaxForward",
        steps=[step0, step1, step2],
        form="memref",
    )


def _softmax_3step_tensor() -> CompositionEntry:
    entry = _softmax_3step()
    return CompositionEntry(
        name="cudnnSoftmaxForward_tensor",
        steps=entry.steps,
        form="tensor",
    )


def _softmax_3step_out_tensor() -> CompositionEntry:
    """Out-of-place 1D softmax:

        max = reduce_max(scores)
        out[i] = exp(scores[i] - max); sum += out[i]
        out[i] /= sum

    This is the standalone attention-softmax fixture shape. The CUDA lowering
    copies scores to out and routes the normalized row through cuDNN softmax.
    """
    step0 = CompositionStep(
        body=Term.Select(
            Term.Cmp("ogt", Term.In(0), Term.Out(0)),
            Term.In(0),
            Term.Out(0),
        ),
        num_ins=1, num_outs=1,
        reduction_dim_count=1, parallel_dim_count=0,
    )
    exp_intermediate = Term.Exp(Term.In(0) - T_cap("%max"))
    step1 = CompositionStep(
        body=exp_intermediate,
        body_per_yield=[
            exp_intermediate,
            Term.Out(1) + exp_intermediate,
        ],
        num_ins=1, num_outs=2,
        reduction_dim_count=1, parallel_dim_count=0,
    )
    step2 = CompositionStep(
        body=Term.Out(0) / T_cap("%sum"),
        num_ins=0, num_outs=1,
        reduction_dim_count=0, parallel_dim_count=1,
    )
    return CompositionEntry(
        name="cudnnSoftmaxForwardOut_tensor",
        steps=[step0, step1, step2],
        form="tensor",
    )


def _rmsnorm_2step() -> CompositionEntry:
    """RMSNorm — 1D root-mean-square normalize + per-element weighted scale.

    cgeist + raise produces two linalg.generic ops in sequence, with the
    scale computation (`scale = 1/sqrt(ss/N + eps)`) inlined between them
    as ordinary scalar arith on the host side:

        Step 0 — ss = sum(x[i]²):  reduction, 1 in (x), 1 scalar out
            body = Out(0) + (In(0) * In(0))

        [inline: load ss; divf ss/N; addf +eps; sqrt; divf 1/sqrt → %scale]

        Step 1 — out = weight * scale * x:  parallel, 2 ins (weight, x),
                                            1 out, captures %scale
            body = In(0) * (Cap("%scale") * In(1))

    The Cap binds to whatever body-external SSA the rewriter sees feeding
    the second linalg's body — typically the `%5 = arith.divf %cst, %4`
    result of the inlined scale computation.

    Lowers to an `rmsnorm` kernel.launch. cuDNN has no native RMSNorm
    entry (its `cudnnNormForward` always mean-centers). The runtime shim
    is the natural place to decide between (a) cuBLAS decomposition
    (cublasSdot for ss + scalar arith on host + per-element fused scale,
    weight, multiply), (b) cuDNN LayerNorm with mean=0 trick
    (version-dependent), or (c) a hand-written CUDA kernel (the
    production choice in TRT-LLM / vLLM).
    """
    step0 = CompositionStep(
        body=Term.Out(0) + (Term.In(0) * Term.In(0)),
        num_ins=1, num_outs=1,
        reduction_dim_count=1, parallel_dim_count=0,
    )
    step1 = CompositionStep(
        body=Term.In(0) * (T_cap("%scale") * Term.In(1)),
        num_ins=2, num_outs=1,
        reduction_dim_count=0, parallel_dim_count=1,
    )
    return CompositionEntry(
        name="rmsnorm_f32",
        steps=[step0, step1],
        form="any",
    )


def _llama_add_f32_tensor() -> CompositionEntry:
    """out = in0 + in1 — residual add in standalone Llama fixtures."""
    return CompositionEntry(
        name="cudaAdd_f32_tensor",
        steps=[CompositionStep(body=Term.In(0) + Term.In(1),
                                num_ins=2, num_outs=1,
                                parallel_dim_count=1, reduction_dim_count=0)],
        form="tensor",
    )


def _llama_mask_select_f32_tensor() -> CompositionEntry:
    """Branchless causal mask fixture:

        drop = (i > pos)
        out = (1 - drop) * scores + drop * NEG_INF

    The `%mask` cap is produced from linalg.index inside the linalg body; the
    rewriter special-cases this symbol and surfaces the real `%pos` operand.
    """
    drop = T_cap("%mask")
    body = (Term.Lit(1.0) - drop) * Term.In(0) + \
           drop * Term.Lit(-3.40282347e38)
    return CompositionEntry(
        name="cudaMaskSelect_f32_tensor",
        steps=[CompositionStep(body=body, num_ins=1, num_outs=1,
                                parallel_dim_count=1, reduction_dim_count=0)],
        form="tensor",
    )


def _llama_swiglu_f32_tensor() -> CompositionEntry:
    """out = (gate / (1 + exp(-gate))) * up."""
    gate = Term.In(0)
    body = (gate / (Term.Exp(Term.Lit(0.0) - gate) + Term.Lit(1.0))) * Term.In(1)
    return CompositionEntry(
        name="cudaSwiGLU_f32_tensor",
        steps=[CompositionStep(body=body, num_ins=2, num_outs=1,
                                parallel_dim_count=1, reduction_dim_count=0)],
        form="tensor",
    )


def _llama_rope_mulmul_sub_f32_tensor() -> CompositionEntry:
    """RoPE split even output: out[h,p] = a[h,p] * b[p] - c[h,p] * d[p]."""
    body = Term.In(0) * Term.In(1) - Term.In(2) * Term.In(3)
    return CompositionEntry(
        name="cudaRopeMulMulSub_f32_tensor",
        steps=[CompositionStep(body=body, num_ins=4, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=0)],
        form="tensor",
    )


def _llama_rope_mulmul_add_f32_tensor() -> CompositionEntry:
    """RoPE split odd output: out[h,p] = a[h,p] * b[p] + c[h,p] * d[p]."""
    body = Term.In(0) * Term.In(1) + Term.In(2) * Term.In(3)
    return CompositionEntry(
        name="cudaRopeMulMulAdd_f32_tensor",
        steps=[CompositionStep(body=body, num_ins=4, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=0)],
        form="tensor",
    )


def _jacobi_1d_3pt() -> CompositionEntry:
    """Jacobi 1D 3-point smoother: out[i] = (a + b + c) * coef
    where a, b, c are the left/center/right neighbors (encoded via subview
    offsets, so the linalg body just sees three identity-accessed inputs)."""
    body = (Term.In(0) + Term.In(1) + Term.In(2)) * T_cap("%coef")
    return CompositionEntry(
        name="jacobi_1d_3pt",
        steps=[CompositionStep(body=body, num_ins=3, num_outs=1,
                                parallel_dim_count=1, reduction_dim_count=0)],
        form="memref",
    )


# Tensor-form variants of the stencils. Multi-root debufferize lifts these
# kernels to tensor-form linalg.generic (with polygeist.submap doing the
# offset work that memref.subview did in the memref form). The body is
# identical, only the operand/result types change — hence a separate entry
# per stencil pointing to a tensor-typed canonical defn in the library.
def _jacobi_1d_3pt_tensor() -> CompositionEntry:
    body = (Term.In(0) + Term.In(1) + Term.In(2)) * T_cap("%coef")
    return CompositionEntry(
        name="jacobi_1d_3pt_tensor",
        steps=[CompositionStep(body=body, num_ins=3, num_outs=1,
                                parallel_dim_count=1, reduction_dim_count=0)],
        form="tensor",
    )


def _jacobi_2d_5pt() -> CompositionEntry:
    """Jacobi 2D 5-point stencil: out[i,j] = (n + s + w + e + c) * coef."""
    body = ((((Term.In(0) + Term.In(1)) + Term.In(2))
              + Term.In(3)) + Term.In(4)) * T_cap("%coef")
    return CompositionEntry(
        name="jacobi_2d_5pt",
        steps=[CompositionStep(body=body, num_ins=5, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=0)],
        form="memref",
    )


def _jacobi_2d_5pt_tensor() -> CompositionEntry:
    body = ((((Term.In(0) + Term.In(1)) + Term.In(2))
              + Term.In(3)) + Term.In(4)) * T_cap("%coef")
    return CompositionEntry(
        name="jacobi_2d_5pt_tensor",
        steps=[CompositionStep(body=body, num_ins=5, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=0)],
        form="tensor",
    )


def _heat_3d_7pt() -> CompositionEntry:
    """Heat 3D 7-point Laplacian update:
        out = (l - 2*c + r)*coef + (d - 2*c + u)*coef + (b - 2*c + f)*coef + c
    where c = In(1) is the center; the other 6 ins are the axial neighbors.
    The encoder pairs ins by subview-offset order: x-neighbors (In(0),In(2)),
    y-neighbors (In(3),In(4)), z-neighbors (In(5),In(6)).
    """
    c = Term.In(1)
    two = T_cap("%two")
    coef = T_cap("%coef")
    dx = (Term.In(0) - c * two + Term.In(2)) * coef
    dy = (Term.In(3) - c * two + Term.In(4)) * coef
    dz = (Term.In(5) - c * two + Term.In(6)) * coef
    body = ((dx + dy) + dz) + c
    return CompositionEntry(
        name="heat_3d_7pt",
        steps=[CompositionStep(body=body, num_ins=7, num_outs=1,
                                parallel_dim_count=3, reduction_dim_count=0)],
        form="memref",
    )


def _heat_3d_7pt_tensor() -> CompositionEntry:
    c = Term.In(1)
    two = T_cap("%two")
    coef = T_cap("%coef")
    dx = (Term.In(0) - c * two + Term.In(2)) * coef
    dy = (Term.In(3) - c * two + Term.In(4)) * coef
    dz = (Term.In(5) - c * two + Term.In(6)) * coef
    body = ((dx + dy) + dz) + c
    return CompositionEntry(
        name="heat_3d_7pt_tensor",
        steps=[CompositionStep(body=body, num_ins=7, num_outs=1,
                                parallel_dim_count=3, reduction_dim_count=0)],
        form="tensor",
    )


def _fdtd_update_2in() -> CompositionEntry:
    """FDTD H-field update: out -= coef * (in0 - in1).
    Used for both H_x and H_y in fdtd-2d's per-time-step body."""
    body = Term.Out(0) - (Term.In(0) - Term.In(1)) * T_cap("%coef")
    return CompositionEntry(
        name="fdtd_update_2in",
        steps=[CompositionStep(body=body, num_ins=2, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=0)],
        form="memref",
    )


def _fdtd_update_2in_tensor() -> CompositionEntry:
    body = Term.Out(0) - (Term.In(0) - Term.In(1)) * T_cap("%coef")
    return CompositionEntry(
        name="fdtd_update_2in_tensor",
        steps=[CompositionStep(body=body, num_ins=2, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=0)],
        form="tensor",
    )


def _fdtd_E_update() -> CompositionEntry:
    """FDTD E-field update: out -= coef * (in0 - in1 + in2 - in3).
    The four ins are paired (curl_x, curl_y) contributions."""
    body = Term.Out(0) - (
        ((Term.In(0) - Term.In(1)) + Term.In(2)) - Term.In(3)
    ) * T_cap("%coef")
    return CompositionEntry(
        name="fdtd_E_update",
        steps=[CompositionStep(body=body, num_ins=4, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=0)],
        form="memref",
    )


def _fdtd_E_update_tensor() -> CompositionEntry:
    body = Term.Out(0) - (
        ((Term.In(0) - Term.In(1)) + Term.In(2)) - Term.In(3)
    ) * T_cap("%coef")
    return CompositionEntry(
        name="fdtd_E_update_tensor",
        steps=[CompositionStep(body=body, num_ins=4, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=0)],
        form="tensor",
    )


def _syr2k_composition() -> CompositionEntry:
    """C[j<=i] = β*C[j<=i] + α*(A*B^T + B*A^T)  (symmetric rank-2k update)."""
    s1 = CompositionStep(
        body=Term.Select(T_cap("%mask1"),
                         Term.Out(0) * T_cap("%beta"),
                         Term.Out(0)),
        num_ins=0, num_outs=1, parallel_dim_count=2, reduction_dim_count=0,
    )
    # Build the body in the same right-associative shape the encoder
    # produces: Out + (part1 + part2). Python's `+` is left-associative, so
    # without these parens we'd build (Out + part1) + part2 — structurally
    # different from the body even though mathematically equivalent.
    part1 = (T_cap("%alpha") * Term.In(0)) * Term.In(1)
    part2 = (T_cap("%alpha") * Term.In(2)) * Term.In(3)
    s2 = CompositionStep(
        body=Term.Select(T_cap("%mask2"),
                         Term.Out(0) + (part1 + part2),
                         Term.Out(0)),
        num_ins=4, num_outs=1, parallel_dim_count=2, reduction_dim_count=1,
    )
    return CompositionEntry(name="cublasDsyr2k", steps=[s1, s2])


def _copy_input() -> CompositionEntry:
    """out[i] = in[i]  — vector copy.

    Tagged memref-form because the canonical defn in kernel_library_phase2.mlir
    is authored for memref operands (used by fdtd-2d's source-injection step
    where a scalar memref broadcasts to a 1D output row). The tensor-form
    twin below handles the multi-root debufferize variant.
    """
    body = Term.In(0)
    return CompositionEntry(
        name="cublasDcopy",
        steps=[CompositionStep(body=body, num_ins=1, num_outs=1,
                                reduction_dim_count=0)],
        form="memref",
    )


def _copy_input_tensor() -> CompositionEntry:
    """Tensor-form variant of cublasDcopy — used by multi-root fdtd-2d's
    source-injection step."""
    body = Term.In(0)
    return CompositionEntry(
        name="cublasDcopy_tensor",
        steps=[CompositionStep(body=body, num_ins=1, num_outs=1,
                                reduction_dim_count=0)],
        form="tensor",
    )


def _axpby() -> CompositionEntry:
    """out = α*in0 + β*out  — gesummv combine step (cublasDaxpby)."""
    body = T_cap("%alpha") * Term.In(0) + T_cap("%beta") * Term.Out(0)
    return CompositionEntry(
        name="cublasDaxpby",
        steps=[CompositionStep(body=body, num_ins=1, num_outs=1,
                                reduction_dim_count=0)],
    )


def _fma3() -> CompositionEntry:
    """out = in0*in1 + in2  — fused-multiply-add over 3 inputs (adi solve step)."""
    body = Term.In(0) * Term.In(1) + Term.In(2)
    return CompositionEntry(
        name="elemwise_fma3",
        steps=[CompositionStep(body=body, num_ins=3, num_outs=1,
                                reduction_dim_count=0)],
    )


def _sub_from_out() -> CompositionEntry:
    """out -= in0  — vector-from-broadcast subtract (covariance centering)."""
    body = Term.Out(0) - Term.In(0)
    return CompositionEntry(
        name="elemwise_sub_from_out",
        steps=[CompositionStep(body=body, num_ins=1, num_outs=1,
                                reduction_dim_count=0)],
    )


def _rank_two_update() -> CompositionEntry:
    """A[i,j] += u1[i]*v1[j] + u2[i]*v2[j]  — gemver A-update stage.

    Could lower to cublasDger × 2 + sum, or stay as a fused kernel.
    """
    body = (Term.Out(0) + Term.In(0) * Term.In(1)
                       + Term.In(2) * Term.In(3))
    return CompositionEntry(
        name="cublasDger_rank2",
        steps=[CompositionStep(body=body, num_ins=4, num_outs=1,
                                parallel_dim_count=2, reduction_dim_count=0)],
    )


def composition_library() -> list[CompositionEntry]:
    """Order: longest compositions first; same-length ordered by specificity
    (more-captures first, more shape-constrained first)."""
    return [
        # Multi-step. Longest compositions first — the matcher is greedy
        # and otherwise a shorter composition would consume bodies the
        # longer one wanted.
        _cudnn_conv_bias_relu_add_fused(),  # 5-step: init + conv + bias + residual + relu
        _cublaslt_gemm_bias_relu_fused(),   # 4-step: init + gemm + bias + relu (cublasLt)
        _darknet_im2col_gemm_fused(),       # 3-step: zero + guarded im2col + sgemm
        _conv1x1_as_gemm_batched(),          # 2-step: init + 4par+1red contraction = 1x1 conv
        _cudnn_conv_bn_relu_fused(),  # 4-step: init + conv + bn-inplace + relu-inplace
        _gemm_composition(),
        _cudnn_conv2d_batched(),  # 2-step: init zero + 7-iter contraction (4 par + 3 red)
        _cudnn_maxpool_batched(), # 2-step: init -inf + 6-iter max-reduce (4 par + 2 red)
        _cudnn_batchnorm_inference(),  # 1-step: 5-in fused normalize+scale+bias (4 par)
        _cudnn_add_tensor_batched(),  # 1-step: Out + In(0) elementwise (4 par)

        # 1-step BLAS with α capture.
        _gemm_alpha_only(),
        _gemv_alpha_accumulate(),
        _axpby(),               # α*in + β*out  — most specific 2-cap form
        _axpy(),
        _scal_1d(),
        _scal_2d(),

        # Triangular / masked / specialty (must come before generic gemm/gemv).
        _syr2k_composition(),
        _syrk_composition(),
        _trmm_masked(),
        _rank_two_update(),
        _centered_sum_squares(),

        # Stencils (Bucket 2).
        _softmax_3step(),       # 3-step composition, max + exp+sum (multi-yield) + div.
        _softmax_3step_tensor(),
        _softmax_3step_out_tensor(),
                                #         Distinctive enough that ordering doesn't
                                #         matter against the rest, but list it
                                #         with the longer-step compositions.
        _rmsnorm_2step(),       # 2-step composition, sum-of-squares + weighted
                                #         scale; sits between softmax (3 steps)
                                #         and the conv shapes (single step) by
                                #         length so longest-first matching picks
                                #         the right one for shared prefixes.
        _conv3d_11pt_weighted(), # 11 ins, 3D parallel — most specific 3D
                                 #         conv shape; relies on egglog
                                 #         factoring to collapse redundant
                                 #         muls in polybench's conv3d body.
        _conv2d_9pt_weighted(), # 9 ins — most specific 2D conv shape; must
                                #         come before jacobi_2d_5pt (5 ins)
                                #         since both target 2D parallel iter.
        _heat_3d_7pt(),       # 7 ins
        _fdtd_E_update(),     # 4 ins
        _jacobi_2d_5pt(),     # 5 ins
        _jacobi_1d_3pt(),     # 3 ins
        _fdtd_update_2in(),   # 2 ins — checked AFTER more-specific 2D shapes

        # Stencils — tensor form (multi-root debufferize).
        _conv2d_9pt_weighted_tensor(),
        _heat_3d_7pt_tensor(),
        _fdtd_E_update_tensor(),
        _jacobi_2d_5pt_tensor(),
        _jacobi_1d_3pt_tensor(),
        _fdtd_update_2in_tensor(),
        _copy_input_tensor(),

        # 1-step BLAS, no α.
        _llama_rope_mulmul_sub_f32_tensor(),
        _llama_rope_mulmul_add_f32_tensor(),
        _llama_swiglu_f32_tensor(),
        _llama_mask_select_f32_tensor(),
        _llama_add_f32_tensor(),
        _gemv_accumulate(),
        _gemm_no_alpha(),
        _sgemm_broadcast3d_memref(),
        _dot(),
        _asum(),
        _reduce_sum_axis(),     # 1 in, 1 out, P=1+R=1: separate from gemv (2 ins)
        _vector_add_no_alpha(), # P=1+R=0
        _copy_input(),          # out = in0 (1 in, 1 out)
        _fma3(),                # in0*in1 + in2 (3 ins)
        _divf_scalar(),
        _subf_inputs(),
        _sub_from_out(),

        # Fill patterns.
        _fill_zero_1d(),
        _fill_zero_2d(),
        _fill_const_1d(),
        _fill_const_2d(),
    ]


def _term_repr(t) -> str:
    """Stable text repr of a Term (uses egglog's default __repr__)."""
    return str(t)


## NOTE: An egglog-driven normaliser (build EGraph, saturate, extract) was
## prototyped here. It worked correctly on small bodies (N ≤ ~10 summands)
## but timed out past 30s on polybenchGpu conv3d's 15-mul body due to
## exponential e-class growth from commutativity + associativity. The
## factoring rules are still registered in `algebra_rules()` for use by
## `equivalent()` (which operates on small canonical-template terms), but
## the body-normalisation hot path uses the Python tuple-AST factoring in
## `_factor_redundant_muls` below — linear time, predictable.


def _looks_like_float(s: str) -> bool:
    """True iff `s` parses as a Python float (used by `_parse_term` to
    distinguish float Lit values like `0.2` or `-1.5` from SSA / type
    tokens)."""
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_term(s: str):
    """Parse the string repr of a Term back into a Python AST (tuples).

    egglog stringifies expressions in a Lisp-y way like
        `Term.Out(0) + Term.Cap("%arg4")`
    We just want a structured tree for our own unification matcher, so
    we parse it as a stripped-down AST of (op, *children) tuples with
    leaves represented as ('In', i) / ('Out', i) / ('Cap', name) / ('Lit', v).
    """
    s = s.strip()
    if not s:
        return None

    def parse_expr(i: int):
        """Returns (node, next_index)."""
        # Skip whitespace
        while i < len(s) and s[i] == " ":
            i += 1
        # Match `Term.<Ctor>(...)` leaf forms.
        for ctor in ("In", "Out", "Cap", "Lit", "Sqrt", "Abs", "Exp", "Select", "Cmp"):
            tag = f"Term.{ctor}("
            if s[i:i+len(tag)] == tag:
                j, args = i + len(tag), []
                depth = 1
                arg_start = j
                # Parse comma-separated arguments respecting nested parens.
                while j < len(s) and depth > 0:
                    c = s[j]
                    if c == '(':
                        depth += 1
                    elif c == ')':
                        depth -= 1
                        if depth == 0:
                            arg = s[arg_start:j].strip()
                            if arg:
                                args.append(arg)
                            break
                    elif c == ',' and depth == 1:
                        arg = s[arg_start:j].strip()
                        if arg:
                            args.append(arg)
                        arg_start = j + 1
                    j += 1
                # Recursively parse each arg.
                parsed_args = []
                for a in args:
                    if a.startswith('"') and a.endswith('"'):
                        parsed_args.append(a[1:-1])
                    elif a.lstrip("-").isdigit():
                        parsed_args.append(int(a))
                    elif _looks_like_float(a):
                        parsed_args.append(float(a))
                    else:
                        sub, _ = parse_expr(0)
                        # If parse_expr fully consumed `a`, use it.
                        if sub is not None:
                            parsed_args.append(sub)
                        else:
                            parsed_args.append(a)
                node = (ctor, *parsed_args)
                return node, j + 1
        # Match a binary operator expression: <lhs> <op> <rhs>
        # The whole expression is parenthesized when nested, but the top
        # level isn't. We'll just handle the * and + operators here.
        # Find the top-level operator by scanning paren-depth = 0.
        depth = 0
        op_idx = -1
        op_char = None
        for j in range(i, len(s)):
            c = s[j]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif depth == 0 and c in "+-*/":
                # Prefer the LAST top-level operator (left-associative parse).
                op_idx = j
                op_char = c
        if op_idx >= 0:
            lhs_str = s[i:op_idx].strip()
            rhs_str = s[op_idx+1:].strip()
            lhs, _ = parse_expr_str(lhs_str)
            rhs, _ = parse_expr_str(rhs_str)
            op_name = {"+": "Add", "-": "Sub", "*": "Mul", "/": "Div"}[op_char]
            return (op_name, lhs, rhs), len(s)
        return None, i

    def parse_expr_str(t: str):
        # Strip wrapping parens.
        t = t.strip()
        while t.startswith('(') and t.endswith(')'):
            # Only strip if these parens match outermost.
            depth = 0
            ok = True
            for k, c in enumerate(t):
                if c == '(': depth += 1
                elif c == ')': depth -= 1
                if depth == 0 and k < len(t) - 1:
                    ok = False
                    break
            if ok:
                t = t[1:-1].strip()
            else:
                break
        # FIRST: try binary operator split at top level (paren depth 0).
        # Lowest precedence first.
        for op_chars in ("+-", "*/"):
            depth = 0
            op_idx = -1
            op_char = None
            for k, c in enumerate(t):
                if c == '(': depth += 1
                elif c == ')': depth -= 1
                elif depth == 0 and c in op_chars:
                    # Prefer the LAST top-level operator (so left-associative).
                    op_idx = k
                    op_char = c
            if op_idx >= 0:
                lhs, _ = parse_expr_str(t[:op_idx])
                rhs, _ = parse_expr_str(t[op_idx+1:])
                op_name = {"+": "Add", "-": "Sub", "*": "Mul", "/": "Div"}[op_char]
                return (op_name, lhs, rhs), len(t)
        # Otherwise try parsing as a Term.Ctor leaf.
        for ctor in ("In", "Out", "Cap", "Lit", "Sqrt", "Abs", "Exp", "Select", "Cmp"):
            tag = f"Term.{ctor}("
            if t.startswith(tag) and t.endswith(")"):
                inner = t[len(tag):-1]
                # Split args at top-level commas.
                args, depth, start = [], 0, 0
                for k, c in enumerate(inner):
                    if c == '(': depth += 1
                    elif c == ')': depth -= 1
                    elif c == ',' and depth == 0:
                        args.append(inner[start:k].strip())
                        start = k + 1
                args.append(inner[start:].strip())
                parsed_args = []
                for a in args:
                    if a.startswith('"') and a.endswith('"'):
                        parsed_args.append(a[1:-1])
                    elif a.lstrip("-").isdigit():
                        parsed_args.append(int(a))
                    elif _looks_like_float(a):
                        parsed_args.append(float(a))
                    else:
                        sub, _ = parse_expr_str(a)
                        parsed_args.append(sub)
                return (ctor, *parsed_args), len(t)
        return None, 0

    node, _ = parse_expr_str(s)
    return node


COMMUTATIVE_OPS = {"Add", "Mul"}


def _unify(body, template, bindings: dict) -> Optional[dict]:
    """Structural unification with commutativity. `template`'s Cap leaves
    are wildcards that bind to a Cap/Lit leaf in the body (i.e., a captured
    scalar — *not* a per-element tensor In/Out value).

    Returns updated bindings on success, None on failure.
    """
    if template is None or body is None:
        return None
    # Template Cap → wildcard, but only matches Cap/Lit body leaves
    # (captured outer scalars). Refuse to bind to per-element In(_)/Out(_)
    # so that axpy `out + alpha*x` doesn't spuriously match a gemv-shaped
    # body `out + a*b`.
    if isinstance(template, tuple) and template[0] == "Cap":
        if not (isinstance(body, tuple) and body[0] in ("Cap", "Lit")):
            return None
        name = template[1]
        if name in bindings:
            return bindings if bindings[name] == body else None
        bindings = dict(bindings)
        bindings[name] = body
        return bindings
    # Otherwise structural equality.
    if not (isinstance(template, tuple) and isinstance(body, tuple)):
        return bindings if template == body else None
    if template[0] != body[0]:
        return None
    if len(template) != len(body):
        return None
    # Leaf variants compare directly.
    if template[0] in {"In", "Out", "Lit"}:
        return bindings if template == body else None
    children_t = template[1:]
    children_b = body[1:]
    if template[0] in COMMUTATIVE_OPS and len(children_t) == 2:
        # Try both orderings.
        b1 = _unify(children_b[0], children_t[0], bindings)
        if b1 is not None:
            b1 = _unify(children_b[1], children_t[1], b1)
            if b1 is not None:
                return b1
        b2 = _unify(children_b[0], children_t[1], bindings)
        if b2 is not None:
            b2 = _unify(children_b[1], children_t[0], b2)
            if b2 is not None:
                return b2
        return None
    # Non-commutative: zip-recurse.
    for tc, bc in zip(children_t, children_b):
        bindings = _unify(bc, tc, bindings)
        if bindings is None:
            return None
    return bindings


def _flatten_addition_chain(node):
    """Walk down ('Add', l, r) nodes, return a flat list of leaf summands
    in source order.

    `((a + b) + c) + d` flattens to `[a, b, c, d]` regardless of bracketing.
    Uses a recursive walk to preserve source order naturally — a stack-based
    pre-order would visit rhs first and need reversing afterwards.
    """
    out: list = []
    def walk(n):
        if isinstance(n, tuple) and len(n) == 3 and n[0] == 'Add':
            walk(n[1])
            walk(n[2])
        else:
            out.append(n)
    walk(node)
    return out


def _try_factor_summand(s):
    """Recognise s as 'Lit(c) * X' or 'X * Lit(c)' for any X. Return (X, c)
    or None if s is not a factorable mul.
    """
    if not (isinstance(s, tuple) and len(s) == 3 and s[0] == 'Mul'):
        return None
    a, b = s[1], s[2]
    if isinstance(a, tuple) and a[0] == 'Lit' and isinstance(a[1], (int, float)):
        return (b, float(a[1]))
    if isinstance(b, tuple) and b[0] == 'Lit' and isinstance(b[1], (int, float)):
        return (a, float(b[1]))
    return None


def _factor_redundant_muls(ast):
    """Fold `c1*x + c2*x + ...` summands sharing a common factor x into
    `(c1+c2+...)*x`. Returns the rewritten tuple AST.

    Used by `body_matches_template` as a fallback when syntactic unification
    against a template fails. Specifically targets polybenchGpu's extracted
    conv3d body, which has 15 muls but only 11 unique input positions — the
    same input appears in multiple muls with different literal coefficients.

    Linear time in the number of summands; deterministic. Replaces an
    earlier egglog-driven attempt that blew up exponentially on bodies of
    this size — see the note above `body_matches_template`.
    """
    summands = _flatten_addition_chain(ast)
    if len(summands) < 2:
        return ast

    # Group factorable summands by their X subtree. `factor_groups` keys
    # are the X tuples (which are hashable since they're nested tuples of
    # hashable leaves). `insertion_order` preserves first-appearance order
    # so the rebuilt AST is deterministic.
    factor_groups: dict = {}
    insertion_order: list = []
    passthrough: list = []
    any_combined = False
    for s in summands:
        pair = _try_factor_summand(s)
        if pair is None:
            passthrough.append(s)
            continue
        X, coeff = pair
        if X not in factor_groups:
            factor_groups[X] = 0.0
            insertion_order.append(X)
        else:
            any_combined = True
        factor_groups[X] += coeff

    # Fast path: if no input was multiplied by more than one constant, no
    # combining happened — return the original AST unchanged. Avoids
    # gratuitously rewriting clean bodies (which would change the
    # bracketing and break downstream binding extraction).
    if not any_combined:
        return ast

    new_summands = [
        ('Mul', ('Lit', factor_groups[X]), X) for X in insertion_order
    ] + passthrough

    # Left-fold the list back into an Add tree.
    result = new_summands[0]
    for s in new_summands[1:]:
        result = ('Add', result, s)
    return result


def body_matches_template(body: Term, template: Term) -> Optional[dict]:
    """Check whether `body` matches `template`, with Cap names in the template
    as wildcards. Returns a binding dict on success, None on failure.

    First tries direct syntactic unification (with commutativity baked into
    `_unify`). If that fails, runs `_factor_redundant_muls` on the body AST
    — which collapses `c1*x + c2*x + ...` patterns into one mul per unique
    input — and retries. This is what lets polybenchGpu's conv3d body
    (15 muls, 11 unique inputs) match the `_conv3d_11pt_weighted` template.
    """
    tmpl_ast = _parse_term(_term_repr(template))
    body_ast = _parse_term(_term_repr(body))
    direct = _unify(body_ast, tmpl_ast, {})
    if direct is not None:
        return direct
    factored = _factor_redundant_muls(body_ast)
    if factored is body_ast:
        return None  # nothing to fold; second attempt would be identical
    return _unify(factored, tmpl_ast, {})


def _is_guarded_im2col_body(g: GenericBody) -> bool:
    """Return true for the raised Darknet im2col workspace-fill body.

    This intentionally checks structural markers rather than exact SSA names:
    the scalar Term encoder cannot model the scf.if/memref.load payload, but
    the surrounding composition and launch rewriter recover the actual operands
    from the matched body text.
    """
    if len(g.ins_arg_names) != 0 or len(g.outs_arg_names) != 1:
        return False
    if sum(1 for it in g.iterator_types if it == "parallel") != 3:
        return False
    if any(it == "reduction" for it in g.iterator_types):
        return False
    body = "\n".join(g.body_lines)
    required = [
        "linalg.index 0",
        "linalg.index 1",
        "linalg.index 2",
        "scf.if",
        "memref.load",
        "arith.cmpi slt",
        "arith.cmpi sge",
        "arith.select",
        "scf.yield",
    ]
    if not all(tok in body for tok in required):
        return False
    # The im2col linearization decomposes the workspace row with div/rem by
    # the kernel size and computes the padded input coordinates from stride
    # and pad. These checks keep the predicate from firing on arbitrary
    # guarded loads.
    return ("arith.remsi" in body and "arith.divsi" in body and
            body.count("scf.yield") >= 2)


def match_composition(
    body_objs: list[GenericBody],
    body_terms: list[Term],
    compositions: list[CompositionEntry],
    start: int = 0,
    body_forms: list[str] | None = None,
) -> Optional[tuple[CompositionEntry, int, dict]]:
    """If a contiguous run of generics starting at index `start` matches a
    composition's full sequence (body + shape gates), return (entry,
    start, bindings). Otherwise None.

    Greedy: tries longest compositions first.

    `body_forms` (optional): per-body "tensor" / "memref" tag. If given, an
    entry only fires when every step's form is compatible (entry.form ==
    body_form, or entry.form == "any"). Keeps the matcher from picking a
    tensor-only library entry for a memref-form body (which would later
    fail in --lower-kernel-launch with a type mismatch).
    """
    for entry in compositions:
        n = len(entry.steps)
        if start + n > len(body_objs):
            continue
        if body_forms is not None and entry.form != "any":
            forms_in_run = body_forms[start : start + n]
            if any(f != entry.form for f in forms_in_run):
                continue
        merged: dict = {}
        ok = True
        for j in range(n):
            step = entry.steps[j]
            g = body_objs[start + j]
            # Shape gates.
            if step.num_ins is not None and step.num_ins != len(g.ins_arg_names):
                ok = False
                break
            if step.num_outs is not None and step.num_outs != len(g.outs_arg_names):
                ok = False
                break
            if step.reduction_dim_count is not None:
                red = sum(1 for it in g.iterator_types if it == "reduction")
                if red != step.reduction_dim_count:
                    ok = False
                    break
            if step.parallel_dim_count is not None:
                par = sum(1 for it in g.iterator_types if it == "parallel")
                if par != step.parallel_dim_count:
                    ok = False
                    break
            # Body match. Two modes:
            #   * Single-yield (the common case): step.body is a single Term;
            #     body_terms[i] is a single Term; one unify call.
            #   * Multi-yield (softmax-style fused exp+sum, etc.): step.body_per_yield
            #     is a list of Terms — one per yield position; the body's
            #     yield Terms come from encode_body_yields stored in
            #     body_yields[i]. We unify each (body_yield, template_yield) pair
            #     and merge bindings.
            if step.special is not None:
                if step.special == "guarded_im2col":
                    if not _is_guarded_im2col_body(g):
                        ok = False
                        break
                    b = {}
                else:
                    ok = False
                    break
            elif step.body_per_yield is not None:
                body_yields_here = body_objs[start + j].__dict__.get(
                    "_yield_terms_cache"
                )
                if body_yields_here is None:
                    body_yields_here = encode_body_yields(body_objs[start + j])
                    body_objs[start + j]._yield_terms_cache = body_yields_here
                if len(body_yields_here) != len(step.body_per_yield):
                    ok = False; break
                step_bindings: dict = {}
                step_ok = True
                for body_t, tmpl_t in zip(body_yields_here, step.body_per_yield):
                    bm = body_matches_template(body_t, tmpl_t)
                    if bm is None:
                        step_ok = False; break
                    for k, v in bm.items():
                        if k in step_bindings and step_bindings[k] != v:
                            step_ok = False; break
                        step_bindings[k] = v
                    if not step_ok:
                        break
                if not step_ok:
                    ok = False; break
                b = step_bindings
            else:
                b = body_matches_template(body_terms[start + j], step.body)
                if b is None:
                    ok = False
                    break
            for k, v in b.items():
                if k in merged and merged[k] != v:
                    ok = False
                    break
                merged[k] = v
            if not ok:
                break
        if ok:
            return entry, start, merged
    return None


# ---------------------------------------------------------------------------
# Original single-body matcher.
# ---------------------------------------------------------------------------

def match(t: Term, entries: list[LibraryEntry],
          want_ins: int, want_outs: int,
          want_maps: list[str], want_iters: list[str]) -> Optional[LibraryEntry]:
    """Match a body Term against the library; return the first matching entry."""
    for e in entries:
        if e.num_ins != want_ins or e.num_outs != want_outs:
            continue
        if e.indexing_maps != want_maps or e.iterator_types != want_iters:
            continue
        if equivalent(e.canonical_body, t):
            return e
    return None


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("usage: kernel_match.py <debuf_dir> [test_kernel.mlir]")
        sys.exit(1)

    root = Path(sys.argv[1])
    print(f"Building library from {root}...")
    lib = build_library_from_dir(root)
    print(f"Library has {len(lib)} unique entries.")
    counts: dict[str, int] = {}
    for e in lib:
        counts[e.source_kernel] = counts.get(e.source_kernel, 0) + 1
    print("Entries per source kernel:")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")

    if len(sys.argv) >= 3:
        # Match every generic in the test file against the library.
        text = Path(sys.argv[2]).read_text()
        gens = parse_generics(text)
        print(f"\nTesting {sys.argv[2]} ({len(gens)} generics):")
        for i, g in enumerate(gens):
            t = encode_body(g)
            hit = match(t, lib, len(g.ins_arg_names), len(g.outs_arg_names),
                        g.indexing_maps, g.iterator_types)
            label = hit.name if hit else "NO_MATCH"
            print(f"  generic #{i} -> {label}")


if __name__ == "__main__":
    main()
