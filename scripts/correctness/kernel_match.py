#!/usr/bin/env python3
"""linalg.generic body matcher using egglog.

This is an iterative prototype of the "match raised linalg to a kernel
library" idea, in three layers:

  1. Regex-based parser for linalg.generic bodies (good enough for the
     debuferized PolyBench output — every body is ~6 lines of arith + yield).
  2. Encoder: linalg-body -> egglog Expr.
  3. Matcher: saturate with algebra rules, then check equivalence between
     a user body and each library pattern.

Production templates come exclusively from imported library reference MLIR.
Scalar equivalence alone never authorizes a library replacement.
"""
from __future__ import annotations
import math
import os
import re
import sys
import json
import time
from collections import Counter
from itertools import product
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from egglog import EGraph, Expr, StringLike, f64, f64Like, i64Like, rewrite, ruleset, vars_


_MATCHER_MODE = "egglog"
_COLLECT_EGRAPH_SIZES = False
_MATCHER_TELEMETRY: dict[str, object] = {}


def configure_matcher(mode: str = "egglog", *, collect_egraph_sizes: bool = False) -> None:
    """Select the scalar acceptance engine and reset per-process telemetry."""
    if mode not in {"egglog", "syntactic"}:
        raise ValueError(f"unknown matcher mode: {mode}")
    global _MATCHER_MODE, _COLLECT_EGRAPH_SIZES, _MATCHER_TELEMETRY
    _MATCHER_MODE = mode
    _COLLECT_EGRAPH_SIZES = collect_egraph_sizes
    _MATCHER_TELEMETRY = {
        "mode": mode,
        "iteration_limit": 8 if mode == "egglog" else 0,
        "proofs_attempted": 0,
        "proofs_matched": 0,
        "proof_elapsed_ms": 0.0,
        "proof_elapsed_max_ms": 0.0,
        "telemetry_elapsed_ms": 0.0,
        "egraph_nodes_sum": 0,
        "egraph_nodes_max": 0,
        "egraph_classes_sum": 0,
        "egraph_classes_max": 0,
        "rule_matches": 0,
        "proofs_over_10s": 0,
        "template_attempts": 0,
        "syntactic_unifications": 0,
        "syntactic_comparisons": 0,
    }


def matcher_telemetry() -> dict[str, object]:
    return dict(_MATCHER_TELEMETRY)


configure_matcher()


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
    @classmethod
    def NegativeZero(cls) -> Term: ...
    @classmethod
    def Access(cls, root: StringLike, mapping: StringLike) -> Term: ...

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
    def Tanh(cls, a: Term) -> Term: ...
    @classmethod
    def Unary(cls, op: StringLike, a: Term) -> Term: ...
    @classmethod
    def Binary(cls, op: StringLike, a: Term, b: Term) -> Term: ...
    @classmethod
    def Select(cls, pred: Term, t: Term, f: Term) -> Term: ...
    @classmethod
    def Cmp(cls, kind: StringLike, a: Term, b: Term) -> Term: ...


# ---------------------------------------------------------------------------
# Algebra rules (cosmetic variations).
# ---------------------------------------------------------------------------

a, b, c, d = vars_("a b c d", Term)


def literal(value: float) -> Term:
    """Build a scalar literal without losing -0 in egglog's primitive cache."""
    value = float(value)
    if value == 0.0 and math.copysign(1.0, value) < 0:
        return Term.NegativeZero()
    return Term.Lit(value)


def numeric_model() -> str:
    model = os.environ.get("POLYGEIST_NUMERIC_MODEL", "strict")
    if model not in {"strict", "relaxed"}:
        raise ValueError(f"unknown numerical model: {model}")
    return model


def exact_integer_selection_rules():
    """Integer order is total; selecting either equal operand has equal bits.

    Both operands may be promoted with the same injective integer extension.
    Floating comparisons and narrowing/noninjective conversions are excluded.
    These scalar identities apply to every imported/application expression.
    """
    rules = []
    casts = [None] + [
        f"arith.{op}:i{src} to i{dst}"
        for op in ("extsi", "extui")
        for src in (8, 16, 32) for dst in (16, 32, 64) if src < dst
    ]
    for cast in casts:
        x, y = ((a, b) if cast is None else
                (Term.Unary(cast, a), Term.Unary(cast, b)))
        for predicate in ("slt", "sle", "ult", "ule"):
            rules.append(rewrite(Term.Select(Term.Cmp(predicate, x, y), a, b))
                         .to(Term.Select(Term.Cmp(predicate, y, x), b, a)))
            rules.append(rewrite(Term.Select(Term.Cmp(predicate, x, y), b, a))
                         .to(Term.Select(Term.Cmp(predicate, y, x), a, b)))
    return rules


def exact_conditional_selection_rules():
    """Move a known-false comparison through a selected literal arm.

    The unchanged literal arm is returned on either side when its predicate
    is true. Otherwise both sides perform the same ordered comparison and
    selection. No floating arithmetic, NaN propagation, or zero sign changes.
    Guards use the actual compared literals, not assumptions about inputs.
    These bounded identities avoid general compare/select distribution.
    """
    predicate, value, alternative = vars_("predicate value alternative", Term)
    literal, bound = vars_("literal bound", f64)
    selected = Term.Select(predicate, Term.Lit(literal), value)
    rules = []
    for kind in ("olt", "ole"):
        left_false = literal >= bound if kind == "olt" else literal > bound
        right_false = bound >= literal if kind == "olt" else bound > literal
        rules.append(rewrite(Term.Select(
            Term.Cmp(kind, selected, Term.Lit(bound)), alternative, selected))
            .to(Term.Select(predicate, Term.Lit(literal), Term.Select(
                Term.Cmp(kind, value, Term.Lit(bound)), alternative, value)),
                left_false))
        rules.append(rewrite(Term.Select(
            Term.Cmp(kind, Term.Lit(bound), selected), alternative, selected))
            .to(Term.Select(predicate, Term.Lit(literal), Term.Select(
                Term.Cmp(kind, Term.Lit(bound), value), alternative, value)),
                right_false))
    return rules


def algebra_rules(include_distributivity: bool = True):
    # Strict mode admits exact selection identities and multiplication operand
    # exchange, matching upstream arith.mulf/muli's Commutative trait. This does
    # not change precision, parenthesization, signed-zero behavior or NaN class.
    # NaN payload identity is not part of the IR/backend numerical contract.
    # Relaxed mode also permits real-arithmetic identities; runtime error still
    # needs validation.
    integer_rules = (exact_integer_selection_rules() +
                     exact_conditional_selection_rules())
    if numeric_model() == "strict":
        return ruleset(*integer_rules, rewrite(a * b).to(b * a),
                       name="strict_scalar_operations")
    one = Term.Lit(1.0)
    zero = Term.Lit(0.0)
    # Numeric literal variables — required for the factoring + folding rules
    # below, where the RHS computes c1+c2 / c1*c2 via egglog's built-in f64
    # arithmetic on the captured constants. `vars_` returns a generator, so
    # single-name calls need tuple-unpack syntax.
    (x,) = vars_("x", Term)
    c1, c2 = vars_("c1 c2", f64)
    rules = [
        *integer_rules,
        # Commutativity
        rewrite(a + b).to(b + a),
        rewrite(a * b).to(b * a),
        # Associativity
        rewrite(a + (b + c)).to((a + b) + c),
        rewrite((a + b) + c).to(a + (b + c)),
        rewrite(a * (b * c)).to((a * b) * c),
        rewrite((a * b) * c).to(a * (b * c)),
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
    ]
    if include_distributivity:
        # These expanding rules are valuable in the isolated audit. Production
        # disables them because a negative proof can grow exponentially before
        # the fixed iteration budget is reached.
        rules.extend([
            rewrite(a * (b + c)).to((a * b) + (a * c)),
            rewrite((a + b) * c).to((a * c) + (b * c)),
        ])
    return ruleset(*rules)


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
    # Result SSA names for this linalg.generic. Multi-result ops use MLIR's
    # `%r:2 = ...` spelling and expose `%r#0`, `%r#1`, ... as users. This is
    # needed for scalar-chain checks such as `%inv_sum = 1.0 / %softmax_sum`.
    result_names: list[str] = None  # type: ignore[assignment]
    # Scalar types of the linalg block arguments, aligned with the complete
    # block-argument list (inputs followed by outputs).  This is more reliable
    # type evidence than `body_lines`: a constant-only initializer commonly
    # has an empty body after the parser removes `linalg.yield`, even though
    # its output block argument still establishes the element type.
    block_arg_types: list[str] = None  # type: ignore[assignment]

    @property
    def yield_value(self) -> str:
        """Back-compat alias for callers written before multi-yield support
        — returns the first yield's SSA name. New code should iterate
        `yield_values` directly."""
        return self.yield_values[0] if self.yield_values else ""


_GEN_RE = re.compile(
    r"(?:(%[\w_\-]+)(?::(\d+))?\s*=\s*)?"
    r"linalg\.generic\s*\{[^}]*indexing_maps\s*=\s*\[([^\]]*)\][^}]*"
    r"iterator_types\s*=\s*\[([^\]]*)\][^}]*\}[^\^]*?"
    # Yield captures one OR MORE comma-separated SSA names. Multi-yield
    # bodies (e.g. softmax's fused exp+sum) write to multiple outs in one
    # op. Single-yield bodies still match unchanged — the (?:...)*
    # group is zero-or-more. The capture is the full operand list as a
    # single string; parse_generics splits on commas to produce the
    # GenericBody.yield_values list.
    r"\^bb0\(([^)]*)\)\s*:\s*(.*?)\s*"
    r"linalg\.yield\s+(%[\w$.\-]+(?:\s*,\s*%[\w$.\-]+)*)\s*:",
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
            if value == "true":
                out[name] = 1.0
            elif value == "false":
                out[name] = 0.0
            # Other non-numeric values (e.g. undef) remain opaque.
    # MLIR elides the `: i1` annotation for canonical boolean constants.
    for name, value in re.findall(
            r"(%[\w_\-]+)\s*=\s*arith\.constant\s+(true|false)\s*(?:$|\n)",
            mlir_text, re.MULTILINE):
        out[name] = 1.0 if value == "true" else 0.0
    return out



_MAP_ALIAS_RE = re.compile(
    # affine_map text contains `->` which has a `>`, so [^>] is wrong here.
    # Match the literal form `affine_map<(...) -> (...)>`.
    r"^\s*(#[A-Za-z_][\w.$-]*)\s*=\s*"
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
                   constants: dict[str, float] | None = None,
                   infer_outputs_from_yield: bool = False) -> list[GenericBody]:
    """Extract every linalg.generic with its body."""
    if constants is None:
        constants = parse_constants(mlir_text)
    mlir_text = _resolve_map_aliases(mlir_text)
    results = []
    for m in _GEN_RE.finditer(mlir_text):
        # SSA names are function-local.  Whole-module parsing used to let a
        # later function's `%cst` overwrite the value of an earlier function's
        # `%cst`, which could make Egglog prove the wrong scalar expression.
        # Only constants and scalar definitions in the enclosing function's
        # dominating textual prefix are visible to this generic.
        function_start = mlir_text.rfind("func.func", 0, m.start())
        scope_start = function_start if function_start >= 0 else 0
        dominating_prefix = mlir_text[scope_start:m.start()]
        scoped_constants = parse_constants(dominating_prefix)
        if not scoped_constants and function_start < 0:
            scoped_constants = constants
        result_base, result_count, maps_str, iters_str, args_str, body_str, yield_operands_str = m.groups()
        if result_base and result_count:
            result_names = [
                f"{result_base}#{i}" for i in range(int(result_count))
            ]
        elif result_base:
            result_names = [result_base]
        else:
            result_names = []
        # Split the yield's operand list on commas (multi-yield bodies have
        # multiple SSAs separated by commas). The regex preserves whitespace
        # around commas, so strip per-token.
        yield_names = [s.strip() for s in yield_operands_str.split(",") if s.strip()]
        # Back-compat for the rest of the local scope: yield_name refers to
        # the FIRST yield. Most local logic (capture detection, etc.) was
        # written assuming a single yield value — keeping it correct for
        # the single-yield case AND for the first slot of multi-yield bodies.
        yield_name = yield_names[0] if yield_names else ""

        block_args = []
        block_arg_types = []
        for piece in args_str.split(","):
            piece = piece.strip()
            if not piece:
                continue
            name, _, arg_type = piece.partition(":")
            name = name.strip()
            block_args.append(name)
            block_arg_types.append(arg_type.strip())
        if infer_outputs_from_yield:
            # Canonical library bodies may name output block arguments `%z`,
            # `%ov`, etc. Their trailing block arguments correspond exactly
            # to the yielded values.
            out_count = len(yield_names)
            ins = block_args[:-out_count] if out_count else block_args
            outs = block_args[-out_count:] if out_count else []
        else:
            # Preserve the raised-corpus convention used by existing matcher
            # patterns. Moving arbitrary input IR to arity-based parsing is a
            # separate migration because several legacy patterns encode this
            # historical `%out` naming contract.
            ins = [name for name in block_args if not name.startswith("%out")]
            outs = [name for name in block_args if name.startswith("%out")]

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
            assigned = re.match(r"(%[\w_\-]+)\s*=", ln)
            if assigned:
                local_defs.add(assigned.group(1))
        for ln in body_lines:
            # Find all %xxx references on the rhs.
            for tok in re.findall(r"%[\w_\-]+", ln):
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

        results.append(GenericBody(
            ins_arg_names=ins,
            outs_arg_names=outs,
            body_lines=body_lines,
            yield_values=yield_names,
            captures=captures,
            indexing_maps=maps,
            iterator_types=iters,
            constants={
                name: scoped_constants[name]
                for name in captures
                if name in scoped_constants
            },
            result_names=result_names,
            block_arg_types=block_arg_types,
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
    # Integer arithmetic retains its opcode and width; real-field rules
    # must never prove division, overflow, or signedness equivalence for it.
    "arith.muli": "integer",
    "arith.addi": "integer",
    "arith.subi": "integer",
    "arith.divsi": "integer",
    "arith.divui": "integer",
    "math.sqrt": "sqrt",
    "math.absf": "abs",
    "math.absi": "abs",
    # Transcendentals — used by softmax (exp), gelu (tanh), crossentropy (log).
    # Encoded as opaque unary Terms; templates can match against `Term.Exp(x)`
    # etc. so the matcher recognises the kernel without trying to fold them.
    "math.exp": "exp",
    "math.tanh": "tanh",
    "math.acos": "unary_acos",
    "math.acosh": "unary_acosh",
    "math.asin": "unary_asin",
    "math.asinh": "unary_asinh",
    "math.atan": "unary_atan",
    "math.atanh": "unary_atanh",
    "math.ceil": "unary_ceil",
    "math.cos": "unary_cos",
    "math.cosh": "unary_cosh",
    "math.floor": "unary_floor",
    "math.erf": "unary_erf",
    "math.exp2": "unary_exp2",
    "math.expm1": "unary_expm1",
    "math.log": "unary_log",
    "math.log1p": "unary_log1p",
    "math.log2": "unary_log2",
    "math.log10": "unary_log10",
    "math.round": "unary_round",
    "math.roundeven": "unary_roundeven",
    "math.trunc": "unary_trunc",
    "math.powf": "binary_pow",
    "math.atan2": "binary_atan2",
    "math.sin": "unary_sin",
    "math.sinh": "unary_sinh",
    "math.tan": "unary_tan",
    "func.call": "call",
    "arith.cmpf": "cmpf",
    "arith.cmpi": "cmpi",
    "arith.xori": "binary_xor",
    "arith.andi": "binary_and",
    "arith.ori": "binary_or",
    "arith.shli": "binary_shl",
    "arith.shrsi": "binary_shrs",
    "arith.select": "select",
    # Casts retain both source and destination types in an uninterpreted
    # node. Equal casts may match; deleting rounding or conversion may not.
    **{op: "cast" for op in (
        "arith.extsi", "arith.extui", "arith.trunci", "arith.sitofp",
        "arith.uitofp", "arith.fptosi", "arith.fptoui", "arith.extf",
        "arith.truncf", "arith.bitcast", "arith.index_cast",
        "arith.index_castui")},
}


def encode_body(g: GenericBody) -> Term:
    return encode_body_yields(g)[0]


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
    env: dict[str, Term] = {}
    for i, name in enumerate(g.ins_arg_names):
        env[name] = Term.In(i)
    for i, name in enumerate(g.outs_arg_names):
        env[name] = Term.Out(i)
    for cap in g.captures:
        if cap in g.constants:
            env[cap] = literal(g.constants[cap])
        else:
            env[cap] = Term.Cap(cap)

    def lookup(name: str) -> Term:
        if name in env:
            return env[name]
        if name in g.constants:
            env[name] = literal(g.constants[name])
        else:
            env[name] = Term.Cap(name)
        return env[name]

    for line in g.body_lines:
        m = re.match(
            r"(%[\w_]+)\s*=\s*(\w+\.\w+)\s+(.*?)\s*:\s*\S+", line.strip()
        )
        if not m:
            raise ValueError(f"unsupported scalar operation: {line.strip()}")
        result, op, args_part = m.group(1), m.group(2), m.group(3)
        arg_toks = [s.strip() for s in args_part.split(",")]

        def resolve(tok: str) -> Term:
            tok = tok.strip()
            if tok.startswith("%"):
                return lookup(tok)
            try:
                return literal(float(tok))
            except ValueError:
                return Term.Lit(float("nan"))

        op_key = _OP_PATTERNS.get(op, op)
        if op_key == "cast":
            signature = re.sub(r"\s+", " ", line.rsplit(":", 1)[1].strip())
            env[result] = Term.Unary(op + ":" + signature, resolve(arg_toks[0]))
            continue
        if op_key == "integer":
            signature = line.rsplit(":", 1)[1].strip()
            env[result] = Term.Binary(op + ":" + signature,
                                      resolve(arg_toks[0]), resolve(arg_toks[1]))
            continue
        if op == "arith.constant":
            try:
                value = float(args_part)
            except ValueError as error:
                raise ValueError(f"unsupported literal: {args_part}") from error
            if not math.isfinite(value):
                raise ValueError("non-finite scalar literal")
            if abs(value) >= 2 ** 53 and line.rsplit(":", 1)[1].strip().startswith("i"):
                raise ValueError("integer literal exceeds exact scalar encoding")
            env[result] = literal(value)
        elif op_key == "mul":
            env[result] = resolve(arg_toks[0]) * resolve(arg_toks[1])
        elif op_key == "add":
            env[result] = resolve(arg_toks[0]) + resolve(arg_toks[1])
        elif op_key == "sub":
            env[result] = resolve(arg_toks[0]) - resolve(arg_toks[1])
        elif op_key == "neg":
            env[result] = Term.Unary("negf", resolve(arg_toks[0]))
        elif op_key == "div":
            env[result] = resolve(arg_toks[0]) / resolve(arg_toks[1])
        elif op_key == "sqrt":
            env[result] = Term.Sqrt(resolve(arg_toks[0]))
        elif op_key == "abs":
            env[result] = Term.Abs(resolve(arg_toks[0]))
        elif op_key == "exp":
            env[result] = Term.Exp(resolve(arg_toks[0]))
        elif op_key == "tanh":
            env[result] = Term.Tanh(resolve(arg_toks[0]))
        elif op_key.startswith("unary_"):
            env[result] = Term.Unary(op_key.removeprefix("unary_"),
                                     resolve(arg_toks[0]))
        elif op_key.startswith("binary_"):
            env[result] = Term.Binary(op_key.removeprefix("binary_"),
                                      resolve(arg_toks[0]),
                                      resolve(arg_toks[1]))
        elif op_key == "call":
            call = re.match(
                r"@([\w.$-]+)\((%[\w_-]+)(?:,\s*(%[\w_-]+))?\)",
                args_part.strip())
            if call:
                callee = re.sub(r"f$", "", call.group(1))
                supported = {
                    "acos", "acosh", "asin", "asinh", "atan", "atanh",
                    "ceil", "cos", "cosh", "exp", "floor", "log",
                    "erf", "erfc", "exp2", "expm1", "log1p", "log2", "log10",
                    "round", "roundeven", "trunc",
                    "sin", "sinh", "sqrt", "tan", "tanh",
                }
                binary = {"fmax": "max", "fmin": "min", "fmod": "mod",
                          "hypot": "hypot", "remainder": "remainder",
                          "pow": "pow"}
                if call.group(3) and callee in binary:
                    env[result] = Term.Binary(
                        binary[callee], resolve(call.group(2)),
                        resolve(call.group(3)))
                else:
                    if callee not in supported:
                        raise ValueError(f"unsupported scalar call: {callee}")
                    # Retain the exact symbol, including float/double suffix.
                    env[result] = Term.Unary("call:" + call.group(1), resolve(call.group(2)))
            else:
                raise ValueError(f"unsupported scalar operation: {line.strip()}")
        elif op_key == "select":
            condition, yes, no = map(resolve, arg_toks)
            if line.rsplit(":", 1)[1].strip() == "i1":
                # The typed i1 select has exact Boolean semantics. Normalize
                # its constant arms just as upstream Arith canonicalization
                # does, without simplifying floating selects or predicates.
                def bool_literal(token):
                    value = g.constants.get(token)
                    return value if value in (0.0, 1.0) else None
                yes_constant, no_constant = map(bool_literal, arg_toks[1:])
                if no_constant == 0.0:
                    env[result] = Term.Binary("and", condition, yes)
                elif yes_constant == 1.0:
                    env[result] = Term.Binary("or", condition, no)
                else:
                    env[result] = Term.Select(condition, yes, no)
            else:
                env[result] = Term.Select(condition, yes, no)
        elif op_key in ("cmpf", "cmpi"):
            kind = arg_toks[0].strip()
            if " " in kind:
                kind, lhs_tok = kind.split(None, 1)
                rhs_tok = arg_toks[1]
            elif len(arg_toks) >= 3:
                lhs_tok, rhs_tok = arg_toks[1], arg_toks[2]
            else:
                raise ValueError(f"malformed comparison: {line.strip()}")
            # Reversing both operands and the direction is exact, including
            # ordered/unordered floating comparisons and signed integers.
            reversed_kind = {"ogt": "olt", "oge": "ole", "ugt": "ult",
                             "uge": "ule", "sgt": "slt", "sge": "sle"}
            if kind in reversed_kind:
                kind, lhs_tok, rhs_tok = reversed_kind[kind], rhs_tok, lhs_tok
            env[result] = Term.Cmp(kind, resolve(lhs_tok), resolve(rhs_tok))
        else:
            raise ValueError(f"unsupported scalar operation: {line.strip()}")

    return [lookup(yv) for yv in g.yield_values]


# ---------------------------------------------------------------------------
# Library + matcher.
# ---------------------------------------------------------------------------

def _literal_identity(node):
    """Keep IEEE zero signs in otherwise structural scalar comparisons."""
    if not isinstance(node, tuple):
        return node
    if node[0] == "Lit":
        return ("LiteralBits", float(node[1]).hex())
    return tuple(_literal_identity(child) for child in node)


def _has_negative_zero(node) -> bool:
    if not isinstance(node, tuple):
        return False
    if node[0] == "Lit":
        return node[1] == 0 and math.copysign(1.0, node[1]) < 0
    return any(_has_negative_zero(child) for child in node[1:])


def equivalent(a: Term, b: Term, include_distributivity: bool = True) -> bool:
    """Are two Terms equivalent under the current algebra rules?"""
    left = _parse_term(_term_repr(a))
    right = _parse_term(_term_repr(b))
    # Raw f64 primitive atoms equate both zero signs. Our negative-zero atom
    # is distinct; keep its strict proofs structural until algebraic rules
    # for that atom have been reviewed explicitly.
    strict_zero = (numeric_model() == "strict" and
                   (_has_negative_zero(left) or _has_negative_zero(right)))
    if _MATCHER_MODE == "syntactic" or strict_zero:
        started = time.perf_counter()
        matched = _literal_identity(left) == _literal_identity(right)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _MATCHER_TELEMETRY["proof_elapsed_ms"] += elapsed_ms
        _MATCHER_TELEMETRY["proof_elapsed_max_ms"] = max(
            _MATCHER_TELEMETRY["proof_elapsed_max_ms"], elapsed_ms)
        if _MATCHER_MODE == "syntactic":
            _MATCHER_TELEMETRY["syntactic_comparisons"] += 1
        if matched:
            _MATCHER_TELEMETRY["proofs_matched"] += 1
        return matched

    _MATCHER_TELEMETRY["proofs_attempted"] += 1
    started = time.perf_counter()
    eg = EGraph()
    eg.register(a, b)
    report = eg.run(algebra_rules(include_distributivity) * 8)
    try:
        eg.check(a == b)
        matched = True
    except Exception:
        matched = False
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    _MATCHER_TELEMETRY["proof_elapsed_ms"] += elapsed_ms
    _MATCHER_TELEMETRY["proof_elapsed_max_ms"] = max(
        _MATCHER_TELEMETRY["proof_elapsed_max_ms"], elapsed_ms)
    _MATCHER_TELEMETRY["rule_matches"] += sum(
        report.num_matches_per_rule.values())
    if elapsed_ms > 10_000.0:
        _MATCHER_TELEMETRY["proofs_over_10s"] += 1
    if matched:
        _MATCHER_TELEMETRY["proofs_matched"] += 1

    if _COLLECT_EGRAPH_SIZES:
        telemetry_started = time.perf_counter()
        function_nodes = sum(size for _, size in eg.all_function_sizes())
        serialized = json.loads(eg._serialize(
            split_primitive_outputs=False,
            include_temporary_functions=True,
        ).to_json())
        node_count = len(serialized.get("nodes", {}))
        class_count = len(serialized.get("class_data", {}))
        # all_function_sizes excludes primitive leaves, whereas serialization
        # includes them. Retain the serialized node count for a complete size
        # and expose the function-only count as a cross-check.
        _MATCHER_TELEMETRY.setdefault("egraph_function_nodes_sum", 0)
        _MATCHER_TELEMETRY["egraph_function_nodes_sum"] += function_nodes
        _MATCHER_TELEMETRY["egraph_nodes_sum"] += node_count
        _MATCHER_TELEMETRY["egraph_nodes_max"] = max(
            _MATCHER_TELEMETRY["egraph_nodes_max"], node_count)
        _MATCHER_TELEMETRY["egraph_classes_sum"] += class_count
        _MATCHER_TELEMETRY["egraph_classes_max"] = max(
            _MATCHER_TELEMETRY["egraph_classes_max"], class_count)
        _MATCHER_TELEMETRY["telemetry_elapsed_ms"] += (
            time.perf_counter() - telemetry_started) * 1000.0
    return matched


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
    # Optional scalar element type required in every matched generic body.
    # This prevents identical algebraic bodies from selecting a library ABI
    # of the wrong precision (for example f32 dot -> cublasDdot).
    element_type: Optional[str] = None


@dataclass
class SemanticCandidate:
    """One possible semantic interpretation of a raised linalg body.

    This is intentionally separate from ABI lowering. A candidate may denote:
      - an exact whole-body/composition match already present in
        `composition_library()`;
      - a semantic node recognized inside a scalar subexpression; or
      - a specialization/completion of one semantic node into another, e.g.
        a 7-point sparse stencil completed into a 3x3x3 convolution by filling
        missing taps with zero.

    The rewriter can still pick the old greedy ABI-lowerable path, while dry
    runs can expose the full candidate set for planning/cost-model work.
    """
    name: str
    body_indices: tuple[int, ...]
    match_kind: str          # "whole" | "composition" | "subterm" | "completion"
    coverage: str            # "whole" | "partial"
    bindings: dict
    entry: Optional[CompositionEntry] = None
    defaults: tuple[tuple[str, str], ...] = ()
    subterm_path: tuple[int, ...] = ()
    source: Optional[str] = None


def composition_library() -> list[CompositionEntry]:
    """Load scalar templates exclusively from imported MLIR definitions.

    These candidates still require whole-region access, dataflow, and ABI
    validation in the production driver before they can authorize a call.
    """
    return list(_mlir_semantic_entries().values())


def _kernel_defn_regions(text: str) -> list[tuple[str, str]]:
    regions = []
    for match in re.finditer(r"kernel\.defn\s+@([A-Za-z0-9_.$-]+)", text):
        brace = text.find("{", match.end())
        if brace < 0:
            continue
        depth = 0
        for index in range(brace, len(text)):
            if text[index] == "{": depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    regions.append((match.group(1), text[match.start():index + 1]))
                    break
    return regions


def _mlir_semantic_entries(path: Path | None = None) -> dict[str, CompositionEntry]:
    default = Path(__file__).resolve().parents[2] / "generic_solver/kernel_library_v3.mlir"
    path = path or Path(os.environ.get("POLYGEIST_KERNEL_LIBRARY", default))
    if not path.is_file():
        return {}
    result = {}
    for symbol, region in _kernel_defn_regions(path.read_text()):
        generics = parse_generics(region, infer_outputs_from_yield=True)
        if not generics:
            continue
        steps = []
        try:
            for generic in generics:
                yields = encode_body_yields(generic)
                steps.append(CompositionStep(
                    body=yields[0],
                    body_per_yield=yields if len(yields) > 1 else None,
                    num_ins=len(generic.ins_arg_names),
                    num_outs=len(generic.outs_arg_names),
                    reduction_dim_count=generic.iterator_types.count("reduction"),
                    parallel_dim_count=generic.iterator_types.count("parallel"),
                ))
        except ValueError:
            continue
        result[symbol] = CompositionEntry(name=symbol, steps=steps)
    return result


def _term_repr(t) -> str:
    """Stable text repr of a Term (uses egglog's default __repr__)."""
    return str(t)


## Egglog proofs in the production matcher use a bounded iteration count.
## The companion egglog_library_variants.py audit additionally places every
## proof in an isolated process, applies a wall timeout, and records the case
## instead of replacing a slow proof with a second algebra implementation.


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

    def _paren_delta(text: str) -> int:
        return text.count("(") - text.count(")")

    def _split_egglog_lets(text: str) -> tuple[list[tuple[str, str]], str]:
        """Split egglog's pretty-printed let form into definitions + body.

        Reused subexpressions print as:

            _Term_1 = Term.Select(...)
            Term.Select(Term.Cmp("ogt", _Term_1, ...), _Term_1, ...)

        The structural matcher wants the fully inlined AST. This splitter is
        deliberately small: it recognizes only the `_Term_N = ...` shape that
        egglog emits for Term aliases, preserving multi-line RHS expressions.
        """
        lines = text.splitlines()
        if not lines or not re.match(r"\s*_Term_\d+\s*=", lines[0]):
            return [], text

        defs: list[tuple[str, str]] = []
        final: list[str] = []
        cur_name: Optional[str] = None
        cur_lines: list[str] = []
        depth = 0

        for line in lines:
            m = re.match(r"\s*(_Term_\d+)\s*=\s*(.*)$", line)
            if m:
                if cur_name is not None:
                    defs.append((cur_name, "\n".join(cur_lines).strip()))
                cur_name = m.group(1)
                cur_lines = [m.group(2)]
                depth = _paren_delta(m.group(2))
                continue

            if cur_name is not None and depth > 0:
                cur_lines.append(line)
                depth += _paren_delta(line)
                continue

            if cur_name is not None:
                defs.append((cur_name, "\n".join(cur_lines).strip()))
                cur_name = None
                cur_lines = []
                depth = 0
            final.append(line)

        if cur_name is not None:
            defs.append((cur_name, "\n".join(cur_lines).strip()))
        return defs, "\n".join(final).strip()

    let_defs, s = _split_egglog_lets(s)
    let_env: dict[str, object] = {}

    def parse_expr(i: int):
        """Returns (node, next_index)."""
        # Skip whitespace
        while i < len(s) and s[i] == " ":
            i += 1
        # Match `Term.<Ctor>(...)` leaf forms.
        for ctor in ("In", "Out", "Cap", "Lit", "NegativeZero", "Access", "Sqrt", "Abs", "Exp",
                     "Tanh", "Unary", "Binary", "Select", "Cmp"):
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
                        sub, _ = parse_expr_str(a)
                        # If parse_expr fully consumed `a`, use it.
                        if sub is not None:
                            parsed_args.append(sub)
                        else:
                            parsed_args.append(a)
                node = ("Lit", -0.0) if ctor == "NegativeZero" else (ctor, *parsed_args)
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
        if t in let_env:
            return let_env[t], len(t)
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
        for ctor in ("In", "Out", "Cap", "Lit", "NegativeZero", "Sqrt", "Abs", "Exp",
                     "Tanh", "Unary", "Binary", "Select", "Cmp"):
            tag = f"Term.{ctor}("
            if t.startswith(tag) and t.endswith(")"):
                inner = t[len(tag):-1]
                # Split args at top-level commas.
                args, depth, start = [], 0, 0
                for k, c in enumerate(inner):
                    if c == '(': depth += 1
                    elif c == ')': depth -= 1
                    elif c == ',' and depth == 0:
                        arg = inner[start:k].strip()
                        if arg:
                            args.append(arg)
                        start = k + 1
                arg = inner[start:].strip()
                if arg:
                    args.append(arg)
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
                return (("Lit", -0.0) if ctor == "NegativeZero" else (ctor, *parsed_args)), len(t)
        return None, 0

    for name, rhs in let_defs:
        let_env[name], _ = parse_expr_str(rhs)
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
            return bindings if _literal_identity(bindings[name]) == _literal_identity(body) else None
        bindings = dict(bindings)
        bindings[name] = body
        return bindings
    # Some front-end/canonicalization paths erase explicit multiplication by
    # one before the matcher sees the linalg body. Let a template term like
    # `In(k) * Cap("%w")` match a bare `In(k)` by binding `%w = 1.0`.
    # This keeps 3x3 filters with unit coefficients (Sobel/Laplacian/emboss)
    # on the same cudnnConvolution2D_9tap path as the fully weighted case.
    if (numeric_model() == "relaxed" and isinstance(template, tuple)
            and template[0] == "Mul" and len(template) == 3):
        for cap_idx, term_idx in ((1, 2), (2, 1)):
            cap = template[cap_idx]
            term = template[term_idx]
            if isinstance(cap, tuple) and cap[0] == "Cap":
                bound = _unify(body, term, bindings)
                if bound is not None:
                    bound = _unify(("Lit", 1.0), cap, bound)
                    if bound is not None:
                        return bound
    # Otherwise structural equality.
    if not (isinstance(template, tuple) and isinstance(body, tuple)):
        return bindings if template == body else None
    if template[0] != body[0]:
        return None
    if len(template) != len(body):
        return None
    # Leaf variants compare directly.
    if template[0] in {"In", "Out", "Lit"}:
        return bindings if _literal_identity(template) == _literal_identity(body) else None
    children_t = template[1:]
    children_b = body[1:]
    if (numeric_model() == "relaxed" and template[0] in COMMUTATIVE_OPS
            and len(children_t) == 2):
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


def _ast_to_term(node) -> Term:
    """Rebuild an Egglog term after substituting template captures."""
    if isinstance(node, list):
        node = tuple(node)
        node = (node[0], *(
            tuple(child) if isinstance(child, list) else child
            for child in node[1:]
        ))
    op = node[0]
    if op == "In": return Term.In(node[1])
    if op == "Out": return Term.Out(node[1])
    if op == "Cap": return Term.Cap(node[1])
    if op == "Lit": return literal(float(node[1]))
    if op == "Access": return Term.Access(node[1], node[2])
    if op == "Add": return _ast_to_term(node[1]) + _ast_to_term(node[2])
    if op == "Sub": return _ast_to_term(node[1]) - _ast_to_term(node[2])
    if op == "Mul": return _ast_to_term(node[1]) * _ast_to_term(node[2])
    if op == "Div": return _ast_to_term(node[1]) / _ast_to_term(node[2])
    if op in {"Sqrt", "Abs", "Exp", "Tanh"}:
        return getattr(Term, op)(_ast_to_term(node[1]))
    if op == "Unary": return Term.Unary(node[1], _ast_to_term(node[2]))
    if op == "Binary":
        return Term.Binary(node[1], _ast_to_term(node[2]), _ast_to_term(node[3]))
    if op == "Cmp":
        return Term.Cmp(node[1], _ast_to_term(node[2]), _ast_to_term(node[3]))
    if op == "Select":
        return Term.Select(*(_ast_to_term(child) for child in node[1:]))
    raise ValueError(f"unsupported Term AST operation: {op}")


def _substitute_caps(node, bindings: dict):
    if node[0] == "Cap" and node[1] in bindings:
        return bindings[node[1]]
    return (node[0], *(
        _substitute_caps(child, bindings)
        if isinstance(child, tuple) else child for child in node[1:]
    ))


def _ordered_unique_ast_nodes(node, wanted: set[str]) -> list:
    found = []
    seen = set()
    def visit(current):
        if not isinstance(current, tuple):
            return
        if current[0] in wanted and current not in seen:
            found.append(current)
            seen.add(current)
        for child in current[1:]:
            visit(child)
    visit(node)
    return found


def _indexed_leaf_counts(node) -> Counter:
    counts = Counter()
    def visit(current):
        if not isinstance(current, tuple):
            return
        if current[0] in {"In", "Out"}:
            counts[current] += 1
        for child in current[1:]:
            visit(child)
    visit(node)
    return counts


def _ast_node_count(node) -> int:
    if not isinstance(node, tuple):
        return 0
    return 1 + sum(_ast_node_count(child) for child in node[1:])


def _ac_identity_fingerprint(node):
    """Cheap necessary-condition filter before constructing an EGraph.

    It canonicalizes only associativity, commutativity, and 0/1 identities.
    Equality is still decided by Egglog; differing fingerprints merely avoid
    asking saturation to disprove obviously different expression topologies.
    """
    if not isinstance(node, tuple):
        return node
    op = node[0]
    if (op == "Select" and node[1][0] == "Cmp" and
            node[1][1] in {"slt", "sle", "ult", "ule"}):
        # This is only a necessary-condition filter: let Egglog decide
        # whether an integer ordered selection is invariant to operand order.
        predicate = node[1]
        compared = sorted((_ac_identity_fingerprint(x)
                           for x in predicate[2:]), key=repr)
        selected = sorted((_ac_identity_fingerprint(x)
                           for x in node[2:]), key=repr)
        return ("IntegerSelect", predicate[1], *compared, *selected)
    if op in {"Add", "Mul"}:
        children = []
        def gather(current):
            if isinstance(current, tuple) and current[0] == op:
                gather(current[1]); gather(current[2])
            else:
                children.append(_ac_identity_fingerprint(current))
        gather(node)
        if op == "Add":
            children = [x for x in children if x != ("Lit", 0.0)]
            identity = ("Lit", 0.0)
        else:
            if ("Lit", 0.0) in children:
                return ("Lit", 0.0)
            children = [x for x in children if x != ("Lit", 1.0)]
            identity = ("Lit", 1.0)
        if not children:
            return identity
        if len(children) == 1:
            return children[0]
        return (op, *sorted(children, key=repr))
    return (op, *(
        _ac_identity_fingerprint(child) if isinstance(child, tuple) else child
        for child in node[1:]
    ))


def _conditional_proof_candidate(left, right) -> bool:
    """Bound the additional EqSat search for conditional topology changes.

    Selection rules preserve the set of scalar leaves but may duplicate their
    occurrences in the printed tree. Count unique DAG nodes, as in the e-graph.
    This only admits a proof attempt; Egglog still decides equivalence.
    """
    def summary(node):
        nodes, leaves = set(), set()
        has_select = False
        def visit(current):
            nonlocal has_select
            if not isinstance(current, tuple) or current in nodes:
                return
            nodes.add(current)
            has_select |= current[0] == "Select"
            if current[0] in {"In", "Out", "Cap", "Lit", "Access"}:
                leaves.add(current)
            for child in current[1:]:
                visit(child)
        visit(node)
        return nodes, leaves, has_select
    ln, ll, ls = summary(left)
    rn, rl, rs = summary(right)
    return ls and rs and ll == rl and max(len(ln), len(rn)) <= 32


def _egglog_accepts_binding(body_ast, template_ast, bindings: dict) -> bool:
    instantiated = _substitute_caps(template_ast, bindings)
    if (numeric_model() == "strict" and
            (_has_negative_zero(body_ast) or _has_negative_zero(instantiated))):
        matched = _literal_identity(body_ast) == _literal_identity(instantiated)
        if matched:
            _MATCHER_TELEMETRY["proofs_matched"] += 1
        return matched
    same_fingerprint = (_ac_identity_fingerprint(body_ast) ==
                        _ac_identity_fingerprint(instantiated))
    conditional_candidate = (not same_fingerprint and
                             _conditional_proof_candidate(body_ast, instantiated))
    if not same_fingerprint and not conditional_candidate:
        return False
    # Literal structural equality is already a complete proof and does not
    # benefit from equality saturation.  In particular, multi-term updates
    # such as GEMVER's rank-2 GER can make the unrestricted AC e-graph grow
    # substantially even though the source and library expressions are
    # byte-for-byte the same after capture substitution.
    if body_ast == instantiated:
        _MATCHER_TELEMETRY["proofs_matched"] += 1
        return True
    # Large equality-saturation jobs run in the isolated audit harness, where
    # a real wall timeout can safely terminate Rust work. The in-process
    # production matcher must remain responsive and therefore declines those
    # cases instead of falling back to a handwritten algebraic implementation.
    # AC-normalized stencil expressions with seven taps are 17 nodes, yet
    # saturate in a few milliseconds because distributivity is disabled and
    # the fingerprint already agrees.  Keep a conservative ceiling for truly
    # large graphs while allowing these ordinary library-sized formulas.
    if (not conditional_candidate and
            max(_ast_node_count(body_ast), _ast_node_count(instantiated)) > 32):
        return False
    return equivalent(_ast_to_term(body_ast), _ast_to_term(instantiated),
                      include_distributivity=False)


def body_matches_template(body: Term, template: Term) -> Optional[dict]:
    """Check whether `body` matches `template`, with Cap names in the template
    as wildcards. Returns a binding dict on success, None on failure.

    Structural unification is only a fast way to propose capture bindings;
    Egglog is the authority that accepts or rejects the instantiated formula.
    When algebraic reassociation prevents structural binding, enumerate the
    small capture domain and let equality saturation test each proposal.
    """
    tmpl_ast = _parse_term(_term_repr(template))
    body_ast = _parse_term(_term_repr(body))
    _MATCHER_TELEMETRY["template_attempts"] += 1
    if _MATCHER_MODE == "syntactic":
        _MATCHER_TELEMETRY["syntactic_unifications"] += 1
        started = time.perf_counter()
        direct = _unify(body_ast, tmpl_ast, {})
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _MATCHER_TELEMETRY["proof_elapsed_ms"] += elapsed_ms
        _MATCHER_TELEMETRY["proof_elapsed_max_ms"] = max(
            _MATCHER_TELEMETRY["proof_elapsed_max_ms"], elapsed_ms)
        if direct is not None:
            _MATCHER_TELEMETRY["proofs_matched"] += 1
        return direct

    direct = _unify(body_ast, tmpl_ast, {})
    if direct is not None and _egglog_accepts_binding(body_ast, tmpl_ast, direct):
        return direct

    cap_nodes = _ordered_unique_ast_nodes(tmpl_ast, {"Cap"})
    cap_names = [node[1] for node in cap_nodes]
    if not cap_names:
        return {} if _egglog_accepts_binding(body_ast, tmpl_ast, {}) else None
    # The current rules reorder and reassociate indexed tensor values but do
    # not rename them. This cheap invariant avoids launching doomed (and
    # potentially costly) binding proofs against opaque scalar bodies.
    if _indexed_leaf_counts(body_ast) != _indexed_leaf_counts(tmpl_ast):
        return None
    candidates = _ordered_unique_ast_nodes(body_ast, {"Cap", "Lit"})
    for identity in (("Lit", 0.0), ("Lit", 1.0)):
        if identity not in candidates:
            candidates.append(identity)
    # Binding search is deliberately bounded. Large proofs are exercised by
    # egglog_library_variants.py in isolated processes with timeout telemetry.
    # Production never falls back to a second hand-written algebraic matcher.
    if len(candidates) ** len(cap_names) > 8:
        return None
    for values in product(candidates, repeat=len(cap_names)):
        proposal = dict(zip(cap_names, values))
        if _egglog_accepts_binding(body_ast, tmpl_ast, proposal):
            return proposal
    return None


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
        # Dispatch-only metadata whose kernel.defn has no semantic body is
        # not a match candidate.  Besides being non-semantic, accepting an
        # empty composition would make the caller advance by zero forever.
        if n == 0:
            continue
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
            if entry.element_type is not None:
                scalar_types = set(re.findall(
                    r"(?:arith\.[A-Za-z0-9_]+|math\.[A-Za-z0-9_]+|"
                    r"func\.call|linalg\.yield)\b[^:]*:\s*"
                    r"(?:\([^)]*\)\s*->\s*)?([A-Za-z0-9]+)",
                    "\n".join(g.body_lines),
                ))
                # Include the region signature.  Initializer bodies such as
                # `outs(%slice : tensor<...xf32>) { linalg.yield %zero }`
                # contain no typed arithmetic operation, but `%out: f32`
                # remains authoritative type evidence.
                scalar_types.update(g.block_arg_types or [])
                if entry.element_type not in scalar_types:
                    ok = False
                    break
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
            if step.body_per_yield is not None:
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


def composition_semantic_candidates(
    body_objs: list[GenericBody],
    body_terms: list[Term],
    compositions: list[CompositionEntry],
    start: int = 0,
    body_forms: list[str] | None = None,
) -> list[SemanticCandidate]:
    """Try every registered composition at `start` and return all hits.

    This is the candidate-producing version of `match_composition`. It calls
    the existing matcher with a single entry at a time, preserving all of its
    arity, iterator, and form checks.
    """
    out: list[SemanticCandidate] = []
    for entry in compositions:
        n = len(entry.steps)
        if start + n > len(body_terms):
            continue
        if any(t is None for t in body_terms[start : start + n]):
            continue
        m = match_composition(
            body_objs, body_terms, [entry], start=start, body_forms=body_forms
        )
        if m is None:
            continue
        matched_entry, _, bindings = m
        out.append(SemanticCandidate(
            name=matched_entry.name,
            body_indices=tuple(range(start, start + n)),
            match_kind="composition" if n > 1 else "whole",
            coverage="whole",
            bindings=bindings,
            entry=matched_entry,
            defaults=(),
            source="composition_library",
        ))
    return out


def enumerate_semantic_candidates(
    body_objs: list[GenericBody],
    body_terms: list[Optional[Term]],
    compositions: list[CompositionEntry],
    start: int = 0,
    body_forms: list[str] | None = None,
) -> list[SemanticCandidate]:
    """Enumerate semantic candidates for a linalg body index.

    Only imported scalar compositions are considered. The production driver
    must additionally prove accesses and dataflow before lowering a call.
    """
    if start >= len(body_objs) or start >= len(body_terms):
        return []
    body_term = body_terms[start]
    if body_term is None:
        return []

    candidates = composition_semantic_candidates(
        body_objs, body_terms, compositions, start=start, body_forms=body_forms
    )
    return candidates


if __name__ == "__main__":
    # Keep one public recognition entry point; there is no library inferred
    # from the benchmark bodies themselves.
    from spec_match import main
    raise SystemExit(main())
