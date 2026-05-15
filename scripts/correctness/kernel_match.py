#!/home/arjaiswal/slacker/.venv/bin/python3
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

from egglog import EGraph, Expr, StringLike, i64Like, rewrite, ruleset, vars_


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
    def Lit(cls, name: StringLike) -> Term: ...

    def __add__(self, other: Term) -> Term: ...
    def __mul__(self, other: Term) -> Term: ...
    def __sub__(self, other: Term) -> Term: ...
    def __truediv__(self, other: Term) -> Term: ...

    @classmethod
    def Sqrt(cls, a: Term) -> Term: ...
    @classmethod
    def Abs(cls, a: Term) -> Term: ...
    @classmethod
    def Select(cls, pred: Term, t: Term, f: Term) -> Term: ...
    @classmethod
    def Cmp(cls, kind: StringLike, a: Term, b: Term) -> Term: ...


# ---------------------------------------------------------------------------
# Algebra rules (cosmetic variations).
# ---------------------------------------------------------------------------

a, b, c, d = vars_("a b c d", Term)


def algebra_rules():
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
        # Subtraction in terms of negation+add  (useful for some kernels)
        # We model `a - b == a + (-1 * b)` only if needed. Leave for later.
    )


# ---------------------------------------------------------------------------
# Parser: extract linalg.generic bodies from MLIR text.
# ---------------------------------------------------------------------------

@dataclass
class GenericBody:
    ins_arg_names: list[str]      # like ['%in', '%in_0', ...]
    outs_arg_names: list[str]     # like ['%out']
    body_lines: list[str]
    yield_value: str              # the SSA name that gets yielded
    captures: list[str]           # outer SSA values referenced in body
    indexing_maps: list[str]      # raw text of each map
    iterator_types: list[str]


_GEN_RE = re.compile(
    r"linalg\.generic\s*\{[^}]*indexing_maps\s*=\s*\[([^\]]*)\][^}]*"
    r"iterator_types\s*=\s*\[([^\]]*)\][^}]*\}[^\^]*?"
    r"\^bb0\(([^)]*)\)\s*:\s*(.*?)\s*linalg\.yield\s+(%[\w_]+)\s*:",
    re.DOTALL,
)


def parse_generics(mlir_text: str) -> list[GenericBody]:
    """Extract every linalg.generic with its body."""
    results = []
    for m in _GEN_RE.finditer(mlir_text):
        maps_str, iters_str, args_str, body_str, yield_name = m.groups()

        # Parse args like "%in: f64, %in_0: f64, %out: f64"
        ins, outs = [], []
        for piece in args_str.split(","):
            piece = piece.strip()
            if not piece:
                continue
            name = piece.split(":")[0].strip()
            (outs if name.startswith("%out") else ins).append(name)

        # Tokenize indexing maps and iterator types as raw substrings.
        maps = [s.strip() for s in re.findall(r"affine_map<[^>]*>", maps_str)]
        iters = [s.strip().strip('"') for s in iters_str.split(",")]

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

        results.append(GenericBody(
            ins_arg_names=ins,
            outs_arg_names=outs,
            body_lines=body_lines,
            yield_value=yield_name,
            captures=captures,
            indexing_maps=maps,
            iterator_types=iters,
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
    "math.sqrt": "sqrt",
    "math.absf": "abs",
    "arith.cmpf": "cmpf",
    "arith.select": "select",
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
        env[cap] = Term.Cap(cap)

    def lookup(name: str) -> Term:
        """Get the Term for an SSA name; fall back to Cap for unknown values."""
        if name in env:
            return env[name]
        # Unknown — synthesize a Cap leaf (covers `%cst`, `%cst_0`, etc.).
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
            # Numeric or other literal.
            return Term.Lit(tok)

        op_key = _OP_PATTERNS.get(op, op)
        if op_key == "mul":
            env[result] = resolve(arg_toks[0]) * resolve(arg_toks[1])
        elif op_key == "add":
            env[result] = resolve(arg_toks[0]) + resolve(arg_toks[1])
        elif op_key == "sub":
            env[result] = resolve(arg_toks[0]) - resolve(arg_toks[1])
        elif op_key == "div":
            env[result] = resolve(arg_toks[0]) / resolve(arg_toks[1])
        elif op_key == "sqrt":
            env[result] = Term.Sqrt(resolve(arg_toks[0]))
        elif op_key == "abs":
            env[result] = Term.Abs(resolve(arg_toks[0]))
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
