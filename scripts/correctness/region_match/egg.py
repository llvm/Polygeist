"""egglog encoding of element equations and the equivalence check.

Scalar arithmetic reuses `Term` and `algebra_rules` from kernel_match.py
(imported read-only).  This module adds:

  RmIx / RmIxL   affine index expressions and index tuples
  RmArr          array references; RmArr.perm(a, "10") is `a` with its
                 dimensions permuted (row-major <-> column-major layout flip)
  rm_elem        Term for an array element read
  rm_ixv         an index used as a Term operand of Cmp
  rm_sum         Sum over an integer range (bound variable is an RmIx so that
                 index renaming by union works under binders)
  rm_le          relation  a <= b  (facts proved by the affine checker)

and the rule set:
  * index arithmetic normalisation (comm/assoc/constant folding),
  * dimension permutation       rm_elem(perm(a, p), idx) -> rm_elem(a, p(idx)),
  * triangular domain <-> mask   Sum[lo,hi) select(r < u, h, 0) <-> Sum[lo,u) h
                                 Sum[lo,hi) select(r >= l, h, 0) <-> Sum[l,hi) h
                                 (side conditions rm_le from the affine oracle),
  * reduction linearity          Sum(c * h) -> c * Sum(h),
  * real-field identities        a - b -> a + (-1)*b,  a / b -> a * (1/b).

Operand binding and index renaming are asserted as unions between the
specification namespace ("s:" prefixed names) and the application names.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_CORR = os.path.dirname(_HERE)
if _CORR not in sys.path:
    sys.path.insert(0, _CORR)

import kernel_match as km  # noqa: E402  (read-only reuse of Term + algebra_rules)
from egglog import (EGraph, Expr, StringLike, String, i64, i64Like, function, relation,  # noqa: E402
                    rewrite, rule, ruleset, union, vars_, var)

from .affine import Aff, atom_info, is_atom  # noqa: E402
from .equations import F, FBin, FConst, FIndex, FParam, FRead, FSel, FSum, FUn  # noqa: E402

Term = km.Term


class RmIx(Expr):
    def __init__(self, name: StringLike) -> None: ...

    @classmethod
    def lit(cls, v: i64Like) -> "RmIx": ...

    def __add__(self, other: "RmIx") -> "RmIx": ...

    def __mul__(self, other: "RmIx") -> "RmIx": ...


class RmIxL(Expr):
    @classmethod
    def nil(cls) -> "RmIxL": ...

    @classmethod
    def cons(cls, h: RmIx, t: "RmIxL") -> "RmIxL": ...


class RmArr(Expr):
    def __init__(self, name: StringLike) -> None: ...

    @classmethod
    def perm(cls, a: "RmArr", p: StringLike) -> "RmArr": ...


@function
def rm_elem(a: RmArr, idx: RmIxL) -> Term: ...


@function
def rm_ixv(i: RmIx) -> Term: ...


@function
def rm_sum(v: RmIx, lo: RmIx, hi: RmIx, body: Term) -> Term: ...


@function
def rm_param(a: RmArr) -> Term: ...


# An index atom (affine.py): the entry value of a read-only integer array
# element.  Encoded structurally, so the operand binding's array unions and
# the index renaming identify a spec atom with the application's.
@function
def rm_ixread(a: RmArr, idx: RmIxL) -> RmIx: ...


rm_le = relation("rm_le", RmIx, RmIx)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def _region_rule_groups() -> dict:
    """Region rules by ablation category (see scripts/correctness/eqsat_ablation.py)."""
    a, b, c = vars_("a b c", RmIx)
    x, y = vars_("x y", i64)
    r, lo, hi, u = vars_("r lo hi u", RmIx)
    h, t1, t2 = vars_("h t1 t2", Term)
    arr = var("arr", RmArr)
    fl = var("fl", km.f64)
    L = RmIx.lit
    zero = Term.Lit(0.0)
    one = Term.Lit(1.0)
    groups: dict[str, list] = {}
    groups["index"] = [
        # index arithmetic
        rewrite(a + b).to(b + a),
        rewrite((a + b) + c).to(a + (b + c)),
        rewrite(a + (b + c)).to((a + b) + c),
        rewrite(a + L(0)).to(a),
        rewrite(L(x) + L(y)).to(L(x + y)),
        rewrite(L(1) * a).to(a),
        rewrite(L(0) * a).to(L(0)),
        rewrite(L(x) * L(y)).to(L(x * y)),
        rewrite(L(x) * (a + b)).to(L(x) * a + L(x) * b),
        rewrite(L(x) * (L(y) * a)).to(L(x * y) * a),
        rewrite(L(x) * a + L(y) * a).to(L(x + y) * a),
        rewrite(a + L(x) * a).to(L(x + 1) * a),
    ]
    groups["mask"] = [
        # triangular domain <-> mask (side conditions from the affine oracle)
        rewrite(rm_sum(r, lo, hi, Term.Select(Term.Cmp("slt", rm_ixv(r), rm_ixv(u)), h, zero)))
        .to(rm_sum(r, lo, u, h), rm_le(u, hi)),
        rewrite(rm_sum(r, lo, hi, Term.Select(Term.Cmp("sge", rm_ixv(r), rm_ixv(u)), h, zero)))
        .to(rm_sum(r, u, hi, h), rm_le(lo, u)),
        rule(rm_sum(r, lo, u, h), rm_le(u, hi)).then(
            union(rm_sum(r, lo, u, h)).with_(
                rm_sum(r, lo, hi, Term.Select(Term.Cmp("slt", rm_ixv(r), rm_ixv(u)), h, zero)))),
        # nested masks commute
        rewrite(Term.Select(t1, Term.Select(t2, h, zero), zero))
        .to(Term.Select(t2, Term.Select(t1, h, zero), zero)),
    ]
    groups["sum"] = [
        # empty range (oracle proves hi <= lo)
        rewrite(rm_sum(r, lo, hi, h)).to(zero, rm_le(hi, lo)),
        # reduction linearity
        rewrite(rm_sum(r, lo, hi, Term.Lit(fl) * h)).to(Term.Lit(fl) * rm_sum(r, lo, hi, h)),
        # scalar parameters never depend on the summation variable
        rewrite(rm_sum(r, lo, hi, rm_param(arr) * h)).to(rm_param(arr) * rm_sum(r, lo, hi, h)),
        rewrite(rm_param(arr) * rm_sum(r, lo, hi, h)).to(rm_sum(r, lo, hi, rm_param(arr) * h)),
        rewrite(rm_sum(r, lo, hi, zero)).to(zero),
        # Contraction direction only. The binder and both bounds must be
        # identical; reassociation is permitted only in relaxed mode.
        rewrite(rm_sum(r, lo, hi, h) + rm_sum(r, lo, hi, t1))
        .to(rm_sum(r, lo, hi, h + t1)),
    ]
    groups["subdiv"] = [
        # real-field identities (exact over R, not bitwise in IEEE-754)
        rewrite(t1 - t2).to(t1 + Term.Lit(-1.0) * t2),
        rewrite(t1 / t2).to(t1 * (one / t2)),
        # Contract a common coefficient without the unbounded expansion of
        # distributivity. Used only under the explicit relaxed model.
        rewrite(t1 * h + t1 * t2).to(t1 * (h + t2)),
        rewrite(t1 - zero).to(t1),
        rewrite(Term.Lit(-0.0)).to(zero),
    ]
    groups["perm"] = []
    # dimension permutations up to rank 3
    i0, i1, i2 = vars_("i0 i1 i2", RmIx)
    C, N = RmIxL.cons, RmIxL.nil()
    for rank, idxv in ((2, [i0, i1]), (3, [i0, i1, i2])):
        for p in itertools.permutations(range(rank)):
            if list(p) == list(range(rank)):
                continue
            ps = "".join(map(str, p))
            lhs_l = N
            for v in reversed(idxv):
                lhs_l = C(v, lhs_l)
            # perm(a, p)[i_0..] = a[i_p^{-1}...]: element k of the permuted view
            # is dimension p[k] of the underlying array.
            inv = [0] * rank
            for k, pk in enumerate(p):
                inv[pk] = k
            rhs_l = N
            for d in reversed(range(rank)):
                rhs_l = C(idxv[inv[d]], rhs_l)
            groups["perm"].append(
                rewrite(rm_elem(RmArr.perm(arr, ps), lhs_l)).to(rm_elem(arr, rhs_l)))
    return groups


def _rules():
    return ruleset(*[r for rs in _region_rule_groups().values() for r in rs],
                   name="region_match_rules")


_RULES = None


def rules():
    """Select rules for the explicit numerical model and ablation."""
    import eqsat_ablation as abl
    m, cats = abl.mode()
    if m == "none":
        return ruleset(name="region_match_no_rules")
    groups = _region_rule_groups()
    if km.numeric_model() == "strict":
        groups = {"index": groups["index"], "perm": groups.get("perm", [])}
    kept = [rule for name, group in groups.items() if name not in cats for rule in group]
    scalar = (km.algebra_rules(include_distributivity=False) if m == "full"
              else abl.scalar_ruleset(km, cats))
    return ruleset(*kept, name="region_match_rules") + scalar


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def enc_ix(a: Aff, ns: str) -> RmIx:
    out = None
    for v, k in a.terms:
        if is_atom(v):
            arr, coords = atom_info(v)
            t = rm_ixread(RmArr(f"{ns}in:{arr}"), enc_idx(coords, ns))
        else:
            t = RmIx(_nsname(v, ns))
        if k != 1:
            t = RmIx.lit(k) * t
        out = t if out is None else out + t
    if a.const or out is None:
        c = RmIx.lit(a.const)
        out = c if out is None else out + c
    return out


def _nsname(v: str, ns: str) -> str:
    if v.startswith("r") and v[1:].isdigit():
        return v                      # bound variables share canonical names
    return ns + v


def enc_idx(idx, ns: str) -> RmIxL:
    out = RmIxL.nil()
    for a in reversed(list(idx)):
        out = RmIxL.cons(enc_ix(a, ns), out)
    return out


def _cond_terms(cons, ns, bound: Optional[str]):
    """Encode affine conditions as Cmp terms; constraints on the innermost
    bound variable use the canonical forms  r < U  /  r >= L."""
    out = []
    for c in cons:
        a = c.aff
        if c.kind == "eq":
            out.append(Term.Cmp("eq", rm_ixv(enc_ix(a, ns)), rm_ixv(RmIx.lit(0))))
            continue
        if bound is not None and abs(a.coeff(bound)) == 1:
            k = a.coeff(bound)
            rest = a - Aff.var(bound, k)
            if k == -1:          # -r + rest >= 0  ->  r < rest + 1
                out.append(Term.Cmp("slt", rm_ixv(RmIx(bound)), rm_ixv(enc_ix(rest + 1, ns))))
            else:                # r + rest >= 0   ->  r >= -rest
                out.append(Term.Cmp("sge", rm_ixv(RmIx(bound)), rm_ixv(enc_ix(-rest, ns))))
            continue
        out.append(Term.Cmp("sge", rm_ixv(enc_ix(a, ns)), rm_ixv(RmIx.lit(0))))
    return out


def encode(f: F, ns: str, arr_name, bound: Optional[str] = None) -> Term:
    """arr_name(arr, ver) -> egglog array name."""
    if isinstance(f, FRead):
        return rm_elem(RmArr(arr_name(f.arr, f.ver)), enc_idx(f.idx, ns))
    if isinstance(f, FConst):
        return Term.Lit(f.v)
    if isinstance(f, FIndex):
        return Term.Unary("integer-data", rm_ixv(enc_ix(f.value, ns)))
    if isinstance(f, FParam):
        return rm_param(RmArr(arr_name(f.name, "param")))
    if isinstance(f, FBin):
        a, b = encode(f.a, ns, arr_name, bound), encode(f.b, ns, arr_name, bound)
        return {"add": a + b, "sub": a - b, "mul": a * b, "div": a / b}.get(f.op) \
            if f.op in ("add", "sub", "mul", "div") else Term.Binary(f.op, a, b)
    if isinstance(f, FUn):
        a = encode(f.a, ns, arr_name, bound)
        if f.op == "sqrt":
            return Term.Sqrt(a)
        if f.op == "absf":
            return Term.Abs(a)
        if f.op == "exp":
            return Term.Exp(a)
        return Term.Unary(f.op, a)
    if isinstance(f, FSel):
        a, b = encode(f.a, ns, arr_name, bound), encode(f.b, ns, arr_name, bound)
        if f.cond[0] == "aff":
            conds = _cond_terms(f.cond[1], ns, bound)
            out = a
            for c in reversed(conds):
                out = Term.Select(c, out, b)
            return out
        ca, cb = encode(f.cond[2], ns, arr_name, bound), encode(f.cond[3], ns, arr_name, bound)
        return Term.Select(Term.Cmp(f.cond[1], ca, cb), a, b)
    if isinstance(f, FSum):
        return rm_sum(RmIx(f.var), enc_ix(f.lo, ns), enc_ix(f.hi, ns),
                      encode(f.body, ns, arr_name, f.var))
    raise TypeError(f)


# ---------------------------------------------------------------------------
# Equivalence check
# ---------------------------------------------------------------------------

class EggResult:
    def __init__(self, ok: bool, ms: float, nodes: int, classes: int, iters: int):
        self.ok, self.ms, self.nodes, self.classes, self.iters = ok, ms, nodes, classes, iters

    def __repr__(self):
        return f"EggResult(ok={self.ok}, {self.ms:.1f} ms, nodes={self.nodes}, classes={self.classes})"


def check_equal(fa: Term, fs: Term, unions: list[tuple], facts: list[tuple],
                iters: int = 10, collect: bool = True) -> EggResult:
    t0 = time.perf_counter()
    eg = EGraph()
    eg.register(fa, fs)
    for x, y in unions:
        eg.register(union(x).with_(y))
    for (lo, hi) in facts:
        eg.register(rm_le(lo, hi))
    ok = False
    n = 0
    sched = rules()
    for n in range(1, iters + 1):
        rep = eg.run(sched)
        if eg.check_bool(fa == fs):
            ok = True
            break
        if not rep.updated:
            break
    ms = (time.perf_counter() - t0) * 1000.0
    nodes = classes = 0
    if collect:
        try:
            ser = json.loads(eg._serialize(split_primitive_outputs=False,
                                           include_temporary_functions=False).to_json())
            nodes = len(ser.get("nodes", {}))
            classes = len(ser.get("class_data", {}))
        except Exception:
            nodes = sum(sz for _, sz in eg.all_function_sizes())
    return EggResult(ok, ms, nodes, classes, n)
