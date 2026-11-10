"""Value-based normalisation: memory-level statement program -> system of
element equations (recurrence equations over final values).

For every output array X and every element e (symbolic index v0..vk) we
compute the final value as a formula over
    X_in[...]   values on entry to the region,
    Y_out[...]  final values of output elements (recurrences),
    scalar parameters and literals,
by composing the writes to e in schedule order.  Non-injective writes become
reductions (Sum over the fiber), scratch temporaries are inlined, and every
read is classified with an affine dependence check:

  * input  : no write to the element precedes the read,
  * final  : no write to the element follows the read,
  * acc    : the read is the accumulator of the write it feeds, and no other
             write to the element lies between the read and that write,
  * otherwise the read observes an intermediate value -> REJECT.

This is the array-dataflow / ADDG view of Feautrier (1991), Shashidhar et
al. (2005) and Verdoolaege et al. (TOPLAS 2012), restricted to the
quasi-affine, unimodular class produced by dense linear-algebra loop nests.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

from .affine import (Aff, Cons, NonAffine, atom_info, atom_table, atoms_in, bounds_of, deps, eq,
                     feasible, feasible_dnf, ge,
                     implies, implies_all, le, lex_lt_dnf, lt, negate, project, simplify,
                     subtract)
from .symexec import (Buffer, CondAff, CondF, EBin, EConst, EParam, ERead, ESel, EUn,
                      EUndef, Expr, Program, Stmt, Unsupported)


class Reject(Exception):
    """The region is outside the class or fails the dependence check."""


# ---------------------------------------------------------------------------
# Formula AST
# ---------------------------------------------------------------------------

class F:
    def key(self):
        raise NotImplementedError

    def __eq__(self, o):
        return isinstance(o, F) and self.key() == o.key()

    def __hash__(self):
        return hash(self.key())


class FRead(F):
    def __init__(self, arr: str, ver: str, idx):
        self.arr, self.ver, self.idx = arr, ver, tuple(idx)

    def key(self):
        return ("rd", self.arr, self.ver, self.idx)

    def __repr__(self):
        return f"{self.arr}@{self.ver}[{', '.join(map(str, self.idx))}]"


class FConst(F):
    def __init__(self, v: float):
        self.v = float(v)

    def key(self):
        return ("c", self.v)

    def __repr__(self):
        return repr(self.v)


class FParam(F):
    def __init__(self, name: str):
        self.name = name

    def key(self):
        return ("p", self.name)

    def __repr__(self):
        return self.name


class FBin(F):
    def __init__(self, op, a, b):
        self.op, self.a, self.b = op, a, b

    def key(self):
        return ("b", self.op, self.a.key(), self.b.key())

    def __repr__(self):
        sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}.get(self.op, self.op)
        return f"({self.a!r} {sym} {self.b!r})"


class FUn(F):
    def __init__(self, op, a):
        self.op, self.a = op, a

    def key(self):
        return ("u", self.op, self.a.key())

    def __repr__(self):
        return f"{self.op}({self.a!r})"


class FSel(F):
    """cond is ('aff', tuple[Cons]) or ('cmpf', kind, F, F)."""

    def __init__(self, cond, a, b):
        self.cond, self.a, self.b = cond, a, b

    def key(self):
        if self.cond[0] == "aff":
            ck = ("aff", tuple((c.aff, c.kind) for c in self.cond[1]))
        else:
            ck = ("cmpf", self.cond[1], self.cond[2].key(), self.cond[3].key())
        return ("s", ck, self.a.key(), self.b.key())

    def __repr__(self):
        if self.cond[0] == "aff":
            c = " && ".join(map(repr, self.cond[1]))
        else:
            c = f"{self.cond[2]!r} {self.cond[1]} {self.cond[3]!r}"
        return f"select({c}, {self.a!r}, {self.b!r})"


class FSum(F):
    """Sum over var in [lo, hi) (integer range, empty if hi <= lo)."""

    def __init__(self, var, lo: Aff, hi: Aff, body: F):
        self.var, self.lo, self.hi, self.body = var, lo, hi, body

    def key(self):
        return ("sum", self.var, self.lo, self.hi, self.body.key())

    def __repr__(self):
        return f"Sum[{self.var} in [{self.lo}, {self.hi})]({self.body!r})"


class FAcc(F):
    def key(self):
        return ("acc",)

    def __repr__(self):
        return "ACC"


ZERO = FConst(0.0)


def fmap_aff(f: F, fn) -> F:
    """Apply fn to every Aff (indices, bounds, constraints)."""
    if isinstance(f, FRead):
        return FRead(f.arr, f.ver, [fn(a) for a in f.idx])
    if isinstance(f, (FConst, FParam, FAcc)):
        return f
    if isinstance(f, FBin):
        return FBin(f.op, fmap_aff(f.a, fn), fmap_aff(f.b, fn))
    if isinstance(f, FUn):
        return FUn(f.op, fmap_aff(f.a, fn))
    if isinstance(f, FSel):
        if f.cond[0] == "aff":
            cond = ("aff", tuple(Cons(fn(c.aff), c.kind) for c in f.cond[1]))
        else:
            cond = ("cmpf", f.cond[1], fmap_aff(f.cond[2], fn), fmap_aff(f.cond[3], fn))
        return FSel(cond, fmap_aff(f.a, fn), fmap_aff(f.b, fn))
    if isinstance(f, FSum):
        return FSum(f.var, fn(f.lo), fn(f.hi), fmap_aff(f.body, fn))
    raise TypeError(f)


def fsubst(f: F, m: dict[str, Aff]) -> F:
    return fmap_aff(f, lambda a: a.subst(m))


def fwalk(f: F):
    yield f
    if isinstance(f, FBin):
        yield from fwalk(f.a)
        yield from fwalk(f.b)
    elif isinstance(f, FUn):
        yield from fwalk(f.a)
    elif isinstance(f, FSel):
        if f.cond[0] == "cmpf":
            yield from fwalk(f.cond[2])
            yield from fwalk(f.cond[3])
        yield from fwalk(f.a)
        yield from fwalk(f.b)
    elif isinstance(f, FSum):
        yield from fwalk(f.body)


def has_acc(f: F) -> bool:
    return any(isinstance(x, FAcc) for x in fwalk(f))


def replace_acc(f: F, v: F) -> F:
    if isinstance(f, FAcc):
        return v
    if isinstance(f, FBin):
        return FBin(f.op, replace_acc(f.a, v), replace_acc(f.b, v))
    if isinstance(f, FUn):
        return FUn(f.op, replace_acc(f.a, v))
    if isinstance(f, FSel):
        cond = f.cond
        if cond[0] == "cmpf":
            cond = ("cmpf", cond[1], replace_acc(cond[2], v), replace_acc(cond[3], v))
        return FSel(cond, replace_acc(f.a, v), replace_acc(f.b, v))
    if isinstance(f, FSum):
        return FSum(f.var, f.lo, f.hi, replace_acc(f.body, v))
    return f


# ---------------------------------------------------------------------------
# Equation system
# ---------------------------------------------------------------------------

@dataclass
class Equation:
    arr: str
    rank: int
    domain: list[Cons]          # over v0..v{rank-1} and params
    formula: F

    def __repr__(self):
        ev = ", ".join(f"v{k}" for k in range(self.rank))
        dom = " && ".join(map(repr, self.domain)) or "true"
        return f"{self.arr}@out[{ev}] = {self.formula!r}   for {dom}"


@dataclass
class System:
    equations: list[Equation]
    outputs: dict[str, Buffer]              # arr -> buffer
    arrays: dict[str, Buffer]               # every array (arg) appearing
    params: dict[str, str]                  # index param var -> source
    float_params: dict[str, str]
    reads: set = field(default_factory=set)  # (arr, ver)
    evidence: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    atoms: dict = field(default_factory=dict)  # index atoms used (affine.atom_table)


_uid = itertools.count()


def _fresh(prefix="r") -> str:
    return f"{prefix}#{next(_uid)}"


def _expr_reads(e, out: list):
    """ERead nodes of an expression (including inside CondF conditions)."""
    if isinstance(e, ERead):
        out.append(e)
    elif isinstance(e, EBin):
        _expr_reads(e.a, out); _expr_reads(e.b, out)
    elif isinstance(e, EUn):
        _expr_reads(e.a, out)
    elif isinstance(e, ESel):
        if isinstance(e.cond, CondF):
            _expr_reads(e.cond.a, out); _expr_reads(e.cond.b, out)
        _expr_reads(e.a, out); _expr_reads(e.b, out)
    return out


def _expr_map_reads(e, fn):
    """Rebuild an expression with every ERead r replaced by fn(r)."""
    if isinstance(e, ERead):
        return fn(e)
    if isinstance(e, EBin):
        return EBin(e.op, _expr_map_reads(e.a, fn), _expr_map_reads(e.b, fn))
    if isinstance(e, EUn):
        return EUn(e.op, _expr_map_reads(e.a, fn))
    if isinstance(e, ESel):
        c = e.cond
        if isinstance(c, CondF):
            c = CondF(c.kind, _expr_map_reads(c.a, fn), _expr_map_reads(c.b, fn))
        return ESel(c, _expr_map_reads(e.a, fn), _expr_map_reads(e.b, fn))
    return e


def _privatise_scalars(stmts: list[Stmt]) -> tuple[list[Stmt], list[str]]:
    """Scalar privatisation of block-local temporaries.

    Raising may keep a C block-local scalar (`double temp2 = 0;` inside
    `for i, for j`) in one scratch cell carried through the loops.  The
    element model needs one cell per iteration.  A rank-0 scratch cell B is
    given the loop coordinates P of a killing write K (a write of B whose
    value does not read B) when, within the window,
      * K runs in every iteration of P in which a read of B runs, and every
        read of B is inside P and strictly after K of the same iteration;
      * writes of B outside P are never read (they are dropped).
    Then every read sees a value written in its own iteration after the
    kill, which is exactly what per-iteration cells give.  Returns the
    rewritten statements and notes."""
    notes: list[str] = []
    scal = {}
    for st in stmts:
        b = st.buf
        if b.kind == "scratch" and b.rank == 0 and not b.priv:
            scal[b.id] = b
    if not scal:
        return stmts, notes
    reads_by_buf: dict[int, list[tuple[Stmt, ERead]]] = {}
    for st in stmts:
        for r in _expr_reads(st.value, []):
            if r.buf.id in scal:
                reads_by_buf.setdefault(r.buf.id, []).append((st, r))
    next_id = max(b.id for b in scal.values()) + 100000
    for bid, B in scal.items():
        writes = [st for st in stmts if st.buf.id == bid]
        reads = reads_by_buf.get(bid, [])
        if not reads:
            continue
        kills = [k for k in writes if k.vars and not any(r.buf.id == bid for r in _expr_reads(k.value, []))]
        chosen = None
        for K in sorted(kills, key=lambda k: -len(k.vars)):
            P = list(K.vars)
            ok = not K.guard
            for (st, r) in reads:
                if not ok:
                    break
                if list(r.vars[:len(P)]) != P or list(st.vars[:len(P)]) != P:
                    ok = False
                    break
                ctx = list(st.domain) + list(r.guard)
                if not implies_all(ctx, K.domain):
                    ok = False
                    break
                if list(r.time) == list(K.time) or feasible_dnf(ctx, lex_lt_dnf(list(r.time), list(K.time))):
                    ok = False
            if ok:
                chosen = (K, P)
                break
        if chosen is None:
            continue
        K, P = chosen
        inner = [w for w in writes if list(w.vars[:len(P)]) == P]
        B2 = Buffer(next_id, B.name, B.kind, 0, B.elem, list(P), [], B.arg_type)
        next_id += 1
        pv = tuple(Aff.var(v) for v in P)

        def remap(r, bid=bid, B2=B2, pv=pv):
            if r.buf.id != bid:
                return r
            return ERead(B2, pv + tuple(r.idx), r.time, r.guard, r.vars)
        out = []
        for st in stmts:
            if st.buf.id == bid and st not in inner:
                continue                  # dead write outside P
            if st.buf.id == bid:
                buf, idx = B2, list(pv) + list(st.idx)
            else:
                buf, idx = st.buf, st.idx
            out.append(Stmt(st.id, st.vars, st.domain, st.guard, st.time, buf, idx,
                            _expr_map_reads(st.value, remap), st.par_dims, st.top, st.origin))
        stmts = out
        notes.append(f"scalar {B.name} privatised over loops {P} (killed by S{K.id})")
    return stmts, notes


class Builder:
    def __init__(self, prog: Program, window: set[int]):
        self.prog = prog
        self.window = window
        self.stmts, self.priv_notes = _privatise_scalars([s for s in prog.stmts if s.top in window])
        self.writers: dict[int, list[Stmt]] = {}
        for s in self.stmts:
            self.writers.setdefault(s.buf.id, []).append(s)
        self.memo: dict[int, list[tuple[list[Cons], F]]] = {}
        self.inprogress: set[int] = set()
        self.stats = {"reads_input": 0, "reads_final": 0, "reads_acc": 0,
                      "reads_scratch_inlined": 0, "remat": 0, "dep_checks": 0,
                      "regions": 0, "chains": 0}
        self.warnings: list[str] = []
        self.params = set(prog.params)

    # -- instance description -----------------------------------------
    def inst(self, W: Stmt, ev_idx: list[Aff], suffix: str) -> list[Cons]:
        """Constraints describing instances J' of W (vars renamed with
        suffix) that write element ev_idx."""
        m = {v: Aff.var(v + suffix) for v in W.vars}
        cs = [c.subst(m) for c in W.domain + W.guard]
        if len(W.idx) != len(ev_idx):
            raise Reject("rank mismatch")
        for a, b in zip(W.idx, ev_idx):
            cs.append(eq(a.subst(m), b))
        return cs

    def time_of(self, W: Stmt, suffix: str) -> list[Aff]:
        m = {v: Aff.var(v + suffix) for v in W.vars}
        return [t.subst(m) for t in W.time]

    def writes_between(self, buf: Buffer, idx: list[Aff], ctx: list[Cons],
                       t_lo: Optional[list[Aff]], t_hi: Optional[list[Aff]]) -> bool:
        """Is there a write to buf[idx] with t_lo < time < t_hi (None = open)?"""
        for k, W in enumerate(self.writers.get(buf.id, [])):
            suf = f"'w{k}"
            base = ctx + self.inst(W, idx, suf)
            tw = self.time_of(W, suf)
            dnfs = []
            if t_lo is not None:
                dnfs.append(lex_lt_dnf(t_lo, tw))
            if t_hi is not None:
                dnfs.append(lex_lt_dnf(tw, t_hi))
            self.stats["dep_checks"] += 1
            if not dnfs:
                if feasible(base):
                    return True
                continue
            combos = dnfs[0] if len(dnfs) == 1 else [a + b for a in dnfs[0] for b in dnfs[1]]
            if feasible_dnf(base, combos):
                return True
        return False

    # -- per-writer element solving --------------------------------------
    def solve(self, W: Stmt, ev: list[str]):
        """Return (sigma, fiber_vars, residual) for f_W(J) = ev."""
        sigma: dict[str, Aff] = {}
        residual: list[Cons] = []
        free = list(W.vars)
        for a, v in zip(W.idx, ev):
            e = a.subst(sigma) - Aff.var(v)
            pick = None
            for x in reversed(free):   # prefer inner vars as pinned
                if abs(e.coeff(x)) == 1:
                    pick = x
                    break
            if pick is None:
                if any(e.coeff(x) for x in free):
                    raise Reject("non-unimodular write access")
                residual.append(Cons(e, "eq"))
                continue
            k = e.coeff(pick)
            rest = e - Aff.var(pick, k)
            sol = rest.scale(-1) if k == 1 else rest
            sigma = {x: y.subst({pick: sol}) for x, y in sigma.items()}
            sigma[pick] = sol
            free.remove(pick)
        return sigma, free, residual

    # -- element equations for a buffer ---------------------------------
    def element_equations(self, buf: Buffer, only: Optional[frozenset] = None) -> list[tuple[list[Cons], F]]:
        """Per-region final element values of buf; with `only`, the value
        after just those writers (the element's value at an earlier time)."""
        key = buf.id if only is None else (buf.id, only)
        if key in self.memo:
            return self.memo[key]
        if key in self.inprogress:
            raise Reject(f"cyclic scratch dependence through {buf}")
        self.inprogress.add(key)
        rank = len(buf.priv) + buf.rank
        ev = [f"v{k}" for k in range(rank)]
        evaff = [Aff.var(v) for v in ev]
        writers = [w for w in self.writers.get(buf.id, []) if only is None or w.id in only]
        info = []
        for W in writers:
            sigma, fiber, residual = self.solve(W, ev)
            dom = [c.subst(sigma) for c in W.domain]
            grd = [c.subst(sigma) for c in W.guard]
            fset = set(fiber)
            # A residual that pins an index atom of a fiber variable (a
            # scatter, y[row[k]]) selects which iterations write the element:
            # it guards the reduction instead of describing the element.
            fres = [c for c in residual if deps(c.vars()) & fset]
            residual = [c for c in residual if not (deps(c.vars()) & fset)]
            grd = grd + fres
            elem_only = [c for c in dom + grd if not (deps(c.vars()) & fset)]
            presence = simplify(elem_only + residual)
            info.append((W, sigma, fiber, residual, dom, grd, presence))
        # region partition ------------------------------------------------
        pieces: list[list[Cons]] = []
        for (W, sigma, fiber, residual, dom, grd, presence) in info:
            if presence is None:
                continue
            rem = [presence]
            for q in pieces:
                nxt = []
                for r in rem:
                    nxt.extend(subtract(r, q))
                rem = nxt
            pieces.extend(rem)
        changed = True
        guard_iter = 0
        while changed and guard_iter < 20:
            changed = False
            guard_iter += 1
            for pi, R in enumerate(list(pieces)):
                for (W, sigma, fiber, residual, dom, grd, presence) in info:
                    if presence is None:
                        continue
                    if implies_all(R, presence) or not feasible(R + presence):
                        continue
                    inside = R + presence
                    outside = subtract(R, presence)
                    pieces = pieces[:pi] + [inside] + outside + pieces[pi + 1:]
                    changed = True
                    break
                if changed:
                    break
        if len(pieces) > 24:
            raise Reject("too many element regions")
        out = []
        for R in pieces:
            R = simplify(R)
            if R is None:
                continue
            present = [x for x in info if x[6] is not None and implies_all(R, x[6])]
            # a reduction whose fiber is provably empty on R does not write R
            present = [x for x in present if not x[2] or feasible(R + x[4] + x[5])]
            chains = [x for x in present if x[2]]
            if len(chains) == 1 and len(present) == 1:
                # only a reduction writes R: e is written iff its fiber is non-empty
                keep = set(ev) | set(self.params)
                x = chains[0]
                shadow = project(x[4] + x[5] + x[3], keep)
                if shadow is not None:
                    # Whether a data-dependent range (a CSR row) is empty is
                    # unknown statically; an element it leaves unwritten keeps
                    # its value, which is the chain's formula over an empty
                    # range, so such conditions need not delimit the region.
                    shadow = [c for c in shadow if not atoms_in(c.vars())]
                R = simplify(R + (shadow or [])) if shadow is not None else None
                if R is None:
                    continue
            if not present:
                continue
            self.stats["regions"] += 1
            order = self.order_writers(R, present, evaff)
            if buf.kind == "arg":
                val: F = FRead(buf.name, "in", evaff)
            else:
                val = None
            for x in order:
                val = self.apply(buf, R, x, val, evaff)
            out.append((R, val))
        self.inprogress.discard(key)
        self.memo[key] = out
        return out

    def order_writers(self, R, present, evaff):
        def before(a, b) -> bool:
            Wa, Wb = a[0], b[0]
            base = R + self.inst(Wa, evaff, "'a") + self.inst(Wb, evaff, "'b")
            ta, tb = self.time_of(Wa, "'a"), self.time_of(Wb, "'b")
            self.stats["dep_checks"] += 1
            return not feasible_dnf(base, lex_lt_dnf(tb, ta))
        items = list(present)
        ordered = []
        while items:
            for c in items:
                if all(c is d or before(c, d) for d in items):
                    ordered.append(c)
                    items.remove(c)
                    break
            else:
                raise Reject("writes to an element interleave between statements")
        return ordered

    # -- applying one writer --------------------------------------------
    def apply(self, buf, R, x, prev: Optional[F], evaff) -> F:
        W, sigma, fiber, residual, dom, grd, presence = x
        ctx = R + dom + grd
        if not feasible(ctx):
            return prev
        f = self.conv(W.value, W, sigma, ctx, buf, evaff, top=True)
        if not fiber:
            if has_acc(f):
                if prev is None:
                    raise Reject(f"read of uninitialised scratch {buf}")
                f = replace_acc(f, prev)
            return f
        # reduction chain over the fiber ------------------------------
        self.stats["chains"] += 1
        if prev is None:
            raise Reject(f"reduction into uninitialised scratch {buf}")
        kind, h = self.decompose_chain(f)
        fset = set(fiber)
        fiber_guard = [c for c in grd if deps(c.vars()) & fset]
        if fiber_guard:
            h = FSel(("aff", tuple(fiber_guard)), h, ZERO)
        fcons = [c for c in dom if deps(c.vars()) & fset]
        body = h
        outer_ctx = list(R) + [c for c in dom if not (deps(c.vars()) & fset)]
        ranges = []
        for k, fv in enumerate(fiber):
            keep = set(fiber[:k + 1]) | {f"v{i}" for i in range(len(evaff))} | self.all_params(fcons + R)
            proj = project(fcons, keep)
            if proj is None:
                return prev   # empty fiber everywhere
            lows, ups, rest = bounds_of(fv, proj)
            lo = self.pick_bound(lows, outer_ctx + rest, fv, lower=True)
            hi = self.pick_bound(ups, outer_ctx + rest, fv, lower=False)
            ranges.append((fv, lo, hi + 1))
            outer_ctx = outer_ctx + [ge(Aff.var(fv), lo), le(Aff.var(fv), hi)]
        for fv, lo, hi in reversed(ranges):
            body = FSum(fv, lo, hi, body)
        if kind == "add":
            return FBin("add", prev, body)
        if kind == "sub":
            return FBin("sub", prev, body)
        raise Reject("unsupported reduction form")

    def all_params(self, cs) -> set[str]:
        out = set()
        for c in cs:
            out |= {v for v in c.vars() if v in self.params}
        return out | set(self.params)

    def pick_bound(self, cands: list[Aff], ctx: list[Cons], var: str, lower: bool) -> Aff:
        """max (lower) / min (upper) of candidate bounds, which must be
        provably attained by a single candidate under ctx."""
        if not cands:
            raise Reject(f"unbounded reduction variable {var}")
        keep = list(dict.fromkeys(cands))
        changed = True
        while changed and len(keep) > 1:
            changed = False
            for a in keep:
                for b in keep:
                    if a is b:
                        continue
                    if implies(ctx, le(a, b) if lower else le(b, a)):
                        keep.remove(a)
                        changed = True
                        break
                if changed:
                    break
        if len(keep) != 1:
            raise Reject(f"reduction bound of {var} is a {'max' if lower else 'min'} of {keep}")
        return keep[0]

    def decompose_chain(self, f: F):
        if isinstance(f, FBin) and f.op == "sub" and isinstance(f.a, FAcc) and not has_acc(f.b):
            return "sub", f.b
        if isinstance(f, FBin) and f.op == "add":
            if isinstance(f.a, FAcc) and not has_acc(f.b):
                return "add", f.b
            if isinstance(f.b, FAcc) and not has_acc(f.a):
                return "add", f.a
        lin = _linear_in_acc(f)
        if lin is not None:
            return "add", lin
        raise Reject(f"accumulation is not of the form acc (+|-) h: {f!r}")

    # -- converting a statement value ------------------------------------
    def conv(self, e: Expr, W: Stmt, sigma, ctx, buf, evaff, top=False) -> F:
        if not top and id(e) in self.prog.stored and not isinstance(e, (ERead, EConst, EParam)):
            S = self.prog.stored[id(e)][0]
            if S is not W and S.top in self.window:
                r = self.try_remat(e, S, W)
                if r is not None:
                    self.stats["remat"] += 1
                    return self.conv(r, W, sigma, ctx, buf, evaff)
        if isinstance(e, ERead):
            return self.resolve(e, W, sigma, ctx, buf, evaff)
        if isinstance(e, EConst):
            return FConst(e.v)
        if isinstance(e, EParam):
            return FParam(e.name)
        if isinstance(e, EBin):
            return FBin(e.op, self.conv(e.a, W, sigma, ctx, buf, evaff),
                        self.conv(e.b, W, sigma, ctx, buf, evaff))
        if isinstance(e, EUn):
            a = self.conv(e.a, W, sigma, ctx, buf, evaff)
            if e.op == "negf":
                return FBin("mul", FConst(-1.0), a)
            return FUn(e.op, a)
        if isinstance(e, ESel):
            if isinstance(e.cond, CondAff):
                cond = ("aff", tuple(c.subst(sigma) for c in e.cond.cons))
            else:
                cond = ("cmpf", e.cond.kind, self.conv(e.cond.a, W, sigma, ctx, buf, evaff),
                        self.conv(e.cond.b, W, sigma, ctx, buf, evaff))
            return FSel(cond, self.conv(e.a, W, sigma, ctx, buf, evaff),
                        self.conv(e.b, W, sigma, ctx, buf, evaff))
        if isinstance(e, EUndef):
            raise Reject("undefined value reaches an output")
        raise Reject(f"unknown expression {type(e).__name__}")

    def try_remat(self, e: Expr, S: Stmt, W: Stmt) -> Optional[ERead]:
        """A scalar value stored by S and reused in W is re-expressed as a
        read of the element S stored it to (value numbering through memory),
        if S's instance is determined by W's and nothing overwrites it in
        between."""
        if S.vars != W.vars[:len(S.vars)]:
            return None
        t_use = list(W.time[:-1]) + [Aff.cst(0)]
        ctx = list(W.domain)
        self.stats["dep_checks"] += 1
        if feasible_dnf(ctx, lex_lt_dnf(t_use, S.time)) or not feasible_dnf(ctx, lex_lt_dnf(S.time, t_use)):
            return None
        if not implies_all(ctx, S.domain + S.guard):
            return None
        if self.writes_between(S.buf, S.idx, ctx, S.time, t_use):
            return None
        return ERead(S.buf, tuple(S.idx), tuple(t_use), tuple(W.guard), tuple(W.vars))

    def resolve(self, r: ERead, W: Stmt, sigma, ctx, buf, evaff) -> F:
        idx = [a.subst(sigma) for a in r.idx]
        t = [a.subst(sigma) for a in r.time]
        rctx = ctx + [c.subst(sigma) for c in r.guard]
        C = r.buf
        tw = [a.subst(sigma) for a in W.time]
        # accumulator read: same element as the write it feeds
        if C is buf and all((a - b).is_const() and (a - b).const == 0 or implies(rctx, eq(a, b))
                            for a, b in zip(idx, evaff)):
            self.stats["dep_checks"] += 1
            if self.writes_between(C, idx, rctx, t, tw):
                v = self.value_at(C, idx, rctx, t)
                if v is None:
                    raise Reject(f"dependence check failed: accumulator read of {C}{idx} is "
                                 "separated from its write by another write")
                return v
            self.stats["reads_acc"] += 1
            return FAcc()
        writers = self.writers.get(C.id, [])
        if not writers:
            if C.kind != "arg":
                raise Reject(f"read of never-written scratch {C}")
            self.stats["reads_input"] += 1
            return FRead(C.name, "in", idx)
        before = self.writes_between(C, idx, rctx, None, t)
        if not before:
            if C.kind != "arg":
                raise Reject(f"read of uninitialised scratch {C}{idx}")
            self.stats["reads_input"] += 1
            return FRead(C.name, "in", idx)
        after = self.writes_between(C, idx, rctx, t, None)
        if after:
            v = self.value_at(C, idx, rctx, t)
            if v is None:
                raise Reject(f"dependence check failed: read of {C}{idx} at {t} observes an "
                             f"intermediate value (written both before and after)")
            return v
        if C.kind == "arg":
            self.stats["reads_final"] += 1
            return FRead(C.name, "out", idx)
        # scratch: inline its final element value
        eqs = self.element_equations(C)
        for (Rc, fc) in eqs:
            m = {f"v{k}": a for k, a in enumerate(idx)}
            Rs = [c.subst(m) for c in Rc]
            if implies_all(rctx, Rs):
                self.stats["reads_scratch_inlined"] += 1
                return _freshen(fsubst(fc, m))
        raise Reject(f"scratch read {C}{idx} spans several element regions")

    def value_at(self, C: Buffer, idx: list[Aff], rctx: list[Cons], t: list[Aff]) -> Optional[F]:
        """Value of element C[idx] at time t: the fold of exactly the writers
        whose instances writing C[idx] (under rctx) all precede t.  None if a
        writer has instances both before and after t (the element is still
        being updated at t), or if the value spans several element regions."""
        before_ids = []
        for k, W in enumerate(self.writers.get(C.id, [])):
            suf = f"'t{k}"
            base = rctx + self.inst(W, idx, suf)
            if not feasible(base):
                continue
            tw = self.time_of(W, suf)
            self.stats["dep_checks"] += 2
            has_before = feasible_dnf(base, lex_lt_dnf(tw, t))
            has_after = feasible_dnf(base, lex_lt_dnf(t, tw))
            if has_before and has_after:
                return None
            if has_before:
                before_ids.append(W.id)
        if not before_ids:
            return FRead(C.name, "in", idx) if C.kind == "arg" else None
        eqs = self.element_equations(C, frozenset(before_ids))
        m = {f"v{k}": a for k, a in enumerate(idx)}
        for (Rc, fc) in eqs:
            Rs = [c.subst(m) for c in Rc]
            if implies_all(rctx, Rs):
                self.stats["reads_at_time"] = self.stats.get("reads_at_time", 0) + 1
                return _freshen(fsubst(fc, m))
        return None

    # -- parallel-annotation check (warning only) ----------------------
    def check_parallel(self):
        for W in self.stmts:
            if not W.par_dims:
                continue
            for r in _reads(W.value):
                if r.buf is not W.buf:
                    continue
                for p in W.par_dims:
                    if p not in W.vars or p not in r.vars:
                        continue          # value captured from outside the generic
                    gen = p.rsplit("_", 1)[0] + "_"
                    a = {v: Aff.var(v + "'x") for v in W.vars}
                    cs = list(W.domain + W.guard) + [c.subst(a) for c in W.domain + W.guard]
                    # same execution of the generic: enclosing loop indices agree
                    cs += [eq(Aff.var(v + "'x"), Aff.var(v)) for v in W.vars if not v.startswith(gen)]
                    cs += [eq(x.subst(a), y) for x, y in zip(W.idx, r.idx)]
                    tx = [x.subst(a) for x in W.time]
                    for alt in ([lt(Aff.var(p + "'x"), Aff.var(p))], [lt(Aff.var(p), Aff.var(p + "'x"))]):
                        if feasible_dnf(cs + alt, lex_lt_dnf(tx, list(r.time))):
                            self.warnings.append(
                                f"linalg.generic dim {p} is annotated parallel but carries a "
                                f"dependence through {W.buf} (memref/sequential semantics used)")
                            break


def _reads(e: Expr):
    if isinstance(e, ERead):
        yield e
    elif isinstance(e, EBin):
        yield from _reads(e.a)
        yield from _reads(e.b)
    elif isinstance(e, EUn):
        yield from _reads(e.a)
    elif isinstance(e, ESel):
        if isinstance(e.cond, CondF):
            yield from _reads(e.cond.a)
            yield from _reads(e.cond.b)
        yield from _reads(e.a)
        yield from _reads(e.b)


def _freshen(f: F) -> F:
    """Rename bound sum variables apart (needed when inlining)."""
    if isinstance(f, FSum):
        nv = _fresh()
        body = fsubst(f.body, {f.var: Aff.var(nv)})
        return FSum(nv, f.lo, f.hi, _freshen(body))
    if isinstance(f, FBin):
        return FBin(f.op, _freshen(f.a), _freshen(f.b))
    if isinstance(f, FUn):
        return FUn(f.op, _freshen(f.a))
    if isinstance(f, FSel):
        cond = f.cond
        if cond[0] == "cmpf":
            cond = ("cmpf", cond[1], _freshen(cond[2]), _freshen(cond[3]))
        return FSel(cond, _freshen(f.a), _freshen(f.b))
    return f


def _linear_in_acc(f: F) -> Optional[F]:
    """If f == acc + g with g acc-free, return g (as a formula)."""
    def lin(x):
        # returns (coef, rest) with rest possibly None for zero
        if isinstance(x, FAcc):
            return 1, None
        if not has_acc(x):
            return 0, x
        if isinstance(x, FBin) and x.op in ("add", "sub"):
            ca, ra = lin(x.a)
            cb, rb = lin(x.b)
            if x.op == "sub":
                cb = -cb
                rb = None if rb is None else FBin("mul", FConst(-1.0), rb)
            rest = ra if rb is None else (rb if ra is None else FBin("add", ra, rb))
            return ca + cb, rest
        if isinstance(x, FSel) and x.cond[0] == "aff":
            ca, ra = lin(x.a)
            cb, rb = lin(x.b)
            if ca == cb:
                return ca, FSel(x.cond, ra or ZERO, rb or ZERO)
        raise Reject("non-linear accumulation")
    try:
        c, r = lin(f)
    except Reject:
        return None
    if c != 1:
        return None
    return r or ZERO


# ---------------------------------------------------------------------------
# Canonicalisation of formulas
# ---------------------------------------------------------------------------

def canonicalize(f: F, ctx: list[Cons], params: set[str]) -> F:
    f = _simplify_sel(f, ctx)
    f = _orient_sums(f, ctx)
    f = _simplify_sel(f, ctx)
    f = _rename_bound(f, 0)
    return f


def _simplify_sel(f: F, ctx: list[Cons]) -> F:
    if isinstance(f, FSel) and f.cond[0] == "aff":
        keep = []
        for c in f.cond[1]:
            if implies(ctx, c):
                continue
            if not feasible(ctx + [c]):
                return _simplify_sel(f.b, ctx)
            keep.append(c)
        a = _simplify_sel(f.a, ctx + keep)
        b = _simplify_sel(f.b, ctx)
        if not keep:
            return a
        if a == b:
            return a
        return FSel(("aff", tuple(keep)), a, b)
    if isinstance(f, FSel):
        return FSel(f.cond, _simplify_sel(f.a, ctx), _simplify_sel(f.b, ctx))
    if isinstance(f, FBin):
        return FBin(f.op, _simplify_sel(f.a, ctx), _simplify_sel(f.b, ctx))
    if isinstance(f, FUn):
        return FUn(f.op, _simplify_sel(f.a, ctx))
    if isinstance(f, FSum):
        v = Aff.var(f.var)
        inner = ctx + [ge(v, f.lo), lt(v, f.hi)]
        if not feasible(inner):
            return ZERO
        return FSum(f.var, f.lo, f.hi, _simplify_sel(f.body, inner))
    return f


def _occurrences(f: F, var: str):
    """Affine index expressions in reads that mention var."""
    for x in fwalk(f):
        if isinstance(x, FRead):
            for a in x.idx:
                if a.coeff(var):
                    yield a


def _orient_sums(f: F, ctx) -> F:
    """Change of variables so that the summation variable appears as a pure
    index (natural index), independent of loop direction / offset."""
    if isinstance(f, FSum):
        body = _orient_sums(f.body, ctx)
        occ = list(_occurrences(body, f.var))
        if occ:
            counts = {}
            for a in occ:
                counts[a] = counts.get(a, 0) + 1
            def rank(a):
                k = a.coeff(f.var)
                return (-counts[a], 0 if k == 1 else 1, len(a.terms), abs(a.const), repr(a))
            target = sorted(counts, key=rank)[0]
            k = target.coeff(f.var)
            if abs(k) == 1 and not (target.terms == ((f.var, 1),) and target.const == 0):
                rest = target - Aff.var(f.var, k)
                nv = f.var
                # target = k*var + rest  =>  var = k*(nv - rest)
                sol = (Aff.var(nv) - rest).scale(k)
                body2 = fsubst(body, {f.var: sol})
                if k == 1:
                    lo, hi = f.lo + rest, f.hi + rest
                else:
                    lo, hi = rest - f.hi + 1, rest - f.lo + 1
                return FSum(nv, lo, hi, body2)
        return FSum(f.var, f.lo, f.hi, body)
    if isinstance(f, FBin):
        return FBin(f.op, _orient_sums(f.a, ctx), _orient_sums(f.b, ctx))
    if isinstance(f, FUn):
        return FUn(f.op, _orient_sums(f.a, ctx))
    if isinstance(f, FSel):
        return FSel(f.cond, _orient_sums(f.a, ctx), _orient_sums(f.b, ctx))
    return f


def _rename_bound(f: F, depth: int) -> F:
    if isinstance(f, FSum):
        nv = f"r{depth}"
        body = fsubst(f.body, {f.var: Aff.var(nv)})
        return FSum(nv, f.lo, f.hi, _rename_bound(body, depth + 1))
    if isinstance(f, FBin):
        return FBin(f.op, _rename_bound(f.a, depth), _rename_bound(f.b, depth))
    if isinstance(f, FUn):
        return FUn(f.op, _rename_bound(f.a, depth))
    if isinstance(f, FSel):
        cond = f.cond
        if cond[0] == "cmpf":
            cond = ("cmpf", cond[1], _rename_bound(cond[2], depth), _rename_bound(cond[3], depth))
        return FSel(cond, _rename_bound(f.a, depth), _rename_bound(f.b, depth))
    return f


def _aff_vars(e: "Equation") -> set[str]:
    """Variables of every affine expression of an equation (domain, read
    indices, sum bounds, affine guards)."""
    out: set[str] = set()
    for c in e.domain:
        out |= c.vars()
    for x in fwalk(e.formula):
        if isinstance(x, FRead):
            for a in x.idx:
                out |= a.vars()
        elif isinstance(x, FSum):
            out |= x.lo.vars() | x.hi.vars()
        elif isinstance(x, FSel) and x.cond[0] == "aff":
            for c in x.cond[1]:
                out |= c.vars()
    return out


def _is_identity(arr: str, f: F, rank: int) -> bool:
    return isinstance(f, FRead) and f.arr == arr and f.ver == "in" and \
        list(f.idx) == [Aff.var(f"v{k}") for k in range(rank)]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_system(prog: Program, window: set[int]) -> System:
    b = Builder(prog, window)
    b.check_parallel()
    outputs = {}
    eqs: list[Equation] = []
    for buf in prog.buffers:
        if buf.kind != "arg" or buf.id not in b.writers:
            continue
        outputs[buf.name] = buf
        for R, f in b.element_equations(buf):
            f = canonicalize(f, R, b.params)
            if _is_identity(buf.name, f, buf.rank):
                continue
            eqs.append(Equation(buf.name, buf.rank, R, f))
    reads = set()
    for e in eqs:
        for x in fwalk(e.formula):
            if isinstance(x, FRead):
                reads.add((x.arr, x.ver))
    # Index atoms stand for the entry value of a read-only integer array: the
    # region must not write it, and it is an input role of the system.
    written = {bb.name for bb in prog.buffers if bb.id in b.writers}
    for e in eqs:
        for name in atoms_in(_aff_vars(e)):
            arr = atom_info(name)[0]
            if arr in written:
                raise Reject(f"index array {arr} is written in the region")
            reads.add((arr, "in"))
    arrays = {bb.name: bb for bb in prog.buffers if bb.kind == "arg"}
    sysm = System(eqs, outputs, arrays, dict(prog.params), dict(prog.float_params), reads,
                  dict(b.stats), list(dict.fromkeys(b.warnings)))
    names: set[str] = set()
    for e in eqs:
        names |= _aff_vars(e)
    sysm.atoms = atom_table(names)
    return sysm
