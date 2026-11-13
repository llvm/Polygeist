"""Tiny affine/Presburger-lite toolkit for the region matcher.

Everything here works on conjunctions of integer affine constraints over
named variables (loop indices, element indices, symbolic parameters).  The
decision procedure is Fourier-Motzkin elimination with integer tightening
(divide by the gcd of the coefficients and floor the constant).  It is
*sound for infeasibility*: when `feasible()` returns False the system has no
integer solution.  When it returns True the rational relaxation is feasible,
which is exact for the unimodular rectangular/triangular systems produced by
dense linear-algebra loop nests (the class this matcher targets).

No external solver (islpy is not available) is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Iterable, Mapping


class Aff:
    """Immutable integer affine expression  sum(c_v * v) + const."""

    __slots__ = ("terms", "const", "_h")

    def __init__(self, terms: Mapping[str, int] | None = None, const: int = 0):
        t = {}
        if terms:
            for k, v in terms.items():
                if v:
                    t[k] = int(v)
        self.terms = tuple(sorted(t.items()))
        self.const = int(const)
        self._h = hash((self.terms, self.const))

    # constructors -------------------------------------------------------
    @staticmethod
    def var(name: str, coeff: int = 1) -> "Aff":
        return Aff({name: coeff})

    @staticmethod
    def cst(c: int) -> "Aff":
        return Aff(None, c)

    # protocol -----------------------------------------------------------
    def __hash__(self):
        return self._h

    def __eq__(self, other):
        return isinstance(other, Aff) and self.terms == other.terms and self.const == other.const

    def coeffs(self) -> dict[str, int]:
        return dict(self.terms)

    def coeff(self, v: str) -> int:
        for k, c in self.terms:
            if k == v:
                return c
        return 0

    def vars(self) -> set[str]:
        return {k for k, _ in self.terms}

    def is_const(self) -> bool:
        return not self.terms

    def __add__(self, o):
        if isinstance(o, int):
            return Aff(self.coeffs(), self.const + o)
        d = self.coeffs()
        for k, v in o.terms:
            d[k] = d.get(k, 0) + v
        return Aff(d, self.const + o.const)

    __radd__ = __add__

    def __neg__(self):
        return Aff({k: -v for k, v in self.terms}, -self.const)

    def __sub__(self, o):
        if isinstance(o, int):
            return self + (-o)
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def scale(self, k: int) -> "Aff":
        return Aff({a: b * k for a, b in self.terms}, self.const * k)

    def __mul__(self, o):
        if isinstance(o, int):
            return self.scale(o)
        if isinstance(o, Aff):
            if o.is_const():
                return self.scale(o.const)
            if self.is_const():
                return o.scale(self.const)
        raise NonAffine("product of two non-constant affine expressions")

    __rmul__ = __mul__

    def subst(self, m: Mapping[str, "Aff"]) -> "Aff":
        out = Aff(None, self.const)
        for k, c in self.terms:
            if k in m:
                out = out + m[k].scale(c)
            elif k in _ATOM_INFO and m:
                out = out + _subst_atom(k, m).scale(c)
            else:
                out = out + Aff({k: c})
        return out

    def rename(self, m: Mapping[str, str]) -> "Aff":
        return self.subst({k: Aff.var(v) for k, v in m.items()})

    def __repr__(self):
        parts = []
        for k, c in self.terms:
            if c == 1:
                parts.append(k)
            elif c == -1:
                parts.append(f"-{k}")
            else:
                parts.append(f"{c}*{k}")
        if self.const or not parts:
            parts.append(str(self.const))
        s = " + ".join(parts)
        return s.replace("+ -", "- ")


class NonAffine(Exception):
    pass


# ---------------------------------------------------------------------------
# Index atoms
# ---------------------------------------------------------------------------
# The value of an integer array element that the analysed region never writes
# (an index array of a sparse format, the bin of a histogram sample) is an
# unknown integer that is used like a parameter: in loop bounds (a CSR row
# range), gathers (x[col[k]]) and scatters (y[row[k]]).  An atom is an
# ordinary variable of this algebra, named "@<n>", whose identity is
# (array, coords): equal reads are the same variable, and substitution renames
# the variables inside its coords (and, through an "@arr:<name>" key, its
# array).  Treating an atom as a free variable only forgets facts about it, so
# feasibility / implication results stay sound; deps() exposes the variables
# an atom depends on to code that must not separate an atom from its
# coordinates (projection, guard and fiber classification).

_ATOM_OF: dict[tuple, str] = {}
_ATOM_INFO: dict[str, tuple[str, tuple]] = {}


def atom(arr: str, coords) -> Aff:
    key = (arr, tuple(coords))
    name = _ATOM_OF.get(key)
    if name is None:
        # content-derived (Aff reprs are canonical), so a name means the same
        # element in every process -- spec systems are pickled to disk
        name = f"@{arr}[{', '.join(map(repr, key[1]))}]"
        _ATOM_OF[key] = name
        _ATOM_INFO[name] = key
    return Aff.var(name)


def atom_table(names) -> dict[str, tuple]:
    """(array, coords) of every atom among names (transitively)."""
    return {n: _ATOM_INFO[n] for n in atoms_in(names)}


def register_atoms(table: Mapping[str, tuple]) -> None:
    """Re-register atoms of a system created in another process."""
    for name, key in table.items():
        _ATOM_INFO.setdefault(name, key)
        _ATOM_OF.setdefault(key, name)


def is_atom(v: str) -> bool:
    return v in _ATOM_INFO


def atom_info(v: str) -> tuple[str, tuple]:
    """(array, coords) of an atom variable."""
    return _ATOM_INFO[v]


def _subst_atom(name: str, m: Mapping[str, Aff]) -> Aff:
    arr, coords = _ATOM_INFO[name]
    new_coords = tuple(a.subst(m) for a in coords)
    r = m.get("@arr:" + arr)
    new_arr = next(iter(r.vars())) if r is not None else arr
    if new_arr == arr and new_coords == coords:
        return Aff.var(name)
    return atom(new_arr, new_coords)


def deps(names) -> set[str]:
    """names plus every variable (and nested atom) their atoms depend on."""
    out: set[str] = set()
    stack = list(names)
    while stack:
        v = stack.pop()
        if v in out:
            continue
        out.add(v)
        if v in _ATOM_INFO:
            for a in _ATOM_INFO[v][1]:
                stack.extend(a.vars())
    return out


def atoms_in(names) -> set[str]:
    return {v for v in deps(names) if v in _ATOM_INFO}


def _closed_keep(cs: list, keep: set[str]) -> set[str]:
    """Keep set for a projection: an atom survives only if every variable it
    depends on survives (otherwise it would name an eliminated variable)."""
    base = {v for v in keep if v not in _ATOM_INFO}
    present = set()
    for c in cs:
        present |= {v for v in c.vars() if v in _ATOM_INFO}
    ok = {a for a in present | (set(keep) - base)
          if {d for d in deps({a}) if d not in _ATOM_INFO} <= base}
    return base | ok


@dataclass(frozen=True)
class Cons:
    """aff >= 0 (kind 'ge') or aff == 0 (kind 'eq')."""
    aff: Aff
    kind: str = "ge"

    def __repr__(self):
        return f"{self.aff} {'>=' if self.kind == 'ge' else '=='} 0"

    def subst(self, m):
        return Cons(self.aff.subst(m), self.kind)

    def rename(self, m):
        return Cons(self.aff.rename(m), self.kind)

    def vars(self):
        return self.aff.vars()


def ge(a: Aff, b: Aff | int = 0) -> Cons:
    """a >= b"""
    return Cons(a - b, "ge")


def le(a: Aff, b: Aff | int) -> Cons:
    """a <= b"""
    if isinstance(b, int):
        b = Aff.cst(b)
    return Cons(b - a, "ge")


def lt(a: Aff, b: Aff | int) -> Cons:
    """a < b  (integers: a + 1 <= b)"""
    if isinstance(b, int):
        b = Aff.cst(b)
    return Cons(b - a - 1, "ge")


def eq(a: Aff, b: Aff | int) -> Cons:
    if isinstance(b, int):
        b = Aff.cst(b)
    return Cons(a - b, "eq")


def negate(c: Cons) -> list[list[Cons]]:
    """Negation as a disjunction of conjunctions."""
    if c.kind == "ge":
        return [[Cons(-c.aff - 1, "ge")]]
    return [[Cons(c.aff - 1, "ge")], [Cons(-c.aff - 1, "ge")]]


# ---------------------------------------------------------------------------
# Normalisation and Fourier-Motzkin
# ---------------------------------------------------------------------------

class _Infeasible(Exception):
    pass


def _normalize(c: Cons) -> Cons | None:
    """gcd-normalise; returns None for a trivially true constraint and raises
    _Infeasible for a trivially false one."""
    a = c.aff
    if a.is_const():
        if c.kind == "ge" and a.const >= 0:
            return None
        if c.kind == "eq" and a.const == 0:
            return None
        raise _Infeasible
    g = 0
    for _, v in a.terms:
        g = gcd(g, abs(v))
    if g > 1:
        if c.kind == "eq":
            if a.const % g:
                raise _Infeasible
            a = Aff({k: v // g for k, v in a.terms}, a.const // g)
        else:
            a = Aff({k: v // g for k, v in a.terms}, a.const // g)  # floor
    if c.kind == "eq":
        # canonical sign: first coefficient positive
        if a.terms[0][1] < 0:
            a = -a
    return Cons(a, c.kind)


def _eliminate_equalities(cs: list[Cons]) -> list[Cons]:
    cs = list(cs)
    while True:
        idx = None
        for i, c in enumerate(cs):
            if c.kind == "eq":
                for v, k in c.aff.terms:
                    if abs(k) == 1:
                        idx = (i, v, k)
                        break
                if idx:
                    break
        if idx is None:
            break
        i, v, k = idx
        c = cs.pop(i)
        # k*v + rest == 0  ->  v = -rest/k
        rest = c.aff - Aff.var(v, k)
        sol = rest.scale(-1) if k == 1 else rest
        new = []
        for d in cs:
            d2 = _normalize(d.subst({v: sol}))
            if d2 is not None:
                new.append(d2)
        cs = new
    # remaining equalities without unit coefficients -> two inequalities
    out = []
    for c in cs:
        if c.kind == "eq":
            for d in (Cons(c.aff, "ge"), Cons(-c.aff, "ge")):
                d2 = _normalize(d)
                if d2 is not None:
                    out.append(d2)
        else:
            out.append(c)
    return out


def _fm(cs: list[Cons], keep: set[str] | None = None, budget: int = 4000) -> list[Cons]:
    """Eliminate all variables not in `keep` (None = eliminate everything).
    Returns the projected inequality system.  Raises _Infeasible."""
    cs = [c for c in (_normalize(c) for c in cs) if c is not None]
    cs = _eliminate_equalities(cs)
    cs = list(dict.fromkeys(cs))
    while True:
        allv = set()
        for c in cs:
            allv |= c.vars()
        elim = [v for v in allv if keep is None or v not in keep]
        if not elim:
            return cs
        # pick the variable producing the fewest new constraints
        best = None
        for v in elim:
            pos = sum(1 for c in cs if c.aff.coeff(v) > 0)
            neg = sum(1 for c in cs if c.aff.coeff(v) < 0)
            score = pos * neg - pos - neg
            if best is None or score < best[0]:
                best = (score, v)
        v = best[1]
        lows = [c for c in cs if c.aff.coeff(v) > 0]
        ups = [c for c in cs if c.aff.coeff(v) < 0]
        rest = [c for c in cs if c.aff.coeff(v) == 0]
        new = list(rest)
        for lo in lows:
            a = lo.aff.coeff(v)
            for up in ups:
                b = -up.aff.coeff(v)
                comb = lo.aff.scale(b) + up.aff.scale(a)
                n = _normalize(Cons(comb, "ge"))
                if n is not None:
                    new.append(n)
        cs = list(dict.fromkeys(new))
        if len(cs) > budget:
            # give up eliminating precisely: drop to "unknown" (feasible)
            return []


def feasible(cs: Iterable[Cons]) -> bool:
    """False only if the integer system is proven infeasible."""
    try:
        _fm(list(cs))
        return True
    except _Infeasible:
        return False


def feasible_dnf(conj: Iterable[Cons], dnf: list[list[Cons]]) -> bool:
    base = list(conj)
    return any(feasible(base + d) for d in dnf)


def implies(ctx: Iterable[Cons], c: Cons) -> bool:
    """ctx |= c (proved)."""
    ctx = list(ctx)
    return not any(feasible(ctx + d) for d in negate(c))


def implies_all(ctx: Iterable[Cons], cs: Iterable[Cons]) -> bool:
    ctx = list(ctx)
    return all(implies(ctx, c) for c in cs)


def project(cs: Iterable[Cons], keep: set[str]) -> list[Cons] | None:
    """Rational shadow onto `keep`; None if infeasible."""
    cs = list(cs)
    keep = _closed_keep(cs, set(keep))
    try:
        return _fm(cs, keep)
    except _Infeasible:
        return None


def simplify(cs: Iterable[Cons], ctx: Iterable[Cons] = ()) -> list[Cons] | None:
    """Remove constraints implied by the others plus ctx.  None if infeasible."""
    ctx = list(ctx)
    cur = []
    for c in cs:
        try:
            n = _normalize(c)
        except _Infeasible:
            return None
        if n is not None and n not in cur:
            cur.append(n)
    if not feasible(ctx + cur):
        return None
    # equalities first: keep them, they are cheap and canonical
    changed = True
    while changed:
        changed = False
        for c in list(cur):
            others = [d for d in cur if d is not c]
            if implies(ctx + others, c):
                cur.remove(c)
                changed = True
                break
    return cur


def equivalent_sets(a: Iterable[Cons], b: Iterable[Cons], ctx: Iterable[Cons] = ()) -> bool:
    a, b, ctx = list(a), list(b), list(ctx)
    return implies_all(ctx + a, b) and implies_all(ctx + b, a)


def bounds_of(var: str, cs: Iterable[Cons]) -> tuple[list[Aff], list[Aff], list[Cons]]:
    """Split constraints into lower bounds (var >= L), upper bounds (var <= U)
    and constraints not mentioning var.  Only unit coefficients supported."""
    lows, ups, rest = [], [], []
    for c in cs:
        k = c.aff.coeff(var)
        if k == 0:
            rest.append(c)
            continue
        if c.kind == "eq":
            if abs(k) != 1:
                raise NonAffine("non-unit equality coefficient")
            other = c.aff - Aff.var(var, k)
            sol = other.scale(-1) if k == 1 else other
            lows.append(sol)
            ups.append(sol)
            continue
        if abs(k) != 1:
            raise NonAffine("non-unit bound coefficient")
        other = c.aff - Aff.var(var, k)
        if k == 1:          # var + other >= 0  -> var >= -other
            lows.append(other.scale(-1))
        else:               # -var + other >= 0 -> var <= other
            ups.append(other)
    return lows, ups, rest


# ---------------------------------------------------------------------------
# Lexicographic order on schedule vectors
# ---------------------------------------------------------------------------

def lex_lt_dnf(a: list[Aff], b: list[Aff]) -> list[list[Cons]]:
    """a <_lex b as a DNF over the variables of a and b."""
    out = []
    prefix: list[Cons] = []
    for x, y in zip(a, b):
        d = y - x
        if d.is_const():
            if d.const > 0:
                out.append(list(prefix))
                return out
            if d.const < 0:
                return out
            continue  # equal at this level, go deeper
        out.append(prefix + [lt(x, y)])
        prefix = prefix + [eq(x, y)]
    return out  # equal prefix: not strictly less


def rename_apart(cs: Iterable[Cons], vars_: Iterable[str], suffix: str) -> tuple[list[Cons], dict[str, str]]:
    m = {v: v + suffix for v in vars_}
    return [c.rename(m) for c in cs], m


# ---------------------------------------------------------------------------
# Unions of convex pieces (used for element-domain partitioning)
# ---------------------------------------------------------------------------

def subtract(a: list[Cons], b: list[Cons], ctx: Iterable[Cons] = ()) -> list[list[Cons]]:
    """Convex pieces of a \\ b (disjoint)."""
    ctx = list(ctx)
    pieces = []
    acc = list(a)
    for c in b:
        for neg in negate(c):
            p = acc + neg
            if feasible(ctx + p):
                pieces.append(p)
        acc = acc + [c]
        if not feasible(ctx + acc):
            break
    return pieces
