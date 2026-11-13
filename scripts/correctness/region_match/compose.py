"""Sequential composition of two specification equation systems.

    S1 ; S2   with a link  {S2 role -> S1 output array}

means: run S1, then run S2 where each linked S2 role is the memory S1 just
wrote.  The combined system is obtained by substituting S1's final element
values into S2's reads of the linked roles (piecewise over S1's element
domains, value on entry outside them) and listing, per linked array, S2's
equations first and S1's equations after (the comparison takes the first
equation whose domain contains a piece, so S1's value survives exactly where
S2 does not write).  Array, index-parameter and float-parameter names of the specs are
renamed apart ("a:" / "b:"); S2 index parameters may be identified with S1
parameters (enumerated by the caller).

Nothing here knows any routine; the result is an ordinary `System` that the
generic binding search and the egglog comparison consume unchanged.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional

from .affine import Aff, Cons
from .equations import (Equation, F, FBin, FConst, FParam, FRead, FSel, FSum, FUn, Reject,
                        System, _freshen, canonicalize, fsubst, fwalk)

P1, P2 = "a:", "b:"


def _pname(prefix: str, p: str) -> str:
    """Index parameter names start with 'p' (see matcher._used_params)."""
    return "p" + prefix + p[1:]


def sum_depth(f: F) -> int:
    if isinstance(f, FSum):
        return 1 + sum_depth(f.body)
    if isinstance(f, FBin):
        return max(sum_depth(f.a), sum_depth(f.b))
    if isinstance(f, FUn):
        return sum_depth(f.a)
    if isinstance(f, FSel):
        d = max(sum_depth(f.a), sum_depth(f.b))
        if f.cond[0] == "cmpf":
            d = max(d, sum_depth(f.cond[2]), sum_depth(f.cond[3]))
        return d
    return 0


@dataclass(frozen=True)
class Signature:
    """Binding-independent necessary facts used to prune composite search."""
    out_ranks: tuple          # sorted ranks of output arrays
    read_ranks: frozenset     # ranks of arrays read
    funops: frozenset
    depth: int                # max summation nesting
    n_fp: int                 # float parameters
    profile: frozenset = frozenset()   # {(version, rank, bound vars in the index)} of reads
    n_eqs: int = 0


def read_profile(sysm: System) -> frozenset:
    """Binding-independent kinds of element reads.  Under any binding a
    spec read keeps its version, rank and the summation variables in its
    index, and the comparison requires equal read sets per piece, so every
    application read kind must occur in the specification."""
    out = set()
    for e in sysm.equations:
        for x in fwalk(e.formula):
            if isinstance(x, FRead) and x.arr in sysm.arrays:
                bound = sorted({v for a in x.idx for v in a.vars() if v.startswith("r") and v[1:].isdigit()})
                out.add((x.ver, sysm.arrays[x.arr].rank, tuple(bound)))
    return frozenset(out)


def signature(sysm: System) -> Signature:
    from .matcher import _fparams, _fun_ops   # late import (cycle)
    reads = {sysm.arrays[a].rank for (a, _v) in sysm.reads if a in sysm.arrays}
    return Signature(tuple(sorted(b.rank for b in sysm.outputs.values())), frozenset(reads),
                     frozenset(_fun_ops(sysm)),
                     max((sum_depth(e.formula) for e in sysm.equations), default=0),
                     len(_fparams(sysm)), read_profile(sysm), len(sysm.equations))


def pair_plausible(app: Signature, s1: Signature, s2: Signature) -> bool:
    if not s1.n_eqs or not s2.n_eqs or not s1.out_ranks:
        return False
    if not app.profile <= s1.profile | s2.profile:
        return False
    if s1.funops | s2.funops != app.funops:
        return False
    if not (max(s1.depth, s2.depth) <= app.depth <= s1.depth + s2.depth):
        return False
    if not s1.read_ranks <= app.read_ranks:
        return False
    if not s2.read_ranks <= app.read_ranks | set(s1.out_ranks):
        return False
    if not app.read_ranks <= s1.read_ranks | s2.read_ranks:
        return False
    if s1.n_fp + s2.n_fp < app.n_fp:
        return False
    # every output rank of the composition must be an application output rank
    if not set(s1.out_ranks) | set(s2.out_ranks) <= set(app.out_ranks):
        return False
    return True


def links(s1: System, s2: System, app_out_ranks: tuple):
    """Non-empty partial injections  S2 role -> S1 output  (same rank and
    element type), such that the composition has the application's output
    ranks."""
    s1_outs = sorted(s1.outputs)
    s2_roles = sorted(s2.arrays)
    want = sorted(app_out_ranks)
    for k in range(1, len(s1_outs) + 1):
        for outs in itertools.permutations(s1_outs, k):
            for roles in itertools.combinations(s2_roles, k):
                ok = all((s2.arrays[r].rank, s2.arrays[r].elem) ==
                         (s1.arrays[o].rank, s1.arrays[o].elem) for r, o in zip(roles, outs))
                if not ok:
                    continue
                link = dict(zip(roles, outs))
                out_ranks = [s1.arrays[o].rank for o in s1_outs] + \
                            [s2.arrays[r].rank for r in s2.outputs if r not in link]
                if sorted(out_ranks) == want:
                    yield link


def param_identifications(s1: System, s2: System, used1, used2):
    """Each S2 index parameter is either kept distinct or identified with
    one S1 parameter (injectively)."""
    p1 = sorted(used1)
    p2 = sorted(used2)
    choices = [[None] + p1 for _ in p2]
    for combo in itertools.product(*choices):
        tgt = [c for c in combo if c is not None]
        if len(tgt) != len(set(tgt)):
            continue
        yield {q: c for q, c in zip(p2, combo) if c is not None}


class Composite:
    """Combined system plus the maps needed to split a binding back."""

    def __init__(self, s1: System, s2: System, link: dict[str, str], ident: dict[str, str]):
        self.s1, self.s2, self.link, self.ident = s1, s2, link, ident
        # name maps (spec name -> combined name)
        self.arr1 = {a: P1 + a for a in s1.arrays}
        self.arr2 = {a: (P1 + link[a]) if a in link else (P2 + a) for a in s2.arrays}
        self.par1 = {p: _pname(P1, p) for p in s1.params}
        self.par2 = {p: (_pname(P1, ident[p]) if p in ident else _pname(P2, p)) for p in s2.params}
        self.fp1 = {p: P1 + p for p in s1.float_params}
        self.fp2 = {p: P2 + p for p in s2.float_params}
        self.system = self._build()

    # -- renaming --------------------------------------------------------------
    @staticmethod
    def _ren(f: F, arrs: dict, pars: dict, fps: dict) -> F:
        m = {p: Aff.var(q) for p, q in pars.items()}

        def go(x: F) -> F:
            if isinstance(x, FRead):
                return FRead(arrs.get(x.arr, x.arr), x.ver, [a.subst(m) for a in x.idx])
            if isinstance(x, FParam):
                return FParam(fps.get(x.name, x.name))
            if isinstance(x, FConst):
                return x
            if isinstance(x, FBin):
                return FBin(x.op, go(x.a), go(x.b))
            if isinstance(x, FUn):
                return FUn(x.op, go(x.a))
            if isinstance(x, FSel):
                if x.cond[0] == "aff":
                    cond = ("aff", tuple(c.subst(m) for c in x.cond[1]))
                else:
                    cond = ("cmpf", x.cond[1], go(x.cond[2]), go(x.cond[3]))
                return FSel(cond, go(x.a), go(x.b))
            if isinstance(x, FSum):
                return FSum(x.var, x.lo.subst(m), x.hi.subst(m), go(x.body))
            raise Reject(f"compose: unexpected formula node {type(x).__name__}")
        return go(f)

    def _build(self) -> System:
        s1, s2 = self.s1, self.s2
        pm1 = {p: Aff.var(q) for p, q in self.par1.items()}
        pm2 = {p: Aff.var(q) for p, q in self.par2.items()}
        eqs1 = [Equation(self.arr1[e.arr], e.rank, [c.subst(pm1) for c in e.domain],
                         self._ren(e.formula, self.arr1, self.par1, self.fp1))
                for e in s1.equations]
        linked_out = {self.arr1[o] for o in self.link.values()}
        # S1 must not observe the *final* (post-S2) value of an array S2 writes
        s2_writes = {self.arr2[a] for a in s2.outputs}
        for e in eqs1:
            for x in fwalk(e.formula):
                if isinstance(x, FRead) and x.ver == "out" and x.arr in s2_writes:
                    raise Reject("compose: S1 reads a final value that S2 overwrites")
        by_arr1: dict[str, list[Equation]] = {}
        for e in eqs1:
            by_arr1.setdefault(e.arr, []).append(e)

        def s1_value(arr: str, idx) -> F:
            out: F = FRead(arr, "in", idx)
            for e in reversed(by_arr1.get(arr, [])):
                m = {f"v{k}": a for k, a in enumerate(idx)}
                dom = tuple(c.subst(m) for c in e.domain)
                out = FSel(("aff", dom), fsubst(_freshen(e.formula), m), out)
            return out

        def subst_links(x: F) -> F:
            if isinstance(x, FRead):
                if x.arr in linked_out and x.ver == "in":
                    return s1_value(x.arr, list(x.idx))
                return x
            if isinstance(x, (FConst, FParam)):
                return x
            if isinstance(x, FBin):
                return FBin(x.op, subst_links(x.a), subst_links(x.b))
            if isinstance(x, FUn):
                return FUn(x.op, subst_links(x.a))
            if isinstance(x, FSel):
                cond = x.cond
                if cond[0] == "cmpf":
                    cond = ("cmpf", cond[1], subst_links(cond[2]), subst_links(cond[3]))
                return FSel(cond, subst_links(x.a), subst_links(x.b))
            if isinstance(x, FSum):
                return FSum(x.var, x.lo, x.hi, subst_links(x.body))
            raise Reject("compose: unexpected node")

        params = set(self.par1.values()) | set(self.par2.values())
        eqs2 = []
        for e in s2.equations:
            dom = [c.subst(pm2) for c in e.domain]
            f = self._ren(e.formula, self.arr2, self.par2, self.fp2)
            # bound variables of S2 must be unique before S1 formulas (with
            # their own canonical r-names) are substituted under them
            f = subst_links(_freshen(f))
            f = canonicalize(f, dom, params)
            eqs2.append(Equation(self.arr2[e.arr], e.rank, dom, f))
        # final values: S2 first (it runs last), then S1 where S2 does not write
        eqs = eqs2 + eqs1
        arrays = {}
        for a, b in s1.arrays.items():
            arrays[self.arr1[a]] = b
        for a, b in s2.arrays.items():
            arrays.setdefault(self.arr2[a], b)
        outputs = {self.arr1[a]: b for a, b in s1.outputs.items()}
        for a, b in s2.outputs.items():
            outputs.setdefault(self.arr2[a], b)
        reads = set()
        for e in eqs:
            for x in fwalk(e.formula):
                if isinstance(x, FRead):
                    reads.add((x.arr, x.ver))
        pdict = {self.par1[p]: v for p, v in s1.params.items()}
        pdict.update({self.par2[p]: v for p, v in s2.params.items()})
        fdict = {self.fp1[p]: v for p, v in s1.float_params.items()}
        fdict.update({self.fp2[p]: v for p, v in s2.float_params.items()})
        warnings = list(s1.warnings) + list(s2.warnings)
        return System(eqs, outputs, arrays, pdict, fdict, reads, {}, warnings)

    # -- splitting a combined binding --------------------------------------
    def split(self, b):
        """Per-spec (arrays, params, fparams) from a binding of the combined
        system.  Returns None if the binding is not a valid sequential
        composition (an unlinked S2 role aliases memory S1 writes)."""
        s1_out_apps = {b.arrays[self.arr1[o]][1] for o in self.s1.outputs if self.arr1[o] in b.arrays}
        arrays1, arrays2 = {}, {}
        for a in self.s1.arrays:
            if self.arr1[a] in b.arrays:
                arrays1[a] = b.arrays[self.arr1[a]]
        for a in self.s2.arrays:
            cn = self.arr2[a]
            if cn not in b.arrays:
                continue
            i_arr, o_arr, p = b.arrays[cn]
            if a in self.link:
                # S2 sees the memory S1 wrote, in place
                mem = b.arrays[cn][1]
                arrays2[a] = (mem, mem if a in self.s2.outputs else None, p)
                continue
            for x in (i_arr, o_arr):
                if x is not None and x in s1_out_apps:
                    return None
            arrays2[a] = (i_arr, o_arr, p)
        params1 = {p: b.params[q] for p, q in self.par1.items() if q in b.params}
        params2 = {p: b.params[q] for p, q in self.par2.items() if q in b.params}
        fp1 = {p: b.fparams[q] for p, q in self.fp1.items() if q in b.fparams}
        fp2 = {p: b.fparams[q] for p, q in self.fp2.items() if q in b.fparams}
        return (arrays1, params1, fp1), (arrays2, params2, fp2)


def build_composites(s1: System, s2: System, app_out_ranks: tuple, used1, used2,
                     limit: int = 64) -> list[Composite]:
    out = []
    for link in links(s1, s2, app_out_ranks):
        # parameters are identified by the (non-injective) binding search
        for ident in ({},):
            try:
                out.append(Composite(s1, s2, link, ident))
            except Reject:
                continue
            if len(out) >= limit:
                return out
    return out
