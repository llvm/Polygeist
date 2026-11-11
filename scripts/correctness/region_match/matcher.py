"""Generic region matcher: equation systems of an application region and of
library specifications are compared under a searched operand binding.

Nothing in this module knows about any particular library routine: the
specifications (C transliterations raised by the same pipeline) and their
manifests are the only per-routine inputs.
"""
from __future__ import annotations

import glob
import hashlib
import itertools
import json
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from .affine import (Aff, Cons, atom_info, atoms_in, bounds_of, deps, eq, feasible, ge, implies,
                     implies_all, is_atom, le, lt, project, register_atoms, simplify, subtract)
from .egg import RmArr, RmIx, Term, check_equal, enc_ix, encode, rm_param
from .equations import (Equation, F, FBin, FConst, FParam, FRead, FSel, FSum, FUn, Reject, System,
                        ZERO, build_system, canonicalize, fsubst, fwalk, _freshen, _simplify_sel)
from .mlir_parse import parse_module
from .literal_policy import candidate_float_literals, literal_ok
from .symexec import NonAffine, Program, Unsupported, execute


@dataclass
class RegionMatch:
    span: tuple[int, int]
    spec: str                       # "S" or "S1+S2" for a composition
    launch_text: str
    binding: dict[str, str]
    evidence: str
    specs: list = field(default_factory=list)   # launched specifications, in order
    # manifest target marked recognition_only: the match is reported with its
    # binding, launch_text is empty and the span must not be rewritten
    recognition_only: bool = False


# ---------------------------------------------------------------------------
# Specifications
# ---------------------------------------------------------------------------

@dataclass
class Spec:
    name: str
    routine: str
    manifest: dict
    entry: dict
    system: System
    func_args: list                   # [(ssa, MLIR type)] of the spec function
    argnames: dict[str, str]          # spec func arg ssa -> C parameter name
    argtypes: list[tuple[str, str]]   # (C name, MLIR type) in order
    dims: dict[str, tuple[str, int]]  # int param name -> (array param, dim)
    source_stage: str
    validation: str
    fixed_ints: dict = field(default_factory=dict)   # C name -> literal (stride specialisation)
    sig: object = None


_SPEC_CACHE: dict[str, list[Spec]] = {}
_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_FORMAT = 4


def _c_prototype_dims(c_path: str) -> dict[str, tuple[str, int]]:
    """From `int lda, double A[][lda]` learn lda == dim 1 of A (generic VLA
    parsing; used only to materialise leading-dimension operands)."""
    out = {}
    try:
        src = open(c_path).read()
    except OSError:
        return out
    for m in re.finditer(r"\b(?:double|float)\s+(\w+)\s*((?:\[[^\]]*\])+)", src):
        name, dims = m.group(1), re.findall(r"\[([^\]]*)\]", m.group(2))
        for k, d in enumerate(dims):
            d = d.strip()
            if re.fullmatch(r"[A-Za-z_]\w*", d):
                out.setdefault(d, (name, k))
    return out


def _load_system(path: str, fname: str, fixed: Optional[dict] = None):
    text = open(path).read()
    funcs, aliases = parse_module(text)
    for f in funcs:
        if f.name == fname and f.body:
            prog = execute(f, aliases, fixed)
            window = {i for i, k in enumerate(prog.top_kinds) if k in ("work", "writeback")}
            return prog, build_system(prog, window)
    raise FileNotFoundError(f"{fname} not in {path}")


def _affinity_failure(ex: Exception) -> bool:
    return isinstance(ex, NonAffine) or "not an affine index" in str(ex)


def _spec_func_args(path: str, fname: str):
    funcs, _ = parse_module(open(path).read())
    for f in funcs:
        if f.name == fname:
            return f.args
    return None


def _stride_specialisations(func_args, cargs, dims, entry):
    """Candidate sets of int parameters to specialise to the literal 1
    (smallest first).  VLA extents are never specialised; the manifest
    assumptions must admit the value (literal_policy)."""
    ints = []
    for k, (ssa, ty) in enumerate(func_args):
        cname = cargs[k] if k < len(cargs) else ssa
        if ty.strip() in ("i32", "i64", "index") and cname not in dims \
                and literal_ok(entry, cname, 1, "int")[0]:
            ints.append((ssa, cname))
    for r in range(1, min(len(ints), 3) + 1):
        for combo in itertools.combinations(ints, r):
            yield {ssa: 1 for ssa, _ in combo}, {c: 1 for _, c in combo}


def _load_one(spec_dir, man, routine, dims, entry, log):
    name = entry["name"]
    stages = [("cgeist-affine", os.path.join(spec_dir, "raised", f"{name}.mlir")),
              ("debufferized", os.path.join(spec_dir, "raised", f"{name}_debuf.mlir"))]
    import_path = os.path.join(spec_dir, 'import_manifest.json')
    if os.path.exists(import_path):
        from lib_v2.spec_provenance import validated_stages
        with open(import_path) as source:
            imported = json.load(source).get('specifications', {}).get(name, {})
        permitted = validated_stages(imported)
        stages = [(stage, path) for stage, path in stages if stage in permitted]
    loaded, errors = {}, {}
    fixed_ssa, fixed_c = None, {}
    for stage, p in stages:
        if not os.path.exists(p):
            errors[stage] = "missing"
            continue
        try:
            loaded[stage] = _load_system(p, name, fixed_ssa)
        except (Reject, Unsupported, Exception) as ex:  # noqa: BLE001
            errors[stage] = f"{type(ex).__name__}: {ex}"
            if stage == "cgeist-affine" and _affinity_failure(ex):
                # generic stride specialisation: an int parameter that makes
                # the access non-affine is bound to the literal 1
                fargs = _spec_func_args(p, name) or []
                for fs, fc in _stride_specialisations(fargs, entry.get("args", []), dims, entry):
                    try:
                        loaded[stage] = _load_system(p, name, fs)
                        fixed_ssa, fixed_c = fs, fc
                        errors.pop(stage, None)
                        break
                    except (Reject, Unsupported, Exception):  # noqa: BLE001
                        continue
    if not loaded:
        if log:
            log(f"spec {name}: no usable stage ({errors})")
        return None
    stage = "cgeist-affine" if "cgeist-affine" in loaded else "debufferized"
    prog, system = loaded[stage]
    cargs = entry.get("args", [])
    argnames = {ssa: (cargs[k] if k < len(cargs) else ssa) for k, (ssa, ty) in enumerate(prog.func.args)}
    argtypes = [(argnames[ssa], ty) for ssa, ty in prog.func.args]
    spec = Spec(name, routine, man, entry, system, list(prog.func.args), argnames, argtypes, dims,
                stage, "", dict(fixed_c))
    spec.source_dir = spec_dir
    if "cgeist-affine" in loaded and "debufferized" in loaded:
        dprog, dsys = loaded["debufferized"]
        ok, info = validate_stages(system, dsys)
        spec.validation = ("debufferized stage equivalent to cgeist stage" if ok
                           else f"debufferized stage NOT equivalent ({info})")
        if dsys.warnings:
            spec.validation += ("; under memref (sequential, aliasing) semantics only -- "
                                + "; ".join(dsys.warnings)
                                + "; tensor-value semantics of this IR differ")
    else:
        spec.validation = f"single stage only ({errors})"
    if fixed_c:
        spec.validation += f"; int parameter(s) specialised: {fixed_c}"
    return spec


def _load_specs_uncached(spec_dir: str, log=None) -> list[Spec]:
    from .compose import signature
    specs = []
    for mpath in sorted(glob.glob(os.path.join(spec_dir, "manifest", "*.json"))):
        man = json.load(open(mpath))
        routine = man.get("routine", os.path.basename(mpath)[:-5])
        dims = _c_prototype_dims(os.path.join(spec_dir, "ref_c", f"{routine}.c"))
        for entry in man.get("specs", []):
            sp = _load_one(spec_dir, man, routine, dims, entry, log)
            if sp is not None:
                sp.sig = signature(sp.system)
                specs.append(sp)
    return specs


def _cache_key(spec_dir: str) -> str:
    """Content hash of everything the parsed spec systems depend on: the
    matcher sources (and kernel_match, whose rules validate stages), the
    manifests, the C references (VLA dims) and every raised spec file."""
    h = hashlib.sha256()
    from kernel_match import numeric_model
    h.update(numeric_model().encode())
    h.update(os.environ.get("EQSAT_ABLATION", "full").encode())
    h.update(f"fmt{_CACHE_FORMAT} py{sys.version_info[:3]}".encode())
    files = sorted(glob.glob(os.path.join(_HERE, "*.py")))
    files.append(os.path.join(os.path.dirname(_HERE), "kernel_match.py"))
    files += sorted(glob.glob(os.path.join(spec_dir, "manifest", "*.json")))
    files += sorted(glob.glob(os.path.join(spec_dir, "ref_c", "*.c")))
    files += sorted(glob.glob(os.path.join(spec_dir, "raised", "*.mlir")))
    files += glob.glob(os.path.join(spec_dir, 'import_manifest.json'))
    for f in files:
        h.update(os.path.relpath(f, spec_dir).encode())
        try:
            with open(f, "rb") as fh:
                h.update(hashlib.sha256(fh.read()).digest())
        except OSError:
            h.update(b"<missing>")
    try:
        import egglog
        h.update(str(getattr(egglog, "__version__", "")).encode())
    except Exception:  # noqa: BLE001
        pass
    return h.hexdigest()


def cache_dir() -> str:
    return os.environ.get("REGION_MATCH_CACHE_DIR",
                          os.path.join(os.path.expanduser("~"), ".cache", "polygeist_region_match"))


def load_specs(spec_dir: str, log=None, use_cache: Optional[bool] = None) -> list[Spec]:
    """Parsed specification systems.  Memoised per process and cached on disk
    (pickle keyed by `_cache_key`; any change to a raised spec file, manifest,
    C reference or matcher source invalidates it).  REGION_MATCH_NO_CACHE=1
    disables the disk cache."""
    key = os.path.abspath(spec_dir)
    if key in _SPEC_CACHE:
        return _SPEC_CACHE[key]
    if use_cache is None:
        use_cache = os.environ.get("REGION_MATCH_NO_CACHE", "") in ("", "0")
    path = None
    if use_cache:
        ck = _cache_key(spec_dir)
        path = os.path.join(cache_dir(), f"specs-{ck[:32]}.pkl")
        try:
            with open(path, "rb") as fh:
                specs, logs = pickle.load(fh)
            if log:
                for m in logs:
                    log(m)
            for s in specs:      # index atoms of systems built in another process
                register_atoms(getattr(s.system, "atoms", {}))
            _SPEC_CACHE[key] = specs
            return specs
        except (OSError, EOFError, pickle.PickleError, AttributeError, ImportError):
            pass
    logs: list[str] = []
    specs = _load_specs_uncached(spec_dir, logs.append)
    if log:
        for m in logs:
            log(m)
    if path:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            old = sys.getrecursionlimit()
            sys.setrecursionlimit(max(old, 20000))
            try:
                with open(tmp, "wb") as fh:
                    pickle.dump((specs, logs), fh, protocol=pickle.HIGHEST_PROTOCOL)
            finally:
                sys.setrecursionlimit(old)
            os.replace(tmp, path)
        except (OSError, pickle.PickleError, RecursionError):
            pass
    _SPEC_CACHE[key] = specs
    return specs


# ---------------------------------------------------------------------------
# Binding search and comparison
# ---------------------------------------------------------------------------

@dataclass
class Binding:
    arrays: dict[str, tuple[str, str, tuple]]     # spec arr -> (app_in, app_out, perm)
    params: dict[str, object]                     # spec index param -> app param | Aff
    fparams: dict[str, object]                    # spec float param -> app param | float literal

    def literals(self) -> int:
        return sum(1 for v in self.fparams.values() if isinstance(v, float))

    def cost(self):
        perms = sum(1 for (_, _, p) in self.arrays.values() if list(p) != sorted(p))
        copies = sum(1 for (i, o, _) in self.arrays.values() if o and i and i != o)
        return (perms, copies, self.literals())


def _used_params(sysm: System) -> set[str]:
    out = set()
    for e in sysm.equations:
        for c in e.domain:
            out |= {v for v in deps(c.vars()) if v.startswith("p")}
        for x in fwalk(e.formula):
            if isinstance(x, FRead):
                for a in x.idx:
                    out |= {v for v in deps(a.vars()) if v.startswith("p")}
            if isinstance(x, FSum):
                out |= {v for v in deps(x.lo.vars() | x.hi.vars()) if v.startswith("p")}
    return out


def _arr_info(sysm: System, name: str):
    b = sysm.arrays.get(name)
    return (b.rank, b.elem) if b else None


def _fun_ops(sysm: System) -> set[str]:
    return {x.op for e in sysm.equations for x in fwalk(e.formula) if isinstance(x, FUn)}


def _fparams(sysm: System) -> list[str]:
    return sorted({x.name for e in sysm.equations for x in fwalk(e.formula)
                   if type(x).__name__ == "FParam"})


def _constants(sysm: System) -> set[float]:
    return {x.v for e in sysm.equations for x in fwalk(e.formula) if isinstance(x, FConst)}


def fparam_assignments(s_fp: list[str], a_fp: list[str], lits: list[float], lit_ok=None):
    """Every spec float parameter binds to an application float parameter or
    to a literal (non-injective allowed); every application parameter must be
    used.  Bindings with fewer literals come first."""
    if not s_fp:
        if not a_fp:
            yield {}
        return
    choices = []
    for sp in s_fp:
        c = list(a_fp) + [v for v in lits if lit_ok is None or lit_ok(sp, v)]
        choices.append(c)
    out = []
    need = set(a_fp)
    for combo in itertools.product(*choices):
        if not need <= {c for c in combo if isinstance(c, str)}:
            continue
        out.append(dict(zip(s_fp, combo)))
    out.sort(key=lambda d: sum(1 for v in d.values() if isinstance(v, float)))
    yield from out


def _index_part(a: Aff) -> Aff:
    return Aff({v: c for v, c in a.terms if v.startswith(("v", "r"))})


def _closed_value(a: Aff) -> bool:
    """Only parameters and atoms over them: a value fixed before the region."""
    return all(not v.startswith(("v", "r")) for v in deps(a.vars()))


def _solve_param(sa: Aff, aa: Aff, s_params: list) -> Optional[tuple[str, Aff]]:
    """If spec expression sa holds exactly one spec parameter s (coefficient
    +-1) and has the same index-variable part as application expression aa,
    the value of s that makes them equal."""
    ps = [(v, k) for v, k in sa.terms if not v.startswith(("v", "r"))]
    if len(ps) != 1 or ps[0][0] not in s_params or abs(ps[0][1]) != 1:
        return None
    (sp, k), = ps
    ipart = _index_part(sa)
    if _index_part(aa) != ipart:
        return None
    rest = aa - ipart
    if not rest.terms or not _closed_value(rest):
        return None
    return sp, (rest - Aff.cst(sa.const)).scale(k)


def _sum_bounds(f: F) -> list[tuple[Aff, Aff]]:
    return [(x.lo, x.hi) for x in fwalk(f) if isinstance(x, FSum)]


def _derived_param_values(app: System, spec: System, out_map: dict, s_outs: list, operm,
                          s_params: list, a_params: list) -> dict[str, list[Aff]]:
    """Spec size parameter -> application values it can stand for, read off
    corresponding bounds -- written element domains and summation ranges: a
    spec bound with exactly one parameter s (coefficient +-1) and the same
    index-variable part as an application bound gives s as an affine
    expression of application values, e.g. s = hi - lo + 1, or
    s = @size[0] (an extent loaded from memory).  Candidates only extend the
    search: every binding is still checked by the domain cover test and
    proved by compare_systems."""
    out: dict[str, list[Aff]] = {}
    plain = {Aff.var(a) for a in a_params}
    for s_out, perm in zip(s_outs, operm):
        rank = spec.arrays[s_out].rank
        vren = {f"v{k}": Aff.var(f"v{perm[k]}") for k in range(rank)}
        s_eqs = [e for e in spec.equations if e.arr == s_out]
        a_eqs = [e for e in app.equations if e.arr == out_map[s_out]]
        pairs = [(cs.aff.subst(vren), ca.aff)
                 for es in s_eqs for cs in es.domain for ea in a_eqs for ca in ea.domain
                 if cs.kind == ca.kind]
        for es in s_eqs:
            sb = _sum_bounds(es.formula)
            for ea in a_eqs:
                ab = _sum_bounds(ea.formula)
                pairs += [(x.subst(vren), y) for (sl, sh) in sb for (al, ah) in ab
                          for x, y in ((sl, al), (sh, ah))]
        for sa, aa in pairs:
            r = _solve_param(sa, aa, s_params)
            if r and r[1] not in plain and r[1] not in out.setdefault(r[0], []):
                out[r[0]].append(r[1])
    return out


def _covers(xs: list[list[Cons]], ys: list[list[Cons]]) -> bool:
    """Every y is contained in the union of xs."""
    for y in ys:
        rem = [y]
        for x in xs:
            nxt = []
            for r in rem:
                nxt.extend(subtract(r, x))
            rem = nxt
            if not rem:
                break
        if any(feasible(r) for r in rem):
            return False
    return True


def enumerate_bindings(app: System, spec: System, cache: Optional[dict] = None, lit_ok=None,
                       literals: bool = True, alias_guard: bool = True):
    """Generic search over operand / layout / parameter bindings.

    Pruning (affects completeness only, never soundness -- every yielded
    binding is still proved by compare_systems):
      * outputs are matched bijectively by rank and element type,
      * the parameter binding must make the written element domains equal,
      * an input role may only bind to an application array that the
        application reads in the same version (value on entry)."""
    if cache is None:
        cache = {}
    s_outs = sorted(spec.outputs)
    a_outs = sorted(app.outputs)
    if len(s_outs) != len(a_outs) or _fun_ops(spec) != _fun_ops(app):
        return
    s_in_roles = sorted({a for (a, v) in spec.reads if v == "in"} | set(s_outs))
    a_read_in = sorted({a for (a, v) in app.reads if v == "in"})
    s_params = sorted(_used_params(spec))
    a_params = sorted(_used_params(app))
    s_fp, a_fp = _fparams(spec), _fparams(app)
    lits = candidate_float_literals(_constants(app)) if literals else []
    if len(s_fp) < len(a_fp) or (not lits and len(s_fp) != len(a_fp)):
        return
    fassigns = list(fparam_assignments(s_fp, a_fp, lits, lit_ok))
    if not fassigns:
        return
    for out_assign in itertools.permutations(a_outs):
        if any(_arr_info(spec, s_) != _arr_info(app, a_) for s_, a_ in zip(s_outs, out_assign)):
            continue
        out_map = dict(zip(s_outs, out_assign))
        out_perms = [list(itertools.permutations(range(spec.arrays[r].rank))) for r in s_outs]
        for operm in itertools.product(*out_perms):
            # non-injective: two spec extents may be the same application
            # extent (e.g. m and n of a square matrix); an extent may also be
            # an expression of application values (_derived_param_values)
            derived = _derived_param_values(app, spec, out_map, s_outs, operm, s_params, a_params)
            choices = [list(a_params) + derived.get(sp, []) for sp in s_params]
            for pperm in itertools.product(*choices):
                params = dict(zip(s_params, pperm))
                ok = True
                for s_out, perm in zip(s_outs, operm):
                    key = ("dom", s_out, out_map[s_out], perm, tuple(sorted(params.items())))
                    if key not in cache:
                        tmp = Binding({s_out: (None, out_map[s_out], perm)}, params, {})
                        ren = _rename_map(tmp, s_out, spec.arrays[s_out].rank)
                        sd = [[c.subst(ren) for c in e.domain] for e in spec.equations if e.arr == s_out]
                        ad = [e.domain for e in app.equations if e.arr == out_map[s_out]]
                        cache[key] = _covers(ad, sd) and _covers(sd, ad)
                    if not cache[key]:
                        ok = False
                        break
                if not ok:
                    continue
                role_choices = []
                for role in s_in_roles:
                    info = _arr_info(spec, role)
                    rank = info[0]
                    if role in out_map:
                        perm = operm[s_outs.index(role)]
                        ins = [out_map[role]]
                        if (role, "in") in spec.reads:
                            ins += [a for a in a_read_in if a != out_map[role]
                                    and _arr_info(app, a) == info and a not in app.outputs]
                        role_choices.append([(role, i, out_map[role], perm) for i in ins])
                    else:
                        perms = list(itertools.permutations(range(rank)))
                        cands = [a for a in a_read_in if _arr_info(app, a) == info]
                        role_choices.append([(role, i, None, p) for i in cands for p in perms])
                for combo in itertools.product(*role_choices):
                    arrays = {r: (i, o, p) for (r, i, o, p) in combo}
                    if alias_guard and not _copy_ins_safe(arrays):
                        continue
                    for fa in fassigns:
                        yield Binding(arrays, params, dict(fa))


def _copy_ins_safe(arrays: dict) -> bool:
    """No role of one launch may read memory that another role of the same
    launch writes: BLAS forbids aliasing a modified argument (Fortran dummy
    argument rules), and a copy-in X -> Y overwrites Y before the launch."""
    written = {o for (_i, o, _p) in arrays.values() if o}
    for (i, o, _p) in arrays.values():
        if i in written and o != i:
            return False
    return True


def _rename_map(b: Binding, out_arr: str, rank: int) -> dict[str, Aff]:
    perm = b.arrays[out_arr][2] if out_arr in b.arrays else tuple(range(rank))
    m = {f"v{k}": Aff.var(f"v{perm[k]}") for k in range(rank)}
    for s, a in b.params.items():
        m[s] = a if isinstance(a, Aff) else Aff.var(a)
    # index atoms name their array: rename it with the operand binding
    for role, (i_arr, _o, _p) in b.arrays.items():
        if i_arr is not None:
            m["@arr:" + role] = Aff.var(i_arr)
    return m


def _param_equalities(params: dict[str, str]) -> list[Cons]:
    """Spec parameters bound to the same application parameter are equal."""
    by_app: dict[str, list[str]] = {}
    for s_, a_ in params.items():
        by_app.setdefault(a_, []).append(s_)
    out = []
    for group in by_app.values():
        for x in group[1:]:
            out.append(eq(Aff.var(group[0]), Aff.var(x)))
    return out


def _inverse_map(m: dict[str, Aff]) -> dict[str, Aff]:
    """Application name -> spec name for the plain renamings in m (spec
    parameters bound to expressions are related separately, see
    _expression_context)."""
    inv = {}
    for k, v in m.items():
        if len(v.terms) != 1 or v.terms[0][1] != 1 or v.const:
            continue
        (name, _), = v.terms
        if k.startswith("@arr:"):
            inv["@arr:" + name] = Aff.var(k[len("@arr:"):])
        else:
            inv[name] = Aff.var(k)
    return inv


def _expression_context(params: dict, inv: dict[str, Aff]) -> tuple[dict[str, Aff], list[Cons]]:
    """For spec parameters bound to expressions: extend the inverse map so
    application names left in the expression (and arrays of its atoms) are
    renamed apart from spec names, and return the defining equalities
    s == expr in that namespace."""
    xb = {s_: a_ for s_, a_ in params.items() if isinstance(a_, Aff)}
    if not xb:
        return inv, []
    inv = dict(inv)
    for a_ in xb.values():
        for v in deps(a_.vars()):
            if is_atom(v):
                arr = atom_info(v)[0]
                inv.setdefault("@arr:" + arr, Aff.var("x:" + arr))
            elif v not in inv:
                inv[v] = Aff.var("x:" + v)
    return inv, [eq(Aff.var(s_), a_.subst(inv)) for s_, a_ in xb.items()]


def _simplify_formula(f: F, ctx: list[Cons]) -> F:
    f = _simplify_sel(f, ctx)
    return _fold_zero(f)


def _fold_zero(f: F) -> F:
    from kernel_match import numeric_model
    if numeric_model() == 'strict':
        return f
    if isinstance(f, FBin):
        a, b = _fold_zero(f.a), _fold_zero(f.b)
        if f.op in ("add", "sub") and b == ZERO:
            return a
        if f.op == "add" and a == ZERO:
            return b
        if f.op == "mul" and (a == ZERO or b == ZERO):
            return ZERO
        return FBin(f.op, a, b)
    if isinstance(f, FUn):
        return FUn(f.op, _fold_zero(f.a))
    if isinstance(f, FSel):
        return FSel(f.cond, _fold_zero(f.a), _fold_zero(f.b))
    if isinstance(f, FSum):
        body = _fold_zero(f.body)
        if body == ZERO:
            return ZERO
        return FSum(f.var, f.lo, f.hi, body)
    return f


def _mask_facts(f: F, ctx: list[Cons], out: list[tuple[Aff, Aff]]):
    """Facts u <= hi / lo <= l for masked reductions, proved under ctx.
    Facts that mention a bound summation variable are not emitted: egglog
    facts are global, and canonical bound names are reused by sibling sums."""
    _mask_facts_rec(f, ctx, out)
    out[:] = [(a, b) for (a, b) in out
              if not any(v.startswith("r") and v[1:].isdigit() for v in a.vars() | b.vars())]


def _mask_facts_rec(f: F, ctx: list[Cons], out: list[tuple[Aff, Aff]]):
    if isinstance(f, FSum):
        r = f.var
        inner = ctx + [ge(Aff.var(r), f.lo), lt(Aff.var(r), f.hi)]
        body = f.body
        while isinstance(body, FSel) and body.cond[0] == "aff" and body.b == ZERO:
            for c in body.cond[1]:
                k = c.aff.coeff(r)
                if c.kind != "ge" or abs(k) != 1:
                    continue
                rest = c.aff - Aff.var(r, k)
                if k == -1:
                    u = rest + 1
                    if implies(ctx, le(u, f.hi)):
                        out.append((u, f.hi))
                    if implies(ctx, le(f.hi, u)):
                        out.append((f.hi, u))
                else:
                    l_ = -rest
                    if implies(ctx, le(f.lo, l_)):
                        out.append((f.lo, l_))
            body = body.a
        _mask_facts_rec(f.body, inner, out)
        # empty range
        if implies(ctx, le(f.hi, f.lo)):
            out.append((f.hi, f.lo))
        return
    if isinstance(f, FBin):
        _mask_facts_rec(f.a, ctx, out)
        _mask_facts_rec(f.b, ctx, out)
    elif isinstance(f, FUn):
        _mask_facts_rec(f.a, ctx, out)
    elif isinstance(f, FSel):
        _mask_facts_rec(f.a, ctx, out)
        _mask_facts_rec(f.b, ctx, out)


def _pieces(doms: list[list[Cons]]) -> list[list[Cons]]:
    """Disjoint convex pieces of the union of doms, each inside-or-outside of
    every dom."""
    pieces: list[list[Cons]] = []
    for d in doms:
        rem = [d]
        for q in pieces:
            nxt = []
            for r in rem:
                nxt.extend(subtract(r, q))
            rem = nxt
        pieces.extend(rem)
    changed = True
    rounds = 0
    while changed and rounds < 30:
        changed = False
        rounds += 1
        for i, P in enumerate(pieces):
            for d in doms:
                if implies_all(P, d) or not feasible(P + d):
                    continue
                pieces = pieces[:i] + [P + d] + subtract(P, d) + pieces[i + 1:]
                changed = True
                break
            if changed:
                break
    out = []
    for P in pieces:
        s = simplify(P)
        if s is not None:
            out.append(s)
    return out


@dataclass
class CompareResult:
    ok: bool
    reason: str = ""
    egg: list = field(default_factory=list)       # EggResult per piece
    pieces: int = 0


def _piece_equalities(P: list[Cons]) -> dict[str, Aff]:
    """Solved equalities of a piece (e.g. the diagonal v0 == v1, or a last
    row v1 == n - 1 given as two opposite inequalities)."""
    ges = {c.aff for c in P if c.kind == "ge"}
    eqs = [c.aff for c in P if c.kind == "eq"] + \
          [a for a in ges if (-a) in ges and repr(a) < repr(-a)]
    sol: dict[str, Aff] = {}
    for a in eqs:
        a = a.subst(sol)
        for v, k in a.terms:
            if abs(k) == 1:
                rest = a - Aff.var(v, k)
                val = rest.scale(-1) if k == 1 else rest
                sol = {x: y.subst({v: val}) for x, y in sol.items()}
                sol[v] = val
                break
    return sol


def _read_set(f: F, arr_key, idx_map) -> set:
    out = set()
    for x in fwalk(f):
        if isinstance(x, FRead):
            k = arr_key(x.arr, x.ver)
            if k is None:
                return None
            out.add((k[0], tuple(idx_map(k[1], x.idx))))
    return out


def compare_systems(app: System, spec: System, b: Binding, ns_spec="s:",
                    cache: Optional[dict] = None) -> CompareResult:
    if cache is None:
        cache = {}
    res = CompareResult(True)
    for s_out, (a_in, a_out, perm) in [(k, v) for k, v in b.arrays.items() if v[1] is not None]:
        rank = spec.arrays[s_out].rank
        ren = _rename_map(b, s_out, rank)
        inv, xctx = _expression_context(b.params, _inverse_map(ren))
        pkey = ("pieces", s_out, a_out, tuple(perm), tuple(sorted(b.params.items())))
        if pkey not in cache:
            s_eqs = [(e, [c.subst(ren) for c in e.domain]) for e in spec.equations if e.arr == s_out]
            a_eqs = [(e, e.domain) for e in app.equations if e.arr == a_out]
            pieces = _pieces([d for _, d in s_eqs] + [d for _, d in a_eqs])
            prepared = []
            for P in pieces:
                fa = next((e.formula for e, d in a_eqs if implies_all(P, d)), None)
                if fa is None:
                    fa = FRead(a_out, "in", [Aff.var(f"v{k}") for k in range(rank)])
                Ps = [c.subst(inv) for c in P] + _param_equalities(b.params) + xctx
                fs = next((e.formula for e, d in s_eqs if implies_all(P, d)), None)
                if fs is None:
                    fs = FRead(s_out, "in", [Aff.var(f"v{k}") for k in range(rank)])
                fa = _simplify_formula(fa, P)
                fs = _simplify_formula(fs, Ps)
                facts_a, facts_s = [], []
                _mask_facts(fa, P, facts_a)
                _mask_facts(fs, Ps, facts_s)
                prepared.append((P, fa, fs, facts_a, facts_s, _piece_equalities(P)))
            cache[pkey] = prepared
        prepared = cache[pkey]
        res.pieces += len(prepared)
        # cheap necessary condition: same set of element reads under the binding
        for (P, fa, fs, facts_a, facts_s, peq) in prepared:
            def app_key(arr, ver):
                return (f"{ver}:{arr}", None)

            def spec_key(arr, ver):
                if arr not in b.arrays:
                    return None
                i_arr, o_arr, p = b.arrays[arr]
                tgt = i_arr if ver == "in" else o_arr
                return None if tgt is None else (f"{ver}:{tgt}", p)

            def norm(a: Aff) -> Aff:
                return a.subst(peq)

            ra = _read_set(fa, app_key, lambda p, idx: [norm(a) for a in idx])

            def spec_idx(p, idx):
                idx = [norm(a.subst(ren)) for a in idx]
                if p is None:
                    return idx
                inv_p = [0] * len(p)
                for k, pk in enumerate(p):
                    inv_p[pk] = k
                return [idx[inv_p[d]] for d in range(len(p))]
            rs = _read_set(fs, spec_key, spec_idx)
            if rs is None or ra != rs:
                res.ok = False
                res.reason = f"element reads differ on piece {P}: app {sorted(map(str, ra or []))} vs spec {sorted(map(str, rs or []))}"
                return res
        for (P, fa, fs, facts_a, facts_s, peq) in prepared:
            app_name = lambda arr, ver: f"{ver}:{arr}"
            spec_name = lambda arr, ver: f"{ns_spec}{ver}:{arr}"
            ta = encode(fa, "", app_name)
            ts = encode(fs, ns_spec, spec_name)
            unions = []
            # operand binding
            for role, (i_arr, o_arr, p) in b.arrays.items():
                ident = list(p) == list(range(len(p)))
                for ver, arr in (("in", i_arr), ("out", o_arr)):
                    if arr is None:
                        continue
                    target = RmArr(f"{ver}:{arr}")
                    if not ident:
                        target = RmArr.perm(target, "".join(map(str, p)))
                    unions.append((RmArr(f"{ns_spec}{ver}:{role}"), target))
            for s_, a_ in b.fparams.items():
                if isinstance(a_, float):
                    # literal binding: the spec parameter IS the literal
                    unions.append((rm_param(RmArr(f"{ns_spec}param:{s_}")), Term.Lit(a_)))
                else:
                    unions.append((RmArr(f"{ns_spec}param:{s_}"), RmArr(f"param:{a_}")))
            # index renaming (atom arrays are unified by the operand binding)
            for k, v in ren.items():
                if not k.startswith("@arr:"):
                    unions.append((RmIx(ns_spec + k), enc_ix(v, "")))
            # equalities of the piece
            for v, val in peq.items():
                unions.append((RmIx(v), enc_ix(val, "")))
            facts = [(enc_ix(x, ""), enc_ix(y, "")) for x, y in facts_a] + \
                    [(enc_ix(x, ns_spec), enc_ix(y, ns_spec)) for x, y in facts_s]
            er = check_equal(ta, ts, unions, facts)
            res.egg.append(er)
            if not er.ok:
                res.ok = False
                res.reason = f"formulas differ on piece {P}: app {fa!r} vs spec {fs!r}"
                return res
    return res


def _identity_fparams(*systems: System) -> dict[str, str]:
    names = set()
    for sysm in systems:
        for e in sysm.equations:
            names |= {x.name for x in fwalk(e.formula) if type(x).__name__ == "FParam"}
    return {n: n for n in names}


def validate_stages(ref: System, other: System) -> tuple[bool, str]:
    """Translation validation of a later pipeline stage of the same function."""
    if sorted(ref.outputs) != sorted(other.outputs):
        return False, f"outputs {sorted(ref.outputs)} vs {sorted(other.outputs)}"
    arrays = {}
    for a in sorted(ref.arrays):
        r = ref.arrays[a].rank
        arrays[a] = (a, a if a in ref.outputs else None, tuple(range(r)))
    params = {p: p for p in _used_params(ref) | _used_params(other)}
    b = Binding(arrays, params, _identity_fparams(ref, other))
    r = compare_systems(other, ref, b)
    return r.ok, r.reason[:300]




# ---------------------------------------------------------------------------
# Regions of the application
# ---------------------------------------------------------------------------

def _defined_names(op) -> set[str]:
    out = set(r.split("#")[0] for r in op.results)
    for reg in op.regions:
        for o in reg:
            out |= _defined_names(o)
    return out


def _windows(prog: Program):
    kinds = prog.top_kinds
    work = [i for i, k in enumerate(kinds) if k == "work"]
    wins = []
    for a in range(len(work)):
        for b in range(a, len(work)):
            lo, hi = work[a], work[b]
            if any(kinds[k] == "opaque" for k in range(lo, hi + 1)):
                continue
            end = hi
            j = hi + 1
            while j < len(kinds) and kinds[j] in ("writeback", "pure"):
                if kinds[j] == "writeback":
                    end = j
                j += 1
            wins.append((lo, hi, end))
    wins.sort(key=lambda w: (-(w[1] - w[0]), w[0]))
    return wins


def _live_outs(text: str, prog: Program, lo: int, end: int) -> list[str]:
    """Names defined in the window and used after it."""
    ops = prog.top_ops
    defined = set()
    for op in ops[lo:end + 1]:
        defined |= _defined_names(op)
    tail = text[ops[end].end:prog.func.end]
    return sorted(n for n in defined if re.search(re.escape(n) + r"(?![\w$.-])", tail))


_REMAT_INDEX_OPS = ("arith.constant", "arith.index_cast", "arith.addi", "arith.subi", "arith.muli",
                 "affine.apply")


def _tensor_type_of(memref_ty: str) -> str:
    m = re.match(r"memref<(.*)>$", memref_ty.strip())
    body = m.group(1) if m else memref_ty
    from .mlir_parse import _split_top
    return f"tensor<{_split_top(body)[0].strip()}>"


def _result_type(line: str) -> Optional[str]:
    m = re.search(r"->\s*(\S+)\s*$", line.strip())
    return m.group(1) if m else None


def _rematerialise(prog: Program, lo: int, end: int, names: list[str], outputs: set[str]):
    """Re-definitions (after the launches) of values that the window defines
    and later code uses: the full current tensor of a buffer the launches
    leave in memory becomes `bufferization.to_tensor <memref>`; index
    arithmetic over values defined before the window is re-emitted.  Returns
    None if some live-out cannot be re-defined."""
    from .symexec import TensorVal
    sync_lo = prog.snaps[lo][0]
    final_ver = prog.snaps[end + 1][2]
    win_defs = {}
    for op in prog.top_ops[lo:end + 1]:
        for r in op.results:
            win_defs[r.split("#")[0]] = op
    inside = set()
    for op in prog.top_ops[lo:end + 1]:
        inside |= _defined_names(op)
    lines = []
    for n in names:
        v = prog.env.get(n)
        op = win_defs.get(n)
        if isinstance(v, TensorVal) and v.view is None and v.buf.kind == "arg" \
                and v.ver == final_ver.get(v.buf.id) and op is not None and len(op.results) == 1:
            if v.buf.name not in outputs and not sync_lo.get(v.buf.id, True):
                return None
            ty = _tensor_type_of(v.buf.arg_type)
            want = _result_type(op.line) or ty
            if want == ty:
                lines.append(f"{n} = bufferization.to_tensor {v.buf.name} : {v.buf.arg_type}")
            else:
                tmp = f"{n}_rm_t"
                lines.append(f"{tmp} = bufferization.to_tensor {v.buf.name} : {v.buf.arg_type}")
                lines.append(f"{n} = tensor.cast {tmp} : {ty} to {want}")
            continue
        if op is not None and op.name in _REMAT_INDEX_OPS and not op.regions and len(op.results) == 1:
            rhs = op.line.split("=", 1)[1]
            if any(o.split("#")[0] in inside for o in re.findall(r"%[\w$.#-]+", rhs)):
                return None
            lines.append(op.line.strip())
            continue
        return None
    return lines


def _sync_plan(prog: Program, app: System, lo: int) -> Optional[dict]:
    """For every application array the launches touch: None if its memref
    holds the current value at window start, else the SSA name of its current
    full tensor (to be materialised with bufferization.to_memref)."""
    from .symexec import TensorVal
    synced, cur_full, versions = prog.snaps[lo]
    plan = {}
    need = {a for (a, v) in app.reads if v == "in"} | set(app.outputs)
    for a in sorted(need):
        buf = prog.arg_buffers.get(a)
        if buf is None:
            return None
        if synced.get(buf.id, True):
            plan[a] = None
            continue
        cur = cur_full.get(buf.id)
        v = prog.env.get(cur) if cur else None
        if not (isinstance(v, TensorVal) and v.view is None and v.ver == versions.get(buf.id)):
            return None
        plan[a] = cur
    return plan


def _written_back(prog: Program, app: System, lo: int, end: int) -> bool:
    """Tensor-level writes in the window are visible to the caller only if
    the buffer is written back at or after the window (otherwise a launch on
    the memref would add a side effect)."""
    for a in app.outputs:
        buf = prog.arg_buffers.get(a)
        if buf is None:
            return False
        tw = [t for t in prog.tensor_writes.get(buf.id, []) if lo <= t <= end]
        if tw and not any(t >= lo for t in prog.writebacks.get(buf.id, [])):
            return False
    return True


# ---------------------------------------------------------------------------
# Launch emission
# ---------------------------------------------------------------------------

_FLOAT_TYS = ("f64", "f32", "f16", "bf16")


def fmt_float(v: float) -> str:
    r = repr(float(v))
    if "e" not in r and "." not in r:
        r += ".0"
    return r


def _elem_type(memref_ty: str) -> str:
    m = re.search(r"x?(f64|f32|f16|bf16|i64|i32|i8|i1|index)\s*(?:,|>)", memref_ty)
    return m.group(1) if m else "f64"


def _rank(memref_ty: str) -> int:
    m = re.match(r"(?:memref|tensor)<(.*)>$", memref_ty.strip())
    if not m:
        return 0
    from .mlir_parse import _split_top
    return len(_split_top(m.group(1))[0].split("x")) - 1


def _shape_part(ty: str) -> str:
    from .mlir_parse import _split_top
    m = re.match(r"(?:memref|tensor)<(.*)>$", ty.strip())
    return _split_top(m.group(1))[0].strip() if m else ty


def _layout_part(ty: str) -> str:
    from .mlir_parse import _split_top
    m = re.match(r"(?:memref|tensor)<(.*)>$", ty.strip())
    return ",".join(p.strip() for p in _split_top(m.group(1))[1:]) if m else ""


def static_strides(memref_ty: str) -> list:
    """Static strides of a memref type (None = dynamic); identity layout gives
    a unit innermost stride."""
    r = _rank(memref_ty)
    m = re.search(r"strided<\[([^\]]*)\]", memref_ty)
    if m:
        out = []
        for t in m.group(1).split(","):
            t = t.strip()
            out.append(int(t) if re.fullmatch(r"-?\d+", t) else None)
        return out
    if "affine_map" in memref_ty:
        return [None] * r
    return [None] * (r - 1) + [1] if r else []


class _Emit:
    """Accumulates the lines of one region replacement."""

    def __init__(self, prog: Program, uid: str):
        self.prog = prog
        self.uid = uid
        self.lines: list[str] = []
        self.n = 0
        self.app_args = dict(prog.func.args)
        self.md: dict[str, str] = {}
        self.consts: dict = {}
        self.operand: dict[str, str] = {}      # app array -> memref operand
        self.notes: list[str] = []

    def fresh(self, stem: str) -> str:
        self.n += 1
        return f"%{self.uid}_{stem}{self.n}"

    def const(self, v, ty: str) -> str:
        key = (repr(v), ty)
        if key not in self.consts:
            nm = self.fresh("c")
            txt = fmt_float(v) if ty in _FLOAT_TYS else str(int(v))
            self.lines.append(f"{nm} = arith.constant {txt} : {ty}")
            self.consts[key] = nm
        return self.consts[key]

    def prepare(self, plan: dict, written: set[str]):
        """Materialise arrays whose current value is not in memory: outputs are
        synchronised into their memref (copy of the current tensor); inputs
        are passed as bufferization.to_memref of the current tensor."""
        for a, cur in sorted(plan.items()):
            if cur is None:
                self.operand[a] = a
                continue
            ty = self.app_args.get(a)
            m = self.fresh("cur")
            self.lines.append(f"{m} = bufferization.to_memref {cur} : {ty}")
            if a in written:
                self.lines.append(f"memref.copy {m}, {a} : {ty} to {ty}")
                self.operand[a] = a
                self.notes.append(f"{a}: current value {cur} copied into memory before the launch")
            else:
                self.operand[a] = m
                self.notes.append(f"{a}: current value {cur} passed via bufferization.to_memref")

    def mem(self, a: str) -> str:
        return self.operand.get(a, a)

    def metadata(self, a: str) -> tuple[str, int]:
        ty = self.app_args.get(a)
        if ty is None or not ty.startswith("memref"):
            raise Reject(f"no memref for {a}")
        r = _rank(ty)
        if a not in self.md:
            nm = self.fresh("md")
            res = ", ".join(["index"] * (1 + 2 * r))
            self.lines.append(f"{nm}:{2 + 2 * r} = memref.extract_strided_metadata {self.mem(a)} : "
                              f"{ty} -> memref<{_elem_type(ty)}>, {res}")
            self.md[a] = nm
        return self.md[a], r

    def leading_dim(self, a: str, k: int, ty: str) -> tuple[str, str]:
        """Extent of C dimension k (k >= 1) of array `a` as laid out in memory:
        stride[k-1] / stride[k] (= stride[k-1] for a unit innermost stride),
        read with memref.extract_strided_metadata.  Returns (operand, text)."""
        md, r = self.metadata(a)
        if not (1 <= k < r):
            raise Reject(f"leading dimension {k} of a rank-{r} array")
        strides = static_strides(self.app_args[a])
        s_hi = f"{md}#{2 + r + k - 1}"
        if k == r - 1 and strides and strides[-1] == 1:
            val, desc = s_hi, f"stride{k - 1}({a})"
        else:
            q = self.fresh("ldq")
            self.lines.append(f"{q} = arith.divui {s_hi}, {md}#{2 + r + k} : index")
            val, desc = q, f"stride{k - 1}({a})/stride{k}({a})"
        if ty == "index":
            return val, desc
        key = ("ld", val, ty)
        if key not in self.consts:
            nm = self.fresh("ld")
            self.lines.append(f"{nm} = arith.index_cast {val} : index to {ty}")
            self.consts[key] = nm
        return self.consts[key], desc


@dataclass
class _LaunchSpec:
    """What a launch needs from a specification (also used for a
    layout-transposed equivalent that was not proved directly)."""
    name: str
    func_args: list
    argnames: dict
    dims: dict
    fixed_ints: dict


def _as_launch_spec(spec: Spec) -> _LaunchSpec:
    return _LaunchSpec(spec.name, spec.func_args, spec.argnames, spec.dims, dict(spec.fixed_ints))


def _materialise_aff(E: _Emit, a: Aff, ty: str, cname: str) -> str:
    """SSA value of a size parameter bound to an affine expression of
    application parameters: index arithmetic, then a cast to ty."""
    acc = E.const(a.const, "index")
    for k, c in a.terms:
        if is_atom(k):
            raise Reject(f"{cname} is a value loaded from memory ({_fmt_param(a)}); "
                         "not materialised for a launch")
        src = E.prog.params.get(k, k)
        md = re.fullmatch(r"dim\((%[\w$.#-]+),(\d+)\)", src)
        if md:
            x = E.fresh("dim")
            E.lines.append(f"{x} = memref.dim {E.mem(md.group(1))}, {E.const(int(md.group(2)), 'index')}"
                           f" : {E.app_args.get(md.group(1))}")
        else:
            aty = E.app_args.get(src)
            if aty is None:
                raise Reject(f"cannot materialise parameter {cname}")
            x = src
            if aty != "index":
                x = E.fresh("idx")
                E.lines.append(f"{x} = arith.index_cast {src} : {aty} to index")
        if c != 1:
            t = E.fresh("mul")
            E.lines.append(f"{t} = arith.muli {x}, {E.const(c, 'index')} : index")
            x = t
        nxt = E.fresh("sum")
        E.lines.append(f"{nxt} = arith.addi {acc}, {x} : index")
        acc = nxt
    if ty != "index":
        nm = E.fresh(re.sub(r"\W", "", cname))
        E.lines.append(f"{nm} = arith.index_cast {acc} : index to {ty}")
        acc = nm
    return acc


def _launch(E: _Emit, spec: _LaunchSpec, b: Binding, ignore_perms: bool = False) -> dict:
    prog = E.prog
    app_args = E.app_args
    operands, types, binding, copy_ins = [], [], {}, []
    for ssa, ty in spec.func_args:
        cname = spec.argnames.get(ssa, ssa)
        if ty.startswith("memref"):
            if ssa not in b.arrays:
                raise Reject(f"spec array {cname} is unbound")
            i_arr, o_arr, p = b.arrays[ssa]
            if not ignore_perms and list(p) != list(range(len(p))):
                raise Reject("binding needs a layout transpose")
            target = o_arr or i_arr
            aty = app_args.get(target, ty)
            opnd = E.mem(target)
            lty = ty
            if aty != ty:
                if _shape_part(aty) == _shape_part(ty):
                    # only the layout differs (strided view / offset): pass the
                    # memref as is -- its descriptor carries offset and strides,
                    # and the leading dimension comes from those strides
                    lty = aty
                    if re.search(r"offset:\s*(\?|[1-9])", aty):
                        E.notes.append(f"{cname}: {target} has a non-zero/dynamic offset carried by its "
                                       "memref descriptor")
                elif _layout_part(aty) == _layout_part(ty):
                    c = E.fresh(re.sub(r"\W", "", cname))
                    E.lines.append(f"{c} = memref.cast {opnd} : {aty} to {ty}")
                    opnd = c
                else:
                    raise Reject(f"{cname}: cannot pass {aty} as {ty}")
            if o_arr and i_arr and i_arr != o_arr:
                copy_ins.append(f"memref.copy {E.mem(i_arr)}, {o_arr} : {app_args.get(i_arr)} to "
                                f"{app_args.get(o_arr)}")
                binding[cname] = f"{o_arr} (in: {i_arr}, copied into {o_arr} before the launch)"
            else:
                binding[cname] = target + (" (in-place)" if o_arr else "")
            operands.append(opnd)
            types.append(lty)
            continue
        if ssa in b.fparams:
            src = b.fparams[ssa]
            if isinstance(src, float):
                operands.append(E.const(src, ty))
                binding[cname] = f"literal {fmt_float(src)}"
            else:
                if app_args.get(src) != ty:
                    raise Reject(f"scalar {cname} has type {app_args.get(src)}, spec expects {ty}")
                operands.append(src)
                binding[cname] = src
            types.append(ty)
            continue
        if cname in spec.fixed_ints:
            operands.append(E.const(spec.fixed_ints[cname], ty))
            binding[cname] = f"literal {spec.fixed_ints[cname]} (specialised)"
            types.append(ty)
            continue
        pv = f"p{ssa}"
        if pv in b.params and isinstance(b.params[pv], Aff):
            operands.append(_materialise_aff(E, b.params[pv], ty, cname))
            binding[cname] = _fmt_param(b.params[pv])
            types.append(ty)
            continue
        if pv in b.params:
            app_pv = b.params[pv]
            src = prog.params.get(app_pv, app_pv)
            md = re.fullmatch(r"dim\((%[\w$.#-]+),(\d+)\)", src)
            aty = app_args.get(src)
            if md:
                d = E.fresh("dim")
                E.lines.append(f"{d} = memref.dim {E.mem(md.group(1))}, {E.const(int(md.group(2)), 'index')}"
                               f" : {app_args.get(md.group(1))}")
                nm = d
                if ty != "index":
                    nm = E.fresh(re.sub(r"\W", "", cname))
                    E.lines.append(f"{nm} = arith.index_cast {d} : index to {ty}")
                operands.append(nm)
            elif aty == ty:
                operands.append(src)
            elif aty is not None:
                nm = E.fresh(re.sub(r"\W", "", cname))
                E.lines.append(f"{nm} = arith.index_cast {src} : {aty} to {ty}")
                operands.append(nm)
            else:
                raise Reject(f"cannot materialise parameter {cname}")
            binding[cname] = src
            types.append(ty)
            continue
        if cname in spec.dims:
            arr_c, k = spec.dims[cname]
            spec_arr = next((s_ for s_, n in spec.argnames.items() if n == arr_c), None)
            if spec_arr is None or spec_arr not in b.arrays:
                raise Reject(f"cannot derive {cname}")
            i_arr, o_arr, p = b.arrays[spec_arr]
            tgt = o_arr or i_arr
            nm, desc = E.leading_dim(tgt, k, ty)
            operands.append(nm)
            types.append(ty)
            binding[cname] = desc
            continue
        raise Reject(f"unbound scalar argument {cname}")
    E.lines += copy_ins
    E.lines.append(f"kernel.launch @{spec.name}({', '.join(operands)}) : ({', '.join(types)}) -> ()")
    return binding


# ---------------------------------------------------------------------------
# Layout-transposed bindings (manifest `layout_transpose`)
# ---------------------------------------------------------------------------
#
# Manifest key (routine level, declarative):
#   "layout_transpose": {
#     "flags": {"<FLAG>": ["<value0>", "<value1>"], ...},
#     "rules": [{"operands": ["A"], "flip": ["TRANSA", "UPLO"], "swap": [["m", "n"]]}, ...]
#   }
# A binding that stores the given (rank-2) operands transposed is equivalent
# to the specification whose `defines` have each listed flag flipped, with
# the listed argument pairs swapped (flips first, then swaps).  The set of
# transposed operands must be covered exactly by disjoint rules.

def transposed_equivalent(spec: Spec, b: Binding):
    """(launch spec, binding with identity layouts, note) or raise Reject."""
    lt = spec.manifest.get("layout_transpose")
    if not lt:
        raise Reject(f"binding needs a layout transpose and manifest of {spec.routine} has no "
                     "layout_transpose metadata")
    T = set()
    for ssa, (_i, _o, p) in b.arrays.items():
        if list(p) != sorted(p):
            if len(p) != 2:
                raise Reject("layout transpose of a rank > 2 operand")
            T.add(spec.argnames.get(ssa, ssa))
    rules = lt.get("rules", [])
    chosen = None
    for r in range(1, len(rules) + 1):
        for combo in itertools.combinations(rules, r):
            sets = [frozenset(x["operands"]) for x in combo]
            if sum(len(x) for x in sets) == len(T) and frozenset().union(*sets) == T:
                chosen = combo
                break
        if chosen:
            break
    if not chosen:
        raise Reject(f"no layout_transpose rule covers transposed operands {sorted(T)}")
    defines = dict(spec.entry.get("defines", {}))
    for rule in chosen:
        for f in rule.get("flip", []):
            v0, v1 = lt["flags"][f]
            if defines.get(f) == v0:
                defines[f] = v1
            elif defines.get(f) == v1:
                defines[f] = v0
            else:
                raise Reject(f"flag {f} has no flippable value in {spec.name}")
    target = next((e for e in spec.manifest.get("specs", []) if e.get("defines") == defines), None)
    if target is None:
        raise Reject(f"no {spec.routine} specification with defines {defines}")
    if target.get("args") != spec.entry.get("args"):
        raise Reject(f"{target['name']} has a different argument list")
    cargs = list(spec.entry.get("args", []))
    src_of = {c: c for c in cargs}
    for rule in chosen:
        for x, y in rule.get("swap", []):
            if x in src_of and y in src_of:
                src_of[x], src_of[y] = src_of[y], src_of[x]
    ssa_of = {c: s_ for s_, c in spec.argnames.items()}
    arrays, params, fparams = {}, {}, {}
    fixed = {}
    for c in cargs:
        new, old = ssa_of.get(c), ssa_of.get(src_of[c])
        if new is None or old is None:
            continue
        if old in b.arrays:
            i, o, p = b.arrays[old]
            arrays[new] = (i, o, tuple(range(len(p))))
        if f"p{old}" in b.params:
            params[f"p{new}"] = b.params[f"p{old}"]
        if old in b.fparams:
            fparams[new] = b.fparams[old]
        if src_of[c] in spec.fixed_ints:
            fixed[c] = spec.fixed_ints[src_of[c]]
    ls = _LaunchSpec(target["name"], spec.func_args, spec.argnames, spec.dims, fixed)
    # Manifest layout rules propose ABI bindings; they are not semantic
    # axioms. Prove the proposed replacement against its imported body.
    target_spec = _load_one(spec.source_dir, spec.manifest, spec.routine,
                            spec.dims, target, None)
    if target_spec is None:
        raise Reject('layout target has no interpretable reference body')
    target_ssa = {c: s for s, c in target_spec.argnames.items()}
    proof_arrays, proof_params, proof_fparams = {}, {}, {}
    for c in cargs:
        new, old = target_ssa.get(c), ssa_of.get(src_of[c])
        if new is None or old is None:
            raise Reject('layout target parameter is unbound')
        if new in target_spec.system.arrays:
            p = b.arrays[old][2]
            inverse = tuple(p.index(k) for k in range(len(p)))
            proof_arrays[new] = (old, old if new in target_spec.system.outputs else None, inverse)
        if 'p' + new in target_spec.system.params:
            proof_params['p' + new] = 'p' + old
        if new in target_spec.system.float_params:
            proof_fparams[new] = old
    proof = compare_systems(spec.system, target_spec.system,
                            Binding(proof_arrays, proof_params, proof_fparams))
    if not proof.ok:
        raise Reject('layout metadata is not proved by the reference bodies: ' + proof.reason)
    rule_txt = "; ".join(f"{'+'.join(r_['operands'])} transposed -> flip {','.join(r_.get('flip', []))}"
                         + (f", swap {r_['swap']}" if r_.get("swap") else "") for r_ in chosen)
    note = (f"{spec.name} proved with operands {sorted(T)} stored transposed; launched as the "
            f"equivalent {target['name']} (manifest layout_transpose: {rule_txt})")
    return ls, Binding(arrays, params, fparams), note


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _lit_checker(spec: Spec, rename=None):
    def ok(sp, v):
        name = rename(sp) if rename else sp
        return literal_ok(spec.entry, spec.argnames.get(name, name), v, "float")[0]
    return ok


def _single_search(app: System, specs: list[Spec], tried: list):
    found = []
    for spec in specs:
        t1 = time.perf_counter()
        cache: dict = {}
        for b in enumerate_bindings(app, spec.system, cache, lit_ok=_lit_checker(spec)):
            r = compare_systems(app, spec.system, b, cache=cache)
            tried.append((spec.name, b, r))
            if r.ok:
                cost = b.cost()
                cost = (cost[0], cost[1], cost[2] + len(spec.fixed_ints))
                found.append({"cost": cost, "key": spec.name, "kind": "single", "spec": spec,
                              "b": b, "r": r, "ms": (time.perf_counter() - t1) * 1000})
    return found


def _composite_search(app: System, specs: list[Spec], tried: list, budget_s: float, stats_w: dict):
    from .compose import P1, P2, build_composites, pair_plausible, signature
    t0 = time.perf_counter()
    asig = signature(app)
    pairs = [(s1, s2) for s1 in specs for s2 in specs
             if s1.sig is not None and s2.sig is not None and pair_plausible(asig, s1.sig, s2.sig)]
    stats_w["composite_pairs"] = len(pairs)
    found = []
    n_comp = 0
    for s1, s2 in pairs:
        if time.perf_counter() - t0 > budget_s:
            stats_w["composite_truncated"] = True
            break
        comps = build_composites(s1.system, s2.system, asig.out_ranks,
                                 _used_params(s1.system), _used_params(s2.system))
        for comp in comps:
            n_comp += 1
            if not asig.profile <= signature(comp.system).profile:
                continue

            def lit_ok(sp, v, comp=comp, s1=s1, s2=s2):
                if sp.startswith(P1):
                    return literal_ok(s1.entry, s1.argnames.get(sp[len(P1):], sp), v, "float")[0]
                return literal_ok(s2.entry, s2.argnames.get(sp[len(P2):], sp), v, "float")[0]
            cache: dict = {}
            # aliasing is checked per launch after the split (the link is
            # sequential by construction)
            for b in enumerate_bindings(app, comp.system, cache, lit_ok=lit_ok, alias_guard=False):
                parts = comp.split(b)
                if parts is None or not all(_copy_ins_safe(pa[0]) for pa in parts):
                    continue
                r = compare_systems(app, comp.system, b, cache=cache)
                tried.append((f"{s1.name}+{s2.name}", b, r))
                if r.ok:
                    cost = b.cost()
                    cost = (cost[0], cost[1], cost[2] + len(s1.fixed_ints) + len(s2.fixed_ints))
                    found.append({"cost": cost, "key": f"{s1.name}+{s2.name}", "kind": "composite",
                                  "specs": (s1, s2), "comp": comp, "parts": parts, "b": b, "r": r,
                                  "ms": (time.perf_counter() - t0) * 1000})
    stats_w["composites_built"] = n_comp
    return found


def _fmt_param(v) -> str:
    """Application value of a spec size parameter: an SSA name, or an
    expression over SSA names and loaded elements (@arr[i] -> arr[i])."""
    if not isinstance(v, Aff):
        return v[1:]
    parts = []
    for k, c in v.terms:
        if is_atom(k):
            arr, coords = atom_info(k)
            name = f"{arr}[{', '.join(_fmt_param(a) for a in coords)}]"
        else:
            name = k[1:] if k.startswith("p") else k
        parts.append(name if c == 1 else f"-{name}" if c == -1 else f"{c}*{name}")
    if v.const or not parts:
        parts.append(str(v.const))
    return " + ".join(parts).replace("+ -", "- ")


def _recognition_only(spec: Spec) -> bool:
    return bool((spec.entry or {}).get("target", {}).get("recognition_only"))


def _recognition_binding(spec: Spec, b: Binding) -> dict[str, str]:
    """C parameter -> application value for a match that is reported, not
    launched.  Scalars the application does not determine (e.g. a column
    count that only sizes x) are reported as unbound."""
    out = {}
    for ssa, _ty in spec.func_args:
        cname = spec.argnames.get(ssa, ssa)
        if ssa in b.arrays:
            i_arr, o_arr, p = b.arrays[ssa]
            val = o_arr or i_arr
            if o_arr and i_arr and i_arr != o_arr:
                val = f"{o_arr} (in: {i_arr})"
            if list(p) != sorted(p):
                val += f" perm{tuple(p)}"
            out[cname] = val
        elif f"p{ssa}" in b.params:
            out[cname] = _fmt_param(b.params[f"p{ssa}"])
        elif ssa in b.fparams:
            out[cname] = str(b.fparams[ssa])
        else:
            out[cname] = "unbound"
    return out


def _emit_one(E: _Emit, spec: Spec, b: Binding):
    """Launch of one proved specification; layout-transposed bindings go
    through the manifest's layout_transpose equivalence."""
    if _recognition_only(spec):
        return spec.name, _recognition_binding(spec, b), "recognition only (no launch)"
    if all(list(p) == sorted(p) for (_i, _o, p) in b.arrays.values()):
        return spec.name, _launch(E, _as_launch_spec(spec), b), None
    ls, b2, note = transposed_equivalent(spec, b)
    if ls.name not in E.allowed_specs:
        raise Reject(f'layout replacement target {ls.name} is unavailable')
    return ls.name, _launch(E, ls, b2, ignore_perms=True), note


def _plan_written(app_outputs, bindings) -> set[str]:
    out = set(app_outputs)
    for b in bindings:
        for (_i, o, _p) in b.arrays.values():
            if o:
                out.add(o)
    return out


# ---------------------------------------------------------------------------
# Partial matches: one library call for the reduction core of an output, plus
# a residual element-wise loop for what the code does around it.
# ---------------------------------------------------------------------------
#
# covariance:   cov[i][j] = (sum_k d[k][i]*d[k][j]) / (n - 1)    (all i, j)
#               -> dgemm (C = d^T d) ; loop cov[e] = cov[e] / (n - 1)
# correlation:  corr[i][j] = sum_k d[k][i]*d[k][j] (i != j), 1.0 (i == j)
#               -> dgemm ; loop corr[e] = 1.0 on the diagonal
#
# Sound by construction: the core (the unique reduction subterm) is proved
# equal across all pieces that have one (e-graph), the call is matched against
# the core over the bounding box B of those pieces with the ordinary spec
# search, B lies inside what the window writes, every input element the core
# reads over B is read by the window anyway, and the residual reads only the
# element itself (its value after the call) or inputs the window never writes.

def _has_sum(f: F) -> bool:
    return any(isinstance(x, FSum) for x in fwalk(f))


def _split_core(f: F, hole: F) -> tuple[F, F]:
    """(core, wrapper): the unique reduction subterm and f with it replaced by
    `hole`.  A node with reductions on both sides is itself the core."""
    if isinstance(f, FSum):
        return f, hole
    if isinstance(f, FBin):
        ha, hb = _has_sum(f.a), _has_sum(f.b)
        if ha and not hb:
            c, w = _split_core(f.a, hole)
            return c, FBin(f.op, w, f.b)
        if hb and not ha:
            c, w = _split_core(f.b, hole)
            return c, FBin(f.op, f.a, w)
        return f, hole
    if isinstance(f, FUn):
        c, w = _split_core(f.a, hole)
        return c, FUn(f.op, w)
    return f, hole


def _expand_copies(arr: str, eqs: list, params=()) -> Optional[list]:
    """Pieces as (domain, formula) with copies of other elements of the output
    (cov[j][i] = cov[i][j]) replaced by the copied piece's formula."""
    def self_reads(f):
        return [x for x in fwalk(f) if isinstance(x, FRead) and x.arr == arr and x.ver == "out"]
    out = []
    for e in eqs:
        if not self_reads(e.formula):
            out.append((e.domain, e.formula))
            continue
        f = e.formula
        if not isinstance(f, FRead):
            return None
        m = {f"v{k}": a for k, a in enumerate(f.idx)}
        tgt = [g for g in eqs if g is not e and not self_reads(g.formula)
               and implies_all(e.domain, [c.subst(m) for c in g.domain])]
        if len(tgt) != 1:
            return None
        out.append((e.domain, canonicalize(_freshen(fsubst(tgt[0].formula, m)), e.domain, set(params))))
    return out


def _formulas_equal(fa: F, fb: F, P: list[Cons]) -> bool:
    name = lambda arr, ver: f"{ver}:{arr}"    # noqa: E731
    unions = [(RmIx(v), enc_ix(val, "")) for v, val in _piece_equalities(P).items()]
    return check_equal(encode(fa, "", name), encode(fb, "", name), unions, []).ok


def _bounding_box(pieces: list[list[Cons]], rank: int) -> Optional[list[Cons]]:
    cands = []
    for P in pieces:
        for c in P:
            vs = {v for v in c.vars() if v.startswith("v")}
            if c.kind == "ge" and len(vs) == 1 and not atoms_in(c.vars()) and c not in cands:
                cands.append(c)
    B = [c for c in cands if all(implies_all(P, [c]) for P in pieces)]
    for k in range(rank):
        lows, ups, _ = bounds_of(f"v{k}", B)
        if not lows or not ups:
            return None
    return simplify(B)


def _residual_ok(f: F, arr: str, ev: list[Aff], outputs) -> bool:
    for x in fwalk(f):
        if isinstance(x, (FSum, FSel)) or type(x).__name__ in ("FAcc", "FAccAt"):
            return False
        if isinstance(x, FRead):
            if x.arr == arr:
                if x.ver != "out" or list(x.idx) != list(ev):
                    return False
            elif x.ver != "in" or x.arr in outputs:
                return False
        if isinstance(x, FBin) and x.op not in ("add", "sub", "mul", "div"):
            return False
        if isinstance(x, FUn) and x.op not in ("negf", "sqrt", "exp", "log", "absf", "tanh", "rsqrt"):
            return False
    return True


def _partial_candidates(app: System):
    """(core system, residual plan) for windows whose single output is a
    reduction core wrapped by element-wise code."""
    if len(app.outputs) != 1:
        return []
    arr = next(iter(app.outputs))
    eqs = [e for e in app.equations if e.arr == arr]
    rank = eqs[0].rank
    ev = [Aff.var(f"v{k}") for k in range(rank)]
    hole = FRead(arr, "out", ev)
    pieces = _expand_copies(arr, eqs, app.params)
    if pieces is None:
        return []
    cores, residual = [], []
    for (P, f) in pieces:
        f = _fold_zero(f)
        if _has_sum(f):
            c, w = _split_core(f, hole)
            if any(isinstance(x, FRead) and x.arr == arr for x in fwalk(c)):
                return []
            cores.append((P, c))
            if w.key() != hole.key():
                residual.append((P, w))
        else:
            residual.append((P, f))
    if not cores or not residual:
        return []
    if any(not _residual_ok(w, arr, ev, app.outputs) for (_P, w) in residual):
        return []
    F0 = cores[0][1]
    for (P, c) in cores[1:]:
        if c.key() != F0.key() and not _formulas_equal(c, F0, P):
            return []
    B = _bounding_box([P for (P, _c) in cores], rank)
    if B is None or not _covers([e.domain for e in eqs], [B]):
        return []
    params = set(app.params)
    # The call runs only where some core piece is non-empty: G = parameter
    # facts implied by every core piece (e.g. M >= 2 when the cores are the
    # off-diagonal elements); with G false the residual alone is the window.
    gs = [project(P, params) or [] for (P, _c) in cores]
    G = [c for c in gs[0] if all(implies_all(g, [c]) for g in gs[1:])]
    for x in fwalk(F0):
        if isinstance(x, FRead):
            for a in x.idx:
                vs = {v for v in a.vars() if v.startswith("v")}
                if len(vs) > 1:
                    return []
                for v in vs:
                    keep = {v} | params
                    pb = project(B + G, keep)
                    pcs = [q for q in (project(P, keep) for (P, _c) in cores) if q is not None]
                    if pb is None or not _covers(pcs, [pb]):
                        return []
    reads = {(x.arr, x.ver) for x in fwalk(F0) if isinstance(x, FRead)}
    core_sys = System([Equation(arr, rank, B, F0)], {arr: app.outputs[arr]}, app.arrays, app.params,
                      app.float_params, reads, dict(app.evidence), list(app.warnings), dict(app.atoms))
    guard = [] if implies_all(project(B, params) or [], G) else G
    return [(core_sys, {"arr": arr, "rank": rank, "box": B, "residual": residual, "guard": guard})]


def _aff_index(E: "_Emit", a: Aff, env: dict) -> str:
    """Index SSA value of an affine expression of loop coordinates (env) and
    application parameters."""
    acc = E.const(a.const, "index")
    for k, c in a.terms:
        if k in env:
            x = env[k]
        else:
            x = _materialise_aff(E, Aff.var(k), "index", k)
        if c != 1:
            t = E.fresh("mul")
            E.lines.append(f"{t} = arith.muli {x}, {E.const(c, 'index')} : index")
            x = t
        nxt = E.fresh("ix")
        E.lines.append(f"{nxt} = arith.addi {acc}, {x} : index")
        acc = nxt
    return acc


def _residual_value(E: "_Emit", f: F, env: dict, elem: str) -> str:
    if isinstance(f, FConst):
        return E.const(f.v, elem)
    if isinstance(f, FParam):
        return f.name
    if isinstance(f, FRead):
        ty = E.app_args.get(f.arr)
        idx = [_aff_index(E, a, env) for a in f.idx]
        v = E.fresh("rv")
        E.lines.append(f"{v} = memref.load {E.mem(f.arr)}[{', '.join(idx)}] : {ty}")
        return v
    if isinstance(f, FBin):
        a, b = _residual_value(E, f.a, env, elem), _residual_value(E, f.b, env, elem)
        v = E.fresh("rv")
        E.lines.append(f"{v} = arith.{f.op}f {a}, {b} : {elem}")
        return v
    if isinstance(f, FUn):
        a = _residual_value(E, f.a, env, elem)
        v = E.fresh("rv")
        op = "arith.negf" if f.op == "negf" else f"math.{f.op}"
        E.lines.append(f"{v} = {op} {a} : {elem}")
        return v
    raise Reject(f"residual: unsupported {type(f).__name__}")


def _emit_cond(E: "_Emit", cons: list[Cons], env: dict) -> str:
    conds = []
    for c in cons:
        lhs = _aff_index(E, c.aff, env)
        cv = E.fresh("g")
        pred = "eq" if c.kind == "eq" else "sge"
        E.lines.append(f"{cv} = arith.cmpi {pred}, {lhs}, {E.const(0, 'index')} : index")
        conds.append(cv)
    cond = conds[0]
    for c2 in conds[1:]:
        nc = E.fresh("g")
        E.lines.append(f"{nc} = arith.andi {cond}, {c2} : i1")
        cond = nc
    return cond


def _guard_launch(E: "_Emit", guard: list[Cons]):
    """Wrap the (last) kernel.launch line in scf.if over the guard."""
    k = max(i for i, l in enumerate(E.lines) if l.lstrip().startswith("kernel.launch"))
    launch = E.lines[k]
    del E.lines[k]
    cond = _emit_cond(E, guard, {})
    E.lines += [f"scf.if {cond} {{", launch, "}"]


def _emit_residual(E: "_Emit", plan: dict, params: set):
    """One loop nest per residual piece: arr[e] = w(e) over the piece's domain
    (bounds from Fourier-Motzkin projection; parameter-only facts guard it)."""
    arr, rank = plan["arr"], plan["rank"]
    ty = E.app_args.get(arr)
    if ty is None or not ty.startswith("memref"):
        raise Reject(f"residual: no memref for {arr}")
    elem = _elem_type(ty)
    one = E.const(1, "index")
    for (P, w) in plan["residual"]:
        coords = [f"v{k}" for k in range(rank)]
        # Coordinates fixed by an equality (the diagonal v1 == v0, a last row
        # v0 == n - 1) are defined, not looped over: solve each coordinate
        # equality for its innermost coordinate and substitute it away.
        cons = list(P)
        defined: dict = {}
        while True:
            eqc = next((c for c in cons if c.kind == "eq" and any(v in coords for v in c.vars())), None)
            if eqc is None:
                break
            cv = max((v for v in eqc.vars() if v in coords and abs(eqc.aff.coeff(v)) == 1),
                     key=coords.index, default=None)
            if cv is None:
                raise Reject("residual: non-unit coordinate equality")
            k = eqc.aff.coeff(cv)
            rest = eqc.aff - Aff.var(cv, k)
            sol = rest.scale(-1) if k == 1 else rest
            cons = [c.subst({cv: sol}) for c in cons if c is not eqc]
            defined = {x: a.subst({cv: sol}) for x, a in defined.items()}
            defined[cv] = sol
        loops = [c for c in coords if c not in defined]
        pvars = sorted({v for c in cons for v in c.vars() if v not in coords} & params)
        guard = project(cons, set(pvars)) if loops else [c for c in cons]
        if guard is None:
            continue                       # empty piece
        guard = [c for c in guard if not c.aff.is_const() or not ((c.kind == "ge" and c.aff.const >= 0)
                                                                   or (c.kind == "eq" and c.aff.const == 0))]
        if any(c.aff.is_const() for c in guard):
            continue                       # contradictory constant fact: empty piece
        cached = set(E.consts)            # constants made inside a region stay local to it
        opened = 0
        if guard:
            cond = _emit_cond(E, guard, {})
            E.lines.append(f"scf.if {cond} {{")
            opened += 1
        env: dict = {}
        for k, cv in enumerate(loops):
            keep = set(loops[:k + 1]) | set(pvars)
            Pk = project(cons, keep)
            if Pk is None:
                raise Reject("residual: cannot project piece domain")
            lows, ups, _ = bounds_of(cv, Pk)
            if not lows or not ups:
                raise Reject("residual: unbounded piece")
            lb = None
            for l in lows:
                x = _aff_index(E, l, env)
                if lb is None:
                    lb = x
                else:
                    t = E.fresh("lb")
                    E.lines.append(f"{t} = arith.maxsi {lb}, {x} : index")
                    lb = t
            ub = None
            for u in ups:
                x = _aff_index(E, u + Aff.cst(1), env)
                if ub is None:
                    ub = x
                else:
                    t = E.fresh("ub")
                    E.lines.append(f"{t} = arith.minsi {ub}, {x} : index")
                    ub = t
            iv = E.fresh(f"{cv}_")
            E.lines.append(f"scf.for {iv} = {lb} to {ub} step {one} {{")
            opened += 1
            env[cv] = iv
        for cv in coords:
            if cv in defined:
                env[cv] = _aff_index(E, defined[cv], env)
        val = _residual_value(E, w, env, elem)
        E.lines.append(f"memref.store {val}, {E.mem(arr)}[{', '.join(env[c] for c in coords)}] : {ty}")
        E.lines += ["}"] * opened
        if opened:
            for key in [k_ for k_ in E.consts if k_ not in cached]:
                del E.consts[key]


def match_function(text: str, prog: Program, specs: list[Spec], log=None, stats=None):
    results = []
    claimed: list[tuple[int, int]] = []
    rejections = []
    t_func = time.perf_counter()
    func_budget = float(os.environ.get("REGION_MATCH_FUNC_BUDGET", "300"))
    comp_budget = float(os.environ.get("REGION_MATCH_COMPOSITE_BUDGET", "20"))
    composites_on = os.environ.get("REGION_MATCH_NO_COMPOSITE", "") in ("", "0")
    for (lo, hi, end) in _windows(prog):
        if time.perf_counter() - t_func > func_budget:
            rejections.append(f"function time budget ({func_budget:.0f}s) exhausted; "
                              "remaining windows skipped")
            break
        ops = prog.top_ops
        span_end = ops[end].end
        while span_end > ops[lo].start and text[span_end - 1] in "\r\n":
            span_end -= 1                     # keep the final newline outside the span
        span = (ops[lo].start, span_end)
        if any(not (span[1] <= a or b <= span[0]) for a, b in claimed):
            continue
        live = _live_outs(text, prog, lo, end)
        window = set(range(lo, end + 1))
        t0 = time.perf_counter()
        try:
            app = build_system(prog, window)
        except (Reject, Unsupported, NotImplementedError) as ex:
            rejections.append(f"window {lo}-{end}: {ex}")
            continue
        norm_ms = (time.perf_counter() - t0) * 1000
        if not app.equations:
            continue
        remat = _rematerialise(prog, lo, end, live, set(app.outputs)) if live else []
        if remat is None:
            continue
        plan = _sync_plan(prog, app, lo)
        if plan is None:
            rejections.append(f"window {lo}-{end}: an operand is neither in memory nor "
                              "available as a full tensor at the window start")
            continue
        if not _written_back(prog, app, lo, end):
            rejections.append(f"window {lo}-{end}: tensor-level output is never written back")
            continue
        tried: list = []
        wstats: dict = {"window": (lo, end)}
        found = _single_search(app, specs, tried)
        if not found and composites_on:
            found = _composite_search(app, specs, tried, comp_budget, wstats)
        if not found and os.environ.get("REGION_MATCH_NO_PARTIAL", "") in ("", "0"):
            for core_sys, rplan in _partial_candidates(app):
                for c in _single_search(core_sys, specs, tried):
                    c.update(kind="partial", residual=rplan)
                    found.append(c)
        if not found:
            rejections.append(f"window {lo}-{end}: no specification matched "
                              f"({len(tried)} bindings tried)")
            if stats is not None:
                wstats.update({"matched": None, "bindings_tried": len(tried), "normalise_ms": norm_ms})
                stats.setdefault("windows", []).append(wstats)
            continue
        found.sort(key=lambda x: (x["cost"], x["kind"] != "single", x["key"]))
        indent = re.match(r"[ \t]*", text[span[0]:]).group(0)
        uid = f"rm{lo}_{end}"
        chosen = None
        last_err = None
        for cand in found:
            E = _Emit(prog, uid)
            E.allowed_specs = {candidate.name for candidate in specs}
            try:
                if cand["kind"] in ("single", "partial"):
                    E.prepare(plan, _plan_written(app.outputs, [cand["b"]]))
                    name, binding, note = _emit_one(E, cand["spec"], cand["b"])
                    launched, notes = [name], [note] if note else []
                    if cand["kind"] == "partial":
                        rp = cand["residual"]
                        if _recognition_only(cand["spec"]):
                            raise Reject("partial match of a recognition-only spec")
                        if rp.get("guard"):
                            _guard_launch(E, rp["guard"])
                        _emit_residual(E, rp, set(prog.params))
                        notes.append(f"partial: {name} computes the reduction core of {rp['arr']} on "
                                     f"{' && '.join(map(repr, rp['box']))}; a residual loop then writes "
                                     + "; ".join(f"{w!r} for {' && '.join(map(repr, P))}"
                                                 for (P, w) in rp['residual']))
                else:
                    s1, s2 = cand["specs"]
                    (a1, p1, f1), (a2, p2, f2) = cand["parts"]
                    b1, b2 = Binding(a1, p1, f1), Binding(a2, p2, f2)
                    E.prepare(plan, _plan_written(app.outputs, [b1, b2]))
                    n1, bd1, note1 = _emit_one(E, s1, b1)
                    n2, bd2, note2 = _emit_one(E, s2, b2)
                    launched = [n1, n2]
                    binding = {f"1:{n1}:{k}": v for k, v in bd1.items()}
                    binding.update({f"2:{n2}:{k}": v for k, v in bd2.items()})
                    notes = [n_ for n_ in (note1, note2) if n_]
            except Reject as ex:
                last_err = f"{cand['key']} matched but {ex}"
                continue
            chosen = cand
            break
        if chosen is None:
            rejections.append(f"window {lo}-{end}: {last_err}")
            continue
        E.lines += remat
        launch = "\n".join(indent + l for l in E.lines)
        r = chosen["r"]
        b = chosen["b"]
        others = sorted({f"{f['key']}[" + ("transposed" if f["cost"][0] else
                                          "copy-in" if f["cost"][1] else
                                          "literal" if f["cost"][2] else "same") + "]"
                         for f in found if f is not chosen})
        warn = list(app.warnings)
        flags = []
        for role, (i_arr, o_arr, p) in b.arrays.items():
            if chosen["kind"] in ("single", "partial") and o_arr and i_arr == o_arr \
                    and (role, "in") in chosen["spec"].system.reads:
                flags.append(f"in-place: {chosen['spec'].argnames.get(role, role)} in/out both bound to {o_arr}")
            if o_arr and i_arr and i_arr != o_arr:
                ia, oa = prog.arg_buffers.get(i_arr), prog.arg_buffers.get(o_arr)
                noalias = bool(ia and oa and ia.restrict and oa.restrict)
                flags.append(f"copy-in {i_arr} -> {o_arr}; assumes {i_arr} and {o_arr} do not alias"
                             + (" (restrict)" if noalias else " (pipeline-wide noalias assumption)"))
        lits = {k: v for k, v in b.fparams.items() if isinstance(v, float)}
        if lits:
            flags.append(f"parameter(s) bound to literals {lits} (e-graph union with Term.Lit)")
        nodes = max((e.nodes for e in r.egg), default=0)
        egg_ms = sum(e.ms for e in r.egg)
        name = "+".join(launched)
        if chosen["kind"] in ("single", "partial"):
            sp = chosen["spec"]
            src = f"spec semantics from {sp.source_stage} stage ({sp.validation})"
        else:
            s1, s2 = chosen["specs"]
            comp = chosen["comp"]
            src = (f"sequential composition {s1.name} ; {s2.name} with link "
                   f"{ {s2.argnames.get(k, k): s1.argnames.get(v, v) for k, v in comp.link.items()} } "
                   f"(S1 final values substituted into S2); spec semantics from "
                   f"{s1.source_stage}/{s2.source_stage} stages")
        ev = (f"{name}: {len(app.equations)} element equation(s), {r.pieces} domain piece(s) "
              f"equal under binding {binding}; e-graph max {nodes} nodes, egglog {egg_ms:.1f} ms; "
              f"dependence checks {app.evidence.get('dep_checks')} (reads: input "
              f"{app.evidence.get('reads_input')}, final {app.evidence.get('reads_final')}, "
              f"accumulator {app.evidence.get('reads_acc')}, scratch-inlined "
              f"{app.evidence.get('reads_scratch_inlined')}); {src}")
        if notes:
            ev += "; " + "; ".join(notes)
        if E.notes:
            ev += "; " + "; ".join(E.notes)
        if remat:
            ev += f"; {len(remat)} live-out value(s) re-defined after the launch"
        if others:
            ev += f"; also equal: {', '.join(others)}"
        if flags:
            ev += "; " + "; ".join(flags)
        if warn:
            ev += "; WARNING: " + "; ".join(warn)
        idx_arrays = sorted({atom_info(n_)[0] for n_ in getattr(app, "atoms", {})})
        if idx_arrays:
            flags.append(f"index arrays {', '.join(idx_arrays)} are read-only in the region and "
                         "assumed not to alias written arrays (pipeline-wide noalias assumption)")
            ev += "; " + flags[-1]
        recog = (chosen["kind"] in ("single", "partial") and _recognition_only(chosen["spec"])) or \
            any(_recognition_only(s_) for s_ in chosen.get("specs", ()) or ())
        if recog:
            launch = ""
            ev = "recognition only, not launched; " + ev
        results.append(RegionMatch(span, name, launch, binding, ev, list(launched), recog))
        claimed.append(span)
        if stats is not None:
            wstats.update({
                "matched": name, "kind": chosen["kind"], "also": others,
                "bindings_tried": len(tried), "pieces": r.pieces,
                "egraph_nodes": [e.nodes for e in r.egg], "egraph_classes": [e.classes for e in r.egg],
                "egg_ms": [round(e.ms, 2) for e in r.egg], "normalise_ms": round(norm_ms, 2),
                "match_ms": round(chosen["ms"], 2)})
            stats.setdefault("windows", []).append(wstats)
    if log:
        for rj in rejections:
            log(rj)
    if stats is not None:
        stats.setdefault("rejections", []).extend(rejections)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _launch_output_resolver(specs: list[Spec], spec_dir: str):
    """spec name -> argument positions a kernel.launch of it writes: from the
    loaded spec systems, else from the generated library metadata."""
    table: dict[str, list[int]] = {}
    for sp in specs:
        table[sp.name] = [k for k, (ssa, _t) in enumerate(sp.func_args) if ssa in sp.system.outputs]
    lib = os.path.join(os.path.dirname(os.path.abspath(spec_dir)), "kernel_library_v2.mlir")
    try:
        txt = open(lib).read()
        for m in re.finditer(r"//\s*lib_v2-meta:\s*(\{.*\})\s*$", txt, re.M):
            d = json.loads(m.group(1))
            if d.get("spec") in table:
                continue
            params = d.get("params", [])
            table[d["spec"]] = [params.index(o) for o in d.get("outputs", []) if o in params]
    except (OSError, ValueError):
        pass
    return lambda name: table.get(name)


def match_text(mlir_text: str, spec_dir: str, log=None, stats=None,
               allowed_specs=None) -> list[RegionMatch]:
    specs = load_specs(spec_dir, log)
    if allowed_specs is not None:
        specs = [s for s in specs if s.name in allowed_specs]
    if not specs:
        return []
    funcs, aliases = parse_module(mlir_text)
    resolver = _launch_output_resolver(specs, spec_dir)
    out = []
    for f in funcs:
        if not f.body:
            continue
        try:
            prog = execute(f, aliases, None, resolver)
        except (Unsupported, Exception) as ex:  # noqa: BLE001
            if log:
                log(f"{f.name}: not analysable ({type(ex).__name__}: {ex})")
            if stats is not None:
                stats.setdefault("rejections", []).append(f"{f.name}: {ex}")
            continue
        out.extend(match_function(mlir_text, prog, specs, log, stats))
    return out
