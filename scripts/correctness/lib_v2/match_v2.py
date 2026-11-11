#!/usr/bin/env python3
"""match_v2.py -- match raised/debufferized MLIR against lib_v2 specs.

Usage:
  match_v2.py INPUT.mlir [-o OUT.mlir] [--library generic_solver/kernel_library_v2.mlir]
              [--report OUT.json] [--inject-defns] [--no-region-match]

Every recognition pattern is derived from the kernel.defn bodies in
kernel_library_v2.mlir (netlib reference code raised by the same pipeline as
the application).  There are no Term formulas and no routine-specific code in
this file.  The existing matcher is reused by import:

  * kernel_match._mlir_semantic_entries
    turn the defn bodies into CompositionEntry steps (body Term templates with
    Cap wildcards + arity / iterator-count gates);
  * kernel_match.parse_generics / encode_body build the application body Terms;
  * kernel_match.match_composition (structural unification + egglog proof)
    decides scalar-body equivalence and returns the Cap bindings.

What this driver adds (the "generalisations"):

  1. dataflow-graph matching of a whole spec body (all its generics, views,
     index arithmetic and write-backs) against a sub-DAG of the application,
     starting from the spec's yielded value;
  2. per-generic search over input/output operand permutations (pi, rho) and
     iteration-dimension bijections (sigma) -- the body check runs on the
     pi-permuted application body, so commutative and non-commutative bodies
     both bind correctly;
  3. access-function comparison after composing every operand's view chain
     (polygeist.submap / tensor.extract_slice) with the generic's indexing map,
     so it does not matter how an access is split between views and maps.
     Row-major application code matches column-major specs because sigma
     relabels the loops and pi exchanges operands: for row-major C = A*B the
     only consistent assignment binds spec A to the application's B and vice
     versa.  Nothing about any routine is encoded here;
  4. ABI completion: spec scalar parameters are bound from slice extents and
     body captures; an i32 spec parameter may bind to an index value via an
     emitted index_cast, a stride symbol may bind to the constant 1, a scalar
     capture may bind to a literal; VLA leading dimensions not referenced by
     the defn body are bound to the bound array's trailing extent.  Any other
     unbound parameter rejects the match.
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterator, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import mlir_lite as ML  # noqa: E402

REPO = HERE.parents[2]
SUITE = os.environ.get("POLYGEIST_SPEC_SUITE", "lib_v3")   # lib_v3 (OpenBLAS-derived) or lib_v2 (netlib)
DEFAULT_LIB = REPO / f"generic_solver/kernel_library_{SUITE.removeprefix('lib_')}.mlir"
SPEC_DIR = REPO / "generic_solver" / SUITE


def _load_literal_policy():
    """The literal-binding policy shared with the region matcher (loaded by
    path: importing the region_match package would import kernel_match before
    POLYGEIST_KERNEL_LIBRARY is set)."""
    import importlib.util
    path = HERE.parent / "region_match" / "literal_policy.py"
    spec = importlib.util.spec_from_file_location("lib_v2_literal_policy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SP = _load_literal_policy()

KM = None  # kernel_match, imported lazily after POLYGEIST_KERNEL_LIBRARY is set


def import_kernel_match(lib_path: Path):
    global KM
    os.environ["POLYGEIST_KERNEL_LIBRARY"] = str(lib_path)
    import kernel_match  # noqa: E402
    KM = kernel_match
    KM.configure_matcher("egglog")
    import eqsat_ablation  # EQSAT_ABLATION env switch; default "full" leaves KM unchanged
    eqsat_ablation.install_linalg_level(KM)
    return KM


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Spec:
    name: str
    order: int
    func: ML.Func
    meta: dict
    entry: object               # kernel_match.CompositionEntry
    generics: list              # ML.Op, textual order
    gidx: dict                  # op idx -> step index
    gbs: dict                   # op idx -> GenericBody
    yields: list
    yield_op: ML.Op
    params: dict                # name -> type


class Library:
    def __init__(self, path: Path):
        self.path = path
        text = path.read_text()
        self.text = text
        self.mod = ML.Module(text)
        metas = {}
        for m in re.finditer(r"//\s*lib_v2-meta:\s*(\{.*\})\s*$", text, re.MULTILINE):
            d = json.loads(m.group(1))
            metas[d["spec"]] = d
        names = [f.name for f in self.mod.funcs]
        # Scalar templates are derived only from imported definition bodies.
        entries = KM._mlir_semantic_entries(path)
        regions = dict(KM._kernel_defn_regions(text))
        # manifest entries (assumptions drive the shared literal policy)
        self.entries: dict[str, dict] = {}
        for mp in sorted((SPEC_DIR / "manifest").glob("*.json")):
            for e in json.loads(mp.read_text()).get("specs", []):
                self.entries[e["name"]] = e
        self.specs: list[Spec] = []
        for order, f in enumerate(self.mod.funcs):
            gens = sorted((op for op in f.ops if op.name == "linalg.generic"),
                          key=lambda o: o.start)
            gb_list = KM.parse_generics(regions[f.name], infer_outputs_from_yield=True)
            if len(gb_list) != len(gens):
                raise RuntimeError(f"{f.name}: generic count mismatch")
            yop = next(op for op in reversed(f.ops) if op.name == "kernel.yield")
            self.specs.append(Spec(
                name=f.name, order=order, func=f, meta=metas.get(f.name, {}),
                entry=entries.get(f.name, KM.CompositionEntry(f.name, [])), generics=gens,
                gidx={g.idx: i for i, g in enumerate(gens)},
                gbs={g.idx: gb for g, gb in zip(gens, gb_list)},
                yields=list(yop.operands), yield_op=yop, params=dict(f.args)))

    def names(self) -> set[str]:
        return {s.name for s in self.specs}


# ---------------------------------------------------------------------------
# Environment (persistent-ish: copied on extension)
# ---------------------------------------------------------------------------

class Env:
    __slots__ = ("vals", "opmap", "gens", "consumed")

    def __init__(self):
        self.vals: dict = {}       # spec value name -> binding tuple
        self.opmap: dict = {}      # spec op idx -> app op idx
        self.gens: dict = {}       # spec generic idx -> (pi, rho, sigma, app idx)
        self.consumed: frozenset = frozenset()

    def copy(self) -> "Env":
        e = Env()
        e.vals = dict(self.vals)
        e.opmap = dict(self.opmap)
        e.gens = dict(self.gens)
        e.consumed = self.consumed
        return e


def same_binding(a: tuple, b: tuple) -> bool:
    if a[0] == "lit" and b[0] == "lit":
        return a[1] == b[1]
    if a[0] != b[0]:
        return False
    if a[0] in ("ssa", "cast_i32", "ld"):
        return a[:2] == b[:2]
    return a == b


@dataclasses.dataclass
class Trace:
    base: object          # spec value name or app key
    access: list          # list of polynomials over iteration dims
    extents: dict         # iteration dim -> spec name / app key
    views: list           # op idxs of the view chain (outermost first)


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

class Matcher:
    def __init__(self, lib: Library, app: ML.Module):
        self.lib = lib
        self.L = lib.mod
        self.A = app
        self.app_gbs: dict[int, object] = {}
        gens = [op for op in app.ops if op.name == "linalg.generic"]
        gens.sort(key=lambda o: o.start)
        gbs = KM.parse_generics(app.text)
        if len(gbs) != len(gens):
            raise RuntimeError(f"generic count mismatch {len(gbs)} vs {len(gens)}")
        for op, gb in zip(gens, gbs):
            self.app_gbs[op.idx] = gb
        self._body_cache: dict = {}
        self._trace_cache: dict = {}
        self.spec: Optional[Spec] = None
        # furthest stage reached per (spec, app generic) -- diagnostics only
        self.stage: dict = {}
        self.policy_rejections: list = []

    def literal_allowed(self, sname: str, value, kind: str) -> bool:
        """Shared policy with the region matcher (region_match/literal_policy)."""
        entry = self.lib.entries.get(self.spec.name, {})
        ok, why = SP.literal_ok(entry, sname.lstrip("%"), value, kind)
        if not ok:
            self.policy_rejections.append(f"{self.spec.name}: {why}")
        return ok

    STAGES = ["shape", "body", "access", "extents", "body-locals", "dataflow", "matched"]

    def note(self, aop, stage: str) -> None:
        k = (self.spec.name, aop.idx)
        cur = self.stage.get(k)
        if cur is None or self.STAGES.index(stage) > self.STAGES.index(cur):
            self.stage[k] = stage

    # -- keys -------------------------------------------------------------
    def akey(self, name: str, ctx: ML.Op):
        """Resolved application value key, or None if not visible."""
        if name is None:
            return None
        d = self.A.def_of(name, ctx)
        if d is not None:
            return ("op", d[0].idx, d[1])
        if name in self.A.arg_types.get(ctx.func, {}):
            return ("arg", ctx.func, name)
        # region block argument (loop induction variable / iter_arg) of an
        # enclosing op
        p = ctx.parent
        while p is not None:
            owner = self.A.ops[p]
            if owner.name != "linalg.generic" and re.search(
                    rf"{re.escape(name)}\s*=", owner.header):
                return ("blk", owner.idx, name)
            p = owner.parent
        return None

    def aname(self, key) -> str:
        if key[0] == "arg":
            return key[2]
        if key[0] == "blk":
            return key[2]
        return self.A.ops[key[1]].results[key[2]]

    def atype(self, key) -> Optional[str]:
        if key[0] == "arg":
            return self.A.arg_types[key[1]][key[2]]
        if key[0] == "blk":
            hdr = self.A.ops[key[1]].header
            m = re.search(r"iter_args\((.*?)\)\s*->\s*\(?([^{]*?)\)?\s*\{", hdr)
            if m:
                names = [x.split("=")[0].strip() for x in ML.split_top(m.group(1))]
                types = ML.split_top(m.group(2))
                if key[2] in names and len(types) == len(names):
                    return types[names.index(key[2])]
            return "index"
        op = self.A.ops[key[1]]
        ty = op.result_types[key[2]] if key[2] < len(op.result_types) else None
        if ty and ty.startswith("tensor<"):
            # a tensor type never carries a layout (to_tensor of a strided
            # memref is parsed with the memref's layout attached)
            parts = ML.split_top(ty[len("tensor<"):-1])
            if len(parts) > 1:
                ty = f"tensor<{parts[0]}>"
        return ty

    def sdef(self, name: str):
        return self.L.def_of(name, self.spec.yield_op)

    # -- views ------------------------------------------------------------
    def trace(self, mod: ML.Module, gop: ML.Op, i: int, is_app: bool) -> Trace:
        ck = (is_app, gop.idx, i)
        if ck in self._trace_cache:
            return self._trace_cache[ck]
        operands = gop.ins + gop.outs
        v = operands[i]
        ctx = gop
        nd = len(gop.iters)
        access = ML.map_apply(gop.maps[i], ML.identity_dims(nd), [])
        extents: dict = {}
        views: list = []
        key = (lambda n, c: self.akey(n, c)) if is_app else (lambda n, c: n)
        first = True
        while True:
            d = mod.def_of(v, ctx)
            if d is None:
                break
            vop, _ = d
            if vop.name == "polygeist.submap":
                mm = re.search(r"map\s*=\s*(affine_map<.*?>|#[\w$.\-]+)\s*\}", vop.text)
                smap = ML.inline_aliases(mm.group(1), mod.aliases)
                _, ns, _ = ML.parse_affine_map(smap)
                syms = vop.operands[1:1 + ns]
                sizes = vop.operands[1 + ns:]
                sym_polys = [{(("v", key(s, vop)),): 1} for s in syms]
                if first:
                    for j, sz in enumerate(sizes):
                        dj = ML.pure_dim(access[j]) if j < len(access) else None
                        if dj is not None:
                            extents[dj] = key(sz, vop)
                access = ML.map_apply(smap, access, sym_polys)
                views.append(vop.idx)
                v, ctx, first = vop.operands[0], vop, False
            elif vop.name == "tensor.extract_slice":
                hdr = vop.text
                mm = re.match(r".*?extract_slice\s+(%[\w$.\-#]+)\s*\[(.*?)\]\s*\[(.*?)\]\s*\[(.*?)\]", hdr, re.S)
                if not mm:
                    break
                offs, sizes, strides = (ML.split_top(mm.group(k)) for k in (2, 3, 4))
                if len(offs) != len(access):
                    break

                def poly(tok):
                    tok = tok.strip()
                    if tok.lstrip("-").isdigit():
                        return ML.p_const(int(tok))
                    return {(("v", key(tok, vop)),): 1}
                if first:
                    for j, sz in enumerate(sizes):
                        dj = ML.pure_dim(access[j])
                        if dj is not None and not sz.strip().isdigit():
                            extents[dj] = key(sz.strip(), vop)
                access = [ML.p_add(poly(o), ML.p_mul(poly(s), a))
                          for o, s, a in zip(offs, strides, access)]
                views.append(vop.idx)
                v, ctx, first = mm.group(1), vop, False
            else:
                break
        t = Trace(key(v, ctx), access, extents, views)
        self._trace_cache[ck] = t
        return t

    # -- body -------------------------------------------------------------
    def body_bindings(self, sop: ML.Op, aop: ML.Op, pi, rho):
        ck = (self.spec.name, sop.idx, aop.idx, pi, rho)
        if ck in self._body_cache:
            return self._body_cache[ck]
        if self.spec.gidx[sop.idx] >= len(self.spec.entry.steps):
            return None
        gb = self.app_gbs[aop.idx]
        res = None
        reference = self.spec.gbs[sop.idx]
        # Scalar algebra does not establish ABI precision or conversion
        # semantics. Compare the actual block types under the operand binding.
        types = gb.block_arg_types or []
        order = list(pi) + [len(gb.ins_arg_names) + r for r in rho]
        if ([types[k] for k in order if k < len(types)] !=
                (reference.block_arg_types or [])):
            self._body_cache[ck] = None
            return None
        if len(gb.ins_arg_names) == len(pi) and len(gb.outs_arg_names) == len(rho):
            perm = dataclasses.replace(
                gb, ins_arg_names=[gb.ins_arg_names[p] for p in pi],
                outs_arg_names=[gb.outs_arg_names[r] for r in rho])
            step = self.spec.entry.steps[self.spec.gidx[sop.idx]]
            one = KM.CompositionEntry(name=self.spec.name, steps=[step], form="tensor")
            try:
                term = KM.encode_body(perm)
                m = KM.match_composition([perm], [term], [one], start=0, body_forms=["tensor"])
            except Exception:
                m = None
            res = None if m is None else dict(m[2])
        self._body_cache[ck] = res
        return res

    @staticmethod
    def _local_defs(body_lines) -> dict[str, str]:
        out = {}
        for ln in body_lines:
            m = re.match(r"(%[\w$.\-]+)\s*=\s*(.*)$", ln.strip())
            if m:
                out[m.group(1)] = m.group(2)
        return out

    def body_local_pairs(self, s_name, a_name, sop, aop, pi, rho, sigma, depth=0):
        """Structural equality of body-local opaque values (linalg.index,
        affine.apply, ...).  Returns extra outer-value pairs or None."""
        if depth > 8:
            return None
        sl = self._local_defs(sop.body)
        al = self._local_defs(aop.body)
        if s_name not in sl or a_name not in al:
            return None
        s_rhs = ML.inline_aliases(sl[s_name], self.L.aliases)
        a_rhs = ML.inline_aliases(al[a_name], self.A.aliases)
        mi_s = re.match(r"linalg\.index\s+(\d+)", s_rhs)
        mi_a = re.match(r"linalg\.index\s+(\d+)", a_rhs)
        if mi_s or mi_a:
            if not (mi_s and mi_a):
                return None
            return [] if sigma[int(mi_s.group(1))] == int(mi_a.group(1)) else None
        if re.sub(r"\s+", " ", ML.SSA_RE.sub("%_", s_rhs)) != re.sub(r"\s+", " ", ML.SSA_RE.sub("%_", a_rhs)):
            return None
        s_ops, a_ops = ML.SSA_RE.findall(s_rhs), ML.SSA_RE.findall(a_rhs)
        if len(s_ops) != len(a_ops):
            return None
        s_args = self.spec.gbs[sop.idx]
        a_args = self.app_gbs[aop.idx]
        pairs = []
        for so, ao in zip(s_ops, a_ops):
            if so in sl:
                sub = self.body_local_pairs(so, ao, sop, aop, pi, rho, sigma, depth + 1)
                if sub is None:
                    return None
                pairs += sub
            elif so in s_args.ins_arg_names:
                i = s_args.ins_arg_names.index(so)
                if ao != a_args.ins_arg_names[pi[i]]:
                    return None
            elif so in s_args.outs_arg_names:
                i = s_args.outs_arg_names.index(so)
                if ao != a_args.outs_arg_names[rho[i]]:
                    return None
            else:
                if ao in al:
                    return None
                k = self.akey(ao, aop)
                if k is None:
                    return None
                pairs.append(("val", so, k))
        return pairs

    # -- access comparison --------------------------------------------------
    @staticmethod
    def _terms(p: dict):
        out = []
        for mono, c in p.items():
            dims = tuple(sorted(a[1] for a in mono if a[0] == "d"))
            vals = [a[1] for a in mono if a[0] == "v"]
            opq = tuple(sorted(a[1] for a in mono if a[0] == "x"))
            out.append((dims, opq, c, vals))
        return out

    def compare_poly(self, ps: dict, pa: dict):
        """ps already renamed into app dims.  Returns pairs or None."""
        st, at = self._terms(ps), self._terms(pa)
        if len(st) != len(at):
            return None
        pairs = []
        used = [False] * len(at)
        for dims, opq, c, vals in sorted(st, key=lambda t: (t[0], t[1], t[2], len(t[3]))):
            hit = None
            for j, (d2, o2, c2, v2) in enumerate(at):
                if not used[j] and d2 == dims and o2 == opq and c2 == c and len(v2) == len(vals):
                    hit = j
                    break
            if hit is not None:
                used[hit] = True
                pairs += [("val", s, a) for s, a in zip(vals, at[hit][3])]
                continue
            # unit-stride specialisation: spec term c*d*s, app term c*d
            if len(vals) == 1:
                for j, (d2, o2, c2, v2) in enumerate(at):
                    if not used[j] and d2 == dims and o2 == opq and c2 == c and not v2:
                        hit = j
                        break
            if hit is None:
                return None
            used[hit] = True
            pairs.append(("unit", vals[0], None))
        return pairs

    def compare_reduced(self, st: Trace, at: Trace, sigma):
        """Spec input of rank k vs application input of rank K > k whose
        K - k extra access components are loop invariant.  The spec parameter
        is then bound to the rank-reducing slice of the application array at
        those fixed coordinates.  Returns (pairs, slice-binding) or None."""
        k, K = len(st.access), len(at.access)
        if K <= k or not isinstance(st.base, str) or st.base not in self.spec.params:
            return None
        fixed = [j for j, pa in enumerate(at.access)
                 if all(a[0] != "d" for mono in pa for a in mono)]
        kept = [j for j in range(K) if j not in fixed]
        if len(kept) != k:
            return None
        pairs = self.compare_access(st.access, [at.access[j] for j in kept], sigma)
        if pairs is None:
            return None
        offs = []
        for j in fixed:
            pa = at.access[j]
            if not pa:
                offs.append(("int", 0))
            elif len(pa) == 1 and list(pa.values())[0] == 1:
                (mono,) = pa.keys()
                if len(mono) == 0:
                    offs.append(("int", 1))
                elif len(mono) == 1 and mono[0][0] == "v" and mono[0][1] is not None:
                    offs.append(("key", mono[0][1]))
                else:
                    return None
            elif len(pa) == 1 and () in pa:
                offs.append(("int", pa[()]))
            else:
                return None
        return pairs, ("slice", at.base, tuple(fixed), tuple(offs), K)

    def compare_access(self, sacc, aacc, sigma):
        if len(sacc) != len(aacc):
            return None
        pairs = []
        for ps, pa in zip(sacc, aacc):
            r = self.compare_poly(ML.rename_dims(ps, sigma), pa)
            if r is None:
                return None
            pairs += r
        return pairs

    # -- value matching -----------------------------------------------------
    def bind(self, env: Env, sname: str, binding: tuple) -> Optional[Env]:
        if sname in env.vals:
            return env if same_binding(env.vals[sname], binding) else None
        e = env.copy()
        e.vals[sname] = binding
        return e

    def match_seq(self, pairs: list, env: Env, ctx: ML.Op) -> Iterator[Env]:
        if not pairs:
            yield env
            return
        head, rest = pairs[0], pairs[1:]
        kind = head[0]
        if kind == "val":
            for e in self.match_value(head[1], head[2], env):
                yield from self.match_seq(rest, e, ctx)
        elif kind == "unit":
            e = self.match_unit(head[1], env)
            if e is not None:
                yield from self.match_seq(rest, e, ctx)
        elif kind == "cap":
            for e in self.match_cap(head[1], head[2], head[3], env):
                yield from self.match_seq(rest, e, ctx)
        elif kind == "slice":
            sname, (tag, base, fixed, offs, rank) = head[1], head[2]
            if base is None or any(o[0] == "key" and o[1] is None for o in offs):
                return
            aty = self.atype(base)
            if aty is None or not aty.startswith("tensor<"):
                return
            e = self.bind(env, sname, ("slice", base, fixed, offs, rank))
            if e is not None:
                yield from self.match_seq(rest, e, ctx)

    def match_unit(self, sname: str, env: Env) -> Optional[Env]:
        d = self.sdef(sname)
        if d is None:
            return None
        sop, _ = d
        if sop.name == "arith.index_cast" and sop.operands and sop.operands[0] in self.spec.params:
            p = sop.operands[0]
            if not self.literal_allowed(p, 1, "int"):
                return None
            return self.bind(env, p, ("int", 1, self.spec.params[p]))
        if sop.name == "arith.constant" and re.search(r"constant\s+1\s*:", sop.text):
            return env
        return None

    def match_cap(self, sname: str, leaf, aop: ML.Op, env: Env) -> Iterator[Env]:
        kind = leaf[0]
        if kind == "Cap":
            k = self.akey(leaf[1], aop)
            if k is None:
                return
            yield from self.match_value(sname, k, env)
        elif kind == "Lit":
            value = float(leaf[1])
            if sname in self.spec.params:
                pty = self.spec.params[sname]
                if not self.literal_allowed(sname, value, "int" if pty.startswith("i") else "float"):
                    return
                gb = self.app_gbs[aop.idx]
                cname = next((n for n, v in gb.constants.items() if v == value), None)
                ckey = self.akey(cname, aop) if cname else None
                e = self.bind(env, sname, ("lit", value, self.spec.params[sname], ckey))
                if e is not None:
                    yield e
                return
            d = self.sdef(sname)
            if d and d[0].name == "arith.constant":
                m = re.search(r"constant\s+([^\s:]+)", d[0].text)
                try:
                    if m and float(m.group(1)) == value:
                        yield env
                except ValueError:
                    pass

    def match_value(self, sname: str, akey, env: Env) -> Iterator[Env]:
        if akey is None:
            return
        if sname in env.vals:
            if same_binding(env.vals[sname], ("ssa", akey)):
                yield env
            return
        sd = self.sdef(sname)
        if sd is None:
            if sname not in self.spec.params:
                return
            pty, aty = self.spec.params[sname], self.atype(akey)
            if aty is None or not types_compatible(pty, aty):
                return
            e = self.bind(env, sname, ("ssa", akey))
            if e is not None:
                yield e
            return
        sop, sk = sd
        aop = self.A.ops[akey[1]] if akey[0] == "op" else None
        if sop.idx in env.opmap:
            if aop is not None and env.opmap[sop.idx] == aop.idx:
                yield env
            return
        if sop.name == "linalg.generic":
            if aop is not None and aop.name == "linalg.generic":
                yield from self.match_generic(sop, sk, aop, akey[2], env)
            return
        if sop.name == "polygeist.submapInverse":
            if aop is not None and aop.name == sop.name:
                yield from self.match_submap_inverse(sop, aop, env)
            return
        if (sop.name == "arith.index_cast" and sop.operands
                and sop.operands[0] in self.spec.params
                and self.spec.params[sop.operands[0]] in ("i32", "i64")):
            p = sop.operands[0]
            if aop is not None and aop.name == "arith.index_cast" and \
                    self.L.skeleton(sop) == self.A.skeleton(aop):
                e = env.copy()
                e.opmap[sop.idx] = aop.idx
                yield from self.match_value(p, self.akey(aop.operands[0], aop), e)
                return
            if self.atype(akey) == "index":
                e = self.bind(env, p, ("cast_i32", akey))
                if e is not None:
                    e.opmap[sop.idx] = akey[1] if akey[0] == "op" else -1
                    yield e
            return
        # generic structural op equality
        if aop is None or aop.name != sop.name or sk != akey[2]:
            return
        if sop.regions or aop.regions:
            return
        if self.L.skeleton(sop) != self.A.skeleton(aop):
            return
        if len(sop.operands) != len(aop.operands):
            return
        e = env.copy()
        e.opmap[sop.idx] = aop.idx
        orders = [list(range(len(sop.operands)))]
        if sop.name in ML.COMMUTATIVE and len(sop.operands) == 2:
            orders.append([1, 0])
        for order in orders:
            pairs = [("val", sop.operands[i], self.akey(aop.operands[j], aop))
                     for i, j in zip(range(len(sop.operands)), order)]
            yield from self.match_seq(pairs, e, aop)

    def match_generic(self, sop, sk, aop, ak, env: Env) -> Iterator[Env]:
        if aop.idx in {g[3] for g in env.gens.values()}:
            return
        self.note(aop, "shape")
        if len(sop.ins) != len(aop.ins) or len(sop.outs) != len(aop.outs):
            return
        if sorted(sop.iters) != sorted(aop.iters):
            return
        nd = len(sop.iters)
        ni, no = len(sop.ins), len(sop.outs)
        try:
            s_tr = [self.trace(self.L, sop, i, False) for i in range(ni + no)]
            a_tr = [self.trace(self.A, aop, i, True) for i in range(ni + no)]
        except ML.AffineParseError:
            return
        sgb = self.spec.gbs[sop.idx]
        s_locals = set(self._local_defs(sop.body))
        for pi in itertools.permutations(range(ni)):
            for rho in itertools.permutations(range(no)):
                if rho[sk] != ak:
                    continue
                b = self.body_bindings(sop, aop, pi, rho)
                if b is None:
                    continue
                self.note(aop, "body")
                for sigma in itertools.permutations(range(nd)):
                    if any(sop.iters[d] != aop.iters[sigma[d]] for d in range(nd)):
                        continue
                    pairs: list = []
                    ok = True
                    amap = list(pi) + [ni + r for r in rho]
                    slice_base: dict = {}
                    for i in range(ni + no):
                        st, at = s_tr[i], a_tr[amap[i]]
                        r = self.compare_access(st.access, at.access, sigma)
                        if r is None and i < ni:
                            r = self.compare_reduced(st, at, sigma)
                            if r is not None:
                                r, slice_base[i] = r
                        if r is None:
                            ok = False
                            break
                        pairs += r
                    if not ok:
                        continue
                    self.note(aop, "access")
                    # iteration extents
                    s_ext, a_ext = {}, {}
                    for i in range(ni + no):
                        for d, v in s_tr[i].extents.items():
                            s_ext.setdefault(d, v)
                        for d, v in a_tr[amap[i]].extents.items():
                            a_ext.setdefault(d, v)
                    for d in range(nd):
                        se, ae = s_ext.get(d), a_ext.get(sigma[d])
                        if (se is None) != (ae is None):
                            ok = False
                            break
                        if se is not None:
                            pairs.append(("val", se, ae))
                    if not ok:
                        continue
                    self.note(aop, "extents")
                    # body captures
                    for cap, leaf in sorted(b.items()):
                        if cap in s_locals:
                            if leaf[0] != "Cap":
                                ok = False
                                break
                            extra = self.body_local_pairs(cap, leaf[1], sop, aop, pi, rho, sigma)
                            if extra is None:
                                ok = False
                                break
                            pairs += extra
                        else:
                            pairs.insert(0, ("cap", cap, leaf, aop))
                    if not ok:
                        continue
                    self.note(aop, "body-locals")
                    # operand bases last (they recurse into producers)
                    for i in range(ni + no):
                        if i in slice_base:
                            pairs.append(("slice", s_tr[i].base, slice_base[i]))
                        else:
                            pairs.append(("val", s_tr[i].base, a_tr[amap[i]].base))
                    e = env.copy()
                    e.opmap[sop.idx] = aop.idx
                    e.gens[sop.idx] = (pi, rho, sigma, aop.idx)
                    views = set()
                    for i in range(ni + no):
                        views.update(a_tr[i].views)
                    e.consumed = e.consumed | frozenset(views) | {aop.idx}
                    for r in self.match_seq(pairs, e, aop):
                        self.note(aop, "matched")
                        yield r
                    self.note(aop, "dataflow")

    def match_submap_inverse(self, sop, aop, env: Env) -> Iterator[Env]:
        def parts(mod, op):
            mm = re.search(r"map\s*=\s*(affine_map<.*?>|#[\w$.\-]+)\s*\}", op.text)
            smap = ML.inline_aliases(mm.group(1), mod.aliases)
            nd, ns, _ = ML.parse_affine_map(smap)
            return smap, op.operands[0], op.operands[1], op.operands[2:2 + ns], op.operands[2 + ns:]
        try:
            s_map, s_base, s_res, s_syms, s_sizes = parts(self.L, sop)
            a_map, a_base, a_res, a_syms, a_sizes = parts(self.A, aop)
        except (ML.AffineParseError, AttributeError):
            return
        if len(s_sizes) != len(a_sizes):
            return
        e0 = env.copy()
        e0.opmap[sop.idx] = aop.idx
        e0.consumed = e0.consumed | {aop.idx}
        for e1 in self.match_value(s_res, self.akey(a_res, aop), e0):
            sd = self.sdef(s_res)
            ad = self.A.def_of(a_res, aop)
            if not sd or not ad or sd[0].name != "linalg.generic" or sd[0].idx not in e1.gens:
                continue
            sg, sk = sd
            ag, ak = ad
            pi, rho, sigma, _ = e1.gens[sg.idx]
            nd = len(sg.iters)
            try:
                s_out = ML.map_apply(sg.maps[len(sg.ins) + sk], ML.identity_dims(nd), [])
                a_out = ML.map_apply(ag.maps[len(ag.ins) + ak], ML.identity_dims(nd), [])
                s_acc = ML.map_apply(s_map, s_out, [{(("v", s),): 1} for s in s_syms])
                a_acc = ML.map_apply(a_map, a_out, [{(("v", self.akey(s, aop)),): 1} for s in a_syms])
            except ML.AffineParseError:
                continue
            pairs = self.compare_access(s_acc, a_acc, sigma)
            if pairs is None:
                continue
            # sizes: result dim j <-> iteration dim
            s_rd = [ML.pure_dim(p) for p in s_out]
            a_rd = [ML.pure_dim(p) for p in a_out]
            ok = True
            for j, d in enumerate(s_rd):
                if d is None or sigma[d] not in a_rd:
                    ok = False
                    break
                pairs.append(("val", s_sizes[j], self.akey(a_sizes[a_rd.index(sigma[d])], aop)))
            if not ok:
                continue
            pairs.append(("val", s_base, self.akey(a_base, aop)))
            yield from self.match_seq(pairs, e1, aop)

    # -- driver ---------------------------------------------------------------
    def find(self, spec: Spec, max_per_root: int = 1) -> list[dict]:
        self.spec = spec
        self._trace_cache = {k: v for k, v in self._trace_cache.items() if k[0]}
        found = []
        y0 = spec.yields[0]
        d = self.sdef(y0)
        if d is None:
            return found
        kind = d[0].name
        for aop in self.A.ops:
            if aop.name != kind or not aop.results:
                continue
            if aop.parent is not None and self.A.ops[aop.parent].name == "linalg.generic":
                continue
            n = 0
            for env in self.match_value(y0, ("op", aop.idx, 0), Env()):
                res = self.finalize(spec, env, aop)
                if res is not None:
                    found.append(res)
                    n += 1
                    if n >= max_per_root:
                        break
        return found

    def finalize(self, spec: Spec, env: Env, root: ML.Op) -> Optional[dict]:
        if len(spec.yields) != 1:
            return None   # multi-output specs: not needed by the current library
        binds = {}
        unbound = []
        for p, ty in spec.func.args:
            if p in env.vals:
                binds[p] = env.vals[p]
            elif p[1:] in spec.meta.get("leading_dims", {}):
                arr = "%" + spec.meta["leading_dims"][p[1:]]
                if arr in env.vals and env.vals[arr][0] == "ssa":
                    binds[p] = ("ld", env.vals[arr][1])
                elif arr in env.vals and env.vals[arr][0] == "slice":
                    binds[p] = ("ld_slice", arr)
                else:
                    unbound.append(p)
            else:
                unbound.append(p)
        if unbound:
            return {"spec": spec.name, "root": root.idx, "rejected": f"unbound parameters {unbound}"}
        gens = [g[3] for g in env.gens.values()]
        if any(self.A.ops[g].block != root.block for g in gens):
            return {"spec": spec.name, "root": root.idx, "rejected": "generics in different blocks"}
        consumed = set(env.consumed) | set(v for v in env.opmap.values() if v >= 0)
        return {"spec": spec.name, "order": spec.order, "root": root.idx, "binds": binds,
                "consumed": consumed, "generics": sorted(gens),
                "gens": {self.L.ops[k].results[0]: {"pi": v[0], "rho": v[1], "sigma": v[2],
                                                    "app": self.A.ops[v[3]].results[0]}
                         for k, v in env.gens.items()}}


def types_compatible(spec_ty: str, app_ty: str) -> bool:
    if spec_ty == app_ty:
        return True
    ms = re.match(r"tensor<((?:\?x)*)(\w+)>", spec_ty)
    ma = re.match(r"tensor<((?:[\d?]+x)*)(\w+)>", app_ty)
    if ms and ma:
        return ms.group(2) == ma.group(2) and ms.group(1).count("x") == ma.group(1).count("x")
    return False


# ---------------------------------------------------------------------------
# Selection + rewrite
# ---------------------------------------------------------------------------

def select(matches: list[dict]) -> list[dict]:
    ok = [m for m in matches if "rejected" not in m]
    ok.sort(key=lambda m: (-len(m["generics"]), -len(m["consumed"]),
                           sum(1 for b in m["binds"].values() if b[0] in ("lit", "int")),
                           m["order"]))
    # Only computations are exclusive: a linalg.generic (and the root whose
    # value the launch replaces) belongs to at most one launch.  Shared pure
    # index arithmetic / views are fine -- dead-code elimination only removes
    # an op once nothing uses it.
    taken_g, taken_roots, out = set(), set(), []
    for m in ok:
        if set(m["generics"]) & taken_g or m["root"] in taken_roots:
            continue
        out.append(m)
        taken_g |= set(m["generics"])
        taken_roots.add(m["root"])
    return out


def fmt_float(v: float) -> str:
    r = repr(float(v))
    if "inf" in r or "nan" in r:
        import struct
        return "0x%016X" % struct.unpack("<Q", struct.pack("<d", v))[0]
    if "e" not in r and "." not in r:
        r += ".0"
    return r


def _rank_of(ty: str) -> int:
    m = re.match(r"(?:memref|tensor)<(.*)>$", ty.strip())
    if not m:
        return 0
    return len(ML.split_top(m.group(1))[0].split("x")) - 1


def _unit_inner_stride(memref_ty: str) -> bool:
    """Statically unit innermost stride: identity layout, or strided<[..., 1]>."""
    m = re.search(r"strided<\[([^\]]*)\]", memref_ty)
    if m:
        return m.group(1).split(",")[-1].strip() == "1"
    return "affine_map" not in memref_ty


def memref_origin(M: "Matcher", lib: "Library", key, depth: int = 0):
    """The function-argument memref whose storage a tensor value occupies
    (in-place bufferization), following destination-passing ops back to the
    `bufferization.to_tensor` it came from.  None if the value has no memref
    origin (a fresh tensor) or is a view of one."""
    if key is None or depth > 64:
        return None
    A = M.A
    if key[0] == "arg":
        ty = M.atype(key)
        return key if ty and ty.startswith("memref") else None
    if key[0] == "blk":
        owner = A.ops[key[1]]
        m = re.search(r"iter_args\((.*?)\)\s*->", owner.header)
        if not m:
            return None
        for part in ML.split_top(m.group(1)):
            a, _, init = part.partition("=")
            if a.strip() == key[2]:
                return memref_origin(M, lib, M.akey(init.strip(), owner), depth + 1)
        return None
    op = A.ops[key[1]]
    k = key[2]

    def via(name):
        return memref_origin(M, lib, M.akey(name, op), depth + 1)
    if op.name in ("bufferization.to_tensor", "polygeist.submapInverse", "tensor.cast") and op.operands:
        return via(op.operands[0])
    if op.name == "tensor.insert_slice" and len(op.operands) > 1:
        return via(op.operands[1])
    if op.name == "linalg.generic" and k < len(op.outs):
        d = A.def_of(op.outs[k], op)
        if d is not None and d[0].name in ("polygeist.submap", "tensor.extract_slice"):
            return None             # result is a view, not the whole storage
        return via(op.outs[k])
    if op.name in ("affine.for", "scf.for"):
        m = re.search(r"iter_args\((.*?)\)\s*->", op.header)
        if m:
            parts = ML.split_top(m.group(1))
            if k < len(parts):
                return via(parts[k].partition("=")[2].strip())
        return None
    if op.name == "kernel.launch":
        mm = re.search(r"@([\w$.\-]+)", op.header)
        spec = next((s_ for s_ in lib.specs if mm and s_.name == mm.group(1)), None)
        if spec is not None:
            outs = spec.meta.get("outputs", [])
            params = spec.meta.get("params", [])
            if k < len(outs) and outs[k] in params and params.index(outs[k]) < len(op.operands):
                return via(op.operands[params.index(outs[k])])
    return None


def ld_source(M: "Matcher", lib: "Library", b: tuple, binds: dict):
    """How a leading-dimension binding is obtained:
       ("stride", memref key, outer dim, inner dim, offset terms) or
       ("extent", tensor name) for a tensor with no memref origin."""
    if b[0] == "ld":
        org = memref_origin(M, lib, b[1])
        if org is not None:
            return ("stride", org, 0, 1, [])
        return ("extent", M.aname(b[1]))
    if b[0] == "ld_slice":
        sl = binds.get(b[1])
        if sl and sl[0] == "slice":
            _, base, fixed, offs, rank = sl
            org = memref_origin(M, lib, base)
            kept = [j for j in range(rank) if j not in fixed]
            if org is not None and len(kept) >= 2:
                return ("stride", org, kept[0], kept[1], list(zip(fixed, offs)))
        return ("extent", None)
    return None


class Rewriter:
    def __init__(self, matcher: Matcher, lib: Library):
        self.M = matcher
        self.A = matcher.A
        self.lib = lib
        self.counter = 0
        self.md: dict = {}

    def leading_dim(self, src, pty: str, pre: list, indent: str) -> str:
        """Leading dimension from the memref stride (extract_strided_metadata):
        stride[outer] (unit inner stride) or stride[outer]/stride[inner]."""
        _, mkey, outer, inner, _offs = src
        name, ty = self.M.aname(mkey), self.M.atype(mkey)
        r = _rank_of(ty)
        if name not in self.md:
            md = self.fresh("md")
            elem = re.search(r"x?(f64|f32|f16|bf16|i64|i32)\s*(?:,|>)", ty)
            pre.append(f"{indent}{md}:{2 + 2 * r} = memref.extract_strided_metadata {name} : {ty} -> "
                       f"memref<{elem.group(1) if elem else 'f64'}>, {', '.join(['index'] * (1 + 2 * r))}")
            self.md[name] = md
        md = self.md[name]
        val = f"{md}#{2 + r + outer}"
        if not (inner == r - 1 and _unit_inner_stride(ty)):
            q = self.fresh("ldq")
            pre.append(f"{indent}{q} = arith.divui {val}, {md}#{2 + r + inner} : index")
            val = q
        nm = self.fresh("i")
        pre.append(f"{indent}{nm} = arith.index_cast {val} : index to {pty}")
        return nm

    def fresh(self, stem: str) -> str:
        self.counter += 1
        return f"%lv2_{stem}{self.counter}"

    def render(self, m: dict) -> tuple[str, list[str]]:
        self.md = {}      # metadata values are local to one launch (dominance)
        spec = next(s for s in self.lib.specs if s.name == m["spec"])
        root = self.A.ops[m["root"]]
        indent = re.match(r"\s*", root.text).group(0)
        pre, ops, tys, uses = [], [], [], []
        c1 = None
        rendered: dict[str, str] = {}
        args = list(spec.func.args)
        # leading dimensions refer to their (rendered) array: render them last
        order = [i for i, (p, _) in enumerate(args) if m["binds"][p][0] not in ("ld", "ld_slice")] + \
                [i for i, (p, _) in enumerate(args) if m["binds"][p][0] in ("ld", "ld_slice")]
        ops = [None] * len(args)
        tys = [pty for _, pty in args]
        for idx in order:
            p, pty = args[idx]
            b = m["binds"][p]
            kind = b[0]
            if kind in ("ld", "ld_slice"):
                src = ld_source(self.M, self.lib, b, m["binds"])
                if src is not None and src[0] == "stride":
                    ops[idx] = self.leading_dim(src, pty, pre, indent)
                    rendered[p] = ops[idx]
                    continue
            if kind == "ld_slice":
                # a fresh dense tensor (no memref origin): its own layout
                arr = rendered[b[1]]
                if c1 is None:
                    c1 = self.fresh("c")
                    pre.append(f"{indent}{c1} = arith.constant 1 : index")
                dn = self.fresh("ld")
                pre.append(f"{indent}{dn} = tensor.dim {arr}, {c1} : {spec.params[b[1]]}")
                nm = self.fresh("i")
                pre.append(f"{indent}{nm} = arith.index_cast {dn} : index to {pty}")
                ops[idx] = nm
                continue
            if kind == "ssa":
                nm, aty = self.M.aname(b[1]), self.M.atype(b[1])
                if aty != pty:
                    cn = self.fresh("cast")
                    pre.append(f"{indent}{cn} = tensor.cast {nm} : {aty} to {pty}")
                    nm = cn
                uses.append(self.M.aname(b[1]))
            elif kind == "lit":
                if b[3] is not None and self.M.atype(b[3]) == pty:
                    nm = self.M.aname(b[3])
                    uses.append(nm)
                else:
                    nm = self.fresh("c")
                    pre.append(f"{indent}{nm} = arith.constant {fmt_float(b[1])} : {pty}")
            elif kind == "int":
                nm = self.fresh("c")
                pre.append(f"{indent}{nm} = arith.constant {b[1]} : {pty}")
            elif kind == "cast_i32":
                src = self.M.aname(b[1])
                nm = self.fresh("i")
                pre.append(f"{indent}{nm} = arith.index_cast {src} : index to {pty}")
                uses.append(src)
            elif kind == "slice":
                _, base, fixed, offs, rank = b
                src, sty = self.M.aname(base), self.M.atype(base)
                offs_t, sizes_t, strides_t = [], [], []
                oi = 0
                for j in range(rank):
                    if j in fixed:
                        o = offs[oi]
                        oi += 1
                        offs_t.append(str(o[1]) if o[0] == "int" else self.M.aname(o[1]))
                        if o[0] == "key":
                            uses.append(self.M.aname(o[1]))
                        sizes_t.append("1")
                    else:
                        cj = self.fresh("c")
                        pre.append(f"{indent}{cj} = arith.constant {j} : index")
                        dj = self.fresh("dim")
                        pre.append(f"{indent}{dj} = tensor.dim {src}, {cj} : {sty}")
                        offs_t.append("0")
                        sizes_t.append(dj)
                    strides_t.append("1")
                nm = self.fresh("slice")
                pre.append(f"{indent}{nm} = tensor.extract_slice {src}[{', '.join(offs_t)}] "
                           f"[{', '.join(sizes_t)}] [{', '.join(strides_t)}] : {sty} to {pty}")
                uses.append(src)
            elif kind == "ld":
                arr, aty = self.M.aname(b[1]), self.M.atype(b[1])
                if c1 is None:
                    c1 = self.fresh("c")
                    pre.append(f"{indent}{c1} = arith.constant 1 : index")
                dn = self.fresh("ld")
                pre.append(f"{indent}{dn} = tensor.dim {arr}, {c1} : {aty}")
                nm = self.fresh("i")
                pre.append(f"{indent}{nm} = arith.index_cast {dn} : index to {pty}")
                uses.append(arr)
            else:
                raise RuntimeError(f"unknown binding {b}")
            ops[idx] = nm
            rendered[p] = nm
        out_ty = spec.func.text.split("\n", 1)[0]
        mres = re.search(r"\)\s*->\s*(.+?)\s*\{\s*$", out_ty)
        rty = mres.group(1) if mres else ""
        res_name = root.results[0]
        aty = root.result_types[0] if root.result_types else rty
        launch_res = res_name if aty == rty else self.fresh("r")
        line = (f"{indent}{launch_res} = kernel.launch @{spec.name}({', '.join(ops)}) : "
                f"({', '.join(tys)}) -> {rty}")
        lines = pre + [line]
        if launch_res != res_name:
            lines.append(f"{indent}{res_name} = tensor.cast {launch_res} : {rty} to {aty}")
        return "\n".join(lines) + "\n", uses

    def rewrite(self, selected: list[dict]) -> str:
        edits = {}
        deleted: set[int] = set()
        extra_uses: list[tuple[ML.Op, str]] = []
        roots = set()
        for m in selected:
            text, uses = self.render(m)
            root = self.A.ops[m["root"]]
            edits[root.idx] = text
            roots.add(root.idx)
            extra_uses += [(root, u) for u in uses]
        candidates = set()
        for m in selected:
            candidates |= {i for i in m["consumed"] if i not in roots}
        # Iterative dead-code elimination restricted to matched ops.
        nested_in = {}
        for op in self.A.ops:
            p = op.parent
            while p is not None:
                nested_in.setdefault(op.idx, set()).add(p)
                p = self.A.ops[p].parent
        changed = True
        while changed:
            changed = False
            live_defs: set[int] = set()
            for op in self.A.ops:
                if op.idx in deleted or (nested_in.get(op.idx, set()) & deleted):
                    continue
                if op.idx in roots:
                    continue
                for u in op.uses:
                    definition = (self.A.def_of(u, op) or
                                  self.A.def_of(u + "#0", op))
                    if definition:
                        live_defs.add(definition[0].idx)
            for root, u in extra_uses:
                definition = (self.A.def_of(u, root) or
                              self.A.def_of(ML.base_name(u) + "#0", root))
                if definition:
                    live_defs.add(definition[0].idx)
            for i in sorted(candidates - deleted):
                op = self.A.ops[i]
                if op.results and i not in live_defs:
                    deleted.add(i)
                    changed = True
        text = self.A.text
        spans = []
        for i, rep in edits.items():
            op = self.A.ops[i]
            spans.append((op.start, op.end, rep))
        for i in deleted:
            op = self.A.ops[i]
            spans.append((op.start, op.end, ""))
        for s, e, rep in sorted(spans, key=lambda t: -t[0]):
            text = text[:s] + rep + text[e:]
        return text


# ---------------------------------------------------------------------------
# Region matcher (Agent 2) splice
# ---------------------------------------------------------------------------

class _RegionTimeout(Exception):
    pass


def splice_region_matches(text: str, report: list, budget: int = 600,
                          allowed_specs=None) -> str:
    try:
        sys.path.insert(0, str(HERE.parent))
        from region_match import match_regions  # type: ignore
    except Exception as e:  # module absent or broken: linalg-level result stands
        report.append({"region_match": f"unavailable ({type(e).__name__})"})
        return text
    import signal

    def _alarm(*_):
        raise _RegionTimeout()
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(budget)
    try:
        regions = match_regions(text, str(SPEC_DIR), allowed_specs=allowed_specs)
    except _RegionTimeout:
        report.append({"region_match": f"timeout after {budget}s"})
        return text
    except Exception as e:
        report.append({"region_match": f"error {type(e).__name__}: {e}"})
        return text
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    launches = [(m.start(), m.end()) for m in re.finditer(r"kernel\.launch[^\n]*", text)]
    keep = []
    for r in sorted(regions, key=lambda r: -r.span[0]):
        if getattr(r, "recognition_only", False):
            # reported with its binding; the loops stay (no lowering exists)
            report.append({"region_match": r.spec, "recognition_only": True,
                           "binding": r.binding, "evidence": r.evidence})
            continue
        if any(not (r.span[1] <= s or r.span[0] >= e) for s, e in launches):
            report.append({"region_match_skipped": r.spec, "reason": "overlaps a linalg-level launch"})
            continue
        text = text[:r.span[0]] + r.launch_text + text[r.span[1]:]
        keep.append(r)
        report.append({"region_match": r.spec, "binding": r.binding, "evidence": r.evidence})
    return text


# ---------------------------------------------------------------------------
# Diagnostics: which spec steps a residual generic's body matches
# ---------------------------------------------------------------------------

def diagnose(matcher: Matcher, lib: Library, consumed: set[int]) -> list[dict]:
    out = []
    for aop in matcher.A.ops:
        if aop.name != "linalg.generic" or aop.idx in consumed:
            continue
        hits = []
        for spec in lib.specs:
            matcher.spec = spec
            for sop in spec.generics:
                if len(sop.ins) != len(aop.ins) or len(sop.outs) != len(aop.outs):
                    continue
                if sorted(sop.iters) != sorted(aop.iters):
                    continue
                for pi in itertools.permutations(range(len(aop.ins))):
                    if matcher.body_bindings(sop, aop, pi, tuple(range(len(aop.outs)))) is not None:
                        hits.append(f"{spec.name}#step{spec.gidx[sop.idx]}")
                        break
        stages = [(Matcher.STAGES.index(st), s) for (s, a), st in matcher.stage.items()
                  if a == aop.idx]
        furthest = None
        if stages:
            best = max(stages)[0]
            furthest = {"stage": Matcher.STAGES[best],
                        "specs": sorted(s for i, s in stages if i == best)[:6]}
        out.append({"app_generic": aop.results[0] if aop.results else f"op{aop.idx}",
                    "func": aop.func, "iters": aop.iters,
                    "form": "tensor" if aop.result_types else "memref",
                    "body_matches": sorted(set(hits))[:12],
                    "n_body_matches": len(set(hits)),
                    "furthest_dag_stage": furthest})
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run(text: str, lib_path: Path, region: bool = True, diag: bool = False,
        region_budget: int = 600, allowed_specs=None):
    import_kernel_match(lib_path)
    lib = Library(lib_path)
    if allowed_specs is not None:
        lib.specs = [s for s in lib.specs if s.name in allowed_specs]
    app = ML.Module(text)
    matcher = Matcher(lib, app)
    all_matches, rejected = [], []
    for spec in lib.specs:
        for m in matcher.find(spec):
            (rejected if "rejected" in m else all_matches).append(m)
    selected = select(all_matches)
    out = Rewriter(matcher, lib).rewrite(selected) if selected else text
    report: list = []
    for m in selected:
        report.append({
            "spec": m["spec"], "root": app.ops[m["root"]].results[0],
            "func": app.ops[m["root"]].func,
            "binding": {p: describe_binding(matcher, b, lib, m["binds"]) for p, b in m["binds"].items()},
            "generics": m["gens"],
            "consumed_ops": len(m["consumed"])})
    alts = sorted({(m["spec"], app.ops[m["root"]].results[0]) for m in all_matches}
                  - {(m["spec"], app.ops[m["root"]].results[0]) for m in selected})
    if region:
        out = splice_region_matches(out, report, region_budget, allowed_specs)
    syms = re.findall(r"kernel\.launch\s+@([\w$.\-]+)", out)
    names = lib.names()
    known = set(lib.entries)
    for s in syms:
        assert s in known, f"launch @{s} has no imported specification"
        assert allowed_specs is None or s in allowed_specs, f"unavailable target @{s}"
    for m in selected:
        assert m["spec"] in names
    extra = {"literal_policy_rejections": sorted(set(matcher.policy_rejections))[:50],
             "alternatives": [f"{a}@{b}" for a, b in alts],
             "rejected": [{"spec": r["spec"], "root": app.ops[r["root"]].results[0],
                           "why": r["rejected"]} for r in rejected]}
    if diag:
        consumed = set()
        for m in selected:
            consumed |= set(m["generics"])
        extra["residual_generics"] = diagnose(matcher, lib, consumed)
    return out, report, extra


def describe_binding(matcher: Matcher, b: tuple, lib: Optional[Library] = None,
                     binds: Optional[dict] = None) -> str:
    k = b[0]
    if k in ("ld", "ld_slice") and lib is not None:
        src = ld_source(matcher, lib, b, binds or {})
        if src is not None and src[0] == "stride":
            _, mkey, outer, inner, offs = src
            nm = matcher.aname(mkey)
            txt = f"stride{outer}({nm})" if inner == _rank_of(matcher.atype(mkey)) - 1 \
                else f"stride{outer}({nm})/stride{inner}({nm})"
            if offs:
                terms = [f"{(str(o[1]) if o[0] == 'int' else matcher.aname(o[1]))}*stride{j}({nm})"
                         for j, o in offs]
                txt += f"  [slice offset offset({nm}) + {' + '.join(terms)}]"
            return txt
        if src is not None:
            return (f"dim({matcher.aname(b[1])}, 1)  (dense tensor value without memref origin: "
                    "leading dimension of its own layout)" if k == "ld" else
                    f"dim(<slice bound to {b[1]}>, 1)  (no memref origin)")
    if k == "ssa":
        return matcher.aname(b[1])
    if k == "lit":
        return f"literal {b[1]!r}" + (f" ({matcher.aname(b[3])})" if b[3] else "")
    if k == "int":
        return f"literal {b[1]}"
    if k == "cast_i32":
        return f"index_cast({matcher.aname(b[1])})"
    if k == "ld":
        return f"dim({matcher.aname(b[1])}, 1)"
    if k == "ld_slice":
        return f"dim(<slice bound to {b[1]}>, 1)"
    if k == "slice":
        _, base, fixed, offs, rank = b
        coords, oi = [], 0
        for j in range(rank):
            if j in fixed:
                o = offs[oi]
                oi += 1
                coords.append(str(o[1]) if o[0] == "int" else matcher.aname(o[1]))
            else:
                coords.append(":")
        return f"{matcher.aname(base)}[{', '.join(coords)}]  (rank-reducing extract_slice)"
    return str(b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--library", default=str(DEFAULT_LIB))
    ap.add_argument("--report")
    ap.add_argument("--inject-defns", action="store_true",
                    help="also splice the used kernel.defn ops into the module")
    ap.add_argument("--no-region-match", action="store_true")
    ap.add_argument("--region-budget", type=int, default=600,
                    help="seconds allowed for the optional region matcher")
    ap.add_argument("--diagnose", action="store_true")
    a = ap.parse_args()
    text = Path(a.input).read_text()
    out, report, extra = run(text, Path(a.library), region=not a.no_region_match,
                             diag=a.diagnose, region_budget=a.region_budget)
    if a.inject_defns:
        syms = sorted(set(re.findall(r"kernel\.launch\s+@([\w$.\-]+)", out)))
        if syms:
            import inject_kernel_library as IKL
            out = IKL.inject(out, Path(a.library).read_text(), syms)
    if a.output:
        Path(a.output).write_text(out)
    else:
        sys.stdout.write(out)
    if a.report:
        Path(a.report).write_text(json.dumps({"launches": report, **extra}, indent=1, default=str))
    n = len(re.findall(r"kernel\.launch\s+@", out))
    print(f"match_v2: {n} kernel.launch ({', '.join(r['spec'] for r in report if 'spec' in r)})",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
