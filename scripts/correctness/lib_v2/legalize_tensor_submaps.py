#!/usr/bin/env python3
"""Test-only legalisation of tensor-form polygeist.submap views for CPU runs.

The production recipe keeps `polygeist.submap` / `submapInverse` views through
`--linalg-debufferize=use-multi-root=true`.  For dynamic shapes, the in-tree
`--lower-polygeist-submap` cannot lower a tensor-form reduction write-back
(NormalizeTensorReductionOutputSubmap requires static sizes) and the loop
fallback only applies to memrefs, so such IR cannot be executed on the host.

This script rewrites every
    %v_i = polygeist.submap(%base_i, syms..., sizes...) {map}      (chains ok)
    %g   = linalg.generic ins(%v...) outs(%v_o)
    %w   = polygeist.submapInverse(%base_o, %g, ...) {map}
group whose composed accesses (view map o indexing map) are, per base
dimension, either `c*d_k*stride + offset` (one iteration dim, unit or symbol
stride) or loop invariant, into
    %s_i = tensor.extract_slice %base_i[offsets][extents][strides]
    %g   = linalg.generic {projected indexing maps} ins(%s...) outs(%s_o)
    %w   = tensor.insert_slice %g into %base_o[...]
which one-shot-bufferize lowers normally.  This is the tensor analogue of the
memref-form lowering the campaign recipe applies before debufferization.
Unsupported groups are left untouched (and later fail to lower, visibly).
Usage: legalize_tensor_submaps.py IN.mlir -o OUT.mlir
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mlir_lite as ML  # noqa: E402


def submap_parts(mod: ML.Module, op: ML.Op):
    mm = re.search(r"map\s*=\s*(affine_map<.*?>|#[\w$.\-]+)\s*\}", op.text)
    smap = ML.inline_aliases(mm.group(1), mod.aliases)
    nd, ns, _ = ML.parse_affine_map(smap)
    return smap, ns


def trace(mod: ML.Module, gop: ML.Op, i: int):
    operands = gop.ins + gop.outs
    v, ctx = operands[i], gop
    nd = len(gop.iters)
    access = ML.map_apply(gop.maps[i], ML.identity_dims(nd), [])
    extents, views, first = {}, [], True
    while True:
        d = mod.def_of(v, ctx)
        if d is None or d[0].name != "polygeist.submap":
            break
        vop = d[0]
        smap, ns = submap_parts(mod, vop)
        syms, sizes = vop.operands[1:1 + ns], vop.operands[1 + ns:]
        if first:
            for j, sz in enumerate(sizes):
                dj = ML.pure_dim(access[j]) if j < len(access) else None
                if dj is not None:
                    extents[dj] = sz
        access = ML.map_apply(smap, access, [{(("v", s),): 1} for s in syms])
        views.append(vop)
        v, ctx, first = vop.operands[0], vop, False
    return v, access, extents, views


def classify(poly: dict):
    """-> ('fixed', offset_terms) | ('dim', d, stride_sym_or_None, coeff, offset_terms) | None"""
    dim_terms = [(m, c) for m, c in poly.items() if any(a[0] == "d" for a in m)]
    rest = {m: c for m, c in poly.items() if not any(a[0] == "d" for a in m)}
    if any(a[0] == "x" for m in poly for a in m):
        return None
    if not dim_terms:
        return ("fixed", rest)
    if len(dim_terms) != 1:
        return None
    mono, c = dim_terms[0]
    dims = [a for a in mono if a[0] == "d"]
    syms = [a for a in mono if a[0] == "v"]
    if len(dims) != 1 or len(syms) > 1 or c <= 0:
        return None
    return ("dim", dims[0][1], syms[0][1] if syms else None, c, rest)


_COUNTER = [0]


class Emitter:
    def __init__(self, indent: str):
        self.indent = indent
        self.lines: list[str] = []

    def fresh(self, stem):
        _COUNTER[0] += 1
        return f"%lgz_{stem}{_COUNTER[0]}"

    def const(self, v: int) -> str:
        nm = self.fresh("c")
        self.lines.append(f"{self.indent}{nm} = arith.constant {v} : index")
        return nm

    def value(self, terms: dict) -> str:
        """Materialise an index polynomial over symbols (no dims)."""
        if not terms:
            return "0"
        if len(terms) == 1:
            (m, c), = terms.items()
            if m == ():
                return str(c)
            if c == 1 and len(m) == 1:
                return m[0][1]
        acc = None
        for m, c in sorted(terms.items(), key=str):
            t = self.const(c) if m == () else None
            if m != ():
                t = m[0][1]
                for a in m[1:]:
                    nm = self.fresh("m")
                    self.lines.append(f"{self.indent}{nm} = arith.muli {t}, {a[1]} : index")
                    t = nm
                if c != 1:
                    k = self.const(c)
                    nm = self.fresh("m")
                    self.lines.append(f"{self.indent}{nm} = arith.muli {t}, {k} : index")
                    t = nm
            if acc is None:
                acc = t
            else:
                nm = self.fresh("a")
                self.lines.append(f"{self.indent}{nm} = arith.addi {acc}, {t} : index")
                acc = nm
        return acc


def split_independent_outputs(text: str) -> tuple[str, int]:
    """Split pure, independent tensor outputs before legalizing broadcast views.

    A raised loop can write an elementwise result and a reduction accumulator
    together. Their composed output accesses need different iterator kinds in
    ordinary Linalg. Splitting is safe only when no result reads another
    output's previous value. Keep each result's scalar dependency closure;
    duplicating pure arithmetic does not change the reduction's evaluation order.
    """
    mod = ML.Module(text)
    count = 0
    for f in reversed(mod.funcs):
        edits, replacements = [], {}
        for g in f.ops:
            if (g.name != "linalg.generic" or g.parent is not None or
                    len(g.outs) < 2 or len(g.results) != len(g.outs) or
                    not all(t.startswith("tensor<") for t in g.in_types + g.out_types)):
                continue
            try:
                traces = [trace(mod, g, len(g.ins) + j) for j in range(len(g.outs))]
            except ML.AffineParseError:
                continue
            if not any(t[3] for t in traces):
                continue
            children = [op for op in mod.ops if op.parent == g.idx]
            if (not children or children[-1].name != "linalg.yield" or
                    any(op.regions or not op.name.startswith(("arith.", "math.", "linalg.index"))
                        for op in children[:-1])):
                continue
            defs = {r: op for op in children[:-1] for r in op.results}
            block = re.search(r"\^bb\d+\((.*?)\):", g.text)
            if block is None:
                continue
            args = ML.split_top(block.group(1))
            names = [a.split(":", 1)[0].strip() for a in args]
            ni = len(g.ins)
            if len(names) != ni + len(g.outs) or len(children[-1].operands) != len(g.outs):
                continue
            output_names = set(names[ni:])
            pieces, new_results = [], []
            for j, (result, out, ty, tr) in enumerate(zip(g.results, g.outs, g.out_types, traces)):
                pending = [children[-1].operands[j]]
                seen, needed = set(), set()
                while pending:
                    v = pending.pop()
                    if v in seen:
                        continue
                    seen.add(v)
                    if v in defs:
                        op = defs[v]
                        needed.add(op.idx)
                        pending.extend(op.operands)
                if (seen & output_names) - {names[ni + j]}:
                    break
                classified = [classify(p) for p in tr[1]]
                if any(c is None for c in classified):
                    break
                kept = [c[1] for c in classified if c[0] == "dim"]
                if len(kept) != len(set(kept)):
                    break
                if any(k not in kept and it != "reduction" for k, it in enumerate(g.iters)):
                    break
                iters = ["parallel" if k in kept else "reduction" for k in range(len(g.iters))]
                em = Emitter(re.match(r"\s*", g.header).group(0))
                fresh = em.fresh("out")
                hdr = ML.inline_aliases(g.header, mod.aliases)
                hdr = re.sub(r"^([ \t]*)%[^=]+ =", lambda m: m[1] + fresh + " =", hdr)
                maps = g.maps[:ni] + [g.maps[ni + j]]
                hdr = re.sub(r"indexing_maps\s*=\s*\[.*?\](?=,\s*iterator_types)",
                             lambda _: "indexing_maps = [" + ", ".join(maps) + "]", hdr)
                hdr = re.sub(r"iterator_types\s*=\s*\[[^]]*\]",
                             lambda _: "iterator_types = [" + ", ".join('"' + it + '"' for it in iters) + "]", hdr)
                hdr = re.sub(r"outs\(.*?\)\s*\{\s*$", f"outs({out} : {ty}) {{", hdr)
                body = "".join(op.text for op in children[:-1] if op.idx in needed)
                ind = em.indent
                elem = args[ni + j].split(":", 1)[1].strip()
                pieces.append(hdr + "\n" + ind + "^bb0(" + ", ".join(args[:ni] + [args[ni + j]]) + "):\n" +
                              body + ind + "  linalg.yield " + children[-1].operands[j] + " : " + elem + "\n" +
                              ind + "} -> " + ty + "\n")
                new_results.append((result, fresh))
            else:
                edits.append((g.start - f.start, g.end - f.start, "".join(pieces)))
                replacements.update(new_results)
        if not edits:
            continue
        ft = text[f.start:f.end]
        for s, e, replacement in reversed(edits):
            ft = ft[:s] + replacement + ft[e:]
        ft = ML.SSA_RE.sub(lambda m: replacements.get(m.group(0), m.group(0)), ft)
        text = text[:f.start] + ft + text[f.end:]
        count += len(edits)
    return text, count


def legalize_func_ops(mod: ML.Module) -> list[tuple[int, int, str]]:
    edits = []
    for g in mod.ops:
        if g.name != "linalg.generic" or not g.results or not g.result_types:
            continue
        if not all(t.startswith("tensor<") for t in g.in_types + g.out_types):
            continue
        n = len(g.ins) + len(g.outs)
        try:
            traces = [trace(mod, g, i) for i in range(n)]
        except ML.AffineParseError:
            continue
        if not any(t[3] for t in traces):
            continue
        if len(g.outs) != 1:
            continue
        # The single result must feed exactly one submapInverse (or nothing
        # if the outs is not a view).
        res = g.results[0]
        users = [op for op in mod.ops if op.func == g.func and op is not g
                 and res in op.operands]
        inv = None
        if traces[-1][3]:
            if len(users) != 1 or users[0].name != "polygeist.submapInverse" or \
                    users[0].operands[1] != res or users[0].block != g.block:
                continue
            inv = users[0]
        nd = len(g.iters)
        extents: dict = {}
        for t in traces:
            for d, v in t[2].items():
                extents.setdefault(d, v)
        indent = re.match(r"\s*", g.text).group(0)
        em = Emitter(indent)
        new_ops, new_types, new_maps = [], [], []
        ok = True
        out_slice = None
        for i, (base, access, _, views) in enumerate(traces):
            if not views:
                new_ops.append((g.ins + g.outs)[i])
                new_types.append((g.in_types + g.out_types)[i])
                new_maps.append(g.maps[i])
                continue
            bty = mod.value_type(base, views[-1])
            if bty is None or not bty.startswith("tensor<"):
                ok = False
                break
            elem = re.match(r"tensor<(?:[\d?]+x)*(\w+)>", bty).group(1)
            offs, sizes, strides, kept = [], [], [], []
            for p in access:
                c = classify(p)
                if c is None:
                    ok = False
                    break
                if c[0] == "fixed":
                    offs.append(em.value(c[1]))
                    sizes.append("1")
                    strides.append("1")
                else:
                    _, d, sym, coeff, rest = c
                    if d in kept or d not in extents:
                        ok = False
                        break
                    kept.append(d)
                    offs.append(em.value(rest))
                    sizes.append(extents[d])
                    if sym is None:
                        strides.append(str(coeff))
                    else:
                        strides.append(sym if coeff == 1 else em.value({(("v", sym),): coeff}))
            if not ok:
                break
            sty = "tensor<" + "?x" * len(kept) + elem + ">"
            nm = em.fresh("slice")
            em.lines.append(f"{indent}{nm} = tensor.extract_slice {base}[{', '.join(offs)}] "
                            f"[{', '.join(sizes)}] [{', '.join(strides)}] : {bty} to {sty}")
            dims = ", ".join(f"d{k}" for k in range(nd))
            new_maps.append(f"affine_map<({dims}) -> ({', '.join(f'd{k}' for k in kept)})>")
            new_ops.append(nm)
            new_types.append(sty)
            if i == n - 1:
                out_slice = (base, offs, sizes, strides, bty, sty)
        if not ok:
            continue
        if inv is not None:
            # the write-back must target the same base with the same access
            try:
                smap, ns = submap_parts(mod, inv)
                ib = inv.operands[0]
                o_acc = ML.map_apply(g.maps[-1], ML.identity_dims(nd), [])
                w_acc = ML.map_apply(smap, o_acc, [{(("v", s),): 1} for s in inv.operands[2:2 + ns]])
            except ML.AffineParseError:
                continue
            if ib != traces[-1][0] or w_acc != traces[-1][1]:
                continue
        # rewrite the generic header
        hdr = g.header
        ni = len(g.ins)
        maps_txt = "[" + ", ".join(new_maps) + "]"
        hdr2 = re.sub(r"indexing_maps\s*=\s*\[.*?\](?=,\s*iterator_types)",
                      lambda m: "indexing_maps = " + maps_txt, ML.inline_aliases(hdr, mod.aliases), count=1)
        if ni:
            hdr2 = re.sub(r"ins\(.*?\)\s*outs\(",
                          f"ins({', '.join(new_ops[:ni])} : {', '.join(new_types[:ni])}) outs(",
                          hdr2, count=1)
        hdr2 = re.sub(r"outs\(.*?\)\s*\{\s*$",
                      f"outs({new_ops[-1]} : {new_types[-1]}) {{", hdr2.rstrip("\n"), count=1) + "\n"
        lines = g.text.split("\n")
        body = "\n".join(lines[1:])
        body = re.sub(r"\}\s*->\s*tensor<[^\n]*$", f"}} -> {new_types[-1]}", body.rstrip("\n")) + "\n"
        new_text = "\n".join(em.lines) + ("\n" if em.lines else "") + hdr2 + body
        edits.append((g.start, g.end, new_text))
        if inv is not None:
            base, offs, sizes, strides, bty, sty = out_slice
            ind = re.match(r"\s*", inv.text).group(0)
            edits.append((inv.start, inv.end,
                          f"{ind}{inv.results[0]} = tensor.insert_slice {res} into {base}"
                          f"[{', '.join(offs)}] [{', '.join(sizes)}] [{', '.join(strides)}]"
                          f" : {sty} into {bty}\n"))
    return edits


def legalize(text: str) -> tuple[str, int]:
    used = [int(x) for x in re.findall(r"%lgz_[a-z]+(\d+)", text)]
    _COUNTER[0] = max([_COUNTER[0], *used])
    text, splits = split_independent_outputs(text)
    mod = ML.Module(text)
    edits = legalize_func_ops(mod)
    for s, e, rep in sorted(edits, key=lambda t: -t[0]):
        text = text[:s] + rep + text[e:]
    return text, len(edits) + splits


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    text = Path(a.input).read_text()
    total = 0
    for _ in range(4):   # nested groups: iterate to a fixpoint
        text, n = legalize(text)
        total += n
        if n == 0:
            break
    Path(a.output).write_text(text)
    print(f"legalize_tensor_submaps: rewrote {total} op(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
