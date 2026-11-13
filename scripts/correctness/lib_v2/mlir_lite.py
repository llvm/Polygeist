#!/usr/bin/env python3
"""Minimal structural reader for the textual MLIR printed by polygeist-opt.

Only what the lib_v2 matcher needs: functions, blocks, ops with spans,
results, operands, free SSA uses, result types, linalg.generic headers and
affine-map algebra (parse / compose / rename) as integer polynomials.
Nothing here knows about any library routine.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

SSA_RE = re.compile(r"%[\w$.\-]+(?:#\d+)?")
ALIAS_DEF_RE = re.compile(r"^(#[\w$.\-]+)\s*=\s*(.+)$", re.MULTILINE)
COMMUTATIVE = {"arith.addi", "arith.muli", "arith.addf", "arith.mulf",
               "arith.maxsi", "arith.minsi", "arith.maxui", "arith.minui",
               "arith.andi", "arith.ori", "arith.xori", "arith.maximumf",
               "arith.minimumf", "arith.maxnumf", "arith.minnumf"}


def base_name(v: str) -> str:
    return v.split("#", 1)[0]


def parse_aliases(text: str) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in ALIAS_DEF_RE.finditer(text)}


def inline_aliases(s: str, aliases: dict[str, str]) -> str:
    if not aliases or "#" not in s:
        return s
    pat = re.compile("|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True))
                     + r"(?![\w$.\-])")
    return pat.sub(lambda m: aliases[m.group(0)], s)


def brace_delta(line: str) -> int:
    d, in_str, esc = 0, False, False
    for c in line:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            d += 1
        elif c == "}":
            d -= 1
    return d


def matching_paren(s: str, i: int) -> int:
    depth = 0
    for j in range(i, len(s)):
        if s[j] == ">" and j > 0 and s[j - 1] == "-":
            continue  # the arrow of `->`
        if s[j] in "([<":
            depth += 1
        elif s[j] in ")]>":
            depth -= 1
            if depth == 0:
                return j
    return -1


def split_top(s: str, sep: str = ",") -> list[str]:
    out, depth, cur = [], 0, []
    prev = ""
    for c in s:
        if c == ">" and prev == "-":
            pass  # the arrow of `->`
        elif c in "([<{":
            depth += 1
        elif c in ")]>}":
            depth -= 1
        prev = c
        if c == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
    if "".join(cur).strip():
        out.append("".join(cur).strip())
    return out


# ---------------------------------------------------------------------------
# Ops / blocks
# ---------------------------------------------------------------------------

@dataclass
class Op:
    idx: int
    name: str
    results: list[str]
    text: str
    start: int
    end: int
    block: int
    func: str
    operands: list[str] = field(default_factory=list)
    uses: set = field(default_factory=set)
    result_types: list[str] = field(default_factory=list)
    regions: list = field(default_factory=list)
    parent: Optional[int] = None
    # linalg.generic specifics
    ins: list[str] = field(default_factory=list)
    outs: list[str] = field(default_factory=list)
    in_types: list[str] = field(default_factory=list)
    out_types: list[str] = field(default_factory=list)
    maps: list[str] = field(default_factory=list)
    iters: list[str] = field(default_factory=list)
    body: list[str] = field(default_factory=list)
    gb: object = None   # kernel_match.GenericBody

    @property
    def header(self) -> str:
        return self.text.split("\n", 1)[0]


@dataclass
class Func:
    name: str
    args: list[tuple[str, str]]
    ops: list[Op]
    start: int
    end: int
    body_start: int
    text: str


def _result_names(first: str) -> tuple[list[str], str]:
    m = re.match(r"\s*(%[\w$.\-]+)(?::(\d+))?\s*=\s*", first)
    if not m:
        return [], first.strip()
    rest = first[m.end():]
    if m.group(2):
        return [f"{m.group(1)}#{i}" for i in range(int(m.group(2)))], rest
    return [m.group(1)], rest


def _defined_within(text: str) -> set[str]:
    d = set()
    for m in re.finditer(r"(%[\w$.\-]+)(?::\d+)?\s*=(?!=)", text):
        d.add(m.group(1))
    for m in re.finditer(r"\^bb\d*\(([^)]*)\)", text):
        for a in re.finditer(r"(%[\w$.\-]+)\s*:", m.group(1)):
            d.add(a.group(1))
    return d


def _result_types(op_text: str, name: str, nres: int) -> list[str]:
    if nres == 0:
        return []
    tail0 = op_text.rsplit(" : ", 1)[-1].strip() if " : " in op_text else ""
    if name in ("memref.collapse_shape", "memref.expand_shape",
                "tensor.collapse_shape", "tensor.expand_shape"):
        # Reshapes spell both types, including layouts: `: source into result`.
        # Only the result describes the new descriptor's rank and strides.
        types = tail0.rsplit(" into ", 1)
        return [types[1].strip()] if len(types) == 2 else []
    if name == "bufferization.to_tensor":
        return [tail0.replace("memref<", "tensor<")]
    if name in ("arith.cmpi", "arith.cmpf"):
        return ["i1"]
    if name in ("tensor.dim", "memref.dim"):
        return ["index"]
    if name == "tensor.extract":
        m = re.match(r"tensor<(?:[\d?]+x)*(\w+)>", tail0)
        return [m.group(1)] if m else []
    last = op_text.rstrip().split("\n")[-1].strip()
    if name == "linalg.generic" or last.startswith("}"):
        m = re.search(r"->\s*(.+)$", last)
        if not m:
            return []
        t = m.group(1).strip()
        if t.startswith("("):
            return split_top(t[1:-1])
        return [t]
    # functional type `: (...) -> T`
    tail = op_text.rsplit(" : ", 1)[-1] if " : " in op_text else ""
    m = re.search(r"\)\s*->\s*(.+)$", tail)
    if m:
        t = m.group(1).strip()
        return split_top(t[1:-1]) if t.startswith("(") else [t]
    m = re.search(r"\bto\s+(\S+)\s*$", tail)
    if m:
        return [m.group(1)]
    if tail:
        parts = split_top(tail)
        return parts[-nres:] if len(parts) >= nres else parts
    return []


def _paren_list(header: str, key: str) -> tuple[list[str], list[str]]:
    i = header.find(key + "(")
    if i < 0:
        return [], []
    j = matching_paren(header, i + len(key))
    inner = header[i + len(key) + 1:j]
    if " : " in inner:
        names, types = inner.split(" : ", 1)
    else:
        names, types = inner, ""
    return ([n.strip() for n in split_top(names) if n.strip()],
            [t.strip() for t in split_top(types) if t.strip()])


class Module:
    """Parsed module: functions with flat op lists (preorder, nested blocks
    tagged by block id)."""

    def __init__(self, text: str):
        self.text = text
        self.aliases = parse_aliases(text)
        self.funcs: list[Func] = []
        self.ops: list[Op] = []
        self._block = 0
        self.block_parent: dict[int, Optional[int]] = {}
        self._parse()
        # Scoped definitions: MLIR reuses SSA names across sibling regions and
        # functions, so a use is resolved by walking up its enclosing blocks.
        self.by_name: dict[tuple[str, str], list[tuple[Op, int]]] = {}
        for op in self.ops:
            for k, r in enumerate(op.results):
                self.by_name.setdefault((op.func, r), []).append((op, k))
                if len(op.results) == 1:
                    self.by_name.setdefault((op.func, base_name(r)), []).append((op, k))
        self.arg_types: dict[str, dict[str, str]] = {
            f.name: dict(f.args) for f in self.funcs}

    # -- parsing ---------------------------------------------------------
    def _lines(self, start: int, end: int) -> list[tuple[int, int, str]]:
        out, pos = [], start
        while pos < end:
            nl = self.text.find("\n", pos, end)
            e = end if nl < 0 else nl + 1
            out.append((pos, e, self.text[pos:e]))
            pos = e
        return out

    def _parse(self) -> None:
        for m in re.finditer(r"^[ \t]*(?:func\.func|kernel\.defn)\s+(?:\w+\s+)?@([\w$.\-]+)\(", self.text, re.MULTILINE):
            name = m.group(1)
            eol = self.text.find("\n", m.start())
            header = self.text[m.start():eol]
            if not header.rstrip().endswith("{"):
                continue  # declaration
            p = header.find("(")
            q = matching_paren(header, p)
            args = []
            for piece in split_top(header[p + 1:q]):
                mm = re.match(r"(%[\w$.\-]+)\s*:\s*(\S+)", piece)
                if mm:
                    args.append((mm.group(1), mm.group(2)))
            depth, pos, end = 0, m.start(), len(self.text)
            for lstart, lend, line in self._lines(m.start(), len(self.text)):
                depth += brace_delta(line)
                if depth == 0:
                    end = lend
                    break
            body_lines = self._lines(eol + 1, end)[:-1]
            f = Func(name, args, [], m.start(), end, eol + 1, self.text[m.start():end])
            self.funcs.append(f)
            self._parse_block(body_lines, f, None)

    def _parse_block(self, lines, f: Func, parent: Optional[int]) -> int:
        bid = self._block
        self._block += 1
        self.block_parent[bid] = parent
        i = 0
        while i < len(lines):
            s, e, line = lines[i]
            st = line.strip()
            if not st or st.startswith("//") or st.startswith("^"):
                i += 1
                continue
            depth = brace_delta(line)
            j = i
            while depth > 0 and j + 1 < len(lines):
                j += 1
                depth += brace_delta(lines[j][2])
            op_text = "".join(l[2] for l in lines[i:j + 1])
            results, rest = _result_names(line)
            nm = re.match(r'([a-z_][\w]*\.[\w.]+|return|"[^"]+")', rest.strip())
            name = nm.group(1).strip('"') if nm else rest.strip().split(" ")[0]
            op = Op(idx=len(self.ops), name=name, results=results, text=op_text,
                    start=s, end=lines[j][1], block=bid, func=f.name, parent=parent)
            self.ops.append(op)
            f.ops.append(op)
            defined = _defined_within(op_text) - set(results)
            hdr = op_text.split("\n", 1)[0] if j > i else op_text
            hdr_rhs = hdr.split("=", 1)[1] if results else hdr
            op.operands = [t for t in SSA_RE.findall(hdr_rhs) if base_name(t) not in defined]
            op.uses = {base_name(t) for t in SSA_RE.findall(op_text)
                       if base_name(t) not in defined and t not in results
                       and base_name(t) not in {base_name(r) for r in results}}
            op.result_types = _result_types(op_text, name, len(results))
            if name == "linalg.generic":
                self._fill_generic(op, hdr)
            if j > i:
                # split region lines on `} ... {` separators at depth 1
                inner = lines[i + 1:j]
                regions, cur, d = [], [], 1
                for ln in inner:
                    dd = brace_delta(ln[2])
                    stp = ln[2].strip()
                    if d == 1 and stp.startswith("}") and stp.endswith("{"):
                        regions.append(cur)
                        cur = []
                    else:
                        cur.append(ln)
                    d += dd
                regions.append(cur)
                for reg in regions:
                    op.regions.append(self._parse_block(reg, f, op.idx))
            i = j + 1
        return bid

    def _fill_generic(self, op: Op, hdr: str) -> None:
        attr = hdr[hdr.find("{"):]
        m = re.search(r"indexing_maps\s*=\s*\[", attr)
        if m:
            k = attr.find("[", m.start())
            j = matching_paren(attr, k)
            op.maps = [inline_aliases(x, self.aliases) for x in split_top(attr[k + 1:j])]
        m = re.search(r"iterator_types\s*=\s*\[([^\]]*)\]", attr)
        if m:
            op.iters = [x.strip().strip('"').replace("#linalg.iterator_type<", "").rstrip(">")
                        for x in m.group(1).split(",")]
        op.ins, op.in_types = _paren_list(hdr, "ins")
        op.outs, op.out_types = _paren_list(hdr, "outs")
        lines = op.text.split("\n")
        op.body = [l.strip() for l in lines[1:-1] if l.strip()]

    # -- queries -----------------------------------------------------------
    def block_chain(self, block: int) -> list[int]:
        chain = [block]
        while True:
            par = self.block_parent.get(chain[-1])
            if par is None:
                return chain
            chain.append(self.ops[par].block)

    def def_of(self, v: str, ctx: Op) -> Optional[tuple[Op, int]]:
        """Definition of SSA value `v` visible from op `ctx` (None for
        function / block arguments)."""
        cands = self.by_name.get((ctx.func, v), [])
        if not cands:
            return None
        chain = self.block_chain(ctx.block)
        best = None
        for op, k in cands:
            if op.block in chain and op.start < ctx.start:
                if best is None or op.start > best[0].start:
                    best = (op, k)
        return best

    def func_of(self, name: str) -> Optional[Func]:
        return next((f for f in self.funcs if f.name == name), None)

    def value_type(self, v: str, ctx: Op) -> Optional[str]:
        d = self.def_of(v, ctx)
        if d:
            op, k = d
            return op.result_types[k] if k < len(op.result_types) else None
        return self.arg_types.get(ctx.func, {}).get(v)

    def skeleton(self, op: Op) -> str:
        """Op text with result names stripped, SSA operands anonymised and
        aliases inlined (used for structural op equality)."""
        t = op.text.split("=", 1)[1] if op.results else op.text
        t = inline_aliases(t, self.aliases)
        t = SSA_RE.sub("%_", t)
        return re.sub(r"\s+", " ", t).strip()


# ---------------------------------------------------------------------------
# Affine maps as integer polynomials
# ---------------------------------------------------------------------------
# A polynomial is a dict {monomial: int coeff}; a monomial is a sorted tuple of
# atoms; atoms are ('d', i) for dims, ('v', ssa) for symbol operands, or
# ('x', text) for opaque sub-expressions (floordiv/mod/ceildiv).

def p_const(c: int) -> dict:
    return {(): c} if c else {}


def p_add(a: dict, b: dict, s: int = 1) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + s * v
        if out[k] == 0:
            del out[k]
    return out


def p_mul(a: dict, b: dict) -> dict:
    out: dict = {}
    for ka, va in a.items():
        for kb, vb in b.items():
            k = tuple(sorted(ka + kb))
            out[k] = out.get(k, 0) + va * vb
            if out[k] == 0:
                del out[k]
    return out


class AffineParseError(Exception):
    pass


def parse_affine_map(s: str) -> tuple[int, int, list[str]]:
    m = re.match(r"\s*affine_map<\(([^)]*)\)(?:\[([^\]]*)\])?\s*->\s*\((.*)\)>\s*$", s)
    if not m:
        raise AffineParseError(s)
    dims = [d.strip() for d in m.group(1).split(",") if d.strip()]
    syms = [d.strip() for d in (m.group(2) or "").split(",") if d.strip()]
    return len(dims), len(syms), split_top(m.group(3))


def expr_to_poly(e: str, dim_polys: list[dict], sym_polys: list[dict]) -> dict:
    toks = re.findall(r"floordiv|ceildiv|mod|d\d+|s\d+|\d+|[()+\-*]", e)
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else None

    def eat():
        nonlocal pos
        pos += 1
        return toks[pos - 1]

    def atom():
        t = eat()
        if t == "(":
            v = add()
            eat()
            return v
        if t == "-":
            return p_mul(p_const(-1), atom())
        if t.startswith("d"):
            return dict(dim_polys[int(t[1:])])
        if t.startswith("s"):
            return dict(sym_polys[int(t[1:])])
        if t.isdigit():
            return p_const(int(t))
        raise AffineParseError(e)

    def mul():
        v = atom()
        while peek() in ("*", "floordiv", "ceildiv", "mod"):
            op = eat()
            r = atom()
            if op == "*":
                v = p_mul(v, r)
            else:
                v = {(("x", f"({_ptext(v)}) {op} ({_ptext(r)})"),): 1}
        return v

    def add():
        v = mul()
        while peek() in ("+", "-"):
            op = eat()
            r = mul()
            v = p_add(v, r, 1 if op == "+" else -1)
        return v

    out = add()
    if pos != len(toks):
        raise AffineParseError(e)
    return out


def _ptext(p: dict) -> str:
    return " + ".join(f"{c}*{'*'.join(str(a) for a in k) or '1'}" for k, c in sorted(p.items(), key=str))


def map_apply(map_text: str, dim_polys: list[dict], sym_polys: list[dict]) -> list[dict]:
    nd, ns, res = parse_affine_map(map_text)
    if nd != len(dim_polys) or ns != len(sym_polys):
        raise AffineParseError(f"arity mismatch {map_text}")
    return [expr_to_poly(r, dim_polys, sym_polys) for r in res]


def identity_dims(n: int) -> list[dict]:
    return [{(("d", i),): 1} for i in range(n)]


def rename_dims(p: dict, sigma: list[int]) -> dict:
    out = {}
    for k, v in p.items():
        nk = tuple(sorted((("d", sigma[a[1]]) if a[0] == "d" else a) for a in k))
        out[nk] = out.get(nk, 0) + v
    return {k: v for k, v in out.items() if v}


def pure_dim(p: dict) -> Optional[int]:
    if len(p) == 1:
        (k, v), = p.items()
        if v == 1 and len(k) == 1 and k[0][0] == "d":
            return k[0][1]
    return None
