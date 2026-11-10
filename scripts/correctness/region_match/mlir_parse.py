"""Minimal line-oriented parser for the MLIR subset emitted by cgeist /
polygeist-opt (affine, arith, math, tensor, bufferization, memref, linalg,
polygeist dialects).

The parser builds an op tree with character spans so that a matched region
can be replaced textually.  Op semantics are interpreted elsewhere
(symexec.py); this module only knows the textual structure.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .affine import Aff, NonAffine


# ---------------------------------------------------------------------------
# Op tree
# ---------------------------------------------------------------------------

@dataclass
class Op:
    name: str
    results: list[str]
    line: str                       # header line text (stripped)
    start: int                      # char offset of line start
    end: int                        # char offset after the op (incl. region)
    regions: list[list["Op"]] = field(default_factory=list)
    block_args: list[list[tuple[str, str]]] = field(default_factory=list)
    tail: str = ""                  # text of the closing line of the last region
    else_region: Optional[list["Op"]] = None

    def __repr__(self):
        return f"Op({self.name}, {self.results})"


@dataclass
class Func:
    name: str
    args: list[tuple[str, str]]     # (ssa, type)
    body: list[Op]
    start: int
    end: int
    body_start: int


_RES_RE = re.compile(r"^(%[\w$.#-]+)(?::(\d+))?\s*=\s*")
_OPNAME_RE = re.compile(r"^\"?([a-z_][\w]*\.[\w.]+|return)\"?")


def _split_results(line: str) -> tuple[list[str], str]:
    m = _RES_RE.match(line)
    if not m:
        # multiple comma separated results "%a, %b = ..."
        m2 = re.match(r"^((?:%[\w$.-]+\s*,\s*)+%[\w$.-]+)\s*=\s*", line)
        if m2:
            names = [x.strip() for x in m2.group(1).split(",")]
            return names, line[m2.end():]
        return [], line
    name, count = m.group(1), m.group(2)
    rest = line[m.end():]
    if count:
        return [f"{name}#{i}" for i in range(int(count))], rest
    return [name], rest


def parse_module(text: str) -> tuple[list[Func], dict[str, str]]:
    """Return functions and the map/set alias table (#map -> 'affine_map<...>')."""
    aliases: dict[str, str] = {}
    for m in re.finditer(r"^(#[\w$.-]+)\s*=\s*(affine_(?:map|set)<.*>)\s*$", text, re.M):
        aliases[m.group(1)] = m.group(2)
    lines = []
    pos = 0
    for raw in text.splitlines(keepends=True):
        lines.append((pos, raw))
        pos += len(raw)
    funcs: list[Func] = []
    i = 0
    while i < len(lines):
        off, raw = lines[i]
        s = raw.strip()
        if s.startswith("func.func") and s.endswith("{"):
            m = re.match(r"func\.func\s+(?:private\s+|public\s+)?@([\w$.-]+)\((.*?)\)\s*(?:->.*?)?(?:attributes\s*\{.*\})?\s*\{$", s)
            if not m:
                i += 1
                continue
            args = []
            for a in _split_top(m.group(2)):
                a = a.strip()
                if not a:
                    continue
                am = re.match(r"(%[\w$.-]+)\s*:\s*([^{]+?)(\s*\{.*\})?$", a)
                if am:
                    args.append((am.group(1), am.group(2).strip()))
            body, j, _tail = _parse_block(lines, i + 1)
            end = lines[j][0] + len(lines[j][1]) if j < len(lines) else len(text)
            funcs.append(Func(m.group(1), args, body, off, end, lines[i + 1][0] if i + 1 < len(lines) else end))
            i = j + 1
            continue
        i += 1
    return funcs, aliases


def _parse_block(lines, i):
    """Parse ops until a closing '}' line.  Returns (ops, index_of_closing_line, closing_text)."""
    ops: list[Op] = []
    while i < len(lines):
        off, raw = lines[i]
        s = raw.strip()
        if not s or s.startswith("//"):
            i += 1
            continue
        if s.startswith("}"):
            return ops, i, s
        if s.startswith("^"):
            i += 1
            continue
        results, rest = _split_results(s)
        nm = _OPNAME_RE.match(rest)
        name = nm.group(1) if nm else rest.split()[0]
        op = Op(name, results, s, off, off + len(raw))
        if s.endswith("{"):
            # region-bearing op
            j = i + 1
            # optional block header
            bargs: list[tuple[str, str]] = []
            if j < len(lines) and lines[j][1].strip().startswith("^"):
                bargs = _parse_block_args(lines[j][1].strip())
                j += 1
            body, k, close = _parse_block(lines, j)
            op.regions.append(body)
            op.block_args.append(bargs)
            while close.startswith("}") and close.rstrip().endswith("{"):
                # '} else {'
                j = k + 1
                bargs = []
                if j < len(lines) and lines[j][1].strip().startswith("^"):
                    bargs = _parse_block_args(lines[j][1].strip())
                    j += 1
                body, k, close = _parse_block(lines, j)
                op.regions.append(body)
                op.block_args.append(bargs)
            op.tail = close
            op.end = lines[k][0] + len(lines[k][1])
            ops.append(op)
            i = k + 1
            continue
        ops.append(op)
        i += 1
    return ops, i, ""


def _parse_block_args(s: str) -> list[tuple[str, str]]:
    m = re.match(r"\^[\w]+\((.*)\)\s*:\s*$", s)
    if not m:
        return []
    out = []
    for a in _split_top(m.group(1)):
        a = a.strip()
        am = re.match(r"(%[\w$.-]+)\s*:\s*(.+)$", a)
        if am:
            out.append((am.group(1), am.group(2).strip()))
    return out


def _split_top(s: str, sep: str = ",") -> list[str]:
    """Split on `sep` at bracket depth 0 (handles <>, (), [], {})."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "<([{":
            depth += 1
        elif ch in ">)]}":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return out


# ---------------------------------------------------------------------------
# Affine expression / map parsing
# ---------------------------------------------------------------------------

_TOK_RE = re.compile(r"\s*(symbol\(|%[\w$.#-]+|[a-zA-Z_][\w]*|\d+|[-+*()])")


def parse_affine_expr(s: str, env: dict[str, Aff]) -> Aff:
    """Parse an affine expression whose identifiers are looked up in env
    (dims d0.., symbols s0.., or SSA names %x)."""
    toks = []
    pos = 0
    s = s.strip()
    while pos < len(s):
        m = _TOK_RE.match(s, pos)
        if not m:
            raise NonAffine(f"cannot tokenize affine expr {s!r}")
        toks.append(m.group(1))
        pos = m.end()
    toks.append(None)
    p = [0]

    def peek():
        return toks[p[0]]

    def take():
        t = toks[p[0]]
        p[0] += 1
        return t

    def atom() -> Aff:
        t = take()
        if t == "(":
            v = expr()
            if take() != ")":
                raise NonAffine("paren")
            return v
        if t == "symbol(":
            v = expr()
            if take() != ")":
                raise NonAffine("symbol paren")
            return v
        if t == "-":
            return -atom()
        if t is not None and t.isdigit():
            return Aff.cst(int(t))
        if t in ("floordiv", "ceildiv", "mod"):
            raise NonAffine(t)
        if t is not None and t in env:
            return env[t]
        raise NonAffine(f"unknown identifier {t!r} in {s!r}")

    def term() -> Aff:
        v = atom()
        while peek() in ("*",) or peek() in ("floordiv", "ceildiv", "mod"):
            op = take()
            if op != "*":
                raise NonAffine(op)
            v = v * atom()
        return v

    def expr() -> Aff:
        v = term()
        while peek() in ("+", "-"):
            op = take()
            r = term()
            v = v + r if op == "+" else v - r
        return v

    out = expr()
    if peek() is not None:
        raise NonAffine(f"trailing tokens in {s!r}")
    return out


@dataclass
class AffineMap:
    dims: list[str]
    syms: list[str]
    results: list[str]   # textual result expressions

    def apply(self, dim_vals: list[Aff], sym_vals: list[Aff]) -> list[Aff]:
        env = {}
        for n, v in zip(self.dims, dim_vals):
            env[n] = v
        for n, v in zip(self.syms, sym_vals):
            env[n] = v
        return [parse_affine_expr(r, env) for r in self.results if r.strip()]


def parse_affine_map(s: str, aliases: dict[str, str]) -> AffineMap:
    s = s.strip()
    if s.startswith("#"):
        s = aliases[s]
    m = re.match(r"affine_map<\(([^)]*)\)(?:\[([^\]]*)\])?\s*->\s*\((.*)\)>$", s)
    if not m:
        raise NonAffine(f"bad affine map {s}")
    dims = [d.strip() for d in m.group(1).split(",") if d.strip()]
    syms = [d.strip() for d in (m.group(2) or "").split(",") if d.strip()]
    res = _split_top(m.group(3)) if m.group(3).strip() else []
    return AffineMap(dims, syms, res)


def parse_affine_set(s: str, aliases: dict[str, str]):
    s = s.strip()
    if s.startswith("#"):
        s = aliases[s]
    m = re.match(r"affine_set<\(([^)]*)\)(?:\[([^\]]*)\])?\s*:\s*\((.*)\)>$", s)
    if not m:
        raise NonAffine(f"bad affine set {s}")
    dims = [d.strip() for d in m.group(1).split(",") if d.strip()]
    syms = [d.strip() for d in (m.group(2) or "").split(",") if d.strip()]
    conds = _split_top(m.group(3))
    return dims, syms, conds
