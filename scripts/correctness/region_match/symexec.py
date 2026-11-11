"""Symbolic execution of cgeist / polygeist-opt functions into a memory-level
statement program.

Tensor SSA values are *re-bufferized*: every tensor value is a version of an
abstract buffer (a function memref argument or a scratch allocation), and
tensor ops are interpreted as in-place memory operations on that buffer.  The
interpretation is checked to be linear (every tensor read refers to the
current version of its buffer); otherwise execution is rejected.  linalg.generic
ops are executed with memref semantics (lexicographic iteration over the
iteration dims), which is what convert-linalg-to-loops and
--lower-polygeist-submap implement for aliasing views.

The result is a list of statements, each a single affine write
    buf[idx(J)] := value(J)       for J in domain, executed at time(J)
where `value` is a scalar expression tree over timed affine reads.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .affine import Aff, Cons, NonAffine, ge, le, lt, eq, negate, atom, is_atom, atoms_in
from .mlir_parse import Func, Op, parse_affine_map, parse_affine_set, parse_affine_expr, _split_top


class Unsupported(Exception):
    pass


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Buffer:
    id: int
    name: str            # memref SSA name for function arguments, "scratch<k>" otherwise
    kind: str            # 'arg' | 'scratch'
    rank: int            # logical rank (without privatisation dims)
    elem: str
    priv: list[str]      # enclosing loop ivs at allocation (privatisation)
    shape: list[Optional[Aff]]
    arg_type: str = ""
    version: int = 0
    written_back: bool = False
    restrict: bool = False

    def __repr__(self):
        return self.name


@dataclass
class View:
    index: list[Aff]               # buffer coords in terms of '@v<k>' placeholders
    sizes: list[Optional[Aff]]


@dataclass
class TensorVal:
    buf: Buffer
    ver: int
    view: Optional[View]
    origin: Optional[tuple] = None   # ('generic', stmt ids) for generic results


@dataclass
class MemVal:
    buf: Buffer
    view: Optional[View]


@dataclass
class ToMem:
    tv: TensorVal


@dataclass
class IdxVal:
    aff: Aff


@dataclass
class CondAff:
    cons: list[Cons]


@dataclass
class CondF:
    kind: str
    a: "Expr"
    b: "Expr"


@dataclass
class FVal:
    e: "Expr"


class Undef:
    pass


# ---------------------------------------------------------------------------
# Scalar expressions
# ---------------------------------------------------------------------------

class Expr:
    pass


@dataclass(eq=False)
class ERead(Expr):
    buf: Buffer
    idx: tuple            # tuple[Aff] (buffer coords incl. priv dims)
    time: tuple           # tuple[Aff]
    guard: tuple = ()     # constraints under which the read is live
    vars: tuple = ()      # iteration vars in scope at the read


@dataclass(eq=False)
class EConst(Expr):
    v: float


@dataclass(eq=False)
class EParam(Expr):
    name: str


@dataclass(eq=False)
class EIndex(Expr):
    """Integer index data, retained beneath its explicit numeric conversion."""
    value: Aff


@dataclass(eq=False)
class EBin(Expr):
    op: str
    a: Expr
    b: Expr


@dataclass(eq=False)
class EUn(Expr):
    op: str
    a: Expr


@dataclass(eq=False)
class ESel(Expr):
    cond: object          # CondAff | CondF
    a: Expr
    b: Expr


@dataclass(eq=False)
class EUndef(Expr):
    pass


# ---------------------------------------------------------------------------
# Statements
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Stmt:
    id: int
    vars: list[str]
    domain: list[Cons]
    guard: list[Cons]
    time: list[Aff]
    buf: Buffer
    idx: list[Aff]
    value: Expr
    par_dims: set = field(default_factory=set)
    top: int = -1                  # index of the top-level op that produced it
    origin: str = ""

    def __repr__(self):
        return f"S{self.id}[{','.join(self.vars)}]: {self.buf}{self.idx} := ..."


@dataclass
class Program:
    func: Func
    stmts: list[Stmt]
    buffers: list[Buffer]
    params: dict[str, str]                 # param var -> func arg ssa
    float_params: dict[str, str]
    top_ops: list[Op]
    top_kinds: list[str]                   # 'work' | 'pure' | 'writeback' | 'return'
    writebacks: dict[int, list[int]]       # buffer id -> top op indices of its write-back
    warnings: list[str]
    stored: dict[int, tuple]               # id(expr) -> (stmt, ...)
    arg_buffers: dict[str, Buffer]
    # memory-sync state before every top-level op (index len(top_ops) = after
    # the last op): (synced, current full tensor value, versions), see
    # Executor.snapshot
    snaps: list = field(default_factory=list)
    tensor_writes: dict = field(default_factory=dict)   # buffer id -> top ops with tensor-level writes
    env: dict = field(default_factory=dict)             # SSA name -> final abstract value


_FLOAT_TYPES = ("f64", "f32", "f16", "bf16")
_INT_TYPES = ("i64", "i32", "i16", "i8", "index")


def _elem_of(ty: str) -> str:
    m = re.match(r"\s*(?:memref|tensor)<(.*)>\s*$", ty)
    if m:                         # first top-level component (drop any layout)
        ty = _split_top(m.group(1))[0].strip() + ">"
    m = re.search(r"x?(f64|f32|f16|bf16|i64|i32|i8|i1|index)>", ty)
    return m.group(1) if m else ty


def _rank_of(ty: str) -> int:
    m = re.match(r"(?:memref|tensor)<(.*)>", ty.strip())
    if not m:
        return 0
    body = _split_top(m.group(1))[0]
    parts = body.split("x")
    return len(parts) - 1


def _static_shape(ty: str) -> list[Optional[Aff]]:
    m = re.match(r"(?:memref|tensor)<(.*)>", ty.strip())
    if not m:
        return []
    body = _split_top(m.group(1))[0]
    parts = body.split("x")[:-1]
    out = []
    for p in parts:
        p = p.strip()
        out.append(Aff.cst(int(p)) if p.isdigit() else None)
    return out


def _subst_view(view: Optional[View], idx: list[Aff]) -> list[Aff]:
    if view is None:
        return list(idx)
    m = {f"@v{k}": a for k, a in enumerate(idx)}
    return [e.subst(m) for e in view.index]


class Executor:
    def __init__(self, func: Func, aliases: dict[str, str], fixed_ints: Optional[dict] = None,
                 launch_outputs=None):
        self.func = func
        self.aliases = aliases
        # int arguments specialised to a constant (spec stride parameters)
        self.fixed_ints: dict[str, int] = dict(fixed_ints or {})
        # spec name -> argument positions written by a kernel.launch of it
        self.launch_outputs = launch_outputs
        # Memory-sync tracking for function-argument buffers.  `synced` is
        # True while the memref argument holds the buffer's current (tensor)
        # version; tensor-level writes clear it, write-backs set it.
        self.synced: dict[int, bool] = {}
        self.cur_full: dict[int, str] = {}       # buffer id -> SSA of its current full tensor
        self.snaps: list = []
        self.tensor_writes: dict[int, list[int]] = {}
        self.env: dict[str, object] = {}
        self.stmts: list[Stmt] = []
        self.buffers: list[Buffer] = []
        self.params: dict[str, str] = {}
        self.float_params: dict[str, str] = {}
        self.loop_vars: list[str] = []
        self.domain: list[Cons] = []
        self.loop_count = 0
        self.gen_count = 0
        self.scratch_count = 0
        self.cur_top = -1
        self.warnings: list[str] = []
        self.stored: dict[int, tuple] = {}
        self.arg_buffers: dict[str, Buffer] = {}
        self.read_time: Optional[list[Aff]] = None
        self.read_guard: tuple = ()
        self.writeback_tops: dict[int, list[int]] = {}
        # index atom -> the read it stands for (used when the integer is
        # consumed as data rather than as an index, e.g. a histogram count)
        self.atom_reads: dict[str, Optional[ERead]] = {}

    # -- helpers --------------------------------------------------------
    def new_buffer(self, name, kind, ty, priv=None, shape=None) -> Buffer:
        b = Buffer(len(self.buffers), name, kind, _rank_of(ty), _elem_of(ty),
                   list(priv or []), shape if shape is not None else _static_shape(ty), ty)
        self.buffers.append(b)
        return b

    def snapshot(self):
        return (dict(self.synced), dict(self.cur_full), {b.id: b.version for b in self.buffers})

    def mark_write(self, buf: Buffer, via_tensor: bool):
        if buf.kind != "arg":
            return
        if via_tensor:
            self.synced[buf.id] = False
            self.tensor_writes.setdefault(buf.id, []).append(self.cur_top)

    def get(self, name: str):
        name = name.strip()
        if name not in self.env:
            raise Unsupported(f"unknown SSA value {name}")
        return self.env[name]

    def idx(self, tok: str) -> Aff:
        tok = tok.strip()
        if re.fullmatch(r"-?\d+", tok):
            return Aff.cst(int(tok))
        v = self.get(tok)
        if isinstance(v, IdxVal):
            return v.aff
        raise Unsupported(f"{tok} is not an affine index ({type(v).__name__})")

    def fval(self, tok: str) -> Expr:
        v = self.get(tok)
        if isinstance(v, FVal):
            return v.e
        if isinstance(v, Undef):
            return EUndef()
        if isinstance(v, IdxVal):
            return self.int_as_data(v.aff, tok)
        raise Unsupported(f"{tok} is not a float value")

    def int_as_data(self, a: Aff, tok: str = "") -> Expr:
        """An integer used as a value: constants and loaded integers (atoms,
        through the read they stand for) combine with exact arithmetic."""
        out: Optional[Expr] = EConst(float(a.const)) if a.const or a.is_const() else None
        for name, c in a.terms:
            r = self.atom_reads.get(name)
            if r is None:
                raise Unsupported(f"{tok} is an index, not a value")
            term: Expr = r if c == 1 else EBin("mul", EConst(float(c)), r)
            out = term if out is None else EBin("add", out, term)
        return out

    def access(self, val, idx: list[Aff]) -> tuple[Buffer, tuple]:
        if isinstance(val, (TensorVal, MemVal)):
            buf = val.buf
            coords = _subst_view(val.view, idx)
        else:
            raise Unsupported(f"access through {type(val).__name__}")
        full = [Aff.var(p) for p in buf.priv] + list(coords)
        return buf, tuple(full)

    def check_current(self, tv):
        if isinstance(tv, TensorVal) and tv.ver != tv.buf.version:
            raise Unsupported(f"non-linear use of a stale tensor version of {tv.buf}")

    def emit(self, buf, idx, value: Expr, time: list[Aff], extra_vars=(), extra_dom=(),
             guard=(), par=(), origin=""):
        st = Stmt(len(self.stmts), list(self.loop_vars) + list(extra_vars),
                  list(self.domain) + list(extra_dom), list(guard), list(time), buf,
                  list(idx), value, set(par), self.cur_top, origin)
        self.stmts.append(st)
        self.stored.setdefault(id(value), (st,))
        return st

    def read(self, val, idx, time, extra_vars=()) -> ERead:
        buf, coords = self.access(val, idx)
        return ERead(buf, coords, tuple(time), tuple(self.read_guard),
                     tuple(self.loop_vars) + tuple(extra_vars))

    def loaded(self, r: ERead):
        """Value of a load: an integer element of a function argument is an
        index atom (usable in bounds, gathers and scatters; equations.py
        rejects it if the region writes that array), anything else a value."""
        if r.buf.kind == "arg" and r.buf.elem in _INT_TYPES and not r.buf.priv:
            a = atom(r.buf.name, r.idx)
            (name, _), = a.terms
            prev = self.atom_reads.get(name, r)
            # the same element loaded at two different times (around a store)
            # has no single value: it can no longer be used as data
            self.atom_reads[name] = r if prev is r or (prev is not None and prev.time == r.time) else None
            return IdxVal(a)
        return FVal(r)

    # -- driver ---------------------------------------------------------
    def run(self) -> Program:
        for name, ty in self.func.args:
            ty = ty.strip()
            if ty.startswith("memref"):
                shape = _static_shape(ty)
                for k, d in enumerate(shape):
                    if d is None:
                        pv = f"pdim{name}_{k}"
                        self.params[pv] = f"dim({name},{k})"
                        shape[k] = Aff.var(pv)
                b = self.new_buffer(name, "arg", ty, shape=shape)
                self.arg_buffers[name] = b
                self.synced[b.id] = True
                self.env[name] = MemVal(b, None)
            elif ty in ("i32", "i64", "index", "i16", "i8"):
                if name in self.fixed_ints:
                    self.env[name] = IdxVal(Aff.cst(int(self.fixed_ints[name])))
                    continue
                pv = f"p{name}"
                self.params[pv] = name
                self.env[name] = IdxVal(Aff.var(pv))
            elif ty in _FLOAT_TYPES:
                self.float_params[name] = name
                self.env[name] = FVal(EParam(name))
            else:
                self.env[name] = Undef()
        kinds = []
        for k, op in enumerate(self.func.body):
            self.cur_top = k
            self.snaps.append(self.snapshot())
            before = len(self.stmts)
            kind = self.exec_op(op, [], k * 4)
            if kind is None:
                kind = "work" if len(self.stmts) > before else "pure"
            kinds.append(kind)
            for r in op.results:
                v = self.env.get(r)
                if isinstance(v, TensorVal) and v.view is None and v.buf.kind == "arg" \
                        and v.ver == v.buf.version:
                    self.cur_full[v.buf.id] = r
        self.snaps.append(self.snapshot())
        return Program(self.func, self.stmts, self.buffers, self.params, self.float_params,
                       self.func.body, kinds, self.writeback_tops, self.warnings,
                       self.stored, self.arg_buffers, self.snaps, self.tensor_writes, self.env)

    def exec_block(self, ops: list[Op], prefix: list[Aff]):
        pos = 1
        for op in ops:
            self.exec_op(op, prefix, pos)
            pos += 4

    # -- op dispatch ----------------------------------------------------
    def exec_op(self, op: Op, prefix: list[Aff], pos: int):
        n = op.name
        line = op.line
        rest = line.split("=", 1)[1].strip() if op.results else line
        T = lambda k: prefix + [Aff.cst(pos + k)]
        res = op.results

        if n == "arith.constant":
            m = re.match(r"arith\.constant\s+(\S+)\s*:\s*(\S+)", rest)
            if not m:
                # e.g. dense<...>
                self.env[res[0]] = Undef()
                return
            v, ty = m.group(1), m.group(2)
            if ty in _FLOAT_TYPES:
                self.env[res[0]] = FVal(EConst(float(v)))
            elif v in ("true", "false"):
                self.env[res[0]] = CondAff([] if v == "true" else [Cons(Aff.cst(-1))])
            else:
                self.env[res[0]] = IdxVal(Aff.cst(int(v)))
            return
        if n in ("arith.index_cast", "arith.index_castui", "arith.extsi", "arith.extui", "arith.trunci"):
            m = re.match(r"\S+\s+(%[\w$.#-]+)\s*:\s*(\S+)\s+to\s+(\S+)", rest)
            if not m:
                raise Unsupported('unparsed integer conversion')
            v = self.get(m.group(1))
            if isinstance(v, Undef):
                # Metadata used only by an existing launch need not be affine.
                # Keep it unknown: using it in a loop bound, index, or element
                # equation will still reject that proof at the point of use.
                self.env[res[0]] = Undef()
            elif isinstance(v, FVal):
                self.env[res[0]] = FVal(EUn(n + ':' + m.group(2) + ' to ' + m.group(3), v.e))
            elif isinstance(v, (CondAff, CondF)) and m.group(2) == 'i1' and n in ('arith.extui', 'arith.extsi'):
                # Boolean data can occur in an unrelated region of the same
                # function. Keep the conversion and condition explicit; i1
                # true has the bit pattern 1, interpreted by the typed cast.
                value = ESel(v, EIndex(Aff.cst(1)), EIndex(Aff.cst(0)))
                self.env[res[0]] = FVal(EUn(n + ':i1 to ' + m.group(3), value))
            elif isinstance(v, IdxVal):
                from .affine import implies
                src, dst = m.group(2), m.group(3)
                source_bits = 64 if src == 'index' else int(src[1:])
                target_bits = 64 if dst == 'index' else int(dst[1:])
                unsigned = n in ('arith.extui', 'arith.index_castui')
                if unsigned and not implies(self.domain, ge(v.aff, 0)):
                    raise Unsupported('unsigned index conversion may change a negative value')
                if target_bits < source_bits:
                    low = 0 if unsigned else -(1 << (target_bits - 1))
                    high = (1 << (target_bits if unsigned else target_bits - 1)) - 1
                    if not (implies(self.domain, ge(v.aff, low)) and
                            implies(self.domain, le(v.aff, high))):
                        raise Unsupported('narrowing index conversion may truncate')
                self.env[res[0]] = v
            else:
                raise Unsupported('unsupported integer conversion operand')
            return
        if n in ("arith.addi", "arith.subi", "arith.muli"):
            m = re.match(r"\S+\s+(%[\w$.#-]+),\s*(%[\w$.#-]+)", rest)
            a, b = self.get(m.group(1)), self.get(m.group(2))
            if isinstance(a, FVal) or isinstance(b, FVal):
                # integer data (e.g. a count read back from memory)
                try:
                    opn = n + ':' + rest.rsplit(':', 1)[1].strip()
                    self.env[res[0]] = FVal(EBin(opn, self.fval(m.group(1)), self.fval(m.group(2))))
                except Unsupported:
                    self.env[res[0]] = Undef()
                return
            if not (isinstance(a, IdxVal) and isinstance(b, IdxVal)):
                self.env[res[0]] = Undef()
                return
            try:
                if n == "arith.addi":
                    r = a.aff + b.aff
                elif n == "arith.subi":
                    r = a.aff - b.aff
                else:
                    r = a.aff * b.aff
                self.env[res[0]] = IdxVal(r)
            except NonAffine:
                self.env[res[0]] = Undef()
            return
        if n in ("arith.divsi", "arith.divui", "arith.floordivsi", "arith.ceildivsi"):
            # exact only: the divisor is a constant that divides every
            # coefficient (e.g. a stride specialised to 1)
            m = re.match(r"\S+\s+(%[\w$.#-]+),\s*(%[\w$.#-]+)", rest)
            a, b = self.get(m.group(1)), self.get(m.group(2))
            if isinstance(a, IdxVal) and isinstance(b, IdxVal) and b.aff.is_const() \
                    and b.aff.const > 0 and all(c % b.aff.const == 0 for _, c in a.aff.terms) \
                    and a.aff.const % b.aff.const == 0:
                d = b.aff.const
                self.env[res[0]] = IdxVal(Aff({k: c // d for k, c in a.aff.terms}, a.aff.const // d))
            else:
                self.env[res[0]] = Undef()
            return
        if n in ("arith.remsi", "arith.remui", "arith.maxsi", "arith.minsi", "affine.min",
                 "affine.max"):
            self.env[res[0]] = Undef()
            return
        if n in ("tensor.dim", "memref.dim"):
            m = re.match(r"\S+\s+(%[\w$.#-]+),\s*(%[\w$.#-]+|\d+)", rest)
            v = self.get(m.group(1))
            k = self.idx(m.group(2))
            size = None
            if isinstance(v, (TensorVal, MemVal)) and k.is_const():
                sizes = v.view.sizes if v.view is not None else v.buf.shape
                if 0 <= k.const < len(sizes):
                    size = sizes[k.const]
            self.env[res[0]] = IdxVal(size) if size is not None else Undef()
            return "pure"
        if n == "memref.extract_strided_metadata":
            for r in res:
                self.env[r] = Undef()
            return "pure"
        if n == "kernel.launch":
            return self._launch(op, rest)
        if n == "affine.apply":
            m = re.match(r"affine\.apply\s+(#[\w$.-]+|affine_map<.*?>)\s*\(([^)]*)\)(?:\[([^\]]*)\])?", rest)
            amap = parse_affine_map(m.group(1), self.aliases)
            dims = [self.idx(x) for x in m.group(2).split(",") if x.strip()]
            syms = [self.idx(x) for x in (m.group(3) or "").split(",") if x.strip()]
            self.env[res[0]] = IdxVal(amap.apply(dims, syms)[0])
            return
        if n in ("arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.maximumf",
                 "arith.minimumf", "arith.maxnumf", "arith.minnumf", "math.powf"):
            m = re.match(r"\S+\s+(%[\w$.#-]+),\s*(%[\w$.#-]+)", rest)
            opn = n.split(".")[1]
            opn = {"addf": "add", "subf": "sub", "mulf": "mul", "divf": "div"}.get(opn, opn)
            self.env[res[0]] = FVal(EBin(opn, self.fval(m.group(1)), self.fval(m.group(2))))
            return
        if n in ("arith.negf", "math.sqrt", "math.exp", "math.log", "math.absf", "math.tanh",
                 "math.rsqrt", "math.sin", "math.cos", "math.log1p", "math.expm1"):
            m = re.match(r"\S+\s+(%[\w$.#-]+)", rest)
            self.env[res[0]] = FVal(EUn(n.split(".")[1], self.fval(m.group(1))))
            return
        if n in ("arith.sitofp", "arith.uitofp", "arith.extf", "arith.truncf",
                 "arith.fptosi", "arith.fptoui", "arith.bitcast"):
            m = re.match(r"\S+\s+(%[\w$.#-]+)\s*:\s*(\S+)\s+to\s+(\S+)", rest)
            if not m:
                raise Unsupported("unparsed numeric conversion")
            v = self.get(m.group(1))
            if isinstance(v, FVal):
                value = v.e
            elif isinstance(v, IdxVal):
                # Parameters used in arithmetic (e.g. a dimension converted
                # to float) remain integer expressions under the typed cast.
                # Loaded integers retain their existing memory-read identity.
                if n in ("arith.sitofp", "arith.uitofp") and not atoms_in(v.aff.vars()):
                    value = EIndex(v.aff)
                else:
                    value = self.int_as_data(v.aff, m.group(1))
            else:
                raise Unsupported("unsupported numeric conversion operand")
            self.env[res[0]] = FVal(EUn(n + ":" + m.group(2) + " to " + m.group(3), value))
            return
        if n == "arith.cmpi":
            m = re.match(r"arith\.cmpi\s+(\w+),\s*(%[\w$.#-]+),\s*(%[\w$.#-]+)", rest)
            kind, a, b = m.group(1), self.get(m.group(2)), self.get(m.group(3))
            if not (isinstance(a, IdxVal) and isinstance(b, IdxVal)):
                self.env[res[0]] = Undef()
                return
            a, b = a.aff, b.aff
            table = {"slt": [lt(a, b)], "ult": [lt(a, b)], "sle": [le(a, b)], "ule": [le(a, b)],
                     "sgt": [lt(b, a)], "ugt": [lt(b, a)], "sge": [le(b, a)], "uge": [le(b, a)],
                     "eq": [eq(a, b)]}
            if kind not in table:
                self.env[res[0]] = Undef()
                return
            self.env[res[0]] = CondAff(table[kind])
            return
        if n == "arith.andi":
            m = re.match(r"\S+\s+(%[\w$.#-]+),\s*(%[\w$.#-]+)", rest)
            a, b = self.get(m.group(1)), self.get(m.group(2))
            if isinstance(a, CondAff) and isinstance(b, CondAff):
                self.env[res[0]] = CondAff(a.cons + b.cons)
            elif isinstance(a, IdxVal) and isinstance(b, IdxVal):
                self.env[res[0]] = Undef()
            else:
                self.env[res[0]] = Undef()
            return
        if n == "arith.cmpf":
            m = re.match(r"arith\.cmpf\s+(\w+),\s*(%[\w$.#-]+),\s*(%[\w$.#-]+)", rest)
            self.env[res[0]] = CondF(m.group(1), self.fval(m.group(2)), self.fval(m.group(3)))
            return
        if n == "arith.select":
            m = re.match(r"arith\.select\s+(%[\w$.#-]+),\s*(%[\w$.#-]+),\s*(%[\w$.#-]+)\s*:\s*(\S+)", rest)
            c = self.get(m.group(1))
            a, b = self.get(m.group(2)), self.get(m.group(3))
            if isinstance(a, FVal) and isinstance(b, FVal) and isinstance(c, (CondAff, CondF)):
                self.env[res[0]] = FVal(ESel(c, a.e, b.e))
            else:
                self.env[res[0]] = Undef()
            return
        if n == "llvm.mlir.undef":
            self.env[res[0]] = Undef()
            return
        if n in ("memref.alloca", "memref.alloc"):
            m = re.search(r"\(([^)]*)\)\s*(?:\{[^}]*\})?\s*:\s*(memref<.*>)", rest)
            ty = m.group(2)
            dyn = [self.idx(x) for x in m.group(1).split(",") if x.strip()]
            shape = _static_shape(ty)
            k = 0
            for i, s in enumerate(shape):
                if s is None and k < len(dyn):
                    shape[i] = dyn[k]
                    k += 1
            self.scratch_count += 1
            b = self.new_buffer(f"scratch{self.scratch_count}", "scratch", ty, self.loop_vars, shape)
            self.env[res[0]] = MemVal(b, None)
            return "pure"
        if n in ("memref.dealloc",):
            return "pure"
        if n == "tensor.empty":
            m = re.search(r"tensor\.empty\(([^)]*)\)\s*:\s*(tensor<.*>)", rest)
            ty = m.group(2)
            dyn = [self.idx(x) for x in m.group(1).split(",") if x.strip()]
            shape = _static_shape(ty)
            k = 0
            for i, s in enumerate(shape):
                if s is None and k < len(dyn):
                    shape[i] = dyn[k]
                    k += 1
            self.scratch_count += 1
            b = self.new_buffer(f"scratch{self.scratch_count}", "scratch", ty, self.loop_vars, shape)
            self.env[res[0]] = TensorVal(b, 0, None)
            return "pure"
        if n == "bufferization.to_tensor":
            m = re.match(r"bufferization\.to_tensor\s+(%[\w$.#-]+)", rest)
            mv = self.get(m.group(1))
            if not isinstance(mv, MemVal):
                raise Unsupported("to_tensor of non-memref")
            if "restrict" in rest:
                mv.buf.restrict = True
            self.env[res[0]] = TensorVal(mv.buf, mv.buf.version, mv.view)
            return "pure"
        if n == "bufferization.to_memref":
            m = re.match(r"bufferization\.to_memref\s+(%[\w$.#-]+)", rest)
            tv = self.get(m.group(1))
            if not isinstance(tv, TensorVal):
                raise Unsupported("to_memref of non-tensor")
            self.env[res[0]] = ToMem(tv)
            return "writeback"
        if n == "memref.copy":
            m = re.match(r"memref\.copy\s+(%[\w$.#-]+),\s*(%[\w$.#-]+)", line)
            src, dst = self.get(m.group(1)), self.get(m.group(2))
            if isinstance(src, ToMem) and isinstance(dst, MemVal) and dst.view is None \
                    and src.tv.buf is dst.buf and src.tv.view is None:
                self.check_current(src.tv)
                dst.buf.written_back = True
                self.writeback_tops.setdefault(dst.buf.id, []).append(self.cur_top)
                if not self.loop_vars:
                    self.synced[dst.buf.id] = True
                return "writeback"
            # generic element-wise copy
            sval = src.tv if isinstance(src, ToMem) else src
            if isinstance(sval, TensorVal):
                self.check_current(sval)
            self._emit_copy(sval, dst, T(0))
            return "work"
        if n in ("affine.load", "memref.load"):
            m = re.match(r"\S+\s+(%[\w$.#-]+)\[(.*)\]\s*:", rest)
            mv = self.get(m.group(1))
            idx = self._index_list(m.group(2), n)
            r = self.read(mv, idx, (self.read_time or T(0)) + ([] if self.read_time else [Aff.cst(0)]))
            self.env[res[0]] = self.loaded(r)
            return
        if n in ("affine.store", "memref.store"):
            m = re.match(r"\S+\s+(%[\w$.#-]+),\s*(%[\w$.#-]+)\[(.*)\]\s*:", line)
            v = self.fval(m.group(1))
            mv = self.get(m.group(2))
            idx = self._index_list(m.group(3), n)
            buf, coords = self.access(mv, idx)
            self.emit(buf, coords, v, T(0) + [Aff.cst(1)], origin=line)
            self.mark_write(buf, isinstance(mv, TensorVal))
            buf.version += 1
            return "work"
        if n == "tensor.extract":
            m = re.match(r"tensor\.extract\s+(%[\w$.#-]+)\[(.*)\]\s*:", rest)
            tv = self.get(m.group(1))
            self.check_current(tv)
            idx = [self.idx(x) for x in m.group(2).split(",") if x.strip()]
            t = (self.read_time if self.read_time is not None else T(0) + [Aff.cst(0)])
            self.env[res[0]] = self.loaded(self.read(tv, idx, t))
            return
        if n == "tensor.insert":
            m = re.match(r"tensor\.insert\s+(%[\w$.#-]+)\s+into\s+(%[\w$.#-]+)\[(.*)\]\s*:", rest)
            v = self.fval(m.group(1))
            tv = self.get(m.group(2))
            self.check_current(tv)
            idx = [self.idx(x) for x in m.group(3).split(",") if x.strip()]
            buf, coords = self.access(tv, idx)
            self.emit(buf, coords, v, T(0) + [Aff.cst(1)], origin=line)
            self.mark_write(buf, True)
            buf.version += 1
            self.env[res[0]] = TensorVal(buf, buf.version, tv.view)
            return "work"
        if n == "polygeist.submap":
            self._submap(op, rest)
            return "pure"
        if n == "polygeist.submapInverse":
            self._submap_inverse(op, rest, T(0))
            return "pure"
        if n == "tensor.extract_slice":
            self._extract_slice(op, rest)
            return "pure"
        if n == "memref.subview":
            m = re.match(
                r"memref\.subview\s+(%[\w$.#-]+)\[(.*?)\]\s*\[(.*?)\]"
                r"\s*\[(.*?)\]\s*:\s*memref<.*>\s+to\s+(memref<.*>)$", rest)
            if not m:
                raise Unsupported(f"cannot parse memref.subview: {rest}")
            base = self.get(m.group(1))
            if not isinstance(base, MemVal):
                raise Unsupported("subview of a non-memref")
            parts = [[self.idx(x) for x in _split_top(m.group(k))]
                     for k in (2, 3, 4)]
            if len({len(p) for p in parts}) != 1:
                raise Unsupported("subview metadata arity")
            self.env[res[0]] = self._slice_view(
                base, *parts, _rank_of(m.group(5)))
            return "pure"
        if n == "memref.reinterpret_cast":
            m = re.match(
                r"memref\.reinterpret_cast\s+(%[\w$.#-]+)\s+to\s+"
                r"offset:\s*\[(.*?)\],\s*sizes:\s*\[(.*?)\],\s*"
                r"strides:\s*\[(.*?)\]\s*:\s*(memref<.*>)\s+to\s+(memref<.*>)$",
                rest)
            if not m:
                raise Unsupported(f"cannot parse reinterpret_cast: {rest}")
            base = self.get(m.group(1))
            # Reinterpret offsets refer to underlying storage, not to the
            # source view. Until storage descriptors are explicit, accept only
            # a full contiguous rank-one base; never guess through a view.
            if (not isinstance(base, MemVal) or base.buf.rank != 1 or
                    base.view is not None or "strided<" in m.group(5)):
                raise Unsupported("reinterpret_cast requires a contiguous flat base")
            offset = self.idx(m.group(2))
            sizes = [self.idx(x) for x in _split_top(m.group(3))]
            strides = [self.idx(x) for x in _split_top(m.group(4))]
            if len(sizes) != len(strides) or len(sizes) != _rank_of(m.group(6)):
                raise Unsupported("reinterpret_cast metadata arity")
            coordinate = offset
            for k, stride in enumerate(strides):
                if not stride.is_const():
                    raise Unsupported("symbolic reinterpret_cast stride")
                coordinate += Aff.var(f"@v{k}").scale(stride.const)
            self.env[res[0]] = MemVal(base.buf, View([coordinate], sizes))
            return "pure"
        if n == "tensor.insert_slice":
            self._insert_slice(op, rest, T(0))
            return "pure"
        if n == "linalg.generic":
            self._generic(op, T(0))
            return "work"
        if n == "linalg.fill":
            self._fill(op, T(0))
            return "work"
        if n == "affine.for" or n == "scf.for":
            self._for(op, prefix, pos)
            return "work"
        if n == "scf.while":
            self._while(op, prefix, pos)
            return "work"
        if n in ("affine.if", "scf.if"):
            self._if(op, prefix, pos)
            return "work"
        if n in ("func.return", "return"):
            return "return"
        if n in ("affine.yield", "scf.yield", "linalg.yield"):
            return
        if n in ("memref.cast", "tensor.cast"):
            m = re.match(r"\S+\s+(%[\w$.#-]+)", rest)
            self.env[res[0]] = self.get(m.group(1))
            return "pure"
        raise Unsupported(f"unsupported op {n}: {line[:120]}")

    # -- op helpers -----------------------------------------------------
    def _index_list(self, s: str, opname: str) -> list[Aff]:
        if not s.strip():
            return []
        out = []
        for part in _split_top(s):
            part = part.strip()
            if opname.startswith("affine"):
                envm = {}
                for tok in re.findall(r"%[\w$.#-]+", part):
                    envm[tok] = self.idx(tok)
                out.append(parse_affine_expr(part, envm))
            else:
                out.append(self.idx(part))
        return out

    def _submap(self, op: Op, rest: str):
        m = re.match(r"polygeist\.submap\((.*?)\)\s*(?:<)?\{map\s*=\s*(#[\w$.-]+|affine_map<.*?>)\}(?:>)?\s*:\s*\((.*)\)\s*->\s*(\S+)", rest)
        if not m:
            raise Unsupported(f"cannot parse submap: {rest}")
        operands = [x.strip() for x in m.group(1).split(",")]
        amap = parse_affine_map(m.group(2), self.aliases)
        rty = m.group(4)
        base = self.get(operands[0])
        nsym = len(amap.syms)
        syms = [self.idx(x) for x in operands[1:1 + nsym]]
        sizes = [self.idx(x) for x in operands[1 + nsym:]]
        rank = _rank_of(rty)
        if len(sizes) != rank:
            raise Unsupported("submap size arity")
        placeholders = [Aff.var(f"@v{k}") for k in range(len(amap.dims))]
        res = amap.apply(placeholders, syms)
        if isinstance(base, (TensorVal, MemVal)):
            if base.view is not None:
                m2 = {f"@v{k}": a for k, a in enumerate(res)}
                idx = [e.subst(m2) for e in base.view.index]
            else:
                idx = res
            view = View(idx, sizes)
            if isinstance(base, TensorVal):
                self.env[op.results[0]] = TensorVal(base.buf, base.ver, view)
            else:
                self.env[op.results[0]] = MemVal(base.buf, view)
        else:
            raise Unsupported("submap of non-shaped value")

    def _same_view(self, a: Optional[View], b: Optional[View]) -> bool:
        if a is None or b is None:
            return a is None and b is None
        return a.index == b.index

    def _submap_inverse(self, op: Op, rest: str, t):
        m = re.match(r"polygeist\.submapInverse\((.*?)\)\s*(?:<)?\{map\s*=\s*(#[\w$.-]+|affine_map<.*?>)\}(?:>)?\s*:", rest)
        operands = [x.strip() for x in m.group(1).split(",")]
        amap = parse_affine_map(m.group(2), self.aliases)
        base = self.get(operands[0])
        viewv = self.get(operands[1])
        nsym = len(amap.syms)
        syms = [self.idx(x) for x in operands[2:2 + nsym]]
        sizes = [self.idx(x) for x in operands[2 + nsym:]]
        placeholders = [Aff.var(f"@v{k}") for k in range(len(amap.dims))]
        res = amap.apply(placeholders, syms)
        if base.view is not None:
            m2 = {f"@v{k}": a for k, a in enumerate(res)}
            idx = [e.subst(m2) for e in base.view.index]
        else:
            idx = res
        target = View(idx, sizes)
        self._scatter(op, base, viewv, target, t)

    def _scatter(self, op, base, viewv, target: View, t):
        if not isinstance(base, TensorVal) or not isinstance(viewv, TensorVal):
            raise Unsupported("scatter of non-tensor")
        if viewv.buf is base.buf and self._same_view(viewv.view, target) and viewv.ver == base.buf.version:
            # in-place: the modified view already lives in the base buffer
            self.env[op.results[0]] = TensorVal(base.buf, base.buf.version, base.view)
            return
        # element-wise copy view -> base[target]
        self.check_current(viewv)
        dst = TensorVal(base.buf, base.buf.version, target)
        self._emit_copy(viewv, dst, t)
        self.env[op.results[0]] = TensorVal(base.buf, base.buf.version, base.view)

    def _emit_copy(self, src, dst, t):
        if isinstance(dst, TensorVal):
            pass
        sizes = None
        if getattr(dst, "view", None) is not None:
            sizes = dst.view.sizes
        elif getattr(src, "view", None) is not None:
            sizes = src.view.sizes
        else:
            sizes = src.buf.shape if src.buf.shape and all(s is not None for s in src.buf.shape) else None
            if sizes is None:
                sizes = [Aff.var(f"pdim{dst.buf.name}_{k}") for k in range(dst.buf.rank)]
                for k in range(dst.buf.rank):
                    self.params[f"pdim{dst.buf.name}_{k}"] = f"dim({dst.buf.name},{k})"
        if sizes is None or any(s is None for s in sizes):
            raise Unsupported("copy with unknown extent")
        self.gen_count += 1
        cv = [f"c{self.gen_count}_{k}" for k in range(len(sizes))]
        dom = []
        for v, s in zip(cv, sizes):
            dom += [ge(Aff.var(v)), lt(Aff.var(v), s)]
        idx = [Aff.var(v) for v in cv]
        old = self.loop_vars
        self.loop_vars = old + cv
        rd = self.read(src, idx, t + idx + [Aff.cst(0)])
        self.loop_vars = old
        buf, coords = self.access(dst, idx)
        self.emit(buf, coords, rd, t + idx + [Aff.cst(1)], extra_vars=cv, extra_dom=dom,
                  par=set(cv), origin="copy")
        self.mark_write(buf, isinstance(dst, TensorVal))
        buf.version += 1

    def _extract_slice(self, op: Op, rest: str):
        m = re.match(r"tensor\.extract_slice\s+(%[\w$.#-]+)\[(.*?)\]\s*\[(.*?)\]\s*\[(.*?)\]\s*:\s*(\S+)\s+to\s+(\S+)", rest)
        if not m:
            raise Unsupported(f"cannot parse extract_slice: {rest}")
        base = self.get(m.group(1))
        offs = [self.idx(x) for x in m.group(2).split(",")]
        sizes = [self.idx(x) for x in m.group(3).split(",")]
        strides = [self.idx(x) for x in m.group(4).split(",")]
        rrank = _rank_of(m.group(6))
        self.env[op.results[0]] = self._slice_view(base, offs, sizes, strides, rrank)

    def _slice_view(self, base, offs, sizes, strides, rrank):
        nd = len(offs)
        drop = nd - rrank
        dropped = set()
        for k in range(nd):
            if drop == 0:
                break
            if sizes[k].is_const() and sizes[k].const == 1:
                dropped.add(k)
                drop -= 1
        if drop:
            raise Unsupported("cannot infer rank reduction")
        coords, vsizes, vk = [], [], 0
        for k in range(nd):
            if k in dropped:
                coords.append(offs[k])
            else:
                if not strides[k].is_const():
                    raise Unsupported("symbolic slice stride")
                coords.append(offs[k] + Aff.var(f"@v{vk}").scale(strides[k].const))
                vsizes.append(sizes[k])
                vk += 1
        if base.view is not None:
            m2 = {f"@v{k}": a for k, a in enumerate(coords)}
            coords = [e.subst(m2) for e in base.view.index]
        view = View(coords, vsizes)
        if isinstance(base, TensorVal):
            return TensorVal(base.buf, base.ver, view)
        return MemVal(base.buf, view)

    def _insert_slice(self, op: Op, rest: str, t):
        m = re.match(r"tensor\.insert_slice\s+(%[\w$.#-]+)\s+into\s+(%[\w$.#-]+)\[(.*?)\]\s*\[(.*?)\]\s*\[(.*?)\]\s*:\s*(\S+)\s+into\s+(\S+)", rest)
        if not m:
            raise Unsupported(f"cannot parse insert_slice: {rest}")
        src = self.get(m.group(1))
        base = self.get(m.group(2))
        offs = [self.idx(x) for x in m.group(3).split(",")]
        sizes = [self.idx(x) for x in m.group(4).split(",")]
        strides = [self.idx(x) for x in m.group(5).split(",")]
        target = self._slice_view(base, offs, sizes, strides, _rank_of(m.group(6))).view
        self._scatter(op, base, src, target, t)

    def _generic(self, op: Op, t):
        line = op.line
        mm = re.search(r"indexing_maps\s*=\s*\[(.*?)\]\s*,\s*iterator_types\s*=\s*\[(.*?)\]", line)
        if not mm:
            raise Unsupported("generic without maps")
        maps = [parse_affine_map(x.strip(), self.aliases) for x in _split_top(mm.group(1))]
        iters = [x.strip().strip('"') for x in mm.group(2).split(",")]
        ins_m = re.search(r"\bins\((.*?)\s*:\s*[^)]*\)\s*outs\(", line)
        outs_m = re.search(r"\bouts\((.*?)\s*:", line)
        ins = [x.strip() for x in ins_m.group(1).split(",")] if ins_m else []
        outs = [x.strip() for x in outs_m.group(1).split(",")] if outs_m else []
        operands = [self.get(x) for x in ins + outs]
        for v in operands:
            if not isinstance(v, (TensorVal, MemVal)):
                raise Unsupported("generic operand is not shaped")
            self.check_current(v)
        ndims = len(iters)
        self.gen_count += 1
        gv = [f"g{self.gen_count}_{d}" for d in range(ndims)]
        # ranges
        ranges: list[Optional[Aff]] = [None] * ndims
        for v, amap in zip(operands, maps):
            sizes = v.view.sizes if v.view is not None else v.buf.shape
            for pos, r in enumerate(amap.results):
                r = r.strip()
                if r in amap.dims:
                    d = amap.dims.index(r)
                    if ranges[d] is None and pos < len(sizes) and sizes[pos] is not None:
                        ranges[d] = sizes[pos]
        if any(r is None for r in ranges):
            raise Unsupported("cannot infer generic iteration range")
        dom = []
        for v, r in zip(gv, ranges):
            dom += [ge(Aff.var(v)), lt(Aff.var(v), r)]
        dimaff = [Aff.var(v) for v in gv]
        rt = t + dimaff + [Aff.cst(0)]
        bargs = op.block_args[0]
        old_vars, old_dom = self.loop_vars, self.domain
        self.loop_vars = old_vars + gv
        self.domain = old_dom + dom
        out_reads = []
        for k, (v, amap) in enumerate(zip(operands, maps)):
            idx = amap.apply(dimaff, [])
            rd = self.read(v, idx, rt)
            self.env[bargs[k][0]] = FVal(rd)
            if k >= len(ins):
                out_reads.append(rd)
        saved_rt = self.read_time
        self.read_time = rt
        yields = None
        for bop in op.regions[0]:
            if bop.name == "linalg.index":
                mi = re.search(r"linalg\.index\s+(\d+)", bop.line)
                self.env[bop.results[0]] = IdxVal(dimaff[int(mi.group(1))])
                continue
            if bop.name == "linalg.yield":
                ym = re.match(r"linalg\.yield\s+(.*?)\s*:", bop.line)
                yields = [x.strip() for x in ym.group(1).split(",")]
                continue
            self.exec_op(bop, rt, 0)
        self.read_time = saved_rt
        self.loop_vars, self.domain = old_vars, old_dom
        if yields is None:
            raise Unsupported("generic without yield")
        par = {gv[d] for d, it in enumerate(iters) if it == "parallel"}
        results = []
        for k, ytok in enumerate(yields):
            val = self.fval(ytok)
            outv = operands[len(ins) + k]
            outmap = maps[len(ins) + k]
            guard: list[Cons] = []
            val = _strip_guards(val, out_reads[k], guard)
            idx = outmap.apply(dimaff, [])
            buf, coords = self.access(outv, idx)
            self.loop_vars = old_vars + gv
            st = Stmt(len(self.stmts), list(old_vars) + gv, list(old_dom) + dom, guard,
                      t + dimaff + [Aff.cst(1 + k)], buf, list(coords), val, par, self.cur_top,
                      "linalg.generic")
            self.loop_vars = old_vars
            self.stmts.append(st)
            self.stored.setdefault(id(val), (st,))
            results.append((outv, buf))
        for (outv, buf) in results:
            self.mark_write(buf, isinstance(outv, TensorVal))
            buf.version += 1
        for k, r in enumerate(op.results):
            outv, buf = results[k]
            self.env[r] = TensorVal(buf, buf.version, outv.view)

    def _fill(self, op: Op, t):
        m = re.search(r"ins\((%[\w$.#-]+)\s*:\s*\S+\)\s*outs\((%[\w$.#-]+)\s*:", op.line)
        if not m:
            raise Unsupported("fill")
        v = self.fval(m.group(1))
        outv = self.get(m.group(2))
        self.check_current(outv)
        sizes = outv.view.sizes if outv.view is not None else outv.buf.shape
        if any(s is None for s in sizes):
            raise Unsupported("fill extent")
        self.gen_count += 1
        gv = [f"g{self.gen_count}_{d}" for d in range(len(sizes))]
        dom = []
        for g, s in zip(gv, sizes):
            dom += [ge(Aff.var(g)), lt(Aff.var(g), s)]
        dimaff = [Aff.var(g) for g in gv]
        buf, coords = self.access(outv, dimaff)
        self.emit(buf, coords, v, t + dimaff + [Aff.cst(1)], extra_vars=gv, extra_dom=dom,
                  par=set(gv), origin="linalg.fill")
        self.mark_write(buf, isinstance(outv, TensorVal))
        buf.version += 1
        if op.results:
            self.env[op.results[0]] = TensorVal(buf, buf.version, outv.view if isinstance(outv, TensorVal) else None)

    def _bound(self, s: str, lower: bool) -> list[Aff]:
        s = s.strip()
        kind = None
        if s.startswith("max "):
            kind, s = "max", s[4:].strip()
        elif s.startswith("min "):
            kind, s = "min", s[4:].strip()
        if re.fullmatch(r"-?\d+", s) or re.fullmatch(r"%[\w$.#-]+", s):
            return [self.idx(s)]
        m = re.match(r"(#[\w$.-]+|affine_map<.*?>)\s*\(([^)]*)\)(?:\[([^\]]*)\])?$", s)
        if not m:
            raise Unsupported(f"loop bound {s}")
        amap = parse_affine_map(m.group(1), self.aliases)
        dims = [self.idx(x) for x in m.group(2).split(",") if x.strip()]
        syms = [self.idx(x) for x in (m.group(3) or "").split(",") if x.strip()]
        vals = amap.apply(dims, syms)
        if len(vals) > 1:
            if lower and kind != "max" or (not lower and kind != "min"):
                raise Unsupported("multi-result bound without max/min")
        return vals

    def _for(self, op: Op, prefix, pos):
        line = op.line
        if op.name == "affine.for":
            m = re.match(r".*?affine\.for\s+(%[\w$.#-]+)\s*=\s*(.*?)\s+to\s+(.*?)(?:\s+step\s+(\d+))?\s*(?:iter_args\((.*?)\)\s*->\s*\((.*?)\))?\s*\{$", line)
        else:
            m = re.match(r".*?scf\.for\s+(%[\w$.#-]+)\s*=\s*(%[\w$.#-]+)\s+to\s+(%[\w$.#-]+)\s+step\s+(%[\w$.#-]+)\s*(?:iter_args\((.*?)\)\s*->\s*\((.*?)\))?.*\{$", line)
        if not m:
            raise Unsupported(f"cannot parse loop: {line}")
        iv, lbs, ubs, step = m.group(1), m.group(2), m.group(3), m.group(4)
        if op.name == "affine.for":
            if step and step != "1":
                raise Unsupported("non-unit step")
            lows = self._bound(lbs, True)
            ups = self._bound(ubs, False)
        else:
            st = self.idx(step)
            if not (st.is_const() and st.const == 1):
                raise Unsupported("non-unit scf step")
            lows, ups = [self.idx(lbs)], [self.idx(ubs)]
        self.loop_count += 1
        var = f"i{self.loop_count}"
        dom = [ge(Aff.var(var), lo) for lo in lows] + [lt(Aff.var(var), up) for up in ups]
        iter_args = []
        if m.group(5):
            for part in _split_top(m.group(5)):
                a, b = part.split("=")
                iter_args.append((a.strip(), b.strip()))
        T0 = prefix + [Aff.cst(pos)]
        body_prefix = prefix + [Aff.cst(pos + 1), Aff.var(var)]
        scal = {}
        for k, (barg, init) in enumerate(iter_args):
            v = self.get(init)
            if isinstance(v, TensorVal):
                self.check_current(v)
                self.env[barg] = TensorVal(v.buf, v.buf.version, v.view)
            elif isinstance(v, (FVal, Undef)):
                self.scratch_count += 1
                b = self.new_buffer(f"scratch{self.scratch_count}", "scratch", "f64", self.loop_vars, [])
                b.rank = 0
                e = v.e if isinstance(v, FVal) else EUndef()
                self.emit(b, [Aff.var(p) for p in b.priv], e, T0 + [Aff.cst(k), Aff.cst(1)], origin="iter_arg init")
                scal[k] = b
            else:
                raise Unsupported("iter_arg kind")
        old_vars, old_dom = self.loop_vars, self.domain
        self.loop_vars = old_vars + [var]
        self.domain = old_dom + dom
        for k, b in scal.items():
            self.env[iter_args[k][0]] = FVal(ERead(b, tuple(Aff.var(p) for p in b.priv), tuple(body_prefix + [Aff.cst(0), Aff.cst(0)]), (), tuple(self.loop_vars)))
        self.env[iv] = IdxVal(Aff.var(var))
        body = op.regions[0]
        yields = []
        bpos = 1
        for bop in body:
            if bop.name in ("affine.yield", "scf.yield"):
                ym = re.match(r"\S+\s+(.*?)\s*:", bop.line)
                yields = [x.strip() for x in ym.group(1).split(",")] if ym else []
                continue
            self.exec_op(bop, body_prefix, bpos)
            bpos += 4
        results = []
        for k, (barg, init) in enumerate(iter_args):
            y = self.get(yields[k])
            if k in scal:
                b = scal[k]
                self.emit(b, [Aff.var(p) for p in b.priv], self.fval(yields[k]),
                          body_prefix + [Aff.cst(bpos), Aff.cst(1)], origin="iter_arg yield")
                results.append(("scal", b))
            else:
                init_v = self.get(init)
                if not isinstance(y, TensorVal) or y.buf is not init_v.buf:
                    raise Unsupported("loop-carried tensor changes buffer (swap)")
                self.check_current(y)
                results.append(("tensor", y))
        self.loop_vars, self.domain = old_vars, old_dom
        for k, r in enumerate(op.results):
            kind, v = results[k]
            if kind == "scal":
                self.env[r] = FVal(ERead(v, tuple(Aff.var(p) for p in v.priv), tuple(prefix + [Aff.cst(pos + 2), Aff.cst(0)]), (), tuple(self.loop_vars)))
            else:
                self.env[r] = TensorVal(v.buf, v.buf.version, v.view)

    def _while(self, op: Op, prefix, pos):
        """A while loop that is a counted loop -- one integer iter_arg iv,
        condition `iv < bound` with a bound computed before the loop that does
        not depend on iv, body ending in `yield iv + 1` -- is executed as
        `scf.for iv = init to bound`.  (cgeist emits this form when the bound
        is re-read in the condition, e.g. `k < colPtr[j + 1]`, and the body
        stores.)  The bound may be an index atom: equations.py rejects atoms
        of arrays the region writes, so its value cannot change in the loop.
        Anything else is unsupported."""
        m = re.match(r".*?scf\.while\s*\((%[\w$.#-]+)\s*=\s*(%[\w$.#-]+)\)\s*:\s*\((\w+)\)", op.line)
        if not m or len(op.regions) != 2 or len(op.block_args[1]) != 1:
            raise Unsupported("while loop is not a counted loop (iter_args)")
        biv, init = m.group(1), m.group(2)
        if not isinstance(self.get(init), IdxVal):
            raise Unsupported("while loop with a non-index iter_arg")
        before, after = op.regions
        cond = before[-1] if before else None
        cm = cond and re.match(r"scf\.condition\((%[\w$.#-]+)\)\s+(%[\w$.#-]+)\s*:", cond.line)
        if not cm or cm.group(2) != biv:
            raise Unsupported("while loop does not forward its counter")
        cmp = next((o for o in before if o.results and o.results[0] == cm.group(1)), None)
        km = cmp and re.match(r".*?arith\.cmpi\s+(\w+),\s*(%[\w$.#-]+),\s*(%[\w$.#-]+)", cmp.line)
        if not km:
            raise Unsupported("while condition is not an integer comparison")
        kind, lhs, rhs = km.groups()
        if lhs == biv and kind in ("slt", "ult", "sle", "ule"):
            bound_tok, closed = rhs, kind in ("sle", "ule")
        elif rhs == biv and kind in ("sgt", "ugt", "sge", "uge"):
            bound_tok, closed = lhs, kind in ("sge", "uge")
        else:
            raise Unsupported(f"while condition {kind} is not iv < bound")
        # the bound is computed outside the loop: run the rest of the before
        # region without the counter in scope (a use of it is unsupported)
        for o in before:
            if o is cmp or o is cond:
                continue
            if o.name in ("affine.store", "memref.store", "tensor.insert") or o.regions:
                raise Unsupported("while condition with side effects")
            self.exec_op(o, prefix, pos)
        bound = self.get(bound_tok)
        if not isinstance(bound, IdxVal):
            raise Unsupported("while bound is not an index")
        aiv = op.block_args[1][0][0]
        body = list(after)
        ym = body and re.match(r"scf\.yield\s+(%[\w$.#-]+)\s*:", body[-1].line)
        inc = ym and next((o for o in body if o.results and o.results[0] == ym.group(1)), None)
        im = inc and re.match(r".*?arith\.addi\s+(%[\w$.#-]+),\s*(%[\w$.#-]+)", inc.line)
        if not im:
            raise Unsupported("while body does not increment its counter")
        step = im.group(2) if im.group(1) == aiv else im.group(1) if im.group(2) == aiv else None
        sv = self.get(step) if step else None
        if not (isinstance(sv, IdxVal) and sv.aff.is_const() and sv.aff.const == 1):
            raise Unsupported("while counter step is not 1")
        rest = [o for o in body[:-1] if o is not inc]
        if any(re.search(re.escape(inc.results[0]) + r"(?![\w$.-])", o.line) for o in rest):
            raise Unsupported("while body uses the incremented counter")
        ub = f"{bound_tok}#ub"
        self.env[ub] = IdxVal(bound.aff + 1 if closed else bound.aff)
        self.env["%__while_one"] = IdxVal(Aff.cst(1))
        loop = Op("scf.for", [], f"scf.for {aiv} = {init} to {ub} step %__while_one {{",
                  op.start, op.end, [rest], [[]])
        self._for(loop, prefix, pos)
        for r in op.results:
            self.env[r] = Undef()

    def _if(self, op: Op, prefix, pos):
        if op.results:
            raise Unsupported("if with results")
        line = op.line
        if op.name == "affine.if":
            m = re.match(r".*?affine\.if\s+(#[\w$.-]+|affine_set<.*?>)\s*\(([^)]*)\)(?:\[([^\]]*)\])?", line)
            dims, syms, conds = parse_affine_set(m.group(1), self.aliases)
            dv = [self.idx(x) for x in m.group(2).split(",") if x.strip()]
            sv = [self.idx(x) for x in (m.group(3) or "").split(",") if x.strip()]
            envm = {d: a for d, a in zip(dims, dv)}
            envm.update({s: a for s, a in zip(syms, sv)})
            cons = []
            for c in conds:
                c = c.strip()
                if c.endswith(">= 0"):
                    cons.append(Cons(parse_affine_expr(c[:-4], envm), "ge"))
                elif c.endswith("== 0"):
                    cons.append(Cons(parse_affine_expr(c[:-4], envm), "eq"))
                else:
                    raise Unsupported(f"affine set {c}")
        else:
            m = re.match(r".*?scf\.if\s+(%[\w$.#-]+)", line)
            c = self.get(m.group(1))
            if not isinstance(c, CondAff):
                raise Unsupported("data-dependent scf.if")
            cons = c.cons
        old = self.domain
        self.domain = old + cons
        self.exec_block([o for o in op.regions[0] if o.name not in ("affine.yield", "scf.yield")],
                        prefix + [Aff.cst(pos), Aff.cst(0)])
        self.domain = old
        if len(op.regions) > 1 and any(o.name not in ("affine.yield", "scf.yield") for o in op.regions[1]):
            if len(cons) != 1:
                raise Unsupported("else-branch of a multi-constraint condition")
            for alt in negate(cons[0]):
                self.domain = old + alt
                self.exec_block([o for o in op.regions[1] if o.name not in ("affine.yield", "scf.yield")],
                                prefix + [Aff.cst(pos), Aff.cst(1)])
            self.domain = old


def _launch_parts(rest: str):
    m = re.match(r"kernel\.launch\s+@([\w$.-]+)\((.*?)\)\s*:", rest)
    if not m:
        raise Unsupported(f"cannot parse kernel.launch: {rest[:120]}")
    return m.group(1), [x.strip() for x in _split_top(m.group(2)) if x.strip()]


def _exec_launch(self, op: Op, rest: str):
    """An already-matched library call: opaque.  Its effect is known only
    as a new version of the operands it writes (taken from the specification's
    outputs); windows never contain it."""
    spec, operands = _launch_parts(rest)
    outs = self.launch_outputs(spec) if self.launch_outputs else None
    if outs is None:
        raise Unsupported(f"kernel.launch of unknown specification @{spec}")
    vals = [self.get(o) for o in operands]
    for v in vals:
        if isinstance(v, TensorVal):
            self.check_current(v)
    if op.results:
        if len(op.results) != len(outs):
            raise Unsupported(f"kernel.launch @{spec}: {len(op.results)} results vs {len(outs)} outputs")
        for r, k in zip(op.results, outs):
            v = vals[k]
            if not isinstance(v, TensorVal) or v.view is not None:
                raise Unsupported(f"kernel.launch @{spec}: output operand {k} is not a full tensor")
            v.buf.version += 1
            self.mark_write(v.buf, True)
            self.env[r] = TensorVal(v.buf, v.buf.version, None)
    else:
        for k in outs:
            v = vals[k]
            if not isinstance(v, MemVal):
                raise Unsupported(f"kernel.launch @{spec}: output operand {k} is not a memref")
            v.buf.version += 1
    return "opaque"


Executor._launch = _exec_launch


def _strip_guards(val: Expr, out_read: ERead, guard: list[Cons]) -> Expr:
    """select(c, X, out) -> X under guard c (the instance is a no-op otherwise)."""
    while isinstance(val, ESel) and isinstance(val.cond, CondAff):
        if val.b is out_read:
            guard.extend(val.cond.cons)
            val = val.a
            continue
        if val.a is out_read and len(val.cond.cons) == 1:
            alts = negate(val.cond.cons[0])
            if len(alts) == 1:
                guard.extend(alts[0])
                val = val.b
                continue
        break
    return val


def execute(func: Func, aliases: dict[str, str], fixed_ints: Optional[dict] = None,
            launch_outputs=None) -> Program:
    return Executor(func, aliases, fixed_ints, launch_outputs).run()
