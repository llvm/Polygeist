#!/usr/bin/env python3
"""lower_v2.py -- lower `kernel.launch @netlib_*` to lib_v2 runtime calls.

MLIR in, MLIR out.  Every `kernel.launch @netlib_X(...)` becomes
`func.call @polygeist_v2_<target.call>(...)` (see runtime/polygeist_v2_rt.h).
The lowering is driven entirely by the ABI table of abi_v2.py (manifests,
spec C prototypes, spec intents, vendor headers); it contains no per-spec or
per-routine code.

Stages
  1. prepare   drop every kernel.defn and synthesise one stub per launched
               spec/signature whose body yields the destination operands (the
               spec's written arrays, in argument order).  kernel.launch's
               BufferizableOpInterface reads destinations from such a yield, so
               One-Shot Bufferize can decide in-place vs copy soundly.  Mark
               to_tensor `restrict writable` (production pre-ABI convention).
  2. bufferize production view cleanup (--lower-polygeist-submap
               --canonicalize-polygeist, loop fallback for leftovers) followed
               by One-Shot Bufferize (allow-unknown-ops,
               bufferize-function-boundaries), as polygeist_build.sh does
               before ABI lowering.  Skipped when the input has no tensors.
  3. rewrite   each (now memref) launch -> func.call:
                 fixed enum args  -> i32 constants (vendor enum values)
                 fixed numbers    -> f64 / i32 constants
                 int / scalar     -> i32 / f64 (cast as needed)
                 arrays           -> !llvm.ptr to element 0 (aligned pointer +
                                     offset * elem size) and i64 byte extent of
                                     the view footprint
                 layout params    -> from memref strides, never from extents:
                                     rank 2: ld = stride[0] (asserting a unit
                                     inner stride); rank 1: inc = stride[0] x
                                     the spec's increment.
               Stub defns are removed.

Usage: lower_v2.py IN.mlir -o OUT.mlir [--no-bufferize] [--report R.json]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from c_types import MLIR_ELEMENT_TYPES
import abi_v2 as A  # noqa: E402
import recipe as R  # noqa: E402

LAUNCH_RE = re.compile(
    r"^(?P<ind>\s*)(?:(?P<res>%[\w$.\-]+)(?::(?P<nres>\d+))?\s*=\s*)?"
    r"kernel\.launch\s+@(?P<sym>[\w$.\-]+)\((?P<ops>[^)]*)\)\s*"
    r"(?P<attrs>\{.*?\})?\s*:\s*\((?P<tys>.*)\)\s*->\s*(?P<rty>.*?)\s*$")

ELEM_BYTES = {"f64": 8, "f32": 4, "i8": 1, "i16": 2, "i32": 4, "i64": 8}


class LowerError(Exception):
    pass


def split_top(s: str) -> list[str]:
    return A._split_top(s)


def parse_results(rty: str) -> list[str]:
    rty = rty.strip()
    if rty in ("()", ""):
        return []
    if rty.startswith("(") and rty.endswith(")"):
        return split_top(rty[1:-1])
    return [rty]


def memref_info(ty: str) -> tuple[int, str]:
    m = re.fullmatch(r"memref<((?:[?\d]+x)*)([a-z]\w*)(?:,.*)?>", ty.strip())
    if not m:
        raise LowerError(f"not a ranked memref type: {ty}")
    rank = len([d for d in m.group(1).split("x") if d])
    return rank, m.group(2)


def find_block(text: str, start: int) -> int:
    """index just past the `}` closing the first `{` at/after start."""
    i = text.index("{", start)
    depth = 0
    for k in range(i, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return k + 1
    raise LowerError("unbalanced braces")


def strip_defns(text: str) -> str:
    out, pos = [], 0
    for m in re.finditer(r"^\s*kernel\.defn\b", text, flags=re.M):
        if m.start() < pos:
            continue
        end = find_block(text, m.end())
        out.append(text[pos:m.start()])
        pos = end
        if pos < len(text) and text[pos] == "\n":
            pos += 1
    out.append(text[pos:])
    return "".join(out)


def insert_after_module_open(text: str, lines: str) -> str:
    m = re.search(r"^module\b[^\n]*\{\s*$", text, flags=re.M)
    if not m:
        raise LowerError("no top-level `module {` line")
    return text[:m.end()] + "\n" + lines + text[m.end():]


# ---------------------------------------------------------------------------
# Stage 1: stub defns
# ---------------------------------------------------------------------------

def prepare(text: str, specs: dict, calls: dict | None = None) -> tuple[str, dict]:
    """-> (text, defn name -> spec)"""
    text = strip_defns(text)
    sigs: dict[tuple, str] = {}
    lines = text.split("\n")
    # Keep application symbols (including declarations and referenced external
    # callees) separate from the temporary kernel definitions used for ABI
    # preparation. A library spec can legitimately share an application's name.
    existing = "\n".join(ln for ln in lines if not LAUNCH_RE.match(ln))
    reserved = set()
    for quoted, plain in re.findall(r'@(?:"((?:[^"\\]|\\.)*)"|([\w$.\-]+))', existing):
        reserved.add(re.sub(r'\\([0-9A-Fa-f]{2})',
                            lambda m: chr(int(m[1], 16)), quoted) if quoted else plain)
    for i, ln in enumerate(lines):
        m = LAUNCH_RE.match(ln)
        if not m:
            continue
        sym = m.group("sym")
        if sym not in specs:
            raise LowerError(f"{sym}: no ABI (manifest/spec/header binding failed)")
        tys = tuple(split_top(m.group("tys")))
        rtys = tuple(parse_results(m.group("rty")))
        key = (sym, tys, rtys)
        if key not in sigs:
            n = sum(1 for k in sigs if k[0] == sym)
            name = sym if n == 0 else f"{sym}__v2sig{n}"
            while name in reserved:
                n += 1
                name = f"{sym}__v2sig{n}"
            sigs[key] = name
            reserved.add(name)
        name = sigs[key]
        # Never trust a capability supplied in input IR. It is derived from
        # the validated reference/backend contract, not the application name.
        capability = "polygeist.elementwise_exact_alias"
        attrs = [a for a in split_top((m.group("attrs") or "{}")[1:-1])
                 if a.strip().split("=", 1)[0].strip() != capability]
        call = (calls or {}).get(specs[sym].call)
        cfg = call.runtime_adapter if call else None
        if (cfg and cfg.get("kind") == "pointwise_reference" and
                cfg.get("alias_policy") == "exact_or_disjoint"):
            attrs.append(capability)
        attr_text = " {" + ", ".join(attrs) + "}" if attrs else ""
        prefix = ln[:m.start("attrs")] if m.group("attrs") else ln[:m.start("tys")].rsplit(":", 1)[0]
        lines[i] = (prefix.rstrip() + attr_text + " : (" + m.group("tys") +
                    ") -> " + m.group("rty"))
        if name != sym:
            lines[i] = lines[i].replace(f"@{sym}(", f"@{name}(", 1)
    stubs, names = [], {}
    for (sym, tys, rtys), name in sigs.items():
        sa: A.SpecAbi = specs[sym]
        written = [k for k, a in enumerate(sa.args) if a.kind == "array" and a.intent in ("out", "inout")]
        args = ", ".join(f"%a{k}: {t}" for k, t in enumerate(tys))
        ret = f" -> ({', '.join(rtys)})" if len(rtys) > 1 else (f" -> {rtys[0]}" if rtys else "")
        if sa.ret != "void":
            # Read-only tensor inputs become memrefs while the ordinary scalar
            # SSA result survives bufferization. Mutating return-value APIs need
            # explicit tensor results as well and remain unsupported here.
            expected = {"float": ("f32",), "double": ("f64",)}.get(sa.ret)
            if written or rtys != expected:
                raise LowerError(f"{sym}: unsupported scalar-return destination contract")
            body = ""
            if rtys:
                body = (f"    %r = arith.constant 0.0 : {rtys[0]}\n"
                        f"    kernel.yield %r : {rtys[0]}")
            else:
                body = "    kernel.yield"
            stubs.append(f"  kernel.defn @{name}({args}){ret} {{\n{body}\n  }}\n")
            names[name] = sym
            continue
        if len(rtys) > len(written):
            raise LowerError(f"{sym}: {len(rtys)} results but only {len(written)} written arrays")
        dests = written[:len(rtys)] if rtys else []
        for k, rt in zip(dests, rtys):
            if tys[k] != rt:
                raise LowerError(f"{sym}: result type {rt} != destination operand type {tys[k]}")
        y = (f"    kernel.yield {', '.join(f'%a{k}' for k in dests)} : {', '.join(rtys)}"
             if rtys else "    kernel.yield")
        stubs.append(f"  kernel.defn @{name}({args}){ret} {{\n{y}\n  }}\n")
        names[name] = sym
    text = "\n".join(lines)
    if stubs:
        text = insert_after_module_open(text, "".join(stubs))
    text = re.sub(r"bufferization\.to_tensor (%[^ ]*) :",
                  r"bufferization.to_tensor \1 restrict writable :", text)
    return text, names


# ---------------------------------------------------------------------------
# Stage 2: bufferize (production pre-ABI flow)
# ---------------------------------------------------------------------------

def run_opt(args: list[str], text: str, log: list) -> str:
    with tempfile.TemporaryDirectory() as d:
        inp, out = Path(d) / "in.mlir", Path(d) / "out.mlir"
        inp.write_text(text)
        cmd = [R.POLYGEIST_OPT, *args, str(inp), "-o", str(out)]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        log.append({"cmd": " ".join(args), "rc": p.returncode, "stderr": p.stderr[-3000:]})
        if p.returncode:
            raise LowerError(f"polygeist-opt {' '.join(args)} failed:\n{p.stderr[-3000:]}")
        return out.read_text()


def _views_left(text: str) -> bool:
    return bool(re.search(r"polygeist\.submap(Inverse)?\(.*tensor<", text))


def _cleanup_views(text: str, log: list) -> str:
    text = run_opt(["--lower-polygeist-submap", "--canonicalize-polygeist"], text, log)
    if "polygeist.submap" in text:
        text = run_opt(["--convert-linalg-to-loops", "--canonicalize",
                        "--lower-polygeist-submap", "--canonicalize"], text, log)
    return text


def legalize(text: str, log: list) -> str:
    """Agent 1's test-only tensor-view legaliser (legalize_tensor_submaps.py)."""
    with tempfile.TemporaryDirectory() as d:
        inp, out = Path(d) / "in.mlir", Path(d) / "out.mlir"
        inp.write_text(text)
        p = subprocess.run([sys.executable, str(HERE / "legalize_tensor_submaps.py"), str(inp),
                            "-o", str(out)], capture_output=True, text=True, timeout=600)
        log.append({"cmd": "legalize_tensor_submaps.py", "rc": p.returncode,
                    "stderr": p.stderr[-3000:]})
        if p.returncode:
            raise LowerError(f"legalize_tensor_submaps.py failed:\n{p.stderr[-2000:]}")
        return out.read_text()


def bufferize(text: str, log: list, allow_legaliser: bool = True) -> str:
    if "tensor<" not in text:
        return text
    src = text
    # production flow first (polygeist_build.sh ABI cleanup)
    text = _cleanup_views(src, log)
    if _views_left(text):
        if not allow_legaliser:
            raise LowerError("tensor-form polygeist.submap/submapInverse survive the production "
                             "view lowering (dynamic-size reduction write-back)")
        log.append({"note": "production view lowering left tensor submaps; "
                            "falling back to legalize_tensor_submaps.py"})
        text = _cleanup_views(legalize(src, log), log)
        if _views_left(text):
            raise LowerError("tensor-form submaps survive even the legaliser")
    # affine.for has no BufferizableOpInterface: tensor iter_args would stay
    # behind to_memref/to_tensor casts.  --lower-affine (which the later
    # mlir-opt pipeline runs anyway) turns them into scf.for first.
    text = run_opt(["--lower-affine", "--empty-tensor-to-alloc-tensor",
                    "--one-shot-bufferize=allow-unknown-ops bufferize-function-boundaries",
                    "--canonicalize", "--cse"], text, log)
    return text


# ---------------------------------------------------------------------------
# Stage 3: launch -> call
# ---------------------------------------------------------------------------

class Emitter:
    def __init__(self, ind: str, tag: str):
        self.ind, self.tag, self.n, self.lines = ind, tag, 0, []
        self.consts: dict[tuple, str] = {}

    def fresh(self, hint: str) -> str:
        self.n += 1
        return f"%pv2_{self.tag}_{hint}{self.n}"

    def op(self, rhs: str, hint: str = "v") -> str:
        v = self.fresh(hint)
        self.lines.append(f"{self.ind}{v} = {rhs}")
        return v

    def const(self, val, ty: str) -> str:
        key = (str(val), ty)
        if key not in self.consts:
            lit = repr(float(val)) if ty.startswith("f") else str(int(val))
            if ty.startswith("f") and "e" not in lit and "." not in lit:
                lit += ".0"
            self.consts[key] = self.op(f"arith.constant {lit} : {ty}", "c")
        return self.consts[key]

    def raw(self, line: str):
        self.lines.append(f"{self.ind}{line}")


def cast_int(e: Emitter, v: str, ty: str) -> str:
    if ty == "i32":
        return v
    if ty == "index":
        return e.op(f"arith.index_cast {v} : index to i32", "i")
    if ty == "i64":
        return e.op(f"arith.trunci {v} : i64 to i32", "i")
    if re.fullmatch(r"i(8|16)", ty):
        return e.op(f"arith.extsi {v} : {ty} to i32", "i")
    raise LowerError(f"cannot pass {ty} as int")


def cast_scalar(e: Emitter, v: str, ty: str, target: str) -> str:
    if ty == target:
        return v
    raise LowerError(f"cannot pass {ty} as {target} library scalar")


class ArrayVal:
    def __init__(self, e: Emitter, v: str, ty: str, spec_elem: str):
        rank, elem = memref_info(ty)
        if elem != spec_elem:
            raise LowerError(f"array {v}: element {elem} != library element {spec_elem}")
        self.rank, self.v, self.ty = rank, v, ty
        nb = ELEM_BYTES[elem]
        idx = ", ".join(["index"] * (1 + 2 * rank))
        md = e.fresh("md")
        e.raw(f"{md}:{2 + 2 * rank} = memref.extract_strided_metadata {v} : {ty} -> memref<{elem}>, {idx}")
        self.off = f"{md}#1"
        self.sizes = [f"{md}#{2 + d}" for d in range(rank)]
        self.strides = [f"{md}#{2 + rank + d}" for d in range(rank)]
        al = e.op(f"memref.extract_aligned_pointer_as_index {v} : {ty} -> index", "al")
        c0, c1, cb = e.const(0, "index"), e.const(1, "index"), e.const(nb, "index")
        offb = e.op(f"arith.muli {self.off}, {cb} : index", "ob")
        addr = e.op(f"arith.addi {al}, {offb} : index", "ad")
        a64 = e.op(f"arith.index_cast {addr} : index to i64", "a")
        self.ptr = e.op(f"llvm.inttoptr {a64} : i64 to !llvm.ptr", "p")
        # footprint: (sum (size_d - 1) * stride_d + 1) elements, 0 if empty
        acc, empty = c1, None
        for s, st in zip(self.sizes, self.strides):
            sm1 = e.op(f"arith.subi {s}, {c1} : index", "s")
            t = e.op(f"arith.muli {sm1}, {st} : index", "t")
            acc = e.op(f"arith.addi {acc}, {t} : index", "f")
            z = e.op(f"arith.cmpi eq, {s}, {c0} : index", "z")
            empty = z if empty is None else e.op(f"arith.ori {empty}, {z} : i1", "z")
        if empty is not None:
            acc = e.op(f"arith.select {empty}, {c0}, {acc} : index", "f")
        nbytes = e.op(f"arith.muli {acc}, {cb} : index", "nb")
        self.bytes = e.op(f"arith.index_cast {nbytes} : index to i64", "nb")
        self._layout = None

    def layout(self, e: Emitter, spec_inc: str | None, what: str) -> str:
        """Leading dimension (rank 2) or increment (rank 1) from strides."""
        c1 = e.const(1, "index")
        if self.rank == 2:
            # BLAS needs a unit inner stride; a single column needs none.
            okstride = e.op(f"arith.cmpi eq, {self.strides[1]}, {c1} : index", "ok")
            onecol = e.op(f"arith.cmpi ule, {self.sizes[1]}, {c1} : index", "ok")
            ok = e.op(f"arith.ori {okstride}, {onecol} : i1", "ok")
            e.raw(f'cf.assert {ok}, "lib_v2 {what}: column-major operand needs unit inner stride"')
            # with one row the stride is irrelevant; BLAS still wants ld >= rows
            onerow = e.op(f"arith.cmpi ule, {self.sizes[0]}, {c1} : index", "r")
            mx = e.op(f"arith.maxui {self.sizes[1]}, {c1} : index", "r")
            ld = e.op(f"arith.select {onerow}, {mx}, {self.strides[0]} : index", "ld")
            return e.op(f"arith.index_cast {ld} : index to i32", "ld")
        if self.rank == 1:
            one = e.op(f"arith.cmpi ule, {self.sizes[0]}, {c1} : index", "r")
            st = e.op(f"arith.select {one}, {c1}, {self.strides[0]} : index", "st")
            st32 = e.op(f"arith.index_cast {st} : index to i32", "st")
            if spec_inc is None:
                return st32
            return e.op(f"arith.muli {st32}, {spec_inc} : i32", "inc")
        raise LowerError(f"{what}: rank-{self.rank} operand has no BLAS layout")



def matrix_needs_layout_adapter(ty: str) -> bool:
    """Only potentially non-unit matrix views need a physical layout copy."""
    if memref_info(ty)[0] != 2:
        return False
    shape = ty[len('memref<'):].split(',', 1)[0].split('x')
    if shape[1] in ('0', '1'):
        return False
    layout = re.search(r'strided<\[([^]]+)\]', ty)
    if layout:
        return split_top(layout[1])[1].strip() != '1'
    return ',' in ty  # nontrivial/unknown layout; helper has a no-copy fast path


def pack_matrix(e: Emitter, av: ArrayVal, written: bool) -> str:
    original = [av.ptr, av.bytes]
    shape = [e.op(f"arith.index_cast {v} : index to i64", "dim")
             for v in av.sizes + av.strides]
    element = e.const(ELEM_BYTES[memref_info(av.ty)[1]], 'i32')
    arguments = original + shape + [element]
    types = '!llvm.ptr, i64, i64, i64, i64, i64, i32'
    av.ptr = e.op(f"func.call @polygeist_v2_matrix_pack({', '.join(arguments)}) : ({types}) -> !llvm.ptr", 'packed')
    c1 = e.const(1, 'index')
    nonunit = e.op(f"arith.cmpi ne, {av.strides[1]}, {c1} : index", 'nonunit')
    multiple = e.op(f"arith.cmpi ugt, {av.sizes[1]}, {c1} : index", 'multiple')
    copied = e.op(f"arith.andi {nonunit}, {multiple} : i1", 'copied')
    row = e.op(f"arith.select {copied}, {av.sizes[1]}, {av.strides[0]} : index", 'stride')
    count = e.op(f"arith.muli {shape[0]}, {shape[1]} : i64", 'count')
    size = e.const(ELEM_BYTES[memref_info(av.ty)[1]], 'i64')
    dense = e.op(f"arith.muli {count}, {size} : i64", 'dense')
    av.bytes = e.op(f"arith.select {copied}, {dense}, {av.bytes} : i64", 'bytes')
    av.strides = [row, c1]
    write = e.const(int(written), 'i32')
    return (f"func.call @polygeist_v2_matrix_release({', '.join([av.ptr] + arguments + [write])}) : "
            f"(!llvm.ptr, {types}, i32) -> ()")


def lower_launch(m: re.Match, sa: A.SpecAbi, ca: A.CallAbi, tag: str) -> tuple[list[str], str]:
    ind = m.group("ind")
    ops = split_top(m.group("ops"))
    tys = split_top(m.group("tys"))
    rtys = parse_results(m.group("rty"))
    if len(ops) != len(sa.args) or len(tys) != len(sa.args):
        raise LowerError(f"{sa.spec}: launch has {len(ops)} operands, spec has {len(sa.args)}")
    e = Emitter(ind, tag)
    arrays: dict[int, ArrayVal] = {}
    releases = []
    elem_of = {p.name: p.elem for p in ca.params if p.tkind == "array"}

    def array(i: int, pname: str) -> ArrayVal:
        if i not in arrays:
            if not tys[i].startswith("memref<"):
                raise LowerError(f"{sa.spec}: array operand {sa.args[i].name} is {tys[i]} (not bufferized)")
            spec_elem = MLIR_ELEMENT_TYPES[elem_of[pname]]
            arrays[i] = ArrayVal(e, ops[i], tys[i], spec_elem)
            if ca.descriptor_adapter or ca.runtime_adapter:
                param = next(p for p in ca.params if p.name == pname)
                if arrays[i].rank != param.rank:
                    raise LowerError(f"{sa.spec}: descriptor array rank mismatch")
        return arrays[i]

    def int_value(b) -> str:
        if b[0] in ("arg", "alias"):
            return cast_int(e, ops[b[1]], tys[b[1]])
        if b[0] == "const_num":
            return e.const(int(b[1]), "i32")
        raise LowerError(f"{sa.spec}: cannot bind int from {b}")

    # Adapt only the ordinary BLAS layout contract. Typed/descriptor adapters
    # keep their own original extent/alias rules and are never repacked here.
    packed_indices = set()
    if not ca.descriptor_adapter and not ca.runtime_adapter:
        for p in ca.params:
            if p.tkind == 'int' and p.layout_of:
                i = sa.binding[p.layout_of][1]
                if matrix_needs_layout_adapter(tys[i]):
                    packed_indices.add(i)
    if packed_indices:
        # Resolve every original descriptor before allocating any copy, so
        # packing cannot hide original operand overlap from the ABI guards.
        for p in ca.params:
            if p.tkind == 'array':
                array(sa.binding[p.name][1], p.name)
        for i in sorted(packed_indices):
            av = arrays[i]
            c1 = e.const(1, 'index')
            nonunit = e.op(f"arith.cmpi ne, {av.strides[1]}, {c1} : index", 'nonunit')
            multiple = e.op(f"arith.cmpi ugt, {av.sizes[1]}, {c1} : index", 'multiple')
            needed = e.op(f"arith.andi {nonunit}, {multiple} : i1", 'pack_needed')
            flag = e.op(f"arith.extui {needed} : i1 to i32", 'pack_flag')
            for j, other in arrays.items():
                if i != j:
                    e.raw(f"func.call @polygeist_v2_matrix_require_disjoint({av.ptr}, {av.bytes}, "
                          f"{other.ptr}, {other.bytes}, {flag}) : (!llvm.ptr, i64, !llvm.ptr, i64, i32) -> ()")
        for i in sorted(packed_indices):
            releases.append(pack_matrix(e, arrays[i], sa.args[i].intent in ('out', 'inout')))
    args, atys = [], []
    for p in ca.params:
        b = sa.binding[p.name]
        if p.tkind == "enum":
            if b[0] != "const_enum":
                raise LowerError(f"{sa.spec}: enum {p.name} must be a fixed constant")
            args.append(e.const(b[1], "i32"))
            atys.append("i32")
        elif p.tkind == "scalar":
            scalar_ty = {"float": "f32", "double": "f64"}[p.elem]
            if b[0] in ("arg", "alias"):
                args.append(cast_scalar(e, ops[b[1]], tys[b[1]], scalar_ty))
            elif b[0] == "const_num":
                args.append(e.const(float(b[1]), scalar_ty))
            else:
                raise LowerError(f"{sa.spec}: scalar {p.name} unbound")
            atys.append(scalar_ty)
        elif p.tkind == "array":
            if b[0] not in ("arg", "alias"):
                raise LowerError(f"{sa.spec}: array {p.name} unbound")
            av = array(b[1], p.name)
            args += [av.ptr, av.bytes]
            atys += ["!llvm.ptr", "i64"]
        elif p.tkind == "int" and p.layout_of:
            ab = sa.binding[p.layout_of]
            av = array(ab[1], p.layout_of)
            spec_inc = None
            if av.rank == 1:
                # the spec's own increment (strided variants / fixed 1) scales
                # the memref stride; for rank 2 the operand is superseded
                spec_inc = int_value(b)
            args.append(av.layout(e, spec_inc, f"{sa.spec}.{p.name}"))
            atys.append("i32")
        elif p.tkind == "int":
            args.append(int_value(b))
            atys.append("i32")
        else:
            raise LowerError(f"{sa.spec}: unsupported ABI kind {p.tkind}")
    callee = f"polygeist_v2_{ca.call}"
    res = m.group("res")
    ret = ca.returns
    ret_ty = {"float": "f32", "double": "f64"}.get(ret)
    # bufferized launches have no results; the only SSA result a library call
    # can produce is the scalar of a value-returning spec
    if rtys and not (ret and rtys == [ret_ty]):
        raise LowerError(f"{sa.spec}: unlowerable launch results {rtys} (not bufferized?)")
    call = f"func.call @{callee}({', '.join(args)}) : ({', '.join(atys)})"
    if ret and rtys and res:
        e.raw(f"{res} = {call} -> {ret_ty}")
    elif ret:
        e.raw(f"{e.fresh('r')} = {call} -> {ret_ty}")
    else:
        e.raw(f"{call} -> ()")
    for release in reversed(releases):
        e.raw(release)
    decl = f"  func.func private @{callee}({', '.join(atys)})" + (f" -> {ret_ty}" if ret else "")
    return e.lines, decl


def rewrite(text: str, names: dict, specs: dict, calls: dict, report: list,
            backend: str = "cpu") -> str:
    out, decls = [], {}
    for n, ln in enumerate(text.split("\n")):
        m = LAUNCH_RE.match(ln)
        if not m:
            out.append(ln)
            continue
        name = m.group("sym")
        spec = names.get(name, name)
        if spec not in specs:
            raise LowerError(f"launch of {name}: not a lib_v2 spec with an ABI")
        sa = specs[spec]
        ca = calls[sa.call]
        lines, decl = lower_launch(m, sa, ca, str(n))
        if backend == "cuda":
            # The CUDA ABI accepts resident device pointers. This is not a
            # graph-capture promise: a wrapper may allocate or synchronize.
            decl += " attributes {polygeist.gpu_runtime}"
        prev = decls.setdefault(ca.call, decl)
        if prev != decl:
            raise LowerError(f"inconsistent declarations for {ca.call}")
        out.extend(lines)
        report.append({"spec": spec, "call": f"polygeist_v2_{ca.call}",
                       "cpu_entry": ca.fortran["entry"] if ca.fortran else None,
                       "gpu_entry": ca.header_symbol,
                       "operand_types": split_top(m.group("tys"))})
    text = strip_defns("\n".join(out))
    if 'func.call @polygeist_v2_matrix_pack(' in text:
        decls['matrix_disjoint'] = ('  func.func private @polygeist_v2_matrix_require_disjoint('
            '!llvm.ptr, i64, !llvm.ptr, i64, i32)')
        decls['matrix_pack'] = ('  func.func private @polygeist_v2_matrix_pack('
            '!llvm.ptr, i64, i64, i64, i64, i64, i32) -> !llvm.ptr')
        decls['matrix_release'] = ('  func.func private @polygeist_v2_matrix_release('
            '!llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i32, i32)')
    if decls:
        text = insert_after_module_open(text, "\n".join(decls.values()) + "\n")
    if "kernel.launch" in text:
        raise LowerError("kernel.launch left after lowering")
    return text


COPY_RE = re.compile(r'^(?P<ind>\s*)memref\.copy (?P<src>%[\w$.\-]+), (?P<dst>%[\w$.\-]+) : (?P<sty>memref<.*>) to (?P<dty>memref<.*>)\s*$')


def rewrite_memory_copies(text: str, backend: str) -> str:
    """CUDA operands may be device pointers; residual memory movement must
    preserve that address space. CPU/PVA emission is unchanged."""
    if backend != 'cuda':
        return text
    lines, used = [], False
    for n, line in enumerate(text.split('\n')):
        m = COPY_RE.match(line)
        if not m:
            lines.append(line)
            continue
        if m['src'] == m['dst'] and m['sty'] == m['dty']:
            continue  # Exact SSA/type identity is a no-op, including empty views.
        rank, elem = memref_info(m['sty'])
        if memref_info(m['dty']) != (rank, elem):
            raise LowerError('memory copy rank/element mismatch')
        e = Emitter(m['ind'], 'copy' + str(n))
        src = ArrayVal(e, m['src'], m['sty'], elem)
        dst = ArrayVal(e, m['dst'], m['dty'], elem)
        for a, b in zip(src.sizes, dst.sizes):
            ok = e.op(f'arith.cmpi eq, {a}, {b} : index', 'shape')
            e.raw(f'cf.assert {ok}, "lib_v2 memory copy: shape mismatch"')
        prefix, e.lines = e.lines, []
        e.ind += '  '
        def metadata(values):
            ty = f'memref<{max(rank, 1)}xi64>'
            storage = e.op(f'memref.alloca() : {ty}', 'meta')
            for i, value in enumerate(values):
                at = e.const(i, 'index')
                val = e.op(f'arith.index_cast {value} : index to i64', 'dim')
                e.raw(f'memref.store {val}, {storage}[{at}] : {ty}')
            address = e.op(f'memref.extract_aligned_pointer_as_index {storage} : {ty} -> index', 'address')
            integer = e.op(f'arith.index_cast {address} : index to i64', 'addr')
            return e.op(f'llvm.inttoptr {integer} : i64 to !llvm.ptr', 'ptr')
        pointers = [metadata(src.sizes), metadata(src.strides), metadata(dst.strides)]
        args = [e.const(rank, 'i64'), e.const(ELEM_BYTES[elem], 'i64'), *pointers, src.ptr, dst.ptr]
        types = ['i64', 'i64'] + ['!llvm.ptr'] * 5
        e.raw(f"func.call @polygeist_v2_memref_copy({', '.join(args)}) : ({', '.join(types)}) -> ()")
        lines.extend(prefix)
        lines.append(m['ind'] + 'memref.alloca_scope {')
        lines.extend(e.lines)
        lines.append(m['ind'] + '  memref.alloca_scope.return')
        lines.append(m['ind'] + '}')
        used = True
    output = '\n'.join(lines)
    if used and 'func.func private @polygeist_v2_memref_copy(' not in output:
        output = insert_after_module_open(output,
            '  func.func private @polygeist_v2_memref_copy(i64, i64, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr)\n')
    return output

def lower(text: str, do_bufferize: bool = True, report: dict | None = None,
          allow_legaliser: bool = True, backend: str = "cpu",
          deallocate: bool = False) -> str:
    report = report if report is not None else {}
    report.setdefault("launches", [])
    report.setdefault("opt", [])
    used = sorted({m.group("sym") for m in map(LAUNCH_RE.match, text.split("\n")) if m})
    if not used:
        report["note"] = "no kernel.launch"
        text = strip_defns(text)
        if do_bufferize:
            text = bufferize(re.sub(r"bufferization\.to_tensor (%[^ ]*) :",
                                    r"bufferization.to_tensor \1 restrict writable :", text),
                             report["opt"], allow_legaliser)
        if deallocate:
            text = run_opt(["--lower-affine", "--buffer-deallocation"], text, report["opt"])
        return text
    calls, specs, errors = A.build_table()
    bad = [s for s in used if s not in specs]
    if bad:
        raise LowerError("no ABI for " + ", ".join(f"{s} ({errors.get(s, 'not a lib_v2 spec')})" for s in bad))
    text, names = prepare(text, specs, calls)
    if do_bufferize:
        text = bufferize(text, report["opt"], allow_legaliser)
    # Resolve lifetimes before rewriting launches to raw-pointer calls. After
    # that rewrite memref liveness cannot see the call's use of the buffer.
    if deallocate:
        text = run_opt(["--lower-affine", "--buffer-deallocation"], text, report["opt"])
    return rewrite(text, names, specs, calls, report["launches"], backend)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--memory-copies-only", action="store_true",
                    help="lower residual copies after higher-level GPU transforms")
    ap.add_argument("--no-bufferize", action="store_true",
                    help="input launches are already memref-form")
    ap.add_argument("--deallocate", action="store_true",
                    help="resolve buffer lifetimes before emitting raw-pointer calls")
    ap.add_argument("--no-legaliser", action="store_true",
                    help="fail instead of falling back to legalize_tensor_submaps.py "
                         "when production view lowering leaves tensor submaps")
    ap.add_argument("--report")
    ap.add_argument("--backend", choices=("cpu", "cuda", "pva"), default="cpu",
                    help="runtime ABI backend; CUDA declarations accept device pointers")
    a = ap.parse_args()
    report: dict = {}
    try:
        if a.memory_copies_only:
            out = rewrite_memory_copies(Path(a.input).read_text(), a.backend)
        else:
            out = lower(Path(a.input).read_text(), not a.no_bufferize, report,
                        allow_legaliser=not a.no_legaliser, backend=a.backend,
                        deallocate=a.deallocate)
    except LowerError as ex:
        print(f"lower_v2: error: {ex}", file=sys.stderr)
        if a.report:
            report["error"] = str(ex)
            Path(a.report).write_text(json.dumps(report, indent=1))
        return 1
    Path(a.output).write_text(out)
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1))
    calls = [l["call"] for l in report.get("launches", [])]
    leg = any("legalize" in str(o.get("cmd", "")) for o in report.get("opt", []))
    print(f"lower_v2: {len(calls)} launch(es) -> {', '.join(calls) or '(none)'}"
          + ("  [legaliser fallback used]" if leg else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
