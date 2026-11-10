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
import abi_v2 as A  # noqa: E402
import recipe as R  # noqa: E402

LAUNCH_RE = re.compile(
    r"^(?P<ind>\s*)(?:(?P<res>%[\w$.\-]+)(?::(?P<nres>\d+))?\s*=\s*)?"
    r"kernel\.launch\s+@(?P<sym>[\w$.\-]+)\((?P<ops>[^)]*)\)\s*"
    r"(?P<attrs>\{.*?\})?\s*:\s*\((?P<tys>.*)\)\s*->\s*(?P<rty>.*?)\s*$")

ELEM_BYTES = {"f64": 8, "f32": 4, "i32": 4, "i64": 8}


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

def prepare(text: str, specs: dict) -> tuple[str, dict]:
    """-> (text, defn name -> spec)"""
    text = strip_defns(text)
    sigs: dict[tuple, str] = {}
    lines = text.split("\n")
    for i, ln in enumerate(lines):
        m = LAUNCH_RE.match(ln)
        if not m or not m.group("sym").startswith(("netlib_", "openblas_")):
            continue
        sym = m.group("sym")
        if sym not in specs:
            raise LowerError(f"{sym}: no ABI (manifest/spec/header binding failed)")
        tys = tuple(split_top(m.group("tys")))
        rtys = tuple(parse_results(m.group("rty")))
        key = (sym, tys, rtys)
        if key not in sigs:
            n = sum(1 for k in sigs if k[0] == sym)
            sigs[key] = sym if n == 0 else f"{sym}__v2sig{n}"
        name = sigs[key]
        if name != sym:
            lines[i] = ln.replace(f"@{sym}(", f"@{name}(", 1)
    stubs, names = [], {}
    for (sym, tys, rtys), name in sigs.items():
        sa: A.SpecAbi = specs[sym]
        written = [k for k, a in enumerate(sa.args) if a.kind == "array" and a.intent in ("out", "inout")]
        args = ", ".join(f"%a{k}: {t}" for k, t in enumerate(tys))
        ret = f" -> ({', '.join(rtys)})" if len(rtys) > 1 else (f" -> {rtys[0]}" if rtys else "")
        if sa.ret != "void":
            # value-returning spec (e.g. a dot product): the only result is
            # the scalar; destinations cannot be expressed for tensor operands
            if any(t.startswith("tensor<") for t in tys) or len(rtys) > 1:
                raise LowerError(f"{sym}: value-returning spec with tensor operands is not "
                                 "supported (kernel.launch bufferization needs destinations)")
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


def cast_scalar(e: Emitter, v: str, ty: str) -> str:
    if ty == "f64":
        return v
    if ty in ("f32", "f16", "bf16"):
        return e.op(f"arith.extf {v} : {ty} to f64", "f")
    raise LowerError(f"cannot pass {ty} as double scalar")


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


def lower_launch(m: re.Match, sa: A.SpecAbi, ca: A.CallAbi, tag: str) -> tuple[list[str], str]:
    ind = m.group("ind")
    ops = split_top(m.group("ops"))
    tys = split_top(m.group("tys"))
    rtys = parse_results(m.group("rty"))
    if len(ops) != len(sa.args) or len(tys) != len(sa.args):
        raise LowerError(f"{sa.spec}: launch has {len(ops)} operands, spec has {len(sa.args)}")
    e = Emitter(ind, tag)
    arrays: dict[int, ArrayVal] = {}
    elem_of = {p.name: p.elem for p in ca.params if p.tkind == "array"}

    def array(i: int, pname: str) -> ArrayVal:
        if i not in arrays:
            if not tys[i].startswith("memref<"):
                raise LowerError(f"{sa.spec}: array operand {sa.args[i].name} is {tys[i]} (not bufferized)")
            spec_elem = {"double": "f64", "float": "f32"}[elem_of[pname]]
            arrays[i] = ArrayVal(e, ops[i], tys[i], spec_elem)
        return arrays[i]

    def int_value(b) -> str:
        if b[0] in ("arg", "alias"):
            return cast_int(e, ops[b[1]], tys[b[1]])
        if b[0] == "const_num":
            return e.const(int(b[1]), "i32")
        raise LowerError(f"{sa.spec}: cannot bind int from {b}")

    args, atys = [], []
    for p in ca.params:
        b = sa.binding[p.name]
        if p.tkind == "enum":
            if b[0] != "const_enum":
                raise LowerError(f"{sa.spec}: enum {p.name} must be a fixed constant")
            args.append(e.const(b[1], "i32"))
            atys.append("i32")
        elif p.tkind == "scalar":
            if b[0] in ("arg", "alias"):
                args.append(cast_scalar(e, ops[b[1]], tys[b[1]]))
            elif b[0] == "const_num":
                args.append(e.const(float(b[1]), "f64"))
            else:
                raise LowerError(f"{sa.spec}: scalar {p.name} unbound")
            atys.append("f64")
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
    # bufferized launches have no results; the only SSA result a library call
    # can produce is the scalar of a value-returning spec
    if rtys and not (ret and rtys == ["f64"]):
        raise LowerError(f"{sa.spec}: unlowerable launch results {rtys} (not bufferized?)")
    call = f"func.call @{callee}({', '.join(args)}) : ({', '.join(atys)})"
    if ret and rtys and res:
        e.raw(f"{res} = {call} -> f64")
    elif ret:
        e.raw(f"{e.fresh('r')} = {call} -> f64")
    else:
        e.raw(f"{call} -> ()")
    decl = f"  func.func private @{callee}({', '.join(atys)})" + (" -> f64" if ret else "")
    return e.lines, decl


def rewrite(text: str, names: dict, specs: dict, calls: dict, report: list) -> str:
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
        prev = decls.setdefault(ca.call, decl)
        if prev != decl:
            raise LowerError(f"inconsistent declarations for {ca.call}")
        out.extend(lines)
        report.append({"spec": spec, "call": f"polygeist_v2_{ca.call}",
                       "cpu_entry": ca.fortran["entry"], "gpu_entry": ca.header_symbol,
                       "operand_types": split_top(m.group("tys"))})
    text = strip_defns("\n".join(out))
    if decls:
        text = insert_after_module_open(text, "\n".join(decls.values()) + "\n")
    if "kernel.launch" in text:
        raise LowerError("kernel.launch left after lowering")
    return text


def lower(text: str, do_bufferize: bool = True, report: dict | None = None,
          allow_legaliser: bool = True) -> str:
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
        return text
    calls, specs, errors = A.build_table()
    bad = [s for s in used if s not in specs]
    if bad:
        raise LowerError("no ABI for " + ", ".join(f"{s} ({errors.get(s, 'not a lib_v2 spec')})" for s in bad))
    text, names = prepare(text, specs)
    if do_bufferize:
        text = bufferize(text, report["opt"], allow_legaliser)
    return rewrite(text, names, specs, calls, report["launches"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--no-bufferize", action="store_true",
                    help="input launches are already memref-form")
    ap.add_argument("--no-legaliser", action="store_true",
                    help="fail instead of falling back to legalize_tensor_submaps.py "
                         "when production view lowering leaves tensor submaps")
    ap.add_argument("--report")
    a = ap.parse_args()
    report: dict = {}
    try:
        out = lower(Path(a.input).read_text(), not a.no_bufferize, report,
                    allow_legaliser=not a.no_legaliser)
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
