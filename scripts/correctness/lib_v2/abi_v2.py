#!/usr/bin/env python3
"""abi_v2.py -- derive the library ABI of every lib_v2 spec automatically.

Nothing in this file names a BLAS/LAPACK routine.  Every fact is read from an
artifact that already exists for another reason:

  manifest/<routine>.json    target.call, fixed_args          (lib_v2 contract)
  raised/<spec>.c            spec C prototype (after the C preprocessor):
                             argument kinds, VLA leading-dimension links
  raised/<spec>.mlir         cgeist stage of the spec: which array arguments
                             are read / written (argument intents)
  third_party/cuda_headers   cuBLAS / cuSOLVER prototypes and enum values
                             (target parameter order and types)
  netlib Fortran sources     Fortran entry-point prototype used by the CPU
                             runtime (BLAS/SRC/<x>.f, SRC/<x>.f)

Binding rules (generic):
  * spec arg -> target param: fixed_args first (a fixed value may be a number,
    an enum constant from the headers, or the name of another spec argument =
    alias); then case-insensitive name equality; then, for the parameters still
    unbound, order-preserving positional pairing with a kind check.  The spec
    arguments are netlib names in netlib order (up to the VLA lda-before-A
    swap, which the name pass resolves), so the positional pass only pairs
    renamed level-1 arguments (e.g. Fortran DA/DX vs cuBLAS alpha/x).
  * target param -> Fortran param: the same name-then-position rule.
  * an int parameter that immediately follows an array parameter in the target
    order is that array's layout parameter (leading dimension for rank 2,
    increment for rank 1).  For rank-2 arrays this is cross-checked against the
    spec's VLA declaration `double A[][lda]`.  The lowering computes these
    values from memref strides.
  * target parameters bound by no spec are "runtime managed" and classified by
    library ABI convention: cuSOLVER workspace (`Workspace` + `Lwork` with a
    `<call>_bufferSize` companion), status output (`devInfo`/`info`), and the
    result pointer of a value-returning spec.
  * enum constant -> Fortran CHARACTER: first letter after the enum's common
    prefix (CUBLAS_OP_T -> 'T', CUBLAS_DIAG_NON_UNIT -> 'N'); cross-checked
    against each spec's manifest `defines` where the flag names coincide.

Usage: abi_v2.py [--json OUT.json]   (prints/dumps the derived table)
"""
from __future__ import annotations

import argparse
import functools
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SPEC_DIR = REPO / "generic_solver" / os.environ.get("POLYGEIST_SPEC_SUITE", "lib_v3")
CUDA_HEADERS = REPO / "third_party/cuda_headers"
NETLIB = Path(os.environ.get("NETLIB_REF", "/home/arjaiswal/netlib_ref/lapack"))
CLANG = os.environ.get("CLANG", str(REPO / "llvm-project/build/bin/clang"))

# Library header families, keyed by the target.library field of the manifest.
HEADER_FAMILIES = {
    "cuBLAS": ["cublas/include/cublas_api.h", "cublas/include/cublas_v2.h"],
    "cuSOLVER": ["cublas/include/cublas_api.h", "cusolver/include/cusolverDn.h"],
    "cuDNN": ["cudnn/include/cudnn_graph.h", "cudnn/include/cudnn_ops.h"],
    "cuTENSOR": ["cutensor/include/cutensor.h", "cutensor/include/cutensor/types.h"],
}


class AbiError(Exception):
    pass


# ---------------------------------------------------------------------------
# C header parsing (cuBLAS / cuSOLVER)
# ---------------------------------------------------------------------------

@dataclass
class CParam:
    ctype: str          # e.g. "const double*", "int", "cublasOperation_t"
    name: str

    @property
    def base(self) -> str:
        return re.sub(r"\bconst\b|\*", "", self.ctype).strip()

    @property
    def is_ptr(self) -> bool:
        return "*" in self.ctype


@dataclass
class CProto:
    name: str
    ret: str
    params: list[CParam]


def _strip_c_comments(t: str) -> str:
    t = re.sub(r"/\*.*?\*/", " ", t, flags=re.S)
    return re.sub(r"//[^\n]*", " ", t)


def _split_top(s: str, sep: str = ",") -> list[str]:
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
    if "".join(cur).strip():
        out.append("".join(cur))
    return [x.strip() for x in out]


def _parse_c_param(p: str) -> Optional[CParam]:
    p = re.sub(r"\s+", " ", p).strip()
    arr = p.endswith("]")
    p = re.sub(r"\s*\[[^\]]*\]\s*$", "", p)
    m = re.fullmatch(r"(.*?)([A-Za-z_]\w*)$", p)
    if not m:
        return None
    if arr:
        p = m.group(1) + "*" + m.group(2)
        m = re.fullmatch(r"(.*?)([A-Za-z_]\w*)$", p)
    ctype = m.group(1).replace(" *", "*").strip()
    ctype = re.sub(r"\s*\*\s*", "*", ctype)
    return CParam(ctype, m.group(2))


@functools.lru_cache(maxsize=None)
def header_info(library: str):
    """-> (prototypes, macro aliases, enums{type: {const: value}})"""
    if library not in HEADER_FAMILIES:
        raise AbiError(f"no header family for library {library!r}")
    protos: dict[str, CProto] = {}
    aliases: dict[str, str] = {}
    enums: dict[str, dict[str, int]] = {}
    for rel in HEADER_FAMILIES[library]:
        path = CUDA_HEADERS / rel
        raw = path.read_text(errors="replace")
        for m in re.finditer(r"^\s*#\s*define\s+(\w+)\s+(\w+)\s*$", raw, flags=re.M):
            aliases[m.group(1)] = m.group(2)
        text = _strip_c_comments(raw)
        text = re.sub(r"^\s*#[^\n]*$", " ", text, flags=re.M)
        for m in re.finditer(r"typedef\s+enum\s*\w*\s*\{([^}]*)\}\s*(\w+)\s*;", text):
            vals, nxt = {}, 0
            for item in _split_top(m.group(1)):
                if not item:
                    continue
                if "=" in item:
                    k, v = (x.strip() for x in item.split("=", 1))
                    try:
                        nxt = int(v, 0)
                    except ValueError:
                        if v in vals:
                            nxt = vals[v]
                        else:
                            continue
                else:
                    k = item.strip()
                vals[k] = nxt
                nxt += 1
            enums[m.group(2)] = vals
        flat = re.sub(r"\s+", " ", text)
        for m in re.finditer(r"\b(\w+(?:_t|Status_t))\s+(?:(?:CUBLASWINAPI|CUSOLVERAPI|CUDNNWINAPI)\s+)?(\w+)\s*\(([^;{}]*?)\)\s*;", flat):
            ret, name, plist = m.groups()
            params = [_parse_c_param(p) for p in _split_top(plist) if p and p != "void"]
            if all(params):
                protos[name] = CProto(name, ret, params)
    return protos, aliases, enums


def resolve_proto(library: str, call: str) -> CProto:
    protos, aliases, _ = header_info(library)
    name, seen = call, set()
    while name not in protos and name in aliases and name not in seen:
        seen.add(name)
        name = aliases[name]
    if name not in protos:
        raise AbiError(f"{library} header has no prototype for {call}")
    return protos[name]


def enum_of_constant(library: str, const: str) -> Optional[tuple[str, int]]:
    _, _, enums = header_info(library)
    for ty, vals in enums.items():
        if const in vals:
            return ty, vals[const]
    return None


def enum_fortran_char(library: str, enum_type: str, const: str) -> str:
    """CUBLAS_OP_T -> 'T': first letter after the enum's common prefix."""
    _, _, enums = header_info(library)
    names = list(enums[enum_type])
    pref = os.path.commonprefix(names)
    pref = pref[:pref.rfind("_") + 1] if "_" in pref else pref
    return const[len(pref)]


# ---------------------------------------------------------------------------
# Fortran (netlib) prototypes, CPU entry points
# ---------------------------------------------------------------------------

@dataclass
class FParam:
    name: str           # upper case
    kind: str           # "char" | "int" | "scalar" | "array"
    ftype: str          # CHARACTER / INTEGER / DOUBLE PRECISION


@dataclass
class FProto:
    entry: str          # C symbol, e.g. dgemm_
    source: str
    is_function: bool
    ret: Optional[str]
    params: list[FParam]


def cpu_entry_name(call: str) -> str:
    """Strip the vendor prefix (lower-case run + optional 'Dn'/'Sp' style
    module tag) of a cuBLAS/cuSOLVER symbol: cublasDgemm -> dgemm,
    cusolverDnDpotrf -> dpotrf."""
    m = re.fullmatch(r"[a-z]+(?:[A-Z][a-z](?=[A-Z]))?([A-Z]\w*)", call)
    if not m:
        raise AbiError(f"cannot derive a Fortran entry from {call}")
    return m.group(1).lower()


@functools.lru_cache(maxsize=None)
def fortran_proto(routine: str) -> FProto:
    cands = [NETLIB / "BLAS/SRC" / f"{routine}.f", NETLIB / "SRC" / f"{routine}.f"]
    path = next((p for p in cands if p.exists()), None)
    if path is None:
        raise AbiError(f"no netlib Fortran source for {routine} under {NETLIB}")
    lines = path.read_text(errors="replace").splitlines()
    # join fixed-form continuation lines, drop comment lines
    stmts: list[str] = []
    for ln in lines:
        if not ln or ln[0] in "*cC!":
            continue
        if len(ln) > 5 and ln[5] not in " 0" and stmts:
            stmts[-1] += ln[6:]
        else:
            stmts.append(ln[6:] if len(ln) > 6 else "")
    head = None
    for i, s in enumerate(stmts):
        m = re.match(r"\s*(?:(DOUBLE PRECISION|INTEGER|REAL)\s+)?(SUBROUTINE|FUNCTION)\s+(\w+)\s*\(([^)]*)\)", s, re.I)
        if m and m.group(3).lower() == routine:
            head = (i, m)
            break
    if head is None:
        raise AbiError(f"no SUBROUTINE/FUNCTION {routine} in {path}")
    i0, m = head
    names = [x.strip().upper() for x in m.group(4).split(",") if x.strip()]
    decl: dict[str, FParam] = {}
    for s in stmts[i0 + 1:]:
        d = re.match(r"\s*(CHARACTER|INTEGER|DOUBLE PRECISION|REAL)\b\s*(.*)$", s, re.I)
        if not d:
            continue
        ftype = d.group(1).upper()
        for item in _split_top(d.group(2)):
            im = re.match(r"(\w+)\s*(\(.*\))?", item)
            if not im:
                continue
            nm = im.group(1).upper()
            if nm not in names or nm in decl:
                continue
            kind = {"CHARACTER": "char", "INTEGER": "int"}.get(ftype)
            if kind is None:
                kind = "array" if im.group(2) else "scalar"
            elif im.group(2):
                kind = "int_array"
            decl[nm] = FParam(nm, kind, ftype)
    missing = [n for n in names if n not in decl]
    if missing:
        raise AbiError(f"{path}: undeclared dummy arguments {missing}")
    is_fn = m.group(2).upper() == "FUNCTION"
    return FProto(f"{routine}_", str(path), is_fn,
                  (m.group(1) or "").upper() or None if is_fn else None,
                  [decl[n] for n in names])


# ---------------------------------------------------------------------------
# Spec side: C prototype (kinds) and cgeist stage (intents)
# ---------------------------------------------------------------------------

@dataclass
class SpecArg:
    name: str
    kind: str                    # "int" | "scalar" | "array"
    rank: int = 0
    vla_ld: Optional[str] = None  # `double A[][lda]` -> "lda"
    intent: Optional[str] = None  # arrays: "in" | "out" | "inout"
    ctype: Optional[str] = None


def _manifest_entries() -> dict[str, tuple[dict, dict]]:
    out = {}
    for mp in sorted(glob.glob(str(SPEC_DIR / "manifest" / "*.json"))):
        man = json.load(open(mp))
        for e in man.get("specs", []):
            out[e["name"]] = (man, e)
    return out


MANIFEST = None


def manifest_entry(spec: str):
    global MANIFEST
    if MANIFEST is None:
        MANIFEST = _manifest_entries()
    if spec not in MANIFEST:
        raise AbiError(f"no manifest entry for {spec}")
    return MANIFEST[spec]


@functools.lru_cache(maxsize=None)
def spec_c_prototype(spec: str) -> tuple[str, list[SpecArg]]:
    """Preprocess raised/<spec>.c and parse the definition of <spec>."""
    src = SPEC_DIR / "raised" / f"{spec}.c"
    if not src.exists():
        raise AbiError(f"missing spec wrapper {src}")
    p = subprocess.run([CLANG, "-E", "-P", str(src)], capture_output=True, text=True)
    if p.returncode:
        raise AbiError(f"preprocessing {src} failed: {p.stderr[-400:]}")
    text = re.sub(r"\s+", " ", p.stdout)
    m = re.search(rf"\b(void|double|float|int)\s+{re.escape(spec)}\s*\(([^)]*)\)\s*\{{", text)
    if not m:
        raise AbiError(f"no definition of {spec} in preprocessed {src}")
    args = []
    for prm in _split_top(m.group(2)):
        pm = re.fullmatch(r"(?:const\s+)?(int|long|double|float)\s+(\w+)\s*((?:\[[^\]]*\])*)", prm.strip())
        if not pm:
            raise AbiError(f"{spec}: cannot classify parameter {prm!r}")
        ty, nm, dims = pm.groups()
        dl = re.findall(r"\[([^\]]*)\]", dims)
        if dl:
            ld = dl[-1].strip() if len(dl) >= 2 and re.fullmatch(r"\w+", dl[-1].strip()) else None
            args.append(SpecArg(nm, "array", len(dl), ld, ctype=ty))
        else:
            args.append(SpecArg(nm, "int" if ty in ("int", "long") else "scalar", ctype=ty))
    return m.group(1), args


_VIEW_OPS = ("polygeist.subindex", "memref.subview", "memref.cast",
             "memref.reinterpret_cast", "memref.collapse_shape",
             "memref.expand_shape", "polygeist.submap")


@functools.lru_cache(maxsize=None)
def spec_intents(spec: str) -> dict[int, str]:
    """arg position -> 'in'|'out'|'inout' for memref arguments of the cgeist
    stage (the faithful semantics of the C transliteration)."""
    path = SPEC_DIR / "raised" / f"{spec}.mlir"
    if not path.exists():
        raise AbiError(f"missing cgeist stage {path}")
    text = path.read_text()
    fm = re.search(rf"func\.func @{re.escape(spec)}\((.*?)\)\s*(?:->[^{{]*)?(?:attributes[^{{]*)?\{{", text)
    if not fm:
        raise AbiError(f"no func @{spec} in {path}")
    root: dict[str, int] = {}
    for k, a in enumerate(_split_top(fm.group(1))):
        am = re.match(r"(%[\w$.\-]+)\s*:\s*memref", a)
        if am:
            root[am.group(1)] = k
    body = text[fm.end():]
    reads: set[int] = set()
    writes: set[int] = set()
    for ln in body.splitlines():
        vm = re.match(r"\s*(%[\w$.\-]+)(?::\d+)?\s*=\s*\"?([\w.]+)\"?\(?\s*(%[\w$.\-]+)", ln)
        if vm and vm.group(2) in _VIEW_OPS and vm.group(3) in root:
            root[vm.group(1)] = root[vm.group(3)]
            continue
        sm = re.search(r"\b(?:affine|memref)\.store\s+%[\w$.\-#]+\s*,\s*(%[\w$.\-]+)\s*\[", ln)
        if sm and sm.group(1) in root:
            writes.add(root[sm.group(1)])
        for lm in re.finditer(r"\b(?:affine|memref)\.load\s+(%[\w$.\-]+)\s*\[", ln):
            if lm.group(1) in root:
                reads.add(root[lm.group(1)])
        if "memref.copy" in ln:
            cm = re.search(r"memref\.copy\s+(%[\w$.\-]+)\s*,\s*(%[\w$.\-]+)", ln)
            if cm:
                if cm.group(1) in root:
                    reads.add(root[cm.group(1)])
                if cm.group(2) in root:
                    writes.add(root[cm.group(2)])
    out = {}
    for k in set(root.values()):
        out[k] = ("inout" if k in reads and k in writes else
                  "out" if k in writes else "in")
    return out


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------

def _pair(left: list, right: list, lname, rname, compat) -> tuple[dict, list, list]:
    """Name-then-position pairing. Returns ({left_idx: right_idx}, unpaired
    left indices, unpaired right indices)."""
    pairs: dict[int, int] = {}
    used_r: set[int] = set()
    for i, l in enumerate(left):
        for j, r in enumerate(right):
            if j not in used_r and lname(l).lower() == rname(r).lower():
                if not compat(l, r):
                    raise AbiError(f"name match {lname(l)}/{rname(r)} with incompatible kinds")
                pairs[i] = j
                used_r.add(j)
                break
    rest_l = [i for i in range(len(left)) if i not in pairs]
    rest_r = [j for j in range(len(right)) if j not in used_r]
    # order-preserving positional pairing of the leftovers with a kind check
    jpos = 0
    still_l = []
    for i in rest_l:
        while jpos < len(rest_r) and not compat(left[i], right[rest_r[jpos]]):
            jpos += 1
        if jpos < len(rest_r):
            pairs[i] = rest_r[jpos]
            used_r.add(rest_r[jpos])
            jpos += 1
        else:
            still_l.append(i)
    return pairs, still_l, [j for j in range(len(right)) if j not in used_r]


def _target_kind(p: CParam, library: str) -> str:
    _, _, enums = header_info(library)
    if p.base in enums and not p.is_ptr:
        return "enum"
    if p.base in ("int", "int64_t") and not p.is_ptr:
        return "int"
    if p.base in ("double", "float") and p.is_ptr:
        return "fptr"
    if p.base in ("int",) and p.is_ptr:
        return "iptr"
    return "other"


@dataclass
class AbiParam:
    """One parameter of polygeist_v2_<call> (target order, handle dropped)."""
    name: str
    tkind: str                  # enum | int | scalar | array
    enum_type: Optional[str] = None
    elem: Optional[str] = None  # array element C type
    const: bool = False
    rank: Optional[int] = None  # array rank (consistent across specs)
    layout_of: Optional[str] = None   # int param: layout param of this array
    layout: Optional[str] = None      # array param: its layout int param
    written: bool = False


@dataclass
class Managed:
    name: str
    role: str                   # workspace | lwork | info | result
    ctype: str


@dataclass
class CallAbi:
    call: str
    library: str
    header_symbol: str
    params: list[AbiParam]
    managed: list[Managed]
    returns: Optional[str] = None    # "double" if a spec returns its value
    buffer_size_symbol: Optional[str] = None
    buffer_size_params: list = field(default_factory=list)
    fortran: Optional[dict] = None
    specs: list = field(default_factory=list)
    descriptor_adapter: Optional[dict] = None
    runtime_adapter: Optional[dict] = None
    backends: list = field(default_factory=lambda: ["cpu", "cuda"])


@dataclass
class SpecAbi:
    spec: str
    call: str
    ret: str
    args: list[SpecArg]
    # target param name -> ("arg", spec_arg_index) | ("const_enum", value, const)
    #                      | ("const_num", value) | ("alias", spec_arg_index)
    binding: dict
    inc_scale: dict             # 1-D array param -> spec arg index of its inc (if spec-provided)


def bind_spec(spec: str) -> SpecAbi:
    man, entry = manifest_entry(spec)
    tgt = entry.get("target") or {}
    call, library = tgt.get("call"), tgt.get("library")
    if not call or not library:
        raise AbiError(f"{spec}: manifest has no target.call/library")
    fixed = dict(tgt.get("fixed_args") or {})
    ret, args = spec_c_prototype(spec)
    if [a.name for a in args] != list(entry.get("args", [])):
        raise AbiError(f"{spec}: C prototype args {[a.name for a in args]} != manifest args {entry.get('args')}")
    intents = spec_intents(spec)
    for k, a in enumerate(args):
        if a.kind == "array":
            a.intent = intents.get(k, "in")
    if tgt.get("runtime_adapter"):
        import runtime_adapter
        return runtime_adapter.bind_spec(sys.modules[__name__], spec, tgt, ret, args)
    proto = resolve_proto(library, call)
    if tgt.get("descriptor_adapter"):
        import descriptor_abi
        return descriptor_abi.bind_spec(sys.modules[__name__], spec, tgt, ret, args, proto)
    tparams = proto.params[1:]          # drop the library handle
    binding: dict[str, tuple] = {}
    names = {a.name: k for k, a in enumerate(args)}
    free_t = []
    for p in tparams:
        if p.name in fixed:
            v = fixed[p.name]
            if isinstance(v, str) and v in names:
                binding[p.name] = ("alias", names[v])
            elif isinstance(v, str):
                e = enum_of_constant(library, v)
                if not e:
                    raise AbiError(f"{spec}: fixed {p.name}={v} is neither a spec arg nor an enum constant")
                binding[p.name] = ("const_enum", e[1], v, e[0])
            else:
                binding[p.name] = ("const_num", v)
        else:
            free_t.append(p)
    unknown_fixed = set(fixed) - {p.name for p in tparams}
    if unknown_fixed:
        raise AbiError(f"{spec}: fixed_args {sorted(unknown_fixed)} are not parameters of {proto.name}")

    def compat(a: SpecArg, p: CParam) -> bool:
        tk = _target_kind(p, library)
        return ((a.kind == "int" and tk == "int") or
                (a.kind in ("scalar", "array") and tk == "fptr"))
    pairs, rest_s, rest_t = _pair(args, free_t, lambda a: a.name, lambda p: p.name, compat)
    if rest_s:
        raise AbiError(f"{spec}: spec args {[args[i].name for i in rest_s]} have no target parameter")
    for i, j in pairs.items():
        binding[free_t[j].name] = ("arg", i)
    for j in rest_t:
        binding[free_t[j].name] = ("managed",)
    # inc params provided by the spec (strided variants): remembered so the
    # lowering multiplies them by the memref stride
    return SpecAbi(spec, call, ret, args, binding, {})


def _layout_links(tparams: list[CParam], kinds: dict[str, str]) -> dict[str, str]:
    """array param -> the int param that immediately follows it."""
    out = {}
    for a, b in zip(tparams, tparams[1:]):
        if kinds.get(a.name) == "array" and kinds.get(b.name) == "int":
            out[a.name] = b.name
    return out


def _classify_managed(proto: CProto, managed_names: list[str], returns: bool, library: str) -> list[Managed]:
    out = []
    pnames = [p.name for p in proto.params]
    for nm in managed_names:
        p = proto.params[pnames.index(nm)]
        k = pnames.index(nm)
        nxt = proto.params[k + 1].name if k + 1 < len(proto.params) else ""
        if re.fullmatch(r"l?work", nm, re.I) or nm.lower() == "lwork":
            out.append(Managed(nm, "lwork", p.ctype))
        elif p.is_ptr and nxt.lower() == "lwork":
            out.append(Managed(nm, "workspace", p.ctype))
        elif p.is_ptr and p.base == "int" and re.search(r"info", nm, re.I):
            out.append(Managed(nm, "info", p.ctype))
        elif p.is_ptr and returns and nm.lower() == "result":
            out.append(Managed(nm, "result", p.ctype))
        else:
            raise AbiError(f"{proto.name}: parameter {nm} bound by no spec and not a known runtime-managed role")
    return out


def build_table(specs: Optional[list[str]] = None) -> tuple[dict[str, CallAbi], dict[str, SpecAbi], dict[str, str]]:
    """Return (calls, specs, errors)."""
    global MANIFEST
    MANIFEST = _manifest_entries()
    names = specs if specs is not None else sorted(MANIFEST)
    # warm the preprocessor cache in parallel (one clang -E per spec)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(min(16, os.cpu_count() or 4)) as ex:
        list(ex.map(lambda s: _safe(spec_c_prototype, s), names))
    spec_abis: dict[str, SpecAbi] = {}
    errors: dict[str, str] = {}
    for s in names:
        try:
            spec_abis[s] = bind_spec(s)
        except (AbiError, OSError) as e:
            errors[s] = str(e)
    calls: dict[str, CallAbi] = {}
    by_call: dict[str, list[SpecAbi]] = {}
    for sa in spec_abis.values():
        by_call.setdefault(sa.call, []).append(sa)
    for call, sas in sorted(by_call.items()):
        try:
            calls[call] = _build_call(call, sas)
        except AbiError as e:
            for sa in sas:
                errors[sa.spec] = f"call {call}: {e}"
                spec_abis.pop(sa.spec, None)
    return calls, spec_abis, errors


def _build_call(call: str, sas: list[SpecAbi]) -> CallAbi:
    target = manifest_entry(sas[0].spec)[1]["target"]
    library = target["library"]
    if target.get("runtime_adapter"):
        import runtime_adapter
        return runtime_adapter.build_call(sys.modules[__name__], call, sas)
    if target.get("descriptor_adapter"):
        import descriptor_abi
        return descriptor_abi.build_call(sys.modules[__name__], call, sas)
    proto = resolve_proto(library, call)
    tparams = proto.params[1:]
    # kind per target param, consistent across specs
    kinds: dict[str, str] = {}
    ranks: dict[str, int] = {}
    written: dict[str, bool] = {}
    managed_names: list[str] = []
    returns = any(sa.ret != "void" for sa in sas)
    for p in tparams:
        tk = _target_kind(p, library)
        seen = set()
        for sa in sas:
            b = sa.binding[p.name]
            if b[0] == "managed":
                seen.add("managed")
            elif b[0] == "const_enum":
                seen.add("enum")
            elif b[0] == "const_num":
                seen.add("scalar" if tk == "fptr" else "int")
            else:
                a = sa.args[b[1]]
                seen.add(a.kind)
                if a.kind in ("array", "scalar") and a.ctype != p.base:
                    raise AbiError(f"{sa.spec}.{p.name}: reference {a.ctype} != vendor {p.base}")
                if a.kind == "array":
                    ranks.setdefault(p.name, a.rank)
                    if ranks[p.name] != a.rank:
                        raise AbiError(f"param {p.name}: inconsistent array rank across specs")
                    if b[0] == "arg" and a.intent in ("out", "inout"):
                        written[p.name] = True
                    if b[0] == "alias" and a.intent in ("out", "inout"):
                        written[p.name] = True
        if tk == "enum":
            seen.discard("enum")
            if seen:
                raise AbiError(f"enum param {p.name} bound to {seen}")
            kinds[p.name] = "enum"
        elif seen == {"managed"}:
            kinds[p.name] = "managed"
            managed_names.append(p.name)
        elif len(seen) != 1:
            raise AbiError(f"param {p.name}: inconsistent kinds across specs {sorted(seen)}")
        else:
            kinds[p.name] = seen.pop()
    links = _layout_links(tparams, kinds)
    # cross-check rank-2 layout links against the spec VLA declarations
    for sa in sas:
        for arr, ld in links.items():
            b = sa.binding[arr]
            if b[0] in ("arg", "alias"):
                a = sa.args[b[1]]
                lb = sa.binding[ld]
                if a.rank == 2 and lb[0] in ("arg", "alias") and sa.args[lb[1]].name != a.vla_ld:
                    raise AbiError(f"{sa.spec}: layout param {ld} of {arr} is bound to "
                                   f"{sa.args[lb[1]].name}, but the spec declares {a.name}[][{a.vla_ld}]")
    params = []
    for p in tparams:
        k = kinds[p.name]
        if k == "managed":
            continue
        ap = AbiParam(p.name, k)
        if k == "enum":
            ap.enum_type = p.base
        if k == "scalar":
            ap.elem = p.base
        if k == "array":
            ap.elem = p.base
            ap.const = p.ctype.startswith("const")
            ap.rank = ranks[p.name]
            ap.layout = links.get(p.name)
            ap.written = written.get(p.name, False)
        if k == "int":
            for arr, ld in links.items():
                if ld == p.name:
                    ap.layout_of = arr
        params.append(ap)
    managed = _classify_managed(proto, managed_names, returns, library)
    return_types = {sa.ret for sa in sas if sa.ret != "void"}
    if len(return_types) > 1:
        raise AbiError(f"{call}: inconsistent return types")
    ca = CallAbi(call, library, proto.name, params, managed,
                 returns=next(iter(return_types), None), specs=sorted(sa.spec for sa in sas))
    if any(m.role == "workspace" for m in managed):
        protos, _, _ = header_info(library)
        bs = protos.get(proto.name + "_bufferSize")
        if not bs:
            raise AbiError(f"{proto.name} has a workspace but no {proto.name}_bufferSize")
        ca.buffer_size_symbol = bs.name
        ca.buffer_size_params = [asdict(p) for p in bs.params[1:]]
    backend_sets = [set(manifest_entry(sa.spec)[1]["target"].get("backends", ["cpu", "cuda"]))
                    for sa in sas]
    if any(not b or not b <= {"cpu", "cuda"} for b in backend_sets):
        raise AbiError(f"{call}: invalid backend declaration")
    if any(b != backend_sets[0] for b in backend_sets):
        raise AbiError(f"{call}: inconsistent backend declarations")
    ca.backends = sorted(backend_sets[0])
    if "cpu" in ca.backends:
        ca.fortran = _fortran_binding(ca, sas)
    return ca


def _fortran_binding(ca: CallAbi, sas: list[SpecAbi]) -> dict:
    routine = cpu_entry_name(ca.call)
    fp = fortran_proto(routine)
    abi = ca.params

    def compat(f: FParam, a: AbiParam) -> bool:
        return ((f.kind == "char" and a.tkind == "enum") or
                (f.kind == "int" and a.tkind == "int") or
                (f.kind in ("scalar", "array") and a.tkind == f.kind and
                 a.elem == {"REAL": "float", "DOUBLE PRECISION": "double"}.get(f.ftype)))
    pairs, rest_f, rest_a = _pair(fp.params, abi, lambda f: f.name, lambda a: a.name, compat)
    args = []
    for i, f in enumerate(fp.params):
        if i in pairs:
            args.append({"fortran": f.name, "kind": f.kind, "abi": abi[pairs[i]].name})
        elif f.kind == "int" and re.fullmatch(r"INFO", f.name):
            args.append({"fortran": f.name, "kind": "info", "abi": None})
        else:
            raise AbiError(f"Fortran {fp.entry} argument {f.name} has no ABI counterpart")
    dropped = [abi[j].name for j in rest_a]
    for d in dropped:
        ap = next(a for a in abi if a.name == d)
        if ap.tkind == "array" or ap.layout_of:
            continue      # checked at run time: must alias a passed array
        raise AbiError(f"ABI param {d} of {ca.call} has no Fortran counterpart")
    # enum -> CHARACTER table, restricted to constants some manifest uses
    chars: dict[str, dict[str, str]] = {}
    for sa in sas:
        _, entry = manifest_entry(sa.spec)
        defines = {k.upper(): v for k, v in (entry.get("defines") or {}).items()}
        for pname, b in sa.binding.items():
            if b[0] != "const_enum":
                continue
            _, val, const, ety = b
            ch = enum_fortran_char(ca.library, ety, const)
            chars.setdefault(ety, {})[str(val)] = ch
            d = defines.get(pname.upper())
            if d and d.strip("'\"") != ch:
                raise AbiError(f"{sa.spec}: {const} -> '{ch}' contradicts manifest define {pname.upper()}={d}")
    return {"entry": fp.entry, "source": os.path.relpath(fp.source, NETLIB),
            "is_function": fp.is_function, "ret": fp.ret, "args": args,
            "dropped": dropped, "enum_chars": chars}


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception:  # noqa: BLE001 -- reported again by the caller
        return None


def table_json(calls, spec_abis, errors) -> dict:
    return {
        "generated_by": "scripts/correctness/lib_v2/abi_v2.py",
        "calls": {c: asdict(v) for c, v in calls.items()},
        "specs": {s: {"call": v.call, "ret": v.ret,
                      "args": [asdict(a) for a in v.args],
                      "binding": {k: list(b) for k, b in v.binding.items()}}
                  for s, v in sorted(spec_abis.items())},
        "errors": errors,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json")
    ap.add_argument("specs", nargs="*")
    a = ap.parse_args()
    calls, specs, errors = build_table(a.specs or None)
    t = table_json(calls, specs, errors)
    if a.json:
        Path(a.json).write_text(json.dumps(t, indent=1))
    for c, v in calls.items():
        sig = ", ".join(f"{p.tkind}:{p.name}" + (f"[ld={p.layout}]" if p.layout else "") +
                        ("!" if p.written else "") for p in v.params)
        print(f"{c:18s} ({sig})" + (f" managed={[m.role for m in v.managed]}" if v.managed else "")
              + f"  cpu={v.fortran['entry'] if v.fortran else 'unavailable'}")
    for s, e in errors.items():
        print(f"ERROR {s}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
