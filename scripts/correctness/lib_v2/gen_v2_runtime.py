#!/usr/bin/env python3
"""gen_v2_runtime.py -- generate the lib_v2 runtime wrappers from abi_v2.py.

Rewrites the regions between `BEGIN GENERATED ...` / `END GENERATED ...`
markers of
  runtime/polygeist_v2_rt.h        prototypes of polygeist_v2_<call>
  runtime/polygeist_v2_rt_cpu.c    Fortran BLAS/LAPACK forwarding wrappers
  runtime/polygeist_v2_rt_cuda.c   cuBLAS / cuSOLVER pass-through wrappers
The hand-written infrastructure outside the markers is left untouched.  No
routine is named here: every wrapper is produced from the derived table
(manifest target.call/fixed_args, vendor headers, netlib Fortran prototypes).

Usage: gen_v2_runtime.py [--check]   (--check: fail if files are stale)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import abi_v2 as A  # noqa: E402

RT = A.REPO / "runtime"
HDR = RT / "polygeist_v2_rt.h"
CPU = RT / "polygeist_v2_rt_cpu.c"
CUDA = RT / "polygeist_v2_rt_cuda.c"


def c_params(ca: A.CallAbi) -> list[str]:
    out = []
    for p in ca.params:
        if p.tkind in ("enum", "int"):
            out.append(f"int32_t {p.name}")
        elif p.tkind == "scalar":
            out.append(f"double {p.name}")
        elif p.tkind == "array":
            cq = "const " if not p.written else ""
            out.append(f"{cq}{p.elem} *{p.name}")
            out.append(f"int64_t {p.name}_bytes")
        else:
            raise A.AbiError(f"{ca.call}: unsupported ABI kind {p.tkind}")
    return out


def proto(ca: A.CallAbi) -> str:
    ret = ca.returns or "void"
    return f"{ret} polygeist_v2_{ca.call}({', '.join(c_params(ca))})"


def gen_header(calls) -> str:
    return "".join(proto(ca) + ";\n" for ca in calls.values())


# ---------------------------------------------------------------------------
# CPU (Fortran)
# ---------------------------------------------------------------------------

def gen_cpu(calls) -> str:
    out = []
    for ca in calls.values():
        f = ca.fortran
        abi = {p.name: p for p in ca.params}
        fargs, decl, nchar = [], [], 0
        for a in f["args"]:
            if a["kind"] == "char":
                fargs.append("const char *")
                nchar += 1
            elif a["kind"] in ("int", "info"):
                fargs.append("const int *" if a["kind"] == "int" else "int *")
            elif a["kind"] == "scalar":
                fargs.append("const double *")
            elif a["kind"] == "array":
                p = abi[a["abi"]]
                fargs.append(("const " if not p.written else "") + f"{p.elem} *")
        fargs += ["polygeist_v2_flen"] * nchar
        fret = "double" if f["is_function"] else "void"
        out.append(f"extern {fret} {f['entry']}({', '.join(fargs)});\n")
        # enum -> CHARACTER converters (only constants some manifest uses)
        for ety, table in sorted(f["enum_chars"].items()):
            fn = f"polygeist_v2_{ca.call}_{ety}_char"
            cases = "".join(f"  case {v}: return '{c}';\n" for v, c in sorted(table.items(), key=lambda x: int(x[0])))
            out.append(f"static char {fn}(int32_t v) {{\n  switch (v) {{\n{cases}  }}\n"
                       f"  polygeist_v2_fatal(\"{ca.call}\", \"enum value without a Fortran flag\");\n"
                       f"  return 0;\n}}\n")
        body = [f"{proto(ca)} {{\n", "  double polygeist_v2_t0 = polygeist_v2_now_ms();\n"]
        call_args, lens = [], []
        for a in f["args"]:
            if a["kind"] == "char":
                p = abi[a["abi"]]
                body.append(f"  char pv2_c_{p.name} = polygeist_v2_{ca.call}_{p.enum_type}_char({p.name});\n")
                call_args.append(f"&pv2_c_{p.name}")
                lens.append("(polygeist_v2_flen)1")
            elif a["kind"] == "info":
                body.append(f"  int pv2_{a['fortran'].lower()} = 0;\n")
                call_args.append(f"&pv2_{a['fortran'].lower()}")
            elif a["kind"] == "array":
                call_args.append(a["abi"])
            else:
                call_args.append(f"&{a['abi']}")
        # parameters the Fortran entry does not take: must alias an argument
        # that the Fortran routine updates in place, with the same layout
        passed_arrays = [abi[a["abi"]] for a in f["args"] if a["kind"] == "array"]
        for d in f["dropped"]:
            p = abi[d]
            if p.tkind != "array":
                continue
            cands = [q for q in passed_arrays if q.written and q.rank == p.rank]
            if not cands:
                raise A.AbiError(f"{ca.call}: dropped {d} has no in-place candidate")
            ld = p.layout or "0"
            body.append(f"  {{ const void *const c[] = {{{', '.join(q.name for q in cands)}}};\n"
                        f"    const int32_t l[] = {{{', '.join(q.layout or '0' for q in cands)}}};\n"
                        f"    polygeist_v2_require_alias(\"{ca.call}\", {d}, {ld}, c, l, {len(cands)}); }}\n")
        for p in ca.params:
            if p.tkind == "array":
                body.append(f"  (void){p.name}_bytes;\n")
        call = f"{f['entry']}({', '.join(call_args + lens)})"
        if f["is_function"]:
            body.append(f"  double polygeist_v2_r = {call};\n")
        else:
            body.append(f"  {call};\n")
        for a in f["args"]:
            if a["kind"] == "info":
                body.append(f"  polygeist_v2_info(\"{ca.call}\", \"{f['entry']}\", pv2_{a['fortran'].lower()});\n")
        body.append(f"  polygeist_v2_done(\"{ca.call}\", \"{f['entry']}\", polygeist_v2_t0);\n")
        if f["is_function"]:
            body.append("  return polygeist_v2_r;\n")
        body.append("}\n\n")
        out.append("".join(body))
    return "".join(out)


# ---------------------------------------------------------------------------
# CUDA (cuBLAS / cuSOLVER)
# ---------------------------------------------------------------------------

def gen_cuda(calls) -> str:
    out = []
    for ca in calls.values():
        hp = A.resolve_proto(ca.library, ca.call)
        abi = {p.name: p for p in ca.params}
        managed = {m.name: m for m in ca.managed}
        check = "POLYGEIST_V2_CUSOLVER" if hp.ret.startswith("cusolver") else "POLYGEIST_V2_CUBLAS"
        handle = "polygeist_v2_cusolver()" if hp.ret.startswith("cusolver") else "polygeist_v2_cublas()"
        b = [f"{proto(ca)} {{\n", "  polygeist_v2_map_t pv2_map;\n",
             f"  polygeist_v2_map_init(&pv2_map, \"{ca.call}\", \"{hp.name}\");\n"]
        slot = {}
        for p in ca.params:
            if p.tkind == "array":
                slot[p.name] = f"pv2_k_{p.name}"
                b.append(f"  int pv2_k_{p.name} = polygeist_v2_map_add(&pv2_map, {p.name}, {p.name}_bytes, {int(p.written)});\n")
        b.append("  polygeist_v2_map_to_device(&pv2_map);\n")
        expr: dict[str, str] = {}
        for prm in hp.params[1:]:
            if prm.name in abi:
                p = abi[prm.name]
                if p.tkind == "enum":
                    expr[prm.name] = f"({p.enum_type}){p.name}"
                elif p.tkind == "int":
                    expr[prm.name] = p.name
                elif p.tkind == "scalar":
                    expr[prm.name] = f"&{p.name}"
                else:
                    expr[prm.name] = f"({prm.ctype})polygeist_v2_map_dev(&pv2_map, {slot[p.name]})"
        for mg in ca.managed:
            if mg.role == "lwork":
                b.append(f"  int {mg.name} = 0;\n")
                expr[mg.name] = mg.name
            elif mg.role == "workspace":
                expr[mg.name] = mg.name
            elif mg.role == "info":
                b.append(f"  int *{mg.name} = polygeist_v2_devinfo();\n")
                expr[mg.name] = mg.name
            elif mg.role == "result":
                b.append(f"  double {mg.name} = 0;\n")
                expr[mg.name] = f"&{mg.name}"
        if ca.buffer_size_symbol:
            bargs = []
            for bp in ca.buffer_size_params:
                if bp["name"] in managed and managed[bp["name"]].role == "lwork":
                    bargs.append(f"&{bp['name']}")
                elif bp["name"] in expr:
                    bargs.append(expr[bp["name"]])
                else:
                    raise A.AbiError(f"{ca.buffer_size_symbol}: cannot bind {bp['name']}")
            b.append(f"  {check}({ca.buffer_size_symbol}({handle}, {', '.join(bargs)}), \"{ca.buffer_size_symbol}\");\n")
            ws = next(m for m in ca.managed if m.role == "workspace")
            elem = ws.ctype.replace("*", "").replace("const", "").strip()
            b.append(f"  {ws.ctype} {ws.name} = ({ws.ctype})polygeist_v2_workspace((size_t){next(m.name for m in ca.managed if m.role == 'lwork')} * sizeof({elem}));\n")
        args = [expr[prm.name] for prm in hp.params[1:]]
        b.append(f"  {check}({hp.name}({handle}, {', '.join(args)}), \"{hp.name}\");\n")
        b.append("  polygeist_v2_map_to_host(&pv2_map);\n")
        for mg in ca.managed:
            if mg.role == "info":
                b.append(f"  polygeist_v2_check_devinfo(\"{ca.call}\", {mg.name});\n")
        if ca.returns:
            res = next(m.name for m in ca.managed if m.role == "result")
            b.append(f"  return {res};\n")
        b.append("}\n\n")
        out.append("".join(b))
    return "".join(out)


def splice(path: Path, tag: str, body: str) -> str:
    text = path.read_text()
    m = re.search(rf"(/\* BEGIN GENERATED {tag} \*/\n)(.*?)(/\* END GENERATED {tag} \*/)", text, re.S)
    if not m:
        raise SystemExit(f"{path}: missing GENERATED {tag} markers")
    return text[:m.start(2)] + body + text[m.end(2):]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--output-dir", type=Path,
                    help="write generated runtime beside build artifacts, preserving source templates")
    a = ap.parse_args()
    calls, _specs, errors = A.build_table()
    for s, e in sorted(errors.items()):
        print(f"note: spec {s} not in the ABI table: {e}", file=sys.stderr)
    new = {HDR: splice(HDR, "PROTOTYPES", gen_header(calls)),
           CPU: splice(CPU, "WRAPPERS", gen_cpu(calls)),
           CUDA: splice(CUDA, "WRAPPERS", gen_cuda(calls))}
    if a.output_dir:
        a.output_dir.mkdir(parents=True, exist_ok=True)
        new = {a.output_dir / p.name: text for p, text in new.items()}
    stale = [p for p, t in new.items() if not p.exists() or p.read_text() != t]
    if a.check:
        for p in stale:
            print(f"stale: {p}", file=sys.stderr)
        return 1 if stale else 0
    for p in stale:
        p.write_text(new[p])
        print(f"wrote {p}")
    print(f"{len(calls)} calls: {', '.join(calls)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
