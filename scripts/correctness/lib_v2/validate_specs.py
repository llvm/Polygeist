#!/usr/bin/env python3
"""Numerically validate raised lib_v2 specs against their C transliteration.

For every spec whose raise produced generic/raised/<spec>_debuf.mlir:
  * lower the debufferized function exactly like e2e_v2.py does
    (legalize_tensor_submaps + production residual lowering) to LLVM;
  * compile the C reference (ref_c via a wrapper that renames the symbol to
    <spec>_ref) with clang;
  * generate a harness that fills every array with deterministic pseudo-random
    values (2-D arrays are padded: 16 rows x leading dimension 16), picks
    distinct sizes for the size parameters (7, 5, 6, 4, ...), 16 for VLA
    leading dimensions, 2 for increments, and non-trivial scalars, calls both
    implementations on copies and compares every array element and the
    return value.

A spec whose raised semantics differ from its C source (e.g. a raise that
drops a loop-carried dependence) is reported UNSOUND.

Usage: validate_specs.py [--only SPEC...] [-j N]   (writes raised/validation.json)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import recipe as R  # noqa: E402
import e2e_v2 as E  # noqa: E402
import mlir_lite as ML  # noqa: E402

REPO = R.REPO
LIB = REPO / "generic_solver" / os.environ.get("POLYGEIST_SPEC_SUITE", "lib_v3")
RAISED = LIB / "raised"
WORK = RAISED / "validate"
ROWS, LD, VEC = 16, 16, 256


def c_param_kinds(spec: str, fn_args: list[tuple[str, str]], names: list[str],
                  lds: set[str]) -> list[dict]:
    out = []
    size_vals = iter([7, 5, 6, 4, 3, 8])
    scal_vals = iter([1.37, -0.61, 0.83, 1.9])
    for (ssa, ty), nm in zip(fn_args, names):
        if ty == "i32":
            if nm in lds:
                out.append({"kind": "int", "name": nm, "val": LD})
            elif nm.startswith("inc"):
                out.append({"kind": "int", "name": nm, "val": 2})
            else:
                out.append({"kind": "int", "name": nm, "val": next(size_vals)})
        elif ty == "f64":
            out.append({"kind": "dbl", "name": nm, "val": next(scal_vals)})
        elif re.match(r"memref<\?x\?xf64>", ty):
            out.append({"kind": "arr2", "name": nm})
        elif re.match(r"memref<\?xf64>", ty):
            out.append({"kind": "arr1", "name": nm})
        else:
            raise ValueError(f"unsupported parameter type {ty}")
    return out


def harness(spec: str, params: list[dict], ret: str | None) -> str:
    L = ["#include <stdio.h>", "#include <stdint.h>", "#include <string.h>", "#include <math.h>", ""]
    # reference prototype (C ABI, VLA parameters decay to pointers)
    ref_args, impl_args = [], []
    for p in params:
        if p["kind"] == "int":
            ref_args.append("int")
            impl_args.append("int32_t")
        elif p["kind"] == "dbl":
            ref_args.append("double")
            impl_args.append("double")
        elif p["kind"] == "arr2":
            ref_args.append("double *")
            impl_args += ["double *", "double *", "int64_t", "int64_t", "int64_t", "int64_t", "int64_t"]
        else:
            ref_args.append("double *")
            impl_args += ["double *", "double *", "int64_t", "int64_t", "int64_t"]
    rt = "double" if ret == "f64" else "void"
    L.append(f"{rt} {spec}_ref({', '.join(ref_args)});")
    L.append(f"{rt} {spec}_impl({', '.join(impl_args)});")
    L.append("static double rnd(unsigned *s) { *s = *s * 1103515245u + 12345u; "
             "return ((*s >> 8) & 0xffff) / 32768.0 - 1.0; }")
    L.append("int main(void) {")
    L.append("  unsigned seed = 12345u;")
    for p in params:
        if p["kind"] in ("arr2", "arr1"):
            n = ROWS * LD if p["kind"] == "arr2" else VEC
            L.append(f"  static double {p['name']}_r[{n}], {p['name']}_i[{n}];")
            L.append(f"  for (int t = 0; t < {n}; t++) {{ {p['name']}_r[t] = rnd(&seed); "
                     f"{p['name']}_i[t] = {p['name']}_r[t]; }}")
    ra, ia = [], []
    for p in params:
        if p["kind"] == "int":
            ra.append(str(p["val"]))
            ia.append(str(p["val"]))
        elif p["kind"] == "dbl":
            ra.append(repr(p["val"]))
            ia.append(repr(p["val"]))
        elif p["kind"] == "arr2":
            ra.append(f"{p['name']}_r")
            ia += [f"{p['name']}_i", f"{p['name']}_i", "0", str(ROWS), str(LD), str(LD), "1"]
        else:
            ra.append(f"{p['name']}_r")
            ia += [f"{p['name']}_i", f"{p['name']}_i", "0", str(VEC), "1"]
    if ret == "f64":
        L.append(f"  double rr = {spec}_ref({', '.join(ra)});")
        L.append(f"  double ri = {spec}_impl({', '.join(ia)});")
    else:
        L.append(f"  {spec}_ref({', '.join(ra)});")
        L.append(f"  {spec}_impl({', '.join(ia)});")
    L.append("  double maxabs = 0, maxrel = 0; long bad = 0;")
    L.append("#define CMP(a, b) do { double d_ = fabs((a) - (b)); if (d_ > maxabs) maxabs = d_; "
             "double r_ = d_ / (fabs(a) > 1e-300 ? fabs(a) : 1e-300); if (d_ > 0 && r_ > maxrel) maxrel = r_; "
             "if (!(d_ <= 1e-12 + 1e-12 * fabs(a))) bad++; } while (0)")
    for p in params:
        if p["kind"] in ("arr2", "arr1"):
            n = ROWS * LD if p["kind"] == "arr2" else VEC
            L.append(f"  for (int t = 0; t < {n}; t++) CMP({p['name']}_r[t], {p['name']}_i[t]);")
    if ret == "f64":
        L.append("  CMP(rr, ri);")
    L.append('  printf("maxabs=%.3e maxrel=%.3e bad=%ld\\n", maxabs, maxrel, bad);')
    L.append("  return bad ? 1 : 0;")
    L.append("}")
    return "\n".join(L) + "\n"


def lower_memref_linalg(inp: Path, d: Path, fn: str, log: Path) -> Path | None:
    """Independent route: memref-form Linalg (before debufferization) lowered
    with the in-tree loop path; no tensor views, no legaliser involved."""
    loops = d / "linalg_loops.mlir"
    if E.sh([R.POLYGEIST_OPT, "--convert-linalg-to-loops", "--lower-polygeist-submap",
             inp, "-o", loops], log):
        return None
    llm, ll = d / "linalg_llvm.mlir", d / "linalg_kernel.ll"
    if E.sh([E.MLIR_OPT, *E.LOWER_TO_LLVM, loops, "-o", llm], log):
        return None
    if E.sh([E.MLIR_TRANSLATE, "--mlir-to-llvmir", llm, "-o", ll], log):
        return None
    ll.write_text(re.sub(rf"@{fn}\b", f"@{fn}_impl", ll.read_text()))
    return ll


def validate(spec_name: str, manifest_spec: dict, ref_c: Path, meta_lds: set[str],
             level: str = "defn") -> dict:
    d = WORK / spec_name / level
    d.mkdir(parents=True, exist_ok=True)
    log = d / "log.txt"
    log.write_text("")
    debuf = RAISED / (f"{spec_name}.mlir" if level == 'affine' else f"{spec_name}_debuf.mlir")
    res = {"spec": spec_name}
    if not debuf.exists():
        return {**res, "verdict": "NO_ARTIFACT"}
    mod = ML.Module(debuf.read_text())
    f = mod.func_of(spec_name)
    if f is None:
        return {**res, "verdict": "NO_FUNCTION"}
    names = manifest_spec.get("args") or [a[0][1:] for a in f.args]
    m = re.search(r"\)\s*->\s*(\S+)\s+attributes", f.text.split("\n", 1)[0])
    ret = m.group(1) if m else None
    try:
        params = c_param_kinds(spec_name, f.args, names, meta_lds)
    except ValueError as e:
        return {**res, "verdict": "UNSUPPORTED_SIGNATURE", "why": str(e)}
    if level == 'affine':
        ll = lower_memref_linalg(debuf, d, spec_name, log)
    elif level == "linalg":
        ll = lower_memref_linalg(RAISED / f"{spec_name}_linalg.mlir", d, spec_name, log)
    else:
        ll = E.lower(debuf, d, "spec", spec_name, log)
    if ll is None:
        return {**res, "verdict": "LOWER_FAIL", "why": "see " + str(log.relative_to(REPO))}
    refw = d / "ref_wrapper.c"
    defs = "".join(f"#define {k} {v}\n" for k, v in manifest_spec.get("defines", {}).items())
    refw.write_text(f"#define NETLIB_SPEC {spec_name}_ref\n{defs}#include \"{ref_c}\"\n")
    (d / "harness.c").write_text(harness(spec_name, params, ret))
    exe = d / "check"
    if E.sh([E.CLANG, "-O0", "-ffp-contract=off", d / "harness.c", refw, ll, "-lm",
             f"-L{E.MLIR_LIBDIR}", f"-Wl,-rpath,{E.MLIR_LIBDIR}", "-lmlir_c_runner_utils",
             "-o", exe], log):
        return {**res, "verdict": "BUILD_FAIL"}
    try:
        p = subprocess.run([exe], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return {**res, "verdict": "RUN_TIMEOUT"}
    out = p.stdout.strip()
    mm = re.search(r"maxabs=(\S+) maxrel=(\S+) bad=(\d+)", out)
    if not mm:
        return {**res, "verdict": "RUN_FAIL", "why": (p.stderr or out)[-300:]}
    maxabs, maxrel, bad = float(mm.group(1)), float(mm.group(2)), int(mm.group(3))
    verdict = "EXACT" if maxabs == 0 else ("WITHIN_1e-12" if bad == 0 else "UNSOUND")
    return {**res, "verdict": verdict, "maxabs": maxabs, "maxrel": maxrel, "mismatches": bad}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("-j", type=int, default=8)
    a = ap.parse_args()
    import build_lib_v2 as B
    specs = B.load_specs()
    results = json.loads((RAISED / "build_results.json").read_text())
    status = {r["spec"]: r for r in results}
    todo = [s for s in specs if (not a.only or s.name in a.only)
            and status.get(s.name, {}).get("status") in ("RAISED", "PARTIAL")]

    def one(s):
        lds = set(B.vla_leading_dims(s.ref_c).keys())
        r = validate(s.name, s.raw, s.ref_c, lds, "defn")
        r2 = validate(s.name, s.raw, s.ref_c, lds, "linalg")
        r["linalg_verdict"] = r2["verdict"]
        r["linalg_maxabs"] = r2.get("maxabs")
        r["build_status"] = status[s.name]["status"]
        return r
    with ThreadPoolExecutor(a.j) as ex:
        out = list(ex.map(one, todo))
    for r in out:
        print(f"{r['verdict']:14s} linalg={r['linalg_verdict']:14s} {r['build_status']:8s} "
              f"{r['spec']} {r.get('maxabs', '')}")
    path = RAISED / "validation.json"
    old = json.loads(path.read_text()) if path.exists() and a.only else {}
    old.update({r["spec"]: r for r in out})
    path.write_text(json.dumps(old, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
