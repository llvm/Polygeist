#!/usr/bin/env python3
"""Numeric cross-check of a RegionMatch: run the application kernel and the
matched spec (C transliteration from ref_c, called exactly as the emitted
launch_text prescribes) on the same random inputs and compare outputs.

Only the emitted launch text is interpreted (memref.copy -> memcpy,
memref.dim %argK, d -> extent d of the K-th application parameter,
kernel.launch operand order -> spec call).  Square 2-D arrays are
initialised symmetric and diagonally dominant (hence SPD), which is a valid
input for both triangular solves and Cholesky.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
LIB = REPO / "generic_solver/lib_v2"
UTIL = REPO / "tools/cgeist/Test/polybench/utilities"


def _c_params(src: str, fn: str) -> list[tuple[str, str, list[str]]]:
    """[(kind, name, dims)] for `fn`'s parameters (kind: int|double|array)."""
    m = re.search(r"\b" + re.escape(fn) + r"\s*\(([^)]*)\)\s*\{", src, re.S)
    if not m:
        raise ValueError(f"prototype of {fn} not found")
    out = []
    for p in m.group(1).split(","):
        p = " ".join(p.split())
        am = re.match(r"(?:const\s+)?(double|float|int)\s+(\w+)\s*((?:\[[^\]]*\])*)$", p)
        if not am:
            raise ValueError(f"cannot parse parameter {p!r}")
        ty, name, dims = am.group(1), am.group(2), re.findall(r"\[([^\]]*)\]", am.group(3))
        kind = "array" if dims else ty
        out.append((kind, name, dims))
    return out


def _spec_entry(spec: str):
    for mp in sorted((LIB / "manifest").glob("*.json")):
        d = json.loads(mp.read_text())
        for s in d.get("specs", []):
            if s["name"] == spec:
                return d, s
    return None, None


def interpret_launch_text(launch_text: str, params) -> tuple[list, list]:
    """Interpret the emitted lines against the C parameters of the
    application function.  Returns (steps, calls_in_order) where each step is
    ('copy', src_k, dst_k) or ('call', spec, [C expressions])."""
    vals: dict[str, str] = {}
    md: dict[str, int] = {}
    steps = []

    def c_dims(k):
        return params[k][2]

    def ref(op: str) -> str:
        op = op.strip()
        if op.startswith("%arg") and op[4:].isdigit():
            return f"@{int(op[4:])}"            # array / scalar parameter K
        m = re.match(r"(%[\w$.-]+)#(\d+)$", op)
        if m and m.group(1) in md:
            k, j = md[m.group(1)], int(m.group(2))
            dm = c_dims(k)
            r = len(dm)
            if j == 1:
                return "0"
            if 2 <= j < 2 + r:
                return f"({dm[j - 2]})"
            s = j - 2 - r
            prod = "*".join(f"({d})" for d in dm[s + 1:]) or "1"
            return f"({prod})"
        if op in vals:
            return vals[op]
        raise ValueError(f"cannot interpret operand {op}")

    for line in launch_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"(%[\w$.-]+):(\d+) = memref\.extract_strided_metadata (%arg\d+)\b", line)
        if m:
            md[m.group(1)] = int(m.group(3)[4:])
            continue
        m = re.match(r"(%[\w$.-]+) = arith\.constant (-?[\d.eE+-]+) : (\w+)", line)
        if m:
            vals[m.group(1)] = m.group(2)
            continue
        m = re.match(r"(%[\w$.-]+) = arith\.index_cast (%[\w$.#-]+)", line)
        if m:
            vals[m.group(1)] = ref(m.group(2))
            continue
        m = re.match(r"(%[\w$.-]+) = arith\.divui (%[\w$.#-]+), (%[\w$.#-]+)", line)
        if m:
            vals[m.group(1)] = f"({ref(m.group(2))}/{ref(m.group(3))})"
            continue
        m = re.match(r"(%[\w$.-]+) = memref\.dim (%arg\d+), (%[\w$.#-]+)", line)
        if m:
            k, d = int(m.group(2)[4:]), int(ref(m.group(3)))
            vals[m.group(1)] = f"({c_dims(k)[d]})"
            continue
        m = re.match(r"(%[\w$.-]+) = memref\.cast (%arg\d+)", line)
        if m:
            vals[m.group(1)] = ref(m.group(2))
            continue
        m = re.match(r"memref\.copy (%arg\d+), (%arg\d+)", line)
        if m:
            steps.append(("copy", int(m.group(1)[4:]), int(m.group(2)[4:])))
            continue
        m = re.match(r"kernel\.launch @(\w+)\((.*)\) :", line)
        if m:
            steps.append(("call", m.group(1), [ref(x) for x in m.group(2).split(",") if x.strip()]))
            continue
        if re.match(r"%[\w$.-]+ = (bufferization\.to_tensor|tensor\.cast)", line):
            continue          # live-out re-definitions after the launch
        raise ValueError(f"cannot interpret launch line: {line[:120]}")
    return steps


def check(app_src: Path, fn: str, launch_text: str, spec: str | None = None,
          cflags: list[str] | None = None, n: int = 23, preprocess: bool = False,
          seed: int = 12345, prefix: tuple | None = None) -> dict:
    """prefix = (C file, function): for a region in the middle of `fn`, the
    application code before the region (same parameter list); it runs on the
    spec side before the emitted launch lines."""
    # the harness includes the kernel in its own translation unit, so the
    # linkage-exposure flag of the raising recipe is not wanted here
    cflags = [f for f in (cflags or []) if not f.startswith("-Dstatic")]
    src = app_src.read_text()
    if preprocess:
        src = subprocess.run(["gcc", "-E", "-P", *cflags, str(app_src)], capture_output=True,
                             text=True, check=True).stdout
    params = _c_params(src, fn)
    try:
        steps = interpret_launch_text(launch_text, params)
    except ValueError as ex:
        return {"ok": False, "error": str(ex)}
    calls = [s_ for s_ in steps if s_[0] == "call"]
    if not calls:
        return {"ok": False, "error": "no kernel.launch in launch_text"}
    specs = list(dict.fromkeys(c[1] for c in calls))
    units = []
    for k, sp in enumerate(specs):
        man, entry = _spec_entry(sp)
        if man is None:
            return {"ok": False, "error": f"spec {sp} not in manifests"}
        u = [f"#define NETLIB_SPEC {sp}"]
        for dk, dv in entry.get("defines", {}).items():
            u.append(f"#define {dk} {dv}")
        u.append(f'#include "{LIB / "ref_c" / (man.get("routine") + ".c")}"')
        units.append("\n".join(u) + "\n")

    def expr(e: str, suffix_arrays: str) -> str:
        m = re.fullmatch(r"@(\d+)", e)
        if m:
            kind, name, dm = params[int(m.group(1))]
            return name + (suffix_arrays if kind == "array" else "")
        return re.sub(r"@(\d+)", lambda mm: params[int(mm.group(1))][1], e)

    lines = ["#include <stdio.h>", "#include <stdlib.h>", "#include <string.h>", "#include <math.h>"]
    for sp in specs:
        lines.append(f"void {sp}();")
    if prefix:
        lines.append(f"void {prefix[1]}();")
    lines.append("#define main polybench_unused_main")
    lines.append(f'#include "{app_src}"')
    lines.append("#undef main")
    lines.append("static double rnd(unsigned *s){*s=*s*1103515245u+12345u;return ((*s>>8)&0xffff)/65536.0;}")
    lines.append(f"int main(void){{ unsigned seed={int(seed)}u;")
    # distinct integer parameters get distinct values, so a swapped extent
    # binding (m <-> n) is caught
    offs = [0, 4, -5, 7, -3, -7, 2]
    k_int = 0
    for kind, name, dm in params:
        if kind == "int":
            lines.append(f"  int {name} = {max(3, n + offs[k_int % len(offs)])};")
            k_int += 1
    for kind, name, dm in params:
        if kind in ("double", "float"):
            lines.append(f"  double {name} = 0.5 + rnd(&seed);")
    for kind, name, dm in params:
        if kind != "array":
            continue
        size = "*".join(f"({d})" for d in dm)
        lines.append(f"  double *{name}_a = malloc(sizeof(double)*{size});")
        lines.append(f"  double *{name}_b = malloc(sizeof(double)*{size});")
        lines.append(f"  for (long q=0;q<{size};q++) {name}_a[q]=rnd(&seed);")
        if len(dm) == 2:
            # not symmetric (so transposed variants are distinguishable), but the
            # symmetric extension of either triangle is diagonally dominant, i.e.
            # SPD -- valid for every triangular solve and Cholesky variant
            lines.append(f"  if (({dm[0]})==({dm[1]})) {{ long N_=({dm[0]});"
                         f" for(long i=0;i<N_;i++) {name}_a[i*N_+i]+=N_; }}")
        lines.append(f"  memcpy({name}_b,{name}_a,sizeof(double)*{size});")

    def cast(kind, name, dm, suffix):
        if kind != "array":
            return name
        if len(dm) == 1:
            return f"{name}{suffix}"
        return f"(double (*){''.join('[' + d + ']' for d in dm[1:])}){name}{suffix}"
    app_args = [cast(k, nm, dm, "_a") for k, nm, dm in params]
    lines.append(f"  {fn}({', '.join(app_args)});")
    if prefix:
        lines.append(f"  {prefix[1]}({', '.join(cast(k, nm, dm, '_b') for k, nm, dm in params)});")
    for st in steps:
        if st[0] == "copy":
            k, nm, dm = params[st[2]]
            size = "*".join(f"({d})" for d in dm)
            lines.append(f"  memcpy({nm}_b, {params[st[1]][1]}_b, sizeof(double)*{size});")
        else:
            lines.append(f"  {st[1]}({', '.join(expr(e, '_b') for e in st[2])});")
    lines.append("  double err=0, mag=0;")
    for kind, name, dm in params:
        if kind != "array":
            continue
        size = "*".join(f"({d})" for d in dm)
        lines.append(f"  for (long q=0;q<{size};q++){{double d=fabs({name}_a[q]-{name}_b[q]);"
                     f"double m=fabs({name}_a[q]); if(d>err)err=d; if(m>mag)mag=m;}}")
    lines.append('  printf("%.3e %.3e\\n", err, mag); return !(err <= 1e-10*(1.0+mag)); }')
    with tempfile.TemporaryDirectory() as td:
        c = Path(td) / "h.c"
        c.write_text("\n".join(lines) + "\n")
        objs = [str(c)]
        if prefix:
            objs.append(str(prefix[0]))
        for k, u in enumerate(units):
            up = Path(td) / f"spec{k}.c"
            up.write_text(u)
            objs.append(str(up))
        exe = Path(td) / "h"
        extra = []
        if preprocess:
            extra = [*cflags, str(UTIL / "polybench.c")]
        r = subprocess.run(["gcc", "-std=gnu99", "-O1", "-w", *objs, *extra, "-o", str(exe), "-lm"],
                           capture_output=True, text=True)
        if r.returncode:
            return {"ok": False, "error": "compile: " + r.stderr[-600:]}
        r = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
        err, mag = (float(x) for x in r.stdout.split()) if r.stdout.strip() else (float("nan"),) * 2
        return {"ok": r.returncode == 0, "max_abs_err": err, "max_abs_val": mag, "n": n,
                "specs": specs}
