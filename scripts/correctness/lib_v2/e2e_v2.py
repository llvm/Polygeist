#!/usr/bin/env python3
"""End-to-end numerical check of lib_v2 matches (v2 analogue of
`run_kernel_e2e.sh --match-canonical`, which is not modified).

For each PolyBench kernel:
  1. reference: clang-compiled original source (all arrays dumped at %.17g);
  2. raise the kernel with the shared lib_v2 recipe (recipe.py);
  3. control  : lower the raised IR without matching ("residual");
  4. v2       : match_v2.py -> inject the used kernel.defn ops ->
                polygeist-opt --lower-kernel-launch (inlines the netlib defn
                bodies at the launch sites with strict operand types) -> lower;
  5. link each lowered kernel with an algorithm-neutral PolyBench driver
     (generate_polybench_harness.py) and gen_wrapper.py, run, compare dumps.

A wrong operand binding (e.g. A/B not swapped, m/n transposed) produces wrong
numbers, so v2 PASS validates the binding; the residual control separates
raise/lowering problems from matching problems.

Usage: e2e_v2.py [--dataset MINI|SMALL|...] [--out DIR] kernel...
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import recipe as R  # noqa: E402
from generate_polybench_harness import generate_harness  # noqa: E402

REPO = R.REPO
UTIL = REPO / "tools/cgeist/Test/polybench/utilities"
MANIFEST = REPO / "issues/polybench_section42/manifest.csv"
LLVM_BIN = REPO / "llvm-project/build/bin"
CLANG = str(LLVM_BIN / "clang")
MLIR_OPT = str(LLVM_BIN / "mlir-opt")
MLIR_TRANSLATE = str(LLVM_BIN / "mlir-translate")
MLIR_LIBDIR = REPO / "llvm-project/build/lib"
LIB = REPO / f"generic_solver/kernel_library_{os.environ.get('POLYGEIST_SPEC_SUITE', 'lib_v3').removeprefix('lib_')}.mlir"
PY = sys.executable

PRECISION = ('\n#undef DATA_PRINTF_MODIFIER\n'
             '#define DATA_PRINTF_MODIFIER "%0.17g "\n')

LOWER_TO_LLVM = [
    "--convert-math-to-llvm", "--convert-math-to-libm",
    "--empty-tensor-to-alloc-tensor", "--lower-affine",
    "--one-shot-bufferize=bufferize-function-boundaries",
    "--convert-linalg-to-loops", "--convert-scf-to-cf", "--expand-strided-metadata",
    "--lower-affine", "--convert-arith-to-llvm", "--convert-index-to-llvm",
    "--finalize-memref-to-llvm", "--convert-func-to-llvm",
    "--reconcile-unrealized-casts"]


def sources() -> dict[str, Path]:
    with MANIFEST.open() as fh:
        return {r["kernel"]: REPO / r["source"] for r in csv.DictReader(fh)}


def sh(cmd, log: Path, timeout=900, stdout=None) -> int:
    with log.open("a") as fh:
        fh.write("$ " + " ".join(map(str, cmd)) + "\n")
        fh.flush()
        try:
            p = subprocess.run([str(c) for c in cmd], stdout=stdout or fh, stderr=fh,
                               timeout=timeout)
            return p.returncode
        except subprocess.TimeoutExpired:
            fh.write("TIMEOUT\n")
            return 124


def with_precision(text: str, kernel_header: str) -> str:
    """Insert the dump-precision override after the kernel's own header."""
    m = re.search(rf'#\s*include\s+"{re.escape(kernel_header)}"[^\n]*\n', text)
    if not m:
        raise RuntimeError(f"cannot find #include \"{kernel_header}\"")
    return text[:m.end()] + PRECISION + text[m.end():]


def lower(inp: Path, d: Path, tag: str, fn: str, log: Path) -> Path | None:
    legal = d / f"{tag}_legal.mlir"
    if sh([PY, HERE / "legalize_tensor_submaps.py", inp, "-o", legal], log):
        return None
    canon = d / f"{tag}_canon.mlir"
    if sh([R.POLYGEIST_OPT, "--lower-polygeist-submap", "--canonicalize-polygeist",
           legal, "-o", canon], log):
        return None
    if "polygeist.submap" in canon.read_text():
        c2 = d / f"{tag}_canon_loops.mlir"
        if sh([R.POLYGEIST_OPT, "--convert-linalg-to-loops", "--canonicalize",
               "--lower-polygeist-submap", "--canonicalize", canon, "-o", c2], log):
            return None
        canon = c2
    t = canon.read_text()
    t = re.sub(r"bufferization\.to_tensor (%[^ ]*) :", r"bufferization.to_tensor \1 restrict :", t)
    canon.write_text(t)
    llvm_mlir = d / f"{tag}_llvm.mlir"
    if sh([MLIR_OPT, *LOWER_TO_LLVM, canon, "-o", llvm_mlir], log):
        return None
    ll = d / f"{tag}_kernel.ll"
    if sh([MLIR_TRANSLATE, "--mlir-to-llvmir", llvm_mlir, "-o", ll], log):
        return None
    ll.write_text(re.sub(rf"@{fn}\b", f"@{fn}_impl", ll.read_text()))
    return ll


def compare(ref: Path, out: Path) -> dict:
    def vals(p):
        t = p.read_text(errors="replace")
        if "BEGIN DUMP_ARRAYS" in t:
            t = t.split("BEGIN DUMP_ARRAYS", 1)[1].split("END   DUMP_ARRAYS", 1)[0]
        nums = re.findall(r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|inf|nan)", t)
        return [float(x) for x in nums]
    a, b = vals(ref), vals(out)
    res = {"n_ref": len(a), "n_out": len(b), "exact_text": ref.read_text() == out.read_text()}
    if len(a) != len(b) or not a:
        res.update(verdict="FAIL", why="different number of dumped values")
        return res
    max_abs = max(abs(x - y) for x, y in zip(a, b))
    max_rel = max(abs(x - y) / max(abs(x), 1e-300) for x, y in zip(a, b) if x != y) if max_abs else 0.0
    ok_tight = all(math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12) for x, y in zip(a, b))
    ok_loose = all(abs(x - y) <= 1.1e-2 + 5e-4 * abs(x) for x, y in zip(a, b))
    res.update(max_abs=max_abs, max_rel=max_rel,
               verdict=("PASS_EXACT" if max_abs == 0 else
                        "PASS_1e-12" if ok_tight else
                        "PASS_LOOSE" if ok_loose else "FAIL"))
    return res


def _func_text(mlir: str, name: str) -> str:
    """The text of `func.func @name` (balanced braces) in a module."""
    k = mlir.index(f"func.func @{name}(")
    k = mlir.rfind("\n", 0, k) + 1
    depth, i, started = 0, mlir.index("{\n", k), False   # the body brace ends the header line
    while i < len(mlir):
        if mlir[i] == "{":
            depth += 1
            started = True
        elif mlir[i] == "}":
            depth -= 1
            if started and depth == 0:
                return mlir[k:i + 1]
        i += 1
    raise ValueError(f"unbalanced func {name}")


def call_spec_functions(matched: Path, specs: set, d: Path) -> Path:
    """Region launches (memref operands) -> func.call of the spec's own
    compiled C (its cgeist function, same operand list), appended to the
    module: an end-to-end check of the binding, guards and residual loops
    that does not depend on the e-graph proof."""
    raised = REPO / "generic_solver" / os.environ.get("POLYGEIST_SPEC_SUITE", "lib_v3") / "raised"
    text = matched.read_text()
    funcs = []
    for s_ in sorted(specs):
        pat = re.compile(rf"kernel\.launch @{re.escape(s_)}\(")
        if not pat.search(text):
            continue
        text = pat.sub(f"func.call @spec_c_{s_}(", text)
        src = (raised / f"{s_}.mlir").read_text()
        ftext = _func_text(src, s_).replace(f"func.func @{s_}(", f"func.func @spec_c_{s_}(", 1)
        # inline the file's attribute aliases (#map, #set): the program
        # defines its own aliases under the same names
        aliases = dict(re.findall(r"^(#[\w$.]+) = (.*)$", src, re.M))
        for name in sorted(aliases, key=len, reverse=True):
            ftext = re.sub(re.escape(name) + r"(?![\w$.])", lambda m_, v=aliases[name]: v, ftext)
        funcs.append(ftext)
    if funcs:
        end = text.rstrip().rfind("}")
        text = text[:end] + "\n".join(funcs) + "\n}\n"
    out = d / "v2_called.mlir"
    out.write_text(text)
    return out


def run_kernel(kernel: str, src: Path, dataset: str, out: Path) -> dict:
    d = out / kernel
    d.mkdir(parents=True, exist_ok=True)
    log = d / "log.txt"
    log.write_text("")
    fn = "kernel_" + kernel.replace("-", "_")
    kdir = src.parent
    flags = [f"-I{UTIL}", f"-I{kdir}", "-DDATA_TYPE_IS_DOUBLE", "-DPOLYBENCH_DUMP_ARRAYS",
             f"-D{dataset}_DATASET", "-DPOLYBENCH_USE_C99_PROTO"]
    row = {"kernel": kernel, "dataset": dataset}
    text = src.read_text()
    header = f"{src.stem}.h"
    # 1. reference
    ref_src = d / "ref_src.c"
    ref_src.write_text(with_precision(text, header))
    if sh([CLANG, "-O1", *flags, "-Dstatic=", ref_src, UTIL / "polybench.c", "-lm",
           "-o", d / "ref_exe"], log):
        row["status"] = "ref_build_fail"
        return row
    with (d / "ref.out").open("w") as fh:
        subprocess.run([d / "ref_exe"], stderr=fh, stdout=subprocess.DEVNULL, timeout=600)
    # 2. raise with the shared recipe
    rr = R.raise_c(R.PRODUCTION, str(src), fn, str(d / "affine"),
                   [*flags, "-Dstatic=__attribute__((noipa))"], 600)
    for step in ("cgeist", "raise", "debuf"):
        if step not in rr or not rr[step].ok:
            row["status"] = f"raise_{step}_fail"
            return row
    debuf = Path(rr["paths"]["debuf"])
    # harness + wrapper
    harness = d / "harness.c"
    harness.write_text(with_precision(generate_harness(text, fn), header))
    with (d / "wrapper.c").open("w") as fh:
        if subprocess.run([PY, HERE.parent / "gen_wrapper.py", src, fn], stdout=fh,
                          stderr=subprocess.DEVNULL).returncode:
            row["status"] = "wrapper_fail"
            return row
    objs = []
    for c in (harness, d / "wrapper.c", UTIL / "polybench.c"):
        o = d / (Path(c).stem + ".o")
        if sh([CLANG, "-c", "-O1", *flags, "-Dstatic=", c, "-o", o], log):
            row["status"] = f"compile_fail_{Path(c).name}"
            return row
        objs.append(o)

    def build_and_run(tag: str, mlir: Path) -> dict:
        ll = lower(mlir, d, tag, fn, log)
        if ll is None:
            return {"verdict": "LOWER_FAIL"}
        exe = d / f"{tag}_exe"
        if sh([CLANG, "-O1", *objs, ll, "-lm", f"-L{MLIR_LIBDIR}",
               f"-Wl,-rpath,{MLIR_LIBDIR}", "-lmlir_c_runner_utils", "-o", exe], log):
            return {"verdict": "LINK_FAIL"}
        outp = d / f"{tag}.out"
        with outp.open("w") as fh:
            try:
                subprocess.run([exe], stderr=fh, stdout=subprocess.DEVNULL, timeout=600)
            except subprocess.TimeoutExpired:
                return {"verdict": "RUN_TIMEOUT"}
        return compare(d / "ref.out", outp)

    # 3. residual control
    row["residual"] = build_and_run("residual", debuf)
    # 4. v2
    matched = d / "v2_matched.mlir"
    rep = d / "v2_report.json"
    region_mode = os.environ.get("E2E_REGION_MATCH", "") not in ("", "0")
    region = [] if region_mode else ["--no-region-match"]
    failed = sh([PY, HERE / "match_v2.py", debuf, "-o", matched, "--report", rep,
                 "--inject-defns", *region], log)
    if failed and region_mode:
        # a region launch of a spec without kernel.defn: call its C instead
        failed = sh([PY, HERE / "match_v2.py", debuf, "-o", matched, "--report", rep, *region], log)
    if failed:
        row["v2"] = {"verdict": "MATCH_FAIL"}
        row["status"] = "ok"
        return row
    report = json.loads(rep.read_text())
    row["v2_launches"] = [l["spec"] for l in report["launches"] if "spec" in l] + \
        [l["region_match"] for l in report["launches"] if "region_match" in l and not l.get("recognition_only")]
    row["v2_bindings"] = [{"spec": l["spec"], "binding": l["binding"]}
                          for l in report["launches"] if "spec" in l]
    if row["v2_launches"] and region_mode:
        regions = {l["region_match"] for l in report["launches"]
                   if "region_match" in l and not l.get("recognition_only")}
        matched = call_spec_functions(matched, regions, d)
    if row["v2_launches"]:
        inlined = d / "v2_inlined.mlir"
        if sh([R.POLYGEIST_OPT, "--lower-kernel-launch", matched, "-o", inlined], log):
            row["v2"] = {"verdict": "LOWER_KERNEL_LAUNCH_FAIL"}
        else:
            row["v2"] = build_and_run("v2", inlined)
    else:
        row["v2"] = {"verdict": "NO_MATCH"}
    row["status"] = "ok"
    (d / "result.json").write_text(json.dumps(row, indent=1))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="MINI")
    ap.add_argument("--out", default=str(REPO / "issues/lib_v2/e2e"))
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--region", action="store_true",
                    help="also splice region-matcher launches (incl. partial call + residual loop)")
    ap.add_argument("kernels", nargs="*")
    a = ap.parse_args()
    if a.region:
        os.environ["E2E_REGION_MATCH"] = "1"
    srcs = sources()
    ks = a.kernels or sorted(srcs)
    out = Path(a.out) / a.dataset
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(a.j) as ex:
        rows = list(ex.map(lambda k: run_kernel(k, srcs[k], a.dataset, out), ks))
    for r in rows:
        print(f"{r['kernel']:14s} {r.get('status'):10s} residual={r.get('residual', {}).get('verdict')}"
              f" v2={r.get('v2', {}).get('verdict')} launches={r.get('v2_launches')}"
              f" maxrel={r.get('v2', {}).get('max_rel')}")
    summary = out / "summary.json"
    old = json.loads(summary.read_text()) if summary.exists() else {}
    old.update({r["kernel"]: r for r in rows})
    summary.write_text(json.dumps(old, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
