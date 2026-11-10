#!/usr/bin/env python3
"""End-to-end PolyBench check of region-matcher launches.

The production driver (lib_v2/match_v2.py) splices Linalg-level launches
(tensor form) and then region-matcher launches (memref form, possibly in the
middle of a function, with operands materialised from tensors and live-out
tensors re-defined after the launch).  This harness checks the result
numerically:

  1. reference : clang-compiled PolyBench source (arrays dumped at %.17g);
  2. raise     : lib_v2 recipe (recipe.py), as e2e_v2.py;
  3. match     : match_v2.py WITH the region matcher;
  4. lower     : every memref-form `kernel.launch @S(...) -> ()` becomes
                 `func.call @S(...)` of the specification's cgeist-stage
                 function (generic_solver/lib_v2/raised/S.mlir, the faithful
                 semantics of the C reference -- NOT the raised body, which is
                 unsound for some specs); tensor-form launches are inlined
                 with --lower-kernel-launch; then e2e_v2.lower();
  5. link with the algorithm-neutral PolyBench driver, run, compare.

A wrong operand binding, flag combination or leading dimension produces
wrong numbers.  Helpers are imported read-only from lib_v2/e2e_v2.py.

Usage: e2e_region.py [--dataset MINI] [--out DIR] kernel...
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORR = HERE.parents[1]
sys.path.insert(0, str(CORR / "lib_v2"))
sys.path.insert(0, str(CORR))
import e2e_v2 as E   # noqa: E402  (read-only reuse)
import recipe as R   # noqa: E402

REPO = R.REPO
SPECS = REPO / "generic_solver/lib_v2/raised"
LIB = REPO / "generic_solver/kernel_library_v2.mlir"
LAUNCH_MEM = re.compile(r"^(\s*)kernel\.launch @([\w$.-]+)\((.*)\) : \((.*)\) -> \(\)\s*$", re.M)


def spec_function(name: str) -> str:
    """The cgeist-stage function of a spec with its #map/#set aliases inlined."""
    text = (SPECS / f"{name}.mlir").read_text()
    aliases = dict(re.findall(r"^(#[\w$.-]+)\s*=\s*(affine_(?:map|set)<.*>)\s*$", text, re.M))
    m = re.search(rf"^(\s*)func\.func @{re.escape(name)}\(.*?^\1\}}\s*$", text, re.M | re.S)
    if not m:
        raise RuntimeError(f"function {name} not found in {name}.mlir")
    body = m.group(0)
    for a in sorted(aliases, key=len, reverse=True):
        body = re.sub(re.escape(a) + r"(?![\w$.-])", aliases[a], body)
    return body


def mutate_first_launch(text: str) -> tuple[str, str]:
    """Negative control: swap the first two distinct operands of equal type
    in the first memref-form launch."""
    m = LAUNCH_MEM.search(text)
    if not m:
        return text, "no memref launch"
    ops = [x.strip() for x in m.group(3).split(",")]
    tys = [x.strip() for x in m.group(4).split(",")]
    pairs = [(i, j) for i in range(len(ops)) for j in range(i + 1, len(ops))
             if tys[i] == tys[j] and ops[i] != ops[j]]
    pairs.sort(key=lambda p: (not tys[p[0]].startswith("memref"), p))   # arrays first
    for i, j in pairs:
        if True:
                ops[i], ops[j] = ops[j], ops[i]
                new = f"{m.group(1)}kernel.launch @{m.group(2)}({', '.join(ops)}) : ({m.group(4)}) -> ()"
                return text[:m.start()] + new + text[m.end():], f"swapped operands {j} and {i} of @{m.group(2)}"
    return text, "nothing to swap"


def lower_region_launches(text: str) -> tuple[str, list[str]]:
    names = []

    def rep(m):
        names.append(m.group(2))
        return f"{m.group(1)}func.call @{m.group(2)}({m.group(3)}) : ({m.group(4)}) -> ()"
    text = LAUNCH_MEM.sub(rep, text)
    names = sorted(set(names))
    if names:
        funcs = "\n".join(spec_function(n) for n in names)
        close = text.rstrip().rfind("}")
        text = text[:close] + funcs + "\n" + text[close:]
    return text, names


def run_kernel(kernel: str, src: Path, dataset: str, out: Path, mutate: bool = False) -> dict:
    d = out / (kernel + ("_mutated" if mutate else ""))
    d.mkdir(parents=True, exist_ok=True)
    log = d / "log.txt"
    log.write_text("")
    fn = "kernel_" + kernel.replace("-", "_")
    flags = [f"-I{E.UTIL}", f"-I{src.parent}", "-DDATA_TYPE_IS_DOUBLE", "-DPOLYBENCH_DUMP_ARRAYS",
             f"-D{dataset}_DATASET", "-DPOLYBENCH_USE_C99_PROTO"]
    row = {"kernel": kernel, "dataset": dataset}
    text = src.read_text()
    header = f"{src.stem}.h"
    ref_src = d / "ref_src.c"
    ref_src.write_text(E.with_precision(text, header))
    if E.sh([E.CLANG, "-O1", *flags, "-Dstatic=", ref_src, E.UTIL / "polybench.c", "-lm",
             "-o", d / "ref_exe"], log):
        row["status"] = "ref_build_fail"
        return row
    with (d / "ref.out").open("w") as fh:
        subprocess.run([d / "ref_exe"], stderr=fh, stdout=subprocess.DEVNULL, timeout=600)
    rr = R.raise_c(R.PRODUCTION, str(src), fn, str(d / "affine"),
                   [*flags, "-Dstatic=__attribute__((noipa))"], 600)
    for step in ("cgeist", "raise", "debuf"):
        if step not in rr or not rr[step].ok:
            row["status"] = f"raise_{step}_fail"
            return row
    debuf = Path(rr["paths"]["debuf"])
    harness = d / "harness.c"
    harness.write_text(E.with_precision(E.generate_harness(text, fn), header))
    with (d / "wrapper.c").open("w") as fh:
        if subprocess.run([E.PY, CORR / "gen_wrapper.py", src, fn], stdout=fh,
                          stderr=subprocess.DEVNULL).returncode:
            row["status"] = "wrapper_fail"
            return row
    objs = []
    for c in (harness, d / "wrapper.c", E.UTIL / "polybench.c"):
        o = d / (Path(c).stem + ".o")
        if E.sh([E.CLANG, "-c", "-O1", *flags, "-Dstatic=", c, "-o", o], log):
            row["status"] = f"compile_fail_{Path(c).name}"
            return row
        objs.append(o)
    matched = d / "matched.mlir"
    rep = d / "report.json"
    if E.sh([E.PY, CORR / "lib_v2/match_v2.py", debuf, "-o", matched, "--report", rep], log):
        row["status"] = "match_fail"
        return row
    report = json.loads(rep.read_text())
    row["linalg_launches"] = [l["spec"] for l in report["launches"] if "spec" in l]
    row["region_launches"] = [l["region_match"] for l in report["launches"] if "region_match" in l]
    row["region_bindings"] = [l.get("binding") for l in report["launches"] if "region_match" in l]
    mtext = matched.read_text()
    if mutate:
        mtext, row["mutation"] = mutate_first_launch(mtext)
    tensor_syms = sorted(set(re.findall(r"=\s*kernel\.launch\s+@([\w$.-]+)", mtext)))
    lowered, called = lower_region_launches(mtext)
    row["called_spec_functions"] = called
    if tensor_syms:
        import inject_kernel_library as IKL
        lowered = IKL.inject(lowered, LIB.read_text(), tensor_syms)
    pre = d / "region_lowered_pre.mlir"
    pre.write_text(lowered)
    inl = d / "region_inlined.mlir"
    if E.sh([R.POLYGEIST_OPT, "--lower-kernel-launch", pre, "-o", inl], log):
        row["v2"] = {"verdict": "LOWER_KERNEL_LAUNCH_FAIL"}
        row["status"] = "ok"
        return row
    ll = E.lower(inl, d, "region", fn, log)
    if ll is None:
        row["v2"] = {"verdict": "LOWER_FAIL"}
        row["status"] = "ok"
        return row
    exe = d / "region_exe"
    if E.sh([E.CLANG, "-O1", *objs, ll, "-lm", f"-L{E.MLIR_LIBDIR}",
             f"-Wl,-rpath,{E.MLIR_LIBDIR}", "-lmlir_c_runner_utils", "-o", exe], log):
        row["v2"] = {"verdict": "LINK_FAIL"}
        row["status"] = "ok"
        return row
    outp = d / "region.out"
    with outp.open("w") as fh:
        subprocess.run([exe], stderr=fh, stdout=subprocess.DEVNULL, timeout=600)
    row["v2"] = E.compare(d / "ref.out", outp)
    row["status"] = "ok"
    (d / "result.json").write_text(json.dumps(row, indent=1))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="MINI")
    ap.add_argument("--out", default=str(REPO / "issues/matcher_features/e2e_region"))
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--mutate", action="store_true",
                    help="negative control: corrupt the first region launch (must FAIL)")
    ap.add_argument("kernels", nargs="+")
    a = ap.parse_args()
    srcs = E.sources()
    out = Path(a.out) / a.dataset
    with ThreadPoolExecutor(a.j) as ex:
        rows = list(ex.map(lambda k: run_kernel(k, srcs[k], a.dataset, out, a.mutate), a.kernels))
    ok_all = True
    for r in rows:
        v = r.get("v2", {}).get("verdict")
        passed = bool(v and v.startswith("PASS"))
        ok_all &= (not passed) if a.mutate else passed
        if a.mutate:
            print(f"   mutation: {r.get('mutation')}")
        print(f"{r['kernel']:10s} {r.get('status'):10s} {v} maxrel={r.get('v2', {}).get('max_rel')} "
              f"linalg={r.get('linalg_launches')} region={r.get('region_launches')}")
    summary = out / "summary.json"
    old = json.loads(summary.read_text()) if summary.exists() else {}
    old.update({r["kernel"] + ("_mutated" if a.mutate else ""): r for r in rows})
    summary.write_text(json.dumps(old, indent=1))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
