#!/usr/bin/env python3
"""Re-raise the 30 PolyBench kernels with the lib_v2 recipe (recipe.py).

Usage: raise_polybench.py [--recipe production|campaign] [--out DIR] [kernel ...]

Writes <out>/<kernel>/{affine,linalg,debuf}.mlir (+ raise_status.json).
Flags are identical to the ones build_lib_v2.py uses for library specs; the
only additional flags are PolyBench's own include/define passthrough taken
from the paper campaign (polygeist_build.sh gcc passthrough).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipe as R  # noqa: E402

REPO = R.REPO
MANIFEST = REPO / "issues/polybench_section42/manifest.csv"
UTIL = REPO / "tools/cgeist/Test/polybench/utilities"


def kernels() -> list[tuple[str, Path]]:
    out = []
    with MANIFEST.open() as fh:
        for row in csv.DictReader(fh):
            out.append((row["kernel"], REPO / row["source"]))
    return out


def polybench_cflags(src: Path) -> list[str]:
    return [f"-I{UTIL}", f"-I{src.parent}", "-Dstatic=__attribute__((noipa))",
            "-DLARGE_DATASET", "-DDATA_TYPE_IS_DOUBLE",
            "-DPOLYBENCH_USE_C99_PROTO", "-DPOLYBENCH_DUMP_ARRAYS"]


def one(kernel: str, src: Path, out: Path, rec: R.Recipe, timeout: int) -> dict:
    d = out / kernel
    d.mkdir(parents=True, exist_ok=True)
    fn = "kernel_" + kernel.replace("-", "_")
    res = R.raise_c(rec, str(src), fn, str(d / "affine"), polybench_cflags(src), timeout)
    # Normalise artifact names to affine/linalg/debuf.mlir.
    for key, name in (("linalg", "affine_linalg.mlir"), ("debuf", "affine_debuf.mlir")):
        p = d / name
        if p.exists():
            p.replace(d / f"{key}.mlir")
    status = {"kernel": kernel, "function": fn, "recipe": rec.name}
    for step in ("cgeist", "raise", "debuf"):
        r = res.get(step)
        if r is None:
            status[step] = "skipped"
            continue
        status[step] = "ok" if r.ok else ("timeout" if r.timed_out else f"rc={r.rc}")
        status[f"{step}_cmd"] = R.shell_line(r.cmd)
        if not r.ok:
            status[f"{step}_stderr"] = r.stderr
    (d / "raise_status.json").write_text(json.dumps(status, indent=1))
    return status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", default="production", choices=sorted(R.RECIPES))
    ap.add_argument("--out", default=str(REPO / "issues/lib_v2/polybench_raised"))
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("-j", type=int, default=6)
    ap.add_argument("only", nargs="*")
    a = ap.parse_args()
    out = Path(a.out)
    rec = R.RECIPES[a.recipe]
    todo = [(k, s) for k, s in kernels() if not a.only or k in a.only]
    with ThreadPoolExecutor(a.j) as ex:
        for st in ex.map(lambda ks: one(ks[0], ks[1], out, rec, a.timeout), todo):
            print(st["kernel"], st["cgeist"], st["raise"], st["debuf"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
