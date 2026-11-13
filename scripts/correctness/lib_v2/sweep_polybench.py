#!/usr/bin/env python3
"""Run match_v2 over the PolyBench kernels and summarise against production.

Usage: sweep_polybench.py [--raised DIR] [--out DIR] [-j N] [kernel ...]

Inputs:  <raised>/<kernel>/debuf.mlir   (default: issues/lib_v2/polybench_raised,
         produced by raise_polybench.py with the shared recipe)
Outputs: <out>/<kernel>/{matched.mlir,report.json} and <out>/summary.json
Production baseline: issues/polybench_section42/ir/<kernel>/**/matched.mlir
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MANIFEST = REPO / "issues/polybench_section42/manifest.csv"
PROD = REPO / "issues/polybench_section42/ir"


def kernels() -> list[str]:
    with MANIFEST.open() as fh:
        return [r["kernel"] for r in csv.DictReader(fh)]


def production(kernel: str) -> dict:
    out = {}
    for p in sorted((PROD / kernel).rglob("matched.mlir")):
        syms = re.findall(r"kernel\.launch\s+@([\w$.\-]+)", p.read_text())
        out[str(p.relative_to(PROD / kernel))] = syms
    return out


def one(kernel: str, raised: Path, out: Path, timeout: int, extra_args: list) -> dict:
    src = raised / kernel / "debuf.mlir"
    d = out / kernel
    d.mkdir(parents=True, exist_ok=True)
    row = {"kernel": kernel, "production": production(kernel)}
    if not src.exists():
        row.update(status="no_input")
        return row
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, str(HERE / "match_v2.py"), str(src),
                            "-o", str(d / "matched.mlir"), "--report", str(d / "report.json"),
                            "--diagnose", *extra_args], capture_output=True, text=True,
                           timeout=timeout)
        rc, err = p.returncode, p.stderr
    except subprocess.TimeoutExpired:
        rc, err = -9, "timeout"
    row["seconds"] = round(time.time() - t0, 1)
    if rc != 0:
        row.update(status="error", error=err[-2000:])
        (d / "error.txt").write_text(err)
        return row
    rep = json.loads((d / "report.json").read_text())
    row["status"] = "ok"
    row["v2"] = [l["spec"] for l in rep["launches"] if "spec" in l]
    row["v2_detail"] = [l for l in rep["launches"] if "spec" in l]
    row["region"] = [l for l in rep["launches"] if "region_match" in l]
    row["alternatives"] = rep.get("alternatives", [])
    row["rejected"] = rep.get("rejected", [])
    row["residual"] = rep.get("residual_generics", [])
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raised", default=str(REPO / "issues/lib_v2/polybench_raised"))
    ap.add_argument("--out", default=str(REPO / "issues/lib_v2/polybench_matched"))
    ap.add_argument("-j", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--no-region-match", action="store_true")
    ap.add_argument("--region-budget", type=int, default=600)
    ap.add_argument("only", nargs="*")
    a = ap.parse_args()
    ks = [k for k in kernels() if not a.only or k in a.only]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.j) as ex:
        extra = (["--no-region-match"] if a.no_region_match
                 else ["--region-budget", str(a.region_budget)])
        rows = list(ex.map(lambda k: one(k, Path(a.raised), out, a.timeout, extra), ks))
    for r in rows:
        prod = sorted({s for v in r["production"].values() for s in v})
        print(f"{r['kernel']:16s} {r.get('status'):6s} {r.get('seconds', '')!s:>6}s  "
              f"v2={r.get('v2', [])}  prod={prod}")
    if not a.only:
        (out / "summary.json").write_text(json.dumps(rows, indent=1, default=str))
    else:
        for r in rows:
            (out / r["kernel"] / "row.json").write_text(json.dumps(r, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
