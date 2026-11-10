#!/usr/bin/env python3
"""Rebuild + numerically validate + translation-validate all lib_v2 specs into a
scratch directory (never touches generic_solver/lib_v2/raised).

The tool directory is selected with POLYGEIST_BIN_DIR (see recipe.py).

usage: POLYGEIST_BIN_DIR=.../bin specs_scratch.py OUTDIR [-j N] [--only SPEC...]
Writes OUTDIR/raised/*, OUTDIR/build_results.json, OUTDIR/spec_validation.md,
OUTDIR/summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/correctness/lib_v2"))
sys.path.insert(0, str(REPO / "scripts/correctness"))
import recipe as R  # noqa: E402
import build_lib_v2 as B  # noqa: E402
import validate_specs as V  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("-j", type=int, default=12)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--src", default=str(REPO / "generic_solver/lib_v2"),
                    help="spec source dir containing manifest/ and ref_c/")
    a = ap.parse_args()
    src = Path(a.src).resolve()
    out = Path(a.out).resolve()
    raised = out / "raised"
    raised.mkdir(parents=True, exist_ok=True)
    # scratch spec-dir layout for translation_validate.py
    for sub in ("manifest", "ref_c"):
        link = out / sub
        if not link.exists():
            link.symlink_to(src / sub)
    # redirect every output path of the (unmodified) lib_v2 modules
    B.MANIFEST = src / "manifest"
    B.REF_C = src / "ref_c"
    B.RAISED = raised
    B.STATUS_MD = out / "RAISE_STATUS.md"
    B.LIBRARY_OUT = out / "kernel_library_v2.mlir"
    V.RAISED = raised
    V.WORK = raised / "validate"
    print("tools:", R.CGEIST, R.POLYGEIST_OPT, file=sys.stderr)
    specs = B.load_specs()
    if a.only:
        specs = [s for s in specs if s.name in a.only or s.routine in a.only]
    rec = R.PRODUCTION
    with ThreadPoolExecutor(a.j) as ex:
        results = list(ex.map(lambda s: B.build_one(s, rec, a.timeout, True), specs))
    B.validate_results(results, {s.name: s for s in specs}, a.j)
    (out / "build_results.json").write_text(json.dumps(
        [{k: v for k, v in r.items() if k not in ("defn", "stderr")} for r in results], indent=1))
    tv = subprocess.run([sys.executable, str(REPO / "scripts/correctness/region_match/translation_validate.py"),
                         "--spec-dir", str(out), "--out", str(out / "spec_validation.md")],
                        capture_output=True, text=True)
    tvtext = (out / "spec_validation.md").read_text() if (out / "spec_validation.md").exists() else tv.stderr
    summary = {}
    for r in results:
        v = r.get("validation") or {}
        summary[r["spec"]] = {"status": r["status"], "defn": v.get("defn"), "linalg": v.get("linalg"),
                              "generics": (r.get("analysis") or {}).get("generics"),
                              "loops": (r.get("analysis") or {}).get("loops")}
    for ln in tvtext.splitlines():
        if ln.startswith("| `"):
            name = ln.split("`")[1]
            if name in summary:
                summary[name]["tv"] = ("NOT_EQUIV" if "NOT equivalent" in ln else
                                       "EQUIV_SEQ_ONLY" if "memref (sequential" in ln else
                                       "EQUIV" if "equivalent to cgeist" in ln else
                                       "SINGLE" if "single stage" in ln else "NA")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    by = {}
    for s in summary.values():
        by[s["status"]] = by.get(s["status"], 0) + 1
    tvc = {}
    for s in summary.values():
        tvc[s.get("tv", "NA")] = tvc.get(s.get("tv", "NA"), 0) + 1
    print("status:", by)
    print("translation validation:", tvc)
    print((tvtext.splitlines() or [""])[2] if len(tvtext.splitlines()) > 2 else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
