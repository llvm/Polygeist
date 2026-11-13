#!/usr/bin/env python3
"""Region-matcher test suite (positives + negatives) with metrics.

Every C test file is raised with scripts/correctness/lib_v2/recipe.py (the
production recipe shared with the library specs).  PolyBench kernels are
raised fresh with the PolyBench flags of raise_polybench.py and are also
checked from the saved campaign IR in issues/polybench_section42/ir/.

Usage:  /usr/bin/python3 scripts/correctness/region_match/tests/run_tests.py
            [--no-raise] [--json OUT] [--only NAME ...]
Exit status 0 iff every expectation holds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RM = HERE.parent
CORR = RM.parent
REPO = CORR.parents[1]
sys.path.insert(0, str(CORR))
sys.path.insert(0, str(CORR / "lib_v2"))
sys.path.insert(0, str(HERE))

import recipe as R                                   # noqa: E402  (shared raise recipe)
from raise_polybench import polybench_cflags          # noqa: E402
from region_match.matcher import (load_specs, match_text, compare_systems, Binding,  # noqa: E402
                                  _used_params, _identity_fparams)
from region_match.mlir_parse import parse_module      # noqa: E402
from region_match.symexec import execute              # noqa: E402
from region_match.equations import build_system       # noqa: E402
from numeric_check import check as numeric_check       # noqa: E402

# The suite is written against the netlib specs; match_v2.py (run as a
# subprocess) reads the same POLYGEIST_SPEC_SUITE, so pin it for both.
os.environ.setdefault("POLYGEIST_SPEC_SUITE", "lib_v2")
SPEC_DIR = REPO / "generic_solver" / os.environ["POLYGEIST_SPEC_SUITE"]
OUT = HERE / "raised"
PB = REPO / "tools/cgeist/Test/polybench/linear-algebra/solvers"
PB_BLAS = REPO / "tools/cgeist/Test/polybench/linear-algebra/blas"
PB_KERNELS = REPO / "tools/cgeist/Test/polybench/linear-algebra/kernels"
SAVED = REPO / "issues/polybench_section42/ir"
PB_RAISED = REPO / "issues/lib_v2/polybench_raised"
C = HERE / "c"

UTN, UTU, LTN, LNN = ("netlib_dtrsv_utn", "netlib_dtrsv_utu", "netlib_dtrsv_ltn",
                      "netlib_dtrsv_lnn")
POTF2_U = "netlib_dpotf2_u"

# name, kind, source, function, expectation
#   expect["match"]   : spec that must be selected (None = no match at all)
#   expect["not"]     : specs that must not be selected
#   expect["flag"]    : substring that must appear in the evidence
TESTS = [
    # ---- trsv positives
    ("pb_trisolv_saved", "saved", SAVED / "trisolv/raised_debufferized.mlir", "kernel_trisolv",
     {"match": UTN, "not": [UTU]}),
    ("pb_trisolv_fresh", "polybench", PB / "trisolv/trisolv.c", "kernel_trisolv",
     {"match": UTN, "not": [UTU]}),
    ("trsv_axpy", "c", HERE / "c/trsv_axpy.c", "trsv_axpy", {"match": UTN, "not": [UTU]}),
    ("trsv_temp", "c", HERE / "c/trsv_temp.c", "trsv_temp", {"match": UTN, "not": [UTU]}),
    ("trsv_unit", "c", HERE / "c/trsv_unit.c", "trsv_unit", {"match": UTU, "not": [UTN]}),
    ("trsv_upper_back", "c", HERE / "c/trsv_upper_back.c", "trsv_upper_back",
     {"match": LTN, "not": [UTN]}),
    ("trsv_inplace (b aliases x)", "c", HERE / "c/trsv_inplace.c", "trsv_inplace",
     {"match": UTN, "flag": "in-place"}),
    # ---- cholesky positives
    ("pb_cholesky_saved", "saved", SAVED / "cholesky/raised_debufferized.mlir", "kernel_cholesky",
     {"match": POTF2_U}),
    ("pb_cholesky_fresh", "polybench", PB / "cholesky/cholesky.c", "kernel_cholesky",
     {"match": POTF2_U}),
    ("cholesky_crout (column-by-column)", "c", HERE / "c/cholesky_crout.c", "cholesky_crout",
     {"match": POTF2_U}),
    ("cholesky_right (right-looking)", "c", HERE / "c/cholesky_right.c", "cholesky_right",
     {"match": POTF2_U}),
    ("cholesky_upper (row-major U^T U)", "c", HERE / "c/cholesky_upper.c", "cholesky_upper",
     {"match": "netlib_dpotf2_l"}),
    # ---- negatives
    ("NEG trsv_transposed L[j][i]", "c", HERE / "c/trsv_transposed.c", "trsv_transposed",
     {"not": [UTN, UTU]}),
    ("NEG trsv_swapdiv L[i][i]/x[i]", "c", HERE / "c/trsv_swapdiv.c", "trsv_swapdiv",
     {"match": None}),
    ("NEG trsv_mask_le j<=i", "c", HERE / "c/trsv_mask_le.c", "trsv_mask_le", {"match": None}),
    ("NEG trsv_alias_shift b=x+1", "c", HERE / "c/trsv_alias_shift.c", "trsv_alias_shift",
     {"match": None}),
    ("NEG trsv_reads_b x-=L*b", "c", HERE / "c/trsv_reads_b.c", "trsv_reads_b", {"match": None}),
    ("NEG trsv_div_first", "c", HERE / "c/trsv_div_first.c", "trsv_div_first", {"match": None}),
    ("NEG ldlt", "c", HERE / "c/ldlt.c", "ldlt", {"match": None}),
    ("NEG cholesky_wrong_tri A[k][j]", "c", HERE / "c/cholesky_wrong_tri.c",
     "cholesky_wrong_tri", {"match": None}),
    # ---- feature 1: literal scalar binding (e-graph union with Term.Lit)
    ("lit axpy_unit (da=1.0)", "c", C / "axpy_unit.c", "axpy_unit",
     {"match": "netlib_daxpy_inc", "flag": "bound to literals"}),
    ("lit axpy_const (da=2.5)", "c", C / "axpy_const.c", "axpy_const",
     {"match": "netlib_daxpy_inc", "flag": "2.5"}),
    ("lit ger_unit (alpha=1.0)", "c", C / "ger_unit.c", "ger_unit",
     {"match": "netlib_dger", "flag": "bound to literals"}),
    ("lit gemm_beta1 (beta!=1 policy)", "c", C / "gemm_beta1.c", "gemm_beta1",
     {"match": "netlib_dgemm_nn_beta1", "not": ["netlib_dgemm_nn"]}),
    ("NEG lit zero_fill (no da=0.0)", "c", C / "zero_fill.c", "zero_fill", {"match": None}),
    ("NEG lit scal_zero x=0.0*x", "c", C / "scal_zero.c", "scal_zero", {"match": None}),
    # ---- feature 2: composition of two specifications
    ("comp axpby = DSCAL;DAXPY", "c", C / "axpby.c", "axpby",
     {"match": "netlib_dscal_inc+netlib_daxpy_inc", "flag": "sequential composition"}),
    ("comp axpby_lit = DSCAL(2);DAXPY(1)", "c", C / "axpby_lit.c", "axpby_lit",
     {"match": "netlib_dscal_inc+netlib_daxpy_inc"}),
    ("comp rank2 = DGER;DGER", "c", C / "rank2.c", "rank2",
     {"match": "netlib_dger+netlib_dger", "flag": "sequential composition"}),
    ("comp pb_gesummv (after linalg)", "combined", PB_RAISED / "gesummv/debuf.mlir", "kernel_gesummv",
     {"match": "netlib_dscal_inc+netlib_daxpy_inc", "flag": "copied into memory"}),
    ("comp pb_gemver (after linalg)", "combined", PB_RAISED / "gemver/debuf.mlir", "kernel_gemver",
     {"match": "netlib_dger+netlib_dger", "flag": "re-defined after the launch"}),
    ("NEG copy-in alias y=2y+x (no DAXPY dx:=y)", "c", C / "axpby_lit.c", "axpby_lit",
     {"not": ["netlib_daxpy_inc"]}),
    ("NEG comp axpby_prod", "c", C / "axpby_prod.c", "axpby_prod", {"match": None}),
    ("NEG comp axpby_shift y[i+1]", "c", C / "axpby_shift.c", "axpby_shift", {"match": None}),
    ("NEG comp rank2_bad u2[j]*v2[j]", "c", C / "rank2_bad.c", "rank2_bad", {"match": None}),
    # ---- feature 3: leading dimension from the memref stride
    ("ld trsv_lda (ld > n)", "c", C / "trsv_lda.c", "trsv_lda",
     {"match": UTN, "flag": "stride0", "launch_has": ["memref.extract_strided_metadata"],
      "launch_not": ["memref.dim"]}),
    # ---- feature 4: layout-transposed bindings emitted via manifest metadata
    ("win pb_ludcmp back-substitution", "polybench", PB / "ludcmp/ludcmp.c", "kernel_ludcmp",
     {"match": "netlib_dtrsv_ltn", "no_saved_compare": True,
      "numeric_prefix": (C / "ludcmp_prefix.c", "kernel_ludcmp_prefix")}),
    ("tr pb_syr2k -> dsyr2k_ut", "polybench", PB_BLAS / "syr2k/syr2k.c", "kernel_syr2k",
     {"match": "netlib_dsyr2k_ut", "flag": "layout_transpose", "no_saved_compare": True}),
    ("tr pb_trmm -> dtrmm_rutu", "polybench", PB_BLAS / "trmm/trmm.c", "kernel_trmm",
     {"match": "netlib_dtrmm_rutu", "flag": "layout_transpose", "no_saved_compare": True}),
    ("NEG tr syr2k_halfT (B only)", "c", C / "syr2k_halfT.c", "syr2k_halfT", {"match": None}),
    ("NEG tr syr2k_beta1 (no flipped spec)", "c", C / "syr2k_beta1.c", "syr2k_beta1",
     {"match": None}),
]


def linalg_then(src: Path, name: str) -> Path:
    """The production driver order: Linalg-level matches are spliced first
    (match_v2.py, separate process: it binds kernel_match to its library),
    the region matcher then sees the spliced text."""
    import subprocess
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / (name.split()[1] + "_after_linalg.mlir")
    subprocess.run([sys.executable, str(CORR / "lib_v2/match_v2.py"), str(src), "-o", str(out),
                    "--no-region-match"], check=True, capture_output=True, text=True)
    return out


def raise_one(name: str, kind: str, src: Path, fn: str, do_raise: bool) -> tuple[Path | None, dict]:
    if kind == "saved":
        return src, {"recipe": "saved campaign artifact"}
    if kind == "combined":
        return linalg_then(src, name), {"recipe": "lib_v2 PolyBench raise + match_v2 linalg splice"}
    stem = OUT / fn
    debuf = Path(f"{stem}_debuf.mlir")
    info = {"recipe": R.PRODUCTION.name}
    if do_raise or not debuf.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        flags = polybench_cflags(src) if kind == "polybench" else []
        res = R.raise_c(R.PRODUCTION, str(src), fn, str(stem), flags)
        info["commands"] = [R.shell_line(res[s].cmd).replace(str(REPO) + "/", "")
                            for s in ("cgeist", "raise", "debuf") if s in res]
        for s in ("cgeist", "raise", "debuf"):
            if s in res and not res[s].ok:
                info["raise_failure"] = f"{s}: {res[s].stderr[-400:]}"
                return None, info
    return debuf, info


def system_of(path: Path, fn: str):
    funcs, aliases = parse_module(path.read_text())
    f = next(f for f in funcs if f.name == fn)
    prog = execute(f, aliases)
    window = {i for i, k in enumerate(prog.top_kinds) if k in ("work", "writeback")}
    return build_system(prog, window)


def same_semantics(p1: Path, p2: Path, fn: str) -> str:
    """Saved vs freshly raised application IR: equal equation systems?"""
    s1, s2 = system_of(p1, fn), system_of(p2, fn)
    arrays = {a: (a, a if a in s1.outputs else None, tuple(range(s1.arrays[a].rank)))
              for a in s1.arrays}
    params = {p: p for p in _used_params(s1) | _used_params(s2)}
    r = compare_systems(s2, s1, Binding(arrays, params, _identity_fparams(s1, s2)))
    return "equivalent" if r.ok else f"DIFFERENT: {r.reason[:200]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-raise", action="store_true")
    ap.add_argument("--json", default=str(REPO / "issues/region_match/results.json"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    t0 = time.perf_counter()
    specs = load_specs(str(SPEC_DIR))
    load_ms = (time.perf_counter() - t0) * 1000
    results = []
    ok_all = True
    print(f"{len(specs)} specifications loaded in {load_ms:.0f} ms")
    for (name, kind, src, fn, expect) in TESTS:
        if a.only and not any(o in name for o in a.only):
            continue
        path, info = raise_one(name, kind, src, fn, not a.no_raise)
        row = {"test": name, "source": str(src.relative_to(REPO)), "function": fn,
               "expect": expect, **info}
        if path is None:
            row.update(selected=None, ok=expect.get("match", "x") is None,
                       reason="raise failed")
            results.append(row)
            ok_all &= row["ok"]
            print(f"{'PASS' if row['ok'] else 'FAIL'}  {name:38s} raise failed")
            continue
        stats: dict = {}
        t1 = time.perf_counter()
        matches = match_text(path.read_text(), str(SPEC_DIR), None, stats)
        ms = (time.perf_counter() - t1) * 1000
        sel = [m.spec for m in matches]
        selected = sel[0] if sel else None
        also = []
        for w in stats.get("windows", []):
            if w.get("matched"):
                also += w.get("also", [])
        ok = True
        if "match" in expect:
            ok &= selected == expect["match"]
        for bad in expect.get("not", []):
            ok &= bad not in sel
        if "flag" in expect:
            ok &= any(expect["flag"] in m.evidence for m in matches)
        for frag in expect.get("launch_has", []):
            ok &= any(frag in m.launch_text for m in matches)
        for frag in expect.get("launch_not", []):
            ok &= all(frag not in m.launch_text for m in matches)
        win = next((w for w in stats.get("windows", []) if w.get("matched")), {})
        row.update(selected=selected, also_equal=also, ok=ok, match_ms=round(ms, 1),
                   windows=stats.get("windows", []), rejections=stats.get("rejections", []),
                   evidence=[m.evidence for m in matches], binding=[m.binding for m in matches],
                   launch=[m.launch_text for m in matches],
                   egraph_nodes_max=max(win.get("egraph_nodes", [0]) or [0]),
                   egg_ms_total=round(sum(win.get("egg_ms", [])), 1))
        if matches and kind == "combined":
            # mid-function window (operands materialised from tensors): checked
            # end-to-end in the PolyBench harness by tests/e2e_region.py
            row["numeric"] = {"ok": True, "deferred": "tests/e2e_region.py"}
        if matches and kind in ("c", "polybench", "saved"):
            csrc = src if kind != "saved" else PB / fn.replace("kernel_", "") / (fn.replace("kernel_", "") + ".c")
            flags = polybench_cflags(csrc) if kind in ("polybench", "saved") else []
            try:
                row["numeric"] = numeric_check(csrc, fn, matches[0].launch_text, matches[0].spec,
                                               flags, preprocess=kind in ("polybench", "saved"),
                                               prefix=expect.get("numeric_prefix"))
            except Exception as ex:  # noqa: BLE001
                row["numeric"] = {"ok": False, "error": f"{type(ex).__name__}: {ex}"}
            ok &= bool(row["numeric"].get("ok"))
            row["ok"] = ok
        if kind == "polybench" and not expect.get("no_saved_compare"):
            saved = SAVED / fn.replace("kernel_", "") / "raised_debufferized.mlir"
            if saved.exists():
                row["saved_vs_fresh"] = same_semantics(saved, path, fn)
                ok &= row["saved_vs_fresh"] == "equivalent"
                row["ok"] = ok
        ok_all &= ok
        results.append(row)
        why = ""
        if not selected:
            why = (stats.get("rejections") or ["no window analysable"])[0][:110]
        else:
            why = f"also={also}" if also else ""
        num = row.get("numeric")
        numtxt = "" if not num else ("num: e2e" if num.get("deferred") else
                                     f"num-err={num.get('max_abs_err', float('nan')):.1e}"
                                     if num.get("ok") else f"NUMERIC FAIL {num.get('error', num)}"[:90])
        print(f"{'PASS' if ok else 'FAIL'}  {name:38s} -> {str(selected):34s} "
              f"{ms:7.0f} ms  nodes={row['egraph_nodes_max']:4d}  {numtxt}  {why}")
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps({"spec_load_ms": round(load_ms, 1),
                                        "spec_validation": {s.name: s.validation for s in specs
                                                            if s.routine in ("dtrsv", "dpotf2")},
                                        "results": results}, indent=1, default=str))
    print(f"{'ALL PASS' if ok_all else 'SOME FAILED'}; results in {a.json}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
