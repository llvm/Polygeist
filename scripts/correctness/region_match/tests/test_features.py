#!/usr/bin/env python3
"""Unit checks for the matcher features that the C-level suite
(run_tests.py) cannot reach directly.  Positive and negative cases:

  cache      disk cache of parsed specs: miss -> hit in a fresh process with
             identical specs; key changes on a raised-file / manifest edit and
             is stable otherwise (feature 5)
  transpose  every proved layout-transposed alternative, mapped through the
             manifest `layout_transpose` metadata, equals the identity-binding
             match the matcher selects (same spec, same operands); an operand
             set no rule covers is rejected (feature 4)
  stride     leading dimensions come from memref strides in both matchers:
             unit inner stride -> stride0, dynamic inner stride ->
             stride0/stride1 (arith.divui); never memref.dim (feature 3)
  literal    both matchers use the same policy object; beta := 1 for a
             general-beta spec and da := 0.0 are refused (feature 1)

Usage: test_features.py [--json OUT] [--only NAME ...]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RM = HERE.parent
CORR = RM.parent
REPO = CORR.parents[1]
SPEC_DIR = REPO / "generic_solver/lib_v2"
RAISED = HERE / "raised"
PB_RAISED = REPO / "issues/lib_v2/polybench_raised"
sys.path.insert(0, str(CORR))

PY = sys.executable
RESULTS: list[dict] = []


def record(name: str, ok: bool, **info):
    RESULTS.append({"test": name, "ok": bool(ok), **info})
    print(f"{'PASS' if ok else 'FAIL'}  {name:58s} {json.dumps(info, default=str)[:150]}")


# ---------------------------------------------------------------------------
# feature 5: cache
# ---------------------------------------------------------------------------

_LOAD = """
import sys, time, json
sys.path.insert(0, {corr!r})
t = time.perf_counter()
from region_match.matcher import load_specs, _cache_key
specs = load_specs({sd!r})
print(json.dumps({{"ms": (time.perf_counter() - t) * 1000, "n": len(specs),
                  "names": [s.name for s in specs], "val": [s.validation for s in specs],
                  "key": _cache_key({sd!r})}}))
"""


def _load(spec_dir: Path, cache: Path, extra_env=None) -> dict:
    env = dict(os.environ, REGION_MATCH_CACHE_DIR=str(cache), **(extra_env or {}))
    out = subprocess.run([PY, "-c", _LOAD.format(corr=str(CORR), sd=str(spec_dir))], env=env,
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_cache():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sd = td / "lib_v2"
        for sub in ("manifest", "ref_c", "raised"):
            shutil.copytree(SPEC_DIR / sub, sd / sub)
        cache = td / "cache"
        miss = _load(sd, cache)
        hit = _load(sd, cache)
        nofile = _load(sd, cache, {"REGION_MATCH_NO_CACHE": "1"})
        record("cache: hit in a fresh process equals the uncached load",
               hit["names"] == miss["names"] == nofile["names"] and hit["val"] == miss["val"],
               miss_ms=round(miss["ms"]), hit_ms=round(hit["ms"]), nocache_ms=round(nofile["ms"]),
               specs=miss["n"])
        record("cache: hit is >= 5x faster than a miss", hit["ms"] * 5 < miss["ms"],
               miss_ms=round(miss["ms"]), hit_ms=round(hit["ms"]))
        record("NEG cache: key stable when nothing changes", hit["key"] == miss["key"])
        files_before = len(list(cache.glob("*.pkl")))
        f = sd / "raised" / "netlib_dtrsv_utn.mlir"
        f.write_text(f.read_text() + "\n// touched\n")
        touched = _load(sd, cache)
        record("cache: editing a raised spec file invalidates (new key, rebuilt)",
               touched["key"] != miss["key"] and len(list(cache.glob("*.pkl"))) == files_before + 1,
               rebuilt_ms=round(touched["ms"]))
        m = sd / "manifest" / "dscal.json"
        d = json.loads(m.read_text())
        d["specs"][0]["assumptions"] = d["specs"][0].get("assumptions", []) + ["touched"]
        m.write_text(json.dumps(d, indent=2))
        touched2 = _load(sd, cache)
        record("cache: editing a manifest invalidates", touched2["key"] != touched["key"])


# ---------------------------------------------------------------------------
# feature 4: transposed equivalences agree with direct identity proofs
# ---------------------------------------------------------------------------

def _app_system(path: Path, fn: str):
    from region_match.equations import build_system
    from region_match.matcher import _windows
    from region_match.mlir_parse import parse_module
    from region_match.symexec import execute
    funcs, aliases = parse_module(path.read_text())
    f = next(f for f in funcs if f.name == fn)
    prog = execute(f, aliases)
    lo, hi, end = _windows(prog)[0]
    return build_system(prog, set(range(lo, end + 1)))


def test_transpose():
    from region_match.equations import Reject
    from region_match.matcher import (Binding, _single_search, load_specs,
                                      transposed_equivalent)
    specs = load_specs(str(SPEC_DIR))
    cases = [(RAISED / "kernel_trisolv_debuf.mlir", "kernel_trisolv"),
             (RAISED / "trsv_axpy_debuf.mlir", "trsv_axpy"),
             (RAISED / "trsv_upper_back_debuf.mlir", "trsv_upper_back"),
             (RAISED / "kernel_cholesky_debuf.mlir", "kernel_cholesky"),
             (RAISED / "cholesky_upper_debuf.mlir", "cholesky_upper"),
             (RAISED / "trsv_transposed_debuf.mlir", "trsv_transposed")]
    for path, fn in cases:
        app = _app_system(path, fn)
        found = _single_search(app, specs, [])
        found.sort(key=lambda x: (x["cost"], x["key"]))
        ident = [f for f in found if f["cost"][0] == 0]
        trans = [f for f in found if f["cost"][0] > 0]
        if not ident or not trans:
            record(f"transpose: {fn} has identity and transposed proofs", False,
                   ident=len(ident), trans=len(trans))
            continue
        best = ident[0]
        agree = []
        for t in trans:
            ls, b2, _note = transposed_equivalent(t["spec"], t["b"])
            same_spec = ls.name == best["spec"].name
            ops = {t["spec"].argnames[k]: v[:2] for k, v in b2.arrays.items()}
            ops_best = {best["spec"].argnames[k]: v[:2] for k, v in best["b"].arrays.items()}
            agree.append((t["key"], ls.name, same_spec and ops == ops_best))
        record(f"transpose: {fn}: manifest equivalent == identity proof",
               all(a[2] for a in agree), identity=best["spec"].name, derived=agree)
    # negative: an operand set that no rule covers
    sp = next(s for s in specs if s.name == "netlib_dsyr2k_un")
    ssa = {c: s_ for s_, c in sp.argnames.items()}
    arrays = {ssa["A"]: ("%a", None, (1, 0)), ssa["B"]: ("%b", None, (0, 1)),
              ssa["C"]: ("%c", "%c", (0, 1))}
    try:
        transposed_equivalent(sp, Binding(arrays, {}, {}))
        record("NEG transpose: A alone transposed for dsyr2k is rejected", False)
    except Reject as ex:
        record("NEG transpose: A alone transposed for dsyr2k is rejected", True, why=str(ex)[:90])
    # positive: the rules compose (A and B transposed for dtrmm -> side, uplo flip, m<->n)
    sp = next(s for s in specs if s.name == "netlib_dtrmm_lltu")
    ssa = {c: s_ for s_, c in sp.argnames.items()}
    arrays = {ssa["A"]: ("%a", None, (1, 0)), ssa["B"]: ("%b", "%b", (1, 0))}
    params = {f"p{ssa['m']}": "pM", f"p{ssa['n']}": "pN"}
    ls, b2, _ = transposed_equivalent(sp, Binding(arrays, params, {ssa["alpha"]: "%al"}))
    record("transpose: dtrmm_lltu with A,B transposed -> dtrmm_rutu, m<->n",
           ls.name == "netlib_dtrmm_rutu" and b2.params[f"p{ssa['m']}"] == "pN"
           and b2.params[f"p{ssa['n']}"] == "pM", got=ls.name)


# ---------------------------------------------------------------------------
# feature 3: leading dimension from the stride
# ---------------------------------------------------------------------------

def _retype(text: str, arg: str, new_ty: str) -> str:
    """Give function argument `arg` a strided memref type everywhere."""
    old_ty = re.search(re.escape(arg) + r":\s*(memref<[^>]*>)", text).group(1)
    text = text.replace(f"{arg}: {old_ty}", f"{arg}: {new_ty}")
    return re.sub(r"(bufferization\.to_tensor " + re.escape(arg) + r"\s*:\s*)" + re.escape(old_ty),
                  r"\g<1>" + new_ty, text)


def test_stride_region():
    from region_match.matcher import match_text
    base = (RAISED / "trsv_axpy_debuf.mlir").read_text()
    arg = "%arg1"            # L
    for layout, expect_div in (("strided<[?, 1], offset: ?>", False), ("strided<[?, ?], offset: ?>", True)):
        t = _retype(base, arg, f"memref<?x?xf64, {layout}>")
        ms = match_text(t, str(SPEC_DIR))
        lt = ms[0].launch_text if ms else ""
        lda = ms[0].binding.get("lda") if ms else None
        ok = bool(ms) and f"memref.extract_strided_metadata {arg}" in lt \
            and ("arith.divui" in lt) == expect_div and "memref.dim" not in lt \
            and "memref.cast" not in lt and layout in lt.splitlines()[-1] \
            and lda == ("stride0(%arg1)/stride1(%arg1)" if expect_div else "stride0(%arg1)")
        record(f"stride region: L {layout} -> lda {lda}", ok)
    ms = match_text(base, str(SPEC_DIR))
    record("NEG stride region: identity layout never uses memref.dim for lda",
           ms and "memref.dim" not in ms[0].launch_text and ms[0].binding.get("lda") == "stride0(%arg1)")


def _match_v2(path: Path, out: Path) -> dict:
    rep = out.with_suffix(".json")
    subprocess.run([PY, str(CORR / "lib_v2/match_v2.py"), str(path), "-o", str(out), "--report",
                    str(rep), "--no-region-match"], check=True, capture_output=True, text=True)
    return json.loads(rep.read_text())


def test_stride_linalg():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        src = PB_RAISED / "gemm/debuf.mlir"
        rep = _match_v2(src, td / "gemm.mlir")
        b = rep["launches"][0]["binding"]
        txt = (td / "gemm.mlir").read_text()
        record("stride linalg: gemm lda/ldb/ldc = stride0(<memref arg>)",
               all(re.fullmatch(r"stride0\(%arg\d+\)", b[k]) for k in ("%lda", "%ldb", "%ldc"))
               and "memref.extract_strided_metadata" in txt, binding={k: b[k] for k in ("%lda", "%ldb", "%ldc")})
        a_arg = b["%A"]
        # the tensor bound to A traces back to a memref argument; give that a
        # dynamic inner stride
        t = src.read_text()
        m = re.search(r"(" + re.escape(a_arg) + r") = bufferization\.to_tensor (%arg\d+)", t)
        arg = m.group(2) if m else None
        if arg:
            t2 = _retype(t, arg, "memref<?x?xf64, strided<[?, ?], offset: ?>>")
            (td / "gemm_strided.mlir").write_text(t2)
            rep2 = _match_v2(td / "gemm_strided.mlir", td / "gemm_strided_out.mlir")
            b2 = rep2["launches"][0]["binding"] if rep2["launches"] else {}
            out2 = (td / "gemm_strided_out.mlir").read_text()
            record("stride linalg: dynamic inner stride -> stride0/stride1 (divui)",
                   b2.get("%lda") == f"stride0({arg})/stride1({arg})" and "arith.divui" in out2,
                   lda=b2.get("%lda"))
        # 2mm: B is the result of the first product (a tensor value): traced to its storage
        rep3 = _match_v2(PB_RAISED / "2mm/debuf.mlir", td / "2mm.mlir")
        lds = [l["binding"].get("%ldb") for l in rep3["launches"]]
        record("stride linalg: tensor produced in the kernel traced to its memref",
               all(x and x.startswith("stride0(%arg") for x in lds), ldb=lds)
        record("NEG stride linalg: no tensor.dim-derived leading dimension in 2mm",
               "tensor.dim" not in (td / "2mm.mlir").read_text())


# ---------------------------------------------------------------------------
# feature 1: one literal policy for both matchers
# ---------------------------------------------------------------------------

def test_literal():
    from region_match import literal_policy as SP
    ent = {}
    for mp in (SPEC_DIR / "manifest").glob("*.json"):
        for e in json.loads(mp.read_text())["specs"]:
            ent[e["name"]] = e
    checks = [
        ("netlib_dgemm_nn", "alpha", 1.0, "float", True),
        ("netlib_dgemm_nn", "beta", 1.0, "float", False),
        ("netlib_dgemm_nn", "beta", 0.0, "float", False),
        ("netlib_dscal_inc", "da", 0.0, "float", False),
        ("netlib_daxpy_inc", "da", 2.5, "float", True),
        ("netlib_daxpy_inc", "incx", 1, "int", True),
        ("netlib_dgemv_n_inc", "incy", 0, "int", False),
    ]
    for spec, p, v, kind, want in checks:
        got, why = SP.literal_ok(ent[spec], p, v, kind)
        record(f"{'' if want else 'NEG '}literal policy: {spec} {p} := {v}", got == want, why=why)
    # the Linalg-level matcher loads the very same file
    src = (CORR / "lib_v2/match_v2.py").read_text()
    record("literal: match_v2 uses region_match/literal_policy.py",
           'HERE.parent / "region_match" / "literal_policy.py"' in src and "SP.literal_ok" in src)
    with tempfile.TemporaryDirectory() as td:
        rep = _match_v2(RAISED / "gemm_beta1_debuf.mlir", Path(td) / "g.mlir")
        specs = [l["spec"] for l in rep["launches"]]
        rej = rep.get("literal_policy_rejections", [])
        record("literal linalg: C += A*B -> dgemm_nn_beta1 (alpha := 1.0), not dgemm_nn",
               specs == ["netlib_dgemm_nn_beta1"], specs=specs)
        rep = _match_v2(RAISED / "scal_zero_debuf.mlir", Path(td) / "z.mlir")
        rej = rep.get("literal_policy_rejections", [])
        record("NEG literal linalg: x = 0.0*x -> no DSCAL, da := 0.0 refused by the shared policy",
               not rep["launches"] and any("netlib_dscal_inc: da := 0.0" in r for r in rej),
               rejections=rej[:3])
    from region_match.matcher import match_text
    ms = match_text((RAISED / "scal_zero_debuf.mlir").read_text(), str(SPEC_DIR))
    record("NEG literal region: x = 0.0*x -> no DSCAL (0.0 is never a candidate literal)",
           not ms, got=[m.spec for m in ms])


TESTS = {"cache": test_cache, "transpose": test_transpose, "stride_region": test_stride_region,
         "stride_linalg": test_stride_linalg, "literal": test_literal}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(REPO / "issues/matcher_features/test_features.json"))
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    t0 = time.perf_counter()
    for name, fn in TESTS.items():
        if a.only and name not in a.only:
            continue
        try:
            fn()
        except Exception as ex:  # noqa: BLE001
            import traceback
            record(f"{name}: crashed", False, error=f"{type(ex).__name__}: {ex}",
                   tb=traceback.format_exc()[-600:])
    ok = all(r["ok"] for r in RESULTS)
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps({"seconds": round(time.perf_counter() - t0, 1),
                                        "results": RESULTS}, indent=1, default=str))
    print(f"{'ALL PASS' if ok else 'SOME FAILED'} ({sum(r['ok'] for r in RESULTS)}/{len(RESULTS)})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
