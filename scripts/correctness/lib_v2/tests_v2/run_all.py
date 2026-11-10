#!/usr/bin/env python3
"""lib_v2 test and robustness harness.

Usage (re-run everything after the raising / matcher code changes):
    source envsetup.sh
    /usr/bin/python3 scripts/correctness/lib_v2/tests_v2/run_all.py all [-j 6]

Subcommands (each writes issues/lib_v2/tests/<suite>/results.json; work
artifacts go to issues/lib_v2/tests/work/<suite>/):
    negatives    suite 1  near-miss negatives (+ positive controls)          P0
    aliasing     suite 2  explicit + call-site aliasing                       P0
    numerics     suite 3  5 seeds x edge sizes for the 12 matched kernels,
                          LARGE via e2e_v2.py, harness sensitivity controls   P0
    irvariants   suite 4  scalar IR variants (eqsat transformations)          P1
    srcvariants  suite 5  source-level C rewrites of the 12 matched kernels   P1
    ablation     suite 6  egglog vs syntactic matcher on originals + 4 + 5    P1
    all          1..6 in order, then the top-level summary
    summary      rebuild issues/lib_v2/tests/summary.json from suite results

Nothing outside scripts/correctness/lib_v2/tests_v2/ and issues/lib_v2/tests/
is written.  Raising always re-runs (current pipeline), matching always runs
fresh match_v2 code in fresh worker processes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import numeric as N  # noqa: E402

WORK = C.OUT_ROOT / "work"
SUITES = ["negatives", "aliasing", "numerics", "irvariants", "srcvariants", "ablation"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def suite_dir(name: str, fresh: bool = True) -> Path:
    d = WORK / name
    if fresh and d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    return d


def finish(name: str, t0: float, summary: dict, cases: list, extra: dict | None = None) -> dict:
    out = {"suite": name, "finished": C.now(), "seconds": round(time.time() - t0, 1),
           "summary": summary, **(extra or {}), "cases": cases}
    C.write_json(C.OUT_ROOT / name / "results.json", out)
    update_summary()
    log(f"{name}: done in {out['seconds']} s -> {C.OUT_ROOT / name / 'results.json'}")
    return out


# ---------------------------------------------------------------------------
# Shared steps
# ---------------------------------------------------------------------------

_PB_CACHE: dict[str, dict] = {}


def polybench_raised(kernels: list[str], j: int) -> dict[str, dict]:
    """Fresh raise of PolyBench kernels with the lib_v2 recipe (cached per process)."""
    todo = [k for k in kernels if k not in _PB_CACHE]
    if todo:
        log(f"raising {len(todo)} PolyBench kernels (fresh, lib_v2 recipe)")
        res = C.pmap(lambda k: C.raise_polybench(k, WORK / "polybench_raised" / k), todo, j)
        for k, r in zip(todo, res):
            stored = C.REPO / "issues/lib_v2/polybench_raised" / k / "debuf.mlir"
            if r["ok"] and stored.exists():
                r["same_as_issues_lib_v2_polybench_raised"] = (
                    Path(r["paths"]["debuf"]).read_bytes() == stored.read_bytes())
            _PB_CACHE[k] = r
    return {k: _PB_CACHE[k] for k in kernels}


_PB_MATCH: dict[str, dict] = {}


def polybench_matched(kernels: list[str], j: int) -> dict[str, dict]:
    """egglog match_v2 (+ region) results of the fresh PolyBench originals."""
    raised = polybench_raised(kernels, j)
    jobs = []
    for k in kernels:
        if k in _PB_MATCH or not raised[k]["ok"]:
            continue
        jobs.append({"id": k, "input": raised[k]["paths"]["debuf"],
                     "out_dir": str(WORK / "polybench_matched" / k), "region": True,
                     "region_diag": True, "linalg_only_inject": True})
    if jobs:
        log(f"matching {len(jobs)} PolyBench originals")
        for k, r in C.match_batch(jobs, j=j, label="pborig").items():
            r["_out_dir"] = str(WORK / "polybench_matched" / k)
            r["input_sha"] = sha(raised[k]["paths"]["debuf"])
            _PB_MATCH[k] = r
    for k in kernels:
        if k not in _PB_MATCH:
            _PB_MATCH[k] = {"status": "raise_failed", "linalg_specs": [], "region_specs": [],
                            "raise": raised[k]}
    return {k: _PB_MATCH[k] for k in kernels}


def miss_reason(res: dict, raise_res: dict | None = None) -> str:
    """Short, data-derived reason why an input produced no launch."""
    if raise_res is not None and not raise_res.get("ok"):
        return f"raise failed at {raise_res.get('failed_step')}: {(raise_res.get('stderr') or '').strip()[-160:]}"
    if res.get("status") != "ok":
        return f"matcher {res.get('status')}: {res.get('error', '')[:200]}"
    parts = []
    ng = res.get("n_generics_in_input", 0)
    if ng == 0:
        parts.append("no linalg.generic in the raised IR (nothing raised to Linalg)")
    rg = res.get("residual_generics", [])
    if rg:
        st = Counter((g.get("furthest") or {}).get("stage", "none") for g in rg)
        parts.append("Linalg level: residual generics furthest DAG stage " +
                     ", ".join(f"{k}x{v}" for k, v in sorted(st.items())))
        order = ["none"] + ["shape", "body", "access", "extents", "body-locals", "dataflow", "matched"]
        best = sorted(rg, key=lambda g: -order.index((g.get("furthest") or {}).get("stage", "none")
                                                     if (g.get("furthest") or {}).get("stage", "none") in order else 0))
        for g in best[:2]:
            fu = g.get("furthest") or {}
            if fu:
                parts.append(f"  {g.get('generic')} {g.get('iters')}: reached '{fu.get('stage')}' "
                             f"for {', '.join(fu.get('specs', [])[:3])}; body matches {g.get('n_body_matches')}")
    rej = res.get("rejected", [])
    if rej:
        parts.append("Linalg rejections: " + "; ".join(f"{r['spec']}: {r['why']}" for r in rej[:2]))
    rd = res.get("region_diag") or {}
    if rd.get("rejections"):
        parts.append("region: " + " | ".join(r[:150] for r in rd["rejections"][:2]))
    elif rd.get("error"):
        parts.append("region diag error: " + rd["error"][:150])
    elif res.get("region_status"):
        parts.append("region: " + "; ".join(res["region_status"]))
    if res.get("region_skipped"):
        parts.append("region skipped: " + "; ".join(
            f"{s['region_match_skipped']} ({s['reason']})" for s in res["region_skipped"]))
    return " || ".join(parts) or "no candidate"


STAGE_CULPRIT = {
    None: "matcher (residual control passes)",
    "debufferized": "debufferize (memref-Linalg stage correct, --linalg-debufferize output wrong; "
                    "the matcher faithfully matched unsound tensor IR)",
    "memref_linalg": "raise-affine-to-linalg (cgeist affine stage correct, Linalg stage wrong)",
    "cgeist_affine": "cgeist / lowering harness (even the affine stage disagrees with clang)",
}


def classify_numeric(num: dict) -> tuple[str, str]:
    """-> (verdict, culprit)."""
    v = N.overall(num)
    if v != "FAIL":
        return v, ""
    culprits = set()
    for k, r in num.items():
        if r.get("verdict") != "FAIL":
            continue
        if k == "linalg":
            rc = (r.get("residual_control") or {}).get("verdict")
            if rc == "PASS":
                culprits.add(STAGE_CULPRIT[None])
            elif rc == "FAIL":
                culprits.add(STAGE_CULPRIT.get(r.get("first_bad_stage"), "undetermined"))
            else:
                culprits.add(f"undetermined (residual control {rc})")
        else:
            culprits.add("matcher (region launch vs spec C)")
    return "FAIL", "; ".join(sorted(culprits))


def fail_verdict(culprit: str) -> str:
    return "FALSE_POSITIVE" if culprit.startswith("matcher") else "E2E_WRONG_MATCH_RAISE_BUG"


def raise_cases(files: list[Path], wd: Path, j: int) -> dict[str, dict]:
    def one(f):
        fn = C.c_function_name(f)
        return f.stem, C.raise_c(f, fn, wd / f.stem / "k", [])
    return dict(C.pmap(one, files, j))


def match_cases(raised: dict, wd: Path, j: int, label: str, mode: str = "egglog") -> dict:
    jobs = [{"id": k, "input": r["paths"]["debuf"], "out_dir": str(wd / k / ("m" if mode == "egglog" else "m_" + mode)),
             "region": True, "region_diag": True, "linalg_only_inject": True}
            for k, r in raised.items() if r["ok"]]
    out = C.match_batch(jobs, j=j, mode=mode, label=label)
    for jb in jobs:
        out[jb["id"]]["_out_dir"] = jb["out_dir"]
    return out


def numeric_for(res: dict, src: Path, fn: str, meta: dict, d: Path, raise_res: dict,
                alias=None, configs=None, always_residual=False) -> dict:
    ext = N.parse_extents(meta)
    n_ints = len(re.findall(r"\bint\s+\w+\s*[,)]", src.read_text().split("{", 1)[0]))
    cfg = configs or N.size_configs(n_ints, square=meta.get("square") == "true")
    init = meta.get("init", "uniform")
    return N.check_result(res, src, fn, [], d, extents=ext, alias=alias, configs=cfg,
                          residual=Path(raise_res["paths"]["debuf"]), init_region="spd",
                          init_linalg=init, always_residual=always_residual)


def reproducer(src: Path, raise_res: dict, res: dict) -> dict:
    return {"c_source": str(src), "raised_debuf": raise_res["paths"]["debuf"],
            "matched": str(Path(res["_out_dir"]) / "matched.mlir"),
            "report": str(Path(res["_out_dir"]) / "report.json"),
            "commands": raise_res.get("commands", []) + [
                f"{C.PY} scripts/correctness/lib_v2/match_v2.py {raise_res['paths']['debuf']} "
                f"-o /tmp/m.mlir --report /tmp/r.json"]}


# ---------------------------------------------------------------------------
# Suite 1: near-miss negatives
# ---------------------------------------------------------------------------

def suite_negatives(a) -> dict:
    t0 = time.time()
    wd = suite_dir("negatives")
    files = sorted((C.CASES / "negatives").glob("*.c"))
    log(f"negatives: {len(files)} kernels")
    raised = raise_cases(files, wd, a.j)
    matched = match_cases(raised, wd, a.j, "neg")
    cases = []

    def one(f):
        meta = C.parse_meta(f)
        fn = C.c_function_name(f)
        rr = raised[f.stem]
        row = {"case": f.stem, "routine": meta.get("routine"), "kind": meta.get("kind"),
               "expect": meta.get("expect"), "source": str(f.relative_to(C.REPO))}
        if not rr["ok"]:
            row.update(verdict="RAISE_FAIL", launches=[], reason=miss_reason({}, rr))
            return row
        res = matched[f.stem]
        launches = C.launch_specs(res)
        row.update(launches=launches, linalg=res.get("linalg_specs"), region=res.get("region_specs"),
                   match_status=res.get("status"), match_seconds=res.get("seconds"),
                   region_evidence=[l.get("evidence", "")[:400] for l in res.get("launches", [])
                                    if l.get("region_match", "").startswith("netlib_")],
                   linalg_bindings=[{"spec": l["spec"], "binding": l["binding"]}
                                    for l in res.get("launches", []) if "spec" in l])
        if res.get("status") != "ok":
            row.update(verdict="MATCHER_" + str(res.get("status")).upper(), error=res.get("error"),
                       reproducer=reproducer(f, rr, res))
            return row
        if not launches:
            row["reason"] = miss_reason(res)
            row["verdict"] = "MISS" if meta.get("expect") == "match" else "PASS_NO_MATCH"
            return row
        num = numeric_for(res, f, fn, meta, wd / f.stem / "num", rr)
        row["numeric"] = num
        v, culprit = classify_numeric(num)
        row["numeric_verdict"] = v
        if v == "PASS":
            row["verdict"] = ("PASS_MATCH" if meta.get("expect") == "match" else
                              "PASS_EQUIVALENT_MATCH")
        elif v == "FAIL":
            row["verdict"] = fail_verdict(culprit)
            row["culprit"] = culprit
            row["reproducer"] = reproducer(f, rr, res)
        else:
            row["verdict"] = "MATCH_NUMERIC_INCONCLUSIVE"
        return row

    cases = C.pmap(one, files, max(1, a.j // 2))
    cnt = Counter(c["verdict"] for c in cases)
    neg = [c for c in cases if c["kind"] != "positive_control"]
    pos = [c for c in cases if c["kind"] == "positive_control"]
    by_kind = defaultdict(Counter)
    for c in neg:
        by_kind[c["kind"]][c["verdict"]] += 1
    summary = {"kernels": len(cases), "near_misses": len(neg), "positive_controls": len(pos),
               "verdicts": dict(cnt),
               "near_miss_pass": sum(c["verdict"] in ("PASS_NO_MATCH", "PASS_EQUIVALENT_MATCH") for c in neg),
               "false_positives": [c["case"] for c in cases if c["verdict"] == "FALSE_POSITIVE"],
               "e2e_wrong_matches_raise_bug": {c["case"]: c.get("culprit") for c in cases
                                               if c["verdict"] == "E2E_WRONG_MATCH_RAISE_BUG"},
               "matcher_crashes": [c["case"] for c in cases if c["verdict"].startswith("MATCHER_")],
               "equivalent_matches": {c["case"]: c["launches"] for c in neg
                                      if c["verdict"] == "PASS_EQUIVALENT_MATCH"},
               "positive_controls_matched": {c["case"]: c.get("launches") for c in pos},
               "by_kind": {k: dict(v) for k, v in by_kind.items()},
               "rtol": N.RTOL, "seeds": N.SEEDS}
    return finish("negatives", t0, summary, cases)


# ---------------------------------------------------------------------------
# Suite 2: aliasing
# ---------------------------------------------------------------------------

# Call-site (runtime) aliasing of kernels whose own body is alias-free:
# (case id, C file, {param: (base param, element offset)}, configs)
RUNTIME_ALIAS = [
    ("rt_gemm_A_is_C", "negatives/gemm__pos.c", {"A": ("C", 0)}, "square"),
    ("rt_gemm_B_is_C", "negatives/gemm__pos.c", {"B": ("C", 0)}, "square"),
    ("rt_gemm_B_is_A", "negatives/gemm__pos.c", {"B": ("A", 0)}, "square"),
    ("rt_gemv_x_is_y", "negatives/gemv_n__pos.c", {"x": ("y", 0)}, "square"),
    ("rt_ger_y_is_x", "negatives/ger__pos.c", {"y": ("x", 0)}, "square"),
    ("rt_axpy_y_is_x_plus1", "negatives/axpy__pos.c", {"y": ("x", 1)}, None),
    ("rt_axpy_x_is_y_plus1", "negatives/axpy__pos.c", {"x": ("y", 1)}, None),
    ("rt_trsv_b_is_x", "aliasing/trsv_ref.c", {"b": ("x", 0)}, None),
    ("rt_syrk_A_is_C", "negatives/syrk__pos.c", {"A": ("C", 0)}, "square"),
    ("rt_symm_B_is_C", "negatives/symm__pos.c", {"B": ("C", 0)}, "square"),
]


def flagged_alias(res: dict, runtime: bool = False) -> bool:
    """Explicit aliasing: any alias / in-place note in the launch evidence.
    Call-site aliasing: only an explicit no-alias assumption counts ("in-place"
    merely says an in/out spec argument is updated in place)."""
    txt = json.dumps(res.get("launches", []))
    return bool(re.search(r"alias" if runtime else r"alias|in-place", txt))


def suite_aliasing(a) -> dict:
    t0 = time.time()
    wd = suite_dir("aliasing")
    explicit = sorted((C.CASES / "aliasing").glob("*.c"))
    explicit = [f for f in explicit if C.parse_meta(f).get("kind") == "explicit"]
    rt_files = sorted({C.CASES / f for _, f, _, _ in RUNTIME_ALIAS}, key=str)
    files = explicit + [f for f in rt_files if f not in explicit]
    log(f"aliasing: {len(explicit)} explicit, {len(RUNTIME_ALIAS)} call-site cases")
    raised = raise_cases(files, wd, a.j)
    matched = match_cases(raised, wd, a.j, "alias")

    def explicit_one(f):
        meta = C.parse_meta(f)
        fn = C.c_function_name(f)
        rr = raised[f.stem]
        row = {"case": f.stem, "type": "explicit", "expect": meta.get("expect"),
               "source": str(f.relative_to(C.REPO))}
        if not rr["ok"]:
            row.update(verdict="RAISE_FAIL", reason=miss_reason({}, rr))
            return row
        res = matched[f.stem]
        launches = C.launch_specs(res)
        row.update(launches=launches, flagged=flagged_alias(res),
                   region_evidence=[l.get("evidence", "")[:500] for l in res.get("launches", [])
                                    if l.get("region_match", "").startswith("netlib_")],
                   linalg_bindings=[{"spec": l["spec"], "binding": l["binding"]}
                                    for l in res.get("launches", []) if "spec" in l])
        if res.get("status") != "ok":
            row.update(verdict="MATCHER_" + str(res.get("status")).upper(), error=res.get("error"))
            return row
        if not launches:
            row.update(verdict="REJECTED" if meta.get("expect") != "match_correct" else "MISS",
                       reason=miss_reason(res))
            return row
        num = numeric_for(res, f, fn, meta, wd / f.stem / "num", rr, always_residual=True)
        row["numeric"] = num
        v, culprit = classify_numeric(num)
        row["numeric_verdict"] = v
        if v == "PASS":
            row["verdict"] = "MATCHED_CORRECT" + ("_FLAGGED" if row["flagged"] else "")
        elif v == "FAIL":
            row.update(verdict=fail_verdict(culprit), culprit=culprit,
                       reproducer=reproducer(f, rr, res))
        else:
            row["verdict"] = "MATCH_NUMERIC_INCONCLUSIVE"
        return row

    def runtime_one(item):
        cid, rel, alias, cfg = item
        f = C.CASES / rel
        meta = C.parse_meta(f)
        fn = C.c_function_name(f)
        rr = raised[f.stem]
        row = {"case": cid, "type": "call-site", "base_kernel": f.stem, "alias": alias,
               "expect": "reject_or_flagged_and_correct"}
        if not rr["ok"]:
            row.update(verdict="RAISE_FAIL")
            return row
        res = matched[f.stem]
        launches = C.launch_specs(res)
        row.update(launches=launches, flagged=flagged_alias(res, runtime=True),
                   flag_text=[m for l in res.get("launches", [])
                              for m in re.findall(r"[^;]*alias[^;]*", l.get("evidence", ""))][:3])
        if not launches:
            row["verdict"] = "NOT_MATCHED"
            return row
        n_ints = len(re.findall(r"\bint\s+\w+\s*[,)]", f.read_text().split("{", 1)[0]))
        configs = N.size_configs(n_ints, square=(cfg == "square"))
        num = numeric_for(res, f, fn, meta, wd / cid / "num", rr, alias=alias, configs=configs,
                          always_residual=True)
        row["numeric"] = num
        v, culprit = classify_numeric(num)
        row["numeric_verdict"] = v
        res_ctrl = (num.get("linalg", {}).get("residual_control") or {}).get("verdict")
        row["residual_control_under_alias"] = res_ctrl
        if v == "PASS":
            row["verdict"] = "CORRECT_UNDER_ALIAS" + ("_FLAGGED" if row["flagged"] else "")
        elif v == "FAIL":
            row["verdict"] = ("WRONG_UNDER_ALIAS_FLAGGED" if row["flagged"] else
                              "WRONG_UNDER_ALIAS_UNFLAGGED")
            row["culprit"] = culprit
        else:
            row["verdict"] = "INCONCLUSIVE"
        return row

    cases = C.pmap(explicit_one, explicit, max(1, a.j // 2))
    cases += C.pmap(runtime_one, RUNTIME_ALIAS, max(1, a.j // 2))
    summary = {"explicit": dict(Counter(c["verdict"] for c in cases if c["type"] == "explicit")),
               "call_site": dict(Counter(c["verdict"] for c in cases if c["type"] == "call-site")),
               "false_positives": [c["case"] for c in cases if c["verdict"] == "FALSE_POSITIVE"],
               "e2e_wrong_matches_raise_bug": {c["case"]: c.get("culprit") for c in cases
                                               if c["verdict"] == "E2E_WRONG_MATCH_RAISE_BUG"},
               "unflagged_wrong_under_alias": [c["case"] for c in cases
                                               if c["verdict"] == "WRONG_UNDER_ALIAS_UNFLAGGED"],
               "note": ("call-site cases call a correctly matched function with aliased "
                        "pointers; C allows this without restrict.  'residual_control_under_alias' "
                        "is the same raised IR lowered without matching: if it is also wrong the "
                        "noalias assumption is pipeline-wide, not introduced by the matcher.")}
    return finish("aliasing", t0, summary, cases)


# ---------------------------------------------------------------------------
# Suite 3: stronger numerics
# ---------------------------------------------------------------------------

def _swap_first_same_type_operands(text: str) -> str | None:
    """Sensitivity control: swap two distinct same-typed tensor operands of the
    first kernel.launch (e.g. gemm's A and B)."""
    m = re.search(r"(kernel\.launch\s+@[\w$.\-]+\()([^)]*)(\)\s*:\s*\()([^)]*)(\))", text)
    if not m:
        return None
    ops = [x.strip() for x in m.group(2).split(",")]
    tys = [x.strip() for x in m.group(4).split(",")]
    for i in range(len(ops)):
        for k in range(i + 1, len(ops)):
            if tys[i] == tys[k] and tys[i].startswith("tensor<") and ops[i] != ops[k]:
                ops[i], ops[k] = ops[k], ops[i]
                return text[:m.start(2)] + ", ".join(ops) + text[m.end(2):]
    return None


def run_e2e_large(kernels: list[str], d: Path, j: int) -> dict:
    """LARGE dataset through the unmodified e2e_v2.py (PolyBench init data)."""
    cmd = [C.PY, str(C.LIBV2 / "e2e_v2.py"), "--dataset", "LARGE", "--out", str(d), "-j", str(j),
           *kernels]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=C.env(), timeout=7200)
    summ = C.read_json(d / "LARGE" / "summary.json", {})
    out = {"command": " ".join(cmd), "rc": p.returncode, "seconds": round(time.time() - t0, 1),
           "stdout_tail": p.stdout[-3000:], "kernels": {}}
    for k in kernels:
        r = summ.get(k, {})
        v2 = r.get("v2", {})
        tight = v2.get("verdict") in ("PASS_EXACT", "PASS_1e-12")
        out["kernels"][k] = {"status": r.get("status"), "launches": r.get("v2_launches"),
                             "v2_verdict": v2.get("verdict"), "v2_max_rel": v2.get("max_rel"),
                             "v2_max_abs": v2.get("max_abs"),
                             "residual_verdict": (r.get("residual") or {}).get("verdict"),
                             "tight_pass": tight}
    return out


def suite_numerics(a) -> dict:
    t0 = time.time()
    wd = suite_dir("numerics")
    pbm = polybench_matched(C.MATCHED_KERNELS, a.j)
    pbr = polybench_raised(C.MATCHED_KERNELS, a.j)
    srcs = C.polybench_sources()

    def one(k):
        res = pbm[k]
        rr = pbr[k]
        src = srcs[k]
        fn = "kernel_" + k.replace("-", "_")
        cfl = C.polybench_cflags(src)
        row = {"kernel": k, "expected_path": "linalg" if k in C.LINALG_KERNELS else "region",
               "launches": C.launch_specs(res), "linalg": res.get("linalg_specs"),
               "region": res.get("region_specs")}
        if not rr.get("ok"):
            row.update(verdict="RAISE_FAIL", reason=miss_reason({}, rr))
            return row
        if not row["launches"]:
            row.update(verdict="NO_MATCH", reason=miss_reason(res))
            return row
        text = N.preprocess(src, cfl)
        params = N.c_params(text, fn)
        n_ints = sum(1 for kk, _, _ in params if kk == "int")
        num = N.check_result(res, src, fn, cfl, wd / k, polybench=True,
                             configs=N.size_configs(n_ints), residual=Path(rr["paths"]["debuf"]))
        row["numeric"] = num
        row["verdict"], row["culprit"] = classify_numeric(num)
        row["max_err"] = max([v.get("max_err") or 0.0 for v in num.values()] or [0.0])
        row["bit_exact"] = all(v.get("bit_exact") for v in num.values())
        return row

    cases = C.pmap(one, C.MATCHED_KERNELS, max(1, a.j // 2))
    # sensitivity controls: a deliberately wrong binding must FAIL
    controls = []
    g = pbm.get("gemm", {})
    inj = Path(g.get("_out_dir", "/nonexistent")) / "linalg_only_injected.mlir"
    if inj.exists():
        bad = _swap_first_same_type_operands(inj.read_text())
        if bad:
            bp = wd / "control_gemm_swapAB.mlir"
            bp.write_text(bad)
            src = srcs["gemm"]
            r = N.check_lowered(src, "kernel_gemm", C.polybench_cflags(src), bp, wd / "control_gemm",
                                polybench=True, configs={"nonsquare": [7, 5, 11]}, seeds=[1])
            controls.append({"control": "gemm launch with A/B operands swapped",
                             "verdict": r["verdict"], "expected": "FAIL", "max_err": r.get("max_err"),
                             "ok": r["verdict"] == "FAIL"})
    t = pbm.get("trisolv", {})
    if t.get("region_blocks"):
        blk = dict(t["region_blocks"][0])
        blk["text"] = re.sub(r"@netlib_dtrsv_utn\b", "@netlib_dtrsv_ltn", blk["text"])
        src = srcs["trisolv"]
        r = N.check_region(src, "kernel_trisolv", C.polybench_cflags(src),
                           Path(t["_out_dir"]) / "matched.mlir", blk, wd / "control_trisolv",
                           polybench=True, configs={"n7": [7]}, seeds=[1])
        controls.append({"control": "trisolv region launch with dtrsv_ltn instead of dtrsv_utn",
                         "verdict": r["verdict"], "expected": "FAIL", "max_err": r.get("max_err"),
                         "ok": r["verdict"] == "FAIL"})
    # hybrid path (mixed Linalg + region launches): swap two same-typed memref
    # operands of the last region launch (e.g. gesummv's daxpy x/y)
    for k in ("gesummv", "gemver"):
        r = pbm.get(k, {})
        mt = Path(r.get("_out_dir", "/nonexistent")) / "matched.mlir"
        if not mt.exists() or not r.get("region_blocks"):
            continue
        lines = mt.read_text().splitlines()
        idx = [i for i, l in enumerate(lines) if re.search(r"kernel\.launch.*->\s*\(\)", l)]
        done = False
        for i in reversed(idx):
            m = re.search(r"\(([^)]*)\)\s*:\s*\(([^)]*)\)", lines[i])
            ops = [x.strip() for x in m.group(1).split(",")]
            tys = [x.strip() for x in m.group(2).split(",")]
            pairs = [(x, y) for x in range(len(ops)) for y in range(x + 1, len(ops))
                     if tys[x] == tys[y] and tys[x].startswith("memref<") and ops[x] != ops[y]]
            if pairs:
                x, y = pairs[0]
                ops[x], ops[y] = ops[y], ops[x]
                lines[i] = lines[i][:m.start(1)] + ", ".join(ops) + lines[i][m.end(1):]
                done = True
                break
        if not done:
            continue
        bp = wd / f"control_{k}_region_swap.mlir"
        bp.write_text("\n".join(lines) + "\n")
        src = srcs[k]
        rr = N.check_full(src, "kernel_" + k, C.polybench_cflags(src), bp, wd / f"control_{k}",
                          polybench=True, configs={"n7": [7]}, seeds=[1])
        controls.append({"control": f"{k}: hybrid path, region launch with two memref operands swapped",
                         "verdict": rr["verdict"], "expected": "FAIL", "max_err": rr.get("max_err"),
                         "ok": rr["verdict"] == "FAIL"})
        break
    large = None
    if not a.no_large:
        log("numerics: LARGE dataset via e2e_v2.py (Linalg-level kernels)")
        large = run_e2e_large(C.LINALG_KERNELS, wd / "e2e_large", max(1, a.j // 2))
    summary = {"verdicts": {c["kernel"]: c["verdict"] for c in cases},
               "pass": sum(c["verdict"] == "PASS" for c in cases), "kernels": len(cases),
               "max_err": {c["kernel"]: c.get("max_err") for c in cases},
               "bit_exact": {c["kernel"]: c.get("bit_exact") for c in cases},
               "runs_per_kernel": len(N.SEEDS) * 4, "seeds": N.SEEDS, "rtol": N.RTOL,
               "size_configs": "n1 (all sizes 1), n2, nonsquare (7,5,11,3,..), medium (23,19,29,17,..)",
               "tolerance": ("per array incl. 64 guard elements: max|test-ref|/max|ref| <= 1e-12, "
                             "NaN/Inf positions equal"),
               "sensitivity_controls": controls,
               "large": None if large is None else {
                   k: (v["v2_verdict"], v["v2_max_rel"]) for k, v in large["kernels"].items()},
               "large_tight_pass": None if large is None else sum(
                   v["tight_pass"] for v in large["kernels"].values())}
    return finish("numerics", t0, summary, cases, {"large_e2e": large})


# ---------------------------------------------------------------------------
# Suite 4: IR variants
# ---------------------------------------------------------------------------

def gen_ir_variants(src_mlir: Path, kernel: str, outdir: Path) -> list[dict]:
    import generate_eqsat_ir_variants as G  # read-only reuse of the paper's transformations
    text = src_mlir.read_text()
    rows = []
    for v in G.VARIANTS:
        m = G.mutate(text, v)
        row = {"kernel": kernel, "variant": v, "sites": m.sites}
        if not m.sites:
            row["status"] = "not_applicable"
            rows.append(row)
            continue
        p = outdir / f"{kernel}__{v}.mlir"
        p.write_text(m.text)
        r = subprocess.run(["polygeist-opt", str(p), "-o", "/dev/null"], capture_output=True,
                           text=True, env=C.env())
        row.update(path=str(p), status="generated" if r.returncode == 0 else "verifier_failed",
                   verifier_stderr=r.stderr[-300:])
        rows.append(row)
    return rows


def retention_rows(orig: dict[str, list[str]], var_results: dict, gen: list[dict]) -> list[dict]:
    rows = []
    for g in gen:
        if g["status"] != "generated":
            continue
        k, v = g["kernel"], g["variant"]
        res = var_results.get(f"{k}__{v}", {})
        o = orig.get(k, [])
        new = C.launch_specs(res)
        rows.append({"kernel": k, "variant": v, "status": res.get("status"),
                     "original": o, "variant_launches": new,
                     "opportunities": len(o), "retained": C.multiset_retained(o, new),
                     "extra": sorted((Counter(new) - Counter(o)).elements()),
                     "seconds": res.get("seconds"),
                     "reason": "" if C.multiset_retained(o, new) == len(o) else miss_reason(res)})
    return rows


def suite_irvariants(a) -> dict:
    t0 = time.time()
    wd = suite_dir("irvariants")
    kernels = list(C.polybench_sources())
    pbm = polybench_matched(kernels, a.j)
    pbr = polybench_raised(kernels, a.j)
    inp = wd / "inputs"
    inp.mkdir(parents=True, exist_ok=True)
    gen = []
    for k in kernels:
        if pbr[k]["ok"]:
            gen += gen_ir_variants(Path(pbr[k]["paths"]["debuf"]), k, inp)
    jobs = [{"id": f"{g['kernel']}__{g['variant']}", "input": g["path"],
             "out_dir": str(wd / "match" / f"{g['kernel']}__{g['variant']}"), "region": True,
             "region_diag": True} for g in gen if g["status"] == "generated"]
    # consumability probe: the paper campaign's own 245 inputs (campaign recipe)
    camp_dir = C.REPO / "issues/equality_saturation_eval/polybench_variants_20260909"
    camp_rows = []
    import csv
    with (camp_dir / "manifest.csv").open() as fh:
        camp_rows = list(csv.DictReader(fh))
    camp_orig = [r for r in camp_rows if r["variant"] == "original"]
    jobs += [{"id": f"campaign::{r['kernel']}", "input": r["path"],
              "out_dir": str(wd / "campaign" / r["kernel"]), "region": True} for r in camp_orig]
    log(f"irvariants: {sum(g['status'] == 'generated' for g in gen)} regenerated variants "
        f"+ {len(camp_orig)} campaign originals")
    res = C.match_batch(jobs, j=a.j, label="irv", chunk=10)
    for jb in jobs:
        res[jb["id"]]["_out_dir"] = jb["out_dir"]
        res[jb["id"]]["input_sha"] = sha(jb["input"])
    # campaign variants only where the campaign original matched something
    camp_hit = [r["kernel"] for r in camp_orig if C.launch_specs(res[f"campaign::{r['kernel']}"])]
    camp_var_jobs = [{"id": f"campaign::{r['kernel']}__{r['variant']}", "input": r["path"],
                      "out_dir": str(wd / "campaign" / f"{r['kernel']}__{r['variant']}"),
                      "region": True}
                     for r in camp_rows if r["variant"] != "original" and r["kernel"] in camp_hit]
    if camp_var_jobs:
        res.update(C.match_batch(camp_var_jobs, j=a.j, label="irvc", chunk=10))
    orig = {k: C.launch_specs(pbm[k]) for k in kernels}
    rows = retention_rows(orig, res, gen)
    primary = [r for r in rows if r["variant"] != "identity_pair"]
    by_var = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        bv = by_var[r["variant"]]
        bv[0] += 1
        bv[1] += r["opportunities"]
        bv[2] += r["retained"]
    camp = {"originals_with_v2_launch": camp_hit,
            "originals": {r["kernel"]: C.launch_specs(res[f"campaign::{r['kernel']}"]) for r in camp_orig},
            "variants_run": len(camp_var_jobs)}
    if camp_var_jobs:
        corig = {k: camp["originals"][k] for k in camp_hit}
        cgen = [{"kernel": r["kernel"], "variant": r["variant"], "status": "generated"}
                for r in camp_rows if r["variant"] != "original" and r["kernel"] in camp_hit]
        cres = {f"{r['kernel']}__{r['variant']}": res[f"campaign::{r['kernel']}__{r['variant']}"]
                for r in camp_rows if r["variant"] != "original" and r["kernel"] in camp_hit}
        crows = retention_rows(corig, cres, cgen)
        camp["retention"] = [sum(r["retained"] for r in crows if r["variant"] != "identity_pair"),
                             sum(r["opportunities"] for r in crows if r["variant"] != "identity_pair")]
    tot_o = sum(r["opportunities"] for r in primary)
    tot_r = sum(r["retained"] for r in primary)
    summary = {
        "source_ir": "fresh raise of all 30 PolyBench kernels with the lib_v2 recipe",
        "fresh_ir_identical_to_issues_lib_v2_polybench_raised": {
            k: pbr[k].get("same_as_issues_lib_v2_polybench_raised") for k in kernels},
        "variants_generated": sum(g["status"] == "generated" for g in gen),
        "variants_not_applicable": sum(g["status"] == "not_applicable" for g in gen),
        "variants_verifier_failed": sum(g["status"] == "verifier_failed" for g in gen),
        "primary_excludes": "identity_pair (as in the production campaign)",
        "egglog_retention": [tot_r, tot_o, round(100.0 * tot_r / tot_o, 1) if tot_o else None],
        "by_variant": {v: {"inputs": x[0], "opportunities": x[1], "retained": x[2]}
                       for v, x in sorted(by_var.items())},
        "kernels_with_original_launches": {k: v for k, v in orig.items() if v},
        "inputs_with_extra_launches": [f"{r['kernel']}__{r['variant']}: {r['extra']}" for r in rows if r["extra"]],
        "matcher_errors": [f"{r['kernel']}__{r['variant']}: {r['status']}" for r in rows if r["status"] != "ok"],
        "production_reference": {"egglog": "207/221 (93.7%)", "syntactic": "98/221 (44.3%)",
                                 "note": "production = old Term-formula matcher on campaign-recipe IR"},
        "campaign_recipe_inputs": camp,
        "syntactic_arm": "see ablation/results.json",
    }
    return finish("irvariants", t0, summary, rows, {"generation": gen})


# ---------------------------------------------------------------------------
# Suite 5: source-level variants
# ---------------------------------------------------------------------------

def suite_srcvariants(a) -> dict:
    t0 = time.time()
    wd = suite_dir("srcvariants")
    pbm = polybench_matched(C.MATCHED_KERNELS, a.j)
    srcs = C.polybench_sources()
    files = sorted((C.CASES / "srcvariants").glob("*/*.c"))
    log(f"srcvariants: {len(files)} variants")
    raised = raise_cases(files, wd, a.j)
    matched = match_cases(raised, wd, a.j, "src")

    def one(f):
        kernel = f.parent.name
        meta = C.parse_meta(f)
        fn = C.c_function_name(f)
        rr = raised[f.stem]
        orig_launches = C.launch_specs(pbm[kernel])
        row = {"case": f.stem, "kernel": kernel, "transform": meta.get("transform"),
               "source": str(f.relative_to(C.REPO)), "original_launches": orig_launches}
        # validity of the hand-written variant itself (variant C vs PolyBench C)
        osrc = srcs[kernel]
        ofn = "kernel_" + kernel.replace("-", "_")
        n_ints = len(re.findall(r"\bint\s+\w+\s*[,)]", f.read_text().split("{", 1)[0]))
        init = "spd" if kernel in C.REGION_KERNELS else "uniform"
        row["variant_valid"] = N.check_c_equivalence(
            f, fn, osrc, ofn, C.polybench_cflags(osrc), wd / f.stem / "equiv",
            extents=N.parse_extents(meta), init=init, configs=N.size_configs(n_ints))
        if not rr["ok"]:
            row.update(capture="none", launches=[], reason=miss_reason({}, rr), verdict="RAISE_FAIL")
            return row
        res = matched[f.stem]
        launches = C.launch_specs(res)
        ret = C.multiset_retained(orig_launches, launches)
        row.update(launches=launches, retained=ret, match_status=res.get("status"),
                   capture=("full" if orig_launches and ret == len(orig_launches) else
                            "partial" if ret or launches else "none"))
        if row["capture"] != "full":
            row["reason"] = miss_reason(res)
        if launches:
            num = N.check_result(res, f, fn, [], wd / f.stem / "num", extents=N.parse_extents(meta),
                                 configs=N.size_configs(n_ints),
                                 residual=Path(rr["paths"]["debuf"]),
                                 init_region="spd", init_linalg=init)
            row["numeric"] = num
            row["numeric_verdict"], row["culprit"] = classify_numeric(num)
            if row["numeric_verdict"] == "FAIL":
                row["reproducer"] = reproducer(f, rr, res)
        row["verdict"] = row["capture"].upper() + (
            "" if not launches else f"/{row['numeric_verdict']}")
        return row

    cases = C.pmap(one, files, max(1, a.j // 2))
    by_t = defaultdict(Counter)
    for c in cases:
        by_t[c["transform"]][c["capture"]] += 1
    by_k = defaultdict(Counter)
    for c in cases:
        by_k[c["kernel"]][c["capture"]] += 1
    summary = {"variants": len(cases),
               "capture": dict(Counter(c["capture"] for c in cases)),
               "capture_by_transform": {t: dict(v) for t, v in sorted(by_t.items())},
               "capture_rate_by_transform": {t: f"{v['full']}/{sum(v.values())}" for t, v in sorted(by_t.items())},
               "capture_by_kernel": {k: dict(v) for k, v in sorted(by_k.items())},
               "numeric": dict(Counter(c.get("numeric_verdict", "not_matched") for c in cases)),
               "numeric_failures": [c["case"] for c in cases if c.get("numeric_verdict") == "FAIL"],
               "invalid_variants": [c["case"] for c in cases
                                    if (c.get("variant_valid") or {}).get("verdict") != "PASS"],
               "capture_definition": ("full = every library launch of the fresh PolyBench original "
                                      "(multiset of spec names) is also emitted for the variant")}
    return finish("srcvariants", t0, summary, cases)


# ---------------------------------------------------------------------------
# Suite 6: egglog vs syntactic
# ---------------------------------------------------------------------------

REGION_SWITCH_NOTE = (
    "kernel_match.configure_matcher(mode='syntactic') switches kernel_match.equivalent() and "
    "body_matches_template() (used by match_v2 via match_composition) to structural "
    "unification.  match_v2.import_kernel_match() hard-codes configure_matcher('egglog'); the "
    "harness wraps it (worker.py) instead of editing match_v2.  region_match does not consult "
    "kernel_match._MATCHER_MODE: region_match/egg.py:check_equal() always builds an EGraph and "
    "runs rules() (index normalisation, layout permutation, triangle<->mask, linearity, field "
    "identities).  Needed switch: a `mode` argument (or a read of km._MATCHER_MODE) in "
    "check_equal() that, in syntactic mode, applies the binding unions as a substitution and "
    "compares the two Terms structurally without running any rewrite; compare_systems() would "
    "pass it through.  Without it the region arm below is egglog in both columns.")


def suite_ablation(a) -> dict:
    t0 = time.time()
    wd = suite_dir("ablation")
    kernels = list(C.polybench_sources())
    pbm = polybench_matched(kernels, a.j)
    pbr = polybench_raised(kernels, a.j)
    # inputs: originals, suite-4 regenerated variants, suite-5 raised variants
    inputs = []
    for k in kernels:
        if pbr[k]["ok"]:
            inputs.append(("original", k, k, Path(pbr[k]["paths"]["debuf"]),
                           Path(pbm[k]["_out_dir"]) / "result.json" if "_out_dir" in pbm[k] else None))
    irv = C.read_json(C.OUT_ROOT / "irvariants" / "results.json", {})
    for g in irv.get("generation", []):
        if g.get("status") == "generated" and Path(g["path"]).exists():
            cid = f"{g['kernel']}__{g['variant']}"
            inputs.append(("irvariant", g["kernel"], cid, Path(g["path"]),
                           WORK / "irvariants" / "match" / cid / "result.json"))
    srcv = C.read_json(C.OUT_ROOT / "srcvariants" / "results.json", {})
    for c in srcv.get("cases", []):
        p = WORK / "srcvariants" / c["case"] / "k_debuf.mlir"
        if p.exists():
            inputs.append(("srcvariant", c["kernel"], c["case"], p,
                           WORK / "srcvariants" / c["case"] / "m" / "result.json"))
    log(f"ablation: {len(inputs)} inputs ({Counter(i[0] for i in inputs)})")
    # egglog arm: reuse earlier results when the input is byte-identical
    egg, redo = {}, []
    for kind, k, cid, p, rp in inputs:
        r = C.read_json(rp, None) if rp else None
        if r and r.get("status") == "ok" and r.get("input") == str(p):
            egg[cid] = r
        else:
            redo.append({"id": cid, "input": str(p), "out_dir": str(wd / "egglog" / cid),
                         "region": True})
    if redo:
        log(f"ablation: egglog arm re-run for {len(redo)} inputs")
        egg.update(C.match_batch(redo, j=a.j, mode="egglog", label="abl_e", chunk=10))
    jobs = [{"id": cid, "input": str(p), "out_dir": str(wd / "syntactic" / cid), "region": True}
            for kind, k, cid, p, rp in inputs]
    syn = C.match_batch(jobs, j=a.j, mode="syntactic", label="abl_s", chunk=10)
    rows = []
    for kind, k, cid, p, rp in inputs:
        e, s = egg.get(cid, {}), syn.get(cid, {})
        rows.append({"kind": kind, "kernel": k, "input": cid,
                     "egglog_linalg": e.get("linalg_specs", []), "syntactic_linalg": s.get("linalg_specs", []),
                     "egglog_region": e.get("region_specs", []), "syntactic_region": s.get("region_specs", []),
                     "egglog_status": e.get("status"), "syntactic_status": s.get("status"),
                     "egglog_s": e.get("seconds"), "syntactic_s": s.get("seconds"),
                     "egglog_proofs": (e.get("telemetry") or {}).get("proofs_attempted"),
                     "syntactic_unifications": (s.get("telemetry") or {}).get("syntactic_unifications")})
    # retention on the common baseline: identities both arms find in the original
    origs = {r["kernel"]: r for r in rows if r["kind"] == "original"}

    def arm(r, which):
        return r[f"{which}_linalg"] + r[f"{which}_region"]

    agg = {}
    for kind in ("irvariant", "srcvariant"):
        sub = [r for r in rows if r["kind"] == kind]
        if kind == "irvariant":
            sub = [r for r in sub if not r["input"].endswith("__identity_pair")]
        tot = {"egglog": [0, 0], "syntactic": [0, 0], "common": 0, "egg_common": 0, "syn_common": 0,
               "egg_linalg_only": [0, 0], "syn_linalg_only": [0, 0]}
        for r in sub:
            o = origs.get(r["kernel"])
            if not o:
                continue
            oe, os_ = arm(o, "egglog"), arm(o, "syntactic")
            common = list((Counter(oe) & Counter(os_)).elements())
            for which, ob in (("egglog", oe), ("syntactic", os_)):
                tot[which][0] += C.multiset_retained(ob, arm(r, which))
                tot[which][1] += len(ob)
            tot["common"] += len(common)
            tot["egg_common"] += C.multiset_retained(common, arm(r, "egglog"))
            tot["syn_common"] += C.multiset_retained(common, arm(r, "syntactic"))
            oel, osl = o["egglog_linalg"], o["syntactic_linalg"]
            tot["egg_linalg_only"][0] += C.multiset_retained(oel, r["egglog_linalg"])
            tot["egg_linalg_only"][1] += len(oel)
            tot["syn_linalg_only"][0] += C.multiset_retained(osl, r["syntactic_linalg"])
            tot["syn_linalg_only"][1] += len(osl)
        agg[kind] = tot
    diff = [r["input"] for r in rows if sorted(arm(r, "egglog")) != sorted(arm(r, "syntactic"))]
    summary = {
        "inputs": dict(Counter(r["kind"] for r in rows)),
        "originals": {k: {"egglog": arm(o, "egglog"), "syntactic": arm(o, "syntactic")}
                      for k, o in origs.items() if arm(o, "egglog") or arm(o, "syntactic")},
        "originals_launches": [sum(len(arm(o, "egglog")) for o in origs.values()),
                               sum(len(arm(o, "syntactic")) for o in origs.values())],
        "retention": {kind: {
            "egglog_vs_own_original": t["egglog"], "syntactic_vs_own_original": t["syntactic"],
            "common_baseline": {"opportunities": t["common"], "egglog": t["egg_common"],
                                "syntactic": t["syn_common"]},
            "linalg_level_only": {"egglog": t["egg_linalg_only"], "syntactic": t["syn_linalg_only"]}}
            for kind, t in agg.items()},
        "inputs_where_arms_differ": len(diff),
        "differing_inputs": diff[:80],
        "errors": [r["input"] for r in rows if "ok" not in (r["egglog_status"], r["syntactic_status"])
                   or r["egglog_status"] != r["syntactic_status"]],
        "production_reference": {"egglog": "207/221", "syntactic": "98/221"},
        "switch": REGION_SWITCH_NOTE,
    }
    return finish("ablation", t0, summary, rows)


# ---------------------------------------------------------------------------

def update_summary() -> None:
    out = {"updated": C.now(), "suites": {}}
    for s in SUITES:
        r = C.read_json(C.OUT_ROOT / s / "results.json", None)
        if r:
            out["suites"][s] = {"finished": r.get("finished"), "seconds": r.get("seconds"),
                                "summary": r.get("summary")}
    C.write_json(C.OUT_ROOT / "summary.json", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", choices=SUITES + ["all", "summary"])
    ap.add_argument("-j", type=int, default=6, help="parallel workers (default 6)")
    ap.add_argument("--no-large", action="store_true", help="numerics: skip the LARGE e2e_v2 run")
    a = ap.parse_args()
    t0 = time.time()
    fns = {"negatives": suite_negatives, "aliasing": suite_aliasing, "numerics": suite_numerics,
           "irvariants": suite_irvariants, "srcvariants": suite_srcvariants,
           "ablation": suite_ablation}
    todo = SUITES if a.suite == "all" else ([] if a.suite == "summary" else [a.suite])
    for s in todo:
        log(f"=== suite {s}")
        try:
            fns[s](a)
        except Exception as e:  # noqa: BLE001  (one suite failing must not stop the others)
            import traceback
            tb = traceback.format_exc()
            log(f"suite {s} crashed: {e}\n{tb}")
            C.write_json(C.OUT_ROOT / s / "results.json",
                         {"suite": s, "finished": C.now(), "harness_error": tb, "summary": {}})
    update_summary()
    total = round(time.time() - t0, 1)
    s = C.read_json(C.OUT_ROOT / "summary.json", {})
    s["last_invocation"] = {"command": " ".join(sys.argv), "seconds": total, "finished": C.now()}
    C.write_json(C.OUT_ROOT / "summary.json", s)
    log(f"total {total} s; summary in {C.OUT_ROOT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
