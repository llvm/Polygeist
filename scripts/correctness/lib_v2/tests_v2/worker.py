#!/usr/bin/env python3
"""Batch match worker: runs match_v2.run() in-process over a list of jobs.

Usage: worker.py --jobs JOBS.json [--mode egglog|syntactic] [--job-timeout S]

JOBS.json: [{"id", "input", "out_dir", "region": bool, "region_diag": bool,
             "linalg_only_inject": bool}]

Per job writes <out_dir>/{matched.mlir, report.json, result.json} and, when
`linalg_only_inject` is set and a Linalg-level launch exists,
<out_dir>/linalg_only_injected.mlir (== match_v2.py --no-region-match
--inject-defns), the input of the numerical lowering path.

match_v2.py is imported read-only.  The matcher mode switch is applied by
wrapping match_v2.import_kernel_match (which hard-codes
configure_matcher("egglog")) so that kernel_match.configure_matcher(mode) is
called after it; nothing in match_v2 / kernel_match is edited.  The region
matcher has no mode switch (region_match/egg.py:check_equal always builds an
e-graph), so it runs unchanged in both modes.
"""
from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402,F401  (sets PATH / sys.path)
import match_v2 as MV  # noqa: E402

MODE = "egglog"
_orig_import = MV.import_kernel_match


def _patched_import(lib_path):
    km = _orig_import(lib_path)
    km.configure_matcher(MODE)
    return km


MV.import_kernel_match = _patched_import


class JobTimeout(Exception):
    pass


def _vt(*_):
    raise JobTimeout()


def region_launch_blocks(text: str, region_specs: set[str]) -> list[dict]:
    """Recover each region-matcher launch block from the matched text: a
    maximal run of `%rm<lo>_<hi>_*` definitions, memref.copy and void
    kernel.launch lines of region specs (one block may hold several launches)."""
    lines = text.splitlines()

    def is_launch(ln):
        m = re.search(r"kernel\.launch\s+@([\w$.\-]+)\(", ln)
        return bool(m and m.group(1) in region_specs and "-> ()" in ln)

    def part(ln):
        return (is_launch(ln) or re.match(r"\s*%rm\d+_\d+_", ln) is not None
                or re.match(r"\s*memref\.copy\b", ln) is not None)
    out, i = [], 0
    while i < len(lines):
        if not part(lines[i]):
            i += 1
            continue
        k = i
        while k < len(lines) and part(lines[k]):
            k += 1
        block = lines[i:k]
        specs = [re.search(r"kernel\.launch\s+@([\w$.\-]+)\(", l).group(1) for l in block if is_launch(l)]
        if specs:
            out.append({"spec": "+".join(specs), "text": "\n".join(block), "line_range": [i, k - 1]})
        i = k
    return out


def run_job(job: dict, timeout: int) -> dict:
    t0 = time.time()
    res = {"id": job["id"], "input": job["input"], "mode": MODE, "region": job.get("region", True)}
    od = Path(job["out_dir"])
    text = Path(job["input"]).read_text()
    signal.signal(signal.SIGVTALRM, _vt)
    signal.setitimer(signal.ITIMER_VIRTUAL, timeout)
    try:
        out, report, extra = MV.run(text, C.LIB, region=job.get("region", True), diag=True,
                                    region_budget=job.get("region_budget", 600))
        (od / "matched.mlir").write_text(out)
        C.write_json(od / "report.json", {"launches": report, **extra})
        res["linalg_specs"] = [l["spec"] for l in report if "spec" in l]
        res["region_matches"] = [l["region_match"] for l in report
                                 if isinstance(l.get("region_match"), str)
                                 and l["region_match"].startswith("netlib_")]
        # one region match may emit several launches ("netlib_dscal_inc+netlib_daxpy_inc")
        res["region_specs"] = [s for m in res["region_matches"] for s in m.split("+")]
        res["region_status"] = [l["region_match"] for l in report
                                if isinstance(l.get("region_match"), str)
                                and not l["region_match"].startswith("netlib_")]
        res["region_skipped"] = [l for l in report if "region_match_skipped" in l]
        res["launches"] = report
        res["alternatives"] = extra.get("alternatives", [])
        res["rejected"] = extra.get("rejected", [])
        res["residual_generics"] = [
            {"generic": g.get("app_generic"), "iters": g.get("iters"), "form": g.get("form"),
             "n_body_matches": g.get("n_body_matches"),
             "body_matches": g.get("body_matches", [])[:4],
             "furthest": g.get("furthest_dag_stage")}
            for g in extra.get("residual_generics", [])]
        res["n_generics_in_input"] = len(re.findall(r"\blinalg\.generic\b", text))
        res["region_blocks"] = region_launch_blocks(out, set(res["region_specs"]))
        try:
            res["telemetry"] = MV.KM.matcher_telemetry()
        except Exception:  # noqa: BLE001
            pass
        res["status"] = "ok"
        if job.get("linalg_only_inject") and res["linalg_specs"]:
            out2, rep2, _ = MV.run(text, C.LIB, region=False, diag=False)
            syms = sorted(set(re.findall(r"kernel\.launch\s+@([\w$.\-]+)", out2)))
            import inject_kernel_library as IKL
            out2 = IKL.inject(out2, C.LIB.read_text(), syms)
            (od / "linalg_only_injected.mlir").write_text(out2)
            res["linalg_only_specs"] = [l["spec"] for l in rep2 if "spec" in l]
        if job.get("region_diag"):
            try:
                from region_match.matcher import match_text
                stats: dict = {}
                ms = match_text(text, str(C.SPEC_DIR), None, stats)
                res["region_diag"] = {
                    "matches_on_input": [m.spec for m in ms],
                    "evidence": [m.evidence[:600] for m in ms],
                    "rejections": [r[:300] for r in stats.get("rejections", [])][:12],
                    "windows": [{k: w.get(k) for k in ("window", "matched", "also", "bindings_tried")}
                                for w in stats.get("windows", [])][:12]}
            except JobTimeout:
                raise
            except Exception as e:  # noqa: BLE001
                res["region_diag"] = {"error": f"{type(e).__name__}: {e}"[:500]}
    except JobTimeout:
        res.update(status="timeout", error=f"job CPU time > {timeout}s")
    except Exception as e:  # noqa: BLE001
        res.update(status="error", error=f"{type(e).__name__}: {e}"[:1500],
                   traceback=traceback.format_exc()[-3000:])
    finally:
        signal.setitimer(signal.ITIMER_VIRTUAL, 0)
    res.setdefault("linalg_specs", [])
    res.setdefault("region_specs", [])
    res.setdefault("launches", [])
    res["seconds"] = round(time.time() - t0, 2)
    C.write_json(od / "result.json", res)
    return res


def main() -> int:
    global MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--mode", default="egglog", choices=["egglog", "syntactic"])
    ap.add_argument("--job-timeout", type=int, default=900)
    a = ap.parse_args()
    MODE = a.mode
    for job in json.loads(Path(a.jobs).read_text()):
        Path(job["out_dir"]).mkdir(parents=True, exist_ok=True)
        r = run_job(job, a.job_timeout)
        print(f"{r['id']}: {r['status']} {r['seconds']}s {r['linalg_specs']} {r['region_specs']}",
              file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
