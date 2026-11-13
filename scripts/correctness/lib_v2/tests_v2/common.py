"""Shared helpers for the lib_v2 test / robustness harness (tests_v2).

Read-only reuse of the lib_v2 drivers:
  * recipe.py           -- the one raising recipe (specs == application code)
  * raise_polybench.py  -- PolyBench cflags + manifest
  * match_v2.py         -- imported in-process by worker.py (never modified)

Everything this harness writes lives under issues/lib_v2/tests/.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIBV2 = HERE.parent
CORR = LIBV2.parent
REPO = CORR.parents[1]
sys.path.insert(0, str(LIBV2))
sys.path.insert(0, str(CORR))

import recipe as R  # noqa: E402

OUT_ROOT = REPO / "issues/lib_v2/tests"
CASES = HERE / "cases"
SPEC_DIR = REPO / "generic_solver/lib_v2"
LIB = REPO / "generic_solver/kernel_library_v2.mlir"
PB_MANIFEST = REPO / "issues/polybench_section42/manifest.csv"
PB_UTIL = REPO / "tools/cgeist/Test/polybench/utilities"
PY = "/usr/bin/python3"

# The 12 PolyBench kernels lib_v2 currently matches (issues/lib_v2/REPORT.md).
LINALG_KERNELS = ["gemm", "2mm", "3mm", "atax", "bicg", "mvt", "gesummv", "gemver", "doitgen"]
REGION_KERNELS = ["trisolv", "cholesky", "syrk"]
MATCHED_KERNELS = LINALG_KERNELS + REGION_KERNELS


def env() -> dict:
    """envsetup.sh equivalent: build/bin first on PATH."""
    e = dict(os.environ)
    e["PATH"] = f"{REPO / 'build/bin'}:{e.get('PATH', '')}"
    e["POLYGEIST_ROOT"] = str(REPO)
    return e


os.environ.update(env())


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=str))


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def pmap(fn, items, j: int):
    items = list(items)
    if j <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(j) as ex:
        return list(ex.map(fn, items))


def parse_meta(c_path: Path) -> dict:
    """`//@ key: value` header lines of a test C file."""
    meta = {}
    for ln in Path(c_path).read_text().splitlines():
        m = re.match(r"\s*//@\s*([\w-]+)\s*:\s*(.*?)\s*$", ln)
        if m:
            meta[m.group(1)] = m.group(2)
    return meta


def c_function_name(c_path: Path) -> str:
    """The test convention: the kernel function is named like the file stem."""
    meta = parse_meta(c_path)
    return meta.get("function", Path(c_path).stem)


def polybench_sources() -> dict[str, Path]:
    import csv
    with PB_MANIFEST.open() as fh:
        return {r["kernel"]: REPO / r["source"] for r in csv.DictReader(fh)}


def polybench_cflags(src: Path) -> list[str]:
    from raise_polybench import polybench_cflags as f  # read-only reuse
    return f(src)


def raise_c(src: Path, fn: str, prefix: Path, cflags: list[str], timeout: int = 600) -> dict:
    """cgeist -> raise -> debuf with the shared production recipe.
    Returns {"ok", "failed_step", "stderr", "paths", "seconds"}."""
    prefix.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = R.raise_c(R.PRODUCTION, str(src), fn, str(prefix), cflags, timeout)
    out = {"paths": res["paths"], "seconds": round(time.time() - t0, 2), "ok": True,
           "commands": []}
    for step in ("cgeist", "raise", "debuf"):
        r = res.get(step)
        if r is None:
            out.update(ok=False, failed_step=out.get("failed_step") or step)
            break
        out["commands"].append(R.shell_line(r.cmd))
        if not r.ok:
            out.update(ok=False, failed_step=step,
                       stderr=("TIMEOUT " if r.timed_out else "") + r.stderr[-1500:])
            break
    return out


def raise_polybench(kernel: str, d: Path) -> dict:
    """Fresh raise of a PolyBench kernel (same flags as raise_polybench.py)."""
    src = polybench_sources()[kernel]
    fn = "kernel_" + kernel.replace("-", "_")
    r = raise_c(src, fn, d / "pb", polybench_cflags(src))
    r.update(kernel=kernel, function=fn, source=str(src))
    return r


# ---------------------------------------------------------------------------
# Batch matching through worker.py (fresh processes; crash / timeout isolated)
# ---------------------------------------------------------------------------

def match_batch(jobs: list[dict], j: int = 6, mode: str = "egglog", chunk: int = 8,
                job_timeout: int = 900, label: str = "match") -> dict[str, dict]:
    """jobs: [{"id", "input", "out_dir", "region": bool, "region_diag": bool}].
    Each job writes <out_dir>/result.json (+ matched.mlir).  Returns id -> result.
    Jobs are run in chunks inside long-lived worker processes so the region
    matcher's spec loading is paid once per chunk; a chunk whose worker dies
    is retried job-by-job, so one crash cannot hide other results."""
    for jb in jobs:
        Path(jb["out_dir"]).mkdir(parents=True, exist_ok=True)
        rp = Path(jb["out_dir"]) / "result.json"
        if rp.exists():
            rp.unlink()
    chunks = [jobs[i:i + chunk] for i in range(0, len(jobs), chunk)]
    tmp = OUT_ROOT / "work" / "_jobs"
    tmp.mkdir(parents=True, exist_ok=True)

    def run_chunk(arg):
        k, ch = arg
        jf = tmp / f"{label}_{mode}_{os.getpid()}_{k}.json"
        jf.write_text(json.dumps(ch))
        try:
            p = subprocess.run([PY, str(HERE / "worker.py"), "--jobs", str(jf), "--mode", mode,
                                "--job-timeout", str(job_timeout)],
                               capture_output=True, text=True,
                               timeout=job_timeout * len(ch) + 120, env=env())
            rc, err = p.returncode, p.stderr[-3000:]
        except subprocess.TimeoutExpired as e:
            rc, err = -9, f"worker wall-clock timeout ({e.timeout}s)"
        missing = [jb for jb in ch if not (Path(jb["out_dir"]) / "result.json").exists()]
        return rc, err, missing

    results = pmap(run_chunk, list(enumerate(chunks)), j)
    retry = []
    for rc, err, missing in results:
        if missing and len(missing) < 999:
            retry += [(jb, rc, err) for jb in missing]
    # retry individually
    def run_one(item):
        jb, rc0, err0 = item
        rc, err, missing = run_chunk((f"retry_{abs(hash(jb['id']))}", [jb]))
        if missing:
            write_json(Path(jb["out_dir"]) / "result.json",
                       {"id": jb["id"], "status": "crash", "worker_rc": rc,
                        "error": err[-2000:], "launches": [], "linalg_specs": [],
                        "region_specs": []})
    pmap(run_one, retry, j)
    out = {}
    for jb in jobs:
        out[jb["id"]] = read_json(Path(jb["out_dir"]) / "result.json",
                                  {"status": "missing", "launches": []})
    return out


def launch_specs(res: dict) -> list[str]:
    """All lib_v2 launches of a worker result (Linalg level + region)."""
    return list(res.get("linalg_specs", [])) + list(res.get("region_specs", []))


def multiset_retained(orig: list[str], new: list[str]) -> int:
    from collections import Counter
    a, b = Counter(orig), Counter(new)
    return sum(min(a[k], b[k]) for k in a)
