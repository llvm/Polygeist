#!/usr/bin/env python3
"""validate_v2.py -- end-to-end check of the lib_v2 library-call path.

For each PolyBench kernel and dataset:
  * reference: the clang-compiled original (e2e_v2.py convention: -O1,
    all arrays dumped at %.17g);
  * candidate: build_v2.sh (raise -> match_v2 -> lower_v2 -> LLVM -> link with
    the v2 runtime), run with POLYGEIST_V2_TRACE=1 so the runtime reports every
    library call it made;
  * evidence that the library is really called: runtime call counters, the
    undefined Fortran/cuBLAS symbols of the executable and its DT_NEEDED
    libraries, and the lower_v2 launch report.

--target=host    build and run on this machine (Fortran BLAS/LAPACK, OpenBLAS)
--target=jetson  cross-compile; with --remote, deploy and run on the Orin
                 through the existing scripted hop used by
                 issues/polybench_section42/run_raised_gpu_timing.sh
                 (ssh <configured SSH hop> -> <configured Jetson target>, BatchMode keys)
--timing         -DPOLYBENCH_TIME build; 1 warm-up + N timed runs.

Outputs: <out>/<target>/<dataset>/<kernel>/{log.txt,result.json,...} and
<out>/<target>_<dataset>_summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import e2e_v2 as E  # noqa: E402  (read-only reuse: sources, with_precision, compare)
from generate_polybench_harness import generate_harness  # noqa: E402

REPO = E.REPO
UTIL = E.UTIL
BUILD = HERE / "build_v2.sh"
DEFAULT_KERNELS = ["gemm", "2mm", "3mm", "atax", "bicg", "mvt", "gesummv", "gemver",
                   "doitgen", "trisolv", "cholesky", "syrk"]
# Access data stays in a private local profile, with environment overrides.
REMOTE_CONFIG = Path(os.environ.get(
    "POLYGEIST_V2_REMOTE_CONFIG",
    str(Path.home() / ".config/polygeist-silicon/lib-v2-remote.json")))
_remote = json.loads(REMOTE_CONFIG.read_text()) if REMOTE_CONFIG.is_file() else {}
HOP = os.environ.get("POLYGEIST_V2_SSH_HOP", _remote.get("hop", ""))
ORIN = os.environ.get("POLYGEIST_V2_SSH_TARGET", _remote.get("target", ""))
HOST_CPU = os.environ.get("POLYGEIST_V2_HOST_CPU", "21")
# Library path = the default of scripts/correctness/run_jetson.sh: the
# Jetson-native (aarch64-linux) cuBLAS in jetson-cuda-libs must precede the
# SBSA toolkit in cuda-12.6/lib64.  With only cuda-12.6/lib64 (as in
# issues/polybench_section42/run_raised_gpu_timing.sh) cuBLAS returns
# CUBLAS_STATUS_SUCCESS on the Orin but leaves every output untouched.
ORIN_LD = os.environ.get(
    "POLYGEIST_JETSON_LD_LIBRARY_PATH",
    "/home/nvidia/jetson-cuda-libs:/home/nvidia/cuda-12.6/lib64:"
    "/home/nvidia/cuda-12.6/targets/aarch64-linux/lib:/home/nvidia/polygeist_cuda_libs:"
    "/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu:"
    "/lib/aarch64-linux-gnu:/home/nvidia/cuda-12.6/targets/sbsa-linux/lib")
ORIN_ENV = f"export LD_LIBRARY_PATH={ORIN_LD}"


def sh(cmd, log: Path, timeout=1800, env=None, stdout=None, stderr=None) -> int:
    with log.open("a") as fh:
        fh.write("$ " + " ".join(shlex.quote(str(c)) for c in cmd) + "\n")
        fh.flush()
        try:
            p = subprocess.run([str(c) for c in cmd], stdout=stdout or fh, stderr=stderr or fh,
                               timeout=timeout, env=env)
            return p.returncode
        except subprocess.TimeoutExpired:
            fh.write("TIMEOUT\n")
            return 124


def flags_for(src: Path, dataset: str, timing: bool) -> list[str]:
    f = [f"-I{UTIL}", f"-I{src.parent}", "-DDATA_TYPE_IS_DOUBLE", f"-D{dataset}_DATASET",
         "-DPOLYBENCH_USE_C99_PROTO"]
    f.append("-DPOLYBENCH_TIME" if timing else "-DPOLYBENCH_DUMP_ARRAYS")
    return f


def normwise(ref: Path, out: Path) -> dict:
    def vals(p):
        t = p.read_text(errors="replace")
        if "BEGIN DUMP_ARRAYS" in t:
            t = t.split("BEGIN DUMP_ARRAYS", 1)[1].split("END   DUMP_ARRAYS", 1)[0]
        return [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|inf|nan)", t)]
    a, b = vals(ref), vals(out)
    if len(a) != len(b) or not a:
        return {"normwise_rel": None}
    num = max(abs(x - y) for x, y in zip(a, b))
    den = max(abs(x) for x in a) or 1.0
    return {"normwise_rel": num / den}


def parse_trace(text: str) -> dict:
    stats = {}
    for m in re.finditer(r"^POLYGEIST_V2_STATS (.*)$", text, flags=re.M):
        kv = dict(x.split("=", 1) for x in m.group(1).split())
        stats[kv["call"]] = kv
    return stats


def evidence(exe: Path, target: str) -> dict:
    nm = "nm" if target == "host" else "aarch64-linux-gnu-nm"
    syms = subprocess.run([nm, str(exe)], capture_output=True, text=True).stdout
    und = sorted({l.split()[-1].split("@")[0] for l in syms.splitlines()
                  if re.search(r" U ", l) and
                  re.search(r"(^|\s)([a-z]+\d*_|cublas\w+|cusolver\w+)(@\S+)?$", l)})
    v2 = sorted({l.split()[-1] for l in syms.splitlines() if " T polygeist_v2_" in l})
    dyn = subprocess.run(["readelf", "-d", str(exe)], capture_output=True, text=True).stdout
    needed = re.findall(r"\(NEEDED\).*\[(.*)\]", dyn)
    return {"undefined_library_symbols": und, "v2_wrappers_linked": v2, "needed": needed}


def run_remote(local: Path, tag: str, log: Path, env: str, stdout: Path, stderr: Path,
               timeout=900) -> int:
    remote = f"/tmp/lib_v2_gpu_{tag}"
    rc = sh(["scp", "-o", "BatchMode=yes", local, f"{HOP}:{remote}.stage"], log)
    if rc == 0:
        rc = sh(["ssh", "-o", "BatchMode=yes", HOP,
                 f"scp -o BatchMode=yes '{remote}.stage' {ORIN}:'{remote}'"], log)
    if rc:
        return 200 + rc
    with stdout.open("w") as so, stderr.open("w") as se:
        rc = sh(["ssh", "-o", "BatchMode=yes", HOP,
                 f"ssh -o BatchMode=yes {ORIN} '{ORIN_ENV}; {env} timeout {timeout}s {remote}'"],
                log, timeout=timeout + 120, stdout=so, stderr=se)
    return rc


def cleanup_remote(tag: str, log: Path):
    remote = f"/tmp/lib_v2_gpu_{tag}"
    sh(["ssh", "-o", "BatchMode=yes", HOP,
        f"rm -f '{remote}.stage'; ssh -o BatchMode=yes {ORIN} 'rm -f {remote}'"], log)


def reference(kernel: str, src: Path, dataset: str, d: Path, log: Path) -> Path | None:
    ref_out = d / "ref.out"
    if ref_out.exists() and ref_out.stat().st_size:
        return ref_out
    ref_src = d / "ref_src.c"
    ref_src.write_text(E.with_precision(src.read_text(), f"{src.stem}.h"))
    flags = flags_for(src, dataset, False)
    if sh([E.CLANG, "-O1", *flags, "-Dstatic=", ref_src, UTIL / "polybench.c", "-lm",
           "-o", d / "ref_exe"], log):
        return None
    with ref_out.open("w") as fh:
        subprocess.run([d / "ref_exe"], stderr=fh, stdout=subprocess.DEVNULL, timeout=1800)
    return ref_out


def build(kernel: str, src: Path, dataset: str, target: str, d: Path, log: Path,
          timing: bool, extra: list[str], reuse: bool = False) -> Path | None:
    fn = "kernel_" + kernel.replace("-", "_")
    harness = d / "harness.c"
    exe = d / f"{kernel}_{target}{'_timed' if timing else ''}"
    if reuse and exe.exists() and (d / "work" / "lower.json").exists():
        return exe
    text = generate_harness(src.read_text(), fn)
    harness.write_text(text if timing else E.with_precision(text, f"{src.stem}.h"))
    rc = sh(["bash", BUILD, f"--target={target}", f"--function={fn}", f"--harness={harness}",
             f"--work={d / 'work'}", "-o", exe, *extra, src, *flags_for(src, dataset, timing),
             "-Dstatic=__attribute__((noipa))"], log)
    return exe if rc == 0 and exe.exists() else None


def run_kernel(kernel: str, src: Path, dataset: str, target: str, out: Path,
               remote: bool, timing: bool, samples: int, extra: list[str],
               reuse: bool = False, remote_env: str = "") -> dict:
    d = out / target / (dataset + ("_timing" if timing else "")) / kernel
    d.mkdir(parents=True, exist_ok=True)
    log = d / "log.txt"
    if not reuse:
        log.write_text("")
    row = {"kernel": kernel, "dataset": dataset, "target": target}
    t0 = time.time()
    exe = build(kernel, src, dataset, target, d, log, timing, extra, reuse)
    row["build_seconds"] = round(time.time() - t0, 1)
    if exe is None:
        row["status"] = "BUILD_FAIL"
        tail = log.read_text().splitlines()[-15:]
        row["error"] = "\n".join(tail)
        return row
    lower_rep = d / "work" / "lower.json"
    if lower_rep.exists():
        lr = json.loads(lower_rep.read_text())
        row["launches"] = [{"spec": l["spec"], "call": l["call"],
                            "cpu_entry": l["cpu_entry"], "gpu_entry": l["gpu_entry"]}
                           for l in lr.get("launches", [])]
    mr = d / "work" / "match.json"
    if mr.exists():
        rep = json.loads(mr.read_text())
        row["matched_specs"] = [l["spec"] for l in rep.get("launches", []) if "spec" in l] + \
            [l["region_match"] for l in rep.get("launches", []) if "region_match" in l and
             str(l["region_match"]).startswith("netlib_")]
    row["evidence"] = evidence(exe, target)
    env = f"POLYGEIST_V2_TRACE=1 {remote_env}".strip()
    row["remote_env"] = env if target != "host" else None
    if target != "host" and not remote:
        row["status"] = "CROSS_BUILT"
        row["binary"] = str(exe)
        (d / "result.json").write_text(json.dumps(row, indent=1))
        return row
    if timing:
        runs, traces = [], []
        for k in range(samples + 1):
            so, se = d / f"run{k}.stdout", d / f"run{k}.stderr"
            if target == "host":
                # production CPU timing convention (run_cpu_timing.sh): one
                # pinned core, single-threaded OpenBLAS
                with so.open("w") as fo, se.open("w") as fe:
                    rc = subprocess.run(["taskset", "-c", HOST_CPU, exe], stdout=fo, stderr=fe,
                                        timeout=1800,
                                        env={**os.environ, "POLYGEIST_V2_TRACE": "1",
                                             "OPENBLAS_NUM_THREADS": "1",
                                             "OMP_NUM_THREADS": "1"}).returncode
            else:
                rc = run_remote(exe, f"{kernel}_{dataset}_t", log, env, so, se)
            if rc:
                row["status"] = f"RUN_FAIL rc={rc}"
                row["error"] = se.read_text()[-1500:] if se.exists() else ""
                return row
            if k == 0:
                continue  # warm-up
            vals = re.findall(r"^\s*([0-9]+\.[0-9]+)\s*$", so.read_text(), flags=re.M)
            runs.append(float(vals[-1]) * 1e3 if vals else None)
            traces.append(parse_trace(se.read_text()))
        if target != "host":
            cleanup_remote(f"{kernel}_{dataset}_t", log)
        row["e2e_ms_samples"] = runs
        row["e2e_ms_median"] = statistics.median([r for r in runs if r is not None]) if any(runs) else None
        per_call = {}
        for tr in traces:
            for c, kv in tr.items():
                pc = per_call.setdefault(c, {"lib_ms": [], "h2d_ms": [], "d2h_ms": [], "count": kv["count"]})
                for key in ("lib_ms", "h2d_ms", "d2h_ms"):
                    if key in kv:
                        pc[key].append(float(kv[key]))
        row["per_call"] = {c: {"count": v["count"],
                               **{k + "_median": statistics.median(v[k]) for k in ("lib_ms", "h2d_ms", "d2h_ms") if v[k]}}
                           for c, v in per_call.items()}
        lib = [sum(float(kv["lib_ms"]) for kv in tr.values()) for tr in traces]
        row["lib_ms_median"] = statistics.median(lib) if lib else None
        mem = [sum(float(kv.get("h2d_ms", 0)) + float(kv.get("d2h_ms", 0)) for kv in tr.values()) for tr in traces]
        row["transfer_ms_median"] = statistics.median(mem) if mem and target != "host" else None
        row["status"] = "TIMED"
        (d / "result.json").write_text(json.dumps(row, indent=1))
        return row
    refdir = out / "ref" / dataset / kernel
    refdir.mkdir(parents=True, exist_ok=True)
    ref = reference(kernel, src, dataset, refdir, log)
    if ref is None:
        row["status"] = "REF_FAIL"
        return row
    cand = d / "v2.out"
    so = d / "v2.stdout"
    t1 = time.time()
    if target == "host":
        with cand.open("w") as fe, so.open("w") as fo:
            rc = subprocess.run([exe], stdout=fo, stderr=fe, timeout=1800,
                                env={**os.environ, "POLYGEIST_V2_TRACE": "1"}).returncode
    elif remote:
        rc = run_remote(exe, f"{kernel}_{dataset}", log, env, so, cand)
        cleanup_remote(f"{kernel}_{dataset}", log)
    else:
        row["status"] = "CROSS_BUILT"
        row["binary"] = str(exe)
        (d / "result.json").write_text(json.dumps(row, indent=1))
        return row
    row["run_seconds"] = round(time.time() - t1, 1)
    if rc:
        row["status"] = f"RUN_FAIL rc={rc}"
        row["error"] = cand.read_text()[-1500:]
        return row
    trace = parse_trace(cand.read_text())
    row["calls"] = {c: int(kv["count"]) for c, kv in trace.items()}
    row["lib_entries"] = {c: kv["entry"] for c, kv in trace.items()}
    cmp_ = E.compare(ref, cand)
    cmp_.update(normwise(ref, cand))
    row["compare"] = cmp_
    expected = len(row.get("launches", []))
    called = sum(row["calls"].values())
    ok_calls = expected > 0 and called >= expected
    row["status"] = ("PASS" if cmp_["verdict"].startswith("PASS") and ok_calls else
                     "FAIL_NO_LIBRARY_CALL" if cmp_["verdict"].startswith("PASS") else "FAIL")
    (d / "result.json").write_text(json.dumps(row, indent=1))
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", default="host", choices=["host", "jetson"])
    ap.add_argument("--datasets", default="MINI,SMALL,LARGE")
    ap.add_argument("--out", default=str(REPO / "issues/lib_v2_gpu"))
    ap.add_argument("--remote", action="store_true", help="run jetson binaries on the Orin")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--build-arg", action="append", default=[])
    ap.add_argument("--remote-env", default="",
                    help="extra VAR=value settings for remote runs (e.g. CUDA_MODULE_LOADING=EAGER)")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse an existing binary in the output tree (build once, run later)")
    ap.add_argument("kernels", nargs="*")
    a = ap.parse_args()
    if a.remote and (not HOP or not ORIN):
        ap.error("--remote requires a local remote profile or "
                 "POLYGEIST_V2_SSH_HOP and POLYGEIST_V2_SSH_TARGET")
    srcs = E.sources()
    ks = a.kernels or DEFAULT_KERNELS
    out = Path(a.out)
    for ds in a.datasets.split(","):
        # remote runs share one device: strictly sequential
        jobs = a.j if (a.target == "host" or not a.remote) else 1
        with ThreadPoolExecutor(jobs) as ex:
            rows = list(ex.map(lambda k: run_kernel(k, srcs[k], ds, a.target, out, a.remote,
                                                    a.timing, a.samples, a.build_arg,
                                                    a.reuse, a.remote_env), ks))
        tag = f"{a.target}_{ds}" + ("_timing" if a.timing else "")
        summ = out / f"{tag}_summary.json"
        old = json.loads(summ.read_text()) if summ.exists() else {}
        old.update({r["kernel"]: r for r in rows})
        summ.write_text(json.dumps(old, indent=1))
        for r in rows:
            c = r.get("compare", {})
            print(f"{tag:22s} {r['kernel']:9s} {r['status']:12s} "
                  f"calls={r.get('calls')} verdict={c.get('verdict')} "
                  f"maxrel={c.get('max_rel')} norm={c.get('normwise_rel')} "
                  f"e2e_ms={r.get('e2e_ms_median')} lib_ms={r.get('lib_ms_median')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
