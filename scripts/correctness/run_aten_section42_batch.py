#!/usr/bin/env python3
"""Run one resumable ATen Section 4.2 silicon batch."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "issues/aten_c_kernels/native_cuda_results"
CAMPAIGN = RESULTS / "section42_campaign"
RUN_JETSON = ROOT / "scripts/correctness/run_jetson.sh"
BUILD_RAISED = ROOT / "scripts/correctness/aten_pointwise_graph_silicon.py"
BUILD_EXACT = ROOT / "scripts/correctness/build_aten_native_exact_region_arm64.sh"
BENCH_SHAPED = RESULTS / "bench_shaped.py"
SHAPE_SPECS = RESULTS / "resident_shape_specs.json"
METRICS = {
    "cpu": ("torch_cpu_us", "cpu_aten_1thread_wall_us"),
    "native": ("torch_resident_us", "native_aten_resident_wall_us"),
    "raised": ("raised_resident_wall_us",),
}


def read_batch(batch: int) -> list[dict[str, str]]:
    path = CAMPAIGN / f"batch_{batch:02d}.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run prepare_aten_section42_campaign.py")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def completed(log_dir: Path, phase: str) -> set[str]:
    metrics = "|".join(re.escape(item) for item in METRICS[phase])
    pattern = re.compile(rf"SAMPLES kernel=(\S+) (?:{metrics})=([^\s]+)")
    sampled = set()
    correct = set()
    for path in log_dir.glob(f"batch_*_{phase}_*.log"):
        text = path.read_text(errors="replace")
        if (phase == "cpu" and "_torch_" in path.name and
                not re.search(r"META .*threads=1 interop_threads=1", text)):
            continue
        for line in text.splitlines():
            match = pattern.search(line)
            if match and len(match.group(2).split(",")) == 5:
                sampled.add(match.group(1))
        for match in re.finditer(
                r"(?:NATIVE_)?RESULT kernel=(\S+).*?errors=(\d+)", text):
            if match.group(2) == "0":
                correct.add(match.group(1))
    return sampled if phase == "cpu" else sampled & correct


def write_wrapper(path: Path, kind: str) -> None:
    common = """#!/usr/bin/env bash
set -uo pipefail
backend=$1
IFS=, read -r -a kernels <<< "$2"
here=$(cd "$(dirname "$0")" && pwd)
rc=0
"""
    if kind == "torch":
        body = """for kernel in "${kernels[@]}"; do
  if [[ "$backend" == cpu ]]; then
    PYTHONPATH=/home/nvidia/tpy-cpu-site ATEN_BENCH_DEVICE=cpu \\
      ATEN_BENCH_CPU_THREADS=1 ATEN_BENCH_CPU_INTEROP_THREADS=1 \\
      OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
      VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
      ATEN_BENCH_TIMING=sync_wall ATEN_BENCH_KERNELS="$kernel" \\
      taskset -c 0 /usr/bin/python3 "$here/bench_shaped.py" \\
      "$here/resident_shape_specs.json" || rc=1
  else
    ATEN_BENCH_DEVICE=cuda ATEN_BENCH_TIMING=sync_wall \\
      ATEN_BENCH_KERNELS="$kernel" \\
      /home/nvidia/venv/bin/python "$here/bench_shaped.py" "$here/resident_shape_specs.json" || rc=1
  fi
done
exit "$rc"
"""
    elif kind == "exact":
        body = """for kernel in "${kernels[@]}"; do
  if [[ "$backend" == cpu ]]; then
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
      taskset -c 0 "$here/aten_native_exact_region_bench" "$kernel" cpu || rc=1
  else
    "$here/aten_native_exact_region_bench" "$kernel" cuda || rc=1
  fi
done
exit "$rc"
"""
    elif kind == "original":
        body = """for kernel in "${kernels[@]}"; do
  taskset -c 0 "$here/${kernel}_cpu_reference" || rc=1
done
exit "$rc"
"""
    else:
        body = """for kernel in "${kernels[@]}"; do
  "$here/$kernel" || rc=1
done
exit "$rc"
"""
    path.write_text(common + body)
    path.chmod(0o755)


def jetson_run(wrapper: Path, extras: list[Path], args: str, tag: str,
                log: Path, dry_run: bool) -> int:
    env = os.environ.copy()
    env.update({
        "POLYGEIST_SILICON_PROFILE": env.get("POLYGEIST_SILICON_PROFILE", "pva-general"),
        "POLYGEIST_JETSON_RUNS": "1",
        "POLYGEIST_JETSON_RUN_ARGS": args,
        "POLYGEIST_JETSON_EXTRA_LIBS": " ".join(str(path) for path in extras),
        # Publication resident timing warms library plans before sampling.
        "POLYGEIST_CUDNN_PLAN_CACHE": env.get(
            "POLYGEIST_CUDNN_PLAN_CACHE", "1"),
    })
    command = [str(RUN_JETSON)]
    if dry_run:
        command.append("--dry-run")
    command += ["--exe", str(wrapper), tag]
    print("+", shlex.join(command), flush=True)
    if dry_run:
        return subprocess.run(command, cwd=ROOT, env=env).returncode
    with log.open("w") as stream:
        return subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                              stderr=subprocess.STDOUT, text=True).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--phase", choices=("build", "cpu", "native", "raised"),
                        required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--work", type=Path, default=Path("/tmp/aten-section42"))
    parser.add_argument(
        "--kernel", action="append", default=[],
        help="limit this batch to a kernel (repeatable)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    rows = read_batch(args.batch)
    if args.kernel:
        requested = set(args.kernel)
        available = {row["kernel"] for row in rows}
        unknown = sorted(requested - available)
        if unknown:
            raise SystemExit(
                f"kernels not in batch {args.batch}: {', '.join(unknown)}")
        rows = [row for row in rows if row["kernel"] in requested]
    args.work.mkdir(parents=True, exist_ok=True)
    log_dir = args.work / "logs"
    log_dir.mkdir(exist_ok=True)
    done = set() if args.no_resume or args.phase == "build" else completed(log_dir, args.phase)
    rows = [row for row in rows if row["kernel"] not in done]
    print(f"batch={args.batch} phase={args.phase} pending={len(rows)} resumed={len(done)}")
    if not rows:
        return 0

    raised_dir = args.work / "raised"
    if args.phase == "build":
        already_built = {
            row["kernel"] for row in rows
            if (raised_dir / row["kernel"] / row["kernel"]).exists()
        }
        rows = [row for row in rows if row["kernel"] not in already_built]
        if already_built:
            print(f"build resume: preserving {len(already_built)} existing binaries")
        if not rows:
            return 0
        python = os.environ.get("PYTHON", "/usr/bin/python3.10")
        if not Path(python).exists():
            python = sys.executable
        command = [python, str(BUILD_RAISED), "--output", str(raised_dir),
                   "--jobs", str(args.jobs)]
        for row in rows:
            command += ["--kernel", row["kernel"]]
        print("+", shlex.join(command), flush=True)
        if args.dry_run:
            return 0
        env = os.environ.copy()
        env["POLYGEIST_FORCE_RESIDENT"] = "1"
        # Compare preserved and normalized projected-view forms and select by
        # residual IR plus legal launch count, never by fixture name.
        env["POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE"] = "auto"
        return subprocess.run(command, cwd=ROOT, env=env).returncode

    groups: list[tuple[str, list[dict[str, str]]]]
    if args.phase == "cpu":
        groups = [
            ("torch", [row for row in rows if row[f"{'cpu' if args.phase == 'cpu' else 'native_gpu'}_runner"] == "torch_recipe"]),
            ("exact", [row for row in rows if row[f"{'cpu' if args.phase == 'cpu' else 'native_gpu'}_runner"] in {"exact_region_harness", "aten_cpp_harness"}]),
            ("original", [row for row in rows
                          if row["cpu_runner"] == "original_c_harness"]),
        ]
    elif args.phase == "native":
        groups = [
            ("torch", [row for row in rows
                       if row["native_gpu_runner"] == "torch_recipe"]),
            ("exact", [row for row in rows if row["native_gpu_runner"] in
                       {"exact_region_harness", "aten_cpp_harness"}]),
        ]
    else:
        groups = [("raised", rows)]

    result = 0
    for kind, selected in groups:
        if not selected:
            continue
        kernels = [row["kernel"] for row in selected]
        wrapper = args.work / f"run_{kind}_batch.sh"
        write_wrapper(wrapper, kind)
        if kind == "torch":
            extras = [BENCH_SHAPED, SHAPE_SPECS]
            backend = "cpu" if args.phase == "cpu" else "cuda"
        elif kind == "exact":
            exact_binary = args.work / "aten_native_exact_region_bench"
            source = RESULTS / "native_exact_region_bench.cpp"
            stale = (not exact_binary.exists() or
                     exact_binary.stat().st_mtime < source.stat().st_mtime)
            if stale and not args.dry_run:
                subprocess.run([str(BUILD_EXACT), str(exact_binary)], cwd=ROOT,
                               check=True)
            extras = [exact_binary]
            extras.extend(Path(path) for path in shlex.split(
                os.environ.get("ATEN_SECTION42_NATIVE_EXTRA_LIBS", "")))
            backend = "cpu" if args.phase == "cpu" else "cuda"
        elif kind == "original":
            binaries = {
                kernel: raised_dir / kernel / f"{kernel}_cpu_reference"
                for kernel in kernels
            }
            missing = [kernel for kernel, path in binaries.items()
                       if not path.exists()]
            if missing and not args.dry_run:
                print("missing original-C CPU binaries: " + ", ".join(missing),
                      file=sys.stderr)
                result |= 1
                continue
            extras = [binaries[kernel] for kernel in kernels]
            backend = "run"
        else:
            binaries = {kernel: raised_dir / kernel / kernel for kernel in kernels}
            missing = [kernel for kernel, path in binaries.items() if not path.exists()]
            if missing and not args.dry_run:
                print("missing raised binaries (left pending): " + ", ".join(missing),
                      file=sys.stderr)
                result |= 1
                kernels = [kernel for kernel in kernels if kernel not in missing]
                selected = [row for row in selected if row["kernel"] in kernels]
                if not kernels:
                    result |= 1
                    continue
            extras = [binaries[kernel] for kernel in kernels]
            backend = "run"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log = log_dir / f"batch_{args.batch:02d}_{args.phase}_{kind}_{stamp}.log"
        rc = jetson_run(wrapper, extras, f"{backend} {','.join(kernels)}",
                        f"aten_s42_b{args.batch:02d}_{args.phase}_{kind}", log,
                        args.dry_run)
        result |= rc
        print(f"group={kind} kernels={len(kernels)} rc={rc} log={log}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
