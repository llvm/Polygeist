#!/usr/bin/env python3
"""Measure fresh end-to-end PolyBench builds sequentially.

Each row runs the complete source-to-AArch64-executable pipeline.  The two
ablation arms differ only in the scalar-expression matcher mode, and both
disable the handwritten semantic fallback.
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[2]
PRIMARY = Path("/home/arjaiswal/Polygeist")
DEFAULT_OUTPUT = ROOT / "issues/equality_saturation_eval/polybench_whole_compile_20260909"
FIELDS = (
    "kernel", "mode", "repetition", "status", "returncode", "timed_out",
    "wall_seconds", "user_seconds", "system_seconds", "peak_rss_kib",
    "started_utc", "finished_utc", "source", "source_sha256",
    "executable_sha256", "log", "time_log", "telemetry", "command",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_completed(path: Path):
    if not path.exists():
        return set()
    with path.open(newline="") as stream:
        return {(row["kernel"], row["mode"], int(row["repetition"]))
                for row in csv.DictReader(stream)}


def append_row(path: Path, row):
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def parse_time(path: Path):
    values = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--kernel", action="append", default=[])
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")

    manifest_path = ROOT / "issues/polybench_section42/manifest.csv"
    rows = read_manifest(manifest_path)
    requested = set(args.kernel)
    if requested:
        rows = [row for row in rows if row["kernel"] in requested]
        missing = requested - {row["kernel"] for row in rows}
        if missing:
            raise SystemExit(f"unknown kernels: {sorted(missing)}")

    args.output.mkdir(parents=True, exist_ok=True)
    summary = args.output / "runs.csv"
    completed = read_completed(summary)
    util = ROOT / "tools/cgeist/Test/polybench/utilities"
    build_script = ROOT / "scripts/correctness/polygeist_build.sh"

    tool_paths = {
        "cgeist": PRIMARY / "build/bin/cgeist",
        "polygeist_opt": PRIMARY / "build/bin/polygeist-opt",
        "mlir_opt": PRIMARY / "llvm-project/build/bin/mlir-opt",
        "mlir_translate": PRIMARY / "llvm-project/build/bin/mlir-translate",
        "clang": PRIMARY / "llvm-project/build/bin/clang",
        "build_script": build_script,
    }
    for name, path in tool_paths.items():
        if not path.exists():
            raise SystemExit(f"missing {name}: {path}")

    metadata = {
        "schema": 1,
        "scope": "C source through linked AArch64 executable",
        "target": "Jetson AArch64 cross-compile on x86 host; executable not run",
        "dataset": "LARGE",
        "datatype": "double",
        "modes": ["egglog", "syntactic"],
        "repetitions_per_mode": args.repetitions,
        "sequential": True,
        "semantic_fallback_disabled": True,
        "matcher_iteration_limit": 8,
        "matcher_ast_node_ceiling": 32,
        "per_candidate_wall_timeout_seconds": None,
        "matcher_bound_note": (
            "The in-process production matcher uses iteration and AST-size "
            "bounds rather than an isolated per-proof wall timer."
        ),
        "timeout_seconds_per_build": args.timeout,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "host": platform.node(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "load_average_at_campaign_start": os.getloadavg(),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "manifest_sha256": sha256(manifest_path),
        "tools": {name: {"path": str(path), "sha256": sha256(path)}
                  for name, path in tool_paths.items()},
        "runner_sha256": sha256(Path(__file__)),
    }
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    jobs = []
    for repetition in range(1, args.repetitions + 1):
        for index, row in enumerate(rows):
            modes = ("egglog", "syntactic")
            if (repetition + index) % 2:
                modes = tuple(reversed(modes))
            for mode in modes:
                jobs.append((row, mode, repetition))

    for job_index, (manifest_row, mode, repetition) in enumerate(jobs, 1):
        kernel = manifest_row["kernel"]
        identity = (kernel, mode, repetition)
        if identity in completed:
            print(f"[{job_index}/{len(jobs)}] skip {kernel} {mode} r{repetition}", flush=True)
            continue

        run_dir = args.output / "runs" / kernel / f"{mode}-r{repetition}"
        run_dir.mkdir(parents=True, exist_ok=True)
        source = ROOT / manifest_row["source"]
        function = "kernel_" + kernel.replace("-", "_")
        executable = run_dir / f"{kernel}.aarch64"
        harness = run_dir / "link_harness.c"
        harness.write_text(
            "#include <stdint.h>\n"
            f"extern void {function}(void);\n"
            f"static void (*volatile keep_kernel)(void) = {function};\n"
            "int main(void) { return keep_kernel == 0; }\n"
        )
        log = run_dir / "build.log"
        time_log = run_dir / "time.txt"
        telemetry = run_dir / "matcher-telemetry.json"
        command = [
            "/usr/bin/timeout", f"{args.timeout}s",
            "/usr/bin/time", "-f",
            "wall_seconds=%e\nuser_seconds=%U\nsystem_seconds=%S\npeak_rss_kib=%M",
            "-o", str(time_log),
            str(build_script), "--target=jetson", f"--function={function}",
            f"--harness={harness}",
            "-o", str(executable), str(source), "-O3", f"-I{util}",
            f"-I{source.parent}", "-Dstatic=__attribute__((noipa))",
            "-DLARGE_DATASET", "-DDATA_TYPE_IS_DOUBLE",
            "-DPOLYBENCH_USE_C99_PROTO", "-DPOLYBENCH_DUMP_ARRAYS",
        ]
        env = os.environ.copy()
        env.update({
            "POLYGEIST_ROOT": str(ROOT),
            "PATH": ":".join([
                str(PRIMARY / "build/bin"),
                str(PRIMARY / "llvm-project/build/bin"),
                env.get("PATH", ""),
            ]),
            "MLIR_OPT": str(tool_paths["mlir_opt"]),
            "MLIR_TRANSLATE": str(tool_paths["mlir_translate"]),
            "CLANG": str(tool_paths["clang"]),
            "PYTHON": "/usr/bin/python3",
            "POLYGEIST_MATCHER_MODE": mode,
            "POLYGEIST_MATCHER_DISABLE_SEMANTIC_FALLBACK": "1",
            "POLYGEIST_MATCHER_TELEMETRY_JSON": str(telemetry),
            "POLYGEIST_CUTENSORNET_ROOT": "/tmp/polygeist_cutensornet_aarch64/unified",
            "POLYGEIST_LOWER_SUBMAP_BEFORE_DEBUFFERIZE": "1",
        })

        print(f"[{job_index}/{len(jobs)}] {kernel} {mode} r{repetition}", flush=True)
        started = dt.datetime.now(dt.timezone.utc)
        with log.open("w") as stream:
            process = subprocess.run(command, cwd=ROOT, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
        finished = dt.datetime.now(dt.timezone.utc)
        timing = parse_time(time_log)
        row = {
            "kernel": kernel,
            "mode": mode,
            "repetition": repetition,
            "status": "ok" if process.returncode == 0 else "timeout" if process.returncode == 124 else "error",
            "returncode": process.returncode,
            "timed_out": str(process.returncode == 124).lower(),
            "wall_seconds": timing.get("wall_seconds", ""),
            "user_seconds": timing.get("user_seconds", ""),
            "system_seconds": timing.get("system_seconds", ""),
            "peak_rss_kib": timing.get("peak_rss_kib", ""),
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "source": str(source.relative_to(ROOT)),
            "source_sha256": sha256(source),
            "executable_sha256": sha256(executable) if executable.exists() else "",
            "log": str(log.relative_to(args.output)),
            "time_log": str(time_log.relative_to(args.output)),
            "telemetry": str(telemetry.relative_to(args.output)),
            "command": shlex.join(command),
        }
        append_row(summary, row)
        completed.add(identity)


if __name__ == "__main__":
    main()
