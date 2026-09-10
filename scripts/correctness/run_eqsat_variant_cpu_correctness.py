#!/usr/bin/env python3
"""Correctness-check Egglog-only PolyBench IR variants on the host CPU.

For each variant row where Egglog preserves an original selected identity that
the exact-syntax arm loses, this runner materializes Egglog's rewrite, lowers
it through the ordinary host/OpenBLAS pipeline, and compares the complete
LARGE FP64 output with a fresh native-Clang reference.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

from generate_polybench_harness import generate_harness


ROOT = Path(__file__).resolve().parents[2]
MATCHER = ROOT / "scripts/correctness/kernel_match_rewrite.py"
BUILDER = ROOT / "scripts/correctness/polygeist_build.sh"
COMPARE = ROOT / "scripts/correctness/compare_polybench_dumps.py"
UTILITIES = ROOT / "tools/cgeist/Test/polybench/utilities"
CLANG = ROOT / "llvm-project/build/bin/clang"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def append_csv(path: Path, row: dict[str, object]) -> None:
    if path.exists():
        columns = next(csv.reader(path.open(newline="")))
        with path.open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=columns,
                           lineterminator="\n").writerow(row)
    else:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row),
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerow(row)


def run(command: list[str], log: Path, *, env=None, stdout=None,
        stderr_output=None, timeout=900) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as errors:
        errors.write("COMMAND: " + shlex.join(command) + "\n")
        errors.flush()
        try:
            result = subprocess.run(
                command, stdout=stdout if stdout is not None else errors,
                stderr=stderr_output if stderr_output is not None else errors,
                env=env, timeout=timeout, check=False)
            return result.returncode
        except subprocess.TimeoutExpired:
            errors.write(f"TIMEOUT after {timeout}s\n")
            return 124


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--polybench-manifest", type=Path,
                        default=ROOT / "issues/polybench_section42/manifest.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int,
                        help="Validate only the first N selected rows (pilot use).")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    sources = {
        row["kernel"]: (ROOT / row["source"]).resolve()
        for row in read_csv(args.polybench_manifest)
    }
    candidates = [row for row in read_csv(args.analysis)
                  if int(row["egglog_only_recovered"]) > 0]
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        candidates = candidates[:args.limit]
    results_path = args.output / "results.csv"
    existing = read_csv(results_path) if results_path.exists() else []
    complete = {(row["kernel"], row["variant"]) for row in existing}

    environment = dict(os.environ)
    environment.update({
        "PATH": f"{ROOT / 'build/bin'}:{ROOT / 'llvm-project/build/bin'}:"
                + environment.get("PATH", ""),
        "MLIR_OPT": str(ROOT / "llvm-project/build/bin/mlir-opt"),
        "MLIR_TRANSLATE": str(ROOT / "llvm-project/build/bin/mlir-translate"),
        "CLANG": str(CLANG),
        "POLYGEIST_CPU_BLAS": "1",
        "POLYGEIST_CPU_BLAS_LIBS": "-lopenblas",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    })
    common_flags = [
        "-O3", f"-I{UTILITIES}", "-DLARGE_DATASET",
        "-DDATA_TYPE_IS_DOUBLE", "-DPOLYBENCH_USE_C99_PROTO",
        "-DPOLYBENCH_DUMP_ARRAYS",
    ]

    references: dict[str, Path] = {}
    harnesses: dict[str, Path] = {}
    for kernel in sorted({row["kernel"] for row in candidates}):
        source = sources[kernel]
        kernel_dir = source.parent
        log_dir = args.output / kernel / "native"
        executable = log_dir / "native"
        reference = log_dir / "output.txt"
        references[kernel] = reference
        if not (reference.is_file() and executable.is_file()):
            compile_command = [
                str(CLANG), *common_flags, f"-I{kernel_dir}", str(source),
                str(UTILITIES / "polybench.c"), "-lm", "-o", str(executable),
            ]
            compile_rc = run(compile_command, log_dir / "build.log", env=environment)
            if compile_rc == 0:
                with reference.open("wb") as output:
                    run_rc = run([str(executable)], log_dir / "run.log",
                                 env=environment, stderr_output=output)
                (log_dir / "provenance.json").write_text(json.dumps({
                    "source": str(source), "source_sha256": file_hash(source),
                    "executable_sha256": file_hash(executable),
                    "output_sha256": file_hash(reference) if run_rc == 0 else None,
                    "compile_command": compile_command, "run_rc": run_rc,
                }, indent=2, sort_keys=True) + "\n")

        function = f"kernel_{kernel.replace('-', '_')}"
        harness = args.output / kernel / "harness.c"
        harness.parent.mkdir(parents=True, exist_ok=True)
        harness.write_text(generate_harness(source.read_text(), function))
        harnesses[kernel] = harness

    total = len(candidates)
    done = len(complete)
    for item in candidates:
        kernel, variant = item["kernel"], item["variant"]
        if (kernel, variant) in complete:
            continue
        source = sources[kernel]
        extracted_variant = (args.variants / "inputs" /
                             f"{kernel}__{variant}.mlir").resolve()
        log_dir = args.output / kernel / variant
        log_dir.mkdir(parents=True, exist_ok=True)
        variant_input = extracted_variant
        generation_rc = subprocess.run(
            [str(ROOT / "build/bin/polygeist-opt"),
             str(variant_input), "-o", "/dev/null"],
            stdout=subprocess.DEVNULL,
            stderr=(log_dir / "verify.log").open("w"),
            check=False).returncode
        matched = log_dir / "matched.mlir"
        executable = log_dir / "raised-openblas"
        candidate_output = log_dir / "output.txt"
        match_command = [
            "/usr/bin/python3", str(MATCHER), str(variant_input),
            "--matcher-mode", "egglog", "--disable-semantic-fallback",
        ]
        if generation_rc == 0:
            with matched.open("wb") as output:
                match_rc = run(match_command, log_dir / "match.log",
                               env=environment, stdout=output)
        else:
            match_rc = -1
        build_command = [
            str(BUILDER), "--target=host",
            f"--function=kernel_{kernel.replace('-', '_')}",
            f"--harness={harnesses[kernel]}",
            f"--semantic-mlir={matched}", "-o", str(executable),
            str(source), *common_flags, f"-I{source.parent}",
        ]
        build_rc = (run(build_command, log_dir / "build.log", env=environment)
                    if match_rc == 0 else -1)
        run_rc = -1
        compare_rc = -1
        if build_rc == 0:
            with candidate_output.open("wb") as output:
                run_rc = run([str(executable)], log_dir / "run.log",
                             env=environment, stderr_output=output)
        reference = references[kernel]
        if run_rc == 0 and reference.is_file():
            compare_command = [
                "/usr/bin/python3", str(COMPARE), str(reference),
                str(candidate_output), "--rtol", "5e-4", "--atol", "1.1e-2",
            ]
            compare_rc = run(compare_command, log_dir / "compare.log",
                             env=environment)
        status = "pass" if (match_rc, build_rc, run_rc, compare_rc) == (0, 0, 0, 0) else "fail"
        row: dict[str, object] = {
            "kernel": kernel, "variant": variant, "status": status,
            "match_rc": match_rc, "build_rc": build_rc,
            "run_rc": run_rc, "compare_rc": compare_rc,
            "generation_rc": generation_rc,
            "source_path": str(source), "source_sha256": file_hash(source),
            "extracted_variant_path": str(extracted_variant),
            "extracted_variant_sha256": file_hash(extracted_variant),
            "variant_path": str(variant_input),
            "variant_sha256": file_hash(variant_input),
            "matched_sha256": file_hash(matched) if matched.is_file() else "",
            "executable_sha256": file_hash(executable) if executable.is_file() else "",
            "reference_sha256": file_hash(reference) if reference.is_file() else "",
            "candidate_sha256": (file_hash(candidate_output)
                                 if candidate_output.is_file() else ""),
            "match_command": shlex.join(match_command),
            "build_command": shlex.join(build_command),
        }
        append_csv(results_path, row)
        done += 1
        print(f"[{done}/{total}] {kernel}::{variant} {status} "
              f"match={match_rc} build={build_rc} run={run_rc} compare={compare_rc}",
              flush=True)

    results = read_csv(results_path)
    summary = {
        "candidate_rows": len(candidates), "result_rows": len(results),
        "passed": sum(row["status"] == "pass" for row in results),
        "failed": sum(row["status"] != "pass" for row in results),
        "dataset": "LARGE", "dtype": "FP64",
        "backend": "host OpenBLAS/CBLAS, one thread",
        "rtol": 5e-4, "atol": 1.1e-2,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
