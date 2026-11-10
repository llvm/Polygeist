#!/usr/bin/env python3
"""Run a Ginsbach-style Clang matcher over retained sequential sources.

This runner preserves every command, LLVM module, report, and diagnostic. It
does not map results to the paper's 60 IDs; that requires separate evidence.
By default sources are passed on stdin, matching IDL-Demo and ensuring the
modified compiler writes its fixed `replace-report--.json` in the unit output
directory rather than trying to embed an absolute source path in a filename.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NPB = ROOT / "third_party/ginsbach-snu-npb/NPB3.3-SER-C"
NPB_PARAMS = (ROOT / "third_party/ginsbach-snu-npb-source/std_benchmarks"
              / "SNU_NPB-1.0.3/NPB3.3-SER-C")
PARBOIL = ROOT / "third_party/gpu-parboil"
POLYBENCH = ROOT / "tools/cgeist/Test/polybench"
NPB_PROGRAMS = ("BT", "CG", "DC", "EP", "FT", "IS", "LU", "MG", "SP", "UA")
PARBOIL_VARIANTS = {
    "bfs": "base", "cutcp": "base", "histo": "base", "lbm": "cpu",
    "mri-gridding": "base", "mri-q": "cpu", "sad": "base",
    "sgemm": "base", "spmv": "cpu", "stencil": "cpu", "tpacf": "base",
}


@dataclass(frozen=True)
class Unit:
    suite: str
    program: str
    source: Path
    includes: tuple[Path, ...]


def ginsbach_corpus() -> list[Unit]:
    units: list[Unit] = []
    for program in NPB_PROGRAMS:
        directory = NPB / program
        for source in sorted(directory.glob("*.c")):
            units.append(Unit("snu-npb", program, source,
                              (directory, NPB_PARAMS / program, NPB / "common")))
    for program, variant in PARBOIL_VARIANTS.items():
        directory = PARBOIL / "benchmarks" / program / "src" / variant
        sources = sorted((*directory.glob("*.c"), *directory.glob("*.cc")))
        for source in sources:
            if program == "sgemm" and source.name == "sgemm_kernel.cc":
                continue  # Included by main.cc, not a standalone benchmark TU.
            extra: tuple[Path, ...] = ()
            if program == "spmv":
                extra = (PARBOIL / "benchmarks/spmv/common_src/convert-dataset",)
            units.append(Unit("parboil", program, source,
                              (source.parent, PARBOIL / "common/include", *extra)))
    return units


def polybench_corpus() -> list[Unit]:
    """Return the canonical PolyBench/C kernel translation units."""
    units: list[Unit] = []
    utilities = POLYBENCH / "utilities"
    for source in sorted(POLYBENCH.rglob("*.c")):
        if utilities in source.parents or source.name.endswith(".orig.c"):
            continue
        units.append(Unit("polybench", source.stem, source,
                          (source.parent, utilities)))
    return units


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clang", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--program", action="append")
    parser.add_argument("--corpus", choices=("ginsbach", "polybench"),
                        default="ginsbach")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--optimization-profile", choices=("demo-o2", "thesis-os"),
        default="demo-o2",
        help=("demo-o2 follows the public IDL demo; thesis-os uses the "
              "normalization flags documented by the author's thesis"))
    args = parser.parse_args()
    args.clang = args.clang.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    units = polybench_corpus() if args.corpus == "polybench" else ginsbach_corpus()
    for index, unit in enumerate(units, 1):
        if args.program and unit.program not in args.program:
            continue
        rel = unit.source.relative_to(ROOT)
        unit_out = args.out / f"{index:03d}-{unit.suite}-{unit.program}-{unit.source.name}"
        unit_out.mkdir(parents=True, exist_ok=True)
        is_cxx = unit.source.suffix == ".cc"
        optimization_flags = (["-O2"] if args.optimization_profile == "demo-o2"
                              else ["-Os", "-fno-unroll-loops", "-fno-vectorize",
                                    "-fno-slp-vectorize", "-ffast-math"])
        command = [str(args.clang), "-std=c++17" if is_cxx else "-std=c99",
                   "-S", "-emit-llvm", *optimization_flags,
                   "-gline-tables-only",
                   "-x", "c++" if is_cxx else "c", "-", "-o", "module.ll"]
        command.extend(f"-I{x}" for x in unit.includes)
        (unit_out / "command.txt").write_text(" ".join(command) + "\n")
        proc = subprocess.Popen(command, cwd=unit_out, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                start_new_session=True)
        try:
            stdout, stderr = proc.communicate(input=unit.source.read_bytes(),
                                              timeout=args.timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            # clang spawns clang -cc1.  Kill the process group so a timed-out
            # solver cannot remain orphaned and consume a core indefinitely.
            os.killpg(proc.pid, signal.SIGKILL)
            stdout, stderr = proc.communicate()
            rc = 124
            stderr += b"\nTIMEOUT\n"
        (unit_out / "stdout.log").write_bytes(stdout)
        (unit_out / "stderr.log").write_bytes(stderr)

        report = unit_out / "replace-report--.json"
        counts: dict[str, int] = {}
        functions: list[str] = []
        lines: list[str] = []
        if report.exists():
            try:
                data = json.loads(report.read_text())
                for hit in data.get("loops", []):
                    functions.append(str(hit.get("function", "")))
                    lines.append(str(hit.get("line", "")))
                    for idiom in hit.get("idioms", []):
                        name = str(idiom.get("type", "unknown"))
                        counts[name] = counts.get(name, 0) + 1
            except (json.JSONDecodeError, OSError) as exc:
                counts["report_parse_error"] = 1
                functions.append(str(exc))
        rows.append({
            "suite": unit.suite,
            "program": unit.program,
            "source": str(rel),
            "compile_rc": rc,
            "report_present": report.exists(),
            "total_loop_solutions": sum(counts.values()),
            "reduction": counts.get("Reduction", 0),
            "histogram": counts.get("Histo", 0),
            "stencil": counts.get("Stencil", 0),
            "gemm": counts.get("GEMM", 0),
            "spmv": counts.get("SPMV", 0),
            "functions": ";".join(functions),
            "source_lines": ";".join(lines),
        })

    with (args.out / "translation_units.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return 0 if all(row["compile_rc"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
