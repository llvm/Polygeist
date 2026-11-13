#!/usr/bin/env python3
"""Audit Polygeist recognition on the 21 programs from Ginsbach ASPLOS'18.

This is deliberately a structural audit, not a performance run.  Every source
translation unit is passed through cgeist, affine-to-linalg raising,
debufferization, and the production library matcher.  Results are aggregated
per program while per-translation-unit diagnostics are retained in OUT/units.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NPB_ROOT = ROOT / "third_party/ginsbach-snu-npb/NPB3.3-SER-C"
NPB_PARAM_ROOT = (ROOT / "third_party/ginsbach-snu-npb-source/std_benchmarks"
                  / "SNU_NPB-1.0.3/NPB3.3-SER-C")
PARBOIL_ROOT = ROOT / "third_party/gpu-parboil"
CGEIST = ROOT / "build/bin/cgeist"
POLYGEIST_OPT = ROOT / "build/bin/polygeist-opt"
MATCHER = ROOT / "scripts/correctness/kernel_match_rewrite.py"
PUBLISHED_MANIFEST = (ROOT / "issues/ginsbach_asplos18"
                      / "published_idiom_manifest.csv")
PYTHON = Path("/usr/bin/python3")


def gcc_builtin_include() -> Path | None:
    """Return the host GCC intrinsic-header directory (which owns omp.h)."""
    try:
        result = subprocess.run(
            ["gcc", "-print-file-name=include"], check=True,
            capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    path = Path(result.stdout.strip())
    return path if path.is_dir() else None

NPB_PROGRAMS = ("BT", "CG", "DC", "EP", "FT", "IS", "LU", "MG", "SP", "UA")
PARBOIL_VARIANTS = {
    "bfs": "base",
    "cutcp": "base",
    "histo": "base",
    "lbm": "cpu",
    "mri-gridding": "base",
    "mri-q": "cpu",
    "sad": "base",
    "sgemm": "base",
    "spmv": "cpu",
    "stencil": "cpu",
    "tpacf": "base",
}


@dataclass(frozen=True)
class Unit:
    suite: str
    program: str
    source: Path
    includes: tuple[Path, ...]
    function: str = "*"
    companions: tuple[Path, ...] = ()


def units() -> list[Unit]:
    result: list[Unit] = []
    gcc_include = gcc_builtin_include()
    for program in NPB_PROGRAMS:
        directory = NPB_ROOT / program
        for source in sorted(directory.glob("*.c")):
            # This unverified local mirror does not check in generated
            # npbparams.h files. Reuse Class-S headers from the separate local
            # source bundle. This is a diagnostic substitution, not yet an
            # exact reconstruction of the paper's compiler input.
            includes = [directory, NPB_PARAM_ROOT / program,
                        NPB_ROOT / "common"]
            # This nominally serial UA source includes omp.h even though it
            # contains no OpenMP calls.  Clang's resource directory does not
            # contain libgomp's header, so mirror the native GCC include path.
            if program == "UA" and gcc_include:
                includes.append(gcc_include)
            companions = ()
            # exact_rhs calls a small polynomial helper inside each grid
            # point. The application build provides both definitions and
            # cgeist can inline it; auditing the caller in isolation leaves an
            # artificial opaque call boundary and C-style while nests.
            if program == "BT" and source.name == "exact_rhs.c":
                companions = (directory / "exact_solution.c",)
            result.append(Unit(
                "snu-npb", program, source,
                tuple(includes), companions=companions))
    for program, variant in PARBOIL_VARIANTS.items():
        directory = PARBOIL_ROOT / "benchmarks" / program / "src" / variant
        for source in sorted((*directory.glob("*.c"), *directory.glob("*.cc"))):
            # The base SGEMM main includes this implementation file directly;
            # it is not a separate translation unit in the benchmark Makefile.
            if program == "sgemm" and source.name == "sgemm_kernel.cc":
                continue
            # mri-q builds computeQ.cc by textual inclusion from main.c; it is
            # not an independent translation unit and intentionally relies on
            # the standard headers included by main.c.
            if program == "mri-q" and source.name == "computeQ.cc":
                continue
            extra_includes: tuple[Path, ...] = ()
            if program == "spmv":
                extra_includes = (
                    PARBOIL_ROOT / "benchmarks/spmv/common_src/convert-dataset",
                )
            function = "*"
            if program == "sgemm" and source.name == "main.cc":
                # main.cc includes the source kernel directly.  Selecting its
                # mangled function keeps the benchmark's actual loop body
                # while excluding unrelated libstdc++ implementation bodies
                # that Linalg debufferization cannot represent.
                function = "_Z10basicSgemmcciiifPKfiS0_ifPfi"
            result.append(Unit(
                "parboil", program, source,
                (directory, PARBOIL_ROOT / "common/include", *extra_includes),
                function,
            ))
    return result


def run(command: list[str], stdout: Path, stderr: Path, timeout: int) -> tuple[int, float]:
    start = time.monotonic()
    with stdout.open("w") as out, stderr.open("w") as err:
        try:
            proc = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, timeout=timeout)
            return proc.returncode, time.monotonic() - start
        except subprocess.TimeoutExpired:
            err.write(f"\nTIMEOUT after {timeout}s\n")
            return 124, time.monotonic() - start


def count(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return len(re.findall(pattern, path.read_text(errors="replace")))


def first_error(path: Path) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    interesting = next((line.strip() for line in lines if "error:" in line.lower()), None)
    return (interesting or next((line.strip() for line in lines if line.strip()), ""))[:300]


def audit_unit(unit: Unit, out_root: Path, timeout: int,
               stencil_backend: str) -> dict[str, object]:
    rel = unit.source.relative_to(ROOT)
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", str(rel))
    directory = out_root / "units" / stem
    directory.mkdir(parents=True, exist_ok=True)
    affine = directory / "affine.mlir"
    linalg = directory / "linalg.mlir"
    matched = directory / "matched.mlir"

    cgeist_cmd = [
        str(CGEIST), str(unit.source), *(str(path) for path in unit.companions),
        f"--function={unit.function}",
        "--resource-dir=/usr/lib/clang/14", "--raise-scf-to-affine",
        "--mlir-print-op-generic", "-fPIC", "-S", "-o", str(affine),
    ]
    for include in unit.includes:
        cgeist_cmd.append(f"-I{include}")
    cgeist_rc, cgeist_s = run(cgeist_cmd, directory / "cgeist.out", directory / "cgeist.err", timeout)

    raise_rc = match_rc = structured_rc = -1
    raise_s = match_s = structured_s = 0.0
    if cgeist_rc == 0 and affine.exists():
        raise_rc, raise_s = run(
            [
                str(POLYGEIST_OPT), "--remove-iter-args", "--affine-parallelize",
                "--raise-affine-to-linalg-pipeline",
                "--linalg-debufferize=use-multi-root=true",
                str(affine), "-o", str(linalg),
            ],
            directory / "raise.out", directory / "raise.err", timeout,
        )
    if raise_rc == 0 and linalg.exists():
        match_rc, match_s = run(
            [str(PYTHON), str(MATCHER), str(linalg),
             "--enable-structured-rewrite",
             f"--stencil-backend={stencil_backend}"],
            matched, directory / "match.err", timeout,
        )
        structured_rc, structured_s = run(
            [str(PYTHON), str(MATCHER), str(linalg), "--dry-run",
             "--show-structured-regions"],
            directory / "structured.out", directory / "structured.err",
            timeout,
        )
    elif cgeist_rc == 0 and affine.exists():
        # Indirect histogram/sparse loops are often precisely the regions that
        # cannot be represented by affine-to-linalg. Preserve their extracted
        # loop form for structured analysis instead of hiding them behind a
        # later raising failure. This path never contributes executable
        # kernel launches.
        structured_rc, structured_s = run(
            [str(PYTHON), str(MATCHER), str(affine), "--dry-run",
             "--show-structured-regions"],
            directory / "structured.out", directory / "structured.err",
            timeout,
        )

    error_file = directory / ("cgeist.err" if cgeist_rc else "raise.err" if raise_rc else "match.err")
    analysis_ir = linalg if linalg.exists() else affine
    return {
        "suite": unit.suite,
        "program": unit.program,
        "source": str(rel),
        "cgeist_rc": cgeist_rc,
        "raise_rc": raise_rc,
        "match_rc": match_rc,
        "affine_loops": count(affine, r"\baffine\.for\b"),
        "residual_loops": count(analysis_ir, r"\b(?:affine|scf)\.(?:for|while)\b"),
        "linalg_generics": count(linalg, r"\blinalg\.generic\b"),
        "kernel_launches": count(matched, r"\bkernel\.launch\b"),
        "structured_fusions": count(
            directory / "structured.err", r"\bstructured_fusion\b"),
        "structured_rejects": count(
            directory / "structured.err", r"\bstructured_reject\b"),
        "structured_reductions": count(
            directory / "structured.err",
            r"\bextracted=(?:scalar|axis)_(?:sum|product|minmax)_reduction\b"),
        "structured_gemms": count(
            directory / "structured.err",
            r"\bextracted=(?:dense_gemm|looped_gemv_as_gemm(?:_schedule)?)\b"),
        "structured_stencils": count(
            directory / "structured.err",
            r"\bextracted=(?:affine_stencil|factorized_linear_stencil3d)\b"),
        "histogram_candidates": count(
            directory / "structured.err",
            r"\bkind=indirect_histogram\b"),
        "csr_spmv_candidates": count(
            directory / "structured.err", r"\bkind=(?:csr|jds)_spmv\b"),
        "structured_rc": structured_rc,
        "seconds": round(cgeist_s + raise_s + match_s + structured_s, 3),
        "error": first_error(error_file) if (cgeist_rc or raise_rc or match_rc) else "",
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/ginsbach_asplos18_audit"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--program", action="append",
                        help="Audit only this benchmark name (repeatable).")
    parser.add_argument(
        "--stencil-backend", choices=("cudnn", "custen"), default="custen",
        help="External stencil library used by the executable matcher.")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    work = [unit for unit in units()
            if not args.program or unit.program in args.program]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        rows = list(executor.map(
            lambda unit: audit_unit(
                unit, args.out, args.timeout, args.stencil_backend), work))
    write_csv(args.out / "translation_units.csv", rows)

    summary: list[dict[str, object]] = []
    for suite, program in [("snu-npb", x) for x in NPB_PROGRAMS] + [
        ("parboil", x) for x in PARBOIL_VARIANTS
    ]:
        selected = [row for row in rows if row["suite"] == suite and row["program"] == program]
        summary.append({
            "suite": suite,
            "program": program,
            "units": len(selected),
            "frontend_ok": sum(row["cgeist_rc"] == 0 for row in selected),
            "raise_ok": sum(row["raise_rc"] == 0 for row in selected),
            "linalg_generics": sum(int(row["linalg_generics"]) for row in selected),
            "kernel_launches": sum(int(row["kernel_launches"]) for row in selected),
            "structured_fusions": sum(int(row["structured_fusions"]) for row in selected),
            "structured_rejects": sum(int(row["structured_rejects"]) for row in selected),
            "structured_reductions": sum(
                int(row["structured_reductions"]) for row in selected),
            "structured_gemms": sum(
                int(row["structured_gemms"]) for row in selected),
            "structured_stencils": sum(
                int(row["structured_stencils"]) for row in selected),
            "histogram_candidates": sum(
                int(row["histogram_candidates"]) for row in selected),
            "csr_spmv_candidates": sum(
                int(row["csr_spmv_candidates"]) for row in selected),
            "residual_loops": sum(int(row["residual_loops"]) for row in selected),
        })
    with PUBLISHED_MANIFEST.open(newline="") as handle:
        published = list(csv.DictReader(handle))
    expected_categories = {
        "scalar_reduction": 45,
        "histogram_reduction": 5,
        "stencil": 6,
        "matrix_operation": 1,
        "sparse_matrix_operation": 3,
    }
    actual_categories = {
        category: sum(row["category"] == category for row in published)
        for category in expected_categories
    }
    if len(published) != 60 or actual_categories != expected_categories:
        raise RuntimeError(
            "published idiom manifest no longer agrees with ASPLOS'18 "
            f"Table 1: rows={len(published)}, categories={actual_categories}")

    published_by_program = {
        program: sum(row["program"] == program for row in published)
        for program in (*NPB_PROGRAMS, *PARBOIL_VARIANTS)
    }
    for row in summary:
        row["published_idioms"] = published_by_program[row["program"]]
    write_csv(args.out / "program_summary.csv", summary)

    print("suite,program,units,frontend_ok,raise_ok,linalg_generics,"
          "kernel_launches,structured_fusions,structured_rejects,"
          "structured_reductions,structured_gemms,structured_stencils,"
          "histogram_candidates,csr_spmv_candidates,residual_loops,"
          "published_idioms")
    for row in summary:
        print(",".join(str(value) for value in row.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
