#!/usr/bin/env python3

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path


KERNELS = [
    "datamining/correlation/correlation.c",
    "datamining/covariance/covariance.c",
    "linear-algebra/blas/gemm/gemm.c",
    "linear-algebra/blas/gemver/gemver.c",
    "linear-algebra/blas/gesummv/gesummv.c",
    "linear-algebra/blas/symm/symm.c",
    "linear-algebra/blas/syr2k/syr2k.c",
    "linear-algebra/blas/syrk/syrk.c",
    "linear-algebra/blas/trmm/trmm.c",
    "linear-algebra/kernels/2mm/2mm.c",
    "linear-algebra/kernels/3mm/3mm.c",
    "linear-algebra/kernels/atax/atax.c",
    "linear-algebra/kernels/bicg/bicg.c",
    "linear-algebra/kernels/doitgen/doitgen.c",
    "linear-algebra/kernels/mvt/mvt.c",
    "linear-algebra/solvers/cholesky/cholesky.c",
    "linear-algebra/solvers/durbin/durbin.c",
    "linear-algebra/solvers/gramschmidt/gramschmidt.c",
    "linear-algebra/solvers/lu/lu.c",
    "linear-algebra/solvers/ludcmp/ludcmp.c",
    "linear-algebra/solvers/trisolv/trisolv.c",
    "medley/deriche/deriche.c",
    "medley/floyd-warshall/floyd-warshall.c",
    "medley/nussinov/nussinov.c",
    "stencils/adi/adi.c",
    "stencils/fdtd-2d/fdtd-2d.c",
    "stencils/heat-3d/heat-3d.c",
    "stencils/jacobi-1d/jacobi-1d.c",
    "stencils/jacobi-2d/jacobi-2d.c",
    "stencils/seidel-2d/seidel-2d.c",
]


def run(command, cwd, timeout=120):
    try:
        completed = subprocess.run(
            command, cwd=cwd, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=timeout
        )
        return completed.returncode, completed.stdout, False
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return 124, output, True


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: run_polybench_replacement_sweep.py POLYBENCH CLANG OUT")
    polybench = Path(sys.argv[1]).resolve()
    clang = Path(sys.argv[2]).resolve()
    output = Path(sys.argv[3]).resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    rows = []
    for relative in KERNELS:
        source = polybench / relative
        name = source.stem
        kernel_dir = output / name
        kernel_dir.mkdir()
        command = [
            "timeout", "120s", str(clang), "-O2", "-fno-vectorize",
            "-fno-slp-vectorize", "-fno-unroll-loops", "-DLARGE_DATASET",
            "-DPOLYBENCH_TIME", "-I", str(polybench / "utilities"),
            "-I", str(source.parent), "-c", str(source), "-o",
            str(kernel_dir / f"{name}.o"),
        ]
        returncode, log, timed_out = run(command, kernel_dir, timeout=135)
        (kernel_dir / "compile.log").write_text(log)
        (kernel_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")

        reports = sorted(kernel_dir.glob("replace-report-*.json"))
        detections = []
        report_error = ""
        for report in reports:
            try:
                detections.extend(json.loads(report.read_text()).get("detected", []))
            except Exception as error:
                report_error = str(error)
        types = [entry.get("type", "unknown") for entry in detections]
        emitted = []
        obj = kernel_dir / f"{name}.o"
        if obj.exists():
            nm = subprocess.run(
                ["nm", "-u", str(obj)], text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            ).stdout
            (kernel_dir / "undefined-symbols.txt").write_text(nm)
            emitted = sorted({
                line.split()[-1] for line in nm.splitlines()
                if line.split() and line.split()[-1].endswith("_harness")
            })
        rows.append({
            "kernel": name,
            "source": relative,
            "compile_status": "timeout" if timed_out else
                              ("pass" if returncode == 0 else "fail"),
            "returncode": returncode,
            "detection_count": len(detections),
            "detected_types": ";".join(types),
            "emitted_harness_symbols": ";".join(emitted),
            "replacement_emitted": "yes" if emitted else "no",
            "report_parse_error": report_error,
        })
        print(json.dumps(rows[-1]), flush=True)

    fields = list(rows[0])
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")

    compile_pass = sum(row["compile_status"] == "pass" for row in rows)
    detected = sum(row["detection_count"] for row in rows)
    kernels_detected = sum(row["detection_count"] > 0 for row in rows)
    replaced = sum(row["replacement_emitted"] == "yes" for row in rows)
    totals = {
        "kernels": len(rows),
        "compile_pass": compile_pass,
        "compile_fail_or_timeout": len(rows) - compile_pass,
        "detections": detected,
        "kernels_with_detection": kernels_detected,
        "kernels_with_emitted_replacement": replaced,
    }
    (output / "totals.json").write_text(json.dumps(totals, indent=2) + "\n")
    print(json.dumps(totals), flush=True)


if __name__ == "__main__":
    main()
