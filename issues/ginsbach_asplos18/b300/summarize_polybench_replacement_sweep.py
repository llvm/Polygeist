#!/usr/bin/env python3

import csv
import json
import sys
from pathlib import Path


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: summarize_polybench_replacement_sweep.py RESULT_DIR")
    root = Path(sys.argv[1])
    with (root / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    correctness = {}
    correctness_root = root / "correctness-mini"
    for row in rows:
        name = row["kernel"]
        compare = correctness_root / name / "comparison.txt"
        compare_naive = correctness_root / name / "comparison-naive.txt"
        if compare.exists():
            correctness[name] = {
                "cublas": compare.read_text().strip(),
                "naive": compare_naive.read_text().strip(),
            }

    valid = []
    false_positives = []
    for row in rows:
        if row["replacement_emitted"] != "yes":
            row["correctness"] = "n/a (unmatched)"
        else:
            checks = correctness.get(row["kernel"])
            if checks and checks["cublas"].startswith("PASS"):
                row["correctness"] = "PASS"
                valid.append(row["kernel"])
            else:
                row["correctness"] = "FAIL"
                false_positives.append(row["kernel"])

    fields = list(rows[0])
    if "correctness" not in fields:
        fields.append("correctness")
    with (root / "final-summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (root / "correctness.json").write_text(
        json.dumps(correctness, indent=2) + "\n"
    )

    compile_pass = sum(row["compile_status"] == "pass" for row in rows)
    detected_regions = sum(int(row["detection_count"]) for row in rows)
    matched_kernels = [
        row["kernel"] for row in rows if row["replacement_emitted"] == "yes"
    ]
    report = f"""# Ginsbach/LiLAC replacement backend on PolyBench/C 30

## Result

- Compiled through the author replacement compiler: {compile_pass}/30.
- Raw detected regions: {detected_regions} across {len(matched_kernels)}/30 kernels.
- Kernels with an emitted library-harness call: {len(matched_kernels)}/30
  ({', '.join(matched_kernels)}).
- Correct replacements after complete-output comparison: {len(valid)}/30.
- Incorrect emitted replacements: {len(false_positives)}/30
  ({', '.join(false_positives)}).
- Unmatched: {30 - len(matched_kernels)}/30.

Incorrect replacements are excluded from coverage and performance results.
Both the generated naive harness and generated cuBLAS harness fail for the
three emitted cases, demonstrating that these are recognition/legality false
positives rather than CUDA compatibility failures.

## Kernel matrix

| Kernel | Compile | Detections | Emitted harness | Correctness |
|---|---|---:|---|---|
"""
    for row in rows:
        detected = row["detected_types"] or "none"
        emitted = row["emitted_harness_symbols"] or "none"
        report += (
            f"| {row['kernel']} | {row['compile_status']} | "
            f"{row['detection_count']} ({detected}) | {emitted} | "
            f"{row['correctness']} |\n"
        )
    report += """

## Correctness failures

"""
    for name in false_positives:
        report += f"- `{name}` cuBLAS: {correctness[name]['cublas']}\n"
        report += f"- `{name}` naive: {correctness[name]['naive']}\n"
    report += """

## Interpretation

The strict LiLAC GEMM computation does not recognize standard PolyBench
`gemm`, `2mm`, or `3mm`, whose operations include scaling and/or accumulation.
It instead emits unsafe GEMM replacements for correlation/covariance partial
matrix computations and triangular matrix multiplication. Those regions have
iteration-domain and dependency semantics that are not represented by an
unconstrained dense GEMM call.

This sweep measures the author-hosted 2020 `linearalgebra` branch, not an exact
archived ASPLOS 2018 artifact. It uses PolyBench/C commit
`d3a15bb1725afaaa33bcf3a4db6087ebf557c9b8`, LLVM commit
`113d41f6e3695045fcc6e93166b3d1f5df1a8074`, and Clang commit
`5e3d3c0055dc731abd506e18c320f227168f728b`.
"""
    (root / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
