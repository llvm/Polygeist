#!/usr/bin/env python3
"""Compare labelled reconstructed solver output with the paper transcription.

The output is diagnostic only.  In particular, equal per-program totals do not
establish occurrence identity or reproduce the unavailable ASPLOS tool.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


AUDIT = Path(__file__).resolve().parent.parent
MANIFEST = AUDIT / "ginsbach_60_manifest.csv"
RUN = AUDIT / "runs/parsed-idl-reconstruction-corpus/translation_units.csv"
OUT = AUDIT / "runs/parsed-idl-reconstruction-corpus/reconstruction_vs_paper.csv"
CATEGORIES = ("scalar", "histogram", "stencil", "gemm", "spmv")


def paper_category(value: str) -> str:
    return {
        "scalar_reduction": "scalar",
        "histogram_reduction": "histogram",
        "stencil": "stencil",
        "matrix_operation": "gemm",
        "sparse_matrix_operation": "spmv",
    }[value]


def main() -> None:
    paper: Counter[tuple[str, str, str]] = Counter()
    with MANIFEST.open(newline="") as handle:
        for row in csv.DictReader(handle):
            paper[(row["benchmark_suite"], row["program"],
                   paper_category(row["idiom_category"]))] += 1

    reconstructed: Counter[tuple[str, str, str]] = Counter()
    compilation: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    with RUN.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["suite"], row["program"])
            compilation[key][row["compile_rc"]] += 1
            for category, column in zip(
                    CATEGORIES,
                    ("reduction", "histogram", "stencil", "gemm", "spmv")):
                reconstructed[(*key, category)] += int(row[column])

    programs = sorted({key[:2] for key in paper} | set(compilation))
    fieldnames = ["suite", "program", "translation_units", "compiled_ok",
                  "timed_out"]
    for category in CATEGORIES:
        fieldnames += [f"paper_{category}", f"reconstructed_{category}"]
    fieldnames += ["paper_total", "reconstructed_total", "classification"]

    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for suite, program in programs:
            counts = compilation[(suite, program)]
            row: dict[str, object] = {
                "suite": suite,
                "program": program,
                "translation_units": sum(counts.values()),
                "compiled_ok": counts["0"],
                "timed_out": counts["124"],
                "classification": (
                    "experimental retained-spec reconstruction; not original tool"
                ),
            }
            for category in CATEGORIES:
                row[f"paper_{category}"] = paper[(suite, program, category)]
                row[f"reconstructed_{category}"] = reconstructed[
                    (suite, program, category)]
            row["paper_total"] = sum(
                int(row[f"paper_{category}"]) for category in CATEGORIES)
            row["reconstructed_total"] = sum(
                int(row[f"reconstructed_{category}"])
                for category in CATEGORIES)
            writer.writerow(row)


if __name__ == "__main__":
    main()
