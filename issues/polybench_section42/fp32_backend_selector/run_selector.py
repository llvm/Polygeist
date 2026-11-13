#!/usr/bin/env python3
"""Exercise the contraction-backend selector on canonical FP32 PolyBench sizes.

This is a policy test, not a claim that every listed source currently matches.
Each launch is marked with neutral semantic contraction metadata and dimensions
from the unmodified PolyBench dataset headers.  The separate raising_audit.csv
records which source kernels actually reach kernel.launch today.
"""

from __future__ import annotations

import csv
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OPT = ROOT / "build/bin/polygeist-opt"

DATASETS = ("mini", "small", "medium", "large", "extralarge")

GEMM = {
    "mini": (20, 25, 30),
    "small": (60, 70, 80),
    "medium": (200, 220, 240),
    "large": (1000, 1100, 1200),
    "extralarge": (2000, 2300, 2600),
}
MM2 = {
    "mini": (16, 18, 22, 24),
    "small": (40, 50, 70, 80),
    "medium": (180, 190, 210, 220),
    "large": (800, 900, 1100, 1200),
    "extralarge": (1600, 1800, 2200, 2400),
}
MM3 = {
    "mini": (16, 18, 20, 22, 24),
    "small": (40, 50, 60, 70, 80),
    "medium": (180, 190, 200, 210, 220),
    "large": (800, 900, 1000, 1100, 1200),
    "extralarge": (1600, 1800, 2000, 2200, 2400),
}
SYRK = {
    "mini": (30, 20),
    "small": (80, 60),
    "medium": (240, 200),
    "large": (1200, 1000),
    "extralarge": (2600, 2000),
}


def cases():
    for dataset in DATASETS:
        ni, nj, nk = GEMM[dataset]
        yield "gemm", dataset, "C=A*B", ni, nj, nk

        ni, nj, nk, nl = MM2[dataset]
        yield "2mm", dataset, "tmp=A*B", ni, nj, nk
        yield "2mm", dataset, "D=tmp*C", ni, nl, nj

        ni, nj, nk, nl, nm = MM3[dataset]
        yield "3mm", dataset, "E=A*B", ni, nj, nk
        yield "3mm", dataset, "F=C*D", nj, nl, nm
        yield "3mm", dataset, "G=E*F", ni, nl, nj

        n, m = SYRK[dataset]
        yield "syrk", dataset, "C=A*A^T", n, n, m
        yield "syr2k", dataset, "C=A*B^T", n, n, m
        yield "syr2k", dataset, "C=B*A^T", n, n, m


def symbol(program: str, dataset: str, occurrence: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]+", "_", occurrence).strip("_")
    return f"pb_{program}_{dataset}_{clean}"


def generate_mlir(rows):
    text = ["module {"]
    for program, dataset, occurrence, m, n, k in rows:
        sym = symbol(program, dataset, occurrence)
        kind = "syr2k" if program == "syr2k" else "syrk" if program == "syrk" else "gemm"
        a = f"tensor<{m}x{k}xf32>"
        b = f"tensor<{k}x{n}xf32>"
        c = f"tensor<{m}x{n}xf32>"
        text += [
            f"  kernel.defn @{sym}(%a: {a}, %b: {b}, %c: {c}) -> {c} {{",
            f"    kernel.yield %c : {c}",
            "  }",
            f"  func.func @{sym}_case(%a: {a}, %b: {b}, %c: {c}) -> {c} {{",
            f"    %0 = kernel.launch @{sym}(%a, %b, %c) "
            f'{{polygeist.contraction.kind = "{kind}", '
            f"polygeist.contraction.m = {m} : i64, "
            f"polygeist.contraction.n = {n} : i64, "
            f"polygeist.contraction.k = {k} : i64}} "
            f": ({a}, {b}, {c}) -> {c}",
            f"    return %0 : {c}",
            "  }",
        ]
    text.append("}")
    return "\n".join(text) + "\n"


def main():
    rows = list(cases())
    input_path = OUT / "canonical_fp32_contractions.mlir"
    selected_path = OUT / "canonical_fp32_selected.mlir"
    csv_path = OUT / "selector_results.csv"
    input_path.write_text(generate_mlir(rows))
    subprocess.run(
        [
            str(OPT),
            str(input_path),
            "--select-contraction-backend=allow-tf32=true target-arch=sm_87",
            "-o",
            str(selected_path),
        ],
        cwd=ROOT,
        check=True,
    )
    selected = selected_path.read_text()
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["program", "dataset", "occurrence", "m", "n", "k", "mnk", "backend", "reason"]
        )
        for program, dataset, occurrence, m, n, k in rows:
            sym = symbol(program, dataset, occurrence)
            line = next(
                line for line in selected.splitlines() if f"kernel.launch @{sym}(" in line
            )
            backend = re.search(r'polygeist\.contraction\.backend = "([^"]+)"', line)
            reason = re.search(r'polygeist\.contraction\.selection_reason = "([^"]+)"', line)
            assert backend and reason
            writer.writerow(
                [program, dataset, occurrence, m, n, k, m * n * k, backend.group(1), reason.group(1)]
            )
    print(csv_path)


if __name__ == "__main__":
    main()
