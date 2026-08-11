#!/usr/bin/env python3
"""Re-run only ATen semantic matching over already-raised debufferized IR."""

from __future__ import annotations

import csv
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "issues/aten_c_kernels/results"
MATCHER = ROOT / "scripts/correctness/kernel_match_rewrite.py"


def count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, re.MULTILINE))


def main() -> None:
    summary = RESULTS / "summary.tsv"
    with summary.open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    def rematch(row: dict[str, str]) -> dict[str, str]:
        row = dict(row)
        kernel = row["kernel"]
        debuf = RESULTS / kernel / "debuf.mlir"
        raised = RESULTS / kernel / "raised.mlir"
        matched = RESULTS / kernel / "matched.mlir"
        if not debuf.exists() or row["status"] != "pass":
            return row
        proc = subprocess.run(
            ["/usr/bin/python3", str(MATCHER), str(debuf)],
            text=True, capture_output=True, check=False, timeout=30,
        )
        if proc.returncode:
            row["status"] = "match_failed"
            (RESULTS / kernel / "match.err").write_text(proc.stderr)
            return row
        matched.write_text(proc.stdout)
        raised_text = raised.read_text() if raised.exists() else ""
        symbols = sorted(set(re.findall(
            r"kernel\.launch @([A-Za-z0-9_]+)", proc.stdout)))
        row.update({
            "linalg_ops": str(count(r"linalg\.(?:generic|matmul|conv)", raised_text)),
            "residual_loops": str(count(r"\b(?:affine|scf)\.(?:for|parallel|while)\b", raised_text)),
            "kernel_launches": str(count(r"kernel\.launch ", proc.stdout)),
            "matched_symbols": ",".join(symbols) if symbols else "-",
        })
        return row
    with ThreadPoolExecutor(max_workers=16) as pool:
        output = list(pool.map(rematch, rows))
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
