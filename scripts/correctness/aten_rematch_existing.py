#!/usr/bin/env python3
"""Re-run only ATen semantic matching over already-raised debufferized IR."""

from __future__ import annotations

import csv
import os
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
    # A targeted aten_c_kernel_sweep.sh invocation intentionally rewrites its
    # summary with only the requested kernels.  The per-kernel directories are
    # authoritative, so reconstruct the index before a corpus-wide rematch
    # instead of silently publishing a truncated audit/CE table.
    artifact_kernels = sorted(
        path.parent.name for path in RESULTS.glob("aten_*/debuf.mlir"))
    indexed = {row["kernel"] for row in rows}
    if len(indexed) < len(artifact_kernels):
        prior = {row["kernel"]: row for row in rows}
        rows = [prior.get(kernel, {
            "kernel": kernel,
            "status": "pass",
            "linalg_ops": "0",
            "residual_loops": "0",
            "kernel_launches": "0",
            "matched_symbols": "-",
        }) for kernel in artifact_kernels]
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
    # A matcher process loads the full Egglog rule set and can consume enough
    # memory that a 16-way sweep is killed by the host OOM controller.  Keep
    # this configurable, but use a conservative default so a rematch cannot
    # leave summary.tsv truncated after a targeted sweep.
    workers = int(os.environ.get("ATEN_REMATCH_WORKERS", "4"))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        output = list(pool.map(rematch, rows))
    with summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
