#!/usr/bin/env python3
"""Fail unless the pinned ATen extraction/raising census is fully accounted."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "issues/aten_c_kernels"


def rows(name: str, delimiter: str = ",") -> list[dict[str, str]]:
    with (CORPUS / name).open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def main() -> None:
    sources = rows("extraction_inventory.csv")
    operators = rows("operator_adjudication.csv")
    dispatch = rows("dispatch_kernel_inventory.csv")
    sweep = rows("results/summary.tsv", "\t")
    cuda_audit = rows("cuda_library_audit.csv")
    fixtures = sorted(path.stem for path in CORPUS.glob("aten_*.c"))

    assert len(sources) == 224, f"expected 224 source files, got {len(sources)}"
    actionable_sources = [
        row for row in sources if row["classification"].startswith("EXTRACT_")
    ]
    assert not actionable_sources, f"unaccounted source files: {actionable_sources}"
    needs_port = [row for row in operators if row["final_status"] == "NEEDS_PORT"]
    assert not needs_port, f"unported operator bodies: {needs_port}"
    remaining_dispatch = [row for row in dispatch if row["status"] == "PENDING"]
    assert not remaining_dispatch, f"unported dispatch kernels: {remaining_dispatch}"
    assert len(sweep) == len(fixtures), (
        f"sweep has {len(sweep)} rows for {len(fixtures)} fixtures"
    )
    assert {row["kernel"] for row in sweep} == set(fixtures), (
        "sweep and standalone-C fixture names differ"
    )
    assert len(cuda_audit) == len(fixtures), (
        f"CUDA-library audit has {len(cuda_audit)} rows for {len(fixtures)} fixtures"
    )
    assert {row["kernel"] for row in cuda_audit} == set(fixtures), (
        "CUDA-library audit and standalone-C fixture names differ"
    )
    assert all(row["rationale"] and row["compiler_gap"] for row in cuda_audit), (
        "CUDA-library audit contains an unadjudicated row"
    )
    early_failures = [
        row for row in sweep
        if row["status"] in {"frontend_failed", "raise_failed", "match_failed"}
    ]
    assert not early_failures, f"frontend/raise/matcher failures: {early_failures}"

    print(
        f"complete: {len(sources)} sources, {len(operators)} named bodies, "
        f"{len(dispatch)} dispatch registrations, {len(fixtures)} C fixtures"
    )


if __name__ == "__main__":
    main()
