#!/usr/bin/env python3
"""Account for every translation unit in the pinned portable ATen census.

This is deliberately a source-file inventory, not an operator count.  One
translation unit may contain no numerical body, one loop kernel, or dozens of
TensorIterator scalar lambdas.  Existing standalone-C fixtures are linked by
the provenance table in build_ce_viewer.py.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = (
    ROOT / "notes/polygeist_raise_to_linalg/aten_raise_sweep_2026_07_21/sources.txt"
)
DEFAULT_PYTORCH = ROOT / "third_party/pytorch"
DEFAULT_OUTPUT = ROOT / "issues/aten_c_kernels/extraction_inventory.csv"
OPERATOR_ADJUDICATION = ROOT / "issues/aten_c_kernels/operator_adjudication.csv"
GENERATED_PROVENANCE_GLOB = "generated*_provenance.csv"
SOURCE_ACCOUNTING = {
    "aten/src/ATen/native/AdaptiveMaxPooling2d.cpp":
        "loop validates non-batch output dimensions; arithmetic dispatches to a registered kernel",
    "aten/src/ATen/native/UpSampleLanczos2d.cpp":
        "loop validates scale factors; resampling arithmetic is in extracted dispatch kernels",
    "aten/src/ATen/native/UpSampleNearest3d.cpp":
        "loops validate five-dimensional shape/scale metadata; arithmetic is dispatched",
    "aten/src/ATen/native/UpSampleTrilinear3d.cpp":
        "loop validates five-dimensional scale metadata; arithmetic is dispatched",
}


def read_provenance() -> dict[str, list[str]]:
    viewer = (ROOT / "scripts/correctness/build_ce_viewer.py").read_text()
    begin = viewer.index("ATEN_C_PROVENANCE:")
    end = viewer.index("ATEN_C_MATCH_ASSESSMENT:", begin)
    block = viewer[begin:end]
    by_source: dict[str, list[str]] = defaultdict(list)
    pattern = re.compile(
        r'"(aten_[^"]+)"\s*:\s*\("(aten/src/ATen/native/[^"]+)"'
    )
    for fixture, source in pattern.findall(block):
        by_source[source].append(fixture)
    for manifest in sorted(
        (ROOT / "issues/aten_c_kernels").glob(GENERATED_PROVENANCE_GLOB)
    ):
        with manifest.open(newline="") as stream:
            for row in csv.DictReader(stream):
                by_source[row["source"]].append(row["kernel"])
    return by_source


def classify(text: str, fixtures: list[str]) -> tuple[str, str, dict[str, int]]:
    metrics = {
        "textual_loops": len(re.findall(r"\b(?:for|while)\s*\(", text)),
        "cpu_kernel_sites": len(
            re.findall(r"\b(?:cpu_kernel(?:_vec)?|cpu_serial_kernel)\s*\(", text)
        ),
        "dispatch_sites": len(
            re.findall(r"\b(?:AT_DISPATCH\w*|REGISTER_DISPATCH|TORCH_IMPL_FUNC)\b", text)
        ),
        "tensor_iterator_mentions": len(re.findall(r"\bTensorIterator\w*\b", text)),
    }
    if fixtures:
        return "HAS_EXTRACTION", "one or more standalone-C fixtures exist", metrics
    if metrics["cpu_kernel_sites"]:
        return (
            "EXTRACT_TENSORITERATOR",
            "scalar lambda(s) are hidden behind TensorIterator/cpu_kernel",
            metrics,
        )
    if metrics["textual_loops"]:
        return (
            "EXTRACT_LOOP_BODY",
            "contains explicit loop(s) requiring framework/type specialization",
            metrics,
        )
    if metrics["dispatch_sites"] or metrics["tensor_iterator_mentions"]:
        return (
            "DISPATCH_ONLY",
            "dispatch/registration wrapper with no local scalar loop body",
            metrics,
        )
    return "NO_LOCAL_NUMERICAL_BODY", "no local loop or TensorIterator kernel body", metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--pytorch", type=Path, default=DEFAULT_PYTORCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    provenance = read_provenance()
    adjudicated: dict[str, list[str]] = defaultdict(list)
    if OPERATOR_ADJUDICATION.exists():
        with OPERATOR_ADJUDICATION.open(newline="") as stream:
            for row in csv.DictReader(stream):
                adjudicated[row["source"]].append(row["final_status"])
    rows = []
    for source in args.sources.read_text().splitlines():
        source = source.strip()
        if not source:
            continue
        path = args.pytorch / source
        text = path.read_text(errors="replace")
        fixtures = sorted(provenance.get(source, []))
        classification, reason, metrics = classify(text, fixtures)
        if source in SOURCE_ACCOUNTING and classification.startswith("EXTRACT_"):
            classification = "ACCOUNTED_NON_STANDALONE"
            reason = SOURCE_ACCOUNTING[source]
        source_statuses = adjudicated.get(source, [])
        if (classification.startswith("EXTRACT_") and source_statuses
                and "NEEDS_PORT" not in source_statuses):
            classification = "ACCOUNTED_NON_STANDALONE"
            reason = (
                "all named iterative bodies are proven helper/composite, "
                "external delegation, non-numerical plumbing, or parser artifact"
            )
        rows.append(
            {
                "source": source,
                "classification": classification,
                "existing_fixtures": ",".join(fixtures),
                **metrics,
                "lines": len(text.splitlines()),
                "reason": reason,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "source", "classification", "existing_fixtures", "textual_loops",
        "cpu_kernel_sites", "dispatch_sites", "tensor_iterator_mentions",
        "lines", "reason",
    ]
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["classification"]] += 1
    print(f"wrote {len(rows)} translation units to {args.output}")
    for name in sorted(counts):
        print(f"{name}: {counts[name]}")


if __name__ == "__main__":
    main()
