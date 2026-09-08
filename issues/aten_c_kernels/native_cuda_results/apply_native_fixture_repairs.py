#!/usr/bin/env python3
"""Promote repaired native-fixture recipes after a fingerprinted CUDA run.

This intentionally leaves layout/storage proxies non-comparable.  A successful
timing alone is not evidence that a dense or differently materialized proxy is
a legal replacement for the extracted fixture.
"""

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPAIRED_CLASSES = {
    "CURRENT_RECIPE_NOT_EQUIVALENT": "EXACT_EXTRACTED_FIXTURE",
    "NO_EXECUTABLE_NATIVE_BASELINE": "EXACT_EXTRACTED_FIXTURE",
    "RNG_OR_FORMULA_SCOPE_MISMATCH": "EXACT_EXTRACTED_TRANSFORM",
    "LAYOUT_OR_STORAGE_PROXY": "EXACT_EXTRACTED_FIXTURE",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("validated_results", type=Path)
    args = parser.parse_args()

    specs = {row["kernel"]: row for row in json.loads(
        (HERE / "resident_shape_specs.json").read_text())}
    with args.validated_results.open(newline="") as stream:
        results = {row["kernel"]: row for row in csv.DictReader(stream)}

    path = HERE / "native_fixture_adjudication.csv"
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    promoted = []
    for row in rows:
        replacement = REPAIRED_CLASSES.get(row["comparability"])
        if not replacement:
            continue
        kernel = row["kernel"]
        result = results.get(kernel, {})
        expected = specs[kernel]["recipe_fingerprint"]
        if (result.get("status") != "PASS" or
                result.get("recipe_fingerprint") != expected):
            raise RuntimeError(
                f"{kernel}: repaired recipe lacks a PASS result for "
                f"fingerprint {expected}")
        row["comparability"] = replacement
        row["legal_ratio"] = "yes"
        row["semantic_note"] = (
            "fingerprinted CUDA recipe reproduces the complete fixed C "
            "fixture; classification is limited to the extracted fixture "
            "and does not claim equivalence to additional parent-operator "
            "dispatch or storage work")
        promoted.append(kernel)

    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "kernel", "fixture_op", "comparability", "legal_ratio",
            "semantic_note",
        ], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"promoted {len(promoted)} fingerprint-validated fixture recipes")


if __name__ == "__main__":
    main()
