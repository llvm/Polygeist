#!/usr/bin/env python3
"""Collect median-of-five ATen Section 4.2 batch logs into one ledger."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "issues/aten_c_kernels/native_cuda_results"
CAMPAIGN = RESULTS / "section42_campaign"
SAMPLE = re.compile(
    r"SAMPLES kernel=(\S+) "
    r"(torch_cpu_us|cpu_aten_1thread_wall_us|torch_resident_us|"
    r"native_aten_resident_wall_us|raised_resident_wall_us)=([^\s]+)"
)
ERROR = re.compile(r"(?:NATIVE_)?RESULT kernel=(\S+).*?errors=(\d+)")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, default=Path("/tmp/aten-section42"))
    parser.add_argument("--output", type=Path,
                        default=CAMPAIGN / "results.csv")
    args = parser.parse_args()

    manifest = read_csv(CAMPAIGN / "manifest.csv")
    values: dict[str, dict[str, object]] = {row["kernel"]: {} for row in manifest}

    three_way = RESULTS / "silicon_threeway_median5.csv"
    if three_way.exists():
        for row in read_csv(three_way):
            if row["kernel"] not in values:
                continue
            values[row["kernel"]].update({
                "native_gpu_samples": row["native_gpu_samples"],
                "raised_samples": row["raised_gpu_samples"],
                "native_gpu_errors": row["native_gpu_errors"],
                "raised_errors": row["raised_gpu_errors"],
            })

    exact_native = RESULTS / "native_exact_region_median5.csv"
    if exact_native.exists():
        for row in read_csv(exact_native):
            if row["kernel"] in values:
                values[row["kernel"]].setdefault("native_gpu_samples", row["samples"])
                values[row["kernel"]].setdefault("native_gpu_errors", row["errors"])

    log_dir = args.work / "logs"
    for path in sorted(log_dir.glob("batch_*.log")) if log_dir.exists() else []:
        phase_match = re.match(r"batch_\d+_(cpu|native|raised)_", path.name)
        if not phase_match:
            continue
        phase = phase_match.group(1)
        text = path.read_text(errors="replace")
        if (phase == "cpu" and "_torch_" in path.name and
                not re.search(r"META .*threads=1 interop_threads=1", text)):
            continue
        for match in SAMPLE.finditer(text):
            kernel, _, samples = match.groups()
            if kernel in values and len(samples.split(",")) == 5:
                values[kernel][f"{phase if phase != 'native' else 'native_gpu'}_samples"] = samples
        for match in ERROR.finditer(text):
            kernel, errors = match.groups()
            if kernel in values:
                values[kernel][f"{phase if phase != 'native' else 'native_gpu'}_errors"] = errors

    output_rows = []
    for row in manifest:
        kernel = row["kernel"]
        found = values[kernel]
        result = dict(row)
        counts = {}
        for phase in ("cpu", "native_gpu", "raised"):
            raw = str(found.get(f"{phase}_samples", ""))
            parsed = [float(value) for value in raw.split(",") if value]
            counts[phase] = len(parsed)
            result[f"{phase}_samples"] = raw
            result[f"{phase}_median_us"] = (
                f"{statistics.median(parsed):.6f}" if len(parsed) == 5 else ""
            )
            result[f"{phase}_errors"] = str(found.get(f"{phase}_errors", ""))
        exact_native_required = row["native_gpu_runner"] in {
            "exact_region_harness", "aten_cpp_harness"
        }
        correct = result["raised_errors"] == "0" and (
            not exact_native_required or result["native_gpu_errors"] == "0"
        )
        result["three_way_status"] = (
            "COMPLETE" if all(value == 5 for value in counts.values()) and correct
            and row["input_alignment"] == "VERIFIED"
            else "PENDING"
        )
        output_rows.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(
        f"kernels={len(output_rows)} "
        f"cpu={sum(bool(row['cpu_median_us']) for row in output_rows)} "
        f"native={sum(bool(row['native_gpu_median_us']) for row in output_rows)} "
        f"raised={sum(bool(row['raised_median_us']) for row in output_rows)} "
        f"complete={sum(row['three_way_status'] == 'COMPLETE' for row in output_rows)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
