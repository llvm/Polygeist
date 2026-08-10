#!/usr/bin/env python3
"""Merge ATen mapped/device-resident Jetson logs into published CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAIN = (ROOT / "issues/aten_c_kernels/silicon_results" /
                "large_problem_comparison.csv")
DEFAULT_OUTPUT = (ROOT / "issues/aten_c_kernels/silicon_results" /
                  "device_residency_comparison.csv")


def parse_logs(roots: list[Path]) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    value_patterns = {
        "mapped": re.compile(r"raised_gpu_us=([0-9.eE+-]+)"),
        "device": re.compile(r"raised_device_us=([0-9.eE+-]+)"),
    }
    for root in roots:
        for log in root.rglob("*.silicon.log"):
            text = log.read_text(errors="replace")
            kernel_match = re.search(r"kernel=(aten_[A-Za-z0-9_]+)", text)
            if not kernel_match:
                continue
            kernel = kernel_match.group(1)
            entry = rows.setdefault(kernel, {"logs": []})
            entry["logs"].append(str(log))
            pass_count = len(re.findall(
                rf"kernel={re.escape(kernel)} correctness=PASS", text))
            for mode, pattern in value_patterns.items():
                values = [float(v) for v in pattern.findall(text)]
                if not values:
                    continue
                # Process run 1 includes library/plan initialization. Publish
                # the median of warm process runs 2-4, matching the existing
                # ATen and MFEM CE convention.
                warm = values[1:4] if len(values) >= 4 else values
                entry[mode] = statistics.median(warm)
                entry[f"{mode}_samples"] = len(values)
                entry[f"{mode}_correct"] = pass_count == len(values)
    return rows


def parse_overrides(values: list[str]) -> dict[str, float]:
    result = {}
    for value in values:
        kernel, timing = value.split("=", 1)
        result[kernel] = float(timing)
    return result


def fmt(value: float | None, digits: int = 6) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", type=Path, required=True)
    parser.add_argument("--main-csv", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resident-override", action="append", default=[],
                        metavar="KERNEL=MICROSECONDS")
    args = parser.parse_args()

    measured = parse_logs(args.run_root)
    resident_overrides = parse_overrides(args.resident_override)
    with args.main_csv.open(newline="") as stream:
        main_rows = list(csv.DictReader(stream))
        fieldnames = list(main_rows[0])
    by_kernel = {row["kernel"]: row for row in main_rows}

    output_rows = []
    for kernel in sorted(measured):
        data = measured[kernel]
        if "mapped" not in data or "device" not in data:
            continue
        main = by_kernel.get(kernel)
        if not main:
            continue
        mapped = float(data["mapped"])
        device = float(data["device"])
        resident = resident_overrides.get(kernel)
        if resident is None and main["resident_cuda_us"] not in {"", "—"}:
            resident = float(main["resident_cuda_us"])

        # Keep the original broad status table current as well as emitting the
        # focused three-way comparison used by the CE performance page.
        main["raised_us"] = fmt(mapped)
        if resident is not None:
            main["resident_cuda_us"] = fmt(resident)
            main["raised_over_resident"] = fmt(mapped / resident)
        main["correctness"] = (
            "PASS" if data.get("mapped_correct") and data.get("device_correct")
            else "FAIL")
        main["statistic"] = "warm median of process runs 2-4"
        main["notes"] = (
            "mapped and cudaMalloc device-resident raised paths correctness-gated")

        output_rows.append({
            "kernel": kernel,
            "correctness": main["correctness"],
            "problem": main["problem"],
            "mapped_raised_us": fmt(mapped),
            "device_resident_us": fmt(device),
            "resident_cuda_us": fmt(resident),
            "mapped_over_resident": fmt(mapped / resident if resident else None),
            "device_over_resident": fmt(device / resident if resident else None),
            "mapped_over_device": fmt(mapped / device),
            "hardware": "Jetson Orin sm87 MAXN CUDA 12.6",
            "statistic": "median process runs 2-4; 20 timed calls/process",
            "notes": ("RMS reduction reassociation tolerance 2e-3" if
                      kernel == "aten_rms_norm" else "correctness-gated"),
        })

    # The new large runs intentionally scale these two legacy fixtures from
    # 1M to 8M elements; their native baselines are passed as overrides.
    for kernel in ("aten_rms_norm", "aten_softmax"):
        if kernel in measured and kernel in by_kernel:
            by_kernel[kernel]["problem"] = "N8388608"
    for row in output_rows:
        if row["kernel"] in {"aten_rms_norm", "aten_softmax"}:
            row["problem"] = "N8388608"

    with args.main_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(main_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_fields = list(output_rows[0]) if output_rows else []
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=out_fields,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"published={len(output_rows)} output={args.output}")
    return 0 if len(output_rows) == len(measured) else 1


if __name__ == "__main__":
    raise SystemExit(main())
