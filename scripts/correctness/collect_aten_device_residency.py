#!/usr/bin/env python3
"""Merge ATen mapped/device-resident Jetson logs into published CSVs."""

from __future__ import annotations

import argparse
import csv
import json
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
            filename_match = re.match(
                r"(aten_[A-Za-z0-9_]+)\.(mapped|device)\.silicon\.log$",
                log.name)
            if not kernel_match and not filename_match:
                continue
            kernel = (kernel_match.group(1) if kernel_match else
                      filename_match.group(1))
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
            if filename_match and not re.search(
                    value_patterns[filename_match.group(2)], text):
                mode = filename_match.group(2)
                entry[f"{mode}_error"] = True
    return rows


def load_manifests(paths: list[Path]) -> dict[str, dict[str, object]]:
    result = {}
    for path in paths:
        payload = json.loads(path.read_text())
        rows = (payload if isinstance(payload, list) else
                payload.get("rows", payload.get("cases", [])))
        for row in rows:
            result[row["kernel"]] = row
    return result


def load_matches(path: Path) -> dict[str, str]:
    with path.open(newline="") as stream:
        return {row["kernel"]: row["current_match"]
                for row in csv.DictReader(stream)}


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
    parser.add_argument("--manifest", action="append", type=Path, default=[])
    parser.add_argument(
        "--match-csv", type=Path,
        default=ROOT / "issues/aten_c_kernels/cuda_library_audit.csv")
    parser.add_argument("--resident-override", action="append", default=[],
                        metavar="KERNEL=MICROSECONDS")
    args = parser.parse_args()

    measured = parse_logs(args.run_root)
    manifests = load_manifests(args.manifest)
    matches = load_matches(args.match_csv)
    resident_overrides = parse_overrides(args.resident_override)
    with args.main_csv.open(newline="") as stream:
        main_rows = list(csv.DictReader(stream))
        fieldnames = list(main_rows[0])
    by_kernel = {row["kernel"]: row for row in main_rows}

    # The exhaustive FULL-match batch includes kernels that were not in the
    # original hand-curated 77-row performance table. Add them from the build
    # manifest so a successful silicon run becomes visible in CE.
    for kernel in sorted(measured):
        if kernel in by_kernel or kernel not in manifests:
            continue
        meta = manifests[kernel]
        row = {name: "—" for name in fieldnames}
        row.update({
            "kernel": kernel,
            "executable_status": "EXECUTED",
            "correctness": "—",
            "problem": str(meta.get("problem", "—")),
            "baseline": matches.get(kernel, "raised public CUDA library call"),
            "hardware": "Jetson Orin sm87 MAXN CUDA 12.6",
            "notes": "exhaustive FULL-raise/FULL-match silicon batch",
        })
        main_rows.append(row)
        by_kernel[kernel] = row

    existing_output = {}
    if args.output.exists():
        with args.output.open(newline="") as stream:
            existing_output = {row["kernel"]: row for row in csv.DictReader(stream)}
    output_by_kernel = dict(existing_output)
    for kernel in sorted(measured):
        data = measured[kernel]
        if "mapped" not in data:
            continue
        main = by_kernel.get(kernel)
        if not main:
            continue
        mapped = float(data["mapped"])
        device = float(data["device"]) if "device" in data else None
        resident = resident_overrides.get(kernel)
        if resident is None and main["resident_cuda_us"] not in {"", "—"}:
            resident = float(main["resident_cuda_us"])

        # Keep the original broad status table current as well as emitting the
        # focused three-way comparison used by the CE performance page.
        main["raised_us"] = fmt(mapped)
        if resident is not None:
            main["resident_cuda_us"] = fmt(resident)
            main["raised_over_resident"] = fmt(mapped / resident)
        main["executable_status"] = "EXECUTED"
        main["correctness"] = "PASS" if data.get("mapped_correct") else "FAIL"
        main["statistic"] = "warm median of process runs 2-4"
        main["notes"] = (
            "mapped and cudaMalloc device-resident raised paths correctness-gated"
            if device is not None else
            "mapped path correctness-gated; cudaMalloc path unavailable because "
            "the generated ABI wrapper performs a host memcpy epilogue")

        output_by_kernel[kernel] = {
            "kernel": kernel,
            "correctness": main["correctness"],
            "problem": main["problem"],
            "mapped_raised_us": fmt(mapped),
            "device_resident_us": fmt(device),
            "resident_cuda_us": fmt(resident),
            "mapped_over_resident": fmt(mapped / resident if resident else None),
            "device_over_resident": fmt(
                device / resident if device is not None and resident else None),
            "mapped_over_device": fmt(
                mapped / device if device is not None else None),
            "hardware": "Jetson Orin sm87 MAXN CUDA 12.6",
            "statistic": "median process runs 2-4; 20 timed calls/process",
            "notes": ("RMS reduction reassociation tolerance 2e-3" if
                      kernel == "aten_rms_norm" else
                      "correctness-gated" if device is not None else
                      "mapped PASS; cudaMalloc ABI wrapper host-memcpy limitation"),
        }

    output_rows = [output_by_kernel[k] for k in sorted(output_by_kernel)]

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
    published_now = sum(1 for kernel in measured if kernel in output_by_kernel)
    print(f"published_now={published_now} total={len(output_rows)} output={args.output}")
    return 0 if published_now == len(measured) else 1


if __name__ == "__main__":
    raise SystemExit(main())
