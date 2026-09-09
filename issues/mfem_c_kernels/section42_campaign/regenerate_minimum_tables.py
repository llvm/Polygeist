#!/usr/bin/env python3
"""Regenerate MFEM tables with retained-sample minima as headline values.

The September 8 campaign used five processes with twenty timed samples each,
not the newer one-process minimum-of-five protocol.  This script preserves
that distinction in the output metadata while making the minimum of the
retained samples the displayed value.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SAMPLE_RE = re.compile(
    r"application=(\S+) implementation=(\S+) sample=\d+ time_us=([0-9.eE+-]+)"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
        "iqr": (statistics.median(ordered[-midpoint:])
                - statistics.median(ordered[:midpoint])),
    }


def summary_distributions(summary: Path, raw_samples: Path) -> dict[tuple[str, str], dict[str, float]]:
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in read_csv(raw_samples):
        samples[(row["kernel"], row["implementation"])].append(
            float(row["runtime_us"])
        )
    expected = {
        (row["kernel"], row["implementation"]):
        int(row["processes"]) * int(row["samples_per_process"])
        for row in read_csv(summary)
    }
    for key, count in expected.items():
        if len(samples[key]) != count:
            raise ValueError(f"{key}: expected {count} samples, found {len(samples[key])}")
    return {key: distribution(values) for key, values in samples.items()}


def parse_application_samples(directory: Path) -> dict[tuple[str, str], dict[str, float]]:
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    for log in sorted(directory.glob("*.silicon.log")):
        for match in SAMPLE_RE.finditer(log.read_text(errors="replace")):
            application, implementation, runtime = match.groups()
            samples[(application, implementation)].append(float(runtime))
    return {key: distribution(values) for key, values in samples.items()}


def ms(stats: dict[str, float], statistic: str) -> float:
    return stats[statistic] / 1000.0


def regenerate_kernels(kernel_summary: Path, kernel_samples: Path,
                       native_summary: Path, native_samples: Path) -> None:
    existing = read_csv(ROOT / "performance_20260908.csv")
    distributions = summary_distributions(kernel_summary, kernel_samples)
    distributions.update(summary_distributions(native_summary, native_samples))
    implementation = {
        "vanilla_cpu": "vanilla_c_cpu",
        "raised_cpu": "polygeist_raised_cpu",
        "raised_gpu": "polygeist_raised_gpu",
        "native_mfem_gpu": "mfem_native_cuda",
    }
    output = []
    for row in existing:
        values = {
            name: distributions[(row["kernel"], impl)]
            for name, impl in implementation.items()
        }
        headline = {name: ms(stats, "min") for name, stats in values.items()}
        result: dict[str, object] = {
            "kernel": row["kernel"],
            "validated_sites": row["validated_sites"],
        }
        for name in implementation:
            result[f"{name}_ms"] = f'{headline[name]:.9f}'
            result[f"{name}_median_ms"] = f'{ms(values[name], "median"):.9f}'
            result[f"{name}_max_ms"] = f'{ms(values[name], "max"):.9f}'
            result[f"{name}_iqr_ms"] = f'{ms(values[name], "iqr"):.9f}'
        result.update({
            "raised_cpu_slowdown_vs_vanilla":
                f'{headline["raised_cpu"] / headline["vanilla_cpu"]:.9f}',
            "raised_gpu_slowdown_vs_vanilla":
                f'{headline["raised_gpu"] / headline["vanilla_cpu"]:.9f}',
            "raised_gpu_slowdown_vs_native_mfem":
                f'{headline["raised_gpu"] / headline["native_mfem_gpu"]:.9f}',
            "processes": row["processes"],
            "samples_per_process": row["samples_per_process"],
            "headline_statistic": "minimum_of_100_retained_samples_legacy_campaign",
            "correctness": row["correctness"],
            "notes": row["notes"],
        })
        output.append(result)
    write_csv(ROOT / "performance_20260908.csv", output)


def regenerate_applications(application_root: Path) -> None:
    existing = read_csv(ROOT / "application_performance_20260908.csv")
    gpu = parse_application_samples(application_root / "raw")
    cpu = parse_application_samples(application_root / "raw-raised-cpu")
    output = []
    for row in existing:
        application = row["id"]
        values = {
            "vanilla_cpu": gpu[(application, "vanilla_cpu")],
            "raised_cpu": cpu[(application, "raised_cpu")],
            "raised_gpu": gpu[(application, "raised_gpu")],
        }
        result: dict[str, object] = dict(row)
        for name, stats in values.items():
            result[f"{name}_us"] = f'{stats["min"]:.9f}'
            result[f"{name}_median_us"] = f'{stats["median"]:.9f}'
            result[f"{name}_max_us"] = f'{stats["max"]:.9f}'
            result[f"{name}_iqr_us"] = f'{stats["iqr"]:.9f}'
        result["raised_cpu_over_vanilla"] = (
            f'{values["raised_cpu"]["min"] / values["vanilla_cpu"]["min"]:.9f}')
        result["raised_gpu_over_vanilla"] = (
            f'{values["raised_gpu"]["min"] / values["vanilla_cpu"]["min"]:.9f}')
        result["headline_statistic"] = (
            "minimum_of_retained_legacy_campaign_samples")
        output.append(result)
    write_csv(ROOT / "application_performance_20260908.csv", output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-summary", type=Path, required=True)
    parser.add_argument("--kernel-samples", type=Path, required=True)
    parser.add_argument("--native-summary", type=Path, required=True)
    parser.add_argument("--native-samples", type=Path, required=True)
    parser.add_argument("--application-root", type=Path, required=True)
    args = parser.parse_args()
    regenerate_kernels(args.kernel_summary, args.kernel_samples,
                       args.native_summary, args.native_samples)
    regenerate_applications(args.application_root)
    print("regenerated MFEM kernel and derived-application minimum tables")


if __name__ == "__main__":
    main()
