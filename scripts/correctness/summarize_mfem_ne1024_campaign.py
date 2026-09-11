#!/usr/bin/env python3
"""Validate and summarize raw logs from run_mfem_ne1024_campaign.sh."""

import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

SAMPLE = re.compile(
    r"^implementation=(\S+) kernel=(\S+) ne=(\d+) sample=(\d+) "
    r"runtime_us=([0-9.eE+-]+)$"
)


def quartiles(values):
    return statistics.quantiles(sorted(values), n=4, method="inclusive")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_root", type=Path)
    args = parser.parse_args()
    root = args.campaign_root
    rows = []
    grouped = defaultdict(lambda: defaultdict(list))

    logs = sorted((root / "raw").glob("process*_*.log"))
    if len(logs) != 255:
        raise SystemExit(f"expected 255 completed logs, found {len(logs)}")
    for path in logs:
        text = path.read_text()
        if "RUN_STATUS=PASS\n" not in text:
            raise SystemExit(f"non-passing log: {path}")
        process_match = re.search(r"^process=(\d+)$", text, re.MULTILINE)
        if not process_match:
            raise SystemExit(f"missing process id: {path}")
        process = int(process_match.group(1))
        samples = []
        for line in text.splitlines():
            match = SAMPLE.match(line)
            if not match:
                continue
            implementation, kernel, ne, sample, runtime = match.groups()
            row = {
                "process": process,
                "implementation": implementation,
                "kernel": kernel,
                "ne": int(ne),
                "sample": int(sample),
                "runtime_us": float(runtime),
                "source_log": path.name,
            }
            rows.append(row)
            samples.append(row)
        if len(samples) != 20 or {r["sample"] for r in samples} != set(range(20)):
            raise SystemExit(f"invalid sample set: {path}")
        key = (samples[0]["kernel"], samples[0]["implementation"])
        grouped[key][process].extend(r["runtime_us"] for r in samples)

    if len(rows) != 5100 or len(grouped) != 51:
        raise SystemExit(f"unexpected totals: samples={len(rows)} groups={len(grouped)}")

    with (root / "raw_samples.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for (kernel, implementation), processes in sorted(grouped.items()):
        if set(processes) != set(range(5)) or any(len(v) != 20 for v in processes.values()):
            raise SystemExit(f"incomplete process set: {kernel} {implementation}")
        process_medians = [statistics.median(processes[p]) for p in range(5)]
        q1, _, q3 = quartiles(process_medians)
        all_samples = [v for values in processes.values() for v in values]
        summaries.append({
            "kernel": kernel,
            "implementation": implementation,
            "ne": 1024,
            "processes": 5,
            "samples_per_process": 20,
            "median_of_process_medians_us": statistics.median(process_medians),
            "q1_process_medians_us": q1,
            "q3_process_medians_us": q3,
            "iqr_process_medians_us": q3 - q1,
            "min_sample_us": min(all_samples),
            "max_sample_us": max(all_samples),
            "process_medians_us": ";".join(f"{v:.9f}" for v in process_medians),
        })

    with (root / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(summaries[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)

    by_kernel = defaultdict(dict)
    for row in summaries:
        by_kernel[row["kernel"]][row["implementation"]] = row
    comparisons = []
    for kernel, implementations in sorted(by_kernel.items()):
        vanilla = implementations["vanilla_c_cpu"]["median_of_process_medians_us"]
        raised_cpu = implementations["polygeist_raised_cpu"]["median_of_process_medians_us"]
        raised_gpu = implementations["polygeist_raised_gpu"]["median_of_process_medians_us"]
        comparisons.append({
            "kernel": kernel,
            "vanilla_cpu_us": vanilla,
            "raised_cpu_us": raised_cpu,
            "raised_gpu_us": raised_gpu,
            "vanilla_over_raised_cpu": vanilla / raised_cpu,
            "vanilla_over_raised_gpu": vanilla / raised_gpu,
            "raised_cpu_over_raised_gpu": raised_cpu / raised_gpu,
            "native_mfem_gpu_us": "",
            "native_mfem_status": "exact counterpart not measured in this fixture campaign",
        })
    with (root / "comparison.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(comparisons[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)

    print(f"validated_logs={len(logs)} raw_samples={len(rows)} summary_rows={len(summaries)} comparison_rows={len(comparisons)}")


if __name__ == "__main__":
    main()
