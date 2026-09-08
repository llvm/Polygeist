#!/usr/bin/env python3
"""Validate native MFEM CUDA logs and merge them with the NE=1024 campaign."""

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
CHECKSUM = re.compile(r"^kernel=(\S+) final_checksum=([0-9.eE+-]+)$", re.MULTILINE)


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("native_root", type=Path)
    parser.add_argument("baseline_root", type=Path)
    parser.add_argument("--publication-dir", type=Path)
    args = parser.parse_args()

    samples = []
    grouped = defaultdict(lambda: defaultdict(list))
    native_checksums = defaultdict(set)
    logs = sorted((args.native_root / "raw").glob("process*_*.log"))
    if len(logs) != 85:
        raise SystemExit(f"expected 85 native logs, found {len(logs)}")
    for path in logs:
        text = path.read_text()
        if "RUN_STATUS=PASS\n" not in text:
            raise SystemExit(f"non-passing log: {path}")
        process = int(re.search(r"^process=(\d+)$", text, re.MULTILINE).group(1))
        checksum = CHECKSUM.search(text)
        if not checksum:
            raise SystemExit(f"missing checksum: {path}")
        native_checksums[checksum.group(1)].add(float(checksum.group(2)))
        local = []
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
            samples.append(row)
            local.append(row)
        if len(local) != 20 or {r["sample"] for r in local} != set(range(20)):
            raise SystemExit(f"invalid samples: {path}")
        grouped[local[0]["kernel"]][process] = [r["runtime_us"] for r in local]
    if len(samples) != 1700 or len(grouped) != 17:
        raise SystemExit(f"unexpected totals: samples={len(samples)} kernels={len(grouped)}")
    write_csv(args.native_root / "raw_samples.csv", samples)

    summaries = []
    for kernel, processes in sorted(grouped.items()):
        if set(processes) != set(range(5)):
            raise SystemExit(f"incomplete processes: {kernel}")
        medians = [statistics.median(processes[p]) for p in range(5)]
        q1, _, q3 = statistics.quantiles(medians, n=4, method="inclusive")
        flat = [sample for process in processes.values() for sample in process]
        summaries.append({
            "kernel": kernel,
            "implementation": "mfem_native_cuda",
            "ne": 1024,
            "processes": 5,
            "samples_per_process": 20,
            "median_of_process_medians_us": statistics.median(medians),
            "q1_process_medians_us": q1,
            "q3_process_medians_us": q3,
            "iqr_process_medians_us": q3 - q1,
            "min_sample_us": min(flat),
            "max_sample_us": max(flat),
            "process_medians_us": ";".join(f"{v:.9f}" for v in medians),
        })
    write_csv(args.native_root / "summary.csv", summaries)

    vanilla_checksums = defaultdict(set)
    for path in (args.baseline_root / "raw").glob("process*_vanilla_cpu.log"):
        match = CHECKSUM.search(path.read_text())
        if match:
            vanilla_checksums[match.group(1)].add(float(match.group(2)))
    audits = []
    for kernel in sorted(grouped):
        native = next(iter(native_checksums[kernel]))
        vanilla = next(iter(vanilla_checksums[kernel]))
        error = abs(native - vanilla)
        tolerance = 1e-11 + 1e-10 * abs(vanilla)
        audits.append({
            "kernel": kernel,
            "vanilla_checksum": f"{vanilla:.17g}",
            "native_mfem_checksum": f"{native:.17g}",
            "absolute_error": f"{error:.17g}",
            "tolerance": f"{tolerance:.17g}",
            "status": "PASS" if error <= tolerance else "FAIL",
        })
    write_csv(args.native_root / "checksum_audit.csv", audits)

    native_by_kernel = {r["kernel"]: r for r in summaries}
    audit_by_kernel = {r["kernel"]: r for r in audits}
    merged = []
    for row in read_csv(args.baseline_root / "comparison.csv"):
        native = float(native_by_kernel[row["kernel"]]["median_of_process_medians_us"])
        raised_gpu = float(row["raised_gpu_us"])
        row["native_mfem_gpu_us"] = native
        row["native_mfem_status"] = (
            "measured; checksum " + audit_by_kernel[row["kernel"]]["status"]
        )
        row["raised_gpu_over_native_mfem"] = raised_gpu / native
        merged.append(row)
    write_csv(args.native_root / "comparison_with_native.csv", merged)
    if args.publication_dir:
        args.publication_dir.mkdir(parents=True, exist_ok=True)
        write_csv(args.publication_dir / "native_summary_20260908.csv", summaries)
        write_csv(args.publication_dir / "native_checksum_audit_20260908.csv", audits)
        write_csv(args.publication_dir / "comparison_with_native_20260908.csv", merged)
    failures = sum(row["status"] != "PASS" for row in audits)
    print(f"validated_logs={len(logs)} raw_samples={len(samples)} "
          f"kernels={len(summaries)} checksum_failures={failures}")


if __name__ == "__main__":
    main()
