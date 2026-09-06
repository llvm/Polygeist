#!/usr/bin/env python3
"""Finalize the explicit Section 4.2 ledger from retained raw timing logs."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "logs"
MANIFEST = ROOT / "manifest.csv"

CPU_LIBRARY_PASS = {"2mm", "atax", "bicg", "gemm", "gemver", "gesummv", "mvt"}
CPU_LIBRARY_FAIL = {"doitgen", "gramschmidt"}
GPU_PASS = {"covariance", "deriche"}
GPU_FAIL = {"2mm", "atax", "bicg", "doitgen", "gemm", "gemver"}
GPU_BLOCKED = {"3mm", "gesummv", "gramschmidt", "mvt"}
COMMON = {"gemm", "syr2k", "2mm", "3mm"}

REASONS = {
    "adi": "raising retains polygeist submap operations; no executable residual",
    "correlation": "raised residual executes but differs from the native output",
    "durbin": "raising retains dynamic tensor submap operations",
    "ludcmp": "raising retains a dynamic tensor submap operation",
    "nussinov": "raising retains memref submap operations",
    "seidel-2d": "raising retains eight tensor submap operations",
    "doitgen": "matched CPU/GPU ABI lowering rejects changed loop-carried tensor values",
    "gramschmidt": "OpenBLAS result is non-finite/wrong; audited GPU rerun blocked by device failure",
    "2mm": "CPU library passes; audited cuBLAS result is numerically wrong; PolyBenchGPU is not equivalent",
    "atax": "CPU library passes; audited cuBLAS result is numerically wrong; PolyBenchGPU is not equivalent",
    "bicg": "CPU library passes; audited cuBLAS result is numerically wrong; PolyBenchGPU is not equivalent",
    "gemm": "CPU library passes; audited cuBLAS result is numerically wrong; external baselines unavailable",
    "gemver": "CPU library passes; audited cuBLAS result is numerically wrong; PolyBenchGPU is not equivalent",
    "3mm": "cuTensorNet rerun and timing blocked after CUDA device failure; CPU library unavailable",
    "gesummv": "CPU library passes; audited GPU run timed out and left CUDA unavailable",
    "mvt": "CPU library passes; audited GPU rerun blocked after CUDA device failure",
    "covariance": "residual and audited GPU correctness pass; no eligible CBLAS match and GPU timing blocked",
    "deriche": "residual and audited GPU correctness pass; no eligible CBLAS match and GPU timing blocked",
    "lu": "residual correctness passes, but its timing warmup was killed by the host; no external-library match",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


rows = read_rows(MANIFEST)
for row in rows:
    kernel = row["kernel"]
    row["native_cpu_status"] = "pass"
    row["polybenchgpu_status"] = "unavailable"
    row["kernelfarer_status"] = "unavailable" if kernel in COMMON else "not_applicable"
    row["polly_status"] = "unavailable" if kernel in COMMON else "not_applicable"
    if kernel in COMMON:
        log = LOGS / kernel / "polly_correctness.log"
        if log.exists() and "correctness=pass" in log.read_text(errors="replace"):
            row["polly_status"] = "pass"
    if kernel in CPU_LIBRARY_PASS:
        row["cpu_library_status"] = "pass"
    elif kernel in CPU_LIBRARY_FAIL:
        row["cpu_library_status"] = "fail"
    else:
        row["cpu_library_status"] = "unavailable"
    if row["residual_cpu_status"] != "pass":
        row["raised_gpu_status"] = "unavailable"
        row["overall_status"] = "fail"
    elif kernel in GPU_PASS:
        row["raised_gpu_status"] = "pass"
        row["overall_status"] = "partial"
    elif kernel in GPU_FAIL:
        row["raised_gpu_status"] = "fail"
        row["overall_status"] = "partial"
    elif kernel in GPU_BLOCKED:
        row["raised_gpu_status"] = "blocked"
        row["overall_status"] = "partial"
    else:
        row["raised_gpu_status"] = "unavailable"
        row["overall_status"] = "partial"
    row["failure_reason"] = REASONS.get(
        kernel,
        "residual correctness passes; no external-library match; PolyBenchGPU is not equivalent",
    )

write_rows(MANIFEST, rows, list(rows[0]))


def samples(path: Path, expected_config: str | None = None) -> list[float]:
    if not path.exists():
        return []
    values = []
    for line in path.read_text(errors="replace").splitlines():
        fields = line.split(",")
        if len(fields) == 5:
            _, config, sample, rc, value = fields
        elif len(fields) == 4:
            config, sample, rc, value = fields
        else:
            continue
        if expected_config and config != expected_config:
            continue
        try:
            if int(sample) in range(1, 6) and int(rc) == 0:
                values.append(float(value))
        except ValueError:
            continue
    return values if len(values) == 5 else []


native_ms: dict[str, float] = {}
cpu_records: list[dict[str, str]] = []


def add_cpu(kernel: str, configuration: str, values: list[float], library: str,
            scope: str, command: str, log: str) -> None:
    if len(values) != 5:
        return
    time_ms = statistics.median(values) * 1000.0
    speedup = "1.000000" if configuration == "native_clang18_noinline" else ""
    if configuration != "native_clang18_noinline" and kernel in native_ms and time_ms:
        speedup = f"{native_ms[kernel] / time_ms:.6f}"
    cpu_records.append({
        "kernel": kernel, "configuration": configuration,
        "dataset": "LARGE", "datatype": "double", "correctness_status": "pass",
        "samples": "5", "warmups": "1", "statistic": "median",
        "time_ms": f"{time_ms:.6f}", "speedup_vs_native": speedup,
        "library": library, "measurement_scope": scope,
        "command": command, "log": log,
    })


for row in rows:
    kernel = row["kernel"]
    values = samples(LOGS / kernel / "native_timing_final_raw.log",
                     "native_clang18_noinline")
    if values:
        native_ms[kernel] = statistics.median(values) * 1000.0
        add_cpu(kernel, "native_clang18_noinline", values, "none",
                "PolyBench kernel call; pinned CPU 21; one process/thread",
                "issues/polybench_section42/run_native_timing_final.sh",
                f"logs/{kernel}/native_timing_final_raw.log")

for row in rows:
    kernel = row["kernel"]
    if row["residual_cpu_status"] == "pass":
        add_cpu(kernel, "raised_residual_cpu",
                samples(LOGS / kernel / "residual_timing_raw.log", "raised_residual_cpu"),
                "none", "PolyBench kernel call; pinned CPU 21; one process/thread",
                "issues/polybench_section42/run_cpu_timing.sh",
                f"logs/{kernel}/residual_timing_raw.log")
    if kernel in CPU_LIBRARY_PASS:
        add_cpu(kernel, "openblas_cblas_1t",
                samples(LOGS / kernel / "cpu_library_timing_raw.log", "openblas_cblas_1t"),
                "OpenBLAS 0.3.20 / CBLAS", "PolyBench kernel call; pinned CPU 21; one BLAS thread",
                "issues/polybench_section42/run_cpu_timing.sh",
                f"logs/{kernel}/cpu_library_timing_raw.log")
    if row["polly_status"] == "pass":
        add_cpu(kernel, "polly14",
                samples(LOGS / kernel / "polly_timing_raw.log", "polly14"),
                "LLVM Polly 14", "PolyBench kernel call; pinned CPU 21; one process/thread",
                "issues/polybench_section42/run_polly_subset.sh",
                f"logs/{kernel}/polly_timing_raw.log")

cpu_fields = [
    "kernel", "configuration", "dataset", "datatype", "correctness_status",
    "samples", "warmups", "statistic", "time_ms", "speedup_vs_native", "library",
    "measurement_scope", "command", "log",
]
write_rows(ROOT / "performance_cpu.csv", cpu_records, cpu_fields)

gpu_fields = [
    "kernel", "configuration", "dataset", "datatype", "correctness_status",
    "samples", "warmups", "statistic", "device_time_ms", "end_to_end_time_ms",
    "speedup_vs_native", "library", "device", "measurement_scope", "command", "log",
]
# No audited GPU configuration reached both correctness and final timing.
write_rows(ROOT / "performance_gpu.csv", [], gpu_fields)

pending = [(row["kernel"], key) for row in rows for key, value in row.items()
           if value.upper() == "PENDING"]
if pending:
    raise SystemExit(f"pending manifest cells remain: {pending}")
print(f"finalized {len(rows)} manifest rows and {len(cpu_records)} CPU records")
