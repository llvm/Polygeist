#!/usr/bin/env python3

import csv
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path


SAMPLE_RE = re.compile(
    r"BENCH_SAMPLE phase=sample index=(\d+) wall_ms=([0-9.]+)"
    r"(?: device_ms=([0-9.]+))?"
)
RESULT_RE = re.compile(
    r"BENCH_RESULT workload=(\S+) shape=(\S+) status=(\S+)"
)
CORRECTNESS_RE = re.compile(r"BENCH_CORRECTNESS ([^\n]+)")

VARIANTS = {
    "gemm_native_cpu": ("native_cpu_single_core", "cpu_10"),
    "gemm_raised_naive": ("ginsbach_raised_naive_cpu", "cpu_10"),
    "gemm_native_cublas": ("direct_cublas", "physical_gpu_0"),
    "gemm_raised_cublas": ("ginsbach_raised_cublas", "physical_gpu_0"),
    "spmv_csr_native_cpu": ("native_cpu_single_core", "cpu_10"),
    "spmv_csr_raised_naive": ("ginsbach_raised_naive_cpu", "cpu_10"),
    "spmv_csr_native_cusparse":
        ("direct_cusparse", "physical_gpu_0"),
    "spmv_csr_raised_cusparse":
        ("ginsbach_raised_cusparse", "physical_gpu_0"),
}


def stats(values):
    ordered = sorted(values)
    return {
        "min_ms": min(values),
        "median_ms": statistics.median(values),
        "max_ms": max(values),
        "iqr_ms": ordered[3] - ordered[1],
        "raw_samples_ms": json.dumps(values),
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: summarize_results.py RESULT_DIR SOURCE_DIR BUILD_DIR PATCH"
        )
    result_dir = Path(sys.argv[1])
    source_dir = Path(sys.argv[2])
    build_dir = Path(sys.argv[3])
    patch_path = Path(sys.argv[4])

    rows = []
    for stem, (variant, device) in VARIANTS.items():
        path = result_dir / f"{stem}.log"
        text = path.read_text()
        matches = SAMPLE_RE.findall(text)
        result = RESULT_RE.search(text)
        correctness = CORRECTNESS_RE.search(text)
        if len(matches) != 5 or not result or not correctness:
            raise RuntimeError(f"incomplete result log: {path}")
        wall = [float(match[1]) for match in matches]
        device_values = [float(match[2]) for match in matches if match[2]]
        row = {
            "workload": result.group(1),
            "shape": result.group(2),
            "variant": variant,
            "device": device,
            "status": result.group(3),
            **stats(wall),
            "device_min_ms": "",
            "device_median_ms": "",
            "device_max_ms": "",
            "device_iqr_ms": "",
            "device_raw_samples_ms": "",
            "correctness": correctness.group(1),
            "protocol": "5 same-process warmups + 5 same-process samples",
            "scope": "steady-state resident synchronized host wall time",
            "log": path.name,
        }
        if device_values:
            device_stats = stats(device_values)
            row.update({
                "device_min_ms": device_stats["min_ms"],
                "device_median_ms": device_stats["median_ms"],
                "device_max_ms": device_stats["max_ms"],
                "device_iqr_ms": device_stats["iqr_ms"],
                "device_raw_samples_ms": device_stats["raw_samples_ms"],
            })
        rows.append(row)

    fields = list(rows[0])
    with (result_dir / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (result_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")

    by_key = {(row["workload"], row["variant"]): row for row in rows}
    gemm_native = by_key[("ginsbach_gemm", "native_cpu_single_core")]["min_ms"]
    gemm_direct = by_key[("ginsbach_gemm", "direct_cublas")]["min_ms"]
    gemm_gpu = by_key[("ginsbach_gemm", "ginsbach_raised_cublas")]["min_ms"]
    spmv_native = by_key[("ginsbach_spmv_csr", "native_cpu_single_core")]["min_ms"]
    spmv_direct = by_key[("ginsbach_spmv_csr", "direct_cusparse")]["min_ms"]
    spmv_gpu = by_key[("ginsbach_spmv_csr", "ginsbach_raised_cusparse")]["min_ms"]

    report = f"""# Ginsbach/LiLAC B300 reproduction

Secondary, non-fixed-clock B300 measurements. Headline values are the minimum
of five synchronized samples following five same-process warmups. CPU variants
were pinned to logical CPU 10. Complete outputs passed the declared checks.

| Workload | Variant | Min wall ms | Median wall ms | IQR ms | Min device ms |
|---|---|---:|---:|---:|---:|
"""
    for row in rows:
        device_min = row["device_min_ms"]
        device_text = f"{device_min:.6f}" if device_min != "" else "n/a"
        report += (
            f"| {row['workload']} ({row['shape']}) | {row['variant']} | "
            f"{row['min_ms']:.6f} | {row['median_ms']:.6f} | "
            f"{row['iqr_ms']:.6f} | {device_text} |\n"
        )
    report += f"""

- GEMM raised cuBLAS speedup over native single-core CPU: {gemm_native / gemm_gpu:.2f}x.
- GEMM raised cuBLAS relative to a direct cuBLAS call: {gemm_direct / gemm_gpu:.3f}x
  (`1.0x` means equal; values below `1.0x` mean the raised path is slower).
- CSR SpMV raised cuSPARSE speedup over native single-core CPU: {spmv_native / spmv_gpu:.2f}x.
- CSR SpMV raised cuSPARSE relative to a direct cuSPARSE call: {spmv_direct / spmv_gpu:.3f}x.
- These are generated-call results: Ginsbach's modified Clang recognized the
  original C loop and emitted `gemm_harness` or `spmv_csr_harness`; the linked
  LiLAC harness invoked the external NVIDIA library.
- The author-hosted 2020 `linearalgebra` branch required the retained CUDA 13
  compatibility/correctness patch. It fixes the removed CSR API, an undefined
  CSR extent name, GEMM row-major argument ordering/leading dimensions, and
  removes harness diagnostics from the timed path.
- The supplied NPB CG Class S application also completed with verification
  successful after automatic CSR-SpMV replacement, but its built-in single
  whole-program timing is not included in the fixed five-sample table.
"""
    (result_dir / "REPORT.md").write_text(report)

    artifacts = [
        source_dir / "gemm_bench.c",
        source_dir / "spmv_csr_bench.c",
        source_dir / "cuda_event_adapter.cu",
        source_dir / "gemm_native_cublas.cu",
        source_dir / "spmv_csr_native_cusparse.cu",
        patch_path,
        build_dir / "gemm_native_cpu",
        build_dir / "gemm_raised_naive",
        build_dir / "gemm_native_cublas",
        build_dir / "gemm_raised_cublas",
        build_dir / "spmv_csr_native_cpu",
        build_dir / "spmv_csr_raised_naive",
        build_dir / "spmv_csr_native_cusparse",
        build_dir / "spmv_csr_raised_cusparse",
    ]
    manifest = [
        {"path": str(path), "sha256": sha256(path)} for path in artifacts
    ]
    (result_dir / "artifact_hashes.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(report)


if __name__ == "__main__":
    main()
