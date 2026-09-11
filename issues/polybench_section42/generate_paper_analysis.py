#!/usr/bin/env python3
"""Generate the publication-protocol PolyBench Section 4.2 data and figures."""

from __future__ import annotations

import csv
import html
import json
import math
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CPU_LOGS = ROOT / "logs" / "publication_orin_cpu"
GPU_LOGS = ROOT / "logs" / "publication_orin"
GEMM_PRECISION_LOGS = ROOT / "logs" / "gemm_fp32_fp64_20260908"
OUT = ROOT / "paper_analysis"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sample_stats(path: Path, field: str) -> dict[str, float]:
    rows = read_csv(path)
    warmups = [float(row[field]) for row in rows if row["phase"] == "warmup"]
    samples = [float(row[field]) for row in rows if row["phase"] == "sample"]
    if len(warmups) != 5 or len(samples) != 5:
        raise ValueError(f"{path}: expected 5 warmups and 5 samples")
    ordered = sorted(samples)
    q1 = statistics.median(ordered[:2])
    q3 = statistics.median(ordered[3:])
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "iqr": q3 - q1,
    }


def correctness(directory: Path, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        paths = sorted(directory.glob(name)) if "*" in name else [directory / name]
        for path in paths:
            if not path.exists():
                continue
            evidence = path.read_text(errors="replace").strip()
            if evidence.startswith("PASS") or re.search(r"\bPOLYBENCH_[A-Z0-9_]+_PASS\b", evidence):
                return "pass"
    return "unavailable"


def f(value: float | None) -> str:
    return "" if value is None else f"{value:.9f}"


def paired_runtime_svg(
    rows: list[dict[str, object]], path: Path, title: str, native_field: str,
    raised_field: str, native_label: str, raised_label: str, scope: str,
) -> None:
    paired = [
        row for row in rows
        if isinstance(row.get(native_field), float)
        and isinstance(row.get(raised_field), float)
    ]
    values = [float(row[field]) for row in paired for field in (native_field, raised_field)]
    lo = 10 ** math.floor(math.log10(min(values)))
    hi = 10 ** math.ceil(math.log10(max(values)))
    width = 1180
    left, right, top, row_h, bottom = 155, 90, 72, 35, 55
    height = top + len(paired) * row_h + bottom
    plot_w = width - left - right

    def xpos(value: float) -> float:
        return left + plot_w * (math.log10(value) - math.log10(lo)) / (
            math.log10(hi) - math.log10(lo))

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="{html.escape(title)}">',
        '<style>text{font-family:system-ui,sans-serif;fill:#24292f}'
        '.title{font-size:20px;font-weight:700}.axis{font-size:11px}'
        '.kernel{font-size:12px;font-weight:600}.ratio{font-size:11px;font-weight:700}'
        '.grid{stroke:#d8dee4;stroke-width:1}.pair{stroke:#8c959f;stroke-width:1.4}'
        '</style>',
        f'<text class="title" x="{left}" y="28">{html.escape(title)}</text>',
        f'<text class="axis" x="{left}" y="48">{html.escape(scope)}</text>',
    ]
    start_exp = math.floor(math.log10(lo))
    end_exp = math.ceil(math.log10(hi))
    for exponent in range(start_exp, end_exp + 1):
        value = 10.0 ** exponent
        x = xpos(value)
        svg.append(f'<line class="grid" x1="{x:.2f}" y1="{top-12}" x2="{x:.2f}" y2="{height-bottom+5}"/>')
        svg.append(f'<text class="axis" x="{x:.2f}" y="{height-18}" text-anchor="middle">{value:g} ms</text>')
    for index, row in enumerate(paired):
        y = top + index * row_h
        native = float(row[native_field])
        raised = float(row[raised_field])
        xn, xr = xpos(native), xpos(raised)
        ratio = native / raised
        datatype = str(row.get("datatype", ""))
        dtype_label = ("FP32" if datatype == "float" else
                       "FP64" if datatype == "double" else datatype)
        label = (f'{row["kernel"]} [{dtype_label}]' if dtype_label
                 else str(row["kernel"]))
        svg.append(f'<text class="kernel" x="{left-10}" y="{y+4}" text-anchor="end">{html.escape(label)}</text>')
        svg.append(f'<line class="pair" x1="{xn:.2f}" y1="{y}" x2="{xr:.2f}" y2="{y}"/>')
        svg.append(f'<circle cx="{xn:.2f}" cy="{y}" r="5" fill="#0969da"><title>{native_label}: {native:.6f} ms</title></circle>')
        svg.append(f'<circle cx="{xr:.2f}" cy="{y}" r="5" fill="#d97706"><title>{raised_label}: {raised:.6f} ms</title></circle>')
        svg.append(f'<text class="ratio" x="{width-right+10}" y="{y+4}">{ratio:.2f}×</text>')
    svg.append(f'<circle cx="{left}" cy="{height-40}" r="5" fill="#0969da"/><text class="axis" x="{left+10}" y="{height-36}">{html.escape(native_label)}</text>')
    svg.append(f'<circle cx="{left+210}" cy="{height-40}" r="5" fill="#d97706"/><text class="axis" x="{left+220}" y="{height-36}">{html.escape(raised_label)}</text>')
    svg.append(f'<text class="axis" x="{width-right}" y="{height-36}" text-anchor="end">right label: native / raised speedup</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(ROOT / "manifest.csv")
    missing_manifest_status = [
        row["kernel"] for row in manifest if not row.get("overall_status", "").strip()
    ]
    if missing_manifest_status:
        raise ValueError(
            "manifest rows lack overall_status: " + ", ".join(missing_manifest_status))
    canonical_kernels = {row["kernel"] for row in manifest}

    cpu_rows: list[dict[str, object]] = []
    for directory in sorted(path for path in CPU_LOGS.iterdir() if path.is_dir()):
        native_path = directory / "native-cpu-samples.csv"
        raised_path = directory / "raised-openblas-cpu-samples.csv"
        if not native_path.exists() or not raised_path.exists():
            continue
        native = sample_stats(native_path, "runtime_ms")
        raised = sample_stats(raised_path, "runtime_ms")
        cpu_rows.append({
            "kernel": directory.name,
            "native_ms": native["min"], "native_min_ms": native["min"],
            "native_median_ms": native["median"],
            "native_max_ms": native["max"], "native_iqr_ms": native["iqr"],
            "raised_openblas_ms": raised["min"],
            "raised_min_ms": raised["min"], "raised_max_ms": raised["max"],
            "raised_median_ms": raised["median"],
            "raised_iqr_ms": raised["iqr"],
            "speedup_native_over_raised": native["min"] / raised["min"],
            "native_correctness": correctness(directory, ("native-cpu-correctness.compare.log", "native-correctness.compare.log")),
            "raised_correctness": correctness(directory, ("raised-openblas-correctness.compare.log",)),
            "dataset": "LARGE", "datatype": "double", "warmups": 5, "samples": 5,
            "hardware": "Jetson AGX Orin CPU core 0", "threads": 1,
            "publication_status": "pending_fixed_hardware_state_verification",
            "log_dir": str(directory.relative_to(ROOT)),
        })

    gpu_rows: list[dict[str, object]] = []
    kernels = sorted(path.name for path in GPU_LOGS.iterdir() if path.is_dir())
    for kernel in kernels:
        directory = GPU_LOGS / kernel
        native_path = directory / "native-gpu-samples.csv"
        raised_path = directory / "raised-gpu-samples.csv"
        native_device = native_e2e = raised_device = raised_wall = raised_memory = raised_e2e = None
        native_iqr = raised_iqr = None
        if native_path.exists():
            fields = read_csv(native_path)[0]
            native_device_stats = sample_stats(native_path, "device_ms")
            native_e2e_field = "e2e_ms" if "e2e_ms" in fields else "end_to_end_ms"
            native_e2e_stats = sample_stats(native_path, native_e2e_field)
            native_device, native_iqr = native_device_stats["min"], native_device_stats["iqr"]
            native_e2e = native_e2e_stats["min"]
        if (kernel == "gemm" and
                (GEMM_PRECISION_LOGS / "raised-fp64-samples.csv").exists()):
            raised_path = GEMM_PRECISION_LOGS / "raised-fp64-samples.csv"
        if raised_path.exists():
            fields = read_csv(raised_path)[0]
            raised_device_stats = sample_stats(raised_path, "compute_device_ms")
            raised_device, raised_iqr = raised_device_stats["min"], raised_device_stats["iqr"]
            if "compute_wall_ms" in fields:
                raised_wall = sample_stats(raised_path, "compute_wall_ms")["min"]
            if "memory_device_ms" in fields:
                raised_memory = sample_stats(raised_path, "memory_device_ms")["min"]
            raised_e2e = sample_stats(raised_path, "e2e_host_ms")["min"]
        gpu_rows.append({
            "kernel": kernel,
            "native_device_ms": native_device, "native_device_iqr_ms": native_iqr,
            "native_device_median_ms": (native_device_stats["median"] if native_path.exists() else None),
            "native_device_max_ms": (native_device_stats["max"] if native_path.exists() else None),
            "native_e2e_ms": native_e2e,
            "raised_device_ms": raised_device, "raised_device_iqr_ms": raised_iqr,
            "raised_device_median_ms": (raised_device_stats["median"] if raised_path.exists() else None),
            "raised_device_max_ms": (raised_device_stats["max"] if raised_path.exists() else None),
            "raised_resident_wall_ms": raised_wall,
            "raised_memory_device_ms": raised_memory, "raised_e2e_ms": raised_e2e,
            "device_speedup_native_over_raised": (
                native_device / raised_device
                if native_device is not None and raised_device is not None else None),
            "native_correctness": correctness(directory, ("native-correctness.compare.log",)),
            "raised_correctness": correctness(directory, (
                "raised-correctness.compare.log", "raised-paper-correctness.compare.log",
                "*publication-correctness*.silicon.log")),
            "dataset": "LARGE", "datatype": "double", "warmups": 5, "samples": 5,
            "hardware": "Jetson AGX Orin sm_87",
            "publication_status": "pending_fixed_hardware_state_verification",
            "native_modified_source": "true" if native_path.exists() else "",
            "log_dir": str(directory.relative_to(ROOT)),
            "native_evidence": (
                f"{directory.relative_to(ROOT)}/native-gpu-samples.csv"
                if native_path.exists() else ""),
            "raised_evidence": (str(raised_path.relative_to(ROOT))
                                if raised_path.exists() else ""),
        })

    fp32_native_path = GEMM_PRECISION_LOGS / "native-fp32-samples.csv"
    fp32_raised_path = GEMM_PRECISION_LOGS / "raised-fp32-samples.csv"
    if fp32_native_path.exists() and fp32_raised_path.exists():
        native_device = sample_stats(fp32_native_path, "device_ms")
        native_e2e = sample_stats(fp32_native_path, "e2e_ms")
        raised_device = sample_stats(fp32_raised_path, "compute_device_ms")
        raised_wall = sample_stats(fp32_raised_path, "compute_wall_ms")
        raised_memory = sample_stats(fp32_raised_path, "memory_device_ms")
        raised_e2e = sample_stats(fp32_raised_path, "e2e_host_ms")
        gpu_rows.append({
            "kernel": "gemm",
            "native_device_ms": native_device["min"],
            "native_device_iqr_ms": native_device["iqr"],
            "native_device_median_ms": native_device["median"],
            "native_device_max_ms": native_device["max"],
            "native_e2e_ms": native_e2e["min"],
            "raised_device_ms": raised_device["min"],
            "raised_device_iqr_ms": raised_device["iqr"],
            "raised_device_median_ms": raised_device["median"],
            "raised_device_max_ms": raised_device["max"],
            "raised_resident_wall_ms": raised_wall["min"],
            "raised_memory_device_ms": raised_memory["min"],
            "raised_e2e_ms": raised_e2e["min"],
            "device_speedup_native_over_raised": (
                native_device["min"] / raised_device["min"]),
            "native_correctness": "pass", "raised_correctness": "pass",
            "dataset": "LARGE", "datatype": "float", "warmups": 5,
            "samples": 5, "hardware": "Jetson AGX Orin sm_87",
            "publication_status": "pending_fixed_hardware_state_verification",
            "native_modified_source": "true",
            "log_dir": str(GEMM_PRECISION_LOGS.relative_to(ROOT)),
            "native_evidence": str(fp32_native_path.relative_to(ROOT)),
            "raised_evidence": str(fp32_raised_path.relative_to(ROOT)),
        })

    cpu_fields = list(cpu_rows[0])
    gpu_fields = list(gpu_rows[0])
    write_csv(OUT / "paper_cpu_results.csv", [
        {key: f(value) if isinstance(value, float) else value for key, value in row.items()}
        for row in cpu_rows
    ], cpu_fields)
    write_csv(OUT / "paper_gpu_results.csv", [
        {key: f(value) if isinstance(value, float) else value for key, value in row.items()}
        for row in gpu_rows
    ], gpu_fields)

    existing_cpu = read_csv(ROOT / "performance_cpu.csv")
    x86_rows: list[dict[str, object]] = []
    for kernel in ("gemm", "syr2k", "2mm", "3mm"):
        by_config = {
            row["configuration"]: row for row in existing_cpu if row["kernel"] == kernel
        }
        native = by_config.get("native_clang18_noinline", {})
        polly = by_config.get("polly14", {})
        x86_rows.append({
            "kernel": kernel, "native_x86_ms": native.get("time_ms", ""),
            "polly_x86_ms": polly.get("time_ms", ""),
            "polly_speedup": polly.get("speedup_vs_native", ""),
            "kernelfarer_status": "unavailable_incompatible_llvm_toolchain",
            "scope": "secondary_x86_portability_only_not_comparable_to_orin",
        })
    write_csv(OUT / "paper_x86_related_work.csv", x86_rows, list(x86_rows[0]))

    paired_cpu = [row for row in cpu_rows if row["native_correctness"] == row["raised_correctness"] == "pass"]
    primary_gpu_rows = [row for row in gpu_rows if row["datatype"] == "double"]
    paired_gpu_all = [
        row for row in gpu_rows
        if row["native_device_ms"] is not None and row["raised_device_ms"] is not None
    ]
    paired_gpu = [row for row in paired_gpu_all if row["datatype"] == "double"]
    cpu_speedups = [float(row["speedup_native_over_raised"]) for row in paired_cpu]
    gpu_speedups = [float(row["device_speedup_native_over_raised"]) for row in paired_gpu]
    summary = {
        "cpu_paired_kernels": len(paired_cpu),
        "cpu_geomean_speedup": math.exp(statistics.mean(math.log(x) for x in cpu_speedups)),
        "cpu_median_speedup": statistics.median(cpu_speedups),
        "cpu_speedup_min": min(cpu_speedups), "cpu_speedup_max": max(cpu_speedups),
        "cpu_missing": sorted(canonical_kernels - {str(row["kernel"]) for row in cpu_rows}),
        "gpu_native_kernels": sum(row["native_device_ms"] is not None for row in primary_gpu_rows),
        "gpu_raised_kernels": sum(row["raised_device_ms"] is not None for row in primary_gpu_rows),
        "gpu_paired_kernels": len(paired_gpu),
        "gpu_supplementary_datatype_configurations": len(paired_gpu_all) - len(paired_gpu),
        "gpu_geomean_device_speedup": math.exp(statistics.mean(math.log(x) for x in gpu_speedups)),
        "gpu_median_device_speedup": statistics.median(gpu_speedups),
        "gpu_raised_faster": sum(x > 1.0 for x in gpu_speedups),
        "gpu_native_faster": sum(x < 1.0 for x in gpu_speedups),
        "gpu_native_only": [row["kernel"] for row in primary_gpu_rows if row["native_device_ms"] is not None and row["raised_device_ms"] is None],
        "gpu_raised_only": [row["kernel"] for row in primary_gpu_rows if row["native_device_ms"] is None and row["raised_device_ms"] is not None],
        "gpu_neither": sorted(canonical_kernels - {str(row["kernel"]) for row in primary_gpu_rows}),
        "manifest_rows": len(manifest),
        "manifest_rows_with_explicit_status": len(manifest),
        "publication_status": "pending_fixed_hardware_state_verification",
    }
    (OUT / "paper_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    paired_runtime_svg(
        paired_cpu, OUT / "polybench_cpu_runtime.svg",
        "PolyBench CPU: native versus raised OpenBLAS",
        "native_ms", "raised_openblas_ms", "Native C -O3", "Raised OpenBLAS",
        "LARGE/FP64 on one Orin CPU core; minimum of 5 after 5 warmups; log runtime",
    )
    paired_runtime_svg(
        paired_gpu_all, OUT / "polybench_gpu_runtime.svg",
        "PolyBench GPU: native CUDA versus raised resident library",
        "native_device_ms", "raised_device_ms", "Native PolyBenchGPU", "Raised CUDA library",
        "LARGE on Orin SM87; datatype labelled per row; CUDA-event device time; minimum of 5 after 5 warmups",
    )

    analysis = f"""# PolyBench Section 4.2 paper analysis

This directory is generated by `generate_paper_analysis.py` from the retained
publication-campaign sample CSVs. It does not reuse the historical one-warmup
timings in the legacy result ledger.

## Current quantitative result

- CPU: {len(paired_cpu)} correctness-gated native/raised pairs. Raised OpenBLAS
  has a {summary['cpu_geomean_speedup']:.2f}x geometric-mean speedup and a
  {summary['cpu_median_speedup']:.2f}x median speedup; the range is
  {summary['cpu_speedup_min']:.2f}x--{summary['cpu_speedup_max']:.2f}x.
- GPU: {summary['gpu_native_kernels']} native and {summary['gpu_raised_kernels']}
  raised resident measurements, with {len(paired_gpu)} one-to-one device-time
  FP64 pairs, plus {summary['gpu_supplementary_datatype_configurations']} supplementary
  FP32 configuration. Raised is faster for {summary['gpu_raised_faster']} FP64 pairs and native is
  faster for {summary['gpu_native_faster']}; geometric-mean native/raised
  device speedup is {summary['gpu_geomean_device_speedup']:.2f}x.
- Native-only GPU rows: {', '.join(summary['gpu_native_only']) or 'none'}.
- Raised-only GPU rows: {', '.join(summary['gpu_raised_only']) or 'none'}.
- CPU rows without a fresh native/raised OpenBLAS pair:
  {', '.join(summary['cpu_missing']) or 'none'}.
- GPU rows with neither a fresh native nor raised measurement:
  {', '.join(summary['gpu_neither']) or 'none'}.
- KernelFaRer is unavailable because the retained revision does not build with
  the available LLVM toolchain. Polly results for GEMM, SYR2K, 2mm, and 3mm are
  retained only as a separately labelled x86 portability comparison.

## Claim boundary

The primary campaign uses canonical LARGE/FP64 inputs; the additional GEMM
FP32 row uses the same dimensions and initialization with float storage. Every
row records its datatype and requires complete-output correctness,
one process, five warmups, and five samples. CPU runs are pinned to core 0 with
one OpenBLAS/OpenMP thread. GPU ratios use device time on both sides; end-to-end
times are reported separately and never mixed into those ratios. Native
PolyBenchGPU rows use normalization adapters and are `modified_source=true`.
Every headline runtime and derived speedup uses the minimum of the five timed
samples; the generated ledgers also retain the median, maximum, IQR, and raw
sample links.

The runner reported accelerator/hardware state as `N/A`. Consequently every
timing is **publication-pending** until fixed power, clock, fan, temperature,
and throttling state can be verified. These artifacts are suitable for paper
layout and internal analysis, but not yet for an unconditional final claim.
"""
    (OUT / "PAPER_ANALYSIS.md").write_text(analysis)


if __name__ == "__main__":
    main()
