#!/usr/bin/env python3
"""Generate the PolyBench kernel-by-kernel related-work reproduction matrix."""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper_analysis" / "polybench_related_work_kernel_comparison.svg"
KERNELS = [
    "2mm", "3mm", "atax", "bicg", "correlation", "covariance", "doitgen",
    "gemm", "gemver", "gesummv", "mvt", "symm", "syr2k", "syrk",
    "trisolv", "trmm",
]

# B300 related-work minima, normalized to milliseconds. These are retained in
# /mnt/forge1/arpitj/results/b300-master-results.csv and summarized in the B300
# data report. A blank means the artifact did not provide that kernel/config.
PB_GPU_FP64 = {
    "2mm": 8.471, "3mm": 8.711, "atax": 3.026, "bicg": 3.259,
    "correlation": 160.415, "covariance": 171.699, "doitgen": 10.322,
    "gemm": 8.641, "gemver": 6.010, "gesummv": 2.711, "mvt": 3.881,
}
PPCG_FP64 = {
    "2mm": 675.308, "3mm": 647.504, "atax": 664.182, "bicg": 662.086,
    "correlation": 662.844, "covariance": 639.464, "doitgen": 1008.874,
    "gemm": 624.870, "gemver": 672.272, "gesummv": 658.738,
    "mvt": 665.828, "symm": 671.319, "syr2k": 647.484,
    "syrk": 706.898, "trisolv": 649.653, "trmm": 680.545,
}
PPCG_FP32 = {
    "2mm": 684.064, "3mm": 653.458, "atax": 633.067, "bicg": 650.371,
    "correlation": 650.572, "covariance": 640.529, "doitgen": 1109.785,
    "gemm": 634.596, "gemver": 623.967, "gesummv": 652.977,
    "mvt": 666.262, "symm": 649.639, "syr2k": 636.441,
    "syrk": 678.432, "trisolv": 705.638, "trmm": 692.698,
}
PB_ACC_FP32_STANDARD = {
    "2mm": 1.760, "3mm": 0.527, "atax": 1.664, "bicg": 1.830,
    "correlation": 791.578, "covariance": 791.420, "doitgen": 5.805,
    "gemm": 0.217, "gemver": 1.730, "gesummv": 1.512, "mvt": 1.656,
    "syr2k": 7.966, "syrk": 4.142,
}

# Reproduced x86 FP32/LARGE comparison. A loop-placement correction prevents
# complete BLAS calls from being inserted in the following loop's header for
# multi-stage programs. Full outputs pass for all four programs.
KERNELFARER = {
    "gemm": 26.172,
    "syr2k": 30.144,
    "2mm": 35.114,
    "3mm": 57.626,
}
KERNELFARER_PARTIAL = set()
PLUTO = {
    "doitgen": 290.895, "atax": 6.142, "3mm": 358.458, "mvt": 18.992,
    "bicg": 7.033, "2mm": 1519.563, "gemm": 1175.589,
    "gesummv": 2.478, "gemver": 17.628, "covariance": 120.006,
    "correlation": 120.617, "syrk": 82.194, "syr2k": 235.539,
    "trmm": 109.043,
}
POLLY = {
    "doitgen": 641.520, "atax": 1.914, "3mm": 7212.046, "mvt": 16.846,
    "bicg": 6.157, "2mm": 4474.442, "gemm": 3625.827,
    "gesummv": 1.139, "gemver": 24.803, "covariance": 4958.991,
    "correlation": 4979.915, "symm": 6723.084, "syrk": 149.899,
    "syr2k": 273.636, "trmm": 704.498,
}
MLIRSYNTH_SET = set(PLUTO) | {"symm"}


def read_minima(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["kernel"]: float(row["e2e_min_ms"])
                for row in csv.DictReader(handle) if row["status"] == "pass"}


def read_column(path: Path, column: str) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["kernel"]: float(row[column])
                for row in csv.DictReader(handle) if row["status"] == "pass"}


def esc(value: str) -> str:
    return html.escape(value)


def fmt(value: float) -> str:
    if value < 1:
        return f"{value:.3f}"
    if value < 10:
        return f"{value:.2f}"
    if value < 100:
        return f"{value:.1f}"
    return f"{value:.0f}"


def draw() -> None:
    ours64 = read_minima(ROOT / "b300_fp64_2026-09-09" / "summary.csv")
    ours32 = read_minima(ROOT / "b300_fp32_2026-09-09" / "summary.csv")
    ours_cpu32 = read_column(
        ROOT / "paper_analysis" / "polybench_cpu_raised_fp32_x86_min5.csv",
        "min_ms")
    polybenchgpu32 = read_column(
        ROOT / "b300_polybenchgpu_fp32_2026-09-09" / "summary.csv",
        "e2e_min_ms")
    columns = [
        ("Polygeist", "FP64 LARGE", ours64, "ours"),
        ("PolyBenchGPU", "FP64 LARGE", PB_GPU_FP64, "pass"),
        ("PPCG", "FP64 LARGE", PPCG_FP64, "pass"),
        ("Polygeist", "FP32 LARGE", ours32, "ours"),
        ("PPCG", "FP32 LARGE", PPCG_FP32, "pass"),
        ("PolyBenchGPU", "FP32 LARGE†", polybenchgpu32, "secondary"),
        ("PolyBench-ACC", "FP32 STANDARD*", PB_ACC_FP32_STANDARD, "different"),
        ("Polygeist", "FP32 CPU LARGE", ours_cpu32, "ours"),
        ("KernelFaRer", "FP32 LARGE", KERNELFARER, "cpu"),
        ("Pluto", "FP32 LARGE", PLUTO, "cpu"),
        ("Polly", "FP32 LARGE", POLLY, "cpu"),
        ("mlirSynth", "FP32 advertised", {}, "pending"),
    ]

    width, height = 3750, 2080
    left, kernel_w, col_w = 50, 250, 282
    title_y, group_y, head_y, row_y, row_h = 65, 190, 240, 365, 91
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: Inter, "Helvetica Neue", Arial, sans-serif; }}
.title {{ font-size: 48px; font-weight: 750; fill: #172033; }}
.sub {{ font-size: 23px; fill: #526072; }}
.group {{ font-size: 21px; font-weight: 750; fill: white; letter-spacing: .5px; }}
.head {{ font-size: 17px; font-weight: 700; fill: white; }}
.head2 {{ font-size: 14px; fill: #bec8d6; }}
.kernel {{ font-size: 19px; font-weight: 750; fill: #172033; }}
.value {{ font-size: 18px; font-weight: 700; fill: #17452c; }}
.missing {{ font-size: 20px; fill: #9aa4b2; }}
.pending {{ font-size: 15px; font-weight: 700; fill: #8a5a00; }}
.fail {{ font-size: 15px; font-weight: 700; fill: #a32929; }}
.foot {{ font-size: 17px; fill: #334155; }}
.note {{ font-size: 15px; fill: #64748b; }}
</style>
<rect width="100%" height="100%" fill="#f7f8fa"/>''']
    parts.append(f'<text x="70" y="{title_y}" class="title">PolyBench related-work reproduction matrix</text>')
    parts.append(f'<text x="70" y="{title_y + 48}" class="sub">Minimum runtime (ms), kernel by kernel • only correctness-passing measurements are shown</text>')

    gpu_x = left + kernel_w
    cpu_x = gpu_x + 7 * col_w
    parts += [
        f'<rect x="{gpu_x}" y="{group_y}" width="{7 * col_w}" height="43" rx="10" fill="#1f4f78"/>',
        f'<text x="{gpu_x + 18}" y="{group_y + 29}" class="group">B300 GPU reproductions</text>',
        f'<rect x="{cpu_x}" y="{group_y}" width="{5 * col_w}" height="43" rx="10" fill="#594477"/>',
        f'<text x="{cpu_x + 18}" y="{group_y + 29}" class="group">x86 CPU related work</text>',
        f'<rect x="{left}" y="{head_y}" width="{kernel_w + len(columns) * col_w}" height="105" rx="12" fill="#172033"/>',
        f'<text x="{left + 25}" y="{head_y + 61}" class="head">KERNEL</text>',
    ]
    for index, (name, detail, _, _) in enumerate(columns):
        x = left + kernel_w + index * col_w + col_w / 2
        parts.append(f'<text x="{x}" y="{head_y + 43}" text-anchor="middle" class="head">{esc(name)}</text>')
        parts.append(f'<text x="{x}" y="{head_y + 72}" text-anchor="middle" class="head2">{esc(detail)}</text>')

    for ri, kernel in enumerate(KERNELS):
        y = row_y + ri * row_h
        fill = "#ffffff" if ri % 2 == 0 else "#eef2f6"
        parts.append(f'<rect x="{left}" y="{y}" width="{kernel_w + len(columns) * col_w}" height="{row_h}" fill="{fill}"/>')
        parts.append(f'<line x1="{left}" y1="{y + row_h}" x2="{left + kernel_w + len(columns) * col_w}" y2="{y + row_h}" stroke="#d8dee7"/>')
        parts.append(f'<text x="{left + 25}" y="{y + 56}" class="kernel">{esc(kernel.upper())}</text>')
        for ci, (name, _, data, kind) in enumerate(columns):
            x0 = left + kernel_w + ci * col_w
            cx = x0 + col_w / 2
            if kernel in data:
                partial = name == "KernelFaRer" and kernel in KERNELFARER_PARTIAL
                cell_fill = ("#fff0d8" if partial or kind == "different"
                             else "#dff5e8")
                parts.append(f'<rect x="{x0 + 8}" y="{y + 10}" width="{col_w - 16}" height="{row_h - 20}" rx="9" fill="{cell_fill}"/>')
                if partial:
                    parts.append(f'<text x="{cx}" y="{y + 36}" text-anchor="middle" class="value">{fmt(data[kernel])} ms</text>')
                    parts.append(f'<text x="{cx}" y="{y + 62}" text-anchor="middle" class="pending">PARTIAL REPLACEMENT</text>')
                else:
                    parts.append(f'<circle cx="{cx - 55}" cy="{y + 46}" r="7" fill="#15803d"/>')
                    parts.append(f'<text x="{cx + 8}" y="{y + 53}" text-anchor="middle" class="value">{fmt(data[kernel])} ms</text>')
            elif kind == "pending" and kernel in MLIRSYNTH_SET:
                parts.append(f'<rect x="{x0 + 8}" y="{y + 10}" width="{col_w - 16}" height="{row_h - 20}" rx="9" fill="#fff1c9"/>')
                parts.append(f'<text x="{cx}" y="{y + 52}" text-anchor="middle" class="pending">TOOLCHAIN PENDING</text>')
            elif kind == "cpu" and name == "Pluto" and kernel == "symm":
                parts.append(f'<rect x="{x0 + 8}" y="{y + 10}" width="{col_w - 16}" height="{row_h - 20}" rx="9" fill="#fde4e4"/>')
                parts.append(f'<text x="{cx}" y="{y + 52}" text-anchor="middle" class="fail">CORRECTNESS FAIL</text>')
            else:
                parts.append(f'<text x="{cx}" y="{y + 54}" text-anchor="middle" class="missing">—</text>')

    fy = row_y + len(KERNELS) * row_h + 52
    parts += [
        f'<circle cx="75" cy="{fy - 6}" r="7" fill="#15803d"/>',
        f'<text x="95" y="{fy}" class="foot">correctness pass</text>',
        f'<rect x="290" y="{fy - 22}" width="34" height="24" rx="5" fill="#fff0d8"/>',
        f'<text x="340" y="{fy}" class="foot">different dataset (STANDARD, not LARGE)</text>',
        f'<text x="70" y="{fy + 39}" class="note">GPU timing scopes differ: Polygeist includes focused-function staging; PPCG includes fresh-process allocation/transfers/context; native artifacts use their source timing regions.</text>',
        f'<text x="70" y="{fy + 68}" class="note">CPU results are from the separate x86 FP32/LARGE reproduction. mlirSynth synthesis is reproduced separately; its public artifact does not include the paper\'s LARGE-FP32 XLA runtime path.</text>',
        f'<text x="70" y="{fy + 97}" class="note">KernelFaRer 2mm and 3mm use corrected loop placement; complete outputs pass and the executed paths contain two and three non-nested CBLAS calls, respectively.</text>',
        f'<text x="70" y="{fy + 126}" class="note">† Native PolyBenchGPU FP32 and Polygeist FP32 ran on different physical B300 workers; values are secondary coverage evidence, not strict same-device speedups.</text>',
        f'<text x="70" y="{fy + 155}" class="note">x86 KernelFaRer, Pluto, and Polly use one warmup plus ten measurements; x86 Polygeist uses five warmups plus five measurements.</text>',
        "</svg>",
    ]
    OUTPUT.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    draw()
