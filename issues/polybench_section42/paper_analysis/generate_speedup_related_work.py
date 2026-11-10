#!/usr/bin/env python3
"""Generate backend-normalized PolyBench related-work speedup charts."""

from __future__ import annotations

import csv
import html
from pathlib import Path

from generate_related_work_comparison import (
    KERNELFARER, KERNELFARER_PARTIAL, PB_ACC_FP32_STANDARD, PB_GPU_FP64, PLUTO, POLLY,
    PPCG_FP32, PPCG_FP64,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper_analysis" / "polybench_fp32_fp64_related_work_speedup.svg"
KERNELS = [
    "2mm", "3mm", "atax", "bicg", "correlation", "covariance", "doitgen",
    "gemm", "gemver", "gesummv", "mvt", "symm", "syr2k", "syrk",
    "trisolv", "trmm",
]
X86_NATIVE = {
    "doitgen": 643.541, "atax": 2.312, "3mm": 6477.873, "mvt": 41.633,
    "bicg": 2.759, "2mm": 3967.405, "gemm": 224.002,
    "gesummv": 1.422, "gemver": 46.557, "covariance": 8475.479,
    "correlation": 8208.152,
    "symm": 6573.945, "syrk": 1923.069, "syr2k": 10417.888,
    "trmm": 4612.607,
}
CAP = 500.0


def read_values(path: Path, value: str) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["kernel"]: float(row[value]) for row in csv.DictReader(handle)
                if row["status"] == "pass"}


def speedups(native: dict[str, float], candidate: dict[str, float]) -> dict[str, float]:
    return {kernel: (native[kernel] / runtime - 1.0) * 100.0
            for kernel, runtime in candidate.items() if kernel in native}


def label(value: float) -> str:
    if abs(value) >= 1000:
        return f"{value / 1000:.1f}k%"
    return f"{value:+.0f}%"


def draw() -> None:
    ours64 = read_values(ROOT / "b300_fp64_2026-09-09" / "summary.csv", "e2e_min_ms")
    ours32 = read_values(ROOT / "b300_fp32_2026-09-09" / "summary.csv", "e2e_min_ms")
    ours_cpu32 = read_values(
        ROOT / "paper_analysis" / "polybench_cpu_raised_fp32_x86_min5.csv",
        "min_ms")
    polybenchgpu32 = read_values(
        ROOT / "b300_polybenchgpu_fp32_2026-09-09" / "summary.csv",
        "e2e_min_ms")

    panels = [
        ("FP64 · B300", "speedup over native CUDA (PolyBenchGPU)",
         [("Polygeist raised", speedups(PB_GPU_FP64, ours64), "#1769e0", "normal"),
          ("PPCG", speedups(PB_GPU_FP64, PPCG_FP64), "#8b5cf6", "normal")]),
        ("FP32 · B300", "speedup over native CUDA (PolyBenchGPU)",
         [("Polygeist raised†", speedups(polybenchgpu32, ours32), "#e85d04", "normal"),
          ("PPCG", speedups(polybenchgpu32, PPCG_FP32), "#8b5cf6", "normal"),
          ("PolyBench-ACC*", {}, "#d09a00", "incompatible")]),
        ("FP32 · x86 CPU", "speedup over native FP32 C",
         [("Polygeist raised", speedups(X86_NATIVE, ours_cpu32), "#e85d04", "normal"),
          ("KernelFaRer", speedups(X86_NATIVE, KERNELFARER), "#c83e4d", "normal"),
          ("Pluto", speedups(X86_NATIVE, PLUTO), "#0088a8", "normal"),
          ("Polly", speedups(X86_NATIVE, POLLY), "#667085", "normal"),
          ("mlirSynth", {}, "#d09a00", "pending")]),
    ]

    width, height = 3780, 2405
    margin, panel_gap, panel_w = 45, 35, 1205
    title_y, panel_y, plot_y, row_h = 68, 180, 405, 101
    kernel_w, plot_left_pad, plot_right_pad = 175, 120, 88
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
text {{ font-family: Inter, "Helvetica Neue", Arial, sans-serif; }}
.title {{ font-size: 48px; font-weight: 760; fill: #172033; }}
.subtitle {{ font-size: 23px; fill: #526072; }}
.panel {{ font-size: 28px; font-weight: 750; fill: #172033; }}
.panelsub {{ font-size: 17px; fill: #64748b; }}
.kernel {{ font-size: 17px; font-weight: 700; fill: #263449; }}
.tick {{ font-size: 14px; fill: #64748b; }}
.legend {{ font-size: 15px; font-weight: 650; fill: #334155; }}
.barlabel {{ font-size: 13px; font-weight: 750; fill: #172033; }}
.xmark {{ font-size: 23px; font-weight: 800; fill: #c53030; }}
.dagger {{ font-size: 23px; font-weight: 800; fill: #b7791f; }}
.foot {{ font-size: 16px; fill: #475569; }}
</style>
<rect width="100%" height="100%" fill="#f7f8fa"/>''']
    parts += [
        f'<text x="65" y="{title_y}" class="title">PolyBench C: speedup over backend-native baseline</text>',
        f'<text x="65" y="{title_y + 48}" class="subtitle">GPU: native CUDA baseline • CPU: native C baseline • retained minimum; protocol differences footnoted • higher is better</text>',
    ]

    for pi, (panel_title, panel_sub, series) in enumerate(panels):
        px = margin + pi * (panel_w + panel_gap)
        plot_x = px + kernel_w + plot_left_pad
        plot_w = panel_w - kernel_w - plot_left_pad - plot_right_pad
        def sx(value: float) -> float:
            return plot_x + (value + 100.0) / 600.0 * plot_w

        parts += [
            f'<rect x="{px}" y="{panel_y}" width="{panel_w}" height="1900" rx="18" fill="#ffffff" stroke="#d8dee7"/>',
            f'<text x="{px + 28}" y="{panel_y + 47}" class="panel">{html.escape(panel_title)}</text>',
            f'<text x="{px + 28}" y="{panel_y + 78}" class="panelsub">{html.escape(panel_sub)}</text>',
        ]
        # Compact legend, wrapped after two entries.
        for si, (name, _, color, _) in enumerate(series):
            lx = px + 30 + (si % 2) * 555
            ly = panel_y + 116 + (si // 2) * 31
            parts.append(f'<rect x="{lx}" y="{ly - 13}" width="23" height="12" rx="3" fill="{color}"/>')
            parts.append(f'<text x="{lx + 33}" y="{ly}" class="legend">{html.escape(name)}</text>')

        axis_y = plot_y - 26
        for tick in (-100, 0, 100, 200, 300, 400, 500):
            x = sx(tick)
            strong = tick == 0
            parts.append(f'<line x1="{x}" y1="{axis_y}" x2="{x}" y2="{plot_y + len(KERNELS) * row_h}" stroke="{"#263449" if strong else "#dbe1e8"}" stroke-width="{2 if strong else 1}"/>')
            parts.append(f'<text x="{x}" y="{axis_y - 10}" text-anchor="middle" class="tick">{tick:+d}%</text>')

        count = len(series)
        bar_h = min(17, 64 / count)
        gap = 4
        group_h = count * bar_h + (count - 1) * gap
        for ri, kernel in enumerate(KERNELS):
            y = plot_y + ri * row_h
            if ri % 2:
                parts.append(f'<rect x="{px + 10}" y="{y}" width="{panel_w - 20}" height="{row_h}" fill="#f5f7fa"/>')
            parts.append(f'<text x="{px + 24}" y="{y + 57}" class="kernel">{html.escape(kernel.upper())}</text>')
            base_y = y + (row_h - group_h) / 2
            for si, (name, values, color, kind) in enumerate(series):
                by = base_y + si * (bar_h + gap)
                if kernel in values:
                    actual = values[kernel]
                    clipped = max(-100.0, min(CAP, actual))
                    x0, x1 = sx(0), sx(clipped)
                    left, bw = min(x0, x1), max(2.5, abs(x1 - x0))
                    partial = name == "KernelFaRer" and kernel in KERNELFARER_PARTIAL
                    partial_attrs = (
                        f' stroke="{color}" stroke-width="2" '
                        'stroke-dasharray="5 3"'
                        if partial else ""
                    )
                    parts.append(
                        f'<rect x="{left}" y="{by}" width="{bw}" '
                        f'height="{bar_h}" rx="3" fill="{color}" '
                        f'opacity="{0.48 if partial else 0.91}"'
                        f'{partial_attrs}/>'
                    )
                    anchor = "end" if clipped < 0 else "start"
                    tx = x1 - 7 if clipped < 0 else x1 + 7
                    if actual > CAP:
                        tx, anchor = sx(CAP) - 5, "end"
                    parts.append(f'<text x="{tx}" y="{by + bar_h - 3}" text-anchor="{anchor}" class="barlabel">{label(actual)}</text>')
                elif kind == "incompatible" and kernel in PB_ACC_FP32_STANDARD:
                    parts.append(f'<text x="{sx(-65)}" y="{by + bar_h}" text-anchor="middle" class="dagger">†</text>')
                else:
                    parts.append(f'<text x="{sx(-65)}" y="{by + bar_h}" text-anchor="middle" class="xmark">×</text>')

    foot_y = 2215
    parts += [
        f'<text x="65" y="{foot_y}" class="foot">Speedup (%) = (backend-native minimum / candidate minimum − 1) × 100. Bars are capped at +500%; labels report the actual value.</text>',
        f'<text x="65" y="{foot_y + 31}" class="foot">× = no comparable measured result. † = PolyBench-ACC exists, but uses STANDARD rather than LARGE and is not assigned a speedup.</text>',
        f'<text x="65" y="{foot_y + 62}" class="foot">B300 panels use precision-matched native PolyBenchGPU CUDA as the 0% baseline. The x86 panel uses native FP32/LARGE C on the CPU as its baseline.</text>',
        f'<text x="65" y="{foot_y + 93}" class="foot">KernelFaRer 2mm and 3mm use corrected loop placement; complete outputs pass and the executed paths contain two and three non-nested CBLAS calls, respectively.</text>',
        f'<text x="65" y="{foot_y + 124}" class="foot">† Native PolyBenchGPU FP32 and Polygeist FP32 ran on different physical B300 workers. Their ratios are secondary cross-worker evidence, not strict same-device speedups.</text>',
        f'<text x="65" y="{foot_y + 155}" class="foot">The x86 KernelFaRer, Pluto, and Polly series use one warmup plus ten measurements; x86 Polygeist uses five warmups plus five measurements. They are secondary comparisons.</text>',
        "</svg>",
    ]
    OUTPUT.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    draw()
