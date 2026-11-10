#!/usr/bin/env python3
"""Generate a dependency-free SVG comparing FP32/FP64 PolyBench raising."""

from __future__ import annotations

import csv
import html
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FP32_CSV = ROOT / "b300_fp32_2026-09-09" / "summary.csv"
FP64_CSV = ROOT / "b300_fp64_2026-09-09" / "summary.csv"
OUTPUT = ROOT / "paper_analysis" / "polybench_fp32_fp64_kernel_comparison.svg"

KERNELS = [
    "2mm", "3mm", "atax", "bicg", "covariance", "doitgen", "gemm",
    "gemver", "gesummv", "mvt", "symm", "syr2k", "syrk", "trisolv",
    "trmm",
]

OPERATIONS = {
    "2mm": "2× GEMM + zero",
    "3mm": "3× GEMM",
    "atax": "2× GEMV + zero",
    "bicg": "2× GEMV + zero",
    "covariance": "mean-center + SYRK",
    "doitgen": "repeated GEMVᵀ + copy",
    "gemm": "GEMM",
    "gemver": "GER2 + 2× GEMV + AXPBY",
    "gesummv": "2× GEMV + AXPBY",
    "mvt": "2× GEMV",
    "symm": "SYMM",
    "syr2k": "SYR2K",
    "syrk": "SYRK",
    "trisolv": "TRSV",
    "trmm": "TRMM semantic (TRMV + SCAL backend)",
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt_ms(value: float) -> str:
    if value < 1:
        return f"{value:.3f} ms"
    if value < 10:
        return f"{value:.2f} ms"
    return f"{value:.1f} ms"


def text(x: int, y: int, value: str, css: str, anchor: str = "start") -> str:
    return (f'<text x="{x}" y="{y}" class="{css}" text-anchor="{anchor}">'
            f'{html.escape(value)}</text>')


def draw() -> None:
    fp32 = {row["kernel"]: row for row in rows(FP32_CSV)}
    fp64 = {row["kernel"]: row for row in rows(FP64_CSV)}
    missing32 = [kernel for kernel in KERNELS if kernel not in fp32]
    missing64 = [kernel for kernel in KERNELS if kernel not in fp64]
    if missing32 or missing64:
        raise RuntimeError(f"missing rows: FP32={missing32}, FP64={missing64}")

    width, height = 1800, 1530
    top, header_y, row_top, row_h = 55, 165, 275, 74
    parts = [f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<style>
  text {{ font-family: Inter, "Helvetica Neue", Arial, sans-serif; }}
  .title {{ font-size: 38px; font-weight: 750; fill: #172033; }}
  .subtitle {{ font-size: 19px; fill: #526072; }}
  .head {{ font-size: 18px; font-weight: 700; fill: white; letter-spacing: .3px; }}
  .headsub {{ font-size: 15px; fill: #b8c4d4; }}
  .kernel {{ font-size: 18px; font-weight: 700; fill: #172033; }}
  .op {{ font-size: 16px; fill: #263449; }}
  .time64 {{ font-size: 17px; font-weight: 700; fill: #2563eb; }}
  .time32 {{ font-size: 17px; font-weight: 700; fill: #ea580c; }}
  .foot {{ font-size: 15px; fill: #334155; }}
  .note {{ font-size: 14px; fill: #64748b; }}
</style>
<rect width="100%" height="100%" fill="#f7f8fa"/>
''']
    parts += [
        text(70, top + 35, "PolyBench library raising: FP64 and FP32", "title"),
        text(70, top + 76,
             "Same B300 GPU • LARGE dataset • correctness-gated • minimum of 5 runs",
             "subtitle"),
        f'<rect x="60" y="{header_y}" width="1680" height="88" rx="14" fill="#172033"/>',
        text(92, header_y + 38, "KERNEL", "head"),
        text(92, header_y + 66, "FP32 speedup", "headsub"),
        text(350, header_y + 34, "FP64  ·  NVIDIA B300", "head"),
        text(350, header_y + 65, "raised operation  |  device minimum", "headsub"),
        text(1070, header_y + 34, "FP32  ·  NVIDIA B300", "head"),
        text(1070, header_y + 65, "raised operation  |  device minimum", "headsub"),
    ]

    for index, kernel in enumerate(KERNELS):
        y = row_top + index * row_h
        fill = "#ffffff" if index % 2 == 0 else "#f0f3f7"
        parts.append(f'<rect x="60" y="{y}" width="1680" height="{row_h}" fill="{fill}"/>')
        parts.append(f'<line x1="60" y1="{y + row_h}" x2="1740" y2="{y + row_h}" stroke="#dce1e8"/>')
        baseline = y + 47
        t64 = float(fp64[kernel]["device_min_ms"])
        t32 = float(fp32[kernel]["device_min_ms"])
        ratio = t64 / t32
        parts += [
            text(92, baseline - 8, kernel.upper(), "kernel"),
            text(92, baseline + 16, f"{ratio:.2f}×", "note"),
            f'<circle cx="322" cy="{baseline - 6}" r="7" fill="#15803d"/>',
            text(350, baseline, OPERATIONS[kernel], "op"),
            text(960, baseline, fmt_ms(t64), "time64", "end"),
            f'<circle cx="1042" cy="{baseline - 6}" r="7" fill="#15803d"/>',
            text(1070, baseline, OPERATIONS[kernel], "op"),
            text(1707, baseline, fmt_ms(t32), "time32", "end"),
        ]

    foot_y = row_top + len(KERNELS) * row_h + 48
    parts += [
        f'<circle cx="76" cy="{foot_y - 6}" r="7" fill="#15803d"/>',
        text(95, foot_y, "correctness pass", "foot"),
        text(290, foot_y, "FP64 and FP32: B300 device minimum", "foot"),
        text(70, foot_y + 34,
             "FP32 speedup = FP64 device minimum / FP32 device minimum; values below 1× mean FP32 was slower.",
             "note"),
        "</svg>",
    ]
    OUTPUT.write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    draw()
