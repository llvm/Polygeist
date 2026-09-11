#!/usr/bin/env python3
"""Generate dependency-free SVG figures for the MFEM Section 4.2 campaign."""

import csv
import html
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rows():
    with (ROOT / "performance_20260908.csv").open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_runtime(data):
    width, height = 1500, 760
    left, right, top, bottom = 105, 30, 48, 225
    plot_w, plot_h = width - left - right, height - top - bottom
    fields = [
        ("vanilla_cpu_ms", "Vanilla C / CPU", "#0969da"),
        ("raised_cpu_ms", "Polygeist raised / CPU", "#8250df"),
        ("raised_gpu_ms", "Polygeist raised / GPU", "#d97706"),
        ("native_mfem_gpu_ms", "Native MFEM / GPU", "#1a7f37"),
    ]
    y_min, y_max = 0.01, 10000.0

    def y(value):
        position = ((math.log10(value) - math.log10(y_min)) /
                    (math.log10(y_max) - math.log10(y_min)))
        return top + plot_h * (1.0 - position)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#24292f}'
        '.axis{font-size:15px}.label{font-size:14px}.title{font-size:22px;font-weight:700}'
        '.grid{stroke:#d8dee8;stroke-width:1}.pair{stroke:#8c959f;stroke-width:1;opacity:.65}'
        '</style>',
        f'<text class="title" x="{width/2}" y="28" text-anchor="middle">'
        'MFEM NE=1024 resident operator runtime</text>',
    ]
    for exponent in range(-2, 5):
        runtime = 10.0 ** exponent
        yp = y(runtime)
        out.extend([
            f'<line class="grid" x1="{left}" y1="{yp:.2f}" x2="{width-right}" y2="{yp:.2f}"/>',
            f'<text class="axis" x="{left-12}" y="{yp+5:.2f}" text-anchor="end">{runtime:g} ms</text>',
        ])
    group = plot_w / len(data)
    offsets = (-17, -6, 6, 17)
    for index, row in enumerate(data):
        center = left + group * (index + 0.5)
        points = []
        for (field, label, color), offset in zip(fields, offsets):
            runtime = float(row[field])
            xp, yp = center + offset, y(runtime)
            points.append(f"{xp:.2f},{yp:.2f}")
            out.append(
                f'<circle cx="{xp:.2f}" cy="{yp:.2f}" r="6" fill="{color}">'
                f'<title>{html.escape(label)}: {runtime:.6f} ms</title></circle>'
            )
        out.append(f'<polyline class="pair" points="{" ".join(points)}" fill="none"/>')
        out.append(
            f'<text class="label" transform="translate({center+5:.2f},{top+plot_h+15}) '
            f'rotate(56)" text-anchor="start">{html.escape(row["kernel"])}</text>'
        )
    out.append(
        f'<text class="axis" x="22" y="{top+plot_h/2}" '
        f'transform="rotate(-90 22 {top+plot_h/2})" text-anchor="middle">'
        'runtime (ms, logarithmic scale; lower is better)</text>'
    )
    legend_x = 300
    for index, (_, label, color) in enumerate(fields):
        x = legend_x + index * 280
        out.append(f'<circle cx="{x}" cy="{height-26}" r="6" fill="{color}"/>')
        out.append(f'<text class="axis" x="{x+12}" y="{height-20}">{html.escape(label)}</text>')
    out.append('</svg>')
    (ROOT / "mfem_four_runtime_log.svg").write_text("\n".join(out))


def write_slowdown(data):
    ordered = sorted(
        data,
        key=lambda row: float(row["raised_gpu_slowdown_vs_native_mfem"]),
        reverse=True,
    )
    width, height = 1300, 790
    left, right, top, bottom = 285, 120, 62, 52
    plot_w, plot_h = width - left - right, height - top - bottom
    maximum = max(float(row["raised_gpu_slowdown_vs_native_mfem"]) for row in ordered)
    row_h = plot_h / len(ordered)
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#24292f}'
        '.axis{font-size:14px}.value{font-size:14px;font-weight:700}'
        '.title{font-size:22px;font-weight:700}.grid{stroke:#d8dee8;stroke-width:1}</style>',
        f'<text class="title" x="{width/2}" y="29" text-anchor="middle">'
        'Polygeist raised GPU slowdown versus native MFEM CUDA</text>',
    ]
    for tick in (0, 50, 100, 150, 200):
        x = left + tick / maximum * plot_w
        out.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}"/>')
        out.append(f'<text class="axis" x="{x:.2f}" y="{height-bottom+24}" text-anchor="middle">{tick}×</text>')
    for index, row in enumerate(ordered):
        ratio = float(row["raised_gpu_slowdown_vs_native_mfem"])
        y = top + row_h * index + row_h * 0.16
        bar_h = row_h * 0.68
        bar_w = ratio / maximum * plot_w
        out.append(
            f'<text class="axis" x="{left-12}" y="{y+bar_h*0.68:.2f}" '
            f'text-anchor="end">{html.escape(row["kernel"])}</text>'
        )
        out.append(f'<rect x="{left}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" rx="3" fill="#d97706"/>')
        out.append(f'<text class="value" x="{left+bar_w+8:.2f}" y="{y+bar_h*0.68:.2f}">{ratio:.2f}×</text>')
    out.append('</svg>')
    (ROOT / "mfem_raised_gpu_vs_native.svg").write_text("\n".join(out))


def main():
    data = rows()
    if len(data) != 17:
        raise SystemExit(f"expected 17 performance rows, found {len(data)}")
    write_runtime(data)
    write_slowdown(data)
    print("generated mfem_four_runtime_log.svg and mfem_raised_gpu_vs_native.svg")


if __name__ == "__main__":
    main()
