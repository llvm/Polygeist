#!/usr/bin/env python3
"""Summarize correctness-gated MFEM-derived application campaign logs."""

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

ENTRIES = [
    ("mtop_iso_elasticity_dfem_2d", "mfem_app_mtop_iso_elasticity_dfem_2d"),
    ("dfem_minimal_surface_2d", "mfem_app_dfem_minimal_surface_2d"),
    ("ex35p_h1_3d", "mfem_app_ex35p_h1_3d"),
    ("ex35p_hcurl_3d", "mfem_app_ex35p_hcurl_3d"),
    ("ex35p_hdiv_3d", "mfem_app_ex35p_hdiv_3d"),
    ("ex9p_mass_convection_2d", "mfem_app_ex9p_mass_convection_2d"),
    ("grad_div_3d", "mfem_app_grad_div_3d"),
    ("abs_l1_mass_3d", "mfem_app_abs_l1_mass_3d"),
    ("abs_l1_diffusion_3d", "mfem_app_abs_l1_diffusion_3d"),
    ("abs_l1_curlcurl_3d", "mfem_app_abs_l1_curlcurl_3d"),
    ("navier_tgv_pa_operators_3d", "mfem_app_navier_tgv_pa_operators_3d"),
]


def complete_processes(path: Path, required: set[str]) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(errors="replace")
    starts = list(re.finditer(r"^\[jetson\] running .* run \d+ \.\.\.$", text,
                              re.MULTILINE))
    results = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        segment = text[start.start():end]
        correctness = re.search(
            r"correctness=(PASS|FAIL) max_abs=([^ ]+) max_rel=([^ ]+)", segment
        )
        samples = {}
        for implementation in required:
            samples[implementation] = [
                float(value) for value in re.findall(
                    rf"implementation={implementation} sample=\d+ time_us=([0-9.]+)",
                    segment,
                )
            ]
        if (correctness and correctness.group(1) == "PASS" and
                all(len(samples[name]) == 20 for name in required)):
            results.append({
                "max_abs": float(correctness.group(2)),
                "max_rel": float(correctness.group(3)),
                "samples": samples,
            })
    return results


def median_of_process_medians(processes: list[dict], implementation: str) -> float:
    medians = [statistics.median(p["samples"][implementation]) for p in processes]
    return statistics.median(medians) if medians else math.nan


def build_counts(path: Path) -> tuple[str, str, str]:
    if not path.exists():
        return "false", "", ""
    text = path.read_text(errors="replace")
    launch = re.findall(r"matched (\d+) kernel\.launch", text)
    calls = re.findall(r"emitted (\d+) func\.call", text)
    complete = "build complete" in text
    return str(complete).lower(), launch[-1] if launch else "", calls[-1] if calls else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for ident, function in ENTRIES:
        gpu = complete_processes(
            args.campaign / "raw" / f"{ident}.deploy.log",
            {"vanilla_cpu", "raised_gpu"},
        )
        cpu_paths = [args.campaign / "raw-raised-cpu" / f"{ident}.deploy.log"]
        if ident == "navier_tgv_pa_operators_3d":
            cpu_paths.append(
                args.campaign / "raw-raised-cpu" /
                "navier_tgv_pa_operators_3d.resume3.deploy.log"
            )
        cpu = []
        for path in cpu_paths:
            cpu.extend(complete_processes(path, {"vanilla_cpu", "raised_cpu"}))
        gpu_build, launches, gpu_calls = build_counts(
            args.campaign / "build-logs" / f"{ident}.log"
        )
        cpu_build, cpu_launches, cpu_calls = build_counts(
            args.campaign / "build-logs-raised-cpu" / f"{ident}.log"
        )
        vanilla = median_of_process_medians(gpu, "vanilla_cpu")
        raised_gpu = median_of_process_medians(gpu, "raised_gpu")
        raised_cpu = median_of_process_medians(cpu, "raised_cpu")
        errors = [p["max_abs"] for p in gpu + cpu]
        rows.append({
            "id": ident,
            "function": function,
            "ne": 1024,
            "structural_launches": launches,
            "gpu_runtime_calls": gpu_calls,
            "cpu_runtime_calls": cpu_calls,
            "raised_gpu_build": gpu_build,
            "raised_cpu_build": cpu_build,
            "gpu_correct_processes": len(gpu),
            "cpu_correct_processes": len(cpu),
            "samples_per_process": 20,
            "vanilla_cpu_us": f"{vanilla:.9f}" if math.isfinite(vanilla) else "",
            "raised_cpu_us": f"{raised_cpu:.9f}" if math.isfinite(raised_cpu) else "",
            "raised_gpu_us": f"{raised_gpu:.9f}" if math.isfinite(raised_gpu) else "",
            "raised_cpu_over_vanilla": (
                f"{raised_cpu / vanilla:.9f}"
                if math.isfinite(raised_cpu) and math.isfinite(vanilla) else ""
            ),
            "raised_gpu_over_vanilla": (
                f"{raised_gpu / vanilla:.9f}"
                if math.isfinite(raised_gpu) and math.isfinite(vanilla) else ""
            ),
            "max_abs": f"{max(errors):.17g}" if errors else "",
            "correctness": (
                "PASS" if len(gpu) == 5 and len(cpu) == 5 else "PARTIAL"
            ),
            "native_mfem_status": "not rerun for derived application path",
        })
        if cpu_launches and launches and cpu_launches != launches:
            raise RuntimeError(f"structural launch mismatch for {ident}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"GPU complete: {sum(r['gpu_correct_processes'] == 5 for r in rows)}/11")
    print(f"CPU complete: {sum(r['cpu_correct_processes'] == 5 for r in rows)}/11")


if __name__ == "__main__":
    main()
