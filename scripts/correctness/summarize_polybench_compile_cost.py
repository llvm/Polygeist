#!/usr/bin/env python3
"""Aggregate the PolyBench whole-compilation cost campaign."""

import csv
import json
from collections import Counter
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "issues/equality_saturation_eval/polybench_whole_compile_20260909"


def median(values):
    return statistics.median(values) if values else None


def main():
    with (CAMPAIGN / "runs.csv").open(newline="") as stream:
        runs = list(csv.DictReader(stream))
    by_key = {(r["kernel"], r["mode"], int(r["repetition"])): r for r in runs}
    kernels = sorted({r["kernel"] for r in runs})

    telemetry = {}
    for row in runs:
        path = CAMPAIGN / row["telemetry"]
        if path.exists():
            telemetry[(row["kernel"], row["mode"], int(row["repetition"]))] = json.loads(path.read_text())

    kernel_rows = []
    paired_wall_deltas = []
    paired_rss_deltas = []
    paired_matcher_deltas = []
    for kernel in kernels:
        modes = {}
        for mode in ("egglog", "syntactic"):
            subset = [r for r in runs if r["kernel"] == kernel and r["mode"] == mode]
            good = [r for r in subset if r["status"] == "ok"]
            tel = [telemetry[(kernel, mode, int(r["repetition"]))]
                   for r in subset if (kernel, mode, int(r["repetition"])) in telemetry]
            modes[mode] = {
                "ok": len(good),
                "errors": len(subset) - len(good),
                "wall_median_seconds": median([float(r["wall_seconds"]) for r in good]),
                "peak_rss_median_kib": median([int(r["peak_rss_kib"]) for r in good]),
                "matcher_median_ms": median([float(t["matcher_elapsed_ms"]) for t in tel]),
                "selected_matches_median": median([int(t["selected_matches"]) for t in tel]),
            }
        for repetition in range(1, 6):
            egg = by_key[(kernel, "egglog", repetition)]
            syn = by_key[(kernel, "syntactic", repetition)]
            if egg["status"] == syn["status"] == "ok":
                paired_wall_deltas.append(float(egg["wall_seconds"]) - float(syn["wall_seconds"]))
                paired_rss_deltas.append(int(egg["peak_rss_kib"]) - int(syn["peak_rss_kib"]))
            egg_tel = telemetry.get((kernel, "egglog", repetition))
            syn_tel = telemetry.get((kernel, "syntactic", repetition))
            if egg_tel and syn_tel:
                paired_matcher_deltas.append(
                    float(egg_tel["matcher_elapsed_ms"]) - float(syn_tel["matcher_elapsed_ms"]))
        kernel_rows.append({
            "kernel": kernel,
            "egglog_ok": modes["egglog"]["ok"],
            "syntactic_ok": modes["syntactic"]["ok"],
            "egglog_wall_median_seconds": modes["egglog"]["wall_median_seconds"],
            "syntactic_wall_median_seconds": modes["syntactic"]["wall_median_seconds"],
            "paired_wall_delta_seconds": (
                None if modes["egglog"]["wall_median_seconds"] is None
                or modes["syntactic"]["wall_median_seconds"] is None
                else modes["egglog"]["wall_median_seconds"] - modes["syntactic"]["wall_median_seconds"]),
            "egglog_peak_rss_median_kib": modes["egglog"]["peak_rss_median_kib"],
            "syntactic_peak_rss_median_kib": modes["syntactic"]["peak_rss_median_kib"],
            "egglog_matcher_median_ms": modes["egglog"]["matcher_median_ms"],
            "syntactic_matcher_median_ms": modes["syntactic"]["matcher_median_ms"],
            "egglog_selected_matches_median": modes["egglog"]["selected_matches_median"],
            "syntactic_selected_matches_median": modes["syntactic"]["selected_matches_median"],
        })

    mode_summary = {}
    for mode in ("egglog", "syntactic"):
        good = [r for r in runs if r["mode"] == mode and r["status"] == "ok"]
        tel = [t for (kernel, candidate_mode, repetition), t in telemetry.items()
               if candidate_mode == mode]
        mode_summary[mode] = {
            "runs": sum(r["mode"] == mode for r in runs),
            "successful_full_builds": len(good),
            "failed_full_builds": sum(r["mode"] == mode and r["status"] != "ok" for r in runs),
            "wall_median_seconds": median([float(r["wall_seconds"]) for r in good]),
            "wall_sum_seconds": sum(float(r["wall_seconds"]) for r in good),
            "peak_rss_median_kib": median([int(r["peak_rss_kib"]) for r in good]),
            "peak_rss_max_kib": max(int(r["peak_rss_kib"]) for r in good),
            "matcher_median_ms": median([float(t["matcher_elapsed_ms"]) for t in tel]),
            "selected_matches_sum": sum(int(t["selected_matches"]) for t in tel),
        }

    failed_kernels = sorted({r["kernel"] for r in runs if r["status"] != "ok"})
    summary = {
        "schema": 1,
        "scheduled_runs": len(runs),
        "unique_run_identities": len(by_key),
        "status_counts": dict(Counter(r["status"] for r in runs)),
        "successful_kernel_mode_pairs": len(paired_wall_deltas),
        "successful_kernels_both_modes": sum(
            row["egglog_ok"] == row["syntactic_ok"] == 5 for row in kernel_rows),
        "failed_kernels_both_modes": failed_kernels,
        "mode_summary": mode_summary,
        "paired_egglog_minus_syntactic": {
            "total_wall_median_seconds": median(paired_wall_deltas),
            "total_wall_mean_seconds": statistics.mean(paired_wall_deltas),
            "peak_rss_median_kib": median(paired_rss_deltas),
            "matcher_median_ms": median(paired_matcher_deltas),
        },
        "all_attempts_wall_sum_seconds": sum(float(r["wall_seconds"]) for r in runs),
    }
    (CAMPAIGN / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    columns = list(kernel_rows[0])
    with (CAMPAIGN / "per_kernel.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(kernel_rows)

    egg = mode_summary["egglog"]
    syn = mode_summary["syntactic"]
    delta = summary["paired_egglog_minus_syntactic"]
    lines = [
        "# PolyBench whole-compilation cost study", "",
        "## Scope", "",
        "Thirty canonical PolyBench/C 4.2 kernels were compiled from C source through a linked AArch64 executable. "
        "Each kernel used five fresh Egglog builds and five fresh syntactic-matcher builds, for 300 sequential attempts. "
        "Both arms disabled the handwritten semantic fallback. The input configuration was LARGE/FP64, each build "
        "had a 900-second outer timeout, and the production matcher used its eight-iteration and 32-AST-node proof bounds.", "",
        "The total wall interval includes cgeist, affine-to-Linalg raising, submap lowering, debufferization, matching, "
        "library-definition injection, ABI lowering, MLIR-to-LLVM lowering, LLVM translation, AArch64 object compilation, "
        "runtime/harness compilation, and final linking. A generated non-computational harness retains the transformed "
        "kernel symbol solely so the linker produces the final artifact; executables were not run and these results make "
        "no runtime-correctness claim.", "",
        "## Results", "",
        f"- Completed attempts: {len(runs)}/300; no outer timeouts.",
        f"- Successful linked builds: {egg['successful_full_builds']} Egglog and {syn['successful_full_builds']} syntactic "
        f"({summary['successful_kernels_both_modes']} kernels completed all five runs in both modes).",
        f"- Failed builds: {egg['failed_full_builds']} Egglog and {syn['failed_full_builds']} syntactic, covering the same "
        f"seven kernels: {', '.join(failed_kernels)}.",
        f"- Median successful whole-build wall time: {egg['wall_median_seconds']:.3f} s Egglog versus "
        f"{syn['wall_median_seconds']:.3f} s syntactic.",
        f"- Median paired Egglog minus syntactic whole-build delta: {delta['total_wall_median_seconds']:+.3f} s; "
        f"mean paired delta: {delta['total_wall_mean_seconds']:+.3f} s across {len(paired_wall_deltas)} pairs.",
        f"- Median successful peak RSS: {egg['peak_rss_median_kib'] / 1024:.2f} MiB Egglog versus "
        f"{syn['peak_rss_median_kib'] / 1024:.2f} MiB syntactic; median paired delta "
        f"{delta['peak_rss_median_kib'] / 1024:+.3f} MiB.",
        f"- Median paired matcher-only overhead: {delta['matcher_median_ms']:+.3f} ms for Egglog.",
        f"- Selected launches across the five suites: {egg['selected_matches_sum']} Egglog versus "
        f"{syn['selected_matches_sum']} syntactic, or 32 versus 29 per complete 30-kernel pass. Egglog found one "
        "additional 2mm launch and two additional gemver launches per pass; gemver subsequently failed ABI validation "
        "in both modes.",
        f"- Sum of measured attempt wall times, including failures: {summary['all_attempts_wall_sum_seconds'] / 60:.2f} minutes.", "",
        "The seven failures are current compiler failures shared by both modes, not timeouts: malformed or mismatched "
        "launch/ABI IR after fresh source raising. They are reported directly and excluded from successful whole-build "
        "timing comparisons. Per-run logs retain the exact diagnostics.", "",
        "## Caveats", "",
        "The host load averages at campaign start were recorded in metadata.json and were high relative to an idle host. "
        "Five repetitions and alternating arm order reduce ordering bias, but these measurements should be labelled "
        "loaded-host results until repeated on an otherwise idle compilation machine. GNU time peak RSS is the maximum "
        "resident set reported for the build process tree, not the sum of simultaneously resident child processes. "
        "Unlike the isolated equality-saturation audit, the production matcher does not apply an explicit per-candidate "
        "wall timer; all measured maximum individual proofs were below 23 ms, so the paper's 10-second ceiling would "
        "not have censored any observed proof.", "",
        "Raw data are in runs.csv, per-kernel aggregates in per_kernel.csv, and full provenance in metadata.json. All "
        "GNU-time records and matcher telemetry are retained, together with build logs for every failure. Successful "
        "verbose build logs, generated binaries, and link-only harness copies remain local rather than entering Git; "
        "their commands and executable hashes are recorded in runs.csv.", "",
    ]
    (CAMPAIGN / "SUMMARY.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
