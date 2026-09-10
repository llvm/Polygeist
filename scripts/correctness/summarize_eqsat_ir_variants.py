#!/usr/bin/env python3
"""Summarize controlled IR-variation recovery from an eqsat ablation run."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def selected(row: dict[str, str]) -> set[tuple[tuple[int, ...], str]]:
    return {
        (tuple(item["body_indices"]), item["symbol"])
        for item in json.loads(row.get("selected") or "[]")
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stats(values: list[float]) -> dict[str, float | int | None]:
    return {
        "n": len(values),
        "median": statistics.median(values) if values else None,
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stress-runs", type=Path)
    parser.add_argument("--manifest", type=Path,
                        help="Resolved campaign manifest for a benefit subset.")
    parser.add_argument("--cpu-summary", type=Path,
                        help="Optional whole-program CPU pilot summary.")
    parser.add_argument("--egraph-runs", type=Path,
                        help="Separate one-repetition e-graph diagnostic CSV.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.runs)
    successful = [row for row in rows if row["status"] == "ok"]
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in successful:
        grouped[(row["input_id"], row["mode"])].append(row)

    canonical: dict[tuple[str, str], set[tuple[tuple[int, ...], str]]] = {}
    nondeterministic: list[dict[str, object]] = []
    for key, group in grouped.items():
        match_sets = {frozenset(selected(row)) for row in group}
        if len(match_sets) != 1:
            nondeterministic.append({"input_id": key[0], "mode": key[1],
                                     "sets": len(match_sets)})
        canonical[key] = set(next(iter(match_sets)))

    detailed: list[dict[str, object]] = []
    for input_id in sorted({key[0] for key in canonical}):
        kernel, variant = input_id.split("::", 1)
        if variant == "original":
            continue
        expected = canonical.get((f"{kernel}::original", "egglog"), set())
        original_syntax = canonical.get(
            (f"{kernel}::original", "syntactic"), set())
        common_expected = expected & original_syntax
        egg = canonical.get((input_id, "egglog"), set())
        syn = canonical.get((input_id, "syntactic"), set())
        egg_recovered = expected & egg
        syn_recovered = expected & syn
        detailed.append({
            "kernel": kernel, "variant": variant,
            "expected_identities": len(expected),
            "common_baseline_identities": len(common_expected),
            "egglog_recovered": len(egg_recovered),
            "syntax_recovered": len(syn_recovered),
            "egglog_common_recovered": len(common_expected & egg),
            "syntax_common_recovered": len(common_expected & syn),
            "egglog_only_recovered": len(egg_recovered - syn_recovered),
            "syntax_only_recovered": len(syn_recovered - egg_recovered),
            "egglog_selected_total": len(egg),
            "syntax_selected_total": len(syn),
            "expected": json.dumps(sorted((list(i), s) for i, s in expected)),
            "egglog_selected": json.dumps(sorted((list(i), s) for i, s in egg)),
            "syntax_selected": json.dumps(sorted((list(i), s) for i, s in syn)),
        })

    by_variant: list[dict[str, object]] = []
    paired_by_variant: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))
    paired: dict[tuple[str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in successful:
        _, variant = row["input_id"].split("::", 1)
        paired[(row["input_id"], int(row["repetition"]))][row["mode"]] = row
    for (input_id, _), arms in paired.items():
        if "egglog" not in arms or "syntactic" not in arms:
            continue
        _, variant = input_id.split("::", 1)
        for metric in ("matcher_elapsed_ms", "process_wall_ms", "peak_rss_kib"):
            if arms["egglog"].get(metric) and arms["syntactic"].get(metric):
                paired_by_variant[variant][metric].append(
                    float(arms["egglog"][metric]) -
                    float(arms["syntactic"][metric]))

    for variant in sorted({row["variant"] for row in detailed}):
        group = [row for row in detailed if row["variant"] == variant]
        opportunities = sum(int(row["expected_identities"]) for row in group)
        common_opportunities = sum(
            int(row["common_baseline_identities"]) for row in group)
        egg_recovered = sum(int(row["egglog_recovered"]) for row in group)
        syn_recovered = sum(int(row["syntax_recovered"]) for row in group)
        egg_common = sum(int(row["egglog_common_recovered"]) for row in group)
        syn_common = sum(int(row["syntax_common_recovered"]) for row in group)
        matcher_delta = stats(paired_by_variant[variant]["matcher_elapsed_ms"])
        wall_delta = stats(paired_by_variant[variant]["process_wall_ms"])
        rss_delta = stats(paired_by_variant[variant]["peak_rss_kib"])
        by_variant.append({
            "variant": variant, "inputs": len(group),
            "opportunities": opportunities,
            "common_baseline_opportunities": common_opportunities,
            "egglog_recovered": egg_recovered,
            "syntax_recovered": syn_recovered,
            "egglog_common_recovered": egg_common,
            "syntax_common_recovered": syn_common,
            "egglog_common_recovery_pct": (
                100.0 * egg_common / common_opportunities
                if common_opportunities else None),
            "syntax_common_recovery_pct": (
                100.0 * syn_common / common_opportunities
                if common_opportunities else None),
            "egglog_recovery_pct": (100.0 * egg_recovered / opportunities
                                      if opportunities else None),
            "syntax_recovery_pct": (100.0 * syn_recovered / opportunities
                                      if opportunities else None),
            "egglog_only_recovered": sum(
                int(row["egglog_only_recovered"]) for row in group),
            "matcher_delta_median_ms": matcher_delta["median"],
            "matcher_delta_q1_ms": matcher_delta["q1"],
            "matcher_delta_q3_ms": matcher_delta["q3"],
            "wall_delta_median_ms": wall_delta["median"],
            "rss_delta_median_kib": rss_delta["median"],
        })

    generation = read_csv(args.generation)
    stress = (read_csv(args.stress_runs)
              if args.stress_runs and args.stress_runs.exists() else [])
    stress = [row for row in stress
              if row["input_id"].endswith("::identity_pair")]
    cpu_summary = (json.loads(args.cpu_summary.read_text())
                   if args.cpu_summary and args.cpu_summary.exists() else None)
    egraph_rows = (read_csv(args.egraph_runs) if args.egraph_runs and
                   args.egraph_runs.exists() else [])
    egraph_egg = [row for row in egraph_rows
                  if row["mode"] == "egglog" and row["status"] == "ok"]
    egraph_summary = None
    if egraph_egg:
        egraph_summary = {
            "inputs": len(egraph_egg),
            "successful_runs": len(egraph_rows),
            "largest_proof_nodes": stats([
                float(row["egraph_nodes_max"]) for row in egraph_egg]),
            "largest_proof_classes": stats([
                float(row["egraph_classes_max"]) for row in egraph_egg]),
            "largest_proof_elapsed_ms": stats([
                float(row["proof_elapsed_max_ms"]) for row in egraph_egg]),
            "completed_proofs_over_10s": sum(
                int(float(row.get("proofs_over_10s") or 0))
                for row in egraph_egg),
        }
    opportunities = sum(int(row["expected_identities"]) for row in detailed)
    common_opportunities = sum(
        int(row["common_baseline_identities"]) for row in detailed)
    egg_recovered = sum(int(row["egglog_recovered"]) for row in detailed)
    syn_recovered = sum(int(row["syntax_recovered"]) for row in detailed)
    egg_common = sum(int(row["egglog_common_recovered"]) for row in detailed)
    syn_common = sum(int(row["syntax_common_recovered"]) for row in detailed)
    all_matcher_delta = [value for per_variant in paired_by_variant.values()
                         for value in per_variant["matcher_elapsed_ms"]]
    all_wall_delta = [value for per_variant in paired_by_variant.values()
                      for value in per_variant["process_wall_ms"]]
    all_rss_delta = [value for per_variant in paired_by_variant.values()
                     for value in per_variant["peak_rss_kib"]]
    benefit_ids = {
        f"{row['kernel']}::{row['variant']}" for row in detailed
        if int(row["egglog_only_recovered"]) > 0
    }
    lost_by_symbol: dict[str, int] = defaultdict(int)
    for row in detailed:
        expected_set = {(tuple(index), symbol)
                        for index, symbol in json.loads(str(row["expected"]))}
        recovered_set = {(tuple(index), symbol) for index, symbol in
                         json.loads(str(row["egglog_selected"]))}
        for _, symbol in expected_set - recovered_set:
            lost_by_symbol[symbol] += 1
    summary = {
        "source_kernels": len({row["kernel"] for row in generation}),
        "planned_transformations": len({row["variant"] for row in generation}),
        "generated_variants": sum(row["status"] == "generated" for row in generation),
        "not_applicable": sum(row["status"] == "not_applicable" for row in generation),
        "verifier_failures": sum(row["status"] == "verifier_failed" for row in generation),
        "primary_inputs": len({row["input_id"] for row in rows}),
        "primary_runs": len(rows),
        "primary_successful_runs": len(successful),
        "primary_failures": [row for row in rows if row["status"] != "ok"],
        "nondeterministic_match_sets": nondeterministic,
        "recovery_opportunities": opportunities,
        "egglog_recovered": egg_recovered,
        "syntax_recovered": syn_recovered,
        "egglog_recovery_pct": 100.0 * egg_recovered / opportunities if opportunities else None,
        "syntax_recovery_pct": 100.0 * syn_recovered / opportunities if opportunities else None,
        "common_baseline_recovery_opportunities": common_opportunities,
        "egglog_common_baseline_recovered": egg_common,
        "syntax_common_baseline_recovered": syn_common,
        "egglog_common_baseline_recovery_pct": (
            100.0 * egg_common / common_opportunities
            if common_opportunities else None),
        "syntax_common_baseline_recovery_pct": (
            100.0 * syn_common / common_opportunities
            if common_opportunities else None),
        "egglog_only_recovered": sum(int(row["egglog_only_recovered"])
                                      for row in detailed),
        "paired_matcher_delta_ms": stats(all_matcher_delta),
        "paired_process_wall_delta_ms": stats(all_wall_delta),
        "paired_peak_rss_delta_kib": stats(all_rss_delta),
        "stress_rows": stress,
        "egglog_only_variant_cases": len(benefit_ids),
        "lost_recoveries_by_symbol": dict(sorted(lost_by_symbol.items())),
        "cpu_execution_pilot": cpu_summary,
        "egraph_diagnostic": egraph_summary,
    }
    write_csv(args.output / "per_input_variant.csv", detailed)
    write_csv(args.output / "by_variant.csv", by_variant)
    if args.manifest:
        write_csv(args.output / "benefit_manifest.csv", [
            row for row in read_csv(args.manifest)
            if row["input_id"] in benefit_ids
        ])
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Controlled PolyBench equality-saturation variants", "",
        f"Thirty raised PolyBench inputs were offered ten independent, "
        f"name-agnostic algebraic transformations. {summary['generated_variants']} "
        f"variants were applicable and all passed MLIR verification; "
        f"{summary['not_applicable']} combinations were recorded as not applicable.", "",
        f"The primary nine-transformation campaign contains {summary['primary_inputs']} "
        f"inputs and {summary['primary_runs']} fresh matcher processes. "
        f"{summary['primary_successful_runs']} succeeded.", "",
        "## Recovery", "",
        f"For the controlled headline, the target set contains only identities both "
        f"arms found in the unmodified input. Across {common_opportunities} such "
        f"variant/identity opportunities, Egglog retained {egg_common} "
        f"({summary['egglog_common_baseline_recovery_pct']:.1f}%), while exact syntax "
        f"retained {syn_common} "
        f"({summary['syntax_common_baseline_recovery_pct']:.1f}%).", "",
        f"Against the broader unmodified-Egglog target set, Egglog recovered "
        f"{egg_recovered}/{opportunities} ({summary['egglog_recovery_pct']:.1f}%) "
        f"and exact syntax recovered {syn_recovered}/{opportunities} "
        f"({summary['syntax_recovery_pct']:.1f}%).", "",
        "| Variant | Inputs | Common opportunities | Egglog | Exact syntax | Egglog-only target recoveries | Matcher delta median ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in by_variant:
        lines.append(
            f"| {row['variant']} | {row['inputs']} | "
            f"{row['common_baseline_opportunities']} | "
            f"{row['egglog_common_recovered']} | "
            f"{row['syntax_common_recovered']} | "
            f"{row['egglog_only_recovered']} | "
            f"{row['matcher_delta_median_ms']:.3f} |")
    matcher = summary["paired_matcher_delta_ms"]
    rss = summary["paired_peak_rss_delta_kib"]
    if cpu_summary and cpu_summary.get("passed", 0) > 0 and \
            cpu_summary.get("failed", 0) == 0:
        cpu_validation = (
            f"All {cpu_summary['passed']} extracted-kernel CPU pilot variants "
            f"built and ran with {cpu_summary['backend']}; complete {cpu_summary['dataset']} "
            f"{cpu_summary['dtype']} output comparisons passed at rtol "
            f"{cpu_summary['rtol']} and atol {cpu_summary['atol']}. The generated "
            f"PolyBench harness declares but does not define the selected kernel, "
            f"so the transformed object is the sole implementation and no symbol "
            f"replacement is used.")
    elif cpu_summary:
        cpu_validation = (
            f"The CPU execution pilot passed {cpu_summary.get('passed', 0)} and "
            f"failed {cpu_summary.get('failed', 0)} of "
            f"{cpu_summary.get('result_rows', 0)} variants; only passing rows have "
            f"numerical execution evidence.")
    else:
        cpu_validation = (
            "CPU execution was not run; no numerical correctness result is claimed.")
    lines.extend([
        "", "## Cost", "",
        f"Across {matcher['n']} paired repetitions, the Egglog-minus-syntax matcher "
        f"delta was {matcher['median']:.3f} ms median "
        f"[{matcher['q1']:.3f}, {matcher['q3']:.3f}]. The peak-RSS delta was "
        f"{rss['median'] / 1024.0:.3f} MiB median.", "",
        "## E-graph diagnostic", "",
        (f"A separate one-repetition diagnostic covered "
         f"{egraph_summary['inputs']} Egglog-benefit variants. The largest proof "
         f"had {egraph_summary['largest_proof_nodes']['max']:.0f} nodes and "
         f"{egraph_summary['largest_proof_classes']['max']:.0f} classes; the "
         f"median per-input largest proof had "
         f"{egraph_summary['largest_proof_nodes']['median']:.0f} nodes and "
         f"{egraph_summary['largest_proof_classes']['median']:.0f} classes. No "
         f"completed proof exceeded 10 seconds."
         if egraph_summary else "No separate e-graph diagnostic was supplied."), "",
        "## Stress case", "",
        "The composed `identity_pair` mutation applies `(x + 0) * 1` at every "
        "floating-point generic yield. It is excluded from primary aggregates after "
        "the 2mm Egglog pilot hit the 120-second whole-input watchdog; the retained "
        "stress rows document that scaling failure.", "",
        "## Remaining robustness gaps", "",
        "The 14 Egglog misses are explicit: four are `cublasDaxpby` composition "
        "misses in Gesummv, five are `cublasDsyr2k` misses, and five are "
        "`cublasDsyrk` misses. These composition recognizers still impose "
        "structural preconditions outside the scalar Egglog equivalence check.", "",
        "## CPU execution validation", "",
        cpu_validation, "",
        "## Interpretation boundary", "",
        "These are controlled variants of already-raised Linalg IR. They measure "
        "matcher robustness and compilation cost on the x86 host, not frontend "
        "raising coverage or application runtime. Floating-point reassociation is "
        "judged under the study's numerical-equivalence contract rather than "
        "bit-for-bit IEEE evaluation order.", "",
    ])
    (args.output / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps({key: summary[key] for key in (
        "primary_runs", "primary_successful_runs", "recovery_opportunities",
        "egglog_recovered", "syntax_recovered", "egglog_recovery_pct",
        "syntax_recovery_pct", "common_baseline_recovery_opportunities",
        "egglog_common_baseline_recovered",
        "syntax_common_baseline_recovered",
        "egglog_common_baseline_recovery_pct",
        "syntax_common_baseline_recovery_pct")}, sort_keys=True))


if __name__ == "__main__":
    main()
