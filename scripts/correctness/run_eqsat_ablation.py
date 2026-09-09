#!/usr/bin/env python3
"""Run the sequential Egglog-vs-syntactic matcher ablation.

The input manifest is CSV with columns: suite,input_id,path. Every invocation
is a fresh process; modes alternate by repetition and no jobs run concurrently.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATCHER = ROOT / "scripts/correctness/kernel_match_rewrite.py"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def median(values):
    return statistics.median(values) if values else None


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def load_manifest(path: Path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"suite", "input_id", "path"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"manifest must contain {sorted(required)}")
    result = []
    seen = set()
    for row in rows:
        input_path = Path(row["path"]).expanduser().resolve()
        key = (row["suite"], row["input_id"])
        if key in seen:
            raise SystemExit(f"duplicate manifest identity: {key}")
        if not input_path.is_file():
            raise SystemExit(f"missing input: {input_path}")
        seen.add(key)
        result.append({**row, "path": str(input_path), "sha256": file_hash(input_path)})
    return result


def affinity_preexec(cpu: int):
    def configure():
        os.sched_setaffinity(0, {cpu})
    return configure


def run_one(item, mode, repetition, output_dir, timeout_s, cpu,
            collect_egraph_sizes):
    fd, telemetry_name = tempfile.mkstemp(
        prefix=f"{item['suite']}-{item['input_id']}-{mode}-",
        suffix=".json", dir=output_dir)
    os.close(fd)
    telemetry_path = Path(telemetry_name)
    command = [
        "/usr/bin/python3", str(MATCHER), item["path"], "--dry-run",
        "--matcher-mode", mode, "--telemetry-json", str(telemetry_path),
        "--disable-semantic-fallback",
    ]
    if collect_egraph_sizes:
        command.append("--collect-egraph-sizes")
    started = time.perf_counter()
    status = "ok"
    returncode = None
    stderr = ""
    telemetry = {}
    try:
        completed = subprocess.run(
            command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, timeout=timeout_s, check=False,
            preexec_fn=affinity_preexec(cpu),
        )
        returncode = completed.returncode
        stderr = completed.stderr
        if returncode:
            status = "error"
        elif telemetry_path.stat().st_size:
            telemetry = json.loads(telemetry_path.read_text())
        else:
            status = "missing_telemetry"
    except subprocess.TimeoutExpired as error:
        status = "input_timeout"
        stderr = error.stderr or ""
    finally:
        wall_ms = (time.perf_counter() - started) * 1000.0
        telemetry_path.unlink(missing_ok=True)
    row = {
        "suite": item["suite"], "input_id": item["input_id"],
        "path": item["path"], "sha256": item["sha256"],
        "mode": mode, "repetition": repetition, "status": status,
        "returncode": returncode, "process_wall_ms": wall_ms,
        "cpu": cpu, "command": json.dumps(command),
        "stderr_tail": stderr[-2000:],
    }
    for key, value in telemetry.items():
        row[key] = json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
    return row


def write_csv(path: Path, rows):
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, row):
    if path.exists():
        with path.open(newline="") as stream:
            columns = next(csv.reader(stream))
        unknown = set(row) - set(columns)
        if unknown:
            raise RuntimeError(f"new CSV fields during resume: {sorted(unknown)}")
        with path.open("a", newline="") as stream:
            csv.DictWriter(stream, fieldnames=columns,
                           lineterminator="\n").writerow(row)
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def selected_set(row):
    selected = json.loads(row.get("selected") or "[]")
    return {
        (tuple(entry["body_indices"]) if isinstance(entry["body_indices"], list)
         else (entry["body_indices"],), entry["symbol"])
        for entry in selected
    }


def summarize(rows, manifest, *, collect_egraph_sizes=False,
              input_timeout_s=120.0, cpu=None):
    successful = [row for row in rows if row["status"] == "ok"]
    by_input_mode = defaultdict(list)
    for row in successful:
        by_input_mode[(row["suite"], row["input_id"], row["mode"])].append(row)

    nondeterministic = []
    canonical = {}
    for key, group in by_input_mode.items():
        variants = {frozenset(selected_set(row)) for row in group}
        if len(variants) != 1:
            nondeterministic.append({"suite": key[0], "input_id": key[1], "mode": key[2]})
        canonical[key] = set(next(iter(variants))) if variants else set()

    suites = sorted({item["suite"] for item in manifest})
    suite_rows = []
    egglog_only_examples = []
    for suite in suites:
        all_ids = [item["input_id"] for item in manifest if item["suite"] == suite]
        ids = [input_id for input_id in all_ids
               if (suite, input_id, "egglog") in canonical
               and (suite, input_id, "syntactic") in canonical]
        egglog_matches = set()
        syntactic_matches = set()
        egglog_bodies = set()
        syntactic_bodies = set()
        for input_id in ids:
            for body_indices, symbol in canonical.get((suite, input_id, "egglog"), set()):
                egglog_matches.add((input_id, body_indices, symbol))
                egglog_bodies.update((input_id, index) for index in body_indices)
            for body_indices, symbol in canonical.get((suite, input_id, "syntactic"), set()):
                syntactic_matches.add((input_id, body_indices, symbol))
                syntactic_bodies.update((input_id, index) for index in body_indices)
        only = sorted(egglog_matches - syntactic_matches, key=repr)
        for value in only[:10]:
            egglog_only_examples.append({
                "suite": suite, "input_id": value[0],
                "body_indices": list(value[1]), "symbol": value[2],
            })
        suite_rows.append({
            "suite": suite, "inputs": len(all_ids),
            "common_success_inputs": len(ids),
            "egglog_selected_matches": len(egglog_matches),
            "syntactic_selected_matches": len(syntactic_matches),
            "egglog_matched_bodies": len(egglog_bodies),
            "syntactic_matched_bodies": len(syntactic_bodies),
            "egglog_only_match_identities": len(egglog_matches - syntactic_matches),
            "syntactic_only_match_identities": len(syntactic_matches - egglog_matches),
        })

    timing = {}
    for mode in ("egglog", "syntactic"):
        group = [row for row in successful if row["mode"] == mode]
        for metric in ("process_wall_ms", "matcher_elapsed_ms", "proof_elapsed_ms",
                       "peak_rss_kib"):
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            timing[f"{mode}_{metric}"] = {
                "median": median(values), "q1": percentile(values, .25),
                "q3": percentile(values, .75), "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
    paired_deltas = {}
    for metric in ("process_wall_ms", "matcher_elapsed_ms", "peak_rss_kib"):
        values = []
        groups = defaultdict(dict)
        for row in successful:
            if row.get(metric) not in (None, ""):
                groups[(row["suite"], row["input_id"], row["repetition"])][row["mode"]] = float(row[metric])
        for pair in groups.values():
            if "egglog" in pair and "syntactic" in pair:
                values.append(pair["egglog"] - pair["syntactic"])
        paired_deltas[metric] = {
            "pairs": len(values), "median": median(values),
            "q1": percentile(values, .25), "q3": percentile(values, .75),
        }
    proof_over_limit = sum(
        int(float(row.get("proofs_over_10s") or 0)) for row in successful)
    return {
        "parameters": {
            "repetitions_per_mode": max((int(r["repetition"]) for r in rows), default=0),
            "modes": ["egglog", "syntactic"], "parallel_jobs": 1,
            "egglog_iteration_limit": 8, "candidate_time_limit_s": 10,
            "candidate_timeout_enforcement": (
                "post-run classification; each fresh input process also has an outer watchdog"),
            "distributivity": False,
            "handwritten_semantic_fallback": False,
            "input_ast_node_ceiling": 32,
            "binding_proposal_limit": 8,
            "explicit_egraph_size_limit": None,
            "explicit_memory_limit": None,
            "outer_input_watchdog_s": input_timeout_s,
            "cpu_affinity": cpu,
            "python": "/usr/bin/python3",
            "egglog_version": "11.4.0",
            "collect_egraph_sizes": collect_egraph_sizes,
        },
        "manifest_inputs": len(manifest), "run_rows": len(rows),
        "successful_rows": len(successful),
        "failures": [
            {k: row[k] for k in ("suite", "input_id", "mode", "repetition", "status")}
            for row in rows if row["status"] != "ok"
        ],
        "nondeterministic_match_sets": nondeterministic,
        "by_suite": suite_rows,
        "egglog_only_examples": egglog_only_examples,
        "timing": timing,
        "paired_deltas": paired_deltas,
        "completed_proofs_over_10s": proof_over_limit,
    }


def write_markdown(path: Path, summary):
    lines = [
        "# Equality-saturation ablation", "",
        f"Inputs: {summary['manifest_inputs']}; runs: {summary['run_rows']}; "
        f"successful: {summary['successful_rows']}.", "",
        "## Match coverage", "",
        "Coverage below uses only inputs with at least one successful run in both arms; whole-input timeouts are reported separately.", "",
        "| Suite | Inputs | Common-success inputs | Egglog matches | Syntactic matches | Egglog bodies | Syntactic bodies | Egglog-only | Syntactic-only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["by_suite"]:
        lines.append(
            f"| {row['suite']} | {row['inputs']} | {row['common_success_inputs']} | {row['egglog_selected_matches']} | "
            f"{row['syntactic_selected_matches']} | {row['egglog_matched_bodies']} | "
            f"{row['syntactic_matched_bodies']} | {row['egglog_only_match_identities']} | "
            f"{row['syntactic_only_match_identities']} |")
    lines.extend(["", "## Performance", "",
                  "| Metric | Egglog median [Q1, Q3] | Syntactic median [Q1, Q3] | Paired Egglog - syntactic median [Q1, Q3] |",
                  "|---|---:|---:|---:|",])
    for metric, label in (("process_wall_ms", "Fresh-process wall time (ms)"),
                          ("matcher_elapsed_ms", "Matcher time (ms)"),
                          ("peak_rss_kib", "Peak RSS (KiB)")):
        egg = summary["timing"][f"egglog_{metric}"]
        syn = summary["timing"][f"syntactic_{metric}"]
        delta = summary["paired_deltas"][metric]
        lines.append(
            f"| {label} | {egg['median']:.3f} [{egg['q1']:.3f}, {egg['q3']:.3f}] | "
            f"{syn['median']:.3f} [{syn['q1']:.3f}, {syn['q3']:.3f}] | "
            f"{delta['median']:.3f} [{delta['q1']:.3f}, {delta['q3']:.3f}] |")
    lines.extend(["", f"Completed individual Egglog proofs over 10 seconds: {summary['completed_proofs_over_10s']}.",
                  "", "## Parameters", "", "```json",
                  json.dumps(summary["parameters"], indent=2), "```", "",
                  "## Egglog-only examples", ""])
    for example in summary["egglog_only_examples"]:
        lines.append(
            f"- `{example['suite']}/{example['input_id']}` bodies "
            f"`{example['body_indices']}` -> `{example['symbol']}`")
    if summary["failures"]:
        lines.extend(["", "## Failures", "", "```json",
                      json.dumps(summary["failures"], indent=2), "```"])
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--input-timeout", type=float, default=120.0)
    parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument("--collect-egraph-sizes", action="store_true",
                        help=("Collect graph nodes/classes; run separately "
                              "because serialization perturbs timing/RSS."))
    parser.add_argument("--limit", type=int,
                        help="Run only the first N manifest inputs (pilot use).")
    args = parser.parse_args()
    if args.repetitions < 1:
        raise SystemExit("--repetitions must be positive")
    if args.cpu not in os.sched_getaffinity(0):
        raise SystemExit(f"CPU {args.cpu} is outside the allowed affinity set")

    manifest = load_manifest(args.manifest)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        manifest = manifest[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.resolved.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    runs_path = args.output / "runs.csv"
    if runs_path.exists():
        with runs_path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        print(f"resuming with {len(rows)} checkpointed runs", flush=True)
    else:
        rows = []
    completed_keys = {
        (row["suite"], row["input_id"], row["mode"], int(row["repetition"]))
        for row in rows
    }
    total = len(manifest) * args.repetitions * 2
    completed_count = len(rows)
    for item_index, item in enumerate(manifest):
        for repetition in range(1, args.repetitions + 1):
            modes = (("egglog", "syntactic")
                     if (item_index + repetition) % 2 else
                     ("syntactic", "egglog"))
            for mode in modes:
                key = (item["suite"], item["input_id"], mode, repetition)
                if key in completed_keys:
                    continue
                row = run_one(
                    item, mode, repetition, args.output,
                    args.input_timeout, args.cpu, args.collect_egraph_sizes)
                rows.append(row)
                completed_count += 1
                append_csv(runs_path, row)
                print(
                    f"[{completed_count}/{total}] {item['suite']}/"
                    f"{item['input_id']} rep={repetition} mode={mode} "
                    f"status={row['status']} wall_ms={row['process_wall_ms']:.1f}",
                    flush=True)

    summary = summarize(
        rows, manifest, collect_egraph_sizes=args.collect_egraph_sizes,
        input_timeout_s=args.input_timeout, cpu=args.cpu)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(args.output / "SUMMARY.md", summary)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "inputs": len(manifest), "runs": len(rows),
        "failures": len(summary["failures"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
