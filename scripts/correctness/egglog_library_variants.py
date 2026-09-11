#!/usr/bin/env python3
"""Measure Egglog equivalence over variants of the real MLIR kernel library.

The canonical expressions come only from linalg.generic bodies inside
kernel_library_phase2.mlir.  For each body we generate five equivalent forms
and run each proof in an isolated subprocess, allowing a real wall-clock
timeout to be reported instead of hanging the full audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from kernel_match import (  # noqa: E402
    EGraph,
    Term,
    _MLIR_SEMANTIC_SOURCE_NAMES,
    _ast_to_term,
    _kernel_defn_regions,
    _parse_term,
    _term_repr,
    algebra_rules,
    composition_library,
    encode_body_yields,
    parse_generics,
)


def map_ast(node, transform):
    if not isinstance(node, tuple) or not node:
        return node
    children = tuple(
        map_ast(child, transform) if isinstance(child, tuple) else child
        for child in node[1:]
    )
    return transform((node[0], *children))


def commute_all(node):
    return map_ast(
        node,
        lambda current: (
            (current[0], current[2], current[1])
            if current[0] in {"Add", "Mul"} and len(current) == 3
            else current
        ),
    )


def flatten(node, op: str) -> list:
    if isinstance(node, tuple) and len(node) == 3 and node[0] == op:
        return flatten(node[1], op) + flatten(node[2], op)
    return [node]


def fold(items: list, op: str, right: bool):
    if len(items) == 1:
        return items[0]
    if right:
        value = items[-1]
        for item in reversed(items[:-1]):
            value = (op, item, value)
        return value
    value = items[0]
    for item in items[1:]:
        value = (op, value, item)
    return value


def reassociate(node, right: bool):
    def transform(current):
        if current[0] not in {"Add", "Mul"} or len(current) != 3:
            return current
        return fold(flatten(current, current[0]), current[0], right)

    return map_ast(node, transform)


def node_count(node) -> int:
    if not isinstance(node, tuple):
        return 0
    return 1 + sum(node_count(child) for child in node[1:])


def variants(ast) -> list[tuple[str, object]]:
    """Five always-valid equivalent variants for every canonical body."""
    return [
        ("commute_all", commute_all(ast)),
        ("associate_left", reassociate(ast, right=False)),
        ("associate_right", reassociate(ast, right=True)),
        ("add_zero", ("Add", ast, ("Lit", 0.0))),
        ("mul_one", ("Mul", ("Lit", 1.0), ast)),
    ]


def worker() -> int:
    payload = json.load(sys.stdin)
    canonical = _ast_to_term(payload["canonical"])
    candidate = _ast_to_term(payload["candidate"])
    started = time.perf_counter()
    egraph = EGraph()
    egraph.register(canonical, candidate)
    report = egraph.run(algebra_rules() * payload["iterations"])
    try:
        egraph.check(canonical == candidate)
        status = "match"
    except Exception:
        status = "no_match"
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    print(json.dumps({
        "status": status,
        "elapsed_ms": elapsed_ms,
        "updated": bool(report.updated),
        "rule_matches": sum(report.num_matches_per_rule.values()),
    }))
    return 0


def run_case(case: dict, timeout_s: float, iterations: int) -> dict:
    payload = {
        "canonical": case.pop("canonical"),
        "candidate": case.pop("candidate"),
        "iterations": iterations,
    }
    started = time.perf_counter()
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker"],
            input=json.dumps(payload),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
        wall_ms = (time.perf_counter() - started) * 1000.0
        if result.returncode:
            return {**case, "status": "error", "proof_elapsed_ms": "",
                    "wall_elapsed_ms": f"{wall_ms:.3f}",
                    "rule_matches": "", "detail": result.stderr.strip()[:500]}
        report = json.loads(result.stdout)
        return {
            **case,
            "status": report["status"],
            "proof_elapsed_ms": f'{report["elapsed_ms"]:.3f}',
            "wall_elapsed_ms": f"{wall_ms:.3f}",
            "rule_matches": report["rule_matches"],
            "detail": "",
        }
    except subprocess.TimeoutExpired:
        wall_ms = (time.perf_counter() - started) * 1000.0
        return {**case, "status": "timeout", "proof_elapsed_ms": "",
                "wall_elapsed_ms": f"{wall_ms:.3f}",
                "rule_matches": "", "detail": f"wall timeout after {timeout_s}s"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path,
                        default=ROOT / "generic_solver/kernel_library_phase2.mlir")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "issues/egglog_library_variants.csv")
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--symbol", action="append",
                        help="audit only this kernel.defn symbol (repeatable)")
    parser.add_argument("--source", choices=("mlir", "production", "all"),
                        default="all", help="formula source(s) to audit")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker()

    cases: list[dict] = []
    semantic_definitions = 0
    formulas: list[tuple[str, str, str, str, Term]] = []
    if args.source in {"mlir", "all"}:
        text = args.library.read_text()
        for symbol, region in _kernel_defn_regions(text):
            if args.symbol and symbol not in args.symbol:
                continue
            bodies = parse_generics(region, infer_outputs_from_yield=True)
            if bodies:
                semantic_definitions += 1
            for body_index, body in enumerate(bodies):
                for yield_index, term in enumerate(encode_body_yields(body)):
                    formulas.append(("mlir", "mlir", symbol,
                                     f"{body_index}:{yield_index}", term))
    if args.source in {"production", "all"}:
        for entry_index, entry in enumerate(composition_library()):
            if args.symbol and entry.name not in args.symbol:
                continue
            for step_index, step in enumerate(entry.steps):
                terms = step.body_per_yield or [step.body]
                for yield_index, term in enumerate(terms):
                    origin = ("mlir" if entry.name in _MLIR_SEMANTIC_SOURCE_NAMES
                              else "python")
                    formulas.append(("production", origin, entry.name,
                                     f"{entry_index}:{step_index}:{yield_index}", term))

    for source, origin, symbol, body_index, term in formulas:
        canonical = _parse_term(_term_repr(term))
        for variant_name, candidate in variants(canonical):
            cases.append({
                "source": source,
                "canonical_origin": origin,
                "symbol": symbol,
                "body_index": body_index,
                "variant": variant_name,
                "canonical_nodes": node_count(canonical),
                "variant_nodes": node_count(candidate),
                "iterations": args.iterations,
                "timeout_s": args.timeout,
                "canonical": canonical,
                "candidate": candidate,
            })

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = [
            executor.submit(run_case, dict(case), args.timeout, args.iterations)
            for case in cases
        ]
        for index, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"completed {index}/{len(futures)}", flush=True)

    rows.sort(key=lambda row: (row["source"], row["symbol"],
                               row["body_index"], row["variant"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source", "canonical_origin", "symbol", "body_index", "variant", "canonical_nodes",
              "variant_nodes", "iterations", "timeout_s", "status", "proof_elapsed_ms",
              "wall_elapsed_ms", "rule_matches", "detail"]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    counts = {status: sum(row["status"] == status for row in rows)
              for status in ("match", "no_match", "timeout", "error")}
    print(
        f"definitions={semantic_definitions} formulas={len(formulas)} cases={len(cases)} "
        + " ".join(f"{key}={value}" for key, value in counts.items())
    )
    print(args.output)
    return 0 if not counts["error"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
