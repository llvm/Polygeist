#!/usr/bin/env python3
"""Generate controlled algebraic variants of raised MLIR inputs.

The transformations are deliberately independent and name-agnostic.  They
operate only on scalar arithmetic inside ``linalg.generic`` regions and never
use a benchmark or source-function name to choose a rewrite.  Each generated
file receives exactly one transformation kind. Swap and identity mutations
cover every applicable generic; reassociation changes one deterministic site.
This keeps recovery attributable without a combinatorial mutation search.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


VARIANTS = (
    "swap_add",
    "swap_mul",
    "swap_add_mul",
    "add_zero_rhs",
    "add_zero_lhs",
    "mul_one_rhs",
    "mul_one_lhs",
    "identity_pair",
    "reassociate_add",
    "reassociate_mul",
)

_ARITH_RE = re.compile(
    r"^(?P<indent>\s*)(?P<result>%[\w.$-]+)\s*=\s*"
    r"arith\.(?P<op>addf|mulf)\s+(?P<a>%[\w.$-]+)\s*,\s*"
    r"(?P<b>%[\w.$-]+)\s*:\s*(?P<type>\S+)\s*$"
)
_YIELD_RE = re.compile(
    r"^(?P<indent>\s*)linalg\.yield\s+(?P<value>%[\w.$-]+)\s*"
    r":\s*(?P<type>\S+)\s*$"
)


@dataclass(frozen=True)
class Mutation:
    text: str
    sites: int
    details: tuple[str, ...]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside_linalg_flags(lines: list[str]) -> list[bool]:
    """Return a conservative line mask for linalg.generic regions."""
    flags = [False] * len(lines)
    active = False
    depth = 0
    for index, line in enumerate(lines):
        if not active and "linalg.generic" in line:
            active = True
            depth = line.count("{") - line.count("}")
            flags[index] = True
            continue
        if active:
            flags[index] = True
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                active = False
                depth = 0
    return flags


def _fresh_prefix(text: str) -> str:
    prefix = "eqsat_variant"
    counter = 0
    while f"%{prefix}" in text:
        counter += 1
        prefix = f"eqsat_variant_{counter}"
    return prefix


def _swap(text: str, wanted: set[str]) -> Mutation:
    lines = text.splitlines()
    flags = _inside_linalg_flags(lines)
    details: list[str] = []
    for index, line in enumerate(lines):
        match = _ARITH_RE.match(line) if flags[index] else None
        if not match or match.group("op") not in wanted:
            continue
        groups = match.groupdict()
        lines[index] = (
            f"{groups['indent']}{groups['result']} = arith.{groups['op']} "
            f"{groups['b']}, {groups['a']} : {groups['type']}"
        )
        details.append(f"line {index + 1}: {groups['op']}")
    return Mutation("\n".join(lines) + "\n", len(details), tuple(details))


def _wrap_yields(text: str, kind: str) -> Mutation:
    lines = text.splitlines()
    flags = _inside_linalg_flags(lines)
    prefix = _fresh_prefix(text)
    # Constants must dominate the generic.  Keeping them outside its region
    # also makes their literal values visible to the matcher's scoped
    # constant analysis, instead of turning them into opaque local captures.
    generic_starts: dict[int, tuple[str, str, str, str]] = {}
    active_start: int | None = None
    for index, line in enumerate(lines):
        if "linalg.generic" in line and flags[index] and active_start is None:
            active_start = index
        match = _YIELD_RE.match(line) if flags[index] else None
        if active_start is not None and match:
            scalar_type = match.group("type")
            if scalar_type.startswith("f"):
                indent = re.match(r"\s*", lines[active_start]).group(0)
                zero = f"%{prefix}_zero_{active_start}"
                one = f"%{prefix}_one_{active_start}"
                generic_starts[active_start] = (indent, zero, one, scalar_type)
            active_start = None
    output: list[str] = []
    details: list[str] = []
    counter = 0
    for index, line in enumerate(lines):
        if index in generic_starts:
            outer_indent, zero, one, scalar_type = generic_starts[index]
            if kind in {"add_zero_rhs", "add_zero_lhs", "identity_pair"}:
                output.append(
                    f"{outer_indent}{zero} = arith.constant 0.000000e+00 : {scalar_type}")
            if kind in {"mul_one_rhs", "mul_one_lhs", "identity_pair"}:
                output.append(
                    f"{outer_indent}{one} = arith.constant 1.000000e+00 : {scalar_type}")
        match = _YIELD_RE.match(line) if flags[index] else None
        if not match or not match.group("type").startswith("f"):
            output.append(line)
            continue
        groups = match.groupdict()
        value = groups["value"]
        scalar_type = groups["type"]
        indent = groups["indent"]
        emitted: list[str] = []

        def binary(op: str, lhs: str, rhs: str) -> str:
            nonlocal counter
            name = f"%{prefix}_v{counter}"
            counter += 1
            emitted.append(
                f"{indent}{name} = arith.{op} {lhs}, {rhs} : {scalar_type}")
            return name

        # Find the constants associated with this enclosing generic.  The
        # nearest preceding start is unambiguous because generics do not nest.
        start = max(candidate for candidate in generic_starts if candidate < index)
        _, zero, one, _ = generic_starts[start]
        if kind == "add_zero_rhs":
            value = binary("addf", value, zero)
        elif kind == "add_zero_lhs":
            value = binary("addf", zero, value)
        elif kind == "mul_one_rhs":
            value = binary("mulf", value, one)
        elif kind == "mul_one_lhs":
            value = binary("mulf", one, value)
        elif kind == "identity_pair":
            value = binary("addf", value, zero)
            value = binary("mulf", one, value)
        else:
            raise ValueError(kind)
        output.extend(emitted)
        output.append(f"{indent}linalg.yield {value} : {scalar_type}")
        details.append(f"line {index + 1}: {scalar_type}")
    return Mutation("\n".join(output) + "\n", len(details), tuple(details))


def _reassociate(text: str, wanted: str) -> Mutation:
    """Rewrite the first ``(a op b) op c`` to ``a op (b op c)``."""
    lines = text.splitlines()
    flags = _inside_linalg_flags(lines)
    prefix = _fresh_prefix(text)
    definitions: dict[str, tuple[str, str, str]] = {}
    details: list[str] = []
    counter = 0
    for index, line in enumerate(lines):
        match = _ARITH_RE.match(line) if flags[index] else None
        if not match:
            continue
        groups = match.groupdict()
        if groups["op"] != wanted:
            definitions[groups["result"]] = (
                groups["op"], groups["a"], groups["b"])
            continue
        left = definitions.get(groups["a"])
        if left is None or left[0] != wanted:
            definitions[groups["result"]] = (
                groups["op"], groups["a"], groups["b"])
            continue
        nested = f"%{prefix}_assoc{counter}"
        counter += 1
        indent = groups["indent"]
        op_name = groups["op"]
        inserted = (
            f"{indent}{nested} = arith.{op_name} {left[2]}, "
            f"{groups['b']} : {groups['type']}"
        )
        replacement = (
            f"{indent}{groups['result']} = arith.{op_name} {left[1]}, "
            f"{nested} : {groups['type']}"
        )
        lines[index:index + 1] = [inserted, replacement]
        details.append(f"line {index + 1}: {op_name}")
        # One site per generic expression is enough; avoid indexing changes
        # and overlapping rewrites by clearing the local definition table.
        definitions.clear()
        break
    return Mutation("\n".join(lines) + "\n", len(details), tuple(details))


def mutate(text: str, variant: str) -> Mutation:
    if variant == "swap_add":
        return _swap(text, {"addf"})
    if variant == "swap_mul":
        return _swap(text, {"mulf"})
    if variant == "swap_add_mul":
        return _swap(text, {"addf", "mulf"})
    if variant.startswith("reassociate_"):
        return _reassociate(text, variant.removeprefix("reassociate_") + "f")
    return _wrap_yields(text, variant)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"suite", "input_id", "path"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"manifest must contain {sorted(required)}")
    return rows


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
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mlir-opt", type=Path,
                        default=Path("build/bin/polygeist-opt"))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=list(VARIANTS))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    variant_dir = args.output / "inputs"
    variant_dir.mkdir(exist_ok=True)
    generated: list[dict[str, object]] = []
    campaign: list[dict[str, object]] = []

    for item in read_manifest(args.manifest):
        source = Path(item["path"]).expanduser().resolve()
        if not source.is_file():
            raise SystemExit(f"missing input: {source}")
        source_text = source.read_text()
        source_hash = sha256(source)
        campaign.append({
            "suite": item["suite"],
            "input_id": f"{item['input_id']}::original",
            "path": str(source),
            "kernel": item["input_id"],
            "variant": "original",
        })
        for variant in args.variants:
            result = mutate(source_text, variant)
            status = "generated" if result.sites else "not_applicable"
            output_path = variant_dir / f"{item['input_id']}__{variant}.mlir"
            verifier_status = "not_run"
            verifier_stderr = ""
            if result.sites:
                output_path.write_text(result.text)
                completed = subprocess.run(
                    [str(args.mlir_opt.resolve()), str(output_path), "-o", "/dev/null"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                    check=False,
                )
                verifier_status = "pass" if completed.returncode == 0 else "fail"
                verifier_stderr = completed.stderr[-2000:]
                if completed.returncode == 0:
                    campaign.append({
                        "suite": item["suite"],
                        "input_id": f"{item['input_id']}::{variant}",
                        "path": str(output_path.resolve()),
                        "kernel": item["input_id"],
                        "variant": variant,
                    })
                else:
                    status = "verifier_failed"
            generated.append({
                "suite": item["suite"], "kernel": item["input_id"],
                "variant": variant, "status": status, "sites": result.sites,
                "source_path": str(source), "source_sha256": source_hash,
                "output_path": str(output_path.resolve()) if result.sites else "",
                "output_sha256": sha256(output_path) if output_path.is_file() else "",
                "verifier_status": verifier_status,
                "details": json.dumps(result.details),
                "verifier_stderr": verifier_stderr,
            })

    write_csv(args.output / "generation.csv", generated)
    write_csv(args.output / "manifest.csv", campaign)
    metadata = {
        "source_manifest": str(args.manifest.resolve()),
        "variants": args.variants,
        "source_inputs": len({row["kernel"] for row in generated}),
        "generated_variants": sum(row["status"] == "generated" for row in generated),
        "not_applicable": sum(row["status"] == "not_applicable" for row in generated),
        "verifier_failures": sum(row["status"] == "verifier_failed" for row in generated),
        "campaign_inputs": len(campaign),
        "mlir_verifier": str(args.mlir_opt.resolve()),
    }
    (args.output / "generation.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
