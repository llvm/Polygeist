#!/usr/bin/env python3
"""Enumerate named ATen functions that contain local numerical iteration.

This complements aten_extraction_inventory.py.  The latter accounts for source
files; this manifest accounts for named bodies inside them so that a file with
twenty kernels cannot be considered complete after extracting only one.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from aten_extraction_inventory import DEFAULT_PYTORCH, DEFAULT_SOURCES, ROOT


DEFAULT_OUTPUT = ROOT / "issues/aten_c_kernels/operator_inventory.csv"
DISPATCH_INVENTORY = ROOT / "issues/aten_c_kernels/dispatch_kernel_inventory.csv"

CONTROL_NAMES = {"if", "for", "while", "switch", "catch"}
MANUAL_COVERED_HELPERS: dict[str, set[str]] = {
    "aten/src/ATen/native/cpu/ScatterGatherKernel.cpp": {"operator"},
    "aten/src/ATen/native/cpu/UpSampleKernel.cpp": {
        "eval", "is_zero_stride", "basic_loop_non_separable",
    },
    "aten/src/ATen/native/cpu/batch_norm_kernel.cpp": {
        "batch_norm_cpu_collect_linear_and_constant_terms",
        "batch_norm_cpu_collect_stats_contiguous_internal",
        "batch_norm_cpu_collect_stats_channels_last_internal",
        "batch_norm_cpu_backward_contiguous_internal",
        "batch_norm_cpu_backward_channels_last_internal",
    },
    "aten/src/ATen/native/cpu/int4mm_kernel.cpp": {
        "tinygemm_kernel", "tinygemm_kernel_",
    },
    "aten/src/ATen/native/cpu/int8mm_kernel.cpp": {
        "tinygemm_kernel", "tinygemm_kernel_",
    },
    "aten/src/ATen/native/cpu/DepthwiseConvKernel.cpp": {
        "convolution_depthwise3x3_winograd_impl",
    },
    "aten/src/ATen/native/FusedAdagrad.cpp": {"_fused_adagrad_kernel_cpu_"},
    "aten/src/ATen/native/FusedAdam.cpp": {
        "_fused_adam_kernel_cpu_", "_fused_adamw_kernel_cpu_",
    },
    "aten/src/ATen/native/FusedSGD.cpp": {"_fused_sgd_kernel_cpu_"},
}
FUNCTION_RE = re.compile(
    r"""
    (?:(?<=\n)|\A)
    (?P<header>
      (?:[ \t]*(?:template[ \t]*<[^;{}]+>|[A-Z_][A-Z0-9_]*\([^{}\n]*\))[ \t]*\n)*
      [ \t]*(?:(?:static|inline|constexpr|const|virtual|extern|C10_ALWAYS_INLINE)
      [ \t]+)*
      [A-Za-z_~][\w:<>,*& \t\n]*?
      [ \t]+(?P<name>[A-Za-z_~]\w*(?:::\w+)*)
      [ \t]*\([^;{}]*?\)
      [ \t]*(?:const[ \t]*)?(?:noexcept[ \t]*)?
    )\{
    """,
    re.VERBOSE,
)
TORCH_IMPL_RE = re.compile(
    r"(?:(?<=\n)|\A)[ \t]*TORCH_IMPL_FUNC\((?P<name>\w+)\)"
    r"[ \t]*\([^;{}]*?\)[ \t]*\{",
    re.DOTALL,
)


def mask_non_code(text: str) -> str:
    """Replace comments and string/char literals while retaining newlines."""
    pattern = re.compile(
        r"//[^\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
        re.DOTALL,
    )
    return pattern.sub(
        lambda match: "".join("\n" if c == "\n" else " " for c in match.group()),
        text,
    )


def matching_brace(text: str, opening: int) -> int:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index + 1
    return len(text)


def provenance() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return fixtures indexed by source and by source token."""
    from build_ce_viewer import ATEN_C_PROVENANCE

    by_source: dict[str, list[str]] = defaultdict(list)
    by_token: dict[str, list[str]] = defaultdict(list)
    for kernel, (source, token) in ATEN_C_PROVENANCE.items():
        by_source[source].append(kernel)
        if token:
            by_token[f"{source}\0{token}"].append(kernel)
    return by_source, by_token


def numerical_sites(body: str) -> tuple[int, int, int]:
    loops = len(re.findall(r"\b(?:for|while)\s*\(", body))
    tensor_iterator = len(
        re.findall(
            r"\b(?:cpu_kernel(?:_vec|_multiple_outputs)?|cpu_serial_kernel)"
            r"\s*\(",
            body,
        )
    )
    parallel = len(
        re.findall(r"\b(?:parallel_for|at::parallel_for|parallel_reduce)\s*\(", body)
    )
    return loops, tensor_iterator, parallel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--pytorch", type=Path, default=DEFAULT_PYTORCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    by_source, by_token = provenance()
    dispatch_roots: dict[str, set[str]] = defaultdict(set)
    if DISPATCH_INVENTORY.exists():
        with DISPATCH_INVENTORY.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if row["status"] == "EXTRACTED":
                    dispatch_roots[row["source"]].add(row["implementation"])
    rows: list[dict[str, object]] = []
    for source in filter(None, map(str.strip, args.sources.read_text().splitlines())):
        path = args.pytorch / source
        original = path.read_text(errors="replace")
        masked = mask_non_code(original)
        matches = list(FUNCTION_RE.finditer(masked)) + list(TORCH_IMPL_RE.finditer(masked))
        seen: set[tuple[str, int]] = set()
        functions: list[dict[str, object]] = []
        for match in sorted(matches, key=lambda item: item.start()):
            name = match.group("name")
            if name in CONTROL_NAMES:
                continue
            opening = masked.find("{", match.start(), match.end() + 1)
            if opening < 0:
                continue
            end = matching_brace(masked, opening)
            body = masked[opening:end]
            line = original.count("\n", 0, match.start()) + 1
            identity = (name, line)
            if identity in seen:
                continue
            seen.add(identity)
            functions.append(
                {
                    "name": name, "line": line, "body": body,
                    "header": original[match.start():opening],
                }
            )

        names = {str(function["name"]) for function in functions}
        roots = set(dispatch_roots.get(source, set()))
        for key in by_token:
            key_source, token = key.split("\0", 1)
            if key_source != source:
                continue
            for name in names:
                if token == name or name in token:
                    roots.add(name)
        reachable = set(roots)
        changed = True
        while changed:
            changed = False
            root_bodies = [
                str(function["body"])
                for function in functions
                if function["name"] in reachable
            ]
            joined = "\n".join(root_bodies)
            for name in names - reachable:
                if re.search(rf"\b{re.escape(name)}\s*(?:<[^;{{}}]*>)?\s*\(", joined):
                    reachable.add(name)
                    changed = True

        for function in functions:
            name = str(function["name"])
            line = int(function["line"])
            body = str(function["body"])
            header = str(function["header"])
            loops, tensor_iterator, parallel = numerical_sites(body)
            if not (loops or tensor_iterator or parallel):
                continue
            exact = []
            for key, fixtures in by_token.items():
                key_source, token = key.split("\0", 1)
                if key_source == source and (
                    token == name or token in header
                    or name in token
                ):
                    exact.extend(fixtures)
            if exact:
                status = "EXTRACTED"
            elif (
                name in reachable
                or name in MANUAL_COVERED_HELPERS.get(source, set())
            ):
                status = "COVERED_BY_EXTRACTED_ENTRY"
            else:
                status = "NEEDS_REVIEW"
            rows.append(
                {
                    "source": source,
                    "symbol": name,
                    "line": line,
                    "textual_loops": loops,
                    "tensor_iterator_sites": tensor_iterator,
                    "parallel_sites": parallel,
                    "exact_fixtures": ",".join(sorted(set(exact))),
                    "source_has_fixture": "yes" if by_source.get(source) else "no",
                    "status": status,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "source", "symbol", "line", "textual_loops", "tensor_iterator_sites",
        "parallel_sites", "exact_fixtures", "source_has_fixture", "status",
    )
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} named numerical bodies to {args.output}")
    print(f"exactly linked extractions: {sum(row['status'] == 'EXTRACTED' for row in rows)}")
    print("covered helper bodies: "
          f"{sum(row['status'] == 'COVERED_BY_EXTRACTED_ENTRY' for row in rows)}")
    print(f"needs review/extraction: {sum(row['status'] == 'NEEDS_REVIEW' for row in rows)}")


if __name__ == "__main__":
    main()
