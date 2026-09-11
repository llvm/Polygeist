#!/usr/bin/env python3
"""Inventory concrete CPU dispatch kernels in the pinned 224-file census."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from aten_extraction_inventory import DEFAULT_PYTORCH, DEFAULT_SOURCES, ROOT


DEFAULT_OUTPUT = ROOT / "issues/aten_c_kernels/dispatch_kernel_inventory.csv"

REGISTER_RE = re.compile(
    r"(?:REGISTER_DISPATCH|ALSO_REGISTER_AVX512_DISPATCH)\s*\(\s*"
    r"(?P<stub>[A-Za-z_]\w*)\s*,\s*"
    r"(?:&(?:(?:CPU_CAPABILITY|DEFAULT)::)?)?"
    r"(?P<impl>[A-Za-z_]\w*)",
    re.DOTALL,
)
UNARY_MACRO_RE = re.compile(
    r"^(?:STATIC_)?IMPLEMENT_(?:FLOAT|COMPLEX)_KERNEL_"
    r"(?:WITH|WITHOUT)_AVX512\((?P<op>\w+)\)",
    re.MULTILINE,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--pytorch", type=Path, default=DEFAULT_PYTORCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from build_ce_viewer import ATEN_C_PROVENANCE

    provenance: dict[str, list[tuple[str, str]]] = {}
    for kernel, (source, token) in ATEN_C_PROVENANCE.items():
        provenance.setdefault(source, []).append((kernel, token or ""))

    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for source in filter(None, map(str.strip, args.sources.read_text().splitlines())):
        text = (args.pytorch / source).read_text(errors="replace")
        targets: list[tuple[str, str, int]] = []
        for match in REGISTER_RE.finditer(text):
            targets.append(
                (
                    match.group("stub"),
                    match.group("impl"),
                    text.count("\n", 0, match.start()) + 1,
                )
            )
        for match in UNARY_MACRO_RE.finditer(text):
            op = match.group("op")
            targets.append(
                (f"{op}_stub", f"{op}_kernel", text.count("\n", 0, match.start()) + 1)
            )
        for stub, implementation, line in targets:
            identity = (source, stub)
            if identity in seen:
                continue
            seen.add(identity)
            fixtures = [
                kernel
                for kernel, token in provenance.get(source, [])
                if token and (
                    token == implementation
                    or implementation in token
                    or token in implementation
                )
            ]
            rows.append(
                {
                    "source": source,
                    "stub": stub,
                    "implementation": implementation,
                    "line": line,
                    "fixtures": ",".join(sorted(fixtures)),
                    "status": (
                        "NO_IMPLEMENTATION"
                        if implementation == "nullptr"
                        else ("EXTRACTED" if fixtures else "NEEDS_EXTRACTION")
                    ),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ("source", "stub", "implementation", "line", "fixtures", "status")
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} concrete dispatch kernels to {args.output}")
    print(f"extracted: {sum(row['status'] == 'EXTRACTED' for row in rows)}")
    print(f"no registered CPU implementation: {sum(row['status'] == 'NO_IMPLEMENTATION' for row in rows)}")
    print(f"remaining: {sum(row['status'] == 'NEEDS_EXTRACTION' for row in rows)}")


if __name__ == "__main__":
    main()
