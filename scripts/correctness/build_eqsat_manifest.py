#!/usr/bin/env python3
"""Build the current paper-facing RQ3/RQ4 raised-MLIR manifest."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def preferred_pairs(directory: Path, suffixes=("_debuf.mlir", "_debuf_mr.mlir")):
    chosen = {}
    for priority, suffix in enumerate(suffixes):
        for path in sorted(directory.glob(f"*{suffix}")):
            name = path.name[:-len(suffix)]
            chosen.setdefault(name, (priority, path))
    return [(name, value[1]) for name, value in sorted(chosen.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []

    for name, path in preferred_pairs(Path("/tmp/polybench_new")):
        rows.append(("polybench", name, path))

    aten_root = ROOT / "issues/aten_c_kernels/results"
    for path in sorted(aten_root.glob("*/debuf.mlir")):
        rows.append(("aten", path.parent.name, path))

    mfem_root = ROOT / "issues/mfem_c_kernels"
    for path in sorted((mfem_root / "match_results").glob("*/debufferized.mlir")):
        rows.append(("mfem_kernel", path.parent.name, path))
    for path in sorted((mfem_root / "application_extractions/results").glob(
            "*.debufferized.mlir")):
        rows.append(("mfem_application", path.name[:-len(".debufferized.mlir")], path))

    for name, path in preferred_pairs(Path("/tmp/llama2c_mlir")):
        rows.append(("llama2c", name, path))
    for name, path in preferred_pairs(Path("/tmp/llama_forward_ops_mlir")):
        rows.append(("llama_forward", name, path))

    identities = set()
    for suite, input_id, path in rows:
        identity = (suite, input_id)
        if identity in identities:
            raise SystemExit(f"duplicate identity: {identity}")
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
        identities.add(identity)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("suite", "input_id", "path"))
        for suite, input_id, path in rows:
            writer.writerow((suite, input_id, str(path.resolve())))
    print(f"wrote {len(rows)} inputs to {args.output}")


if __name__ == "__main__":
    main()
