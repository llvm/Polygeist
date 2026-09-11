#!/usr/bin/env python3
"""Generate the fast MFEM coverage ledger from retained debufferized IR.

This intentionally does not rerun raising.  It applies the current audited
matcher to the retained 20 normalized debufferized modules and records hashes
so the result cannot be confused with a clean end-to-end compiler rebuild.
"""

import argparse
import csv
import hashlib
import re
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    corpus = repo / "issues/mfem_c_kernels"
    matcher = repo / "scripts/correctness/kernel_match_rewrite.py"
    manifest = list(csv.DictReader((corpus / "manifest.csv").open()))
    originals = {
        (r["family"], r["dimension"], r["operation"], r["upstream_file"],
         r["upstream_symbol"]): r
        for r in manifest if r["variant"] == "original"
    }
    normalized = [r for r in manifest if r["variant"] == "normalized"]
    if len(originals) != 20 or len(normalized) != 20:
        raise SystemExit("expected exactly 20 original and 20 normalized rows")

    rows = []
    for row in normalized:
        key = (row["family"], row["dimension"], row["operation"],
               row["upstream_file"], row["upstream_symbol"])
        original = originals.get(key)
        if original is None:
            raise SystemExit(f"no original partner for {row['id']}")
        debuf = corpus / "match_results" / row["id"] / "debufferized.mlir"
        dry = subprocess.run(
            ["/usr/bin/python3", str(matcher), str(debuf), "--dry-run"],
            text=True, capture_output=True, timeout=300,
        )
        rewritten = subprocess.run(
            ["/usr/bin/python3", str(matcher), str(debuf)],
            text=True, capture_output=True, timeout=300,
        )
        report = dry.stdout + dry.stderr
        total = re.search(r"total:\s+(\d+) matched / (\d+) bodies", report)
        launches = re.findall(
            r"kernel\.launch\s+@([A-Za-z0-9_.$-]+)", rewritten.stdout
        )
        rows.append({
            "semantic_id": original["id"],
            "normalized_id": row["id"],
            "family": row["family"],
            "dimension": row["dimension"],
            "operation": row["operation"],
            "upstream_file": row["upstream_file"],
            "upstream_symbol": row["upstream_symbol"],
            "matcher_ok": str(dry.returncode == 0 and rewritten.returncode == 0).lower(),
            "matched_region_count": total.group(1) if total else "0",
            "matcher_body_count": total.group(2) if total else "0",
            "external_launch_count": str(len(launches)),
            "external_symbols": ";".join(sorted(set(launches))),
            "residual_linalg_ops": str(rewritten.stdout.count("linalg.")),
            "debufferized_ir_sha256": sha256(debuf),
            "matcher_sha256": sha256(matcher),
            "notes": "fixed matcher over retained debufferized IR; raising not regenerated",
        })
        print(f"{row['id']}: launches={len(launches)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    matched = sum(int(r["external_launch_count"]) > 0 for r in rows)
    launches = sum(int(r["external_launch_count"]) for r in rows)
    residual = sum(int(r["residual_linalg_ops"]) for r in rows)
    print(f"semantic_kernels={len(rows)} kernels_with_matches={matched} "
          f"external_launches={launches} residual_linalg_ops={residual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
