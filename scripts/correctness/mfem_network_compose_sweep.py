#!/usr/bin/env python3
"""Apply cuTensorNet network composition to existing MFEM matcher results.

This is deliberately separate from raising/matching: it lets us evaluate and
publish the network optimization without rewriting the established MFEM
artifacts or making its result depend on later matcher changes.
"""

from __future__ import annotations

import csv
from pathlib import Path

from mfem_network_compose import compose_file, launch_symbols, network_symbols


ROOT = Path(__file__).resolve().parents[2]
MFEM = ROOT / "issues" / "mfem_c_kernels"


def compose_rows(summary_path: Path, results_dir: Path, application: bool):
    rows = list(csv.DictReader(summary_path.open()))
    output = []
    for row in rows:
        ident = row["function"] if application else row["id"]
        base = results_dir / ident if not application else results_dir / ident
        matched = (base / "matched.mlir") if not application else base.with_suffix(".matched.mlir")
        composed = (base / "composed.mlir") if not application else base.with_suffix(".composed.mlir")
        log = (base / "compose.log") if not application else base.with_suffix(".compose.log")
        before = launch_symbols(matched)
        rc, message = compose_file(matched, composed, log)
        after = launch_symbols(composed) if rc == 0 else before
        networks = network_symbols(composed) if rc == 0 else []
        result = {
            "id": ident,
            "composition_ok": str(rc == 0).lower(),
            "matcher_launches": len(before),
            "composed_launches": len(after),
            "network_launches": len(networks),
            "matcher_symbols": ",".join(sorted(set(before))),
            "composed_symbols": ",".join(sorted(set(after))),
            "network_symbols": ",".join(sorted(set(networks))),
            "error": "" if rc == 0 else next(
                (line for line in message.splitlines() if "error:" in line),
                message.strip()[:300],
            ),
        }
        output.append(result)
        status = "ok" if rc == 0 else "invalid input"
        print(f"{ident:<48} {len(before):>3}->{len(after):<3} "
              f"networks={len(networks):<2} {status}", flush=True)
    return output


def write_summary(path: Path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    kernel_dir = MFEM / "match_results"
    kernel_rows = compose_rows(kernel_dir / "summary.csv", kernel_dir, False)
    write_summary(kernel_dir / "composition_summary.csv", kernel_rows)

    app_dir = MFEM / "application_extractions" / "results"
    app_rows = compose_rows(app_dir / "summary.csv", app_dir, True)
    write_summary(app_dir / "composition_summary.csv", app_rows)


if __name__ == "__main__":
    main()
