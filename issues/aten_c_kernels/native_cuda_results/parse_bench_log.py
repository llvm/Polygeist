#!/usr/bin/env python3
"""Convert bench_shaped.py output into a provenance-bearing CSV."""

import argparse
import csv
import re
from datetime import date
from pathlib import Path


LINE = re.compile(
    r"^kernel=(\S+)\s+(torch_resident_us|torch_cpu_us)=([^\s]+)"
    r"(?:\s+timing=(\S+))?.*?\s+shape='([^']*)'"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--existing", type=Path,
                        help="merge new results into an existing parsed CSV")
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--torch-version", default="2.6.0")
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()

    by_kernel = {}
    if args.existing and args.existing.exists():
        with args.existing.open(newline="") as stream:
            by_kernel.update({row["kernel"]: row for row in csv.DictReader(stream)
                              if row.get("kernel")})
    for log in args.logs:
        for raw in log.read_text().splitlines():
            match = LINE.match(raw)
            if not match:
                continue
            kernel, metric, value, timing, shape = match.groups()
            by_kernel[kernel] = {
                "kernel": kernel,
                "time_us": "" if value == "SKIP" else value,
                "status": "SKIP" if value == "SKIP" else "PASS",
                "metric": metric,
                "timing": timing or "unspecified",
                "shape": shape,
                "hardware": args.hardware,
                "framework": f"torch_{args.torch_version}",
                "date": args.date,
                "raw_result": raw,
            }
    rows = [by_kernel[kernel] for kernel in sorted(by_kernel)]

    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "kernel", "time_us", "status", "metric", "timing", "shape",
            "hardware", "framework", "date", "raw_result",
        ], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    passed = sum(row["status"] == "PASS" for row in rows)
    print(f"wrote {args.output}: {passed}/{len(rows)} PASS")


if __name__ == "__main__":
    main()
