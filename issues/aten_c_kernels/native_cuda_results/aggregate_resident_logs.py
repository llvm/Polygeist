#!/usr/bin/env python3
"""Merge correctness-gated resident RESULT logs into resident_silicon.csv.

Only kernels for which every parsed RESULT has ``errors=0`` are accepted.
Repeated process runs are represented by their median timing.  Existing rows
are retained unless a newly accepted result for the same kernel is supplied.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
import statistics
from pathlib import Path

RESULT = re.compile(
    r"RESULT kernel=(?P<kernel>\S+) warm_us=(?P<warm>[0-9.]+) "
    r"resident_us=(?P<resident>[0-9.]+) errors=(?P<errors>\d+) "
    r"max_error=(?P<max_error>\S+) shape=(?P<shape>\S+)"
)
FIELDS = ["kernel", "resident_us", "warm_us", "device_speedup", "shape",
          "errors", "hardware", "date", "timing", "process_runs",
          "warmups", "correctness_scope"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--existing", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--date", required=True)
    args = parser.parse_args()

    rows: dict[str, dict[str, str]] = {}
    if args.existing and args.existing.exists():
        rows.update({r["kernel"]: r for r in csv.DictReader(args.existing.open())})

    samples: dict[str, list[dict[str, str]]] = {}
    for pattern in args.logs:
        for name in glob.glob(pattern):
            for line in Path(name).read_text(errors="replace").splitlines():
                if match := RESULT.search(line):
                    samples.setdefault(match["kernel"], []).append(match.groupdict())

    rejected = []
    for kernel, values in sorted(samples.items()):
        if any(int(v["errors"]) != 0 for v in values):
            rejected.append(kernel)
            continue
        resident = statistics.median(float(v["resident"]) for v in values)
        warm = statistics.median(float(v["warm"]) for v in values)
        rows[kernel] = {
            "kernel": kernel,
            "resident_us": f"{resident:.6f}",
            "warm_us": f"{warm:.6f}",
            "device_speedup": f"{warm / resident:.6f}",
            "shape": values[-1]["shape"],
            "errors": "0",
            "hardware": "Jetson_AGX_Orin_sm87_CUDA12.6",
            "date": args.date,
            "timing": "best_of_20_synchronized_wall_us",
            "process_runs": str(len(values)),
            "warmups": "5",
            "correctness_scope": "device_pointer_output_vs_C_reference",
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows))
    print(f"wrote {len(rows)} rows; accepted {len(samples) - len(rejected)}; "
          f"rejected {len(rejected)}: {', '.join(rejected) or 'none'}")


if __name__ == "__main__":
    main()
