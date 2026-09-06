#!/usr/bin/env python3
"""Numerically compare PolyBench text dumps with explicit tolerances."""

import argparse
import math
import re
from pathlib import Path


NUMBER = re.compile(
    r"(?<![A-Za-z_])[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|"
    r"inf(?:inity)?|nan)(?![A-Za-z_])", re.IGNORECASE)


def values(path: Path) -> list[float]:
    result: list[float] = []
    for line in path.read_text(errors="replace").splitlines():
        if "DUMP_ARRAYS" in line or "dump:" in line:
            continue
        result.extend(float(token) for token in NUMBER.findall(line))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--rtol", type=float, default=5.0e-4)
    parser.add_argument("--atol", type=float, default=1.1e-2)
    args = parser.parse_args()

    reference = values(args.reference)
    candidate = values(args.candidate)
    if len(reference) != len(candidate):
        print(f"FAIL count reference={len(reference)} candidate={len(candidate)}")
        return 1

    failures = 0
    max_abs = 0.0
    max_rel = 0.0
    first_failure = None
    for index, (expected, actual) in enumerate(zip(reference, candidate)):
        if not (math.isfinite(expected) and math.isfinite(actual)):
            failures += 1
            first_failure = first_failure or (index, expected, actual)
            continue
        absolute = abs(actual - expected)
        relative = absolute / max(abs(expected), args.atol)
        max_abs = max(max_abs, absolute)
        max_rel = max(max_rel, relative)
        if absolute > args.atol + args.rtol * abs(expected):
            failures += 1
            first_failure = first_failure or (index, expected, actual)

    status = "PASS" if failures == 0 else "FAIL"
    print(
        f"{status} values={len(reference)} failures={failures} "
        f"max_abs={max_abs:.9g} max_rel={max_rel:.9g} "
        f"rtol={args.rtol:.9g} atol={args.atol:.9g}")
    if first_failure:
        index, expected, actual = first_failure
        print(f"first_failure index={index} reference={expected} candidate={actual}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
