#!/usr/bin/env python3
"""Audit complete Llama logits with both original numerical gates retained."""
import argparse
import json
import math
from pathlib import Path
import re


def read_logits(path, count):
    values = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"(?:LOGIT,)?(\d+),([^,]+)", line)
        if match:
            index, value = int(match[1]), float(match[2])
            if index in values:
                raise ValueError(f"{path}: duplicate logit {index}")
            if not math.isfinite(value):
                raise ValueError(f"{path}: non-finite logit {index}")
            values[index] = value
    if set(values) != set(range(count)):
        raise ValueError(f"{path}: expected exactly {count} indexed logits")
    return [values[i] for i in range(count)]


def compare(actual, reference):
    errors = [abs(a - b) for a, b in zip(actual, reference)]
    return {
        "count": len(errors),
        "max_abs": max(errors),
        "rms_error": math.sqrt(sum(e * e for e in errors) / len(errors)),
        "unequal": sum(a != b for a, b in zip(actual, reference)),
        "strict_failures": sum(e > 0.001 + 0.0001 * abs(b)
                               for e, b in zip(errors, reference)),
        "cross_implementation_failures": sum(e > 0.01 + 0.0001 * abs(b)
                                             for e, b in zip(errors, reference)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("native", "raised", "expert", "oracle"):
        parser.add_argument("--" + name, type=Path, required=name != "oracle")
    parser.add_argument("--count", type=int, default=32000)
    args = parser.parse_args()
    outputs = {name: read_logits(path, args.count) for name in
               ("native", "raised", "expert", "oracle")
               if (path := getattr(args, name)) is not None}
    report = {"gates": {"strict": {"atol": 0.001, "rtol": 0.0001},
                        "cross_implementation": {"atol": 0.01, "rtol": 0.0001}}}
    pairs = [("raised", "native"), ("expert", "native"), ("raised", "expert")]
    if "oracle" in outputs:
        pairs += [(name, "oracle") for name in ("native", "raised", "expert")]
    for actual, reference in pairs:
        report[actual + "_vs_" + reference] = compare(outputs[actual], outputs[reference])
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
