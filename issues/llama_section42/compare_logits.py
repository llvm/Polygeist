#!/usr/bin/env python3
import argparse
import math
from pathlib import Path


def load_polygeist(path: Path) -> list[float]:
    values = []
    for line in path.read_text().splitlines():
        if line.startswith("LOGIT,"):
            values.append(float(line.split(",", 2)[2]))
    return values


def load_ggml(path: Path) -> list[float]:
    lines = path.read_text().splitlines()
    start = lines.index("output_index,value") + 1
    return [float(line.split(",", 1)[1]) for line in lines[start:]]


def report(label: str, reference: list[float], values: list[float],
           atol: float, rtol: float) -> bool:
    if len(reference) != len(values):
        raise ValueError(f"{label}: {len(values)} values, expected {len(reference)}")
    errors = [abs(a - b) for a, b in zip(reference, values)]
    worst = max(range(len(errors)), key=errors.__getitem__)
    failures = sum(error > atol + rtol * abs(expected)
                   for expected, error in zip(reference, errors))
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    print(f"{label}: values={len(values)} failures={failures} "
          f"max_abs={errors[worst]:.9g} worst_index={worst} "
          f"mean_abs={sum(errors) / len(errors):.9g} rmse={rmse:.9g} "
          f"atol={atol:g} rtol={rtol:g}")
    return failures == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("polygeist", type=Path)
    parser.add_argument("ggml", type=Path)
    args = parser.parse_args()
    reference = load_polygeist(args.reference)
    polygeist_ok = report("polygeist", reference,
                          load_polygeist(args.polygeist), 1e-3, 1e-4)
    ggml_ok = report("ggml", reference, load_ggml(args.ggml), 1e-2, 1e-4)
    return 0 if polygeist_ok and ggml_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
