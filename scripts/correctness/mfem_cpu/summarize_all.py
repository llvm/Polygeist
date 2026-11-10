#!/usr/bin/env python3
"""Summarise run_all_cpu.sh: launches before/after network composition,
residual Linalg stages, full-output validation per backend, and median/min
runtime per variant against plain C.

Usage: summarize_all.py BUILD_ROOT OUT_DIR
"""
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

LAUNCH_RE = re.compile(r"kernel\.launch @([A-Za-z0-9_]+)")
VARIANTS = ["vanilla_c", "raised_loops", "raised_openblas", "raised_xsmm",
            "raised_xsmm_noslice"]


def launches(path: Path) -> str:
    if not path.exists():
        return "-"
    syms = Counter(re.sub(r"_\d+$", "", s) for s in LAUNCH_RE.findall(path.read_text()))
    return " ".join(f"{n}x{s}" for s, n in sorted(syms.items())) or "none"


def residual_generics(path: Path, fn: str) -> str:
    """linalg.generic ops left in the kernel function (compiled, not a call)."""
    if not path.exists():
        return "-"
    lines = path.read_text().split("\n")
    start = next((i for i, l in enumerate(lines) if f"func.func @{fn}(" in l), None)
    if start is None:
        return "?"
    indent = len(lines[start]) - len(lines[start].lstrip())
    count = 0
    for line in lines[start + 1:]:
        if line.strip() == "}" and len(line) - len(line.lstrip()) == indent:
            break
        count += "linalg.generic" in line
    return str(count)


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for kdir in sorted(p for p in out.iterdir() if p.is_dir()):
        name = kdir.name
        build_log = root / f"{name}.log"
        fn = ""
        if build_log.exists():
            m = re.search(r"function=(\S+)", build_log.read_text())
            fn = m.group(1) if m else ""
        if not fn:
            blog = root / name / "base_build.log"
            m = re.search(r"function=(\S+)", blog.read_text()) if blog.exists() else None
            fn = m.group(1) if m else ""
        before = launches(root / name / "base_work" / "matched.mlir")
        after = launches(root / name / "with_defns_composed.mlir")
        resid = residual_generics(root / name / "with_defns_composed.mlir", fn)
        run_log = (kdir / "run.log").read_text() if (kdir / "run.log").exists() else ""
        valid = {}
        for be, line in re.findall(r"-- backend=(\w+)\n(.*)", run_log):
            m = re.search(r"correctness=(\w+) failures=(\d+) max_abs=(\S+)", line)
            valid[be] = f"{m.group(1)}({m.group(3)})" if m else "NO-OUTPUT"
        times = {}
        tfile = kdir / "timings.txt"
        if tfile.exists():
            for line in tfile.read_text().splitlines():
                parts = line.split()
                vals = [float(v) for v in parts[1:] if re.fullmatch(r"[0-9.]+", v)]
                if vals:
                    times[parts[0]] = (statistics.median(vals), min(vals))
        rows.append((name, before, after, resid, valid, times))

    for name, before, after, resid, valid, times in rows:
        print(f"* {name}")
        print(f"    matcher launches : {before}")
        print(f"    after composition: {after}   residual linalg.generic: {resid}")
        print("    validation       : " + (", ".join(f"{b}={v}" for b, v in valid.items()) or "none"))
        if times:
            base = times.get("vanilla_c")
            parts = []
            for v in VARIANTS:
                if v in times:
                    med, mn = times[v]
                    sp = f" ({base[0] / med:.2f}x)" if base and v != "vanilla_c" else ""
                    parts.append(f"{v.replace('raised_', '')} {med:.0f}/{mn:.0f}{sp}")
            print("    median/min us    : " + "; ".join(parts))
    print("\nspeedup = plain-C median / variant median (>1 means faster than plain C)")


if __name__ == "__main__":
    main()
