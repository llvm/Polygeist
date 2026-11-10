#!/usr/bin/env python3
"""CLI:  cli.py <input.mlir> --spec-dir generic_solver/lib_v2 [--rewrite OUT] [--json]"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from region_match.matcher import load_specs, match_text  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mlir")
    ap.add_argument("--spec-dir", required=True)
    ap.add_argument("--rewrite", help="write the input with matched regions replaced")
    ap.add_argument("--json", action="store_true", help="print machine-readable results")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    text = open(args.mlir).read()
    log = (lambda m: print(f"[region_match] {m}", file=sys.stderr)) if args.verbose else None
    t0 = time.perf_counter()
    specs = load_specs(args.spec_dir, log)
    t1 = time.perf_counter()
    stats: dict = {}
    matches = match_text(text, args.spec_dir, log, stats)
    t2 = time.perf_counter()
    if args.json:
        print(json.dumps({
            "input": args.mlir,
            "specs_loaded": len(specs),
            "spec_validation": {s.name: s.validation for s in specs},
            "spec_load_ms": round((t1 - t0) * 1000, 1),
            "match_ms": round((t2 - t1) * 1000, 1),
            "matches": [{"span": m.span, "spec": m.spec, "binding": m.binding,
                         "launch_text": m.launch_text, "evidence": m.evidence,
                         "recognition_only": m.recognition_only} for m in matches],
            "stats": stats,
        }, indent=2, default=str))
    else:
        print(f"// {len(specs)} specs loaded in {(t1 - t0) * 1000:.0f} ms; "
              f"matching took {(t2 - t1) * 1000:.0f} ms")
        if not matches:
            print("// no region matched")
            for r in stats.get("rejections", []):
                print(f"//   {r}")
        for m in matches:
            print(f"// MATCH {m.spec} span={m.span}")
            print(f"//   binding: {m.binding}")
            print(f"//   evidence: {m.evidence}")
            print(m.launch_text)
    if args.rewrite:
        out = text
        for m in sorted(matches, key=lambda m: -m.span[0]):
            if m.recognition_only:
                continue            # reported only; the loops stay
            out = out[:m.span[0]] + m.launch_text + out[m.span[1]:]
        with open(args.rewrite, "w") as f:
            f.write(out)
    return 0 if matches else 1


if __name__ == "__main__":
    sys.exit(main())
