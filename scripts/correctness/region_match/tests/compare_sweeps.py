#!/usr/bin/env python3
"""Compare two PolyBench sweeps of lib_v2/match_v2.py (Linalg-level +
region-matcher launches) kernel by kernel.  A match is LOST if a launch of
the baseline (multiset of launched specification names per kernel) is
missing from the new sweep.

Usage: compare_sweeps.py BASE_DIR NEW_DIR [--json OUT]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path


def launches(d: Path) -> dict:
    out = {}
    for rep in sorted(d.glob("*/report.json")):
        r = json.loads(rep.read_text())
        lin = [l["spec"] for l in r.get("launches", []) if "spec" in l]
        reg = []
        notes = []
        for l in r.get("launches", []):
            if "region_match" in l:
                v = l["region_match"]
                if v.startswith("netlib_"):
                    reg += v.split("+")
                else:
                    notes.append(v)
            if "region_match_skipped" in l:
                notes.append(f"skipped {l['region_match_skipped']}: {l.get('reason')}")
        out[rep.parent.name] = {"linalg": lin, "region": reg, "notes": notes}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--json")
    a = ap.parse_args()
    base, new = launches(Path(a.base)), launches(Path(a.new))
    rows, lost_any = [], False
    for k in sorted(set(base) | set(new)):
        b = base.get(k, {"linalg": [], "region": [], "notes": []})
        n = new.get(k, {"linalg": [], "region": [], "notes": ["missing report"]})
        bc = collections.Counter(b["linalg"] + b["region"])
        nc = collections.Counter(n["linalg"] + n["region"])
        lost = sorted((bc - nc).elements())
        gained = sorted((nc - bc).elements())
        lost_any |= bool(lost)
        rows.append({"kernel": k, "base": b, "new": n, "lost": lost, "gained": gained})
        if lost or gained or n["notes"]:
            print(f"{k:16s} lost={lost} gained={gained} notes={n['notes'][:2]}")
    nb = sum(len(r["base"]["linalg"]) + len(r["base"]["region"]) for r in rows)
    nn = sum(len(r["new"]["linalg"]) + len(r["new"]["region"]) for r in rows)
    full_b = sum(1 for r in rows if r["base"]["linalg"] or r["base"]["region"])
    full_n = sum(1 for r in rows if r["new"]["linalg"] or r["new"]["region"])
    print(f"launches: base {nb} -> new {nn}; kernels with >=1 launch: base {full_b} -> new {full_n}; "
          f"{'MATCHES LOST' if lost_any else 'no match lost'}")
    if a.json:
        Path(a.json).write_text(json.dumps({"lost_any": lost_any, "rows": rows}, indent=1))
    return 1 if lost_any else 0


if __name__ == "__main__":
    sys.exit(main())
