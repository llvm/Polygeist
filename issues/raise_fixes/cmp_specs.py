#!/usr/bin/env python3
"""Compare two specs_scratch.py summaries (status / validation / generics)."""
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
rank = {"RAISED": 3, "UNVALIDATED": 2, "PARTIAL": 1, "UNSOUND": 0, "NOT_RAISED": 0}
for k in sorted(a):
    x, y = a[k], b.get(k, {})
    keys = ("status", "defn", "linalg", "generics", "loops", "tv")
    if any(x.get(t) != y.get(t) for t in keys):
        tag = ""
        if x["status"] == "RAISED" and y.get("status") != "RAISED":
            tag = "  <-- REGRESSION (left the library)"
        print(f"{k:32s} " + " ".join(f"{t}={x.get(t)}->{y.get(t)}" for t in keys if x.get(t) != y.get(t)) + tag)
