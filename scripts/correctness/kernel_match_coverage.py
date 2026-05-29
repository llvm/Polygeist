#!/usr/bin/env python3
"""Cross-coverage analysis: for every (kernel, body), what library entries match?

This tells us how many distinct "library kernels" we actually need to cover
the 26 lowering-clean PolyBench kernels — and where sharing happens.
"""
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from kernel_match import (
    build_library_from_dir, parse_generics, encode_body, match,
)

root = Path("/tmp/polybench_new")
print(f"Building library...", flush=True)
lib = build_library_from_dir(root)
print(f"Library has {len(lib)} entries.\n", flush=True)

# Now cross-match: for each body in each kernel, which library entry hits?
rows = []
for f in sorted(root.glob("*_debuf.mlir")):
    text = f.read_text()
    try:
        gens = parse_generics(text)
    except Exception:
        continue
    kernel = f.stem.replace("_debuf", "")
    for i, g in enumerate(gens):
        try:
            t = encode_body(g)
        except Exception as e:
            rows.append((kernel, i, "ENCODE_FAIL"))
            continue
        hit = match(t, lib, len(g.ins_arg_names), len(g.outs_arg_names),
                    g.indexing_maps, g.iterator_types)
        rows.append((kernel, i, hit.name if hit else "NO_MATCH"))

# Group by kernel.
from collections import defaultdict
matches = defaultdict(list)
for k, i, name in rows:
    matches[k].append((i, name))

print(f"{'kernel':<20} {'generic#':<10} {'matched library entry'}")
print("-" * 80)
for k in sorted(matches):
    for i, name in matches[k]:
        print(f"{k:<20} #{i:<9} {name}")

# Summary
total = len(rows)
matched = sum(1 for _, _, n in rows if n not in ("NO_MATCH", "ENCODE_FAIL"))
enc_fail = sum(1 for _, _, n in rows if n == "ENCODE_FAIL")
no_match = sum(1 for _, _, n in rows if n == "NO_MATCH")
print(f"\n{matched}/{total} bodies match a library entry "
      f"({no_match} no-match, {enc_fail} encoder-fail).")
