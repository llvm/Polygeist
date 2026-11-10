#!/usr/bin/env python3
"""Debug aid: why does an application region not match a given spec?

Prints both equation systems and, for every candidate binding, the first
reason compare_systems gives.

Usage: explain_pair.py APP_DEBUF.mlir APP_FUNC SPEC_NAME [--max N]
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORR = HERE.parents[1]
REPO = CORR.parents[1]
sys.path[:0] = [str(CORR), str(CORR / "lib_v2")]
from region_match.matcher import (load_specs, enumerate_bindings, compare_systems,  # noqa: E402
                                  _windows)
from region_match.mlir_parse import parse_module                         # noqa: E402
from region_match.symexec import execute                                 # noqa: E402
from region_match.equations import build_system                          # noqa: E402


def main() -> int:
    app_path, fn, spec_name = sys.argv[1:4]
    limit = int(sys.argv[sys.argv.index("--max") + 1]) if "--max" in sys.argv else 8
    specs = {s.name: s for s in load_specs(str(REPO / "generic_solver/lib_v2"))}
    spec = specs[spec_name]
    funcs, aliases = parse_module(Path(app_path).read_text())
    func = next(f for f in funcs if f.name == fn)
    prog = execute(func, aliases)
    for (lo, hi, end) in _windows(prog)[:1]:
        app = build_system(prog, set(range(lo, end + 1)))
        print(f"== window {lo}-{end}")
        for e in app.equations:
            print("  APP ", e)
        for e in spec.system.equations:
            print("  SPEC", e)
        print("  app reads", sorted(app.reads), " spec reads", sorted(spec.system.reads))
        n = 0
        for b in enumerate_bindings(app, spec.system):
            r = compare_systems(app, spec.system, b)
            print(f"  binding arrays={b.arrays} params={b.params} fparams={b.fparams}")
            print(f"    -> {'MATCH' if r.ok else r.reason[:400]}")
            n += 1
            if r.ok or n >= limit:
                break
        if n == 0:
            print("  (no candidate binding)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
