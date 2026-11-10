#!/usr/bin/env python3
"""Run the region matcher over every PolyBench kernel raised with the shared
production recipe (issues/lib_v2/polybench_raised/<k>/debuf.mlir, produced by
scripts/correctness/lib_v2/raise_polybench.py) against all lib_v2 specs.
Writes issues/region_match/polybench_sweep.json."""
from __future__ import annotations

import json
import os
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE.parents[1]))
from region_match.matcher import load_specs, match_text  # noqa: E402

SD = str(REPO / "generic_solver/lib_v2")
ROOT = REPO / "issues/lib_v2/polybench_raised"


class _TO(Exception):
    pass


def main() -> int:
    t0 = time.perf_counter()
    load_specs(SD)
    print(f"specs loaded in {(time.perf_counter() - t0) * 1000:.0f} ms")
    signal.signal(signal.SIGALRM, lambda *a: (_ for _ in ()).throw(_TO()))
    out = {}
    for k in sorted(os.listdir(ROOT)):
        p = ROOT / k / "debuf.mlir"
        if not p.exists():
            continue
        st: dict = {}
        t = time.perf_counter()
        signal.alarm(120)
        try:
            ms = match_text(p.read_text(), SD, None, st)
            status = "ok"
        except _TO:
            ms, status = [], "timeout"
        except Exception as e:  # noqa: BLE001
            ms, status = [], f"crash {type(e).__name__}: {e}"
        finally:
            signal.alarm(0)
        dt = (time.perf_counter() - t) * 1000
        rej = st.get("rejections", [])
        out[k] = {"status": status, "ms": round(dt), "matches": [
            {"spec": m.spec, "binding": m.binding, "evidence": m.evidence} for m in ms],
            "first_rejection": rej[0] if rej else None}
        print(f"{k:16s} {dt:7.0f} ms  {status:8s} {[m.spec for m in ms]}  "
              f"{'' if ms else (rej[0][:100] if rej else '')}", flush=True)
    (REPO / "issues/region_match").mkdir(parents=True, exist_ok=True)
    (REPO / "issues/region_match/polybench_sweep.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
