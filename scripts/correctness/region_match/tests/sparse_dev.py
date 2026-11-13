#!/usr/bin/env python3
"""Development set for indirect-access region matching (CSR, CSC, histogram).

The C files are written from the libraries' documented semantics (cuSPARSE
SpMV, CUB DeviceHistogram / NumPy bincount), not from any evaluation
benchmark.  Each is raised with the production recipe and matched against
the lib_v2 specs; a positive must select exactly the expected spec, a
negative must select none of the sparse/histogram specs.  The cases after
lim_csc_noalias were added after the held-out evaluation, together with the
int64 / 1-based specs and expression-valued extents.

Usage: /usr/bin/python3 scripts/correctness/region_match/tests/sparse_dev.py [-v]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORR = HERE.parents[1]
REPO = CORR.parents[1]
sys.path[:0] = [str(CORR), str(CORR / "lib_v2")]
import recipe as R                                    # noqa: E402
from region_match import match_regions                # noqa: E402

SPEC_DIR = REPO / "generic_solver/lib_v2"
OUT = HERE / "raised"
SPARSE = ("scipy_csr", "scipy_csc", "numpy_bincount", "sparskit_")

CASES = [
    ("sp_csr_rowsum", "scipy_csr_matvec_beta0"),
    ("sp_csr_inmem", "scipy_csr_matvec"),
    ("sp_csc_scatter", "scipy_csc_matvec"),
    ("sp_csc_zeroed", "scipy_csc_matvec_beta0"),
    ("sp_hist_count", "numpy_bincount"),
    ("sp_hist_accum", "numpy_bincount_accumulate"),
    ("sp_hist_weighted", "numpy_bincount_weights"),
    ("neg_csr_bound", None),
    ("neg_csr_nogather", None),
    ("neg_csc_twotargets", None),
    ("neg_hist_plus2", None),
    ("neg_hist_saturate", None),
    ("neg_hist_selfindex", None),
    ("lim_csc_noalias", "scipy_csc_matvec"),   # scf.while + noalias assumption
    # follow-up (written after the held-out run): 64-bit indices, 1-based
    # CSR, and extents that are expressions or loaded from memory
    ("sp_csr_i64", "scipy_csr_matvec_beta0_i64"),
    ("sp_csc_i64", "scipy_csc_matvec_i64"),
    ("sp_csr_onebased", "sparskit_amux"),
    ("sp_csr_droplast", "scipy_csr_matvec_beta0"),
    ("sp_csr_sizeptr", "scipy_csr_matvec_beta0"),
    ("sp_hist_window", "numpy_bincount_accumulate"),
    ("neg_csr_onebased_mixed", None),
    ("neg_csr_rows_shifted", None),
    ("neg_hist_window_scaled", None),
]


def main() -> int:
    verbose = "-v" in sys.argv
    OUT.mkdir(exist_ok=True)
    failures = 0
    for name, expect in CASES:
        res = R.raise_c(R.PRODUCTION, str(HERE / "c" / f"{name}.c"), name,
                        str(OUT / name), [])
        debuf = Path(res["paths"]["debuf"])
        if not debuf.exists():
            print(f"{name:20s} RAISE_FAIL"); failures += 1; continue
        log: list = []
        t0 = time.perf_counter()
        try:
            got = [m.spec for m in match_regions(debuf.read_text(), str(SPEC_DIR))]
        except Exception as e:  # noqa: BLE001
            got = [f"ERROR {type(e).__name__}: {e}"]
        ms = (time.perf_counter() - t0) * 1000
        sparse = [g for g in got if g.startswith(SPARSE)]
        ok = (sparse == [expect]) if expect else not sparse
        failures += not ok
        print(f"{name:20s} {'PASS' if ok else 'FAIL'}  expect={expect}  got={got}  ({ms:.0f} ms)")
    print(f"{len(CASES) - failures}/{len(CASES)} pass")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
