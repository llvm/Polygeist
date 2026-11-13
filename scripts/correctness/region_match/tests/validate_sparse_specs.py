#!/usr/bin/env python3
"""Numerically validate the sparse / histogram specs against their reference C.

The region matcher takes a spec's semantics from cgeist's compilation of the
reference transliteration (generic_solver/lib_v2/raised/<spec>.mlir).  This
check guards that stage against the reference itself, independently of any
application: the cgeist output is lowered to native code and run against the
clang-compiled raised/<spec>.c wrapper on pseudo-random CSR/CSC matrices
(with empty rows/columns) and histograms; outputs must be bit-identical.

Usage: /usr/bin/python3 scripts/correctness/region_match/tests/validate_sparse_specs.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RAISED = REPO / "generic_solver/lib_v2/raised"
LLVM = REPO / "llvm-project/build/bin"
LOWER = ["--convert-math-to-llvm", "--lower-affine", "--convert-scf-to-cf",
         "--expand-strided-metadata", "--lower-affine", "--convert-arith-to-llvm",
         "--convert-index-to-llvm", "--finalize-memref-to-llvm",
         "--convert-func-to-llvm", "--reconcile-unrealized-casts"]

# spec -> (data kind, index type IDX, list of ("i", value[, ctype]) | ("buf", ctype, name)
# in C parameter order).  Index arrays are IDX; "csr1" data is 1-based.
MATVEC = [("i", "NR", "IDX"), ("i", "NC", "IDX"), ("buf", "IDX", "Ap"), ("buf", "IDX", "Aj"),
          ("buf", "double", "Ax"), ("buf", "double", "Xx"), ("buf", "double", "Yx")]
AMUX = [("i", "NR"), ("buf", "double", "Xx"), ("buf", "double", "Yx"), ("buf", "double", "Ax"),
        ("buf", "IDX", "Aj"), ("buf", "IDX", "Ap")]
SPECS = {
    "scipy_csr_matvec": ("csr", MATVEC),
    "scipy_csr_matvec_beta0": ("csr", MATVEC),
    "scipy_csc_matvec": ("csc", MATVEC),
    "scipy_csc_matvec_beta0": ("csc", MATVEC),
    "scipy_csr_matvec_i64": ("csr", MATVEC, "long long"),
    "scipy_csr_matvec_beta0_i64": ("csr", MATVEC, "long long"),
    "scipy_csc_matvec_i64": ("csc", MATVEC, "long long"),
    "scipy_csc_matvec_beta0_i64": ("csc", MATVEC, "long long"),
    "sparskit_amux": ("csr1", AMUX),
    "numpy_bincount": ("hist", [("i", "LEN"), ("buf", "int", "num"), ("i", "NB"), ("buf", "int", "cnt")]),
    "numpy_bincount_accumulate": ("hist", [("i", "LEN"), ("buf", "int", "num"), ("i", "NB"),
                                           ("buf", "int", "cnt")]),
    "numpy_bincount_weights": ("hist", [("i", "LEN"), ("buf", "int", "num"), ("buf", "double", "w"),
                                        ("i", "NB"), ("buf", "double", "dcnt")]),
}

SETUP = r"""
#include <stdio.h>
#include <stdint.h>
#include <string.h>
enum { NR = 9, NC = 7, LEN = 40, NB = 6, CAP = 64 };
static IDX Ap[3][CAP], Aj[3][CAP];
static int num[3][CAP], cnt[3][CAP];
static double Ax[3][CAP], Xx[3][CAP], Yx[3][CAP], w[3][CAP], dcnt[3][CAP];
static unsigned rng = 12345u;
static unsigned next(void) { rng = rng * 1103515245u + 12345u; return rng >> 8; }
static void fill(int outer, int inner) {   /* compressed pointers + indices */
  int nnz = 0;
  for (int r = 0; r < outer; r++) {
    Ap[0][r] = nnz;
    int len = (int)(next() % 4);                       /* 0..3, incl. empty */
    for (int k = 0; k < len; k++, nnz++) {
      Aj[0][nnz] = (int)(next() % inner);
      Ax[0][nnz] = 0.25 + (double)(next() % 1000) / 333.0;
    }
  }
  Ap[0][outer] = nnz;
}
"""


def proto(name, args, tag):
    ps = []
    for a in args:
        if a[0] == "i":
            ps.append(a[2] if len(a) > 2 else "int32_t")
        elif tag == "c":
            ps.append(f"{a[1]}*")
        else:
            ps += [f"{a[1]}*", f"{a[1]}*", "int64_t", "int64_t", "int64_t"]
    return f"void {name}{'' if tag == 'c' else '_mlir'}({', '.join(ps)});"


def driver(name, kind, args, idx="int") -> str:
    lines = [f"typedef {idx} IDX;", SETUP, proto(name, args, "c"), proto(name, args, "mlir"),
             "int main(void) {"]
    if kind == "csr":
        lines.append("  fill(NR, NC);")
    elif kind == "csr1":   # 1-based pointers and column indices
        lines.append("  fill(NR, NC);")
        lines.append("  for (int r = 0; r <= NR; r++) Ap[0][r] += 1;")
        lines.append("  for (int k = 0; k < Ap[0][NR] - 1; k++) Aj[0][k] += 1;")
    elif kind == "csc":
        lines.append("  fill(NC, NR);")
    else:
        lines.append("  for (int i = 0; i < LEN; i++) { num[0][i] = (int)(next() % NB);"
                     " w[0][i] = (double)(next() % 977) / 97.0; }")
    lines.append("  for (int i = 0; i < CAP; i++) { Xx[0][i] = 1.0 + (double)(next() % 500) / 77.0;"
                 " Yx[0][i] = (double)(next() % 300) / 31.0; cnt[0][i] = (int)(next() % 5);"
                 " dcnt[0][i] = (double)(next() % 300) / 13.0; }")
    bufs = [a[2] for a in args if a[0] == "buf"]
    for b in bufs:
        lines.append(f"  memcpy({b}[1], {b}[0], sizeof {b}[0]); memcpy({b}[2], {b}[0], sizeof {b}[0]);")
    for v, tag in ((1, "c"), (2, "mlir")):
        call = []
        for a in args:
            if a[0] == "i":
                call.append(a[1])
            elif tag == "c":
                call.append(f"{a[2]}[{v}]")
            else:
                call.append(f"{a[2]}[{v}], {a[2]}[{v}], 0, CAP, 1")
        lines.append(f"  {name}{'' if tag == 'c' else '_mlir'}({', '.join(call)});")
    lines.append("  int bad = 0;")
    for b in bufs:
        lines.append(f"  if (memcmp({b}[1], {b}[2], sizeof {b}[1])) {{ printf(\"%s differs\\n\", \"{b}\"); bad = 1; }}")
    lines.append("  return bad;\n}")
    return "\n".join(lines) + "\n"


def sh(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-800:])
    return r


def main() -> int:
    fails = 0
    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        for name, (kind, args, *idx) in SPECS.items():
            try:
                mlir = RAISED / f"{name}.mlir"
                sh([LLVM / "mlir-opt", *LOWER, mlir, "-o", wd / f"{name}.llvm.mlir"])
                sh([LLVM / "mlir-translate", "--mlir-to-llvmir", wd / f"{name}.llvm.mlir",
                    "-o", wd / f"{name}.ll"])
                ll = (wd / f"{name}.ll").read_text()
                (wd / f"{name}.ll").write_text(re.sub(rf"@{name}\b", f"@{name}_mlir", ll))
                (wd / "drv.c").write_text(driver(name, kind, args, *idx))
                exe = wd / f"{name}.exe"
                sh([LLVM / "clang", "-O1", "-w", wd / "drv.c", RAISED / f"{name}.c",
                    "-I", REPO / "generic_solver/lib_v2", wd / f"{name}.ll", "-o", exe])
                r = subprocess.run([exe], capture_output=True, text=True)
                ok = r.returncode == 0
                print(f"{name:28s} {'BIT-IDENTICAL (cgeist stage == clang reference)' if ok else 'MISMATCH ' + r.stdout.strip()}")
                fails += not ok
            except RuntimeError as e:
                print(f"{name:28s} ERROR {str(e)[:300]}")
                fails += 1
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
