// RUN: /usr/bin/python3 %S/Inputs/check-scaled-gemv-matching.py polygeist-opt %S/../../scripts/correctness/kernel_match_rewrite.py %S/kernel-match-scaled-gemv-transpose.mlir | FileCheck %s
// CHECK: PASS rename invariance and literal/unit coefficient matching
