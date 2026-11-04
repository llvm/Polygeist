// Bitwise not on an integer wider than 64 bits must flip every bit. Found by the
// W11 validation (~/hls-est-work/w11val/diff, kernels k173/k178/k182 and, through
// AP_WRAP_SM's `Base::V = ~Base::V` (etc/ap_fixed_base.h:329), k571/k574): cgeist
// built the all-ones mask with APInt::getAllOnes(w).getSExtValue() and
// ConstantIntOp(int64_t, iN), which zero-extends -1, so only the low 64 bits
// were inverted (Lib/clang-mlir.cc UO_Not). Present before W11 as well (the
// control binary built from 95a5ce16 gives the wrong MLIR). Fixed in the W11
// review fixes: the mask is a full-width IntegerAttr.
// RUN: cgeist %s --function='*' -S -o - | FileCheck %s

// CHECK-LABEL: func.func @not100(
// CHECK: %[[M:.*]] = arith.constant -1 : i100
// CHECK: arith.xori %{{.*}}, %[[M]] : i100
extern "C" unsigned _BitInt(100) not100(unsigned _BitInt(100) x) { return ~x; }

// CHECK-LABEL: func.func @not64(
// CHECK: arith.constant -1 : i64
extern "C" unsigned long long not64(unsigned long long x) { return ~x; }
