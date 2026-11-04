// Polygeist's canonicaliser rewrites `x - C` as `x + (-C)`. -C must be
// computed in the integer's own width: it was built from a 64-bit value and
// zero-extended, so on an integer wider than 64 bits `x - 1` became
// `x + 18446744073709551615` (x - 1 + 2^64). Widths <= 64 are unchanged.
// RUN: cgeist %s --function='*' -S | FileCheck %s

typedef unsigned _BitInt(128) u128;
typedef unsigned _BitInt(512) u512;
typedef unsigned _BitInt(513) u513;

// CHECK-NOT: 18446744073709551615

// CHECK-LABEL: func.func @sub1(%arg0: i128) -> i128
// CHECK: %[[C:.*]] = arith.constant -1 : i128
// CHECK: arith.addi %arg0, %[[C]] : i128
u128 sub1(u128 x) { return x - 1; }

// CHECK-LABEL: func.func @mask16() -> i512
// CHECK: arith.constant 65535 : i512
static u512 mask(int n) { return (u512)((((u513)1) << n) - 1); }
u512 mask16(void) { return mask(16); }

// CHECK-LABEL: func.func @sub3(%arg0: i32) -> i32
// CHECK: %[[C3:.*]] = arith.constant -3 : i32
// CHECK: arith.addi %arg0, %[[C3]] : i32
int sub3(int x) { return x - 3; }
