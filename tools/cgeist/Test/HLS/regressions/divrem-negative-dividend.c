// C's / and % truncate toward zero, the affine floordiv and mod round down:
// they agree only for a non-negative dividend and a positive divisor. A div /
// rem of any width, int and index included, is raised to floordiv / mod only
// when its dividend provably is non-negative; this is rounding, not overflow,
// so it does not depend on -affine-cfg-assume-nowrap-width.
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

#include <stdint.h>

// (i - 5) / 2 for i in [0, 20): -2 for i = 0, where floordiv gives -3.
// CHECK-LABEL: func.func @div_negative(
// CHECK: %[[D:.*]] = arith.divsi %{{.*}}, %{{.*}} : i32
// CHECK: %[[X:.*]] = arith.addi %[[D]], %{{.*}} : i32
// CHECK: %[[I:.*]] = arith.index_cast %[[X]] : i32 to index
// CHECK: memref.load %{{.*}}[%[[I]]]
void div_negative(int A[256], int O[32]) {
  for (int i = 0; i < 20; i++)
    O[i] = A[(i - 5) / 2 + 10];
}

// i in [5, 25): i - 5 >= 0, raised.
// CHECK-LABEL: func.func @div_nonnegative(
// CHECK-NOT: arith.divsi {{.*}} : i32
// CHECK: return
void div_nonnegative(int A[256], int O[32]) {
  for (int i = 5; i < 25; i++)
    O[i] = A[(i - 5) / 2 + 10];
}

// An int8_t i in [-100, 99]: i % 16 is negative for negative i.
// CHECK-LABEL: func.func @rem_negative(
// CHECK: %[[R:.*]] = arith.remsi %{{.*}}, %{{.*}} : i32
// CHECK: %[[X:.*]] = arith.addi %[[R]], %{{.*}} : i32
// CHECK: %[[I:.*]] = arith.index_cast %[[X]] : i32 to index
// CHECK: memref.load %{{.*}}[%[[I]]]
void rem_negative(int A[256], int O[256]) {
  for (int8_t i = -100; i < 100; i++)
    O[i + 100] = A[(i % 16) + 16];
}

// A runtime bound: i in [0, n - 1] is non-negative, raised.
// CHECK-LABEL: func.func @div_runtime_bound(
// CHECK-NOT: arith.divsi {{.*}} : i32
// CHECK: return
void div_runtime_bound(int A[4096], int O[4096], int n) {
  for (int i = 0; i < n; i++)
    O[i] = A[i / 2];
}

// Unsigned w - i wraps: (w - i) % 100 is not (w - i) mod 100.
// CHECK-LABEL: func.func @urem_wrapping(
// CHECK: %[[D:.*]] = arith.subi %{{.*}}, %{{.*}} : i32
// CHECK: %[[R:.*]] = arith.remui %[[D]], %{{.*}} : i32
// CHECK: %[[I:.*]] = arith.index_castui %[[R]] : i32 to index
// CHECK: memref.load %{{.*}}[%[[I]]]
void urem_wrapping(int A[256], int O[64], int s) {
  unsigned w = (unsigned)s & 63;
  for (unsigned i = 0; i < 64; i++)
    O[i] = A[(w - i) % 100];
}
