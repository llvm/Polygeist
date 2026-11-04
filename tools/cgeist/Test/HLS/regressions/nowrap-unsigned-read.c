// AffineCFG raises a marked (polygeist.nsw) add / sub / mul and a cast at
// least 32 bits wide without a range proof: each is exact as a value read
// signed. Read unsigned (through an extui / index_castui: a conversion to an
// unsigned type), a negative iN value is 2^N more, so such a value read
// unsigned narrower than 64 bits is raised (and an index -> i32 -> index
// round trip read unsigned is folded) only when it is provably non-negative.
// Every function below is defined C; raised as before, f01-f05 read the
// wrong element at the first iterations and f06's inner loop ran no
// iteration at i = -1 (W5 review D1; checked against native clang).
// -affine-cfg-assume-nowrap-width=32 keeps the former rule, for comparison.
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -affine-cfg-assume-nowrap-width=32 -o - | FileCheck %s --check-prefix=W32

#include <stddef.h>
#include <stdint.h>

// (unsigned)(i - 3) is 4294967293 at i = 0, so C reads A[4294967293 % 7] =
// A[1]; the raised (i - 3) mod 7 reads A[4].
// CHECK-LABEL: func.func @f01_sizet_rem(
// CHECK: arith.extui %{{.*}} : i32 to i64
// CHECK-NEXT: arith.remui
void f01_sizet_rem(int A[64], int O[64]) {
  for (int i = 0; i < 8; i++) {
    size_t k = (unsigned)(i - 3);
    O[i] = A[k % 7];
  }
}

// 3 for i < 4 (the unsigned view of a negative difference), 0 after.
// CHECK-LABEL: func.func @f02_u64_div(
// CHECK: arith.extui %{{.*}} : i32 to i64
// CHECK-NEXT: arith.divui
void f02_u64_div(int A[64], int O[64]) {
  for (int i = 0; i < 8; i++)
    O[i] = A[(uint64_t)(uint32_t)(i - 4) / 0x40000000ull];
}

// Unary minus of a marked int.
// CHECK-LABEL: func.func @f03_neg_view(
// CHECK: arith.extui %{{.*}} : i32 to i64
// CHECK-NEXT: arith.remui
void f03_neg_view(int A[64], int O[64]) {
  for (int i = 0; i < 8; i++)
    O[i] = A[(unsigned long long)(unsigned)(-i) % 9ull];
}

// No arithmetic: an int counter that starts negative, read unsigned (a wide
// truncation of the index counter).
// CHECK-LABEL: func.func @f04_iv_view(
// CHECK: arith.extui %{{.*}} : i32 to i64
// CHECK-NEXT: arith.remui
void f04_iv_view(int A[64], int O[64]) {
  for (int i = -3; i < 5; i++)
    O[i + 3] = A[(unsigned long)(unsigned)i % 7ul];
}

// A short counter promoted to int, read unsigned (a wide sign extension).
// CHECK-LABEL: func.func @f05_short_view(
// CHECK: arith.extsi %{{.*}} : i16 to i32
// CHECK-NEXT: arith.extui %{{.*}} : i32 to i64
// CHECK-NEXT: arith.remui
void f05_short_view(int A[64], int O[64]) {
  for (short s = -3; s < 5; s++)
    O[s + 3] = A[(unsigned long)(unsigned)s % 7ul];
}

// The bound (unsigned)i of an int counter i = -1 is 4294967295 (the inner
// loop runs; c counts every eighth k): the index -> i32 -> index round trip
// read unsigned is not folded to i.
// CHECK-LABEL: func.func @f06_rt_bound(
// CHECK: %[[B:.*]] = arith.index_castui %{{.*}} : i32 to index
// CHECK-NEXT: scf.for %{{.*}} = %{{.*}} to %[[B]]
// W32-LABEL: func.func @f06_rt_bound(
// W32: affine.for %{{.*}} = 0 to #map(%{{.*}})
void f06_rt_bound(int O[64]) {
  for (int i = -1; i < 3; i++) {
    unsigned c = 0;
    for (unsigned k = 0; k < (unsigned)i; k++)
      c += (k & 7) == 0;
    O[i + 1] = c;
  }
}

// Provably non-negative: raised.
// CHECK-LABEL: func.func @f07_nonneg(
// CHECK-NOT: arith.remui
// CHECK: return
void f07_nonneg(int A[64], int O[64]) {
  for (int i = 0; i < 8; i++)
    O[i] = A[(unsigned long)(unsigned)(i + 3) % 7ul];
}

// CHECK-LABEL: func.func @f08_iv_nonneg(
// CHECK-NOT: arith.remui
// CHECK: return
void f08_iv_nonneg(int A[64], int O[64]) {
  for (int i = 0; i < 8; i++)
    O[i] = A[(unsigned long)(unsigned)i % 7ul];
}

// An unsigned counter against a runtime bound: its index is non-negative.
// CHECK-LABEL: func.func @f09_u_runtime(
// CHECK-NOT: arith.remui
// CHECK: return
void f09_u_runtime(int A[64], int O[64], unsigned n) {
  for (unsigned i = 0; i < n; i++)
    O[i] = A[(unsigned long)i % 7ul];
}

// A 64-bit value read unsigned keeps its bits as an index: raised.
// CHECK-LABEL: func.func @f10_sizet_sum(
// CHECK: affine.load %{{.*}}[%{{.*}} + symbol(%{{.*}})]
void f10_sizet_sum(int A[4096], int O[64], int h) {
  for (int j = 0; j < 4; j++)
    O[j] = A[(size_t)(h + j)];
}
