// C's signed integer +, -, * at least as wide as int is undefined on overflow
// (C11 6.5p5), so cgeist marks such an arith.addi / subi / muli polygeist.nsw
// and AffineCFG raises it to an affine expression without a range proof.
// Unsigned arithmetic wraps and stays unmarked: it is raised only when it
// provably keeps its value, at every width (an affine expression is
// mathematics: (i * c) / c simplifies to i, which the wrapped C value is
// not). The mark is removed from the output unless -keep-frontend-assumptions
// is given; -affine-cfg-assume-nowrap-width=32 restores the former width rule
// (every 32-bit op raised without a proof) for comparison.
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s --implicit-check-not=polygeist.nsw
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -keep-frontend-assumptions -o - | FileCheck %s --check-prefix=MARK
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -affine-cfg-assume-nowrap-width=32 -o - | FileCheck %s --check-prefix=W32

#include <stdint.h>

// 3 * (2^31 - 1) wraps to 2147483645, so C reads A[0] at i = 3; raised as i,
// the former default read A[3].
// CHECK-LABEL: func.func @u32_mul_div(
// CHECK: %[[M:.*]] = arith.muli %{{[^ ]+}}, %{{[^ ]+}} : i32
// CHECK: %[[D:.*]] = arith.divui %[[M]], %{{[^ ]+}} : i32
// CHECK: %[[X:.*]] = arith.index_castui %[[D]] : i32 to index
// CHECK: memref.load %{{.*}}[%[[X]]]
// W32-LABEL: func.func @u32_mul_div(
// W32: affine.for %[[I:.*]] = 0 to 4 {
// W32: affine.load %{{.*}}[%[[I]]]
void u32_mul_div(int A[8], int O[8]) {
  for (unsigned i = 0; i < 4; i++)
    O[i] = A[(i * 0x7FFFFFFFu) / 0x7FFFFFFFu];
}

// The same at 64 bits: an i64 op is not index arithmetic either.
// CHECK-LABEL: func.func @u64_mul_div(
// CHECK: arith.muli %{{[^ ]+}}, %{{[^ ]+}} : i64
// CHECK: arith.divui
// CHECK: memref.load
void u64_mul_div(int A[8], int O[8]) {
  for (unsigned long i = 0; i < 4; i++)
    O[i] = A[(i * 0x7FFFFFFFFFFFFFFFul) / 0x7FFFFFFFFFFFFFFFul];
}

// int arithmetic on an unbounded argument: marked, raised without a proof.
// CHECK-LABEL: func.func @s32_arg(
// CHECK-NOT: memref.load
// CHECK: affine.load %{{.*}}[%{{.*}} + symbol(%{{.*}}) * 4 + 2000]
// CHECK: return
void s32_arg(int A[4096], int O[64], int h) {
  for (int j = 0; j < 4; j++)
    O[j] = A[h * 4 + j + 2000];
}

// The unsigned counterpart has no proof (h * 4 wraps for h >= 2^30).
// CHECK-LABEL: func.func @u32_arg(
// CHECK: arith.muli %{{[^ ]+}}, %{{[^ ]+}} : i32
// CHECK: memref.load
// W32-LABEL: func.func @u32_arg(
// W32-NOT: memref.load
// W32: affine.load
void u32_arg(int A[4096], int O[64], unsigned h) {
  for (int j = 0; j < 4; j++)
    O[j] = A[h * 4u + j + 2000u];
}

// Compound assignments compute in int: marked. The canonicaliser folds
// `k -= 1` into `k + (-1)` and that with `+ 2000` into `k + 1999`; the folds
// keep the mark.
// CHECK-LABEL: func.func @s32_compound(
// CHECK-NOT: memref.load
// CHECK: affine.load %{{.*}}[%{{.*}} + symbol(%{{.*}}) * 4 + 1999]
// CHECK: return
void s32_compound(int A[4096], int O[64], int h) {
  for (int j = 0; j < 4; j++) {
    int k = h;
    k *= 4;
    k += j;
    k -= 1;
    O[j] = A[k + 2000];
  }
}

// A fold whose constant overflows (2^30 + 2^30) is left to the arith
// patterns, unmarked: h + (-2^31) is not h + 2^31. The unmarked sum, hoisted
// out of the loop, is an opaque symbol of the access, not part of its map.
// CHECK-LABEL: func.func @s32_const_overflow(
// CHECK: %[[S:.*]] = arith.addi %{{[^ ]+}}, %{{[^ ]+}} : i32
// CHECK: %[[X:.*]] = arith.index_cast %[[S]] : i32 to index
// CHECK: affine.load %{{.*}}[%{{.*}} + symbol(%[[X]])]
void s32_const_overflow(int A[4096], int O[64], int h) {
  for (int j = 0; j < 4; j++)
    O[j] = A[h + 0x40000000 + 0x40000000 + j];
}

// `h * 2 * 3` is folded into `h * 6`, marked.
// CHECK-LABEL: func.func @s32_mul_chain(
// CHECK-NOT: memref.load
// CHECK: affine.load %{{.*}}[%{{.*}} + symbol(%{{.*}}) * 6 + 2000]
// CHECK: return
void s32_mul_chain(int A[4096], int O[64], int h) {
  for (int j = 0; j < 4; j++)
    O[j] = A[h * 2 * 3 + j + 2000];
}

// long, unary minus and ++ / -- on int: marked.
// CHECK-LABEL: func.func @s64_arg(
// CHECK-NOT: memref.load
// CHECK: affine.load %{{.*}}[%{{.*}} + symbol(%{{.*}}) * 4 + 2000]
// CHECK: return
void s64_arg(int A[4096], int O[64], long h) {
  for (long j = 0; j < 4; j++)
    O[j] = A[h * 4 + j + 2000];
}

// The invariant `-h` is hoisted out of the loop and read as a symbol.
// CHECK-LABEL: func.func @s32_neg_inc(
// CHECK-NOT: memref.load
// CHECK: %[[N:.*]] = arith.subi %{{[^ ]+}}, %{{[^ ]+}} : i32
// CHECK: %[[X:.*]] = arith.index_cast %[[N]] : i32 to index
// CHECK-NOT: memref.load
// CHECK: affine.load %{{.*}}[%{{.*}} * 3 + symbol(%[[X]]) + 2002]
// CHECK: return
void s32_neg_inc(int A[4096], int O[64], int h) {
  for (int j = 0; j < 4; j++) {
    int k = j;
    k++;
    ++k;
    k--;
    O[j] = A[-h + k * 3 - 1 + 2000];
  }
}

// A promoted narrow operand is unmarked; its range proves the int op.
// CHECK-LABEL: func.func @s16_promoted(
// CHECK-NOT: memref.load
// CHECK: affine.load %{{.*}}[%{{.*}} + symbol(%{{.*}}) * 3 + 2000]
// CHECK: return
void s16_promoted(int A[4096], int O[64], short h) {
  for (int j = 0; j < 4; j++)
    O[j] = A[h * 3 + j + 2000];
}

// Which ops carry the mark (-keep-frontend-assumptions): signed int and long
// arithmetic, compound assignments and ++ / --; not unsigned, promoted narrow
// or narrow _BitInt arithmetic.
// MARK-LABEL: func.func @f_s(
// MARK: %[[M:.*]] = arith.muli %{{[^ ]+}}, %{{[^ ]+}} {polygeist.nsw} : i32
// MARK: arith.addi %[[M]], %{{[^ ]+}} {polygeist.nsw} : i32
int f_s(int a, int b) { return a * b + 1; }

// MARK-LABEL: func.func @f_u(
// MARK: %[[M:.*]] = arith.muli %{{[^ ]+}}, %{{[^ ]+}} : i32
// MARK: arith.addi %[[M]], %{{[^ ]+}} : i32
unsigned f_u(unsigned a, unsigned b) { return a * b + 1u; }

// MARK-LABEL: func.func @f_n(
// MARK: %[[M:.*]] = arith.muli %{{[^ ]+}}, %{{[^ ]+}} : i32
// MARK: arith.subi %[[M]], %{{[^ ]+}} : i32
int f_n(short a, short b) { return a * b - a; }

// `-(r - 1)` is folded into `1 - r`, marked.
// MARK-LABEL: func.func @f_l(
// MARK: %[[A:.*]] = arith.addi %{{[^ ]+}}, %{{[^ ]+}} {polygeist.nsw} : i64
// MARK: %[[M:.*]] = arith.muli %[[A]], %{{[^ ]+}} {polygeist.nsw} : i64
// MARK: %[[S:.*]] = arith.subi %{{[^ ]+}}, %[[M]] {polygeist.nsw} : i64
// MARK-NEXT: return %[[S]]
long f_l(long a, int b) {
  long r = a;
  r += b;
  r *= 3;
  r -= 1;
  return -r;
}

// x + 1 + 1 - 1 folds to x + 1, marked.
// MARK-LABEL: func.func @f_inc(
// MARK: %[[A:.*]] = arith.addi %{{[^ ]+}}, %{{[^ ]+}} {polygeist.nsw} : i32
// MARK-NEXT: return %[[A]]
int f_inc(int a) {
  int x = a;
  x++;
  ++x;
  x--;
  return x;
}

// MARK-LABEL: func.func @f_bitint8(
// MARK: arith.addi %{{[^ ]+}}, %{{[^ ]+}} : i8
signed _BitInt(8) f_bitint8(signed _BitInt(8) a, signed _BitInt(8) b) {
  signed _BitInt(8) k = a;
  k += b;
  return k;
}
