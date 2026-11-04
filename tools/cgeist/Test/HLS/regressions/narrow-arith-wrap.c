// Integer arithmetic narrower than int wraps. An iN add / sub / mul (the
// `+=` / `-=` of a narrow C integer, _BitInt arithmetic), and an index read
// of it, is raised to an affine expression only when it provably does not
// wrap; otherwise the access stays a memref.load of the iN value read with
// its own signedness. A narrow constant is read with the signedness of the
// expression it is in (an unsigned _BitInt(7) 64 is 64, not -64).
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

#include <stdint.h>

// k = (10 - i) mod 256: 266 - i for i > 10.
// CHECK-LABEL: func.func @u8_sub_wrap(
// CHECK: %[[K:.*]] = arith.subi %{{.*}}, %{{.*}} : i8
// CHECK: %[[X:.*]] = arith.index_castui %[[K]] : i8 to index
// CHECK: memref.load %{{.*}}[%[[X]]]
void u8_sub_wrap(int A[256], int O[64]) {
  for (uint8_t i = 0; i < 64; i++) {
    uint8_t k = 10;
    k -= i;
    O[i] = A[k];
  }
}

// 100 - i stays in 37..100: exact.
// CHECK-LABEL: func.func @u8_sub_in_range(
// CHECK: affine.for %[[I:.*]] = 0 to 64 {
// CHECK: affine.load %{{.*}}[-%[[I]] + 100]
void u8_sub_in_range(int A[256], int O[64]) {
  for (uint8_t i = 0; i < 64; i++) {
    uint8_t k = 100;
    k -= i;
    O[i] = A[k];
  }
}

// 250 + i wraps to 0..56 for i >= 6.
// CHECK-LABEL: func.func @u8_add_wrap(
// CHECK: %[[K:.*]] = arith.addi %{{.*}}, %{{.*}} : i8
// CHECK: %[[X:.*]] = arith.index_castui %[[K]] : i8 to index
// CHECK: memref.load %{{.*}}[%[[X]]]
void u8_add_wrap(int A[256], int O[64]) {
  for (uint8_t i = 0; i < 64; i++) {
    uint8_t k = 250;
    k += i;
    O[i] = A[k];
  }
}

// n + 1 is 0 for n = 255: an opaque symbol, not n + 1.
// CHECK-LABEL: func.func @u8_symbol(
// CHECK: %[[K:.*]] = arith.addi %{{.*}}, %{{.*}} : i8
// CHECK: %[[X:.*]] = arith.index_castui %[[K]] : i8 to index
// CHECK: affine.load %{{.*}}[symbol(%[[X]])]
void u8_symbol(int A[256], int *O, uint8_t n) {
  uint8_t k = n;
  k += 1;
  *O = A[k];
}

// 64 + i over 0..63 is 64..127, in the range of an unsigned 7-bit value.
// CHECK-LABEL: func.func @u7_add(
// CHECK: affine.for %[[I:.*]] = 0 to 64 {
// CHECK: affine.load %{{.*}}[%[[I]] + 64]
void u7_add(int A[128], int O[64]) {
  for (unsigned _BitInt(7) i = 0; i < 64; i++) {
    unsigned _BitInt(7) k = 64;
    k += i;
    O[i] = A[k];
  }
}

// 5 - i modulo 128.
// CHECK-LABEL: func.func @u7_sub_wrap(
// CHECK: %[[K:.*]] = arith.subi %{{.*}}, %{{.*}} : i7
// CHECK: %[[X:.*]] = arith.index_castui %[[K]] : i7 to index
// CHECK: memref.load %{{.*}}[%[[X]]]
void u7_sub_wrap(int A[128], int O[64]) {
  for (unsigned _BitInt(7) i = 0; i < 64; i++) {
    unsigned _BitInt(7) k = 5;
    k -= i;
    O[i] = A[k];
  }
}

// `*=` computes in int and truncates: 8 i wraps from i = 32.
// CHECK-LABEL: func.func @u8_mul_wrap(
// CHECK: arith.muli %{{.*}} : i32
// CHECK: %[[K:.*]] = arith.trunci %{{.*}} : i32 to i8
// CHECK: %[[X:.*]] = arith.index_castui %[[K]] : i8 to index
// CHECK: memref.load %{{.*}}[%[[X]]]
void u8_mul_wrap(int A[256], int O[64]) {
  for (uint8_t i = 0; i < 64; i++) {
    uint8_t k = i;
    k *= 8;
    O[i] = A[k];
  }
}

// C's remainder of a negative i is negative (-4..4), not the affine mod
// (0..4).
// CHECK-LABEL: func.func @s8_rem(
// CHECK: arith.remsi %{{.*}}, %{{.*}} : i8
// CHECK: memref.load
void s8_rem(int A[256], int O[64]) {
  for (signed _BitInt(8) i = -32; i < 32; i++) {
    signed _BitInt(8) k = i % (signed _BitInt(8))5;
    O[i + 32] = A[k + 8];
  }
}
