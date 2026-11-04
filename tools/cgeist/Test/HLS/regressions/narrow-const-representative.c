// An iN constant stands for every integer congruent to it modulo 2^N, and an
// iN add / sub / mul gives the same bits with any of them. MLIR folds
// `k - 5` into `k + (-5)`, and the i8 -5 read unsigned is 251, so a narrow
// decrement is raised with the representative that keeps the exact result in
// the range of the iN as it is read (-5, not 251), when there is one.
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

#include <stdint.h>

// k = i - 5 for i in [5, 99] is 0..94: exact.
// CHECK-LABEL: func.func @u8_dec_in_range(
// CHECK: affine.for %[[I:.*]] = 5 to 100 {
// CHECK: affine.load %{{.*}}[%[[I]] - 5]
void u8_dec_in_range(int A[256], int O[256]) {
  for (uint8_t i = 5; i < 100; i++) {
    uint8_t k = i;
    k -= 5;
    O[i] = A[k];
  }
}

// i - 5 wraps to 251..255 for i < 5: no representative fits.
// CHECK-LABEL: func.func @u8_dec_wrap(
// CHECK: %[[K:.*]] = arith.addi %{{.*}}, %{{.*}} : i8
// CHECK: %[[X:.*]] = arith.index_castui %[[K]] : i8 to index
// CHECK: memref.load %{{.*}}[%[[X]]]
void u8_dec_wrap(int A[256], int O[256]) {
  for (uint8_t i = 0; i < 100; i++) {
    uint8_t k = i;
    k -= 5;
    O[i] = A[k];
  }
}

// A countdown from a runtime n: i is in [1, n], k = i - 1 in [0, n - 1].
// CHECK-LABEL: func.func @u8_countdown_dec(
// CHECK: affine.for %[[I:.*]] = 1 to #{{.*}}()[%[[N:.*]]] {
// CHECK: affine.load %{{.*}}[-%[[I]] + symbol(%[[N]])]
// CHECK: affine.store %{{.*}}, %{{.*}}[-%[[I]] + symbol(%[[N]])]
void u8_countdown_dec(int A[256], int O[256], int s) {
  uint8_t n = s;
  for (uint8_t i = n; i > 0; i--) {
    uint8_t k = i;
    k -= 1;
    O[k] = A[k];
  }
}

// int8 k = i + 128 for i in [-128, -101]: the i8 constant is -128, read
// signed i - 128 does not fit, i + 128 (0..27) does.
// CHECK-LABEL: func.func @s8_add_128(
// CHECK: affine.for %[[I:.*]] = -128 to -100 {
// CHECK: affine.load %{{.*}}[%[[I]] + 128]
void s8_add_128(int A[256], int O[64]) {
  for (int8_t i = -128; i < -100; i++) {
    int8_t k = i;
    k += 128;
    O[i + 128] = A[k];
  }
}
