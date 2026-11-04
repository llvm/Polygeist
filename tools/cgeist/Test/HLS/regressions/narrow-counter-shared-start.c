// A second narrow counter that starts at the same value as the loop's own
// counter (the same SSA value: a CSE'd constant, a shared runtime start) but
// steps differently is folded into the induction variable only when its
// values provably stay in range. The loop's own counter, or one stepping
// with it, keeps the range the while loop's test proved.
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

#include <stdint.h>

// j = 0, 30, .., 240, then 270 wraps to 14.
// CHECK-LABEL: func.func @shared_const(
// CHECK: affine.for %{{.*}} = 0 to 10 iter_args(%[[J:.*]] = %{{.*}}) -> (i8) {
// CHECK: %[[X:.*]] = arith.index_castui %[[J]] : i8 to index
// CHECK: memref.store %{{.*}}, %{{.*}}[%[[X]]]
void shared_const(int O[300]) {
  for (uint8_t i = 0, j = 0; i < 10; i++, j += 30)
    O[j] = i;
}

// j = 0, 255, 254, ..
// CHECK-LABEL: func.func @shared_down(
// CHECK: affine.for %{{.*}} = 0 to 10 iter_args(%[[J:.*]] = %{{.*}}) -> (i8) {
// CHECK: %[[X:.*]] = arith.index_castui %[[J]] : i8 to index
// CHECK: memref.store %{{.*}}, %{{.*}}[%[[X]]]
void shared_down(int O[256]) {
  for (uint8_t i = 0, j = 0; i < 10; i++, j--)
    O[j] = i;
}

// j = s + 30 k wraps once it passes 255.
// CHECK-LABEL: func.func @shared_runtime(
// CHECK: affine.for %{{.*}} = %{{.*}} to 10 iter_args(%[[J:.*]] = %{{.*}}) -> (i8) {
// CHECK: %[[X:.*]] = arith.index_castui %[[J]] : i8 to index
// CHECK: memref.store %{{.*}}, %{{.*}}[%[[X]]]
void shared_runtime(int O[300], uint8_t s) {
  for (uint8_t i = s, j = s; i < 10; i++, j += 30)
    O[j] = i;
}

// i counts down 200..101 while j counts up 200..299, which wraps at 256.
// CHECK-LABEL: func.func @countdown_shared(
// CHECK: affine.for %{{.*}} = 101 to 201 iter_args(%[[J:.*]] = %{{.*}}) -> (i8) {
// CHECK: %[[X:.*]] = arith.index_castui %[[J]] : i8 to index
// CHECK: memref.store %{{.*}}, %{{.*}}[%[[X]]]
void countdown_shared(int O[256]) {
  for (uint8_t i = 200, j = 200; i > 100; i--, j += 1)
    O[j] = i;
}

// j steps with i: the same values.
// CHECK-LABEL: func.func @shared_same_step(
// CHECK: affine.for %[[I:.*]] = 0 to 10 {
// CHECK: affine.store %{{.*}}, %{{.*}}[%[[I]]]
void shared_same_step(int O[256]) {
  for (uint8_t i = 0, j = 0; i < 10; i++, j++)
    O[j] = i;
}

// j = 20 k stays in 0..180: exact.
// CHECK-LABEL: func.func @shared_in_range(
// CHECK: affine.for %[[I:.*]] = 0 to 10 {
// CHECK: affine.store %{{.*}}, %{{.*}}[%[[I]] * 20]
void shared_in_range(int O[256]) {
  for (uint8_t i = 0, j = 0; i < 10; i++, j += 20)
    O[j] = i;
}

// The loop's own counter from a runtime start: n, n - 1, .., 1, in range.
// CHECK-LABEL: func.func @own_countdown(
// CHECK: affine.for %[[I:.*]] = 1 to #{{.*}}()[%[[N:.*]]] {
// CHECK: affine.load %{{.*}}[-%[[I]] + symbol(%[[N]]) + 1]
void own_countdown(int A[256], int O[256], uint8_t n) {
  for (uint8_t i = n; i > 0; i--)
    O[i] = A[i];
}
