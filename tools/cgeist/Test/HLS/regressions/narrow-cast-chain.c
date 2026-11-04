// Casts are folded into an affine index only when they keep the value: a
// sign or zero extension read with its own signedness, a truncation of an
// index that fits the narrow type as it is read. `(uint16_t)i` of an int8_t
// -8 is 65528, `(uint8_t)i` of it 248. A narrow counter passed to a call is
// read as the callee reads it, which inlining exposes after the counter has
// been folded, so the counter stays an iter_arg unless its values fit both
// signednesses.
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

#include <stdint.h>

// CHECK-LABEL: func.func @s8_to_u16(
// CHECK: affine.for %[[I:.*]] = -8 to 8 {
// CHECK: %[[T:.*]] = arith.index_cast %[[I]] : index to i8
// CHECK: %[[E:.*]] = arith.extsi %[[T]] : i8 to i16
// CHECK: %[[X:.*]] = arith.index_castui %[[E]] : i16 to index
// CHECK: memref.load %{{.*}}[%[[X]]]
// CHECK: affine.store %{{.*}}, %{{.*}}[%[[I]] + 8]
void s8_to_u16(int A[65536], int O[16]) {
  for (int8_t i = -8; i < 8; i++)
    O[i + 8] = A[(uint16_t)i];
}

// 0..15: every reading agrees.
// CHECK-LABEL: func.func @s8_to_u16_in_range(
// CHECK: affine.for %[[I:.*]] = 0 to 16 {
// CHECK: affine.load %{{.*}}[%[[I]]]
void s8_to_u16_in_range(int A[65536], int O[16]) {
  for (int8_t i = 0; i < 16; i++)
    O[i] = A[(uint16_t)i];
}

static inline int read_u8(int A[256], uint8_t k) { return A[k]; }
static inline int read_s8(int A[256], int8_t k) { return A[k + 128]; }

// The callee reads -8..-1 as 248..255.
// CHECK-LABEL: func.func @call_s8_as_u8(
// CHECK: affine.for %{{.*}} = -8 to 8 iter_args(%[[C:.*]] = %{{.*}}) -> (i8) {
// CHECK: %[[X:.*]] = arith.index_castui %[[C]] : i8 to index
// CHECK: memref.load %{{.*}}[%[[X]]]
void call_s8_as_u8(int A[256], int O[16]) {
  for (int8_t i = -8; i < 8; i++)
    O[i + 8] = read_u8(A, i);
}

// The callee reads 128..135 as -128..-121.
// CHECK-LABEL: func.func @call_u8_as_s8(
// CHECK: affine.for %{{.*}} = 120 to 136 iter_args(%[[C:.*]] = %{{.*}}) -> (i8) {
// CHECK: arith.index_cast %[[C]] : i8 to index
// CHECK: memref.load
void call_u8_as_s8(int A[256], int O[16]) {
  for (uint8_t i = 120; i < 136; i++)
    O[i - 120] = read_s8(A, i);
}

// 0..15 reads the same either way.
// CHECK-LABEL: func.func @call_in_range(
// CHECK: affine.for %[[I:.*]] = 0 to 16 {
// CHECK-DAG: affine.load %{{.*}}[%[[I]]]
// CHECK-DAG: affine.load %{{.*}}[%[[I]] + 128]
void call_in_range(int A[256], int O[16]) {
  for (uint8_t i = 0; i < 16; i++)
    O[i] = read_u8(A, i) + read_s8(A, i);
}
