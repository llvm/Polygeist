// Narrow loop counters, unsigned bounds and `+=` follow C:
//  - a loop whose counter (narrower than int) would wrap before its
//    constant bound's test fails never ends in C and stays scf.while; one
//    provably in range is raised, a countdown from the top of the range
//    included (runtime bounds: narrow-counter-assume-termination.c);
//  - a bound compared unsigned is cast to index with index_castui;
//  - a counter read both zero- and sign-extended across 2^(W-1) stays an
//    iter_arg instead of being folded into the induction variable;
//  - `+=` on an integer LHS goes through the computation type.
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

#include <stdbool.h>
#include <stdint.h>

// 250, 251, .., 255, 0, 1, ..: never ends.
// CHECK-LABEL: func.func @u8_lt300(
// CHECK-NOT: affine.for
// CHECK: scf.while
void u8_lt300(int *A, int *O) {
  for (uint8_t i = 250; i < 300; i++)
    O[i & 63] = A[i & 63];
}

// i < 16 holds for every 4-bit value: never ends.
// CHECK-LABEL: func.func @u4_lt16(
// CHECK-NOT: affine.for
// CHECK: scf.while
void u4_lt16(int *A, int *O) {
  for (unsigned _BitInt(4) i = 0; i < 16; i++)
    O[i] = A[i];
}

// CHECK-LABEL: func.func @u4_lt15(
// CHECK: affine.for %{{.*}} = 0 to 15 {
void u4_lt15(int *A, int *O) {
  for (unsigned _BitInt(4) i = 0; i < 15; i++)
    O[i] = A[i];
}

// Never ends for n > 255: raised by assuming that the loop terminates, so
// n <= 255 (C11 6.8.5p6), and marked hls.uncounted
// (narrow-counter-assume-termination.c).
// CHECK-LABEL: func.func @u8_int_bound(
// CHECK: affine.for %{{.*}} = 0 to %{{.*}} {
// CHECK: } {hls.uncounted}
void u8_int_bound(int *A, int *O, int n) {
  for (uint8_t i = 0; i < n; i++)
    O[i] = A[i];
}

// 0, 2, .., 62: `+=` keeps the counter update an i8 add.
// CHECK-LABEL: func.func @u8_step2(
// CHECK: affine.for %{{.*}} = 0 to 64 step 2 {
void u8_step2(int *A, int *O) {
  for (uint8_t i = 0; i < 64; i += 2)
    O[i] = A[i];
}

// 254 + 2 wraps to 0 < 255: never ends.
// CHECK-LABEL: func.func @u8_step2_255(
// CHECK-NOT: affine.for
// CHECK: scf.while
void u8_step2_255(int *A, int *O) {
  for (uint8_t i = 0; i < 255; i += 2)
    O[i] = A[i];
}

// 255 .. 1: the upper bound is 256, not (uint8_t)(255 + 1) = 0.
// CHECK-LABEL: func.func @u8_top(
// CHECK: affine.for %[[I:.*]] = 1 to 256 {
// CHECK: affine.load %{{.*}}[-%[[I]] + 256]
void u8_top(int *A, int *O) {
  for (uint8_t i = 255; i > 0; i--)
    O[i] = A[i];
}

// `i >= 0` holds for every unsigned value: never ends.
// CHECK-LABEL: func.func @u8_ge0(
// CHECK-NOT: affine.for
// CHECK: scf.while
void u8_ge0(int *A, int *O) {
  for (uint8_t i = 10; i >= 0; i--)
    O[i & 63] = A[i & 63];
}

// An 8-bit unsigned bound of 128 or more is a large bound, not a negative one.
// CHECK-LABEL: func.func @u8_len(
// CHECK: %[[N:.*]] = arith.index_castui %{{.*}} : i8 to index
// CHECK: affine.for %{{.*}} = 0 to %[[N]] {
void u8_len(int *A, int *O, unsigned _BitInt(8) len) {
  for (unsigned _BitInt(8) i = 0; i < len; i++)
    O[i] = A[i];
}

// i runs over 200..209: i - 200 is 0..9 and (int8_t)i is -56..-47, which no
// single index can stand for.
// CHECK-LABEL: func.func @u8_mixed_views(
// CHECK: affine.for %{{.*}} = 200 to 210 iter_args(%[[C:.*]] = %{{.*}}) -> (i8) {
// CHECK-DAG: arith.extui %[[C]] : i8 to i32
// CHECK-DAG: arith.extsi %[[C]] : i8 to i32
void u8_mixed_views(int *A, int *O) {
  for (uint8_t i = 200; i < 210; i++)
    O[i - 200] = A[((int8_t)i) & 0xff];
}

// Read only as (int8_t)i over 200..209 (all >= 128): -56 + k, exact.
// CHECK-LABEL: func.func @u8_signed_view(
// CHECK: affine.for %[[I:.*]] = 200 to 210 {
// CHECK: affine.store %{{.*}}, %{{.*}}[%[[I]] - 156]
void u8_signed_view(int *O) {
  for (uint8_t i = 200; i < 210; i++)
    O[((int8_t)i) + 100] = 1;
}

// CHECK-LABEL: func.func @int_add_double(
// CHECK: arith.sitofp %{{.*}} : i32 to f64
// CHECK: arith.addf
// CHECK: arith.fptosi %{{.*}} : f64 to i32
void int_add_double(int *p) { *p += 1.5; }

// CHECK-LABEL: func.func @uint_add_double(
// CHECK: arith.uitofp %{{.*}} : i32 to f64
// CHECK: arith.addf
// CHECK: arith.fptoui %{{.*}} : f64 to i32
void uint_add_double(unsigned *p) { *p += 1.5; }

// b + 255 is 255 or 256, never 0: the stored bool is the sum tested against
// zero in int, not the sum truncated to 8 bits.
// CHECK-LABEL: func.func @bool_add(
// CHECK: arith.addi %{{.*}} : i32
// CHECK: arith.cmpi ne, %{{.*}} : i32
// CHECK: arith.extui %{{.*}} : i1 to i8
void bool_add(bool *b) { *b += 255; }
