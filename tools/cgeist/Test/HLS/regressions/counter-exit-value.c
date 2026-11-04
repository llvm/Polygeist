// The value a counter leaves its loop with is start + trips * step, with
// trips = max(0, ceil((ub - lb) / step)): the start when the loop does not
// run, a last partial step counted whole.
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

#include <stdint.h>

// 0, 3, 6, 9, then 12 fails the test.
// CHECK-LABEL: func.func @u8_step3(
// CHECK: %[[C:.*]] = arith.constant 12 : i8
// CHECK: affine.store %[[C]], %{{.*}}[0]
void u8_step3(int A[256], uint8_t *O) {
  uint8_t i;
  for (i = 0; i < 10; i += 3)
    A[i] = 0;
  *O = i;
}

// s >= 100: the loop does not run and i stays s.
// CHECK-LABEL: func.func @u8_runtime_start(
// CHECK: %[[S:.*]] = arith.index_castui %{{.*}} : i8 to index
// CHECK: %[[D:.*]] = arith.subi %{{.*}}, %[[S]] : index
// CHECK: %[[T:.*]] = arith.maxsi %[[D]], %{{.*}} : index
// CHECK: %[[E:.*]] = arith.addi %[[S]], %[[T]] : index
// CHECK: %[[V:.*]] = arith.index_cast %[[E]] : index to i8
// CHECK: affine.store %[[V]], %{{.*}}[0]
void u8_runtime_start(int A[256], uint8_t *O, uint8_t s) {
  uint8_t i;
  for (i = s; i < 100; i++)
    A[i] = 0;
  *O = i;
}

// CHECK-LABEL: func.func @int_step4(
// CHECK: %[[T:.*]] = arith.ceildivsi %{{.*}}, %[[C4:.*]] : index
// CHECK: %[[M:.*]] = arith.maxsi %[[T]], %{{.*}} : index
// CHECK: %[[E:.*]] = arith.muli %[[M]], %[[C4]] : index
// CHECK: %[[V:.*]] = arith.index_cast %[[E]] : index to i32
// CHECK: affine.store %[[V]], %{{.*}}[0]
void int_step4(int A[256], int *O, int n) {
  int i;
  for (i = 0; i < n; i += 4)
    A[i] = 0;
  *O = i;
}

// An unsigned bound is never below the start 0: no max.
// CHECK-LABEL: func.func @unsigned_bound(
// CHECK-NOT: arith.maxsi
// CHECK: affine.store %{{.*}}, %{{.*}}[0]
void unsigned_bound(int A[256], unsigned *O, unsigned n) {
  unsigned i;
  for (i = 0; i < n; i++)
    A[i] = 0;
  *O = i;
}
