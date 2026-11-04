// RUN: cgeist %s --function='*' -S --raise-scf-to-affine | FileCheck %s

// A loop counter used after the loop holds start + trips * step, with
// trips = max(0, ceil((ub - lb) / step)).

// 0, 3, 6, 9, then 12 fails the test.
// CHECK-LABEL: func.func @step3(
// CHECK: %[[C:.*]] = arith.constant 12 : i32
// CHECK: affine.store %[[C]], %{{.*}}[0]
void step3(int *A, int *O) {
  int i;
  for (i = 0; i < 10; i += 3)
    A[i] = 0;
  *O = i;
}

// n <= 0: the loop does not run and i stays 0.
// CHECK-LABEL: func.func @runtime_bound(
// CHECK: %[[N:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: %[[T:.*]] = arith.maxsi %[[N]], %{{.*}} : index
// CHECK: %[[V:.*]] = arith.index_cast %[[T]] : index to i32
// CHECK: affine.store %[[V]], %{{.*}}[0]
void runtime_bound(int *A, int *O, int n) {
  int i;
  for (i = 0; i < n; i++)
    A[i] = 0;
  *O = i;
}

// s >= 100: the loop does not run and i stays s.
// CHECK-LABEL: func.func @runtime_start(
// CHECK: %[[S:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: %[[D:.*]] = arith.subi %{{.*}}, %[[S]] : index
// CHECK: %[[T:.*]] = arith.maxsi %[[D]], %{{.*}} : index
// CHECK: %[[E:.*]] = arith.addi %[[S]], %[[T]] : index
// CHECK: %[[V:.*]] = arith.index_cast %[[E]] : index to i32
// CHECK: affine.store %[[V]], %{{.*}}[0]
void runtime_start(int *A, int *O, int s) {
  int i;
  for (i = s; i < 100; i++)
    A[i] = 0;
  *O = i;
}
