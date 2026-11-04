// RUN: cgeist %s --function='*' -S | FileCheck %s
// RUN: cgeist %s --function='*' -S --raise-scf-to-affine | FileCheck %s --check-prefix=AFF

// A value stored before a loop is not forwarded into a load in the loop when
// the loop itself writes the location: the write reaches the load in the next
// iteration.

// CHECK-LABEL: func.func @acc(
// CHECK: affine.store %{{.*}}, %[[O:.*]][1]
// CHECK: scf.for
// CHECK: %[[L:.*]] = affine.load %[[O]][1]
// CHECK: %[[S:.*]] = arith.addi %[[L]], %{{.*}} : i32
// CHECK: affine.store %[[S]], %[[O]][1]
// AFF-LABEL: func.func @acc(
// AFF: affine.for
// AFF: %[[L:.*]] = affine.load %[[O:.*]][1]
// AFF: arith.addi %[[L]], %{{.*}} : i32
void acc(int *a, int *o) {
  o[1] = 5;
  for (int i = 0; i < 64; i++)
    o[1] = o[1] + a[i];
}

// CHECK-LABEL: func.func @acc_while(
// CHECK: scf.while
// CHECK: } do {
// CHECK: %[[L:.*]] = affine.load %[[O:.*]][0]
// CHECK: arith.muli %[[L]]
// AFF-LABEL: func.func @acc_while(
// AFF: scf.while
// AFF: } do {
// AFF: %[[L:.*]] = affine.load %[[O:.*]][0]
// AFF: arith.muli %[[L]]
void acc_while(int *a, int *o) {
  o[0] = 3;
  int i = 0;
  while (a[i] > 0) {
    o[0] = o[0] * 2 + a[i];
    i++;
  }
}

// The loop writes another element only: the stored value is forwarded.
// CHECK-LABEL: func.func @other(
// CHECK: %[[V:.*]] = affine.load %{{.*}}[0]
// CHECK: affine.store %[[V]], %[[O:.*]][0]
// CHECK-NOT: affine.load %[[O]]
// CHECK: return
// AFF-LABEL: func.func @other(
// AFF: %[[V:.*]] = affine.load %{{.*}}[0]
// AFF: affine.store %[[V]], %[[O:.*]][0]
// AFF-NOT: affine.load %[[O]]
// AFF: return
void other(int *a, int *o) {
  o[0] = a[0];
  for (int i = 0; i < 64; i++)
    o[1] = o[0] + a[i];
}
