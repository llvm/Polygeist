// A loop whose counter steps by itself (`m += m` doubles it) is no constant-
// step loop: the while-to-for conversion must leave it a while loop. It used
// to build an scf.for whose step was its own iteration argument ("operand #0
// does not dominate this use", machsuite sort_merge, W8b). An inner loop whose
// step is a value computed in the outer body still becomes an scf.for.
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

// CHECK-LABEL: func.func @top(
// CHECK:         %[[M:.*]] = scf.while (%[[A:.*]] = %{{.*}}) : (i32) -> i32 {
// CHECK:           arith.cmpi slt, %[[A]], %{{.*}} : i32
// CHECK:         } do {
// CHECK:         ^bb0(%[[B:.*]]: i32):
// CHECK:           %[[D:.*]] = arith.addi %[[B]], %[[B]] : i32
// CHECK:           %[[S:.*]] = arith.index_cast %[[D]] : i32 to index
// CHECK:           scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[S]] {
// CHECK:           scf.yield %[[D]] : i32
void top(int a[64]) {
  int m, i;
  for (m = 1; m < 64; m += m)
    for (i = 0; i < 64; i += m + m)
      a[i] += m;
}
