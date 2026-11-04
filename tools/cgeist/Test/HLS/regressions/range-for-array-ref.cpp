// A range-based for over an array reference parameter: the deduced
// `auto &&__range` is an array type behind an AutoType; getting its shape
// asserted (cast<ConstantArrayType> on the sugared type; OPEN_ISSUES O91, the
// two cgeist crashes under the harness flags, reproducer c_rangefor.cpp).
// RUN: cgeist %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// CHECK-LABEL: func.func @_Z9range_forRA8_i(%arg0: memref<8xi32>)
// CHECK: scf.while
// CHECK: affine.load
// CHECK: arith.addi
// CHECK: affine.store
void range_for(int (&a)[8]) {
  for (int &x : a)
    x += 1;
}

// A nested range-based for over the rows of a 2-D array reference: the row
// (`auto &row = *__begin`) is the pointer to it, one rank above the row, and
// its decay to `int *` is the pointer's row 0 (W9 round 8 review defect 8:
// before, the decay kept the extra rank and `int v = *__begin2` asserted in
// ValueCategory::getValue; probe rf2.cpp, bit-exact under check_semantics).
// CHECK-LABEL: func.func @_Z4rowsRA4_A8_i(%arg0: memref<4x8xi32>)
// CHECK: scf.while
// CHECK: "polygeist.subindex"(%{{.*}}, %{{.*}}) : (memref<?x8xi32>, index) -> memref<?xi32>
// CHECK: scf.while
// CHECK: affine.load %{{.*}}[0] : memref<?xi32>
int rows(int (&m)[4][8]) {
  int s = 0;
  for (auto &row : m)
    for (int v : row)
      s += v;
  return s;
}
