// With --scalar-wrapper-abi a struct of one scalar member is that scalar; a
// braced initialiser of it (`return V{a.x + b.x};`) initialises the scalar
// (before, assert(0) in InitializeValueByInitListExpr; OPEN_ISSUES O91,
// reproducer c_initlist.cpp).
// RUN: cgeist %s %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --scalar-wrapper-abi -o - | FileCheck %s
// CHECK-LABEL: func.func @_Z3add1VS_(%arg0: i32, %arg1: i32) -> i32
// CHECK: %[[S:.*]] = arith.addi %arg0, %arg1 : i32
// CHECK: return %[[S]] : i32
struct V {
  int x;
};
V add(V a, V b) { return V{a.x + b.x}; }
