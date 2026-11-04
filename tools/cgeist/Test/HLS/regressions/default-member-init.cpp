// A member that a braced initialiser leaves out takes its default member
// initialiser (C++11 `uint32_t id_{ 0 };`). cgeist visited the
// CXXDefaultInitExpr in the init list, which needs a `this` the aggregate
// does not have (assertion `ThisVal.val`, class_ece8893_2023_spring
// FocusedACO, W8b); it now evaluates the initialiser's own expression.
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

struct seg {
  unsigned first;
  unsigned last;
  unsigned id = 7;
};

// CHECK-LABEL: func.func @top(
// CHECK-DAG:     arith.constant 7 : i32
// CHECK:         return
extern "C" unsigned top(unsigned a, unsigned b) {
  seg s = {a, b};
  return s.first + s.last + s.id;
}
