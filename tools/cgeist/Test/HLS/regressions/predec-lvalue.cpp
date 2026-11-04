// In C++ a prefix decrement is an lvalue, like a prefix increment: the
// condition of `for (i = 8; --i;)` reads it through an lvalue-to-rvalue cast.
// cgeist returned the decremented value instead of the reference and asserted
// in VisitCastExpr (hlsyn-vitis aes `L4: for (i = 8; --i;)`, W8b).
// RUN: cgeist %s %resourcedir --function='*' -S -o - | FileCheck %s

// CHECK-LABEL: func.func @_Z3cntv(
// CHECK:         scf.while
// CHECK:         arith.addi
// CHECK:         return
int cnt() {
  unsigned char i;
  int n = 0;
  for (i = 8; --i;)
    n += 1;
  return n;
}
