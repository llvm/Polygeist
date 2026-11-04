// A static data member named through an object (`t.N`, Vitis_Libraries
// solver svd.hpp) is the variable itself; cgeist asserted in
// cast<FieldDecl> before (CIRCT OPEN_ISSUES O219 crashes).
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

// CHECK: memref.global @_ZN6Traits1NE : memref<1xi32> = dense<12>
// CHECK: func.func @_Z3topi
// CHECK: memref.get_global @_ZN6Traits1NE
struct Traits { static const int N = 12; int pad; };
int top(int x) {
  Traits t;
  t.pad = x;
  return x * t.N + t.pad;
}
