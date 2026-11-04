// A builtin that clang evaluates to a constant (INFINITY is
// `__builtin_inff()`) is emitted as the constant; it used to stay a call of
// an undefined function `__builtin_inff`, which cannot be linked (W8b).
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <math.h>

// CHECK-LABEL: func.func @top(
// CHECK:         %[[NINF:.*]] = arith.constant 0xFF800000 : f32
// CHECK-NOT:     call @__builtin_inff
// CHECK:         affine.for {{.*}} iter_args(%{{.*}} = %[[NINF]]) -> (f32)
float top(float a[8]) {
  float m = -INFINITY;
  for (int i = 0; i < 8; i++)
    if (a[i] > m) m = a[i];
  return m;
}
