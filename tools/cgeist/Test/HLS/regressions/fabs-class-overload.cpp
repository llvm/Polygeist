// cgeist emits `fabs` / `fabsf` of a number as math.absf (or a select for an
// integer). An overload of the same name for a class (`fabs(fixed)`, as the
// Vitis hls_math headers declare for ap_fixed) is an ordinary call: its
// argument was taken for an integer (assertion `cast<IntegerType>`, flowgnn
// dgn, accelerators GP_selection, W8b).
// RUN: cgeist %s %resourcedir --function='*' -S --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <math.h>

struct fixed { int raw; };
fixed fabs(fixed x) {
  fixed r;
  r.raw = x.raw < 0 ? -x.raw : x.raw;
  return r;
}

// CHECK-LABEL: func.func @top(
// CHECK:         math.absf
// CHECK-NOT:     math.absf
// CHECK:         arith.cmpi slt, %{{.*}}, %{{.*}} : i32
// CHECK:         arith.subi
// CHECK:         return
extern "C" int top(float a, int b, float *out) {
  out[0] = fabsf(a);
  fixed x;
  x.raw = b;
  return fabs(x).raw;
}
