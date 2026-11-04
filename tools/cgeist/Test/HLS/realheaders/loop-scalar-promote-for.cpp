// XlxLoopScalarPromote through scf.for and scf.if: an ap_uint accumulator
// updated under a condition in a loop with a run-time trip count. The pass
// adds an iteration argument to the loop (replaceWithAdditionalYields runs
// the yield callback before it moves the body into the new loop; the value is
// threaded through the body that holds the yield) and a result to the scf.if
// (accelerators/GP_selection: an assertion before the fix).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <ap_int.h>

int acc_top(int n, int a[64]) {
  ap_uint<12> s = 0;
  for (int i = 0; i < n; i++)
    if (a[i] > 0)
      s += a[i];
  return s;
}

// CHECK-LABEL: func.func @_Z7acc_topiPi
// CHECK-NOT: memref.alloca() : memref<i12>
// CHECK: affine.for {{.*}} iter_args({{.*}}) -> (i12) {
// CHECK: scf.if {{.*}} -> (i12) {
// CHECK: return
