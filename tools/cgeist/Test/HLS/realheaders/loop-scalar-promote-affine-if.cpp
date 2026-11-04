// XlxLoopScalarPromote through an affine.if (the pass runs after
// AffineCFG): an ap_uint accumulator updated under a condition on the loop
// counter. The affine.if gets a result and an else region yielding the value
// (an affine.if with results needs one: the first version asserted in
// AffineIfOp::build, vitis_examples perfect_nested_loops, flowgnn x 5).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <ap_int.h>

int aif_top(int a[64]) {
  ap_uint<12> s = 0;
  for (int i = 0; i < 64; i++)
    if (i < 32)
      s += a[i];
  return s;
}

// CHECK-LABEL: func.func @_Z7aif_topPi
// CHECK-NOT: memref.alloca() : memref<i12>
// CHECK: affine.for {{.*}} iter_args({{.*}}) -> (i12) {
// CHECK: affine.if {{.*}} -> i12 {
// CHECK: } else {
// CHECK-NEXT: affine.yield %{{.*}} : i12
