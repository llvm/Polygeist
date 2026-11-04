// XlxLoopScalarPromote on an accumulator each iteration of the outer loops
// resets before use (hyungseok lab2's inner product): the outer loops carry
// no value into their bodies (the bodies start from an undefined value, as
// LLVM's mem2reg builds no live phi there; the first version chained the
// i loop's value into the j loop), and the pass runs after LICM, so
// the inner loop, now free of memory effects and invariant in j, stays inside
// the j loop (Vitis never hoists a loop; before, LICM moved it out).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <ap_fixed.h>
typedef ap_fixed<16, 5> T;

void dot_top(T q[4][128], T w[128], T out[4][100]) {
  T sum;
  for (int i = 0; i < 4; i++)
    for (int j = 0; j < 100; j++) {
      sum = 0;
      for (int d = 0; d < 128; d++)
        sum += q[i][d] * w[d];
      out[i][j] = sum * j;
    }
}

// CHECK-LABEL: func.func @_Z7dot_top
// CHECK: affine.for %{{.*}} = 0 to 4 iter_args(
// CHECK-NEXT: %[[U:.*]] = llvm.mlir.undef : i16
// CHECK-NEXT: affine.for %{{.*}} = 0 to 100 iter_args(%{{.*}} = %[[U]])
// CHECK: affine.for %{{.*}} = 0 to 128 iter_args
// CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<4x100xi16>
