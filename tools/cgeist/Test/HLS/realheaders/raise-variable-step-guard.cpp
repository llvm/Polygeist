// The trip count of a variable-step loop is computed above the guards and
// loops that protect the loop (raise-variable-step.cpp), so it must not trap
// where the loop never runs: a zero step divides by one and an empty range
// gives 0. Before, `if (s > 0) for (i = 0; i < n; i += s)` inside another
// loop divided by zero for s == 0 (SIGFPE where the native build returns;
// W9r7 review defect 1, check_semantics probe vs_if: 60 runs bit-exact).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

// CHECK-LABEL: func.func @_Z5vs_ifPiii
// CHECK: %[[NONEMPTY:.*]] = arith.cmpi sgt
// CHECK: %[[NUM:.*]] = arith.select %[[NONEMPTY]], %{{.*}}, %{{.*}} : index
// CHECK: %[[ZERO:.*]] = arith.cmpi eq
// CHECK: %[[DEN:.*]] = arith.select %[[ZERO]], %{{.*}}, %{{.*}} : index
// CHECK: %[[TRIP:.*]] = arith.divui %[[NUM]], %[[DEN]] : index
// CHECK: affine.for
// CHECK: affine.if
// CHECK: affine.for %{{.*}} = 0 to %[[TRIP]]
void vs_if(int out[64], int s, int n) {
  for (int j = 0; j < 4; j++) {
    if (s > 0) {
      for (int i = 0; i < n; i += s)
        out[i & 63] += j + 1;
    }
  }
}
