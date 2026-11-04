// `out[0] += a[i] * b[i]` on ap_fixed<16,5> through the real headers: the sum
// is the header's RType plus width (ap_fixed<33,11>), and the guard store of
// the conversion back to ap_fixed<16,5> (etc/ap_fixed_base.h:691
// `if (!op) Base::V = 0;`, overwritten by the unconditional assignment that
// follows) is gone, as in Vitis (forgebench/mult_op_module_dot: no
// conditional store in db/a.o.1.ll's dot_product).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <ap_fixed.h>

extern "C" {
// CHECK-LABEL: func.func @dot(%arg0: memref<64xi16>, %arg1: memref<64xi16>, %arg2: memref<1xi16>)
// CHECK: affine.for
// CHECK: arith.muli {{.*}} : i32
// CHECK: arith.addi {{.*}} : i33
// CHECK-NOT: scf.if
// CHECK: affine.store {{.*}} : memref<1xi16>
// CHECK-NOT: scf.if
// CHECK: return
void dot(ap_fixed<16, 5> a[64], ap_fixed<16, 5> b[64], ap_fixed<16, 5> out[1]) {
  out[0] = 0;
  for (int i = 0; i < 64; i++)
    out[0] += a[i] * b[i];
}
}
