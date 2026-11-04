// A compound assignment (`/=`, `*=`, `+=`, `-=`, `++`, `--`) to an ap_fixed object is
// marked on its stores (`hls.ap_compound_assign`): Vitis keeps two stores per
// bank for it (`if (!op) V = 0` at the head of the cross-format
// ap_fixed_base::operator=, then the result) when the array is partitioned
// and the bank is chosen at run time, so the array is never RAM_1WNR (CIRCT
// VITIS_NOTES N38 R-cmpd, kernels w9r_d1div / d2mul / d3add); cgeist's own
// store forwarding leaves one store, the mark tells hls-est. A plain
// assignment (w9r_d4asg) and an int `/=` (w9r_d6intdiv) are not marked.
// `++` / `--` go through operator+= / -= and the same operator= (kernel
// w9r7rev_rc_inc: Vitis 14, two stores per bank; W9r7 review defect 5).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <ap_fixed.h>
typedef ap_fixed<16, 5> T;

// CHECK-LABEL: func.func @_Z3cmpPA64_8ap_fixed
// CHECK: affine.for
// CHECK: affine.store {{.*}} {hls.ap_compound_assign} : memref<2x64xi16>
// CHECK: affine.for
// CHECK-NOT: hls.ap_compound_assign
// CHECK: return
void cmp(T A[2][64], T c, T w, int I[2][64], int d) {
  for (int b = 0; b < 2; b++)
    for (int i = 0; i < 64; i++)
      A[b][i] /= c;
  for (int b = 0; b < 2; b++)
    for (int i = 0; i < 64; i++) {
      A[b][i] = w * A[b][i];
      I[b][i] /= d;
    }
}

// CHECK-LABEL: func.func @_Z3incPA64_8ap_fixed
// CHECK: affine.store {{.*}} {hls.ap_compound_assign} : memref<2x64xi16>
// CHECK: affine.store {{.*}} {hls.ap_compound_assign} : memref<2x64xi16>
// CHECK: return
void inc(T A[2][64]) {
  for (int b = 0; b < 2; b++)
    for (int i = 0; i < 64; i++)
      A[b][i]++;
  for (int b = 0; b < 2; b++)
    for (int i = 0; i < 64; i++)
      --A[b][i];
}
