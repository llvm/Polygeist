// ap_fixed conversions in every quantisation and overflow mode compile from
// the real headers to integer arithmetic on the raw iN (the values are
// checked against the Vitis C simulation by the hls-est harness's
// check_shim_cgeist.py --shim real).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <ap_fixed.h>
#define CONV(Q, O)                                                             \
  extern "C" void conv_##Q##_##O(ap_fixed<24, 10> a,                           \
                                 ap_fixed<8, 3, Q, O> &r) {                    \
    r = a;                                                                     \
  }
// CHECK-LABEL: func.func @conv_AP_RND_AP_SAT(%arg0: i24, %arg1: memref<?xi8>)
// CHECK-NOT: call
// CHECK: return
CONV(AP_RND, AP_SAT)
// CHECK-LABEL: func.func @conv_AP_RND_ZERO_AP_SAT_ZERO(%arg0: i24, %arg1: memref<?xi8>)
// CHECK-NOT: call
// CHECK: return
CONV(AP_RND_ZERO, AP_SAT_ZERO)
// CHECK-LABEL: func.func @conv_AP_RND_MIN_INF_AP_SAT_SYM(%arg0: i24, %arg1: memref<?xi8>)
// CHECK-NOT: call
// CHECK: return
CONV(AP_RND_MIN_INF, AP_SAT_SYM)
// CHECK-LABEL: func.func @conv_AP_RND_INF_AP_WRAP(%arg0: i24, %arg1: memref<?xi8>)
// CHECK-NOT: call
// CHECK: return
CONV(AP_RND_INF, AP_WRAP)
// CHECK-LABEL: func.func @conv_AP_RND_CONV_AP_WRAP_SM(%arg0: i24, %arg1: memref<?xi8>)
// CHECK-NOT: call
// CHECK: return
CONV(AP_RND_CONV, AP_WRAP_SM)
// CHECK-LABEL: func.func @conv_AP_TRN_AP_SAT(%arg0: i24, %arg1: memref<?xi8>)
// CHECK-NOT: call
// CHECK: return
CONV(AP_TRN, AP_SAT)
// CHECK-LABEL: func.func @conv_AP_TRN_ZERO_AP_WRAP(%arg0: i24, %arg1: memref<?xi8>)
// CHECK-NOT: call
// CHECK: return
CONV(AP_TRN_ZERO, AP_WRAP)
