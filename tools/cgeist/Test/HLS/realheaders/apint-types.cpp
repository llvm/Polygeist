// ap_int / ap_uint / ap_fixed from the real Vitis HLS headers: every value is
// the iN of its declared width (ssdm_int's bitwidth(N), -fxlx-bitwidth), the
// operators' result widths are those of the headers' RType, and the classes
// are passed as their scalar (--scalar-wrapper-abi).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <ap_fixed.h>
#include <ap_int.h>

// CHECK-LABEL: func.func @{{.*}}add17{{.*}}(%arg0: i17, %arg1: i17) -> i18
// The 17-bit operands are promoted to int (-fxlx-apint-promote, as Vitis).
// CHECK: arith.addi {{.*}} : i32
// CHECK: arith.trunci {{.*}} : i32 to i18
ap_int<18> add17(ap_int<17> a, ap_int<17> b) { return a + b; }
// CHECK-LABEL: func.func @{{.*}}mul17{{.*}}(%arg0: i17, %arg1: i17) -> i34
// CHECK: arith.muli {{.*}} : i34
ap_int<34> mul17(ap_int<17> a, ap_int<17> b) { return a * b; }
// CHECK-LABEL: func.func @{{.*}}sub33{{.*}}(%arg0: i33, %arg1: i33) -> i33
ap_uint<33> sub33(ap_uint<33> a, ap_uint<33> b) { return a - b; }
// CHECK-LABEL: func.func @{{.*}}shl7{{.*}}(%arg0: i7, %arg1: i32) -> i7
ap_int<7> shl7(ap_int<7> a, int s) { return a << s; }
// CHECK-LABEL: func.func @{{.*}}shr65{{.*}}(%arg0: i65) -> i65
// CHECK: arith.shrui {{.*}} : i65
ap_uint<65> shr65(ap_uint<65> a) { return a >> 3; }
// CHECK-LABEL: func.func @{{.*}}lt12{{.*}}(%arg0: i12, %arg1: i12) -> i8
// CHECK: arith.cmpi slt
bool lt12(ap_int<12> a, ap_uint<12> b) { return a < b; }
// CHECK-LABEL: func.func @{{.*}}or256{{.*}}(%arg0: i256, %arg1: i256) -> i256
// CHECK: arith.ori {{.*}} : i256
ap_uint<256> or256(ap_uint<256> a, ap_uint<256> b) { return a | b; }
// ap_int<1> is a signed 1-bit integer.
// CHECK-LABEL: func.func @{{.*}}neg1{{.*}}(%arg0: i1) -> i1
ap_int<1> neg1(ap_int<1> a) { return -a; }
// CHECK-LABEL: func.func @{{.*}}div16{{.*}}(%arg0: i16, %arg1: i9) -> i16
// CHECK: arith.divsi {{.*}} : i32
ap_int<16> div16(ap_int<16> a, ap_int<9> b) { return a / b; }
// CHECK-LABEL: func.func @{{.*}}fadd{{.*}}(%arg0: i16, %arg1: i12) -> i16
ap_fixed<16, 5> fadd(ap_fixed<16, 5> a, ap_fixed<12, 4> b) { return a + b; }
// CHECK-LABEL: func.func @{{.*}}fconv{{.*}}(%arg0: i24) -> i8
ap_fixed<8, 3, AP_RND, AP_SAT> fconv(ap_fixed<24, 10> a) { return a; }
// CHECK-LABEL: func.func @{{.*}}arr{{.*}}(%arg0: memref<1024xi12>, %arg1: memref<1024xi12>)
// CHECK: memref.alloca() {{.*}}: memref<1024xi12>
void arr(ap_int<12> in[1024], ap_int<12> out[1024]) {
  ap_int<12> buf[1024];
  for (int i = 0; i < 1024; i++)
    buf[i] = in[i] + 1;
  for (int i = 0; i < 1024; i++)
    out[i] = buf[1023 - i];
}
