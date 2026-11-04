// hls::exp on ap_fixed through the real headers (etc/hls_exp_apfixed.h): its
// lookup tables are `const static ap_ufixed<...> t[N] = {"0x1.0000p0", ...}`
// (string constructors, folded at compile time as by the Vitis front end,
// XlxStringBits.cc) and its body reads ap_range_ref / ap_bit_ref proxies
// inside the if / loop regions of the inlined helpers (forwarded across
// regions, XlxProxyScalarize.cc) and converts doubles through the
// doubleToRawBits union. Before these, a 256-entry table took cgeist minutes
// and hls::exp did not finish within 15 minutes. The result has the tables
// as dense globals and nothing hls-est cannot read.
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o %t.mlir
// RUN: FileCheck %s < %t.mlir
// RUN: FileCheck %s --check-prefix=CLEAN < %t.mlir
#include <ap_fixed.h>
#include <hls_math.h>

// CLEAN-NOT: _ssdm_string2bits
// CLEAN-NOT: llvm.struct
// CLEAN-NOT: memref2pointer
// CHECK-DAG: memref.global {{.*}}exp_x_msb_1_table{{.*}} : memref<32xi25> = dense<
// CHECK-DAG: memref.global {{.*}}f_x_msb_4_h_table{{.*}} : memref<8xi6> = dense<
// CHECK-LABEL: func.func @e12
// CHECK: memref.get_global
extern "C" int e12(double a) {
  ap_fixed<12, 4, AP_TRN, AP_WRAP> x = a;
  auto r = hls::exp(x);
  return r.range().to_int();
}
