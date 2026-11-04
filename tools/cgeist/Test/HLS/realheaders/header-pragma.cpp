// A #pragma HLS inside a header template reaches the MLIR: hls::sinpi on an
// ap_fixed<16,..> instantiates etc/hls_hotbm_apfixed.h's sin_cos_fixed_16,
// whose BIND_STORAGE pragmas name namespace-qualified tables
// (variable=first_order_fixed_16::sin_cos_K0, type=ROM_1P impl=LUTRAM).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - 2>&1 | FileCheck %s --check-prefixes=CHECK,AUTO
// RUN: cgeist %s %xlx_real %resourcedir --hls-auto-inline=false --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - 2>&1 | FileCheck %s --check-prefixes=CHECK,NOAUTO
// hls::sinpi itself is a core of the math library, kept as a call marked
// hls.math_core (XlxMathCore.h). Inside it, the automatic inlining that
// mimics Vitis (W10, on by default) keeps the large sinf_or_cosf instance as
// a call; without it everything else is inlined.
#include <ap_fixed.h>
#include <hls_math.h>

// CHECK-NOT: error
// CHECK-DAG: memref.global @_ZN13hotbm_apfixed20first_order_fixed_1610sin_cos_K0E : memref<256xi19> {hls.bind_storage = [{impl = "lutram", type = "rom_1p", variable = "first_order_fixed_16::sin_cos_K0"}]}
// CHECK-DAG: memref.global @_ZN13hotbm_apfixed20first_order_fixed_1610sin_cos_K1E : memref<256xi11> {hls.bind_storage = [{impl = "lutram", type = "rom_1p", variable = "first_order_fixed_16::sin_cos_K1"}]}
// CHECK-LABEL: func.func @hp(%arg0: memref<64xi16>, %arg1: memref<64xi16>)
// CHECK: call @_ZN3hls5sinpiILi16ELi4EEE{{.*}} {hls.math_core = {function = "sinpi", type = "ap_fixed<16,4>"}} : (i16) -> i16
// CHECK: return
// CHECK-LABEL: func.func @_ZN3hls5sinpiILi16ELi4EEE{{.*}}(%arg0: i16) -> i16 attributes {hls.math_core = {function = "sinpi", type = "ap_fixed<16,4>"}
// AUTO: call @_ZN13hotbm_apfixed12sinf_or_cosf{{.*}}hls.inline_auto_off
// NOAUTO-NOT: call
// CHECK: return
extern "C" void hp(ap_fixed<16, 4> in[64], ap_fixed<16, 2> out[64]) {
  for (int i = 0; i < 64; i++)
    out[i] = hls::sinpi(in[i]);
}
