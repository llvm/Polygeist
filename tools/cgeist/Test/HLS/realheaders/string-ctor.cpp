// ap_fixed / ap_int constructed from a C string. Under __SYNTHESIS__ the
// headers call _ssdm_string2bits (etc/ap_fixed_base.h:458-471,
// etc/ap_int_base.h:398-410), which the Vitis front end evaluates at compile
// time; with --xlx-headers cgeist folds it the same way (XlxStringBits.cc):
// a static table becomes a dense global with no run-time initialisation, a
// scalar a constant. Expected bits: the Vitis C simulation of the same
// declarations (g++ + Vitis/include), printed as signed decimals.
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s --check-prefix=CLEAN
#include <ap_fixed.h>
#include <ap_int.h>

// CLEAN-NOT: _ssdm_string2bits
// CLEAN-NOT: llvm.struct
// CLEAN-NOT: @init
// CLEAN-NOT: call

// ap_fixed<16,5>: 1.5, -2.5 (hex with a point), 5.25 (binary), pi truncated
// to 11 fraction bits (AP_TRN), 0x1.8p-2 = 0.375 (binary exponent), 1.25e1.
// CHECK-DAG: memref.global "private" @"tbl@static@{{.*}}" : memref<6xi16> = dense<[3072, -5120, 10752, 6433, 768, 25600]>
// ap_ufixed<8,4,AP_RND,AP_SAT>: 100 saturates to all ones, -1 to 0, 2.0.
// CHECK-DAG: memref.global "private" @"utbl@static@{{.*}}" : memref<3xi8> = dense<[-1, 0, 32]>
extern "C" {
// CHECK-LABEL: func.func @tbl
// CHECK: memref.get_global @"tbl@static@
int tbl(int i) {
  const static ap_fixed<16, 5> tbl[6] = {"1.5",   "-0x2.8p0", "0b101.01",
                                         "3.14159", "0x1.8p-2", "1.25e1"};
  return tbl[i % 6].range(15, 0).to_int();
}

int utbl(int i) {
  const static ap_ufixed<8, 4, AP_RND, AP_SAT> utbl[3] = {"100", "-1", "2.0"};
  return utbl[i % 3].range(7, 0).to_int();
}

// A scalar: -2.5615234375 * 2^9 = -1311.5 rounds to the even -1312
// (AP_RND_CONV). (range().to_int() of a signed ap_fixed sign-extends in the
// synthesis headers, etc/ap_fixed_ref.h:541-542.)
// CHECK-LABEL: func.func @scal
// CHECK: arith.constant -1312 : i32
int scal() {
  ap_fixed<12, 3, AP_RND_CONV, AP_WRAP> c = "-2.5615234375";
  return c.range(11, 0).to_int();
}

// An integer wider than 64 bits from a decimal string: bits [99:68] of
// 1234567890123456789012345678901 are 0xF951A8FA.
// CHECK-LABEL: func.func @big
// CHECK: arith.constant -112088582 : i32
int big() {
  ap_uint<100> x = "1234567890123456789012345678901";
  return x.range(99, 68).to_int();
}
}
