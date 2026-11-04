// A loop counted by an ap_uint (rosetta 3d-rendering OUTPUT_FB_ROW / COL,
// Dehazing debayer): the counter's ++ and < are the header's inlined member
// functions, so after inlining the counter lives in a memref<i16> that the
// loops read and update. XlxLoopScalarPromote threads it through the loops as
// a loop-carried value (OPEN_ISSUES O101 in the hls-est tree): the outer loop
// becomes an affine.for and the inner one (step 4) an scf.while carrying the
// counter, which hls-est raises as a counted loop. With
// --hls-promote-loop-scalars=false both stay scf.while over the alloca.
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-promote-loop-scalars=false -o - | FileCheck %s --check-prefix=OFF
#include <ap_int.h>

void apc_top(ap_uint<8> fb[256][256], ap_uint<32> out[16384]) {
  for (ap_uint<16> i = 0; i < 256; i++)
    for (ap_uint<16> j = 0; j < 256; j = j + 4) {
      ap_uint<32> o = 0;
      o(7, 0) = fb[i][j];
      o(15, 8) = fb[i][j + 1];
      out[i * 64 + j / 4] = o;
    }
}

// CHECK-LABEL: func.func @_Z7apc_top
// CHECK-NOT: memref.alloca() : memref<i16>
// CHECK: affine.for %{{.*}} = 0 to 256 {
// CHECK: scf.while (%{{.*}} = %{{.*}}) : (i16) -> i16 {
// CHECK: scf.yield %{{.*}} : i16

// OFF-LABEL: func.func @_Z7apc_top
// OFF: memref.alloca() : memref<i16>
// OFF: scf.while : () -> () {
