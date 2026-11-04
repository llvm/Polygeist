// An array of ap_int<12> from the real headers is a memref of i12, so hls-est
// costs it like Vitis: 1024 x 12 bits is one BRAM_18K (an ignored bitwidth
// attribute gave i32 and 2).
// REQUIRES: vitis-autopilot, xlx-bitwidth, hls-est
// RUN: cgeist %s %xlx_real %resourcedir --function=arr -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o %t.mlir
// RUN: sed -e "s/^module attributes {.*} {$/module {/" %t.mlir > %t.clean.mlir
// RUN: %hls-est %t.clean.mlir --allow-unregistered-dialect --affine-loop-normalize --memory-banking-bram --convert-affine-to-loopschedule --bram-analysis | FileCheck %s
// CHECK: 1024x12b
// CHECK: Total BRAM: 1
#include <ap_int.h>

extern "C" void arr(ap_int<12> in[1024], ap_int<12> out[1024]) {
  ap_int<12> buf[1024];
  for (int i = 0; i < 1024; i++)
    buf[i] = in[i] + 1;
  for (int i = 0; i < 1024; i++)
    out[i] = buf[1023 - i];
}
