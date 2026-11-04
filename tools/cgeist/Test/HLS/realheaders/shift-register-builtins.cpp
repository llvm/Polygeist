// ap_shift_reg's `__fpga_shift_register_peek(Array, Addr)` reads Array[Addr]
// and `__fpga_shift_register_shift(DataIn, Array, Enable)` moves every
// element one place up and stores DataIn at 0 when Enable; the depth is the
// extent of the array the header casts to a pointer (CIRCT OPEN_ISSUES O219:
// Vitis_Libraries dsp 1-D FFT, `undeclared identifier
// '__fpga_shift_register_peek'` before).
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

// CHECK: affine.for %{{.*}} = 0 to 64 {
// CHECK:   %[[IN:.*]] = affine.load %arg0
// CHECK:   %[[OUT3:.*]] = llvm.load
// CHECK:   affine.for %[[K:.*]] = 0 to 3 {
// CHECK:     llvm.load
// CHECK:     llvm.store
// CHECK:   }
// CHECK:   llvm.store %[[IN]]
#include <ap_shift_reg.h>
void top(const int in[64], int out[64]) {
  static ap_shift_reg<int, 4> sr;
  for (int i = 0; i < 64; i++) {
    int o = sr.shift(in[i], 3);
    out[i] = o + sr.read(1);
  }
}
