// The size Vitis measures for its automatic inlining (VITIS_NOTES N18) with
// the real headers: the Xilinx bit builtins the headers' ap code calls
// (__builtin_bit_select, __builtin_bit_part_select, ...) are one intrinsic
// each in Vitis's front end and fold for constant operands, and every member
// of the ap classes is inlined before Vitis measures - also the one the
// headers do not mark AP_INLINE, the ap_fixed_base(double) constructor
// (no row in any design-size report). Measured with the builtins as calls
// through pointers and the constructor as a callee of its own, the
// function below measured 147 and was kept, where Vitis inlines it (GradCAM
// avg_pool_core, [HLS 214-178]; cam_core likewise, 102 before).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --hls-auto-inline-report -o /dev/null 2>&1 | FileCheck %s
#include <ap_fixed.h>

typedef ap_fixed<16, 3> fm_t;

// CHECK-NOT: callee '_ZN13ap_fixed_base
// CHECK: caller 'top' callee 'avg_core' size {{[0-9]+}} calls 1: inlined
void avg_core(fm_t in[16][7][7], fm_t out[16]) {
  const fm_t factor = (fm_t)(16.0 / 49);
  for (int c = 0; c < 16; c++) {
    fm_t sum = 0;
    for (int i = 0; i < 7; i++)
      for (int j = 0; j < 7; j++)
        sum += in[c][i][j];
    out[c] = sum * factor;
  }
}

void top(fm_t in[16][7][7], fm_t out[16]) { avg_core(in, out); }
