// A loop with a variable step (`m += sz`, sz = 1 << stage) inside another
// loop: --raise-scf-to-affine computes its trip count above the enclosing
// loops, where the step and bounds are valid symbols; built next to the loop
// it made the affine map composition assert ("operations must be in a
// block", hp_fft_hls no_StagePipeline; CIRCT OPEN_ISSUES O76).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s

// CHECK-LABEL: func.func @_Z2stPfS_i
// CHECK: affine.for %[[K:.*]] = 0 to
// CHECK: affine.for %[[M:.*]] = 0 to
// CHECK: affine.store
void st(float ld[256], float st_[256], int stage) {
  int sz = 1 << stage, step = sz >> 1;
  for (int k = 0; k < step; ++k) {
    for (int m = 0; m < 256; m += sz) {
      st_[k + m] = ld[k + m] + ld[k + m + step];
    }
  }
}
