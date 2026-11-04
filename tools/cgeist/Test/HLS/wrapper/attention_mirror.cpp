// The attention-kernel pattern (CLAUDE.md step 2) through a fixed-point
// scalar wrapper with --scalar-wrapper-abi: every ap_fixed-like array is an
// i16 memref, the arithmetic is plain integer arith, and nothing is left as
// an LLVM struct or a pointer. With head_dim = 16, `(data_t)16` wraps to -16,
// sqrt of a negative value is 0, and the scale is `1.0 / 0`: (data_t)1.0 is
// 2048 raw, aligned to the 11 fraction bits of the divisor = 4194304, divided
// by zero. That division stays in the IR for hls-est's Vitis-undef fold.
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --scalar-wrapper-abi -I %S/Inputs 2>&1 | FileCheck %s
#include "mini_fixed.h"
typedef fx<16, 5> data_t;

// CHECK-NOT: llvm.struct
// CHECK-NOT: memref2pointer
// CHECK-NOT: llvm.getelementptr

// CHECK-LABEL: func.func @_Z6kernelPA16_2fxILi16ELi5EES2_S2_PA8_S0_(%arg0: memref<8x16xi16>, %arg1: memref<16x16xi16>, %arg2: memref<16x16xi16>, %arg3: memref<8x8xi16>)
// CHECK-DAG: memref.alloca() {hls.preserve, polygeist.varname = "Q"} : memref<8x16xi16>
// CHECK-DAG: memref.alloca() {hls.preserve, polygeist.varname = "K"} : memref<8x16xi16>
// CHECK-DAG: memref.alloca() {hls.preserve, polygeist.varname = "scores"} : memref<8x8xi16>
// CHECK-DAG: arith.divsi %c4194304_i128, %c0_i128 : i128
// CHECK: return
void kernel(data_t in[8][16], data_t Wq[16][16], data_t Wk[16][16], data_t out[8][8]) {
  const int head_dim = 16;
  const data_t scale = (data_t)1.0 / hls::sqrt((data_t)head_dim);
  data_t Q[8][16];
  data_t K[8][16];
  data_t scores[8][8];
  for (int i = 0; i < 8; i++)
    for (int d = 0; d < 16; d++) {
      Q[i][d] = 0;
      K[i][d] = 0;
      for (int k = 0; k < 16; k++) {
        Q[i][d] += in[i][k] * Wq[d][k];
        K[i][d] += in[i][k] * Wk[d][k];
      }
    }
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++) {
      scores[i][j] = 0;
      for (int d = 0; d < head_dim; d++)
        scores[i][j] += Q[i][d] * K[j][d];
      scores[i][j] *= scale;
    }
  for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
      out[i][j] = scores[i][j];
}
