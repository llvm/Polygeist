// Control for attention_mirror.cpp: head_dim = 4 fits ap_fixed<16,5>, so the
// scale is 1.0 / sqrt(4) = 0.5, raw 4194304 / 4096 = 1024 in the <28,17>
// quotient, folded by cgeist, and the Q / K arrays stay live (they feed the
// scores product that is multiplied by it). Same lowering: i16 memrefs, no
// structs, no pointers.
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --scalar-wrapper-abi -I %S/Inputs 2>&1 | FileCheck %s
#include "mini_fixed.h"
typedef fx<16, 5> data_t;

// CHECK-NOT: llvm.struct
// CHECK-NOT: memref2pointer
// CHECK-NOT: llvm.getelementptr

// CHECK-LABEL: func.func @_Z11kernel_ctrlPA16_2fxILi16ELi5EES2_S2_PA8_S0_(%arg0: memref<8x16xi16>, %arg1: memref<16x16xi16>, %arg2: memref<16x16xi16>, %arg3: memref<8x8xi16>)
// CHECK-DAG: memref.alloca() {hls.preserve, polygeist.varname = "Q"} : memref<8x16xi16>
// CHECK-DAG: memref.alloca() {hls.preserve, polygeist.varname = "K"} : memref<8x16xi16>
// CHECK-DAG: memref.alloca() {hls.preserve, polygeist.varname = "scores"} : memref<8x8xi16>
// CHECK-NOT: arith.divsi
// CHECK: affine.load %arg0
// CHECK: arith.muli %{{.*}}, %c1024_i128 : i128
// CHECK: return
void kernel_ctrl(data_t in[8][16], data_t Wq[16][16], data_t Wk[16][16], data_t out[8][8]) {
  const int head_dim = 4;
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
