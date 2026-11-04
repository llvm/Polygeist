// An unroll factor written as an expression that folds below 1 (hp_fft_hls:
// `unroll factor=UF>>(STAGE-1)` with UF 1 and STAGE 2). Vitis synthesises
// those designs and ignores the pragma with a warning ([HLS 207-5577],
// kernel w9r6_uf0, CIRCT OPEN_ISSUES O70 (b)); so does cgeist, as for a
// literal factor below 1 (errors.c).
// RUN: cgeist %s --function='*' -S --hls-annotate -o - 2>&1 | FileCheck %s

template <int UF, int STAGE> void work(int a[64]) {
  for (int i = 0; i < 64; i++) {
// CHECK: factor-expr-zero.cpp:[[@LINE+1]]:27: warning: option 'factor' in '#pragma HLS unroll' is 0, below 1; the pragma is ignored, as Vitis ignores it
#pragma HLS unroll factor=UF>>(STAGE-1)
    a[i] += STAGE;
  }
}
// CHECK-LABEL: func.func @_Z5c_topPi
// CHECK-NOT: hls.unroll
void c_top(int a[64]) { work<1, 2>(a); }
