// `d[i] = std::complex<float>(a[i], b[i])`: the implicit copy assignment of
// std::complex<float> copies its `_Complex float` member (`_M_value =
// __z._M_value`, both rows of memref<?x2xf32>); reading the row as one value
// asserted ("must have shape 1"), which failed the 16 hp_fft_hls designs
// (CIRCT OPEN_ISSUES O70 (a)). The two parts are now copied one by one, like
// a struct of one member type. The values are checked bit for bit against
// g++ by the harness's check_semantics.py (the 12 hp_fft_hls UF designs).
// REQUIRES: vitis-autopilot, xlx-bitwidth
// RUN: cgeist %s %xlx_real %resourcedir --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank -o - | FileCheck %s
#include <complex>

// CHECK-LABEL: func.func @_Z5c_topPSt7complexIfEPKfS3_
// CHECK: affine.for %[[I:.*]] = 0 to 4 {
// CHECK-DAG: %[[A:.*]] = affine.load %arg1[%[[I]]] : memref<4xf32>
// CHECK-DAG: %[[B:.*]] = affine.load %arg2[%[[I]]] : memref<4xf32>
// CHECK: llvm.store %[[A]]
// CHECK: llvm.store %[[B]]
void c_top(std::complex<float> d[4], const float a[4], const float b[4]) {
  for (int i = 0; i < 4; i++)
    d[i] = std::complex<float>(a[i], b[i]);
}

// std::complex<float> arithmetic assigned into an array element: libstdc++
// builds `operator*` / `operator+` from `*=` / `+=` on its `_ComplexT`
// member (a typedef of `_Complex float`), which asserted "must have shape 1"
// before (CIRCT W9 round 6 review defect 2, repros cg/m4.cpp, cg/mx.cpp).
// Vitis's FPGA product: (ac - bd) + i(ad + bc), no libcall.
// CHECK-LABEL: func.func @_Z5m_topPSt7complexIfE
// CHECK-NOT: call
// CHECK: arith.mulf
// CHECK: arith.subf
// CHECK: arith.addf
// CHECK: return
void m_top(std::complex<float> a[8]) {
  for (int i = 0; i < 8; i++)
    a[i] = a[i] * a[(i + 1) & 7] + a[(i + 2) & 7];
}
