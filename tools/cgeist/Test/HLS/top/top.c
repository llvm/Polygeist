// #pragma HLS top: hls.top = [{name}] on the enclosing function.
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 --raise-scf-to-affine 2>&1 | FileCheck %s
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

// CHECK-LABEL: func.func @kernel_impl(
// CHECK-SAME: hls.top = [{name = "kernel"}]
void kernel_impl(int *a) {
#pragma HLS top name=kernel
  for (int i = 0; i < 8; ++i)
    a[i] = i;
}

// CHECK-LABEL: func.func @other(
// CHECK-SAME: hls.top = [{name = "Top_2"}]
void other(int *a) {
  a[0] = 1;
  _Pragma("HLS TOP NAME=Top_2")
}

// Without name= the enclosing function is the top (Vitis 2025.1 accepts it;
// Vitis_Libraries dsp ssr_fft tests main.cpp:29).
// CHECK-LABEL: func.func @noname(
// CHECK-SAME: hls.top = [{}]
void noname(int *a) {
#pragma HLS top
  a[0] = 2;
}

// An empty `#pragma HLS` is ignored without a message, as in Vitis 2025.1
// (Vitis_Libraries xf_max_suppression.hpp:203).
// CHECK-LABEL: func.func @empty_pragma(
void empty_pragma(int *a) {
  for (int i = 0; i < 4; ++i) {
#pragma HLS
    a[i] = i;
  }
}

#ifdef ERRORS
void errors(int *a) {
#pragma HLS top name=a
// ERR: top.c:[[@LINE+1]]:1: error: '#pragma HLS top' conflicts with an earlier pragma on the same function
#pragma HLS top name=b
  a[0] = 0;
}
#endif
