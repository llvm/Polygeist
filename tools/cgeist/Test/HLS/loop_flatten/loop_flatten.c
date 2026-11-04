// #pragma HLS loop_flatten: hls.loop_flatten = [{}] on the loop, or
// hls.loop_flatten_off with `off`. Loop-only.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --strict-whitespace --check-prefixes=CHECK,NOINLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O2 2>&1 | FileCheck %s --strict-whitespace --check-prefixes=CHECK,INLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s --strict-whitespace --check-prefixes=CHECK,INLINE
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

// Placed in the innermost loop of a perfect nest (as Vitis recommends).
// CHECK-LABEL: func.func @perfect(
// CHECK: {{^}}      } {hls.loop_flatten = [{}]}
// CHECK-NEXT: {{^}}    }{{$}}
void perfect(int a[8][8]) {
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) {
#pragma HLS loop_flatten
      a[i][j] = i * j;
    }
}

// CHECK-LABEL: func.func @off(
// CHECK: {{^}}    } {hls.loop_flatten_off}
void off(int a[8][8]) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS LOOP_FLATTEN OFF
    for (int j = 0; j < 8; ++j)
      a[i][j] = i + j;
  }
}

// CHECK-LABEL: func.func @while_loop(
// CHECK: } {hls.loop_flatten = [{}]}
void while_loop(int *a, int n) {
  int i = 0;
  while (i < n) {
    _Pragma("hls loop_flatten")
    a[i] = i;
    i++;
  }
}

// CHECK-LABEL: func.func @do_while(
// CHECK: } attributes {hls.loop_flatten = [{}]}
void do_while(int *a, int n) {
  int i = 0;
  do {
#pragma HLS loop_flatten
    a[i] = i;
    i++;
  } while (i < n);
}

// CHECK-LABEL: func.func @top(
// NOINLINE: call @perfect
// INLINE: } {hls.loop_flatten = [{}]}
void top(int a[8][8]) { perfect(a); }

#ifdef ERRORS
void errors(int *a) {
// ERR: loop_flatten.c:[[@LINE+1]]:1: error: '#pragma HLS loop_flatten' must be placed inside a loop body
#pragma HLS loop_flatten
  for (int i = 0; i < 8; ++i) {
#pragma HLS loop_flatten
// ERR: loop_flatten.c:[[@LINE+1]]:1: error: '#pragma HLS loop_flatten' conflicts with an earlier pragma on the same loop
#pragma HLS loop_flatten off
    a[i] = i;
  }
}
#endif
