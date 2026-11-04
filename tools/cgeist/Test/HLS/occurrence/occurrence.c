// #pragma HLS occurrence: hls.occurrence = [{cycle}] on the enclosing loop
// (typically placed in a conditional region of a pipelined loop) or function.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

// CHECK-LABEL: func.func @in_condition(
// CHECK: } {hls.occurrence = [{cycle = 4 : i32}], hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
void in_condition(int *a, int *b) {
  for (int i = 0; i < 64; ++i) {
#pragma HLS pipeline II=1
    a[i] = i;
    if (i % 4 == 0) {
#pragma HLS occurrence cycle=4
      b[i / 4] = a[i];
    }
  }
}

// CHECK-LABEL: func.func @function_level(
// CHECK-SAME: hls.occurrence = [{cycle = 2 : i32}]
void function_level(int *a, int c) {
  if (c) {
#pragma HLS OCCURRENCE cycle=2
    a[0] = 1;
  }
}

#ifdef ERRORS
void errors(int *a) {
  for (int i = 0; i < 8; ++i) {
// ERR: occurrence.c:[[@LINE+1]]:1: error: '#pragma HLS occurrence' requires option 'cycle'
#pragma HLS occurrence
    a[i] = i;
  }
  for (int i = 0; i < 8; ++i) {
// ERR: occurrence.c:[[@LINE+1]]:30: error: option 'cycle' in '#pragma HLS occurrence' must be >= 1, got 0
#pragma HLS occurrence cycle=0
    a[i] = i;
  }
}
#endif
