// unroll is loop-only; conflicts and invalid factors are diagnosed. An unroll
// outside a loop body is ignored with a warning, as Vitis 2025.1 ignores it
// ([HLS 207-5589]).
// RUN: not cgeist %s --function='*' -S --hls-annotate -o /dev/null 2>&1 | FileCheck %s

void outside_loop(int *a) {
// CHECK: errors.c:[[@LINE+1]]:1: warning: '#pragma HLS unroll' can only be applied inside a loop body; it is ignored, as Vitis ignores it
#pragma HLS unroll
  a[0] = 1;
}

void before_loop(int *a) {
// CHECK: errors.c:[[@LINE+1]]:1: warning: '#pragma HLS unroll' can only be applied inside a loop body; it is ignored, as Vitis ignores it
#pragma HLS unroll factor=2
  for (int i = 0; i < 8; ++i)
    a[i] = i;
}

void conflict(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS unroll factor=2
// CHECK: errors.c:[[@LINE+1]]:1: error: '#pragma HLS unroll' conflicts with an earlier pragma on the same loop
#pragma HLS unroll off
    a[i] = i;
  }
}

// A literal factor below 1 is ignored with a warning, as Vitis 2025.1 does
// ([HLS 207-5577], kernel w9r6rev uflit0: factor=0 and factor=-1).
void zero(int *a) {
  for (int i = 0; i < 8; ++i) {
// CHECK: errors.c:[[@LINE+1]]:27: warning: option 'factor' in '#pragma HLS unroll' is 0, below 1; the pragma is ignored, as Vitis ignores it
#pragma HLS unroll factor=0
    a[i] = i;
  }
}

void minus_one(int *a) {
  for (int i = 0; i < 8; ++i) {
// CHECK: errors.c:[[@LINE+1]]:27: warning: option 'factor' in '#pragma HLS unroll' is -1, below 1; the pragma is ignored, as Vitis ignores it
#pragma HLS unroll factor=-1
    a[i] = i;
  }
}

// CHECK-NOT: error
