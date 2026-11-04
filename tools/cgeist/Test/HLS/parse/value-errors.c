// Option values that are syntactically fine but do not evaluate to a valid
// constant are rejected when the pragma is bound to its function.
// RUN: not cgeist %s --function='*' -S --hls-annotate -o /dev/null 2>&1 | FileCheck %s

#define ZERO 0
void f(int *a, int n) {
  int buf[16][16];
// CHECK: value-errors.c:[[@LINE+1]]:56: error: cannot resolve 'UNDEFINED_FACTOR' to an integer constant in '#pragma HLS array_partition'
#pragma HLS array_partition variable=buf cyclic factor=UNDEFINED_FACTOR
// CHECK: value-errors.c:[[@LINE+1]]:56: error: cannot resolve 'n' to an integer constant in '#pragma HLS array_partition'
#pragma HLS array_partition variable=buf cyclic factor=n dim=2
// CHECK: value-errors.c:[[@LINE+1]]:56: error: option 'factor' in '#pragma HLS array_partition' must be >= 1, got 0
#pragma HLS array_partition variable=buf cyclic factor=ZERO
// CHECK: value-errors.c:[[@LINE+1]]:46: error: option 'dim' in '#pragma HLS array_partition' must be >= 0, got -1
#pragma HLS array_partition variable=buf dim=-1
// CHECK: value-errors.c:[[@LINE+1]]:56: error: invalid integer expression '4/0' for option 'factor' in '#pragma HLS array_partition'
#pragma HLS array_partition variable=buf cyclic factor=4/0
// CHECK: value-errors.c:[[@LINE+1]]:55: error: option 'factor' in '#pragma HLS array_partition' is out of range: 4294967296
#pragma HLS array_partition variable=buf block factor=1<<32
  for (int i = 0; i < 16; ++i) {
// CHECK: value-errors.c:[[@LINE+1]]:25: error: option 'II' in '#pragma HLS pipeline' must be >= 1, got 0
#pragma HLS pipeline II=0
    a[i] = buf[i][i];
  }
}

// CHECK-NOT: error
// CHECK-NOT: Stack dump
