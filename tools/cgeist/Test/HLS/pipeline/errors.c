// Conflicting and duplicate pipeline pragmas on one target, and `off` with
// other options.
// RUN: not cgeist %s --function='*' -S --hls-annotate -o /dev/null 2>&1 | FileCheck %s

void conflict_off(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=1
// CHECK: errors.c:[[@LINE+2]]:1: error: '#pragma HLS pipeline' conflicts with an earlier pragma on the same loop
// CHECK: errors.c:[[@LINE-2]]:1: note: previous pragma is here
#pragma HLS pipeline off
    a[i] = i;
  }
}

void conflict_ii(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=1
// CHECK: errors.c:[[@LINE+1]]:1: error: '#pragma HLS pipeline' conflicts with an earlier pragma on the same loop
#pragma HLS pipeline II=2
    a[i] = i;
  }
}

void conflict_function(int *a) {
#pragma HLS pipeline
// CHECK: errors.c:[[@LINE+1]]:1: error: '#pragma HLS pipeline' conflicts with an earlier pragma on the same function
#pragma HLS pipeline off
  a[0] = 0;
}

void duplicate(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=1
// CHECK: errors.c:[[@LINE+1]]:1: warning: duplicate '#pragma HLS pipeline' on the same loop is ignored
#pragma HLS PIPELINE ii=1
    a[i] = i;
  }
}

void off_with_options(int *a) {
  for (int i = 0; i < 8; ++i) {
// CHECK: errors.c:[[@LINE+1]]:22: warning: option 'II' is ignored because '#pragma HLS pipeline' has 'off'
#pragma HLS pipeline II=2 off
    a[i] = i;
  }
}

// Different loops never conflict.
void no_conflict(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=1
    for (int j = 0; j < 8; ++j) {
#pragma HLS pipeline off
      a[i * 8 + j] = i;
    }
  }
}

// Outside any loop the pragma applies to the function; if a loop follows,
// that is probably not what was meant.
void before_loop(int *a) {
// CHECK: errors.c:[[@LINE+1]]:1: warning: '#pragma HLS pipeline' is outside the loop that follows it and applies to the enclosing function; place it inside the loop body to apply it to the loop
#pragma HLS pipeline II=1
  for (int i = 0; i < 8; ++i)
    a[i] = i;
}

// CHECK-NOT: error
// CHECK-NOT: warning
