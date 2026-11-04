// #pragma HLS loop_tripcount: hls.loop_tripcount = [{min, max, avg}] on the
// loop. max is required; min defaults to 0 and avg to (min + max) / 2.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --strict-whitespace
// RUN: cgeist %s --function='*' -S --hls-annotate -O2 2>&1 | FileCheck %s --strict-whitespace --check-prefixes=CHECK,INLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s --strict-whitespace --check-prefixes=CHECK,INLINE
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

#define MAXN 1024

// CHECK-LABEL: func.func @max_only(
// CHECK: } {hls.loop_tripcount = [{avg = 512 : i32, max = 1024 : i32, min = 0 : i32}]}
void max_only(int *a, int n) {
  for (int i = 0; i < n; ++i) {
#pragma HLS loop_tripcount max=MAXN
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @min_max(
// CHECK: } {hls.loop_tripcount = [{avg = 15 : i32, max = 20 : i32, min = 10 : i32}]}
void min_max(int *a, int n) {
  for (int i = 0; i < n; ++i) {
#pragma HLS loop_tripcount min=10 max=20
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @all(
// CHECK: } {hls.loop_tripcount = [{avg = 12 : i32, max = 20 : i32, min = 10 : i32}]}
void all(int *a, int n) {
  for (int i = 0; i < n; ++i) {
#pragma HLS LOOP_TRIPCOUNT min = 10 max = 20 avg = 12
    a[i] = i;
  }
}

// Innermost of a nest with data-dependent bounds.
// CHECK-LABEL: func.func @nested(
// CHECK: {{^}}      } attributes {hls.loop_tripcount = [{avg = 8 : i32, max = 16 : i32, min = 1 : i32}]}
// CHECK-NEXT: {{^}}    } {hls.loop_tripcount = [{avg = 50 : i32, max = 100 : i32, min = 0 : i32}]}
void nested(int *a, int n, int *m) {
  for (int i = 0; i < n; ++i) {
#pragma HLS loop_tripcount max=100
    for (int j = 0; j < m[i]; ++j) {
#pragma HLS loop_tripcount min=1 max=16
      a[i * 16 + j] = j;
    }
  }
}

// CHECK-LABEL: func.func @while_loop(
// CHECK: attributes {hls.loop_tripcount = [{avg = 32 : i32, max = 64 : i32, min = 0 : i32}]}
void while_loop(int *a) {
  int i = 0;
  while (a[i] != 0) {
#pragma HLS loop_tripcount max=64
    i++;
  }
  a[0] = i;
}

// CHECK-LABEL: func.func @top(
// INLINE: } {hls.loop_tripcount = [{avg = 15 : i32, max = 20 : i32, min = 10 : i32}]}
void top(int *a, int n) { min_max(a, n); }

#ifdef ERRORS
void errors(int *a, int n) {
  for (int i = 0; i < n; ++i) {
// ERR: loop_tripcount.c:[[@LINE+1]]:1: warning: '#pragma HLS loop_tripcount' without 'max' is ignored, as Vitis ignores it
#pragma HLS loop_tripcount min=4
    a[i] = i;
  }
  for (int i = 0; i < n; ++i) {
// ERR: loop_tripcount.c:[[@LINE+1]]:28: error: min=8 exceeds max=4 in '#pragma HLS loop_tripcount'
#pragma HLS loop_tripcount min=8 max=4
    a[i] = i;
  }
  for (int i = 0; i < n; ++i) {
// ERR: loop_tripcount.c:[[@LINE+1]]:40: error: avg=9 is not within [min, max] = [1, 4] in '#pragma HLS loop_tripcount'
#pragma HLS loop_tripcount min=1 max=4 avg=9
    a[i] = i;
  }
// ERR: loop_tripcount.c:[[@LINE+1]]:1: error: '#pragma HLS loop_tripcount' must be placed inside a loop body
#pragma HLS loop_tripcount max=4
}
#endif
