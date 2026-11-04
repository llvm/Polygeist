// #pragma HLS loop_merge: hls.loop_merge = [{force}] on the enclosing loop,
// or on the function when placed outside loops (merges its child loops).
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --strict-whitespace
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s --strict-whitespace --check-prefixes=CHECK,INLINE
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

// CHECK-LABEL: func.func @function_level(
// CHECK-SAME: hls.loop_merge = [{force = false}]
void function_level(int *a, int *b) {
#pragma HLS loop_merge
  a[0] = 0;
  for (int i = 0; i < 8; ++i)
    a[i] = i;
  for (int i = 0; i < 8; ++i)
    b[i] = i;
}

// CHECK-LABEL: func.func @loop_level(
// CHECK: {{^}}    } {hls.loop_merge = [{force = true}]}
void loop_level(int a[8][8], int b[8][8]) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS loop_merge force
    for (int j = 0; j < 8; ++j)
      a[i][j] = j;
    for (int j = 0; j < 8; ++j)
      b[i][j] = j;
  }
}

// CHECK-LABEL: func.func @while_loop(
// CHECK: } {hls.loop_merge = [{force = false}]}
void while_loop(int *a, int n) {
  int i = 0;
  while (i < n) {
#pragma hls LOOP_MERGE force=false
    a[i] = i;
    ++i;
  }
}

// CHECK-LABEL: func.func @top(
// INLINE: } {hls.loop_merge = [{force = true}]}
void top(int a[8][8], int b[8][8]) { loop_level(a, b); }

#ifdef ERRORS
void errors(int *a) {
#pragma HLS loop_merge
// ERR: loop_merge.c:[[@LINE+1]]:1: error: '#pragma HLS loop_merge' conflicts with an earlier pragma on the same function
#pragma HLS loop_merge force
  a[0] = 0;
}
#endif
