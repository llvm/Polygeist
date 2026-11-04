// #pragma HLS expression_balance: hls.expression_balance = [{}] on the
// enclosing loop or function; `off` gives hls.expression_balance_off.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

// CHECK-LABEL: func.func @function_on(
// CHECK-SAME: hls.expression_balance = [{}]
int function_on(int *a) {
#pragma HLS expression_balance
  return a[0] + a[1] + a[2] + a[3];
}

// CHECK-LABEL: func.func @function_off(
// CHECK-SAME: hls.expression_balance_off
float function_off(float *a) {
#pragma HLS EXPRESSION_BALANCE off
  return a[0] + a[1] + a[2] + a[3];
}

// CHECK-LABEL: func.func @loop_off(
// CHECK: } {hls.expression_balance_off}
void loop_off(float *a, float *b) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS expression_balance off
    b[i] = a[i] + a[i + 1] + a[i + 2];
  }
}

// CHECK-LABEL: func.func @top(
// INLINE: } {hls.expression_balance_off}
void top(float *a, float *b) { loop_off(a, b); }

#ifdef ERRORS
int errors(int *a) {
#pragma HLS expression_balance
// ERR: expression_balance.c:[[@LINE+1]]:1: error: '#pragma HLS expression_balance' conflicts with an earlier pragma on the same function
#pragma HLS expression_balance off
  return a[0];
}
#endif
