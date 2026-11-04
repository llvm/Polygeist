// Regression: `inline recursive off` was accepted with only an "option
// ignored" warning because semantic checks were skipped for `off` pragmas.
// RUN: not cgeist %s --function='*' -S --hls-annotate -o /dev/null 2>&1 | FileCheck %s

// CHECK: inline-off-recursive-not-diagnosed.c:[[@LINE+3]]:1: error: 'recursive' conflicts with 'off' in '#pragma HLS inline'
// CHECK-NOT: warning
void f(int *a) {
#pragma HLS inline recursive off
  a[0] = 0;
}
