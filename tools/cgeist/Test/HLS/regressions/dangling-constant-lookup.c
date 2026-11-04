// Regression: resolving an identifier in an integer option (e.g. `II=fast`)
// called through a function_ref to a destroyed lambda and crashed cgeist.
// RUN: not cgeist %s --function='*' -S --hls-annotate -o /dev/null 2>&1 | FileCheck %s

// CHECK: dangling-constant-lookup.c:[[@LINE+5]]:25: error: cannot resolve 'fast' to an integer constant in '#pragma HLS pipeline'
// CHECK-NOT: Stack dump
#define K 3
void f(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=fast
#pragma HLS unroll factor=K
    a[i] = i;
  }
}
