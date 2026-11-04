// Regression: two different unknown pragmas on the same loop were reported
// as conflicting because they shared an empty conflict key.
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s

// CHECK-NOT: error
// CHECK: } {hls.unknown_pragmas = ["loop_name L1", "foo"]}
void f(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS loop_name L1
#pragma HLS foo
    a[i] = i;
  }
}
