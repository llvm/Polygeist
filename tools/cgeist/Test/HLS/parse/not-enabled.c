// Without --hls-annotate, HLS pragmas are consumed silently (no attributes,
// no diagnostics), even malformed ones. Other unknown pragmas keep clang's
// behaviour (ignored silently by default).
// RUN: cgeist %s --function='*' -S 2>&1 | FileCheck %s

// CHECK-NOT: warning
// CHECK-NOT: error
// CHECK-NOT: hls.
void f(int *a) {
#pragma foo
#pragma HLS array_partition variable=nope
#pragma HLS pipeline II=bad
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline II=1
    a[i] = i;
  }
}
