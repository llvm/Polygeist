// Regression (HLSFactory flowgnn): integer options given by calls to
// constexpr functions, including function templates, e.g.
//   #pragma HLS LOOP_TRIPCOUNT min=ceildiv(ANALYSIS_MIN_NODES, NODE_PARALLEL)
// were rejected with "expected an integer constant".
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s

// CHECK-NOT: error
// CHECK-NOT: warning

#define ANALYSIS_MAX_NODES 100
#define NODE_PARALLEL 8

template <typename T> static constexpr T ceildiv(T dividend, T divisor) {
  return (dividend + divisor - 1) / divisor;
}
constexpr int twice(int x) { return 2 * x; }
namespace util {
constexpr unsigned half(unsigned x) { return x / 2; }
}

// Instantiated elsewhere (as in the original design), so a specialization
// exists for the pragma to call.
int dims[ceildiv(ANALYSIS_MAX_NODES, NODE_PARALLEL)];

// CHECK-LABEL: func.func @_Z6kernelPi(
// CHECK: } {hls.loop_tripcount = [{avg = 7 : i32, max = 13 : i32, min = 1 : i32}], hls.pipeline = [{II = 6 : i32, rewind = false, style = "stp"}]}
void kernel(int *a) {
  for (int i = 0; i < 64; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=ceildiv(1, NODE_PARALLEL) max=ceildiv(ANALYSIS_MAX_NODES, NODE_PARALLEL)
#pragma HLS PIPELINE II=twice(util::half(6))
    a[i] = dims[i % 13];
  }
}
