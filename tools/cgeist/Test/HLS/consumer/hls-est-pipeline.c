// hls.pipeline (II), hls.pipeline_off and polygeist.varname as consumed by
// hls-est's infer-vitis-split-array; a user hls.array_partition overrides
// inference.
// REQUIRES: hls-est
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine -o %t.mlir
// RUN: sed -i 's/^module attributes {.*} {$/module {/' %t.mlir
// RUN: %hls-est %t.mlir --allow-unregistered-dialect "--infer-vitis-split-array=td=auto emit-remarks=true" -o /dev/null 2>&1 | FileCheck %s
// RUN: %hls-est %t.mlir --allow-unregistered-dialect "--infer-vitis-split-array=td=auto emit-remarks=true auto-pipeline=true" -o /dev/null 2>&1 | FileCheck %s --check-prefix=AUTO

// Two reads of `buf` per iteration at the user's II=1: split by 2, reported
// under the C name carried by polygeist.varname.
// CHECK-DAG: remark: Inferring pragma 'array_partition type=cyclic factor=2 dim=1' for array 'user_ii' due to pipeline pragma

// II=2 leaves room for both reads: nothing to infer for `relaxed`.
// CHECK-NOT: for array 'relaxed'
// A user partition stands: nothing inferred for `given`.
// CHECK-NOT: for array 'given'
// pipeline off: never treated as pipelined, even with auto-pipeline.
// CHECK-NOT: for array 'unpiped'
// AUTO-NOT: for array 'unpiped'
// AUTO-NOT: for array 'relaxed'

void user_ii_kernel(float *x, float *y) {
  float user_ii[64];
  for (int i = 0; i < 64; ++i)
    user_ii[i] = x[i];
  for (int i = 0; i < 32; ++i) {
#pragma HLS pipeline II=1
    y[i] = user_ii[2 * i] + user_ii[2 * i + 1];
  }
}

void relaxed_kernel(float *x, float *y) {
  float relaxed[64];
  for (int i = 0; i < 64; ++i)
    relaxed[i] = x[i];
  for (int i = 0; i < 32; ++i) {
#pragma HLS pipeline II=2
    y[i] = relaxed[2 * i] + relaxed[2 * i + 1];
  }
}

void given_kernel(float *x, float *y) {
  float given[64];
#pragma HLS array_partition variable=given block factor=2
  for (int i = 0; i < 64; ++i)
    given[i] = x[i];
  for (int i = 0; i < 32; ++i) {
#pragma HLS pipeline II=1
    y[i] = given[2 * i] + given[2 * i + 1];
  }
}

void unpiped_kernel(float *x, float *y) {
  float unpiped[64];
  for (int i = 0; i < 64; ++i)
    unpiped[i] = x[i];
  for (int i = 0; i < 32; ++i) {
#pragma HLS pipeline off
    y[i] = unpiped[2 * i] + unpiped[2 * i + 1];
  }
}
