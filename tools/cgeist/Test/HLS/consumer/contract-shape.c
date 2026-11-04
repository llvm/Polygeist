// The exact attribute shapes hls-est (~/circt, lib/Dialect/Resource) reads.
// This test runs without hls-est; the hls-est-*.c tests exercise the same
// attributes through the tool itself.
//
//  - hls.array_partition: ArrayAttr of {kind: "cyclic"|"block"|"complete",
//    factor: i32 (cyclic/block only), dim: i32 (always present, 1-based,
//    0 = all), variable: string}, on memref.alloca / memref.global or as a
//    func.func argument attribute (MemoryInterface.cpp parsePartitionSpec,
//    MemoryDS.cpp getPartitionSpecs, InferVitisSplitArray.cpp).
//  - hls.pipeline: ArrayAttr of {II: i32, ...} on affine.for; II > 0 is a
//    user target, 0 means "not specified" (InferVitisSplitArray.cpp).
//  - hls.pipeline_off: unit attribute on affine.for.
//  - polygeist.varname: string on the defining op of a memref.
//  - hls.preserve: unit attribute on multi-element local arrays (read by the
//    Polygeist canonicalizer).
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine 2>&1 | FileCheck %s --strict-whitespace

// CHECK-NOT: warning

float G[64];

// CHECK: memref.global @G : memref<64xf32> = uninitialized {hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "block", variable = "G"}]}

// CHECK-LABEL: func.func @shapes(
// CHECK-SAME: %arg0: memref<?x16xf32> {hls.array_partition = [{dim = 2 : i32, kind = "complete", variable = "a"}]}
// CHECK: %[[BUF:.*]] = memref.alloca() {hls.array_partition = [{dim = 1 : i32, factor = 2 : i32, kind = "cyclic", variable = "buf"}], hls.preserve, polygeist.varname = "buf"} : memref<64xf32>
// CHECK: affine.for
// CHECK: {{^}}    } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
// CHECK: affine.for
// CHECK: {{^}}    } {hls.pipeline = [{II = 0 : i32, rewind = false, style = "stp"}]}
// CHECK: affine.for
// CHECK: {{^}}    } {hls.pipeline_off}
void shapes(float a[8][16], float *x, float *y) {
  float buf[64];
#pragma HLS array_partition variable=a complete dim=2
#pragma HLS array_partition variable=buf cyclic factor=2
#pragma HLS array_partition variable=G block factor=4
  for (int i = 0; i < 64; ++i) {
#pragma HLS pipeline II=1
    buf[i] = x[i] + G[i];
  }
  for (int i = 0; i < 32; ++i) {
#pragma HLS pipeline
    y[i] = buf[2 * i] + buf[2 * i + 1];
  }
  for (int i = 0; i < 8; ++i) {
#pragma HLS pipeline off
    y[i] += a[i][i];
  }
}
