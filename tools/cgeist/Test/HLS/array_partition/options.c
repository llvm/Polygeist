// #pragma HLS array_partition: hls.array_partition = [{variable, kind, factor?,
// dim}] on the storage of the variable. kind defaults to "complete" and dim
// to 1; factor is omitted for complete partitioning. `off` gives
// hls.array_partition_off = [{variable, dim}].
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine 2>&1 | FileCheck %s

// CHECK-NOT: warning
// CHECK-NOT: error

#define USE(buf, n)                                                            \
  for (int i = 0; i < (n); ++i)                                                \
    buf[i] = a[i];                                                             \
  for (int i = 0; i < (n); ++i)                                                \
    a[(n) - 1 - i] = buf[i];

// CHECK-LABEL: func.func @defaults(
// CHECK: memref.alloca() {hls.array_partition = [{dim = 1 : i32, kind = "complete", variable = "buf"}], hls.preserve, polygeist.varname = "buf"}
void defaults(int *a) {
  int buf[16];
#pragma HLS array_partition variable=buf
  USE(buf, 16)
}

// CHECK-LABEL: func.func @complete(
// CHECK: {hls.array_partition = [{dim = 1 : i32, kind = "complete", variable = "buf"}]
void complete(int *a) {
  int buf[16];
#pragma HLS array_partition variable=buf type=complete
  USE(buf, 16)
}

// CHECK-LABEL: func.func @cyclic(
// CHECK: {hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "buf"}]
void cyclic(int *a) {
  int buf[16];
#pragma HLS array_partition variable=buf type=cyclic factor=4
  USE(buf, 16)
}

// CHECK-LABEL: func.func @block(
// CHECK: {hls.array_partition = [{dim = 1 : i32, factor = 2 : i32, kind = "block", variable = "buf"}]
void block(int *a) {
  int buf[16];
#pragma HLS array_partition variable=buf type=block factor=2
  USE(buf, 16)
}

// Positional type (older Vivado spelling).
// CHECK-LABEL: func.func @positional(
// CHECK: {hls.array_partition = [{dim = 1 : i32, factor = 8 : i32, kind = "cyclic", variable = "buf"}]
void positional(int *a) {
  int buf[16];
#pragma HLS ARRAY_PARTITION variable=buf cyclic factor=8
  USE(buf, 16)
}

// CHECK-LABEL: func.func @dim_explicit(
// CHECK: {hls.array_partition = [{dim = 2 : i32, kind = "complete", variable = "buf"}]
void dim_explicit(int *a) {
  int buf[4][4];
#pragma HLS array_partition variable=buf complete dim=2
  for (int i = 0; i < 16; ++i)
    buf[i / 4][i % 4] = a[i];
  for (int i = 0; i < 16; ++i)
    a[i] = buf[i % 4][i / 4];
}

// dim=0 partitions every dimension.
// CHECK-LABEL: func.func @dim_zero(
// CHECK: {hls.array_partition = [{dim = 0 : i32, factor = 2 : i32, kind = "cyclic", variable = "buf"}]
void dim_zero(int *a) {
  int buf[4][4];
#pragma HLS array_partition variable=buf cyclic factor=2 dim=0
  for (int i = 0; i < 16; ++i)
    buf[i / 4][i % 4] = a[i];
  for (int i = 0; i < 16; ++i)
    a[i] = buf[i % 4][i / 4];
}

// All options together, in any order.
// CHECK-LABEL: func.func @all(
// CHECK: {hls.array_partition = [{dim = 1 : i32, factor = 2 : i32, kind = "block", variable = "buf"}]
void all(int *a) {
  int buf[16];
#pragma HLS array_partition dim=1 factor=2 type=block variable=buf
  USE(buf, 16)
}

// CHECK-LABEL: func.func @off(
// CHECK: {hls.array_partition_off = [{dim = 1 : i32, variable = "buf"}], hls.preserve, polygeist.varname = "buf"}
void off(int *a) {
  int buf[16];
#pragma HLS array_partition variable=buf off
  USE(buf, 16)
}

// CHECK-LABEL: func.func @off_dim(
// CHECK: {hls.array_partition_off = [{dim = 2 : i32, variable = "buf"}]
void off_dim(int *a) {
  int buf[4][4];
#pragma HLS array_partition variable=buf dim=2 off=true
  for (int i = 0; i < 16; ++i)
    buf[i / 4][i % 4] = a[i];
  for (int i = 0; i < 16; ++i)
    a[i] = buf[i % 4][i / 4];
}
