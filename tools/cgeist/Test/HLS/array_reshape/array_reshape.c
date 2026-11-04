// #pragma HLS array_reshape: same options and shape as array_partition, under
// hls.array_reshape, plus the `object` flag; `off` gives
// hls.array_reshape_off.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --check-prefixes=CHECK,NOINLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

int gtab[64];

// CHECK-DAG: memref.global @gtab : memref<64xi32> = uninitialized {hls.array_reshape = [{dim = 1 : i32, factor = 4 : i32, kind = "block", variable = "gtab"}]}

// CHECK-LABEL: func.func @local(
// CHECK: memref.alloca() {hls.array_reshape = [{dim = 1 : i32, kind = "complete", variable = "buf"}], hls.preserve, polygeist.varname = "buf"} : memref<8xi32>
void local(int *a) {
  int buf[8];
#pragma HLS array_reshape variable=buf
  for (int i = 0; i < 8; ++i)
    buf[i] = a[i];
  for (int i = 0; i < 8; ++i)
    a[i] = buf[7 - i];
}

// CHECK-LABEL: func.func @multi(
// CHECK: memref.alloca() {hls.array_partition = [{dim = 1 : i32, kind = "complete", variable = "m"}], hls.array_reshape = [{dim = 2 : i32, factor = 2 : i32, kind = "cyclic", object = true, variable = "m"}], hls.preserve, polygeist.varname = "m"} : memref<4x8xi32>
void multi(int *a) {
  int m[4][8];
#pragma HLS ARRAY_RESHAPE variable=m cyclic factor=2 dim=2 object
#pragma HLS array_partition variable=m complete dim=1
  for (int i = 0; i < 32; ++i)
    m[i / 8][i % 8] = a[i];
  for (int i = 0; i < 32; ++i)
    a[i] = m[i % 4][i / 4];
}

// CHECK-LABEL: func.func @params(
// CHECK-SAME: %arg0: memref<?x8xi32> {hls.array_reshape = [{dim = 0 : i32, kind = "complete", variable = "p"}]}
// CHECK-SAME: %arg1: memref<?xi32> {hls.array_reshape_off = [{dim = 1 : i32, variable = "q"}]}
void params(int p[4][8], int *q) {
#pragma HLS array_reshape variable=p complete dim=0
#pragma HLS array_reshape variable=q off
#pragma HLS array_reshape variable=gtab block factor=4
  for (int i = 0; i < 32; ++i)
    q[i] = p[i / 8][i % 8] + gtab[i];
}

// CHECK-LABEL: func.func @top(
// INLINE: memref.alloca() {hls.array_reshape = [{dim = 1 : i32, kind = "complete", variable = "buf"}], hls.preserve, polygeist.varname = "buf"} : memref<8xi32>
// NOINLINE: call @local
void top(int *a) { local(a); }

#ifdef ERRORS
void errors(int *a) {
  int buf[8];
  int x;
// ERR-DAG: array_reshape.c:[[@LINE+1]]:1: error: '#pragma HLS array_reshape' with type=cyclic requires option 'factor'
#pragma HLS array_reshape variable=buf cyclic
// ERR-DAG: array_reshape.c:[[@LINE+1]]:27: error: '#pragma HLS array_reshape' requires an array or pointer, but 'x' has type 'int'
#pragma HLS array_reshape variable=x complete
// ERR-DAG: array_reshape.c:[[@LINE+1]]:40: error: dim=2 is out of range in '#pragma HLS array_reshape': 'buf' has 1 dimension(s)
#pragma HLS array_reshape variable=buf dim=2
  x = 0;
  buf[0] = x;
  a[0] = buf[0];
}
#endif
