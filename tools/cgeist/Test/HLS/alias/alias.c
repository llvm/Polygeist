// #pragma HLS alias: hls.alias = [{ports, distance | offset}] on the function.
// ports must be at least two parameters; exactly one of distance/offset,
// with one offset per port.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

// CHECK-LABEL: func.func @distance(
// CHECK-SAME: hls.alias = [{distance = 1024 : i32, ports = ["a", "b", "c"]}]
void distance(int *a, int *b, int *c) {
#pragma HLS alias ports=a,b,c distance=1024
  for (int i = 0; i < 16; ++i)
    c[i] = a[i] + b[i];
}

// CHECK-LABEL: func.func @offsets(
// CHECK-SAME: hls.alias = [{offset = [0 : i32, 256 : i32], ports = ["x", "y"]}]
#define OFF 256
void offsets(int *x, int *y) {
#pragma HLS ALIAS ports=x, y offset=0, OFF
  for (int i = 0; i < 16; ++i)
    y[i] = x[i];
}

#ifdef ERRORS
void errors(int *a, int *b) {
  int local[4];
// ERR-DAG: alias.c:[[@LINE+1]]:19: error: '#pragma HLS alias' requires at least two ports
#pragma HLS alias ports=a distance=4
// ERR-DAG: alias.c:[[@LINE+1]]:1: error: '#pragma HLS alias' requires exactly one of 'distance' or 'offset'
#pragma HLS alias ports=a,b
// ERR-DAG: alias.c:[[@LINE+1]]:1: error: '#pragma HLS alias' requires exactly one of 'distance' or 'offset'
#pragma HLS alias ports=a,b distance=4 offset=0,4
// ERR-DAG: alias.c:[[@LINE+1]]:29: error: '#pragma HLS alias' needs one offset per port (2 ports, 3 offsets)
#pragma HLS alias ports=a,b offset=0,4,8
// ERR-DAG: alias.c:[[@LINE+1]]:19: error: 'local' in '#pragma HLS alias' is not a function parameter
#pragma HLS alias ports=a,local distance=4
  local[0] = 1;
  a[0] = b[0] + local[0];
}
#endif
