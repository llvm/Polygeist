// #pragma HLS latency: hls.latency = [{min?, max?}] on the enclosing loop or
// function. Absent min/max are omitted (unconstrained).
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --strict-whitespace
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s --strict-whitespace --check-prefixes=CHECK,INLINE
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

// CHECK-LABEL: func.func @function_minmax(
// CHECK-SAME: hls.latency = [{max = 10 : i32, min = 2 : i32}]
void function_minmax(int *a) {
#pragma HLS latency min=2 max=10
  a[0] = a[1] * a[2];
}

// CHECK-LABEL: func.func @function_max(
// CHECK-SAME: hls.latency = [{max = 5 : i32}]
void function_max(int *a) {
#pragma HLS latency max=5
  a[0] = a[1] * a[2];
}

// CHECK-LABEL: func.func @function_min(
// CHECK-SAME: hls.latency = [{min = 1 : i32}]
void function_min(int *a) {
#pragma HLS LATENCY MIN=1
  a[0] = a[1] * a[2];
}

// CHECK-LABEL: func.func @empty(
// CHECK-SAME: hls.latency = [{}]
void empty(int *a) {
#pragma HLS latency
  a[0] = 1;
}

// CHECK-LABEL: func.func @loop_level(
// CHECK: {{^}}      } {hls.latency = [{max = 3 : i32, min = 0 : i32}]}
// CHECK-NEXT: {{^}}    }{{$}}
void loop_level(int a[8][8]) {
  for (int i = 0; i < 8; ++i)
    for (int j = 0; j < 8; ++j) {
#pragma HLS latency min=0 max=3
      a[i][j] = i * j;
    }
}

// CHECK-LABEL: func.func @do_while(
// CHECK: } attributes {hls.latency = [{max = 4 : i32}]}
void do_while(int *a, int n) {
  int i = 0;
  do {
#pragma HLS latency max=4
    a[i] = i;
    ++i;
  } while (i < n);
}

// CHECK-LABEL: func.func @top(
// INLINE: } {hls.latency = [{max = 3 : i32, min = 0 : i32}]}
void top(int a[8][8]) { loop_level(a); }

#ifdef ERRORS
void errors(int *a) {
// ERR: latency.c:[[@LINE+1]]:21: error: min=5 exceeds max=2 in '#pragma HLS latency'
#pragma HLS latency min=5 max=2
  a[0] = 1;
}
#endif
