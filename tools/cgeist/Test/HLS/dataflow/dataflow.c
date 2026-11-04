// #pragma HLS dataflow: hls.dataflow = [{disable_start_propagation}] on the
// function (outside loops) or on the enclosing loop.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --strict-whitespace
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s --strict-whitespace
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

static void produce(int *a, int *b) {
  for (int i = 0; i < 16; ++i)
    b[i] = a[i] + 1;
}
static void consume(int *b, int *c) {
  for (int i = 0; i < 16; ++i)
    c[i] = 2 * b[i];
}

// CHECK-LABEL: func.func @function_dataflow(
// CHECK-SAME: hls.dataflow = [{disable_start_propagation = false}]
void function_dataflow(int *a, int *c) {
#pragma HLS dataflow
  int b[16];
  produce(a, b);
  consume(b, c);
}

// CHECK-LABEL: func.func @function_dataflow_nsp(
// CHECK-SAME: hls.dataflow = [{disable_start_propagation = true}]
void function_dataflow_nsp(int *a, int *c) {
  int b[16];
  produce(a, b);
#pragma HLS DATAFLOW disable_start_propagation
  consume(b, c);
}

// CHECK-LABEL: func.func @loop_dataflow(
// CHECK: {{^}}    } {hls.dataflow = [{disable_start_propagation = false}]}
void loop_dataflow(int a[4][16], int c[4][16]) {
  for (int k = 0; k < 4; ++k) {
#pragma HLS dataflow
    int b[16];
    for (int i = 0; i < 16; ++i)
      b[i] = a[k][i] + 1;
    for (int i = 0; i < 16; ++i)
      c[k][i] = 2 * b[i];
  }
}

#ifdef ERRORS
void errors(int *a, int *c) {
#pragma HLS dataflow
// ERR: dataflow.c:[[@LINE+1]]:1: warning: duplicate '#pragma HLS dataflow' on the same function is ignored
#pragma HLS dataflow
// ERR: dataflow.c:[[@LINE+1]]:1: error: '#pragma HLS dataflow' conflicts with an earlier pragma on the same function
#pragma HLS dataflow disable_start_propagation
  c[0] = a[0];
}
#endif
