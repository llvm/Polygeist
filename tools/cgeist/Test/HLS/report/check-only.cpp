// --hls-pragma-check-only: every function body with a definition (template
// instantiations included, no --function filter) is checked and reported at
// the AST level; no code is generated, so designs cgeist cannot lower still
// get their pragmas validated.
// RUN: cgeist %s --function=nothing -S --hls-annotate --hls-pragma-check-only -o %t.mlir 2>&1 | FileCheck %s
// RUN: FileCheck %s --check-prefix=EMPTY < %t.mlir
// RUN: not cgeist %s --function=nothing -S --hls-annotate --hls-pragma-check-only -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// EMPTY-NOT: func.func

int g[8];

static void kernel(int *a) {
  int buf[8];
// CHECK-DAG: check-only.cpp:[[@LINE+1]]:1: remark: '#pragma HLS array_partition variable=buf complete' -> variable 'buf' in kernel (check-only)
#pragma HLS array_partition variable=buf complete
// CHECK-DAG: check-only.cpp:[[@LINE+1]]:1: remark: '#pragma HLS interface m_axi port=a' -> argument #0 (variable 'a') of @kernel
#pragma HLS interface m_axi port=a
// CHECK-DAG: check-only.cpp:[[@LINE+1]]:1: remark: '#pragma HLS reset variable=g' -> global 'g' (check-only)
#pragma HLS reset variable=g
  int m = a[0] * a[1];
// CHECK-DAG: check-only.cpp:[[@LINE+1]]:1: remark: '#pragma HLS bind_op variable=m op=mul' -> operation assigned to variable 'm' in kernel (check-only)
#pragma HLS bind_op variable=m op=mul
L:
  for (int i = 0; i < 8; ++i) {
// CHECK-DAG: check-only.cpp:[[@LINE+1]]:1: remark: '#pragma HLS pipeline II=1' -> loop 'L' in kernel (check-only)
#pragma HLS pipeline II=1
    buf[i] = a[i] + m + g[i];
  }
  a[0] = buf[3];
}

// Each instantiation separately; the uninstantiated pattern is not.
// CHECK-DAG: remark: '#pragma HLS unroll factor=F' -> loop in twice<2> (check-only)
// CHECK-DAG: remark: '#pragma HLS unroll factor=F' -> loop in twice<4> (check-only)
template <int F> void twice(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS unroll factor=F
    a[i] *= 2;
  }
}
void use(int *a) {
  twice<2>(a);
  twice<4>(a);
}

#ifdef ERRORS
void errors(int *a) {
// ERR: check-only.cpp:[[@LINE+1]]:1: error: '#pragma HLS loop_tripcount' must be placed inside a loop body
#pragma HLS loop_tripcount max=4
  a[0] = 0;
}
#endif
