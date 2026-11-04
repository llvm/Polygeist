// #pragma HLS function_instantiate: argument attribute
// hls.function_instantiate = [{variable}] on the named parameter.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 --raise-scf-to-affine 2>&1 | FileCheck %s
// RUN: not cgeist %s --function='*' -S --hls-annotate -O0 -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

// CHECK-LABEL: func.func @scale(
// CHECK-SAME: %arg0: memref<?xi32>, %arg1: i32 {hls.function_instantiate = [{variable = "mode"}]}, %arg2: i32 {hls.function_instantiate = [{variable = "shift"}]})
void scale(int *a, int mode, int shift) {
#pragma HLS function_instantiate variable=mode
#pragma HLS FUNCTION_INSTANTIATE variable=shift
  for (int i = 0; i < 8; ++i)
    a[i] = mode ? a[i] << shift : a[i];
}

#ifdef ERRORS
int g;
void errors(int *a, int m) {
  int local = 0;
// ERR: function_instantiate.c:[[@LINE+1]]:43: error: 'local' in '#pragma HLS function_instantiate' is not a function parameter
#pragma HLS function_instantiate variable=local
// ERR: function_instantiate.c:[[@LINE+1]]:43: error: 'g' in '#pragma HLS function_instantiate' is not a function parameter
#pragma HLS function_instantiate variable=g
// ERR: function_instantiate.c:[[@LINE+1]]:43: error: use of undeclared variable 'q' in '#pragma HLS function_instantiate'
#pragma HLS function_instantiate variable=q
#pragma HLS function_instantiate variable=m
// ERR: function_instantiate.c:[[@LINE+1]]:1: warning: duplicate '#pragma HLS function_instantiate' on the same variable 'm' is ignored
#pragma HLS function_instantiate variable=m
  a[0] = local + m;
}
#endif
