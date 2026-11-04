// #pragma HLS cache: hls.cache = [{port, lines?, depth?}] as an argument
// attribute on the named (m_axi) parameter.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

// CHECK-LABEL: func.func @cached(
// CHECK-SAME: %arg0: memref<{{.*}}> {hls.cache = [{depth = 64 : i32, lines = 4 : i32, port = "in"}], hls.interface = [{bundle = "gmem", mode = "m_axi", port = "in"}]}
// CHECK-SAME: %arg1: memref<{{.*}}> {hls.cache = [{port = "out"}]}
void cached(int *in, int *out) {
#pragma HLS interface m_axi port=in bundle=gmem
#pragma HLS cache port=in lines=4 depth=64
#pragma HLS CACHE port=out
  for (int i = 0; i < 64; ++i)
    out[i] = in[i] + in[63 - i];
}

#ifdef ERRORS
void errors(int *p) {
  int local[4];
// ERR-DAG: cache.c:[[@LINE+1]]:24: error: 'local' in '#pragma HLS cache' is not a function parameter
#pragma HLS cache port=local lines=1
// ERR-DAG: cache.c:[[@LINE+1]]:32: error: option 'lines' in '#pragma HLS cache' must be >= 1, got 0
#pragma HLS cache port=p lines=0
  local[0] = 0;
  p[0] = local[0];
}
#endif
