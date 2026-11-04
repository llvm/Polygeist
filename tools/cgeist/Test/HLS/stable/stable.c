// #pragma HLS stable: hls.stable = [{variable}] on the parameter (argument
// attribute), the global, or the local buffer it names.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

int coeffs[16];

// CHECK-DAG: memref.global @coeffs : memref<16xi32> = uninitialized {hls.stable = [{variable = "coeffs"}]}
// CHECK-DAG: func.func @dataflow_region(%arg0: memref<?xi32> {hls.stable = [{variable = "cfg"}]}, %arg1: i32 {hls.stable = [{variable = "n"}]}, %arg2: memref<?xi32>)
void dataflow_region(int *cfg, int n, int *out) {
#pragma HLS stable variable=cfg
#pragma HLS STABLE variable=n
#pragma HLS stable variable=coeffs
  for (int i = 0; i < 16; ++i)
    out[i] = cfg[i] * coeffs[i] + n;
}

// Local buffers of a dataflow region (HLSFactory designs do this).
// CHECK-LABEL: func.func @local_buffer(
// CHECK: hls.stable = [{variable = "buf"}]
void local_buffer(int *a) {
  int buf[4];
#pragma HLS stable variable=buf
  for (int i = 0; i < 4; ++i)
    buf[i] = a[i];
  for (int i = 0; i < 4; ++i)
    a[i] = buf[3 - i];
}

#ifdef ERRORS
void errors(int *a) {
// ERR: stable.c:[[@LINE+1]]:29: error: use of undeclared variable 'nope' in '#pragma HLS stable'
#pragma HLS stable variable=nope
  a[0] = 1;
}
#endif
