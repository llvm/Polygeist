// #pragma HLS protocol: hls.protocol = [{mode}] on the enclosing loop or
// function; mode is "fixed" or "floating" (default), also positional.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

// CHECK-LABEL: func.func @default_mode(
// CHECK-SAME: hls.protocol = [{mode = "floating"}]
void default_mode(volatile int *a) {
  {
#pragma HLS protocol
    a[0] = 1;
    a[0] = 0;
  }
}

// CHECK-LABEL: func.func @fixed(
// CHECK-SAME: hls.protocol = [{mode = "fixed"}]
void fixed(volatile int *a) {
  {
#pragma HLS PROTOCOL fixed
    a[0] = 1;
    a[0] = 0;
  }
}

// CHECK-LABEL: func.func @loop_floating(
// CHECK: } {hls.protocol = [{mode = "floating"}]}
void loop_floating(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS protocol mode=floating
    a[i] = i;
  }
}

#ifdef ERRORS
void errors(int *a) {
// ERR: protocol.c:[[@LINE+1]]:27: error: invalid value 'strict' for option 'mode' in '#pragma HLS protocol': expected one of: fixed, floating
#pragma HLS protocol mode=strict
  a[0] = 0;
}
#endif
