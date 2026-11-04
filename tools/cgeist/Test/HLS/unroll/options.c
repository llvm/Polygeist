// #pragma HLS unroll: hls.unroll = [{factor, skip_exit_check}] on the loop;
// factor = 0 means "unroll completely" (the default); `off` gives
// hls.unroll_off.
// RUN: cgeist %s --function='*' -S --hls-annotate 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine 2>&1 | FileCheck %s

// CHECK-NOT: warning
// CHECK-NOT: error

// CHECK-LABEL: func.func @defaults(
// CHECK: } {hls.unroll = [{factor = 0 : i32, skip_exit_check = false}]}
void defaults(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS unroll
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @factor(
// CHECK: } {hls.unroll = [{factor = 4 : i32, skip_exit_check = false}]}
void factor(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS unroll factor=4
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @skip_exit_check(
// CHECK: } {hls.unroll = [{factor = 0 : i32, skip_exit_check = true}]}
void skip_exit_check(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS unroll skip_exit_check
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @all(
// CHECK: } {hls.unroll = [{factor = 2 : i32, skip_exit_check = true}]}
void all(int *a, int n) {
  for (int i = 0; i < n; ++i) {
#pragma HLS UNROLL factor=2 skip_exit_check=true
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @off(
// CHECK-NOT: hls.unroll =
// CHECK: } {hls.unroll_off}
void off(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS unroll off
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @off_true(
// CHECK: } {hls.unroll_off}
void off_true(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS unroll off=true
    a[i] = i;
  }
}

// unroll and pipeline on one loop are independent attributes.
// CHECK-LABEL: func.func @with_pipeline(
// CHECK: } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}], hls.unroll = [{factor = 2 : i32, skip_exit_check = false}]}
void with_pipeline(int *a) {
  for (int i = 0; i < 8; ++i) {
#pragma HLS unroll factor=2
#pragma HLS pipeline II=1
    a[i] = i;
  }
}
