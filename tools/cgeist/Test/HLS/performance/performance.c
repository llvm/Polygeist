// #pragma HLS performance: hls.performance = [{target_ti?, target_tl?, unit}]
// on the enclosing loop or function. unit defaults to "cycle"; targets are
// integers, or f64 when written as floating-point literals (unit=sec).
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --strict-whitespace
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s --strict-whitespace --check-prefixes=CHECK,INLINE
// RUN: not cgeist %s --function='*' -S --hls-annotate -DERRORS -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERR

// CHECK-NOT: warning

// CHECK-LABEL: func.func @loop_ti(
// CHECK: {{^}}    } {hls.performance = [{target_ti = 100 : i32, unit = "cycle"}]}
void loop_ti(int *a) {
  for (int i = 0; i < 64; ++i) {
#pragma HLS performance target_ti=100
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @loop_tl(
// CHECK: {{^}}    } {hls.performance = [{target_tl = 64 : i32, unit = "cycle"}]}
void loop_tl(int *a) {
  for (int i = 0; i < 64; ++i) {
#pragma HLS performance target_tl=64 unit=cycle
    a[i] = i;
  }
}

// CHECK-LABEL: func.func @function_sec(
// CHECK-SAME: hls.performance = [{target_ti = 1.500000e-06 : f64, target_tl = 2.000000e-06 : f64, unit = "sec"}]
void function_sec(int *a) {
#pragma HLS PERFORMANCE target_ti=1.5e-6 target_tl=2.0e-6 unit=sec
  for (int i = 0; i < 64; ++i)
    a[i] = i;
}

// CHECK-LABEL: func.func @top(
// INLINE: } {hls.performance = [{target_ti = 100 : i32, unit = "cycle"}]}
void top(int *a) { loop_ti(a); }

#ifdef ERRORS
void errors(int *a) {
  for (int i = 0; i < 64; ++i) {
// ERR-DAG: performance.c:[[@LINE+1]]:1: error: '#pragma HLS performance' requires 'target_ti' or 'target_tl'
#pragma HLS performance unit=cycle
    a[i] = i;
  }
  for (int i = 0; i < 64; ++i) {
// ERR-DAG: performance.c:[[@LINE+1]]:35: error: option 'target_ti' in '#pragma HLS performance' must be positive, got 0
#pragma HLS performance target_ti=0
    a[i] = i;
  }
  for (int i = 0; i < 64; ++i) {
// ERR-DAG: performance.c:[[@LINE+1]]:42: error: invalid value 'minutes' for option 'unit' in '#pragma HLS performance': expected one of: cycle, sec
#pragma HLS performance target_ti=4 unit=minutes
    a[i] = i;
  }
}
#endif
