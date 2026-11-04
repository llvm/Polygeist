// hls.unroll survives -O levels, affine raising and inlining.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --check-prefixes=CHECK,NOINLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O2 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 --raise-scf-to-affine --memref-fullrank 2>&1 | FileCheck %s --check-prefixes=CHECK,NOINLINE

// CHECK-NOT: warning

// CHECK-LABEL: func.func @callee(
// CHECK: } {hls.unroll = [{factor = 4 : i32, skip_exit_check = false}]}
void callee(int *a) {
  for (int i = 0; i < 32; ++i) {
#pragma HLS unroll factor=4
    a[i] += i;
  }
}

// CHECK-LABEL: func.func @top(
// NOINLINE: call @callee
// INLINE: } {hls.unroll = [{factor = 4 : i32, skip_exit_check = false}]}
void top(int *a) { callee(a); }
