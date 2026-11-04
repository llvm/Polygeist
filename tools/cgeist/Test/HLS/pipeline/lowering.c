// hls.pipeline survives every cgeist pipeline configuration: each -O level,
// affine raising, inlining of the callee into its caller, and later
// unrolling/canonicalization of the annotated loop.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --check-prefixes=CHECK,NOINLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O1 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O2 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 --raise-scf-to-affine 2>&1 | FileCheck %s --check-prefixes=CHECK,NOINLINE,AFF
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE,AFF
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine --memref-fullrank 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE,AFF
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine -o - \
// RUN:   | polygeist-opt --affine-loop-unroll="unroll-factor=2" --canonicalize \
// RUN:   | FileCheck %s --check-prefix=UNROLL

// CHECK-NOT: warning

// The callee keeps its own annotated loop.
// CHECK-LABEL: func.func @callee(
// AFF: affine.for %{{.*}} = 0 to 64 {
// CHECK: } {hls.pipeline = [{II = 2 : i32, rewind = false, style = "stp"}]}
void callee(int *a) {
  for (int i = 0; i < 64; ++i) {
#pragma HLS pipeline II=2
    a[i] = 3 * a[i];
  }
}

// After inlining, the copy of the loop in the caller is annotated too;
// the caller's own loop keeps its own pragma.
// CHECK-LABEL: func.func @top(
// NOINLINE: call @callee
// INLINE-NOT: call @callee
// INLINE: } {hls.pipeline = [{II = 2 : i32, rewind = false, style = "stp"}]}
// CHECK: } {hls.pipeline = [{II = 1 : i32, rewind = false, style = "stp"}]}
void top(int *a, int *b) {
  callee(a);
  for (int i = 0; i < 64; ++i) {
#pragma HLS pipeline II=1
    b[i] = a[i] + 1;
  }
}

// Partial unrolling rewrites the loop in place and keeps its attributes.
// UNROLL-LABEL: func.func @callee(
// UNROLL: affine.for %{{.*}} = 0 to 64 step 2 {
// UNROLL: } {hls.pipeline = [{II = 2 : i32, rewind = false, style = "stp"}]}
