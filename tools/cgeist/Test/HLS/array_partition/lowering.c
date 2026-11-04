// hls.array_partition survives every pipeline configuration; when a callee
// is inlined, its local arrays are cloned into the caller with their
// attributes, and a caller's array passed to a callee keeps its own.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 2>&1 | FileCheck %s --check-prefixes=CHECK,NOINLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O1 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O2 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 --raise-scf-to-affine 2>&1 | FileCheck %s --check-prefixes=CHECK,NOINLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: cgeist %s --function='*' -S --hls-annotate -O3 --raise-scf-to-affine --memref-fullrank 2>&1 | FileCheck %s --check-prefixes=CHECK,INLINE
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine -o - \
// RUN:   | polygeist-opt --canonicalize --cse --affine-loop-unroll="unroll-factor=2" \
// RUN:   | FileCheck %s --check-prefixes=CHECK,INLINE

// CHECK-NOT: warning

// CHECK-LABEL: func.func @filter(
// CHECK: memref.alloca() {hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "taps"}], hls.preserve, polygeist.varname = "taps"} : memref<16xi32>
void filter(int *x, int *y) {
  int taps[16];
#pragma HLS array_partition variable=taps cyclic factor=4
  for (int i = 0; i < 16; ++i)
    taps[i] = x[i] * 3;
  for (int i = 0; i < 16; ++i) {
    int acc = 0;
    for (int j = 0; j < 16; ++j)
      acc += taps[j] * x[i + j];
    y[i] = acc;
  }
}

// CHECK-LABEL: func.func @top(
// CHECK: memref.alloca() {hls.array_partition = [{dim = 1 : i32, kind = "complete", variable = "tmp"}], hls.preserve, polygeist.varname = "tmp"} : memref<32xi32>
// INLINE: memref.alloca() {hls.array_partition = [{dim = 1 : i32, factor = 4 : i32, kind = "cyclic", variable = "taps"}], hls.preserve, polygeist.varname = "taps"} : memref<16xi32>
// NOINLINE: call @filter
// INLINE-NOT: call @filter
void top(int *x, int *y) {
  int tmp[32];
#pragma HLS array_partition variable=tmp complete
  for (int i = 0; i < 32; ++i)
    tmp[i] = x[i] + 1;
  filter(tmp, y);
}
