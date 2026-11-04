// The Xilinx HLS bitwidth(N) attribute through cgeist: with
// -Xclang -fxlx-bitwidth it is _BitInt(N), so the MLIR has iN; without the
// flag it is ignored as before (i32). The declarations below are written for
// this test from the attribute's documented behaviour; none is taken from the
// Vitis-installed headers.
// REQUIRES: xlx-bitwidth
// RUN: cgeist %s -Xclang -fxlx-bitwidth --function='*' -S -o - | FileCheck %s
// RUN: cgeist %s -Xclang -fxlx-bitwidth -Xclang -fxlx-apint-promote --function='*' -S -o - | FileCheck %s --check-prefix=PROMOTE
// RUN: cgeist %s --function='*' -S -o - 2>&1 | FileCheck %s --check-prefix=OFF

// A field whose width is a template parameter.
template <int N> struct Word {
  unsigned bits __attribute__((bitwidth(N)));
};

// OFF: warning: unknown attribute 'bitwidth' ignored

// CHECK-LABEL: func.func @add17(
// CHECK-SAME: %arg0: i17, %arg1: i17) -> i17
// CHECK: arith.addi %arg0, %arg1 : i17
// OFF-LABEL: func.func @add17(
// OFF-SAME: %arg0: i32, %arg1: i32) -> i32
extern "C" int __attribute__((bitwidth(17))) add17(int __attribute__((bitwidth(17))) a,
                                                   int __attribute__((bitwidth(17))) b) {
  return a + b;
}

// A template-dependent width.
// CHECK-LABEL: func.func @wide(
// CHECK-SAME: %arg0: i300) -> i300
extern "C" unsigned __attribute__((bitwidth(300))) wide(unsigned __attribute__((bitwidth(300))) x) {
  Word<300> w;
  w.bits = x;
  return w.bits + 1;
}

// An array of a 12-bit type stays a memref of i12.
// CHECK-LABEL: func.func @arr(
// CHECK-SAME: %arg0: memref<?xi12>
// OFF-LABEL: func.func @arr(
// OFF-SAME: %arg0: memref<?xi32>
extern "C" int arr(int __attribute__((bitwidth(12))) *buf, int i) {
  return buf[i];
}

// The promotion: a shift of an 8-bit value is done in 32 bits with
// -fxlx-apint-promote (Vitis), in 8 bits without.
// CHECK-LABEL: func.func @shl8(
// CHECK: arith.shli {{.*}} : i8
// PROMOTE-LABEL: func.func @shl8(
// PROMOTE: arith.extui {{.*}} : i8 to i32
// PROMOTE: arith.shli {{.*}} : i32
extern "C" int shl8(unsigned __attribute__((bitwidth(8))) x) { return x << 4; }
