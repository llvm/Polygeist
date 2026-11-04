// Value-initialisation of scalar-wrapper SUBOBJECTS under
// --scalar-wrapper-abi: a base `W2() : F() {}` or member `b()` initialiser
// zeroes exactly the one object being constructed (one store at index 0 of
// `this`), not every element of the memref `this` points into (it used to
// loop over memref.dim of `this`, writing past the object when it is an
// array element). `new F[n]()` zeroes the n elements.
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --scalar-wrapper-abi 2>&1 | FileCheck %s

struct F {
  short v;
  F() = default;
  F(int x) : v((short)x) {}
};
struct W2 : F {
  W2() : F() {}
};
struct P {
  F a;
  F b;
  P() : a(5), b() {}
};
void sinkW(W2 *p);
void sinkP(P *p);
void sink(F *p);

// CHECK-NOT: memref.dim

// CHECK-LABEL: func.func @_Z2k1v()
// CHECK: %[[A:.*]] = memref.alloca() {hls.preserve, polygeist.varname = "arr"} : memref<4xi16>
// CHECK: affine.for %[[I:.*]] = 0 to 4 {
// CHECK-NEXT: affine.store %c0_i16, %[[A]][%[[I]]] : memref<4xi16>
// CHECK-NEXT: }
void k1() {
  W2 arr[4];
  sinkW(arr);
}

// CHECK-LABEL: func.func @_ZN2W2C1Ev(%arg0: memref<?xi16>)
// CHECK-NEXT: %[[Z:.*]] = arith.constant 0 : i16
// CHECK-NEXT: affine.store %[[Z]], %arg0[0] : memref<?xi16>
// CHECK-NEXT: return

// CHECK-LABEL: func.func @_ZN1PC1Ev(
// CHECK-NOT: affine.for
// CHECK: llvm.store %c5_i16
// CHECK-NOT: affine.for
// CHECK: llvm.store %c0_i16
// CHECK-NEXT: return
void k2() {
  P p;
  sinkP(&p);
}

// CHECK-LABEL: func.func @_Z2k4i(%arg0: i32)
// CHECK: %[[N:.*]] = arith.index_cast %arg0 : i32 to index
// CHECK: %[[M:.*]] = memref.alloc(%[[N]]) : memref<?xi16>
// CHECK: affine.for %[[I:.*]] = 0 to %[[N]] {
// CHECK-NEXT: affine.store %c0_i16, %[[M]][%[[I]]] : memref<?xi16>
void k4(int n) {
  F *p = new F[n]();
  sink(p);
}
