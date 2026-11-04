// Construction of scalar wrappers under --scalar-wrapper-abi: a defaulted
// default constructor emits nothing, a user constructor is called with
// `this : memref<?xi16>`, a trivial copy is a load and a store, every element
// of an N-D array with a user default constructor is constructed (one loop
// per dimension; this used to pass the whole sub-array as `this`), an init
// list constructs each element, value-initialisation stores a zero instead of
// a memset. The objects are passed to an external `sink` so that they survive
// cgeist's own optimisations.
// RUN: cgeist %s --function='*' -S --hls-annotate -O0 --scalar-wrapper-abi 2>&1 | FileCheck %s

struct F {
  short v;
  F() = default;
  F(const F &) = default;
  F &operator=(const F &) = default;
  F(int x) : v((short)x) {}
};
struct G {
  short v;
  G() { v = 7; }
};
void sink(F *p);
void sinkG(G *p);


// CHECK-NOT: llvm.struct
// CHECK-NOT: llvm.intr.memset
// CHECK-NOT: memref2pointer

// CHECK-LABEL: func.func @_Z8defaultsv()
// CHECK: memref.alloca() {polygeist.varname = "a"} : memref<1xi16>
// CHECK: call @_ZN1FC1Ei(%{{.*}}, %c3_i32) : (memref<?xi16>, i32) -> ()
// CHECK-NOT: call @_ZN1FaSERKS_
// CHECK: call @_Z4sinkP1F(%{{.*}}) : (memref<?xi16>) -> ()
// CHECK: return
void defaults() {
  F a;
  F b(3);
  a = b;
  sink(&a);
}

// CHECK-LABEL: func.func @_Z4copyP1F(
// CHECK: %[[L:.*]] = affine.load %arg0[0] : memref<?xi16>
// CHECK: affine.store %[[L]], %{{.*}}[0] : memref<1xi16>
// CHECK-NOT: call @_ZN1FC1ERKS_
// CHECK: call @_Z4sinkP1F(
void copy(F *in) {
  F b = in[0];
  sink(&b);
}

// CHECK-LABEL: func.func @_Z4gridv()
// CHECK: memref.alloca() {hls.preserve, polygeist.varname = "g"} : memref<4x4xi16>
// CHECK: scf.for
// CHECK: scf.for
// CHECK: call @_ZN1GC1Ev(%{{.*}}) : (memref<?xi16>) -> ()
void grid() {
  G g[4][4];
  sinkG(&g[0][0]);
}

// CHECK-LABEL: func.func @_Z4listv()
// CHECK: call @_ZN1FC1Ei(%{{.*}}, %c1_i32) : (memref<?xi16>, i32) -> ()
// CHECK: call @_ZN1FC1Ei(%{{.*}}, %c2_i32) : (memref<?xi16>, i32) -> ()
void list() {
  F c[2] = {1, 2};
  sink(c);
}

// CHECK-LABEL: func.func @_Z4zerov()
// CHECK: affine.store %c0_i16, %{{.*}}[0] : memref<1xi16>
// CHECK: call @_Z4sinkP1F(
void zero() {
  F z{};
  sink(&z);
}

// CHECK-LABEL: func.func @_Z8zeroGridv()
// CHECK: memref.alloca() {hls.preserve, polygeist.varname = "z"} : memref<2x3xi16>
// CHECK: %c0_i16
// CHECK: call @_Z4sinkP1F(
void zeroGrid() {
  F z[2][3] = {};
  sink(&z[1][2]);
}
