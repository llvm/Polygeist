// A class that is NOT a scalar wrapper (it has a member of its own) but
// derives from one through a wrapper chain keeps the aggregate lowering; a
// conversion between it and its bases steps into the scalar once and is the
// identity along the wrapper part of the path (this used to assert in
// GetAddressOfBaseClass when the walk reached the wrapper base).
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --scalar-wrapper-abi 2>&1 | FileCheck %s

struct B {
  short v;
};
struct D : B {};
struct E : D {
  int tag;
};

// CHECK-NOT: Assertion
// CHECK-LABEL: func.func @_Z1gP1Ei(%arg0: memref<?x!llvm.struct<(i16, i32)>>, %arg1: i32) -> i16
// CHECK: %[[P:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.load %[[P]] : !llvm.ptr -> i16
// CHECK: llvm.getelementptr %[[P]][0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, i32)>
short g(E *e, int i) { return e[i].v + (short)e[i].tag; }

// CHECK-LABEL: func.func @_Z1hP1B(%arg0: memref<?xi16>) -> i16
// CHECK: llvm.getelementptr %{{.*}}[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i16, i32)>
short h(B *b) { return static_cast<E *>(b)->tag; }

// CHECK-LABEL: func.func @_Z1kR1D(%arg0: memref<?xi16>) -> i16
// CHECK-NEXT: %[[V:.*]] = affine.load %arg0[0] : memref<?xi16>
// CHECK-NEXT: return %[[V]] : i16
short k(D &d) {
  B &b = d;
  return b.v;
}
