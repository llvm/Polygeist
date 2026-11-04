// A wrapper chain (a derived class without members of its own over a wrapper
// base) is the same scalar throughout: `this->v` in a D method, `b.v` through
// a B&, and the derived-to-base conversion are all the one memref, with no
// base-offset GEP and no null check on the conversion.
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --scalar-wrapper-abi 2>&1 | FileCheck %s

struct B {
  _BitInt(20) v;
};
struct D : B {
  D() = default;
  D(int x) { v = (_BitInt(20))x; }
  _BitInt(20) get() const { return this->v; }
};

// CHECK-NOT: llvm.struct
// CHECK-NOT: llvm.icmp
// CHECK-NOT: llvm.getelementptr
// CHECK-NOT: memref2pointer

// CHECK-LABEL: func.func @_Z7viaBaseRK1B(%arg0: memref<?xi20>) -> i20
// CHECK: affine.load %arg0[0] : memref<?xi20>
_BitInt(20) viaBase(const B &b) { return b.v; }

// CHECK-LABEL: func.func @_Z7derivedPA5_1DRS_(%arg0: memref<3x5xi20>, %arg1: memref<?xi20>) -> i20
// CHECK: affine.load %arg0[1, 2] : memref<3x5xi20>
_BitInt(20) derived(D a[3][5], D &d) { return viaBase(d) + a[1][2].get(); }

// CHECK-LABEL: func.func @_ZNK1D3getEv(%arg0: memref<?xi20>) -> i20
// CHECK: affine.load %arg0[0] : memref<?xi20>
