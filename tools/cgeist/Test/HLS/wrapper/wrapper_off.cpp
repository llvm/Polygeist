// --scalar-wrapper-abi is opt-in: without it (or with =false) a scalar
// wrapper class keeps the lowering it had before the option existed (a
// merged memref for a plain one-member class, an LLVM struct for a class
// with a base); with it, both are the scalar.
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --scalar-wrapper-abi=false 2>&1 | FileCheck %s --check-prefix=OFF
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --scalar-wrapper-abi 2>&1 | FileCheck %s --check-prefix=ON

struct W {
  int v;
  W() = default;
  int get() const { return v; }
};
struct B {
  _BitInt(12) v;
};
struct D : B {
  D() = default;
};

// OFF-LABEL: func.func @_Z6kernelP1WRS_(%arg0: memref<?x1xi32>, %arg1: memref<?x1xi32>) -> i32
// ON-LABEL: func.func @_Z6kernelP1WRS_(%arg0: memref<?xi32>, %arg1: memref<?xi32>) -> i32
int kernel(W *p, W &r) { return p[2].get() + r.v; }

// OFF-LABEL: func.func @_Z5chainP1D(%arg0: memref<?x!llvm.struct<(struct<(i12)>)>>) -> i12
// ON-LABEL: func.func @_Z5chainP1D(%arg0: memref<?xi12>) -> i12
_BitInt(12) chain(D *d) { return d[1].v; }
