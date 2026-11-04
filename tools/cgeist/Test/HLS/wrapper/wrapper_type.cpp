// A "scalar wrapper" (a class whose only data member is one scalar) is
// lowered to that scalar under --scalar-wrapper-abi: values are i32,
// references / `this` memref<?xi32>, arrays memref<4x8xi32>.
// wrapper_off.cpp checks that without the option the old lowering is kept.
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --scalar-wrapper-abi 2>&1 | FileCheck %s

struct W {
  int v;
  W() = default;
  W(int x) : v(x) {}
  int get() const { return v; }
  void set(int x) { v = x; }
};

// CHECK-NOT: llvm.struct
// CHECK-NOT: memref2pointer

// CHECK-LABEL: func.func @_Z5byval1W(%arg0: i32) -> i32
W byval(W a) { return a; }

// CHECK-LABEL: func.func @_Z6kernelPA8_1WPS_RS_i(%arg0: memref<4x8xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: i32) -> i32
// CHECK: memref.alloca() {hls.preserve, polygeist.varname = "loc"} : memref<4xi32>
// CHECK: affine.load %arg0[%{{.*}}, 0] : memref<4x8xi32>
// CHECK: affine.load %arg0[2, 3] : memref<4x8xi32>
int kernel(W a[4][8], W *p, W &r, int k) {
  W loc[4];
  for (int i = 0; i < 4; i++)
    loc[i] = a[i][0];
  loc[1] = r;
  p->set(a[2][3].get() + loc[k].v);
  return p->v;
}

// CHECK-LABEL: func.func @_ZNK1W3getEv(%arg0: memref<?xi32>) -> i32
// CHECK: affine.load %arg0[0] : memref<?xi32>
