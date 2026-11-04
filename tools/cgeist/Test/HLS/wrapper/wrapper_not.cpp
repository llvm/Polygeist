// Control: records that are NOT scalar wrappers keep their lowering with
// --scalar-wrapper-abi, identical to the default: two fields (merged memref),
// an array field, a virtual method, a reference member, a union, and a
// member with an explicit alignment (the record is larger than its scalar).
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --scalar-wrapper-abi 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank 2>&1 | FileCheck %s

struct Two {
  int a, b;
};
struct Arr {
  int a[4];
};
struct Virt {
  int v;
  virtual int f() { return v; }
};
struct Ref {
  int &r;
  Ref(int &x) : r(x) {}
};
union U {
  int a;
  float b;
};

// CHECK-LABEL: func.func @_Z3twoR3Two(%arg0: memref<?x2xi32>) -> i32
int two(Two &t) { return t.a + t.b; }

// CHECK-LABEL: func.func @_Z3arrR3Arr(%arg0: memref<?x!llvm.struct<(array<4 x i32>)>>) -> i32
int arr(Arr &t) { return t.a[2]; }

// CHECK-LABEL: func.func @_Z4virtR4Virt(%arg0: memref<?x!llvm.struct<packed (ptr, i32, array<4 x i8>)>>) -> i32
int virt(Virt &t) { return t.v; }

// CHECK-LABEL: func.func @_Z3refR3Ref(%arg0: memref<?x1xmemref<?xi32>>) -> i32
int ref(Ref &t) { return t.r; }

// CHECK-LABEL: func.func @_Z3uniR1U(%arg0: memref<?x!llvm.struct<(i32)>>) -> i32
int uni(U &t) { return t.a; }

struct AW {
  alignas(16) int v;
};

// CHECK-LABEL: func.func @_Z2awR2AW(%arg0: memref<?x!llvm.struct<(i32)>>) -> i32
int aw(AW &t) { return t.v; }
