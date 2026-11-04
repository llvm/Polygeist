// Constant initialisers are evaluated by clang's constant evaluator under
// --scalar-wrapper-abi: a `static const int F = W - I` member reads as the
// constant (no global, no load), and a global array or scalar of a scalar
// wrapper with constexpr constructors gets a dense initial value.
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --scalar-wrapper-abi 2>&1 | FileCheck %s

struct Q {
  short v;
  constexpr Q() : v(0) {}
  constexpr Q(int x) : v((short)(x * 4)) {}
  constexpr Q(double d) : v((short)(d * 4)) {}
};
template <int W, int I> struct FX {
  static const int F = W - I;
  short v;
};

const Q tbl[4] = {1, 2.5, 3, 4};
const Q one = 1;
Q mutableQ = 2;

// CHECK-NOT: warning not initializing global
// CHECK-DAG: memref.global "private" @_ZL3tbl : memref<4xi16> = dense<[4, 10, 12, 16]>
// CHECK-DAG: memref.global "private" @_ZL3one : memref<1xi16> = dense<4>
// CHECK-DAG: memref.global @mutableQ : memref<1xi16> = dense<8>

// CHECK-LABEL: func.func @_Z4useFi(%arg0: i32) -> i32
// CHECK-NOT: memref.get_global
// CHECK: arith.addi %arg0, %c11_i32 : i32
int useF(int x) { return x + FX<16, 5>::F; }

// CHECK-LABEL: func.func @_Z7readTbli(%arg0: i32) -> i16
// CHECK: memref.get_global @_ZL3tbl : memref<4xi16>
short readTbl(int i) { return tbl[i].v; }

// CHECK-LABEL: func.func @_Z7readOnev() -> i16
// CHECK: memref.get_global @_ZL3one : memref<1xi16>
short readOne() { return one.v; }

// CHECK-LABEL: func.func @_Z11readMutablev() -> i16
// CHECK: memref.get_global @mutableQ
short readMutable() { return mutableQ.v; }
