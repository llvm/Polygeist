// Static locals under --scalar-wrapper-abi. A static local with constant
// initialisation (C++ [basic.start.static]) is initialised before the program
// runs: its global carries clang's evaluated value as a dense initial value
// and nothing runs in the function (no guard flag, no stores), so a
// `static const` table of wrappers is a ROM. A static local whose initialiser
// is not a constant expression keeps the run-time guarded initialisation; it
// is stored through the global's static shape (an initialiser of array type
// with cleanups used to assert in ValueCategory::store).
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --scalar-wrapper-abi 2>&1 | FileCheck %s
// RUN: cgeist %s --function='*' -S --hls-annotate --raise-scf-to-affine --memref-fullrank --scalar-wrapper-abi 2>&1 | FileCheck %s --check-prefix=NOINIT

struct W {
  short v;
  W() = default;
  template <class T> constexpr W(const T &x) : v((short)x) {}
};
template <class T> const T &mx(const T &a, const T &b) { return a > b ? a : b; }

// NOINIT-NOT: _ZZ3tabiE2lt@init
// NOINIT-NOT: _ZZ7counteriE1c@init

// CHECK-DAG: memref.global "private" @"_Z3tabi@static@_ZZ3tabiE2lt" : memref<4xi16> = dense<[1, 2, 3, -1]>
// CHECK-DAG: memref.global "private" @"_Z7counteri@static@_ZZ7counteriE1c" : memref<1xi32> = dense<3>
// CHECK-DAG: memref.global "private" @"_Z3runi@static@_ZZ3runiE2lt@init" : memref<1xi1> = dense<true>

// CHECK-LABEL: func.func @_Z3tabi(%arg0: i32) -> i16
// CHECK-NOT: scf.if
// CHECK: memref.get_global @"_Z3tabi@static@_ZZ3tabiE2lt" : memref<4xi16>
// CHECK-NOT: affine.store
// CHECK: return
short tab(int i) {
  static const W lt[4] = {1, 2, 3, 65535};
  return lt[i].v;
}

// CHECK-LABEL: func.func @_Z7counteri(%arg0: i32) -> i32
// CHECK-NOT: scf.if
// CHECK: return
int counter(int n) {
  static int c = 3;
  c += n;
  return c;
}

// CHECK-LABEL: func.func @_Z3runi(%arg0: i32) -> i16
// CHECK: scf.if
// CHECK: affine.store %c2_i16, %{{.*}}[0] : memref<4xi16>
// CHECK: affine.store %c5_i16, %{{.*}}[3] : memref<4xi16>
short run(int i) {
  static const short lt[4] = {mx<short>(1, 2), 3, 4, 5};
  return lt[i];
}

// A static wrapper object with a run-time constructor (an `hls::stream`
// stand-in) is constructed through the dynamic-shape view of its global,
// which is the `this` type of the constructor.
struct Stream {
  int depth;
  Stream(const char *name) : depth(name[0]) {}
  void write(int v) { depth += v; }
};

// CHECK-LABEL: func.func @_Z6streami(%arg0: i32) -> i32
// CHECK: scf.if
// CHECK: return
int stream(int v) {
  static Stream s("fifo");
  s.write(v);
  return s.depth;
}
