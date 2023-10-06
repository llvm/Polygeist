// RUN: cgeist %s %stdinclude --function=* -S | FileCheck %s

#include <stdio.h>
#include <new>

struct A {
  float x, y;
};

void f(A *a) { printf("a.x = %f, a.y = %f\n", a->x, a->y); }

int main(int argc, char const *argv[]) {
  auto *a = new A{1.0f, 2.0f};
  f(a);
  return 0;
}

extern "C" char *malloc(int asd);

class SimStream {
public:
  int n;
  SimStream(int _n) {
    n = _n;
  }
};

int bar() {
  SimStream a(2);
  return a.n;
}

SimStream *baz() {
  SimStream *b = new SimStream(30);
  return b;
}

int *bat() {
  int *b = new int[10];
  return b;
}

int *baf() {
  int *b = new int[3] {1, 2, 3};
  return b;
}

int foo() {
  int e = 1;
  int * __new_start((int *)malloc(sizeof(int)));
  ::new((void*)__new_start) int(e);
  return (int) __new_start[0];
}

// CHECK-LABEL:   func.func @main(
// CHECK-SAME:                    %[[VAL_0:[A-Za-z0-9_]*]]: i32,
// CHECK-SAME:                    %[[VAL_1:[A-Za-z0-9_]*]]: memref<?xmemref<?xi8>>) -> i32
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = arith.constant 2.000000e+00 : f64
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = arith.constant 1.000000e+00 : f64
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:[A-Za-z0-9_]*]] = llvm.mlir.addressof @str0 : !llvm.ptr
// CHECK:           %[[VAL_6:[A-Za-z0-9_]*]] = llvm.getelementptr %[[VAL_5]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<20 x i8>
// CHECK:           %[[VAL_7:[A-Za-z0-9_]*]] = llvm.call @printf(%[[VAL_6]], %[[VAL_3]], %[[VAL_2]]) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, f64, f64) -> i32
// CHECK:           return %[[VAL_4]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func @_Z3barv() -> i32
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = arith.constant 2 : i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func @_ZN9SimStreamC1Ei(
// CHECK-SAME:                                 %[[VAL_0:[A-Za-z0-9_]*]]: memref<?x1xi32>,
// CHECK-SAME:                                 %[[VAL_1:[A-Za-z0-9_]*]]: i32)
// CHECK:           affine.store %[[VAL_1]], %[[VAL_0]][0, 0] : memref<?x1xi32>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_Z3bazv() -> memref<?x1xi32>
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = arith.constant 30 : i32
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = memref.alloc() : memref<1x1xi32>
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = memref.cast %[[VAL_1]] : memref<1x1xi32> to memref<?x1xi32>
// CHECK:           affine.store %[[VAL_0]], %[[VAL_1]][0, 0] : memref<1x1xi32>
// CHECK:           return %[[VAL_2]] : memref<?x1xi32>
// CHECK:         }

// CHECK-LABEL:   func.func @_Z3batv() -> memref<?xi32>
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = memref.alloc() : memref<10xi32>
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = memref.cast %[[VAL_0]] : memref<10xi32> to memref<?xi32>
// CHECK:           return %[[VAL_1]] : memref<?xi32>
// CHECK:         }

// CHECK-LABEL:   func.func @_Z3bafv() -> memref<?xi32>
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = arith.constant 3 : i32
// CHECK:           %[[VAL_1:[A-Za-z0-9_]*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:[A-Za-z0-9_]*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:[A-Za-z0-9_]*]] = memref.alloc() : memref<3xi32>
// CHECK:           %[[VAL_4:[A-Za-z0-9_]*]] = memref.cast %[[VAL_3]] : memref<3xi32> to memref<?xi32>
// CHECK:           affine.store %[[VAL_2]], %[[VAL_3]][0] : memref<3xi32>
// CHECK:           affine.store %[[VAL_1]], %[[VAL_3]][1] : memref<3xi32>
// CHECK:           affine.store %[[VAL_0]], %[[VAL_3]][2] : memref<3xi32>
// CHECK:           return %[[VAL_4]] : memref<?xi32>
// CHECK:         }

// CHECK-LABEL:   func.func @_Z3foov() -> i32
// CHECK:           %[[VAL_0:[A-Za-z0-9_]*]] = arith.constant 1 : i32
// CHECK:           return %[[VAL_0]] : i32
// CHECK:         }

