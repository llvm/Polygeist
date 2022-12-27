// RUN: cgeist %s %stdinclude --function=* -S | FileCheck %s

#include <stdio.h>
#include <new>

struct A {
  float x, y;
};

void f(A *a) { printf("a.x = %f, a.y = %f\n", a->x, a->y); }

int main(int argc, char const *argv[]) {
  // CHECK-DAG: %[[two:.*]] = arith.constant 2.000000e+00 : f32
  // CHECK-DAG: %[[one:.*]] = arith.constant 1.000000e+00 : f32
  // CHECK: %[[alloc:.*]] = memref.alloc() : memref<1x2xf32>
  // CHECK: affine.store %[[one]], %[[alloc]][0, 0] : memref<1x2xf32>
  // CHECK: affine.store %[[two]], %[[alloc]][0, 1] : memref<1x2xf32>
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
  // CHECK: func.func @_Z3barv() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
  // CHECK-NEXT:    %[[two:.*]] = arith.constant 2 : i32
  // CHECK-NEXT:    return %[[two:.*]] : i32
  SimStream a(2);
  return a.n;
}

SimStream *baz() {
  // CHECK:  func.func @_Z3bazv()
  // CHECK-NEXT:    %[[thirty:.*]] = arith.constant 30 : i32
  // CHECK-NEXT:    %[[alloc:.*]] = memref.alloc() : memref<1x1xi32>
  // CHECK-NEXT:    %[[cast:.*]] = memref.cast %[[alloc]] : memref<1x1xi32> to memref<?x1xi32>
  // CHECK-NEXT:    affine.store %[[thirty]], %[[alloc]][0, 0] : memref<1x1xi32>
  // CHECK-NEXT:    return %[[cast]] : memref<?x1xi32>
  SimStream *b = new SimStream(30);
  return b;
}

int *bat() {
  // CHECK:  func.func @_Z3batv()
  // CHECK-NEXT:    %[[alloc:.*]] = memref.alloc() : memref<10xi32>
  // CHECK-NEXT:    %[[cast:.*]] = memref.cast %[[alloc]] : memref<10xi32> to memref<?xi32>
  // CHECK-NEXT:    return %[[cast]] : memref<?xi32>
  int *b = new int[10];
  return b;
}

int *baf() {
  // CHECK:  func.func @_Z3bafv()
  // CHECK-DAG:    %[[three:.*]] = arith.constant 3 : i32
  // CHECK-DAG:    %[[two:.*]] = arith.constant 2 : i32
  // CHECK-DAG:    %[[one:.*]] = arith.constant 1 : i32
  // CHECK-NEXT:    %[[alloc:.*]] = memref.alloc() : memref<3xi32>
  // CHECK-NEXT:    %[[cast:.*]] = memref.cast %[[alloc]] : memref<3xi32> to memref<?xi32>
  // CHECK-DAG:    affine.store %[[one]], %alloc[0] : memref<3xi32>
  // CHECK-DAG:    affine.store %[[two]], %alloc[1] : memref<3xi32>
  // CHECK-DAG:    affine.store %[[three]], %alloc[2] : memref<3xi32>
  // CHECK-NEXT:    return %[[cast]] : memref<?xi32>
  int *b = new int[3] {1, 2, 3};
  return b;
}

int foo() {
  // CHECK:  func.func @_Z3foov()
  // CHECK-DAG:    %[[one:.*]] = arith.constant 1 : i32
  // CHECK-DAG:    %[[alloc:.*]] = memref.alloc() : memref<1xi32>
  // CHECK-NEXT:    %[[v1:.*]] = "polygeist.memref2pointer"(%[[alloc]]) : (memref<1xi32>) -> !llvm.ptr<i32>
  // CHECK-NEXT:    llvm.store %[[one]], %[[v1]] : !llvm.ptr<i32>
  // CHECK-NEXT:    %[[v2:.*]] = affine.load %[[alloc]][0] : memref<1xi32>
  // CHECK-NEXT:    return %[[v2]] : i32
  int e = 1;
  int * __new_start((int *)malloc(sizeof(int)));
  ::new((void*)__new_start) int(e);
  return (int) __new_start[0];
}

