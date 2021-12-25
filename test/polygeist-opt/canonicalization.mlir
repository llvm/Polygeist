// RUN: polygeist-opt --canonicalize --split-input-file %s | FileCheck %s
// XFAIL: *
// CHECK: func @main(%arg0: index) -> memref<30xi32> {
// CHECK:   %0 = memref.alloca() : memref<30x30xi32>
// CHECK:   %1 = memref.subview %0[%arg0, 0] [1, 30] [1, 1] : memref<30x30xi32> to memref<30xi32>
// CHECK:   return %1 : memref<30xi32>
// CHECK: }
module {
  func @main(%arg0 : index) -> memref<30xi32> {
    %0 = memref.alloca() : memref<30x30xi32>
    %1 = "polygeist.subindex"(%0, %arg0) : (memref<30x30xi32>, index) -> memref<30xi32>
    return %1 : memref<30xi32>
  }
}

// -----

// CHECK:  func @main(%arg0: index) -> memref<1000xi32> {
// CHECK:    %0 = memref.alloca() : memref<2x1000xi32>
// CHECK:    %1 = memref.subview %0[%arg0, 0] [1, 1000] [1, 1] : memref<2x1000xi32> to memref<1000xi32>
// CHECK:    return %1 : memref<1000xi32>
// CHECK:  }
func @main(%arg0 : index) -> memref<1000xi32> {
  %c0 = arith.constant 0 : index
  %1 = memref.alloca() : memref<2x1000xi32>
    %3 = "polygeist.subindex"(%1, %arg0) : (memref<2x1000xi32>, index) -> memref<?x1000xi32>
    %4 = "polygeist.subindex"(%3, %c0) : (memref<?x1000xi32>, index) -> memref<1000xi32>
  return %4 : memref<1000xi32>
}
