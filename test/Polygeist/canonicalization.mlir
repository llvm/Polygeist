// RUN: polygeist-opt --canonicalize --split-input-file %s | FileCheck %s

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
