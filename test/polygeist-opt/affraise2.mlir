// RUN: polygeist-opt --raise-scf-to-affine --split-input-file %s | FileCheck %s

module {
  func @main(%12 : i1, %14 : i32, %18 : memref<?xf32>, %19 : memref<?xf32> ) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.if %12 {
      %15 = arith.index_cast %14 : i32 to index
      %16 = arith.muli %15, %c4 : index
      %17 = arith.divui %16, %c4 : index
      scf.for %arg2 = %c0 to %17 step %c1 {
        %20 = memref.load %19[%arg2] : memref<?xf32>
        memref.store %20, %18[%arg2] : memref<?xf32>
      }
    }
    return
  }
}

// CHECK:   func @withinif(%arg0: memref<?xf64>, %arg1: i32, %arg2: memref<?xf64>, %arg3: i1) {
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     scf.if %arg3 {
// CHECK-NEXT:       affine.for %arg4 = 1 to %0 {
// CHECK-NEXT:         %1 = memref.load %arg0[%arg4] : memref<?xf64>
// CHECK-NEXT:         memref.store %1, %arg2[%arg4] : memref<?xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
