// RUN: polygeist-opt --raise-scf-to-affine --split-input-file %s | FileCheck %s

module {
  func @withinif(%arg0: memref<?xf64>, %arg1: i32, %arg2: memref<?xf64>, %arg3: i1) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.if %arg3 {
      %3 = arith.index_cast %arg1 : i32 to index
      scf.for %arg6 = %c1 to %3 step %c1 {
        %4 = memref.load %arg0[%arg6] : memref<?xf64>
        memref.store %4, %arg2[%arg6] : memref<?xf64>
      }
    }
    return
  }
}

// CHECK:   func @withinif(%arg0: memref<?xf64>, %arg1: i32, %arg2: memref<?xf64>, %arg3: i1) {
// CHECK-NEXT:     %0 = arith.index_cast %arg1 : i32 to index
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     scf.if %arg3 {
// CHECK-NEXT:       affine.for %arg4 = 1 to %0 {
// CHECK-NEXT:         %1 = memref.load %arg0[%arg4] : memref<?xf64>
// CHECK-NEXT:         memref.store %1, %arg2[%arg4] : memref<?xf64>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
