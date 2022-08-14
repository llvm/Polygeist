// RUN: polygeist-opt --parallel-licm --split-input-file %s | FileCheck %s

module {
func.func @dist(%arg0: memref<?x1xmemref<?xf32>>) -> f32 {
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %cst) -> (f32) {
    %1 = arith.index_cast %arg1 : index to i32
    %2 = affine.load %arg0[0, 0] : memref<?x1xmemref<?xf32>>
    %3 = memref.load %2[%arg1] : memref<?xf32>
    %4 = arith.addf %arg2, %3 : f32
    scf.yield %4 : f32
  }
  return %0 : f32
}
}

// CHECK:   func.func @dist(%arg0: memref<?x1xmemref<?xf32>>) -> f32 
// CHECK-NEXT:     %c10 = arith.constant 10 : index
// CHECK-NEXT:     %c0 = arith.constant 0 : index
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %0 = arith.addi %c0, %c1 : index
// CHECK-NEXT:     %1 = arith.cmpi sle, %0, %c10 : index
// CHECK-NEXT:     %2 = scf.if %1 -> (f32) {
// CHECK-NEXT:       %3 = affine.load %arg0[0, 0] : memref<?x1xmemref<?xf32>>
// CHECK-NEXT:       %4 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %cst) -> (f32) {
// CHECK-NEXT:         %5 = arith.index_cast %arg1 : index to i32
// CHECK-NEXT:         %6 = memref.load %3[%arg1] : memref<?xf32>
// CHECK-NEXT:         %7 = arith.addf %arg2, %6 : f32
// CHECK-NEXT:         scf.yield %7 : f32
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %4 : f32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %cst : f32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %2 : f32
// CHECK-NEXT:   }
