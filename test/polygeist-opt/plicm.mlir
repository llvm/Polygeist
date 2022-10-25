// RUN: polygeist-opt --parallel-licm --split-input-file %s | FileCheck %s

module {
func.func @dist(%arg0: memref<?x1xmemref<?xf32>>) -> f32 {
  %c10 = arith.constant 10 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = scf.for %arg1 = %c0 to %c10 step %c1 iter_args(%arg2 = %cst) -> (f32) {
    %2 = affine.load %arg0[0, 0] : memref<?x1xmemref<?xf32>>
    %3 = memref.load %2[%arg1] : memref<?xf32>
    %4 = arith.addf %arg2, %3 : f32
    scf.yield %4 : f32
  }
  return %0 : f32
}
}

// CHECK:   func.func @dist(%[[arg0:.+]]: memref<?x1xmemref<?xf32>>) -> f32
// CHECK-NEXT:     %[[c10:.+]] = arith.constant 10 : index
// CHECK-NEXT:     %[[c0:.+]] = arith.constant 0 : index
// CHECK-NEXT:     %[[c1:.+]] = arith.constant 1 : index
// CHECK-NEXT:     %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %[[i1:.+]] = arith.cmpi slt, %[[c0]], %[[c10]] : index
// CHECK-NEXT:     %[[i2:.+]] = scf.if %[[i1]] -> (f32) {
// CHECK-NEXT:       %[[i3:.+]] = affine.load %[[arg0]][0, 0] : memref<?x1xmemref<?xf32>>
// CHECK-NEXT:       %[[i4:.+]] = scf.for %[[arg1:.+]] = %[[c0]] to %[[c10]] step %[[c1]] iter_args(%[[arg2:.+]] = %[[cst]]) -> (f32) {
// CHECK-NEXT:         %[[i6:.+]] = memref.load %[[i3]][%[[arg1]]] : memref<?xf32>
// CHECK-NEXT:         %[[i7:.+]] = arith.addf %[[arg2]], %[[i6]] : f32
// CHECK-NEXT:         scf.yield %[[i7]] : f32
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[i4]] : f32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %[[cst]] : f32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[i2]] : f32
// CHECK-NEXT:   }
