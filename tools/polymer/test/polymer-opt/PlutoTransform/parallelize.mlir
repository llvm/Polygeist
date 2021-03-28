// RUN: polymer-opt %s -reg2mem -extract-scop-stmt -pluto-opt="parallelize" | FileCheck %s

func @single_loop(%A: memref<?xf32>, %N: index) {
  affine.for %i = 0 to %N {
    %0 = affine.load %A[%i] : memref<?xf32>
    %1 = addf %0, %0 : f32
    affine.store %1, %A[%i] : memref<?xf32>
  }
  return
}

// CHECK:      func private @single_loop_opt(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index) {
// CHECK-NEXT:   affine.if #[[SET:.*]]()[%[[ARG1]]] {
// CHECK-NEXT:     affine.parallel (%[[ARG2:.*]]) = (0) to ((symbol(%[[ARG1]]) - 1) floordiv 32 + 1) {
// CHECK-NEXT:       affine.for %[[ARG3:.*]] = #[[MAP0:.*]](%[[ARG2]]) to min #[[MAP1:.*]](%[[ARG2]])[%[[ARG1]]] {
// CHECK-NEXT:         call @[[S0:.*]](%[[ARG0]], %[[ARG3]]) : (memref<?xf32>, index) -> ()
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

// Should trigger wavefront parallelism in Pluto.
// func @not_parallelizable_bounds(%A: memref<?x?xf32>, %N: index) {
//   affine.for %i = 0 to %N {
//     affine.for %j = 0 to %N {
//       %0 = affine.load %A[%i - 1, %j] : memref<?x?xf32>
//       %1 = affine.load %A[%i, %j - 1] : memref<?x?xf32>
//       %2 = addf %0, %1 : f32
//       affine.store %2, %A[%i, %j] : memref<?x?xf32>
//     }
//   }
//   return
// }