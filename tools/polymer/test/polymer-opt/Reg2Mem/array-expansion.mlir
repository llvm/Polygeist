// RUN: polymer-opt %s -demote-loop-reduction -reg2mem -array-expansion -split-input-file | FileCheck %s

func @single_reduction_loop(%A: memref<?xf32>) -> (f32) {
  %c0 = constant 0 : index
  %N = dim %A, %c0 : memref<?xf32>

  %sum_0 = constant 0.0 : f32
  %prod_0 = constant 1.0 : f32

  %sum, %prod = affine.for %i = 0 to %N iter_args(%sum_iter=%sum_0, %prod_iter=%prod_0) -> (f32, f32) {
    %0 = affine.load %A[%i] : memref<?xf32>
    %1 = addf %sum_iter, %0 : f32
    %2 = mulf %prod_iter, %0 : f32
    affine.yield %1, %2 : f32, f32
  }

  %ans = addf %sum, %prod : f32

  return %ans : f32
}

// CHECK: #[[MAP:.*]] = affine_map<()[s0] -> (s0)>
// CHECK-LABEL: func @single_reduction_loop
// CHECK: %[[VAL0:.*]] = dim %{{.*}}, %{{.*}} : memref<?xf32>
// CHECK: %[[VAL1:.*]] = affine.max #map()[%[[VAL0]]]
// CHECK: %[[VAL2:.*]] = alloca(%[[VAL1]]) {scop.scratchpad} : memref<?xf32>
// CHECK: %[[VAL3:.*]] = affine.max #map()[%[[VAL0]]]
// CHECK: %[[VAL4:.*]] = alloca(%[[VAL3]]) {scop.scratchpad} : memref<?xf32>
// CHECK: affine.store %{{.*}}, %[[VAL4]][0] : memref<?xf32>
// CHECK: affine.store %{{.*}}, %[[VAL2]][0] : memref<?xf32>
// CHECK: affine.for %[[ARG1:.*]] = 0 to %[[VAL0]] {
// CHECK:   %[[VAL8:.*]] = affine.load %[[VAL2]][%[[ARG1]]] : memref<?xf32>
// CHECK:   %[[VAL9:.*]] = affine.load %[[VAL4]][%[[ARG1]]] : memref<?xf32>
// CHECK:   affine.store %{{.*}}, %[[VAL4]][(%[[ARG1]] + 1) mod symbol(%[[VAL0]])] : memref<?xf32>
// CHECK:   affine.store %{{.*}}, %[[VAL2]][(%[[ARG1]] + 1) mod symbol(%[[VAL0]])] : memref<?xf32>
// CHECK: }
// CHECK: %{{.*}} = affine.load %[[VAL2]][0] : memref<?xf32>
// CHECK: %{{.*}} = affine.load %[[VAL4]][0] : memref<?xf32>


// func @nested(%A: memref<?xf32>, %B: memref<?x?xf32>) {
//   %c0 = constant 0 : index
//   %c1 = constant 0 : index
//   %N = dim %A, %c0 : memref<?xf32>
//   %M = dim %B, %c0 : memref<?x?xf32>
//   %K = dim %B, %c1 : memref<?x?xf32>

//   affine.for %i = 0 to %N {
//     %0 = affine.load %A[%i] : memref<?xf32>
//     affine.for %j = 0 to %M {
//       affine.for %k = 0 to %K {
//         %1 = affine.load %B[%j,%k] : memref<?x?xf32>
//         %2 = addf %0, %1 : f32
//         affine.store %2, %B[%j,%k] : memref<?x?xf32>
//       }
//     }
//     affine.for %j = 0 to %M {
//       affine.for %k = 0 to %K {
//         %1 = affine.load %B[%j,%k] : memref<?x?xf32>
//         %2 = mulf %0, %1 : f32
//         affine.store %2, %B[%j,%k] : memref<?x?xf32>
//       }
//     }
//   }

//   return
// }
