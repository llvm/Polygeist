// RUN: polymer-opt %s -extract-scop-stmt | FileCheck %s


/// The data-flow of the following program. Due to the existence of the dominating-store edge,
/// we should replace the load edge on the left by a scratchpad.
///

///                   +---load A[i] ---+
///                   |     |          |
///                   |     |          |
///                   |     v       dominating
///                   |   mulf         |
///   replace --->    |     |          |
///                   |     |          v
///                   |     +----->store A[i]
///                   |     |          +
///                   |     |       dominating
///                   |     v          |
///                   +-->addf <-------+
///                         |
///                         +----->store A[i]

func @foo(%A: memref<10xf32>) {
  affine.for %i = 0 to 10 {
    %0 = affine.load %A[%i] : memref<10xf32>
    %1 = arith.mulf %0, %0 : f32
    affine.store %1, %A[%i] : memref<10xf32>
    // Should replace %0 by a load from a scratchpad.
    %2 = arith.addf %1, %0 : f32
    affine.store %2, %A[%i] : memref<10xf32>
  }
  return
}

// CHECK: func private @S0(%[[ARG0:.*]]: memref<10xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: memref<1xf32>) attributes {scop.stmt} 
// CHECK:   %[[VAL0:.*]] = affine.load %[[ARG0]][symbol(%[[ARG1]])]
// CHECK:   affine.store %[[VAL0]], %[[ARG2]][0]
// CHECK:   %[[VAL1:.*]] = arith.mulf %[[VAL0]], %[[VAL0]]
// CHECK:   affine.store %[[VAL1]], %[[ARG0]][symbol(%[[ARG1]])]

// CHECK: func private @S1(%[[ARG0:.*]]: memref<10xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: memref<1xf32>) attributes {scop.stmt} 
// CHECK:   %[[VAL0:.*]] = affine.load %[[ARG0]][symbol(%[[ARG1]])]
// CHECK:   %[[VAL1:.*]] = affine.load %[[ARG2]][0]
// CHECK:   %[[VAL2:.*]] = arith.addf %[[VAL0]], %[[VAL1]]
// CHECK:   affine.store %[[VAL2]], %[[ARG0]][symbol(%[[ARG1]])]

// CHECK: func @foo(%[[ARG0:.*]]: memref<10xf32>) 
// CHECK:   %[[VAL0:.*]] = memref.alloca()
// CHECK:   affine.for %[[ARG1:.*]] = 0 to 10 
// CHECK:     call @S0(%[[ARG0]], %[[ARG1]], %[[VAL0]])
// CHECK:     call @S1(%[[ARG0]], %[[ARG1]], %[[VAL0]])
