// RUN: polymer-opt %s -extract-scop-stmt | FileCheck %s

func @foo(%A: memref<1xf32>) {
  %c0 = arith.constant 0 : index
  %0 = affine.load %A[0] : memref<1xf32>
  affine.store %0, %A[0] : memref<1xf32>
  affine.store %0, %A[0] : memref<1xf32>
  return
}

// CHECK: func private @S0(%[[A:.*]]: memref<1xf32>, %[[SCRATCHPAD:.*]]: memref<1xf32>)
// CHECK-NEXT: %[[VAL0:.*]] = affine.load %[[A]][0]
// CHECK-NEXT: affine.store %[[VAL0]], %[[SCRATCHPAD]][0]
// CHECK-NEXT: affine.store %[[VAL0]], %[[A]][0]

// CHECK: func private @S1(%[[A:.*]]: memref<1xf32>, %[[SCRATCHPAD:.*]]: memref<1xf32>)
// CHECK-NEXT: %[[VAL0:.*]] = affine.load %[[SCRATCHPAD]][0]
// CHECK-NEXT: affine.store %[[VAL0]], %[[A]][0]

// CHECK: func @foo(%[[A:.*]]: memref<1xf32>)
// CHECK-NEXT: %[[SCRATCHPAD:.*]] = memref.alloca() : memref<1xf32>
// CHECK-NEXT: call @S0(%[[A]], %[[SCRATCHPAD]])
// CHECK-NEXT: call @S1(%[[A]], %[[SCRATCHPAD]])
