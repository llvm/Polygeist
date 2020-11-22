// RUN: polymer-opt %s -extract-scop-stmt -split-input-file | FileCheck %s

func @load_store(%A: memref<?xf32>, %B: memref<?xf32>) {
  %c0 = constant 0 : index
  %N = dim %A, %c0 : memref<?xf32> 

  affine.for %i = 0 to %N {
    %0 = affine.load %A[%i] : memref<?xf32>
    %1 = mulf %0, %0 : f32
    affine.store %1, %B[%i] : memref<?xf32>
  }

  return
}

// CHECK: func @load_store(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>) {
// CHECK-NEXT:   %[[C0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[DIM0:.*]] = dim %[[ARG0]], %[[C0]] : memref<?xf32>
// CHECK-NEXT:   affine.for %[[ARG2:.*]] = 0 to %[[DIM0]] {
// CHECK-NEXT:     call @S0(%[[ARG1]], %[[ARG2]], %[[ARG0]]) : (memref<?xf32>, index, memref<?xf32>) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func @S0(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: memref<?xf32>) attributes {scop.stmt} {
// CHECK-NEXT:   %[[VAL0:.*]] = affine.load %[[ARG2]][%[[ARG1]]] : memref<?xf32>
// CHECK-NEXT:   %[[VAL1:.*]] = mulf %[[VAL0]], %[[VAL0]] : f32
// CHECK-NEXT:   affine.store %[[VAL1]], %[[ARG0]][%[[ARG1]]] : memref<?xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

func @load_multi_stores(%A: memref<?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %N = dim %B, %c0 : memref<?x?xf32> 
  %M = dim %B, %c1 : memref<?x?xf32> 

  affine.for %i = 0 to %N {
    affine.for %j = 0 to %M {
      %0 = affine.load %A[%i] : memref<?xf32>
      %1 = mulf %0, %0 : f32
      affine.store %1, %B[%i, %j] : memref<?x?xf32>
      %2 = addf %0, %1 : f32
      affine.store %2, %C[%i, %j] : memref<?x?xf32>
    }
  }

  return
}

// CHECK: func @load_multi_stores(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?x?xf32>) {
// CHECK-NEXT:   %[[CST0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[CST1:.*]] = constant 1 : index
// CHECK-NEXT:   %[[DIM0:.*]] = dim %[[ARG1]], %[[CST0]] : memref<?x?xf32>
// CHECK-NEXT:   %[[DIM1:.*]] = dim %[[ARG1]], %[[CST1]] : memref<?x?xf32>
// CHECK-NEXT:   affine.for %[[I:.*]] = 0 to %[[DIM0]] {
// CHECK-NEXT:     affine.for %[[J:.*]] = 0 to %[[DIM1]] {
// CHECK-NEXT:       call @S0(%[[ARG1]], %[[I]], %[[J]], %[[ARG0]]) : (memref<?x?xf32>, index, index, memref<?xf32>) -> ()
// CHECK-NEXT:       call @S1(%[[ARG2]], %[[I]], %[[J]], %[[ARG0]]) : (memref<?x?xf32>, index, index, memref<?xf32>) -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func @S0(%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: memref<?xf32>) attributes {scop.stmt} {
// CHECK-NEXT:   %[[VAL0:.*]] = affine.load %[[ARG3]][%[[ARG1]]] : memref<?xf32>
// CHECK-NEXT:   %[[VAL1:.*]] = mulf %[[VAL0]], %[[VAL0]] : f32
// CHECK-NEXT:   affine.store %[[VAL1]], %[[ARG0]][%[[ARG1]], %[[ARG2]]] : memref<?x?xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// CHECK: func @S1(%[[ARG0:.*]]: memref<?x?xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: memref<?xf32>) attributes {scop.stmt} {
// CHECK-NEXT:   %[[VAL0:.*]] = affine.load %[[ARG3]][%[[ARG1]]] : memref<?xf32>
// CHECK-NEXT:   %[[VAL1:.*]] = mulf %[[VAL0]], %[[VAL0]] : f32
// CHECK-NEXT:   %[[VAL2:.*]] = addf %[[VAL0]], %[[VAL1]] : f32
// CHECK-NEXT:   affine.store %[[VAL2]], %[[ARG0]][%[[ARG1]], %[[ARG2]]] : memref<?x?xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

// multiple functions

func @f1(%A: memref<?xf32>, %B: memref<?xf32>) {
  %c0 = constant 0 : index
  %N = dim %A, %c0 : memref<?xf32> 

  affine.for %i = 0 to %N {
    %0 = affine.load %A[%i] : memref<?xf32>
    %1 = mulf %0, %0 : f32
    affine.store %1, %B[%i] : memref<?xf32>
  }

  return
}

func @f2(%A: memref<?xf32>, %B: memref<?xf32>) {
  %c0 = constant 0 : index
  %N = dim %A, %c0 : memref<?xf32> 

  affine.for %i = 0 to %N {
    %0 = affine.load %A[%i] : memref<?xf32>
    %1 = addf %0, %0 : f32
    affine.store %1, %B[%i] : memref<?xf32>
  }

  return
}

// CHECK: func @S0
// CHECK: func @S1

// -----

// Alloc and alloca ops in the code.

func @alloc_and_alloca() {
  %A = alloc() : memref<32xf32>
  %B = alloca() : memref<32xf32>

  affine.for %i = 0 to 32 {
    %0 = affine.load %A[%i] : memref<32xf32>
    %1 = mulf %0, %0 : f32
    affine.store %1, %B[%i] : memref<32xf32>
  }

  return
}

// CHECK: func @alloc_and_alloca() {
// CHECK-NEXT:   %[[VAL0]] = alloc() : memref<32xf32>
// CHECK-NEXT:   %[[VAL1]] = alloca() : memref<32xf32>
// CHECK-NEXT:   affine.for %[[ARG0:.*]] = 0 to 32 {
// CHECK-NEXT:     call @S0(%[[ARG0]], %[[VAL1]], %[[VAL0]]) : (index, memref<32xf32>, memref<32xf32>) -> ()
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }
// CHECK: func @S0(%[[ARG0:.*]]: index, %[[ARG1:.*]]: memref<32xf32>, %[[ARG2:.*]]: memref<32xf32>) attributes {scop.stmt} {
// CHECK-NEXT:   %[[VAL0:.*]] = affine.load %[[ARG2]][%[[ARG0]]] : memref<32xf32>
// CHECK-NEXT:   %[[VAL1:.*]] = mulf %[[VAL0]], %[[VAL0]] : f32
// CHECK-NEXT:   affine.store %[[VAL1]], %[[ARG1]][%[[ARG0]]] : memref<32xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

// This pass should not crash if there is no write op in the function.

func @no_write() {
  return
}

// CHECK: func @no_write() {
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

// Storing constants.

func @write_const(%A: memref<?xf32>) {
  %i = constant 0 : index
  %j = constant 1 : index
  %cst = constant 3.217 : f32
  affine.store %cst, %A[%i] : memref<?xf32>
  affine.store %cst, %A[%j] : memref<?xf32>
  return
}

// CHECK: func @write_const(%[[ARG0:.*]]: memref<?xf32>) {
// CHECK-NEXT:   call @S0(%[[ARG0]]) : (memref<?xf32>) -> ()
// CHECK-NEXT:   call @S1(%[[ARG0]]) : (memref<?xf32>) -> ()
// CHECK-NEXT:   return
// CHECK-NEXT: }
// CHECK: func @S0(%[[ARG0:.*]]: memref<?xf32>) attributes {scop.stmt} {
// CHECK-NEXT:   %[[CST:.*]] = constant 3.217000e+00 : f32
// CHECK-NEXT:   %[[C0:.*]] = constant 0 : index
// CHECK-NEXT:   affine.store %[[CST]], %[[ARG0]][%[[C0]]] : memref<?xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }
// CHECK: func @S1(%[[ARG0:.*]]: memref<?xf32>) attributes {scop.stmt} {
// CHECK-NEXT:   %[[CST:.*]] = constant 3.217000e+00 : f32
// CHECK-NEXT:   %[[C0:.*]] = constant 1 : index
// CHECK-NEXT:   affine.store %[[CST]], %[[ARG0]][%[[C0]]] : memref<?xf32>
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

// AffineApplyOp result used in both loop bounds and load/store addresses
// should be treated differently.

#map0 = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
#map1 = affine_map<(d0) -> (d0)>

func @use_affine_apply(%A: memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index
  %NI = dim %A, %c0 : memref<?x?xf32>
  %NJ = dim %A, %c1 : memref<?x?xf32>

  affine.for %i = 0 to %NI {
    %0 = affine.apply #map0(%i)[%NI]
    affine.for %j = #map1(%0) to %NJ {
      %1 = affine.load %A[%0, %j] : memref<?x?xf32>
      %2 = addf %1, %1 : f32 
      affine.store %2, %A[%0, %j] : memref<?x?xf32>
    }
  }

  return
} 
