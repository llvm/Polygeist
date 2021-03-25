// RUN: polymer-opt %s -reg2mem -split-input-file | FileCheck %s

// This is a general test case that covers many different aspects for checking.

func @load_store_dep(%A: memref<?xf32>, %B: memref<?x?xf32>) {
  %c0 = constant 0 : index
  %c1 = constant 1 : index

  %someValue = constant 1.23 : f32

  %NI = dim %A, %c0 : memref<?xf32>
  %NJ = dim %B, %c1 : memref<?x?xf32>

  affine.for %i = 0 to %NI {
    %0 = affine.load %A[%i] : memref<?xf32>
    %1 = mulf %0, %0 : f32
    affine.store %someValue, %A[%i] : memref<?xf32>

    affine.for %j = 0 to %NJ {
      %2 = mulf %1, %0 : f32
      %3 = addf %1, %2 : f32
      %4 = subf %3, %someValue : f32
      affine.store %4, %B[%i, %j] : memref<?x?xf32>
    }

    affine.store %1, %A[%i] : memref<?xf32>
  }

  return 
}

// CHECK:       func @load_store_dep(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?x?xf32>) {
// CHECK-NEXT:   %[[MEM0:.*]] = alloca() {scop.scratchpad} : memref<1xf32>
// CHECK-NEXT:   %[[MEM1:.*]] = alloca() {scop.scratchpad} : memref<1xf32>
// CHECK-NEXT:   %[[C0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[C1:.*]] = constant 1 : index
// CHECK-NEXT:   %[[CST:.*]] = constant 1.230000e+00 : f32
// CHECK-NEXT:   %[[VAL2:.*]] = dim %[[ARG0]], %[[C0]] : memref<?xf32>
// CHECK-NEXT:   %[[VAL3:.*]] = dim %[[ARG1]], %[[C1]] : memref<?x?xf32>
// CHECK-NEXT:   affine.for %[[ARG2:.*]] = 0 to %[[VAL2]] {
// CHECK-NEXT:     %[[VAL4:.*]] = affine.load %[[ARG0]][%[[ARG2]]] : memref<?xf32>
// CHECK-NEXT:     affine.store %[[VAL4]], %[[MEM0]][0] : memref<1xf32>
// CHECK-NEXT:     %[[VAL5:.*]] = mulf %[[VAL4]], %[[VAL4]] : f32
// CHECK-NEXT:     affine.store %[[VAL5]], %[[MEM1]][0] : memref<1xf32>
// CHECK-NEXT:     affine.store %[[CST]], %[[ARG0]][%[[ARG2]]] : memref<?xf32>
// CHECK-NEXT:     affine.for %[[ARG3:.*]] = 0 to %[[VAL3]] {
// CHECK-NEXT:       %[[VAL6:.*]] = affine.load %[[MEM0]][0] : memref<1xf32>
// CHECK-NEXT:       %[[VAL7:.*]] = affine.load %[[MEM1]][0] : memref<1xf32>
// CHECK-NEXT:       %[[VAL8:.*]] = mulf %[[VAL7]], %[[VAL6]] : f32
// CHECK-NEXT:       %[[VAL9:.*]] = addf %[[VAL7]], %[[VAL8]] : f32
// CHECK-NEXT:       %[[VAL10:.*]] = subf %[[VAL9]], %[[CST]] : f32
// CHECK-NEXT:       affine.store %[[VAL10]], %[[ARG1]][%[[ARG2]], %[[ARG3]]] : memref<?x?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:     affine.store %[[VAL5]], %[[ARG0]][%[[ARG2]]] : memref<?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }


// -----

// No scratchpad memref will be created for load op that are not used by values to be stored.

// CHECK: func @load_no_use(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>) {
func @load_no_use(%A: memref<?xf32>, %B: memref<?xf32>) {
  // CHECK: %[[C0:.*]] = constant 0 : index
  %c0 = constant 0 : index
  // CHECK-NEXT: %[[VAL0:.*]] = dim %[[ARG0]], %[[C0]] : memref<?xf32>
  %NI = dim %A, %c0 : memref<?xf32>
  // CHECK-NEXT: %[[VAL1:.*]] = affine.load %[[ARG0]][0] : memref<?xf32>
  %0 = affine.load %A[0] : memref<?xf32>
  // CHECK-NEXT: affine.for %[[ARG2:.*]] = 0 to %[[VAL0]] {
  affine.for %i = 0 to %NI {
    %1 = affine.load %A[%i] : memref<?xf32>
    affine.store %1, %B[%i] : memref<?xf32>
  }

  return
}


// -----

// Should not generate scratchpad for values being used in the same block.

func @load_use_in_same_block(%A: memref<?xf32>, %B: memref<?xf32>) {
  %c0 = constant 0 : index
  %NI = dim %A, %c0 : memref<?xf32>
  affine.for %i = 0 to %NI {
    %0 = affine.load %A[%i] : memref<?xf32>
    affine.store %0, %B[%i] : memref<?xf32>
  }

  return
}

// CHECK: func @load_use_in_same_block(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>) {
// CHECK-NEXT: %[[C0:.*]] = constant 0 : index
// CHECK-NEXT: %[[DIM0:.*]] = dim %[[ARG0]], %[[C0]] : memref<?xf32>
// CHECK-NEXT: affine.for %[[I:.*]] = 0 to %[[DIM0]] {
// CHECK-NEXT:   %[[VAL0:.*]] = affine.load %[[ARG0]][%[[I]]] : memref<?xf32>
// CHECK-NEXT:   affine.store %[[VAL0]], %[[ARG1]][%[[I]]] : memref<?xf32>

// -----

// Should generate multiple loads for uses of the same value at different blocks.

func @multi_uses_at_diff_blocks(%A: memref<?xf32>, %B: memref<?x?xf32>, %C: memref<?x?x?xf32>) {
  %c0 = constant 0 : index 
  %c1 = constant 1 : index 
  %c2 = constant 2 : index 

  %NI = dim %C, %c0 : memref<?x?x?xf32>
  %NJ = dim %C, %c1 : memref<?x?x?xf32>
  %NK = dim %C, %c2 : memref<?x?x?xf32>

  affine.for %i = 0 to %NI {
    %0 = affine.load %A[%i] : memref<?xf32>
    affine.for %j = 0 to %NJ {
      affine.store %0, %B[%i, %j] : memref<?x?xf32>
      affine.for %k = 0 to %NK {
        affine.store %0, %C[%i, %j, %k] : memref<?x?x?xf32>
      }
    }
  }

  return
}

// CHECK: func @multi_uses_at_diff_blocks(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?x?x?xf32>) {
// CHECK-NEXT: %[[MEM0:.*]] = alloca() {scop.scratchpad} : memref<1xf32>
// CHECK-NEXT: %[[C0:.*]] = constant 0 : index
// CHECK-NEXT: %[[C1:.*]] = constant 1 : index
// CHECK-NEXT: %[[C2:.*]] = constant 2 : index
// CHECK-NEXT: %[[DIM0:.*]] = dim %[[ARG2]], %[[C0]] : memref<?x?x?xf32>
// CHECK-NEXT: %[[DIM1:.*]] = dim %[[ARG2]], %[[C1]] : memref<?x?x?xf32>
// CHECK-NEXT: %[[DIM2:.*]] = dim %[[ARG2]], %[[C2]] : memref<?x?x?xf32>
// CHECK-NEXT: affine.for %[[I:.*]] = 0 to %[[DIM0]] {
// CHECK-NEXT:   %[[VAL0:.*]] = affine.load %[[ARG0]][%[[I]]] : memref<?xf32>
// CHECK-NEXT:   affine.store %[[VAL0]], %[[MEM0]][0] : memref<1xf32>
// CHECK-NEXT:   affine.for %[[J:.*]] = 0 to %[[DIM1]] {
// CHECK-NEXT:     %[[VAL1:.*]] = affine.load %[[MEM0]][0] : memref<1xf32>
// CHECK-NEXT:     affine.store %[[VAL1]], %[[ARG1]][%[[I]], %[[J]]] : memref<?x?xf32>
// CHECK-NEXT:     affine.for %[[K:.*]] = 0 to %[[DIM2]] {
// CHECK-NEXT:       %[[VAL2:.*]] = affine.load %[[MEM0]][0] : memref<1xf32>
// CHECK-NEXT:       affine.store %[[VAL2]], %[[ARG2]][%[[I]], %[[J]], %[[K]]] : memref<?x?x?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }

// ----- 

// Should only generate one load for multiple uses of the same value in the same block.

func @multi_uses_at_same_block(%A: memref<?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  %c0 = constant 0 : index 
  %c1 = constant 1 : index 

  %NI = dim %C, %c0 : memref<?x?xf32>
  %NJ = dim %C, %c1 : memref<?x?xf32>

  affine.for %i = 0 to %NI {
    %0 = affine.load %A[%i] : memref<?xf32>
    affine.for %j = 0 to %NJ {
      affine.store %0, %B[%i, %j] : memref<?x?xf32>
      affine.store %0, %C[%i, %j] : memref<?x?xf32>
    }
  }

  return
}

// CHECK: func @multi_uses_at_same_block(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?x?xf32>, %[[ARG2:.*]]: memref<?x?xf32>) {
// CHECK-NEXT: %[[MEM0:.*]] = alloca() {scop.scratchpad} : memref<1xf32>
// CHECK-NEXT: %[[C0:.*]] = constant 0 : index
// CHECK-NEXT: %[[C1:.*]] = constant 1 : index
// CHECK-NEXT: %[[DIM0:.*]] = dim %[[ARG2]], %[[C0]] : memref<?x?xf32>
// CHECK-NEXT: %[[DIM1:.*]] = dim %[[ARG2]], %[[C1]] : memref<?x?xf32>
// CHECK-NEXT: affine.for %[[I:.*]] = 0 to %[[DIM0]] {
// CHECK-NEXT:   %[[VAL0:.*]] = affine.load %[[ARG0]][%[[I]]] : memref<?xf32>
// CHECK-NEXT:   affine.store %[[VAL0]], %[[MEM0]][0] : memref<1xf32>
// CHECK-NEXT:   affine.for %[[J:.*]] = 0 to %[[DIM1]] {
// CHECK-NEXT:     %[[VAL1:.*]] = affine.load %[[MEM0]][0] : memref<1xf32>
// CHECK-NEXT:     affine.store %[[VAL1]], %[[ARG1]][%[[I]], %[[J]]] : memref<?x?xf32>
// CHECK-NEXT:     affine.store %[[VAL1]], %[[ARG2]][%[[I]], %[[J]]] : memref<?x?xf32>
// CHECK-NEXT:   }
// CHECK-NEXT: }


// ----- 

// Should replace uses in conditionals.

func @use_in_conds(%A: memref<?xf32>, %B: memref<?xf32>, %C: memref<?xf32>) {
  %c0 = constant 0 : index 
  %N = dim %A, %c0 : memref<?xf32>
  %M = dim %B, %c0 : memref<?xf32>

  affine.for %i = 0 to %N {
    %0 = affine.load %A[%i] : memref<?xf32>
    affine.if affine_set<(d0)[s0]: (s0 - d0 - 1 >= 0)>(%i)[%M] {
      affine.store %0, %B[%i] : memref<?xf32>
    } else {
      affine.store %0, %C[%i] : memref<?xf32>
    }
  }

  return
}

// CHECK: func @use_in_conds(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>, %[[ARG2:.*]]: memref<?xf32>) {
// CHECK-NEXT:   %[[MEM0:.*]] = alloca() {scop.scratchpad} : memref<1xf32>
// CHECK-NEXT:   %[[C0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[VAL0:.*]] = dim %[[ARG0]], %[[C0]] : memref<?xf32>
// CHECK-NEXT:   %[[VAL1:.*]] = dim %[[ARG1]], %[[C0]] : memref<?xf32>
// CHECK-NEXT:   affine.for %[[ARG3:.*]] = 0 to %[[VAL0]] {
// CHECK-NEXT:     %[[VAL2:.*]] = affine.load %[[ARG0]][%[[ARG3]]] : memref<?xf32>
// CHECK-NEXT:     affine.store %[[VAL2]], %[[MEM0]][0] : memref<1xf32>
// CHECK-NEXT:     affine.if #[[SET0:.*]](%[[ARG3]])[%[[VAL1]]] {
// CHECK-NEXT:       %[[VAL3:.*]] = affine.load %[[MEM0]][0] : memref<1xf32>
// CHECK-NEXT:       affine.store %[[VAL3]], %[[ARG1]][%[[ARG3]]] : memref<?xf32>
// CHECK-NEXT:     } 
// CHECK-NEXT:     affine.if #[[SET1:.*]](%[[ARG3]])[%[[VAL1]]] {
// CHECK-NEXT:       %[[VAL3:.*]] = affine.load %[[MEM0]][0] : memref<1xf32>
// CHECK-NEXT:       affine.store %[[VAL3]], %[[ARG2]][%[[ARG3]]] : memref<?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }

// -----

// Dealing with function call.

func @f(%x: f32, %y: f32) -> (f32) {
  %0 = addf %x, %y : f32
  return %0 : f32
}

func @use_by_arith_call(%A: memref<?xf32>, %B: memref<?x?xf32>) {
  %c0 = constant 0 : index 
  %c1 = constant 1 : index 

  %N = dim %B, %c0 : memref<?x?xf32>
  %M = dim %B, %c1 : memref<?x?xf32>

  affine.for %i = 0 to %N {
    %0 = affine.load %A[%i] : memref<?xf32>
    affine.for %j = 1 to %M {
      %k = affine.apply affine_map<(d0)[] -> (d0 - 1)>(%j)
      %1 = affine.load %B[%i, %k] : memref<?x?xf32>
      %2 = call @f(%0, %1) : (f32, f32) -> (f32)
      affine.store %2, %B[%i, %j] : memref<?x?xf32>
    }
  }

  return
}

// CHECK: func @use_by_arith_call(%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?x?xf32>) {
// CHECK-NEXT:   %[[MEM0:.*]] = alloca() {scop.scratchpad} : memref<1xf32>
// CHECK-NEXT:   %[[CST0:.*]] = constant 0 : index
// CHECK-NEXT:   %[[CST1:.*]] = constant 1 : index
// CHECK-NEXT:   %[[DIM0:.*]] = dim %[[ARG1]], %[[CST0]] : memref<?x?xf32>
// CHECK-NEXT:   %[[DIM1:.*]] = dim %[[ARG1]], %[[CST1]] : memref<?x?xf32>
// CHECK-NEXT:   affine.for %[[I:.*]] = 0 to %[[DIM0]] {
// CHECK-NEXT:     %[[VAL0:.*]] = affine.load %[[ARG0]][%[[I]]] : memref<?xf32>
// CHECK-NEXT:     affine.store %[[VAL0]], %[[MEM0]][0] : memref<1xf32>
// CHECK-NEXT:     affine.for %[[J:.*]] = 1 to %[[DIM1]] {
// CHECK-NEXT:       %[[VAL1:.*]] = affine.load %[[MEM0]][0] : memref<1xf32>
// CHECK-NEXT:       %[[K:.*]] = affine.apply #[[MAP2:.*]](%[[J]])
// CHECK-NEXT:       %[[VAL2:.*]] = affine.load %[[ARG1]][%[[I]], %[[K]]] : memref<?x?xf32>
// CHECK-NEXT:       %[[VAL3:.*]] = call @f(%[[VAL1]], %[[VAL2]]) : (f32, f32) -> f32
// CHECK-NEXT:       affine.store %[[VAL3]], %[[ARG1]][%[[I]], %[[J]]] : memref<?x?xf32>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return
// CHECK-NEXT: }


// -----

// TODO: make this test case work.

func @g(%0: f32, %i: index, %mem: memref<?xf32>) {
  affine.store %0, %mem[%i] : memref<?xf32>
  return 
}

func @use_by_store_call(%A: memref<?xf32>) {
  %c0 = constant 0 : index 
  %N = dim %A, %c0 : memref<?xf32>

  %cst = constant 1.23 : f32

  affine.for %i = 0 to %N {
    call @g(%cst, %i, %A) : (f32, index, memref<?xf32>) -> ()
  }

  return 
}

// -----

// Affine apply defines the loop bounds and addresses accessed.

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

// CHECK: func @use_affine_apply(%[[ARG0:.*]]: memref<?x?xf32>) {
// CHECK:   %[[C0:.*]] = constant 0 : index
// CHECK:   %[[C1:.*]] = constant 1 : index
// CHECK:   %[[DIM0:.*]] = dim %[[ARG0]], %[[C0]] : memref<?x?xf32>
// CHECK:   %[[DIM1:.*]] = dim %[[ARG0]], %[[C1]] : memref<?x?xf32>
// CHECK:   affine.for %[[ARG1:.*]] = 0 to %[[DIM0]] {
// CHECK:     %[[VAL0:.*]] = affine.apply #[[MAP0:.*]](%[[ARG1]])[%[[DIM0]]]
// CHECK:     affine.for %[[ARG2:.*]] = #[[MAP2:.*]](%[[VAL0]]) to %[[DIM1]] {
// CHECK:       %[[VAL1:.*]] = affine.load %[[ARG0]][%[[VAL0]], %[[ARG2]]] : memref<?x?xf32>
// CHECK:       %[[VAL2:.*]] = addf %[[VAL1]], %[[VAL1]] : f32
// CHECK:       affine.store %[[VAL2]], %[[ARG0]][%[[VAL0]], %[[ARG2]]] : memref<?x?xf32>
// CHECK:     }
// CHECK:   }
// CHECK:   return
// CHECK: }

