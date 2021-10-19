// RUN: polymer-opt %s -pluto-opt="diamond-tiling" | FileCheck %s

#map = affine_map<()[s0] -> (s0 - 1)>

func private @S0(%arg0: memref<120xf32>, %arg1: index, %arg2: memref<120xf32>) attributes {scop.stmt} {
  %cst = arith.constant 3.333330e-01 : f32
  %0 = affine.load %arg2[symbol(%arg1) + 1] : memref<120xf32>
  %1 = affine.load %arg2[symbol(%arg1) - 1] : memref<120xf32>
  %2 = affine.load %arg2[symbol(%arg1)] : memref<120xf32>
  %3 = arith.addf %1, %2 : f32
  %4 = arith.addf %0, %3 : f32
  %5 = arith.mulf %cst, %4 : f32
  affine.store %5, %arg0[symbol(%arg1)] : memref<120xf32>
  return
}

func private @S1(%arg0: memref<120xf32>, %arg1: index, %arg2: memref<120xf32>) attributes {scop.stmt} {
  %0 = affine.load %arg2[symbol(%arg1)] : memref<120xf32>
  affine.store %0, %arg0[symbol(%arg1)] : memref<120xf32>
  return
}

func @jacobi(%A: memref<120xf32>, %B: memref<120xf32>) {
  %cst = arith.constant 0.333333 : f32
  affine.for %i = 0 to 40 {
    affine.for %j = 1 to 119 {
      call @S0(%B, %j, %A): (memref<120xf32>, index, memref<120xf32>) -> ()
    }
    affine.for %j = 1 to 119 {
      call @S1(%A, %j, %B): (memref<120xf32>, index, memref<120xf32>) -> ()
    }
  }
  return
}

// CHECK-LABEL: func @jacobi
// CHECK: affine.for %[[I:.*]] = -4 to 3 
// CHECK:   affine.for %[[J:.*]] = max #{{.*}}(%[[I]]) to min #{{.*}}(%[[I]]) 
// CHECK:     affine.if #{{.*}}(%[[I]], %[[J]])
// CHECK:     affine.for %[[K:.*]] = max #{{.*}}(%[[I]], %[[J]]) to min #{{.*}}(%[[I]], %[[J]]) 
// CHECK:       affine.if #{{.*}}(%[[I]], %[[K]], %[[J]]) 
// CHECK:       affine.for %[[L:.*]] = max #{{.*}}(%[[J]], %[[K]], %[[I]]) to #{{.*}}(%[[I]], %[[K]]) 
// CHECK:       affine.for %[[L:.*]] = max #{{.*}}(%[[J]], %[[K]], %[[I]]) to min #{{.*}}(%[[I]], %[[K]], %[[J]]) 
// CHECK:       affine.for %[[L:.*]] = #{{.*}}(%[[I]], %[[K]]) to min #{{.*}}(%[[J]], %[[K]], %[[I]]) 
// CHECK:       affine.if #{{.*}}(%[[I]], %[[K]], %[[J]]) 
// CHECK:     affine.if #{{.*}}(%[[I]], %[[J]])
