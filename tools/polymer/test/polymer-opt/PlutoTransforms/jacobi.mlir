// RUN: polymer-opt %s -pluto-opt | FileCheck %s

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

// CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0) -> (d0 * 2)>                                                                                                                                                                                                                                                                           
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0) -> (7, d0 * 2 + 6)>                                                                                                                                                                                                                                                                    
// CHECK-DAG: #[[MAP2:.*]] = affine_map<(d0, d1) -> (d0 * 32, d1 * 16 - 59)>                                                                                                                                                                                                                                                        
// CHECK-DAG: #[[MAP3:.*]] = affine_map<(d0, d1) -> (40, d0 * 32 + 32, d1 * 16 + 15)>
// CHECK-DAG: #[[MAP4:.*]] = affine_map<(d0, d1) -> (d0 * 32, d1 * 2 + 2)>
// CHECK-DAG: #[[MAP5:.*]] = affine_map<(d0, d1) -> (d0 * 32 + 32, d1 * 2 + 119)>
// CHECK-DAG: #[[MAP6:.*]] = affine_map<(d0, d1) -> (d0 * -2 + d1)>
// CHECK-DAG: #[[MAP7:.*]] = affine_map<(d0, d1) -> (d0 * -2 + d1 - 1)>
// CHECK-DAG: #[[SET0:.*]] = affine_set<(d0, d1) : (d1 floordiv 16 - d0 >= 0)>
// CHECK-DAG: #[[SET1:.*]] = affine_set<(d0, d1) : (d0 - (d1 + 44) ceildiv 16 >= 0)>
// CHECK-DAG: #[[SET2:.*]] = affine_set<(d0, d1) : (d0 == 0, -d1 + 1 >= 0)>

// CHECK-LABEL: func @jacobi
// CHECK: affine.for %[[I:.*]] = 0 to 2
// CHECK: affine.for %[[J:.*]] = #[[MAP0]](%[[I]]) to min #[[MAP1]](%[[I]])
// CHECK: affine.for %[[K:.*]] = max #[[MAP2]](%[[I]], %[[J]]) to min #[[MAP3]](%[[I]], %[[J]])
// CHECK: affine.for %[[L:.*]] = max #[[MAP4]](%[[J]], %[[K]]) to min #[[MAP5]](%[[J]], %[[K]])
