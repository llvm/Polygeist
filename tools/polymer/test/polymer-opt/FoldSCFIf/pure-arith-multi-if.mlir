// RUN: polymer-opt %s -fold-scf-if | FileCheck %s

func @foo(%a: f32, %b: f32, %c: i1, %d: i1) -> f32 {
  %0 = scf.if %c -> f32 {
    %0 = scf.if %d -> f32 {
      scf.yield %a : f32
    } else {
      scf.yield %b : f32
    }
    %1 = arith.addf %0, %b : f32
    scf.yield %1 : f32
  } else {
    %1 = arith.mulf %a, %b : f32
    scf.yield %1 : f32
  }
  return %0 : f32
}

// CHECK: func @foo(%[[a:.*]]: f32, %[[b:.*]]: f32, %[[c:.*]]: i1, %[[d:.*]]: i1) -> f32 
// CHECK-NEXT:   %[[v0:.*]] = arith.select %[[d]], %[[a]], %[[b]] : f32
// CHECK-NEXT:   %[[v1:.*]] = arith.addf %[[v0]], %[[b]] : f32
// CHECK-NEXT:   %[[v2:.*]] = arith.mulf %[[a]], %[[b]] : f32
// CHECK-NEXT:   %[[v3:.*]] = arith.select %[[c]], %[[v1]], %[[v2]] : f32
// CHECK-NEXT:   return %[[v3]] : f32
