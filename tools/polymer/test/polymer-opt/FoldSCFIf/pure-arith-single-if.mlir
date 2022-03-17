// RUN: polymer-opt %s -fold-scf-if | FileCheck %s

func @foo(%a: f32, %b: f32, %c: i1) -> f32 {
  %0 = scf.if %c -> f32 {
    %1 = arith.addf %a, %b : f32
    scf.yield %1 : f32
  } else {
    %1 = arith.mulf %a, %b : f32
    scf.yield %1 : f32
  }
  return %0 : f32
}

// CHECK: func @foo(%[[a:.*]]: f32, %[[b:.*]]: f32, %[[c:.*]]: i1) -> f32 
// CHECK-NEXT:   %[[v0:.*]] = arith.addf %[[a]], %[[b]] : f32
// CHECK-NEXT:   %[[v1:.*]] = arith.mulf %[[a]], %[[b]] : f32
// CHECK-NEXT:   %[[v2:.*]] = arith.select %[[c]], %[[v0]], %[[v1]] : f32
// CHECK-NEXT:   return %[[v2]] : f32
