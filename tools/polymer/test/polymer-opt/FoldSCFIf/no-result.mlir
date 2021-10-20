// RUN: polymer-opt %s -fold-scf-if | FileCheck %s
func @foo(%a: f32, %b: f32, %c: i1) {
  scf.if %c {
    %0 = arith.addf %a, %b : f32
  } else {
    %0 = arith.mulf %a, %b : f32
  }

  return
}

// CHECK: func @foo
// CHECK-NEXT: arith.addf
// CHECK-NEXT: arith.mulf
// CHECK-NEXT: return
