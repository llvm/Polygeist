// RUN: polygeist-opt --remove-iter-args %s | FileCheck %s

func.func @invariant_tensor(%n: index, %input: tensor<?xf64>) -> tensor<?xf64> {
  %result = affine.for %i = 0 to %n iter_args(%carried = %input) -> tensor<?xf64> {
    func.call @opaque_side_effect(%carried) : (tensor<?xf64>) -> ()
    affine.yield %carried : tensor<?xf64>
  }
  return %result : tensor<?xf64>
}
func.func private @opaque_side_effect(tensor<?xf64>)

// CHECK-LABEL: func.func @invariant_tensor
// CHECK-SAME: (%{{.*}}: index, %[[INPUT:.*]]: tensor<?xf64>)
// CHECK-NOT: iter_args
// CHECK: func.call @opaque_side_effect(%[[INPUT]])
// CHECK: return %[[INPUT]]
