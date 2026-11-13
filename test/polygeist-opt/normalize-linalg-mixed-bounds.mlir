// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-parallel-dims=2 max-operand-rank=2" | FileCheck %s
func.func @mixed_bounds(%a:tensor<3x4xf64>,%b:tensor<4x5xf64>,%c:tensor<3x5xf64>)->tensor<3x5xf64> {
  %c0=arith.constant 0:index
  %c1=arith.constant 1:index
  %c2=arith.constant 2:index
  %c4=arith.constant 4:index
  %c5=arith.constant 5:index
  %bview=polygeist.submap(%b,%c4,%c5) {map=affine_map<(d0,d1)->(d0,d1)>} : (tensor<4x5xf64>,index,index)->tensor<4x5xf64>
  %out=scf.for %i=%c0 to %c2 step %c1 iter_args(%acc=%c)->tensor<3x5xf64> {
    %r=linalg.generic {indexing_maps=[affine_map<(i,j,k)->(i,k)>,affine_map<(i,j,k)->(k,j)>,affine_map<(i,j,k)->(i,j)>],iterator_types=["parallel","parallel","reduction"]} ins(%a,%bview:tensor<3x4xf64>,tensor<4x5xf64>) outs(%acc:tensor<3x5xf64>) {
    ^bb0(%x:f64,%y:f64,%z:f64):
      %p=arith.mulf %x,%y:f64
      %s=arith.addf %z,%p:f64
      linalg.yield %s:f64
    }->tensor<3x5xf64>
    scf.yield %r:tensor<3x5xf64>
  }
  return %out:tensor<3x5xf64>
}
// CHECK-LABEL: func.func @mixed_bounds
// CHECK: %[[B:[a-zA-Z0-9_]+]] = polygeist.submap
// CHECK: scf.for
// CHECK: %[[A:[a-zA-Z0-9_]+]] = polygeist.submap
// CHECK: %[[C:[a-zA-Z0-9_]+]] = polygeist.submap
// CHECK: ins(%[[A]], %[[B]]
// CHECK: outs(%[[C]]
// CHECK: polygeist.submapInverse
