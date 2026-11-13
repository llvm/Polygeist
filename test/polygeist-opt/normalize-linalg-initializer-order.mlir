// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=2 fuse-initializers=true" | FileCheck %s
#A = affine_map<(b,i,j,k)->(b,i,k)>
#B = affine_map<(b,i,j,k)->(k,j)>
#C = affine_map<(b,i,j,k)->(b,i,j)>
#I = affine_map<(b,i,j)->(b,i,j)>
func.func @seeded_shared_matrix(%a:tensor<2x3x4xf32>,%b:tensor<4x5xf32>,%c:tensor<2x3x5xf32>,%seed:f32) -> tensor<2x3x5xf32> {
 %c2=arith.constant 2:index
 %c3=arith.constant 3:index
 %c5=arith.constant 5:index
 %init=linalg.generic {indexing_maps=[#I],iterator_types=["parallel","parallel","parallel"]} outs(%c:tensor<2x3x5xf32>) {
 ^bb0(%old:f32):
  linalg.yield %seed:f32
 } -> tensor<2x3x5xf32>
 %written=polygeist.submapInverse(%c,%init,%c2,%c3,%c5) {map=#I} : (tensor<2x3x5xf32>,tensor<2x3x5xf32>,index,index,index)->tensor<2x3x5xf32>
 %view=polygeist.submap(%written,%c2,%c3,%c5) {map=#I} : (tensor<2x3x5xf32>,index,index,index)->tensor<2x3x5xf32>
 %result=linalg.generic {indexing_maps=[#A,#B,#C],iterator_types=["parallel","parallel","parallel","reduction"]} ins(%a,%b:tensor<2x3x4xf32>,tensor<4x5xf32>) outs(%view:tensor<2x3x5xf32>) {
 ^bb0(%x:f32,%y:f32,%z:f32):
  %p=arith.mulf %x,%y:f32
  %v=arith.addf %z,%p:f32
  linalg.yield %v:f32
 } -> tensor<2x3x5xf32>
 return %result:tensor<2x3x5xf32>
}
// CHECK-LABEL: func.func @seeded_shared_matrix
// CHECK-NOT: linalg.generic
// CHECK: scf.for
// CHECK: linalg.generic
// CHECK: linalg.yield %arg3
// CHECK-NOT: scf.for
// CHECK: linalg.generic
// CHECK: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: scf.yield
// CHECK-NOT: scf.for
// CHECK: return

