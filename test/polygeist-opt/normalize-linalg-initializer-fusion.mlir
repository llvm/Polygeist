// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=2 fuse-initializers=true" | FileCheck %s
// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=2" | FileCheck %s --check-prefix=DEFAULT
#A = affine_map<(b,i,j,k)->(b,i,k)>
#B = affine_map<(b,i,j,k)->(b,k,j)>
#C = affine_map<(b,i,j,k)->(b,i,j)>
#I = affine_map<(b,i,j)->(b,i,j)>
func.func @seeded_batch(%a:tensor<2x3x4xf32>,%b:tensor<2x4x5xf32>,%c:tensor<2x3x5xf32>,%seed:f32) -> tensor<2x3x5xf32> {
 %c2=arith.constant 2:index
 %c3=arith.constant 3:index
 %c5=arith.constant 5:index
 %init=linalg.generic {indexing_maps=[#I],iterator_types=["parallel","parallel","parallel"]} outs(%c:tensor<2x3x5xf32>) {
 ^bb0(%old:f32):
  linalg.yield %seed:f32
 } -> tensor<2x3x5xf32>
 %written=polygeist.submapInverse(%c,%init,%c2,%c3,%c5) {map=#I} : (tensor<2x3x5xf32>,tensor<2x3x5xf32>,index,index,index)->tensor<2x3x5xf32>
 %view=polygeist.submap(%written,%c2,%c3,%c5) {map=#I} : (tensor<2x3x5xf32>,index,index,index)->tensor<2x3x5xf32>
 %result=linalg.generic {indexing_maps=[#A,#B,#C],iterator_types=["parallel","parallel","parallel","reduction"]} ins(%a,%b:tensor<2x3x4xf32>,tensor<2x4x5xf32>) outs(%view:tensor<2x3x5xf32>) {
 ^bb0(%x:f32,%y:f32,%z:f32):
  %p=arith.mulf %x,%y:f32
  %v=arith.addf %z,%p:f32
  linalg.yield %v:f32
 } -> tensor<2x3x5xf32>
 return %result:tensor<2x3x5xf32>
}
// CHECK-LABEL: func.func @seeded_batch
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

func.func @same_view(%base:tensor<16xf32>,%part:tensor<8xf32>) -> tensor<8xf32> {
 %c8=arith.constant 8:index
 %written=polygeist.submapInverse(%base,%part,%c8) {map=affine_map<(i)->(i)>} : (tensor<16xf32>,tensor<8xf32>,index)->tensor<16xf32>
 %read=polygeist.submap(%written,%c8) {map=affine_map<(i)->(i)>} : (tensor<16xf32>,index)->tensor<8xf32>
 return %read:tensor<8xf32>
}
// CHECK-LABEL: func.func @same_view
// CHECK-NOT: polygeist.submap
// CHECK: return %arg1

func.func @different_bounds(%base:tensor<16xf32>,%part:tensor<8xf32>) -> tensor<6xf32> {
 %c8=arith.constant 8:index
 %c6=arith.constant 6:index
 %written=polygeist.submapInverse(%base,%part,%c8) {map=affine_map<(i)->(i)>} : (tensor<16xf32>,tensor<8xf32>,index)->tensor<16xf32>
 %read=polygeist.submap(%written,%c6) {map=affine_map<(i)->(i)>} : (tensor<16xf32>,index)->tensor<6xf32>
 return %read:tensor<6xf32>
}
// CHECK-LABEL: func.func @different_bounds
// CHECK: polygeist.submapInverse
// CHECK: polygeist.submap(

func.func @repeated_coordinate(%base:tensor<2xf32>,%part:tensor<2x3xf32>) -> tensor<2x3xf32> {
 %c2=arith.constant 2:index
 %c3=arith.constant 3:index
 %written=polygeist.submapInverse(%base,%part,%c2,%c3) {map=affine_map<(i,j)->(i)>} : (tensor<2xf32>,tensor<2x3xf32>,index,index)->tensor<2xf32>
 %read=polygeist.submap(%written,%c2,%c3) {map=affine_map<(i,j)->(i)>} : (tensor<2xf32>,index,index)->tensor<2x3xf32>
 return %read:tensor<2x3xf32>
}
// CHECK-LABEL: func.func @repeated_coordinate
// CHECK: polygeist.submapInverse
// CHECK: polygeist.submap(

func.func @overwrite_same_view(%base:tensor<16xf32>,%first:tensor<8xf32>,%second:tensor<8xf32>) -> tensor<16xf32> {
 %c8=arith.constant 8:index
 %one=polygeist.submapInverse(%base,%first,%c8) {map=affine_map<(i)->(i)>} : (tensor<16xf32>,tensor<8xf32>,index)->tensor<16xf32>
 %two=polygeist.submapInverse(%one,%second,%c8) {map=affine_map<(i)->(i)>} : (tensor<16xf32>,tensor<8xf32>,index)->tensor<16xf32>
 return %two:tensor<16xf32>
}
// CHECK-LABEL: func.func @overwrite_same_view
// CHECK: polygeist.submapInverse(%arg0, %arg2,
// CHECK-NOT: polygeist.submapInverse
// CHECK: return

func.func @overwrite_different_bounds(%base:tensor<16xf32>,%first:tensor<8xf32>,%second:tensor<6xf32>) -> tensor<16xf32> {
 %c8=arith.constant 8:index
 %c6=arith.constant 6:index
 %one=polygeist.submapInverse(%base,%first,%c8) {map=affine_map<(i)->(i)>} : (tensor<16xf32>,tensor<8xf32>,index)->tensor<16xf32>
 %two=polygeist.submapInverse(%one,%second,%c6) {map=affine_map<(i)->(i)>} : (tensor<16xf32>,tensor<6xf32>,index)->tensor<16xf32>
 return %two:tensor<16xf32>
}
// CHECK-LABEL: func.func @overwrite_different_bounds
// CHECK: polygeist.submapInverse
// CHECK: polygeist.submapInverse
// CHECK: return

func.func @overwrite_repeated_coordinate(%base:tensor<2xf32>,%first:tensor<2x3xf32>,%second:tensor<2x3xf32>) -> tensor<2xf32> {
 %c2=arith.constant 2:index
 %c3=arith.constant 3:index
 %one=polygeist.submapInverse(%base,%first,%c2,%c3) {map=affine_map<(i,j)->(i)>} : (tensor<2xf32>,tensor<2x3xf32>,index,index)->tensor<2xf32>
 %two=polygeist.submapInverse(%one,%second,%c2,%c3) {map=affine_map<(i,j)->(i)>} : (tensor<2xf32>,tensor<2x3xf32>,index,index)->tensor<2xf32>
 return %two:tensor<2xf32>
}
// CHECK-LABEL: func.func @overwrite_repeated_coordinate
// CHECK: polygeist.submapInverse
// CHECK: polygeist.submapInverse
// CHECK: return

// DEFAULT-LABEL: func.func @seeded_batch
// DEFAULT: linalg.generic
// DEFAULT: scf.for
// DEFAULT: linalg.generic
// DEFAULT: return
