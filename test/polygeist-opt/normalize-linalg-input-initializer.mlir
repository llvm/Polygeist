// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=2 fuse-initializers=true" | FileCheck %s
#id=affine_map<(i,j)->(i,j)>
#x=affine_map<(i,j)->(i)>
#y=affine_map<(i,j)->(j)>
func.func @captured_seed(%x:tensor<2xf64>,%y:tensor<3xf64>,%out:tensor<2x3xf64>,%seed:f64)->tensor<2x3xf64> {
 %init=linalg.generic {indexing_maps=[#id],iterator_types=["parallel","parallel"]} outs(%out:tensor<2x3xf64>) {
 ^bb0(%old:f64):
  linalg.yield %seed:f64
 } -> tensor<2x3xf64>
 %result=linalg.generic {indexing_maps=[#x,#y,#id],iterator_types=["parallel","parallel"]} ins(%x,%y:tensor<2xf64>,tensor<3xf64>) outs(%init:tensor<2x3xf64>) {
 ^bb0(%a:f64,%b:f64,%v:f64):
  %p=arith.mulf %a,%b:f64
  %r=arith.addf %v,%p:f64
  linalg.yield %r:f64
 } -> tensor<2x3xf64>
 return %result:tensor<2x3xf64>
}
// CHECK-LABEL: func.func @captured_seed
// CHECK: linalg.generic
// CHECK-NOT: linalg.generic
// CHECK: arith.mulf
// CHECK: arith.addf %arg3
// CHECK-NOT: linalg.generic
// CHECK: return

// A producer with element computation is outside this initializer-only option.
func.func @computed_seed(%x:tensor<6xi32>,%old:tensor<6xi32>)->tensor<6xi32> {
 %a=linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]} ins(%x:tensor<6xi32>) outs(%old:tensor<6xi32>) {
 ^bb0(%v:i32,%o:i32):
  %r=arith.muli %v,%v:i32
  linalg.yield %r:i32
 } -> tensor<6xi32>
 %b=linalg.generic {indexing_maps=[affine_map<(i)->(i)>,affine_map<(i)->(i)>],iterator_types=["parallel"]} ins(%a:tensor<6xi32>) outs(%old:tensor<6xi32>) {
 ^bb0(%v:i32,%o:i32):
  %r=arith.addi %v,%v:i32
  linalg.yield %r:i32
 } -> tensor<6xi32>
 return %b:tensor<6xi32>
}
// CHECK-LABEL: func.func @computed_seed
// CHECK: linalg.generic
// CHECK: arith.muli
// CHECK: linalg.generic
// CHECK: arith.addi
// CHECK: return

// A partial initialized view is overwritten. Preserve the untouched columns.
func.func @partial_view(%x:tensor<2xf64>,%y:tensor<3xf64>,%base:tensor<2x5xf64>,%seed:f64)->tensor<2x5xf64> {
 %c2=arith.constant 2:index
 %c3=arith.constant 3:index
 %old=polygeist.submap(%base,%c2,%c3) {map=#id} : (tensor<2x5xf64>,index,index)->tensor<2x3xf64>
 %init=linalg.generic {indexing_maps=[#id],iterator_types=["parallel","parallel"]} outs(%old:tensor<2x3xf64>) {
 ^bb0(%v:f64):
  linalg.yield %seed:f64
 } -> tensor<2x3xf64>
 %written=polygeist.submapInverse(%base,%init,%c2,%c3) {map=#id} : (tensor<2x5xf64>,tensor<2x3xf64>,index,index)->tensor<2x5xf64>
 %view=polygeist.submap(%written,%c2,%c3) {map=#id} : (tensor<2x5xf64>,index,index)->tensor<2x3xf64>
 %value=linalg.generic {indexing_maps=[#x,#y,#id],iterator_types=["parallel","parallel"]} ins(%x,%y:tensor<2xf64>,tensor<3xf64>) outs(%view:tensor<2x3xf64>) {
 ^bb0(%a:f64,%b:f64,%v:f64):
  %p=arith.mulf %a,%b:f64
  %r=arith.addf %v,%p:f64
  linalg.yield %r:f64
 } -> tensor<2x3xf64>
 %result=polygeist.submapInverse(%written,%value,%c2,%c3) {map=#id} : (tensor<2x5xf64>,tensor<2x3xf64>,index,index)->tensor<2x5xf64>
 return %result:tensor<2x5xf64>
}
// CHECK-LABEL: func.func @partial_view
// CHECK: linalg.generic
// CHECK-NOT: linalg.generic
// CHECK: arith.addf %arg3
// CHECK: } -> tensor<2x3xf64>
// CHECK-NOT: linalg.generic
// CHECK: polygeist.submapInverse(%arg2,
// CHECK: return
