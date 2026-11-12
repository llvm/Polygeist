// RUN: polygeist-opt %s --normalize-linalg-matching-views="max-operand-rank=1 fuse-initializers=true" | FileCheck %s
#row = affine_map<(r,k)->(r)>
#input = affine_map<(r,k)->(r,k)>
#id = affine_map<(r)->(r)>
func.func @rows(%x:tensor<2x3xf64>,%out:tensor<2xf64>,%seed:f64)->tensor<2xf64> {
 %init=linalg.generic {indexing_maps=[#id],iterator_types=["parallel"]} outs(%out:tensor<2xf64>) {
 ^bb0(%old:f64):
  linalg.yield %seed:f64
 } -> tensor<2xf64>
 %sum=linalg.generic {indexing_maps=[#input,#row],iterator_types=["parallel","reduction"]} ins(%x:tensor<2x3xf64>) outs(%init:tensor<2xf64>) {
 ^bb0(%a:f64,%acc:f64):
  %s=arith.addf %acc,%a:f64
  linalg.yield %s:f64
 } -> tensor<2xf64>
 return %sum:tensor<2xf64>
}
// CHECK-LABEL: func.func @rows
// CHECK: scf.for
// CHECK: tensor.insert %arg2
// CHECK: polygeist.submap(
// CHECK: linalg.generic
// CHECK: iterator_types = ["reduction"]
// CHECK: polygeist.submapInverse
// CHECK: scf.yield

func.func @scalar_read(%base:tensor<8xi32>,%v:i32)->tensor<i32> {
 %c2=arith.constant 2:index
 %written=tensor.insert %v into %base[%c2]:tensor<8xi32>
 %view=polygeist.submap(%written) {map=affine_map<()->(2)>} : (tensor<8xi32>)->tensor<i32>
 return %view:tensor<i32>
}
// CHECK-LABEL: func.func @scalar_read
// CHECK-NOT: polygeist.submap
// CHECK: tensor.insert %arg1
// CHECK: return

func.func @dynamic_scalar_write(%base:tensor<8xi32>,%v:i32,%part:tensor<i32>,%i:index)->tensor<8xi32> {
 %written=tensor.insert %v into %base[%i]:tensor<8xi32>
 %updated=polygeist.submapInverse(%written,%part,%i) {map=affine_map<()[s0]->(s0)>} : (tensor<8xi32>,tensor<i32>,index)->tensor<8xi32>
 return %updated:tensor<8xi32>
}
// CHECK-LABEL: func.func @dynamic_scalar_write
// CHECK-NOT: tensor.insert
// CHECK: polygeist.submapInverse(%arg0, %arg2, %arg3)
// CHECK: return

func.func @different_scalar_index(%base:tensor<8xi32>,%v:i32,%i:index,%j:index)->tensor<i32> {
 %written=tensor.insert %v into %base[%i]:tensor<8xi32>
 %view=polygeist.submap(%written,%j) {map=affine_map<()[s0]->(s0)>} : (tensor<8xi32>,index)->tensor<i32>
 return %view:tensor<i32>
}
// CHECK-LABEL: func.func @different_scalar_index
// CHECK: tensor.insert
// CHECK: polygeist.submap(
// CHECK: return

func.func @point_slice(%base:tensor<8xi32>,%v:tensor<i32>)->tensor<8xi32> {
 %updated=tensor.insert_slice %v into %base[2][1][1]:tensor<i32> into tensor<8xi32>
 return %updated:tensor<8xi32>
}
// CHECK-LABEL: func.func @point_slice
// CHECK-NOT: tensor.insert_slice
// CHECK: polygeist.submapInverse(%arg0, %arg1)
// CHECK: return

func.func @nonpoint_slice(%base:tensor<8xi32>,%v:tensor<2xi32>)->tensor<8xi32> {
 %updated=tensor.insert_slice %v into %base[2][2][1]:tensor<2xi32> into tensor<8xi32>
 return %updated:tensor<8xi32>
}
// CHECK-LABEL: func.func @nonpoint_slice
// CHECK: tensor.insert_slice
// CHECK-NOT: polygeist.submapInverse
// CHECK: return

func.func @different_write_index(%base:tensor<8xi32>,%v:i32,%part:tensor<i32>)->tensor<8xi32> {
 %c2=arith.constant 2:index
 %written=tensor.insert %v into %base[%c2]:tensor<8xi32>
 %updated=polygeist.submapInverse(%written,%part) {map=affine_map<()->(3)>} : (tensor<8xi32>,tensor<i32>)->tensor<8xi32>
 return %updated:tensor<8xi32>
}
// CHECK-LABEL: func.func @different_write_index
// CHECK: tensor.insert
// CHECK: polygeist.submapInverse
// CHECK: return
