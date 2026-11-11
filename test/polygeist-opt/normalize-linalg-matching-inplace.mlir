// RUN: polygeist-opt %s --normalize-linalg-matching-views | FileCheck %s

// CHECK-LABEL: func.func @dynamic_outer
// CHECK: linalg.generic
// CHECK: ins({{.*}}, {{.*}} :
// CHECK: polygeist.submapInverse{{.*}} : (tensor<?xf32>, tensor<6xf32>
func.func @dynamic_outer(%a: tensor<?x3xf32>, %b: tensor<?x3xf32>) -> tensor<?x3xf32> {
  %n = arith.constant 2 : index
  %m = arith.constant 3 : index
  %av = polygeist.submap(%a, %n, %m) {map=affine_map<(i,j)->(i,j)>} : (tensor<?x3xf32>, index, index) -> tensor<2x3xf32>
  %bv = polygeist.submap(%b, %n, %m) {map=affine_map<(i,j)->(i,j)>} : (tensor<?x3xf32>, index, index) -> tensor<2x3xf32>
  %r = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%av: tensor<2x3xf32>) outs(%bv: tensor<2x3xf32>) {
  ^bb0(%x:f32,%old:f32):
    %v = arith.addf %x, %old : f32
    linalg.yield %v : f32
  } -> tensor<2x3xf32>
  %updated = polygeist.submapInverse(%b, %r, %n, %m) {map=affine_map<(i,j)->(i,j)>} : (tensor<?x3xf32>, tensor<2x3xf32>, index, index) -> tensor<?x3xf32>
  return %updated : tensor<?x3xf32>
}

// CHECK-LABEL: func.func @unknown_inner
// CHECK: linalg.generic
// CHECK: ins({{.*}}, {{.*}} :
// CHECK: polygeist.submapInverse{{.*}} : (tensor<?x?xf32>, tensor<2x3xf32>
func.func @unknown_inner(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %n = arith.constant 2 : index
  %m = arith.constant 3 : index
  %av = polygeist.submap(%a, %n, %m) {map=affine_map<(i,j)->(i,j)>} : (tensor<?x?xf32>, index, index) -> tensor<2x3xf32>
  %bv = polygeist.submap(%b, %n, %m) {map=affine_map<(i,j)->(i,j)>} : (tensor<?x?xf32>, index, index) -> tensor<2x3xf32>
  %r = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%av: tensor<2x3xf32>) outs(%bv: tensor<2x3xf32>) {
  ^bb0(%x:f32,%old:f32):
    %v = arith.addf %x, %old : f32
    linalg.yield %v : f32
  } -> tensor<2x3xf32>
  %updated = polygeist.submapInverse(%b, %r, %n, %m) {map=affine_map<(i,j)->(i,j)>} : (tensor<?x?xf32>, tensor<2x3xf32>, index, index) -> tensor<?x?xf32>
  return %updated : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @zero_inner
// CHECK: linalg.generic
// CHECK: ins({{.*}}, {{.*}} :
// CHECK: polygeist.submapInverse{{.*}} : (tensor<?x0xf32>, tensor<2x0xf32>
func.func @zero_inner(%a: tensor<?x0xf32>, %b: tensor<?x0xf32>) -> tensor<?x0xf32> {
  %n = arith.constant 2 : index
  %m = arith.constant 0 : index
  %av = polygeist.submap(%a, %n, %m) {map=affine_map<(i,j)->(i,j)>} : (tensor<?x0xf32>, index, index) -> tensor<2x0xf32>
  %bv = polygeist.submap(%b, %n, %m) {map=affine_map<(i,j)->(i,j)>} : (tensor<?x0xf32>, index, index) -> tensor<2x0xf32>
  %r = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]} ins(%av: tensor<2x0xf32>) outs(%bv: tensor<2x0xf32>) {
  ^bb0(%x:f32,%old:f32):
    %v = arith.addf %x, %old : f32
    linalg.yield %v : f32
  } -> tensor<2x0xf32>
  %updated = polygeist.submapInverse(%b, %r, %n, %m) {map=affine_map<(i,j)->(i,j)>} : (tensor<?x0xf32>, tensor<2x0xf32>, index, index) -> tensor<?x0xf32>
  return %updated : tensor<?x0xf32>
}
