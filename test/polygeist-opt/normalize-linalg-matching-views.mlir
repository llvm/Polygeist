// RUN: polygeist-opt %s --normalize-linalg-matching-views | FileCheck %s

#id3 = affine_map<(i,j,k)->(i,j,k)>
#out = affine_map<(i,j,k)->(i,j)>
#shared = affine_map<(i,j,k)->(k)>
#per_group = affine_map<(i,j,k)->(i,k)>

// Arbitrary dimensions, with a shared vector: collapse the two parallel loops.
// CHECK-LABEL: func.func @shared_vector
// CHECK: memref.collapse_shape {{.*}} {{\[\[0, 1\], \[2\]\]}}
// CHECK: iterator_types = ["parallel", "reduction"]
// CHECK: tensor.expand_shape
func.func @shared_vector(%a: memref<3x5x7xf32>, %x: tensor<7xf32>, %y: tensor<3x5xf32>) -> tensor<3x5xf32> {
  %h = arith.constant 3 : index
  %d = arith.constant 5 : index
  %k = arith.constant 7 : index
  %a0 = bufferization.to_tensor %a : memref<3x5x7xf32>
  %av = polygeist.submap(%a0, %h, %d, %k) {map=#id3} : (tensor<3x5x7xf32>, index,index,index)->tensor<?x?x?xf32>
  %xv = polygeist.submap(%x, %h, %d, %k) {map=#shared} : (tensor<7xf32>, index,index,index)->tensor<?x?x?xf32>
  %yv = polygeist.submap(%y, %h, %d, %k) {map=#out} : (tensor<3x5xf32>, index,index,index)->tensor<?x?x?xf32>
  %r = linalg.generic {indexing_maps=[#id3,#id3,#id3], iterator_types=["parallel","parallel","reduction"]} ins(%av,%xv: tensor<?x?x?xf32>,tensor<?x?x?xf32>) outs(%yv: tensor<?x?x?xf32>) {
  ^bb0(%v:f32,%w:f32,%o:f32):
    %p = arith.mulf %v,%w : f32
    %s = arith.addf %o,%p : f32
    linalg.yield %s : f32
  } -> tensor<?x?x?xf32>
  %result = polygeist.submapInverse(%y,%r,%h,%d,%k) {map=#out} : (tensor<3x5xf32>,tensor<?x?x?xf32>,index,index,index)->tensor<3x5xf32>
  return %result : tensor<3x5xf32>
}

// A vector varies with one parallel loop. Flattening would hide that variation.
// CHECK-LABEL: func.func @per_group_vector
// CHECK-NOT: collapse_shape
// CHECK: iterator_types = ["parallel", "parallel", "reduction"]
// CHECK-NOT: collapse_shape
// CHECK: return
func.func @per_group_vector(%a: tensor<3x5x7xf32>, %x: tensor<3x7xf32>, %y: tensor<3x5xf32>) -> tensor<3x5xf32> {
  %h = arith.constant 3 : index
  %d = arith.constant 5 : index
  %k = arith.constant 7 : index
  %av = polygeist.submap(%a, %h, %d, %k) {map=#id3} : (tensor<3x5x7xf32>, index,index,index)->tensor<?x?x?xf32>
  %xv = polygeist.submap(%x, %h, %d, %k) {map=#per_group} : (tensor<3x7xf32>, index,index,index)->tensor<?x?x?xf32>
  %yv = polygeist.submap(%y, %h, %d, %k) {map=#out} : (tensor<3x5xf32>, index,index,index)->tensor<?x?x?xf32>
  %r = linalg.generic {indexing_maps=[#id3,#id3,#id3], iterator_types=["parallel","parallel","reduction"]} ins(%av,%xv: tensor<?x?x?xf32>,tensor<?x?x?xf32>) outs(%yv: tensor<?x?x?xf32>) {
  ^bb0(%v:f32,%w:f32,%o:f32):
    %p = arith.mulf %v,%w : f32
    %s = arith.addf %o,%p : f32
    linalg.yield %s : f32
  } -> tensor<?x?x?xf32>
  %result = polygeist.submapInverse(%y,%r,%h,%d,%k) {map=#out} : (tensor<3x5xf32>,tensor<?x?x?xf32>,index,index,index)->tensor<3x5xf32>
  return %result : tensor<3x5xf32>
}

// This is also valid for unrelated scalar bodies, with no reduction at all.
// CHECK-LABEL: func.func @pointwise
// CHECK: iterator_types = ["parallel"]
func.func @pointwise(%a: tensor<2x6xf64>, %b: tensor<2x6xf64>) -> tensor<2x6xf64> {
  %r = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>], iterator_types=["parallel","parallel"]} ins(%a:tensor<2x6xf64>) outs(%b:tensor<2x6xf64>) {
  ^bb0(%v:f64,%o:f64):
    %s = arith.negf %v : f64
    linalg.yield %s : f64
  } -> tensor<2x6xf64>
  return %r : tensor<2x6xf64>
}

// Padded rows cannot become a zero-copy collapsed memref. The logical tensor
// reshape may remain, but its physical implementation must preserve the holes.
// CHECK-LABEL: func.func @padded_rows
// CHECK-NOT: memref.collapse_shape
// CHECK: tensor.collapse_shape
// CHECK-NOT: memref.collapse_shape
// CHECK: return
func.func @padded_rows(%a: memref<2x3xf32, strided<[4, 1]>>, %b: tensor<2x3xf32>) -> tensor<2x3xf32> {
  %t = bufferization.to_tensor %a : memref<2x3xf32, strided<[4, 1]>>
  %r = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>], iterator_types=["parallel","parallel"]} ins(%t:tensor<2x3xf32>) outs(%b:tensor<2x3xf32>) {
  ^bb0(%v:f32,%o:f32):
    %s = arith.negf %v : f32
    linalg.yield %s : f32
  } -> tensor<2x3xf32>
  return %r : tensor<2x3xf32>
}

// This MLIR version cannot restore two unknown factors from their product.
// CHECK-LABEL: func.func @dynamic_output
// CHECK-NOT: collapse_shape
// CHECK: iterator_types = ["parallel", "parallel"]
// CHECK-NOT: collapse_shape
// CHECK: return
func.func @dynamic_output(%a: tensor<?x?xf64>, %b: tensor<?x?xf64>) -> tensor<?x?xf64> {
  %r = linalg.generic {indexing_maps=[affine_map<(i,j)->(i,j)>,affine_map<(i,j)->(i,j)>], iterator_types=["parallel","parallel"]} ins(%a:tensor<?x?xf64>) outs(%b:tensor<?x?xf64>) {
  ^bb0(%v:f64,%o:f64):
    %s = arith.negf %v : f64
    linalg.yield %s : f64
  } -> tensor<?x?xf64>
  return %r : tensor<?x?xf64>
}
