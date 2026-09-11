// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s --disable-kernel cudaAdd_f32_tensor | sed '/^\/\/ CHECK/d' | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0)[s0] -> (s0, d0)>

module {
  func.func @identity_submap(
      %x: tensor<?xf32>, %y: tensor<?xf32>, %out: tensor<?xf32>,
      %n: index) -> tensor<?xf32> {
    %xv = polygeist.submap(%x, %n) {map = #map}
        : (tensor<?xf32>, index) -> tensor<?xf32>
    %yv = polygeist.submap(%y, %n) {map = #map}
        : (tensor<?xf32>, index) -> tensor<?xf32>
    %ov = polygeist.submap(%out, %n) {map = #map}
        : (tensor<?xf32>, index) -> tensor<?xf32>
    %sum = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel"]}
        ins(%xv, %yv : tensor<?xf32>, tensor<?xf32>)
        outs(%ov : tensor<?xf32>) {
    ^bb0(%in: f32, %in_8: f32, %out: f32):
      %r = arith.addf %in, %in_8 : f32
      linalg.yield %r : f32
    } -> tensor<?xf32>
    %updated = polygeist.submapInverse(%out, %sum, %n) {map = #map}
        : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    return %updated : tensor<?xf32>
  }

  func.func @offset_row_rejected(
      %x: tensor<?x?xf32>, %out: tensor<?xf32>, %row_index: index,
      %n: index) -> tensor<?xf32> {
    %row_view = polygeist.submap(%x, %row_index, %n) {map = #map1}
        : (tensor<?x?xf32>, index, index) -> tensor<?xf32>
    %copy = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel"]}
        ins(%row_view : tensor<?xf32>) outs(%out : tensor<?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %r = arith.addf %in, %in : f32
      linalg.yield %r : f32
    } -> tensor<?xf32>
    return %copy : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @identity_submap
// CHECK: kernel.launch @cudnnPointwiseGraph_f32
// CHECK-LABEL: func.func @offset_row_rejected
// CHECK: linalg.generic
// CHECK-NOT: kernel.launch @cudnnPointwiseGraph_f32
