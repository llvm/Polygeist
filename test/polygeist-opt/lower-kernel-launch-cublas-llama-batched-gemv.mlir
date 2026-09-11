// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cublasSgemm_broadcast3d_simple(
      %a: tensor<?x?x?xf32>, %b: tensor<?x?x?xf32>,
      %c: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    kernel.yield %c : tensor<?x?x?xf32>
  }

  func.func @llama_split_projection(
      %weights: tensor<?x?x?xf32>, %x: tensor<?xf32>,
      %out: tensor<?x?xf32>, %heads: index, %pairs: index, %width: index)
      -> tensor<?x?xf32> {
    %a = polygeist.submap(%weights, %heads, %pairs, %width)
        {map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>} :
        (tensor<?x?x?xf32>, index, index, index) -> tensor<?x?x?xf32>
    %b = polygeist.submap(%x, %heads, %pairs, %width)
        {map = affine_map<(d0, d1, d2) -> (d2)>} :
        (tensor<?xf32>, index, index, index) -> tensor<?x?x?xf32>
    %c = polygeist.submap(%out, %heads, %pairs, %width)
        {map = affine_map<(d0, d1, d2) -> (d0, d1)>} :
        (tensor<?x?xf32>, index, index, index) -> tensor<?x?x?xf32>
    %r = kernel.launch @cublasSgemm_broadcast3d_simple(%a, %b, %c) :
        (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>) ->
        tensor<?x?x?xf32>
    %updated = polygeist.submapInverse(
        %out, %r, %heads, %pairs, %width)
        {map = affine_map<(d0, d1, d2) -> (d0, d1)>} :
        (tensor<?x?xf32>, tensor<?x?x?xf32>, index, index, index) ->
        tensor<?x?xf32>
    return %updated : tensor<?x?xf32>
  }
}

// CHECK-LABEL: func.func @llama_split_projection
// CHECK: %[[ROWS:.*]] = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: call @polygeist_cublas_sgemm(%[[ROWS]], %[[ONE]],
// CHECK-NOT: kernel.launch
