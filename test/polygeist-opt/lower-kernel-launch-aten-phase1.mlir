// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

#window = affine_map<(d0,d1,d2,d3,d4,d5,d6,d7) ->
                     (d4, d5 + d1, d6 + d2, d7 + d3)>

module {
  kernel.defn @cublasDgemm_outer_product(
      %u: tensor<?xf64>, %v: tensor<?xf64>, %c: tensor<?x?xf64>)
      -> tensor<?x?xf64> {
    kernel.yield %c : tensor<?x?xf64>
  }

  kernel.defn @cublasSgemm_strided_batched_broadcast_rhs(
      %a: tensor<?x?x?xf32>, %b: tensor<?x?xf32>,
      %c: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    kernel.yield %c : tensor<?x?x?xf32>
  }

  kernel.defn @cudnnConvolution3D_f32_bias(
      %window: tensor<?x?x?x?x?x?x?x?xf32>,
      %filter: tensor<?x?x?x?x?xf32>, %bias: tensor<?xf32>,
      %out: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    kernel.yield %out : tensor<?x?x?x?xf32>
  }

  // CHECK-LABEL: func.func @outer
  // CHECK: call @polygeist_cublas_dgemm_outer_product
  func.func @outer(%u: tensor<?xf64>, %v: tensor<?xf64>,
                   %c: tensor<?x?xf64>) -> tensor<?x?xf64> {
    %0 = kernel.launch @cublasDgemm_outer_product(%u, %v, %c)
        : (tensor<?xf64>, tensor<?xf64>, tensor<?x?xf64>)
          -> tensor<?x?xf64>
    return %0 : tensor<?x?xf64>
  }

  // CHECK-LABEL: func.func @broadcast_bmm
  // CHECK: call @polygeist_cublas_sgemm_strided_batched_broadcast_rhs
  func.func @broadcast_bmm(%a: tensor<?x?x?xf32>, %b: tensor<?x?xf32>,
                           %c: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %0 = kernel.launch @cublasSgemm_strided_batched_broadcast_rhs(%a, %b, %c)
        : (tensor<?x?x?xf32>, tensor<?x?xf32>, tensor<?x?x?xf32>)
          -> tensor<?x?x?xf32>
    return %0 : tensor<?x?x?xf32>
  }

  // CHECK-LABEL: func.func @conv3d_bias
  // CHECK: call @polygeist_cudnn_conv3d_channels_f32
  func.func @conv3d_bias(
      %input: tensor<?x?x?x?xf32>, %filter: tensor<?x?x?x?x?xf32>,
      %bias: tensor<?xf32>, %out: tensor<?x?x?x?xf32>,
      %oc: index, %od: index, %oh: index, %ow: index,
      %ic: index, %kd: index, %kh: index, %kw: index)
      -> tensor<?x?x?x?xf32> {
    %window = polygeist.submap(
        %input, %oc, %od, %oh, %ow, %ic, %kd, %kh, %kw) {map = #window}
        : (tensor<?x?x?x?xf32>, index, index, index, index,
           index, index, index, index) -> tensor<?x?x?x?x?x?x?x?xf32>
    %0 = kernel.launch @cudnnConvolution3D_f32_bias(
        %window, %filter, %bias, %out)
        : (tensor<?x?x?x?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>,
           tensor<?xf32>, tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %0 : tensor<?x?x?x?xf32>
  }
}
