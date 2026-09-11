// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

#window = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4 + d2)>
module {
  kernel.defn @cudnnConvolution1D_f32_bias(
      %windows: tensor<?x?x?x?x?xf32>, %filter: tensor<?x?x?xf32>,
      %bias: tensor<?xf32>, %out: tensor<?x?x?xf32>)
      -> tensor<?x?x?xf32> { kernel.yield %out : tensor<?x?x?xf32> }
  func.func @conv1d(%input: tensor<?x?x?xf32>,
                    %filter: tensor<?x?x?xf32>, %bias: tensor<?xf32>,
                    %out: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
    %c1 = arith.constant 1 : index
    %windows = polygeist.submap(%input, %c1, %c1, %c1, %c1, %c1)
        {map = #window} :
        (tensor<?x?x?xf32>, index, index, index, index, index) ->
        tensor<?x?x?x?x?xf32>
    %r = kernel.launch @cudnnConvolution1D_f32_bias(
        %windows, %filter, %bias, %out) :
        (tensor<?x?x?x?x?xf32>, tensor<?x?x?xf32>, tensor<?xf32>,
         tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
    return %r : tensor<?x?x?xf32>
  }
}

// CHECK: call @polygeist_cudnn_conv1d_bias_f32
// CHECK-NOT: kernel.launch
