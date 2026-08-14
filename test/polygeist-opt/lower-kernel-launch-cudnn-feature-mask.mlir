// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s
module {
  kernel.defn @cudnnFeatureMaskScale_f32_tensor(
      %input: tensor<?x?x?x?xf32>, %mask: tensor<?x?xf32>, %scale: f32,
      %out: tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32> {
    kernel.yield %out : tensor<?x?x?x?xf32>
  }
  func.func @dropout(%input: tensor<?x?x?x?xf32>, %mask: tensor<?x?xf32>,
                     %scale: f32, %out: tensor<?x?x?x?xf32>)
      -> tensor<?x?x?x?xf32> {
    %r = kernel.launch @cudnnFeatureMaskScale_f32_tensor(
        %input, %mask, %scale, %out) :
        (tensor<?x?x?x?xf32>, tensor<?x?xf32>, f32,
         tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xf32>
    return %r : tensor<?x?x?x?xf32>
  }
}
// CHECK: call @polygeist_cudnn_feature_mask_scale_f32
// CHECK-NOT: kernel.launch
