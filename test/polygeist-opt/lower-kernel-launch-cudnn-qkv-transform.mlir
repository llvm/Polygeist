// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s
module {
  kernel.defn @cudnnTransformBiasRescaleQKV_f32_memref(
      %qkv: memref<?x?x?x?x?xf32>, %bias: memref<?x?x?xf32>, %scale: f32,
      %q: memref<?x?x?x?xf32>, %k: memref<?x?x?x?xf32>,
      %v: memref<?x?x?x?xf32>) { kernel.yield }
  func.func @qkv(%qkv: memref<?x?x?x?x?xf32>,
                 %bias: memref<?x?x?xf32>, %scale: f32,
                 %q: memref<?x?x?x?xf32>, %k: memref<?x?x?x?xf32>,
                 %v: memref<?x?x?x?xf32>) {
    kernel.launch @cudnnTransformBiasRescaleQKV_f32_memref(
        %qkv, %bias, %scale, %q, %k, %v) :
        (memref<?x?x?x?x?xf32>, memref<?x?x?xf32>, f32,
         memref<?x?x?x?xf32>, memref<?x?x?x?xf32>,
         memref<?x?x?x?xf32>) -> ()
    return
  }
}
// CHECK: call @polygeist_cudnn_transform_bias_rescale_qkv_f32
// CHECK-NOT: kernel.launch
