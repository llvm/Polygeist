// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s

module {
  kernel.defn @cudnnDepthwiseConvolution2D_f32_memref(
      %input: memref<?x?x?x?xf32>, %filter: memref<?x?x?xf32>,
      %bias: memref<?xf32>, %output: memref<?x?x?x?xf32>) { kernel.yield }

  func.func @depthwise(%input: memref<?x?x?x?xf32>,
                       %filter: memref<?x?x?xf32>,
                       %bias: memref<?xf32>,
                       %output: memref<?x?x?x?xf32>) {
    kernel.launch @cudnnDepthwiseConvolution2D_f32_memref(
        %input, %filter, %bias, %output) :
        (memref<?x?x?x?xf32>, memref<?x?x?xf32>, memref<?xf32>,
         memref<?x?x?x?xf32>) -> ()
    return
  }
}

// CHECK: call @polygeist_cudnn_depthwise_conv2d_f32
// CHECK-NOT: kernel.launch
