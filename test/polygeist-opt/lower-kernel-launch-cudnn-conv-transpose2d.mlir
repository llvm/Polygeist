// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s
module {
  kernel.defn @cudnnConvolutionTranspose2D_f32_memref(
      %input: memref<?x?x?x?xf32>, %filter: memref<?x?x?x?xf32>,
      %output: memref<?x?x?x?xf32>) { kernel.yield }
  func.func @conv_transpose(%input: memref<?x?x?x?xf32>,
                           %filter: memref<?x?x?x?xf32>,
                           %output: memref<?x?x?x?xf32>) {
    kernel.launch @cudnnConvolutionTranspose2D_f32_memref(
        %input, %filter, %output) :
        (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>,
         memref<?x?x?x?xf32>) -> ()
    return
  }
}
// CHECK: call @polygeist_cudnn_conv_transpose2d_f32
// CHECK-NOT: kernel.launch
