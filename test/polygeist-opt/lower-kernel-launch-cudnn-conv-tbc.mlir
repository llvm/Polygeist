// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s
module {
  kernel.defn @cudnnConvolutionTBC_f32_memref(
      %input: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>,
      %output: memref<?x?x?xf32>) { kernel.yield }
  func.func @conv_tbc(%input: memref<?x?x?xf32>,
                      %filter: memref<?x?x?xf32>,
                      %output: memref<?x?x?xf32>) {
    kernel.launch @cudnnConvolutionTBC_f32_memref(
        %input, %filter, %output) :
        (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
    return
  }
}
// CHECK: call @polygeist_cudnn_conv_tbc_f32
// CHECK-NOT: kernel.launch
