// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cudnnConvolutionTranspose3D_f32_memref(
      %input: memref<?x?x?x?xf32>, %filter: memref<?x?x?x?x?xf32>,
      %output: memref<?x?x?x?xf32>) { kernel.yield }
  kernel.defn @cudnnConvolutionBackwardFilter3D_f32_memref(
      %input: memref<?x?x?x?xf32>, %gradient: memref<?x?x?x?xf32>,
      %filter: memref<?x?x?x?x?xf32>) { kernel.yield }
  kernel.defn @cudnnConvolutionTBCBackward_f32_memref(
      %gradient: memref<?x?x?xf32>, %filter: memref<?x?x?xf32>,
      %output: memref<?x?x?xf32>) { kernel.yield }

  func.func @routes(
      %input3d: memref<?x?x?x?xf32>,
      %filter3d: memref<?x?x?x?x?xf32>,
      %output3d: memref<?x?x?x?xf32>,
      %gradient3d: memref<?x?x?x?xf32>,
      %gradient_filter3d: memref<?x?x?x?x?xf32>,
      %gradient_tbc: memref<?x?x?xf32>,
      %filter_tbc: memref<?x?x?xf32>,
      %output_tbc: memref<?x?x?xf32>) {
    kernel.launch @cudnnConvolutionTranspose3D_f32_memref(
        %input3d, %filter3d, %output3d) :
        (memref<?x?x?x?xf32>, memref<?x?x?x?x?xf32>,
         memref<?x?x?x?xf32>) -> ()
    kernel.launch @cudnnConvolutionBackwardFilter3D_f32_memref(
        %input3d, %gradient3d, %gradient_filter3d) :
        (memref<?x?x?x?xf32>, memref<?x?x?x?xf32>,
         memref<?x?x?x?x?xf32>) -> ()
    kernel.launch @cudnnConvolutionTBCBackward_f32_memref(
        %gradient_tbc, %filter_tbc, %output_tbc) :
        (memref<?x?x?xf32>, memref<?x?x?xf32>, memref<?x?x?xf32>) -> ()
    return
  }
}

// CHECK-LABEL: func.func @routes
// CHECK: call @polygeist_cudnn_conv_transpose3d_f32
// CHECK: call @polygeist_cudnn_conv_backward_filter3d_f32
// CHECK: call @polygeist_cudnn_conv_tbc_backward_f32
// CHECK-NOT: kernel.launch
