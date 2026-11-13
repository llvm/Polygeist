// RUN: polygeist-opt --lower-kernel-launch-to-cublas %s | FileCheck %s

module {
  kernel.defn @cudnnStencil3D7pt_f32_flat_tensor(
      %A: tensor<?xf32>, %C: tensor<?xf32>, %center: f32, %neighbor: f32,
      %ny: index, %nx: index, %ox: index, %oy: index, %oz: index)
      -> tensor<?xf32> {
    kernel.yield %C : tensor<?xf32>
  }

  func.func @stencil(%A: tensor<?xf32>, %C: tensor<?xf32>,
                     %center: f32, %neighbor: f32,
                     %ny: index, %nx: index,
                     %ox: index, %oy: index, %oz: index) -> tensor<?xf32> {
    %result = kernel.launch @cudnnStencil3D7pt_f32_flat_tensor(
        %A, %C, %center, %neighbor, %ny, %nx, %ox, %oy, %oz) :
        (tensor<?xf32>, tensor<?xf32>, f32, f32, index, index, index,
         index, index) -> tensor<?xf32>
    return %result : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @stencil
// CHECK: arith.index_cast
// CHECK: call @polygeist_cudnn_stencil3d_7pt_f32_flat
// CHECK-NOT: kernel.launch
// CHECK: return %{{.*}} : tensor<?xf32>
