// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas | FileCheck %s

module {
  kernel.defn @customStencil3D7pt_f32_tensor(
      %a0: tensor<?x?x?xf32>, %a1: tensor<?x?x?xf32>,
      %a2: tensor<?x?x?xf32>, %a3: tensor<?x?x?xf32>,
      %a4: tensor<?x?x?xf32>, %a5: tensor<?x?x?xf32>,
      %a6: tensor<?x?x?xf32>, %out: tensor<?x?x?xf32>,
      %base0: f32, %base_extra: f32, %coeff_extra: f32,
      %c0: f32, %c1: f32, %c2: f32, %c3: f32,
      %c4: f32, %c5: f32, %c6: f32) -> tensor<?x?x?xf32> {
    kernel.yield %out : tensor<?x?x?xf32>
  }

  func.func @stencil(
      %a0: tensor<?x?x?xf32>, %a1: tensor<?x?x?xf32>,
      %a2: tensor<?x?x?xf32>, %a3: tensor<?x?x?xf32>,
      %a4: tensor<?x?x?xf32>, %a5: tensor<?x?x?xf32>,
      %a6: tensor<?x?x?xf32>, %out: tensor<?x?x?xf32>,
      %neighbor: f32, %center: f32) -> tensor<?x?x?xf32> {
    %zero = arith.constant 0.0 : f32
    %result = kernel.launch @customStencil3D7pt_f32_tensor(
        %a0, %a1, %a2, %a3, %a4, %a5, %a6, %out,
        %zero, %zero, %zero,
        %neighbor, %neighbor, %neighbor, %neighbor, %neighbor, %neighbor,
        %center) :
        (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>,
         tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>,
         tensor<?x?x?xf32>, tensor<?x?x?xf32>,
         f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
         -> tensor<?x?x?xf32>
    return %result : tensor<?x?x?xf32>
  }
}

// CHECK-LABEL: func.func @stencil
// CHECK: call @polygeist_custom_stencil3d_7pt_flat_f32
// CHECK-NOT: kernel.launch
