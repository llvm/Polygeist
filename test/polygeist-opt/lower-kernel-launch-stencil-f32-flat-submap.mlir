// RUN: polygeist-opt %s --lower-kernel-launch-to-cublas --canonicalize | FileCheck %s

#center = affine_map<(d0, d1, d2)[s0, s1] ->
    (((d2 + 1) * s0 + d1 + 1) * s1 + d0 + 1)>

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

  func.func @flat_submap_stencil(
      %input: memref<?xf32>, %output: memref<?xf32>,
      %ny: index, %nx: index, %sx: index, %sy: index, %sz: index,
      %neighbor: f32, %center_coeff: f32) -> tensor<?xf32> {
    %zero = arith.constant 0.0 : f32
    %input_tensor = bufferization.to_tensor %input restrict
        : memref<?xf32>
    %output_tensor = bufferization.to_tensor %output restrict writable
        : memref<?xf32>
    %input_view = polygeist.submap(
        %input_tensor, %ny, %nx, %sx, %sy, %sz) {map = #center}
        : (tensor<?xf32>, index, index, index, index, index)
          -> tensor<?x?x?xf32>
    %output_view = polygeist.submap(
        %output_tensor, %ny, %nx, %sx, %sy, %sz) {map = #center}
        : (tensor<?xf32>, index, index, index, index, index)
          -> tensor<?x?x?xf32>
    %result = kernel.launch @customStencil3D7pt_f32_tensor(
        %input_view, %input_view, %input_view, %input_view,
        %input_view, %input_view, %input_view, %output_view,
        %zero, %zero, %zero,
        %neighbor, %neighbor, %neighbor, %neighbor, %neighbor, %neighbor,
        %center_coeff) :
        (tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>,
         tensor<?x?x?xf32>, tensor<?x?x?xf32>, tensor<?x?x?xf32>,
         tensor<?x?x?xf32>, tensor<?x?x?xf32>,
         f32, f32, f32, f32, f32, f32, f32, f32, f32, f32)
         -> tensor<?x?x?xf32>
    %updated = polygeist.submapInverse(
        %output_tensor, %result, %ny, %nx, %sx, %sy, %sz) {map = #center}
        : (tensor<?xf32>, tensor<?x?x?xf32>, index, index, index, index,
           index) -> tensor<?xf32>
    return %updated : tensor<?xf32>
  }
}

// CHECK-LABEL: func.func @flat_submap_stencil
// CHECK: affine.apply #{{.*}}()[%{{.*}}, %{{.*}}]
// CHECK: call @polygeist_custom_stencil3d_7pt_strided_f32
// CHECK-NOT: kernel.launch
// CHECK-NOT: polygeist.submap
// CHECK: return
