// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | FileCheck %s

#map = affine_map<(d0)[s0] -> (d0 + s0 * 3)>
#identity = affine_map<(d0) -> (d0)>
module {
  func.func @aten_max_pool1d_cpu(%input: memref<?xf32>,
                                 %output: memref<?xf32>,
                                 %indices: memref<?xi32>) {
    %c3 = arith.constant 3 : index
    %output_tensor = bufferization.to_tensor %output : memref<?xf32>
    %index_tensor = bufferization.to_tensor %indices : memref<?xi32>
    %result:2 = affine.for %i = 0 to 2
        iter_args(%out = %output_tensor, %idx = %index_tensor)
        -> (tensor<?xf32>, tensor<?xi32>) {
      %scratch_i = memref.alloca(%c3) : memref<?xi32>
      %scratch_i_tensor = bufferization.to_tensor %scratch_i : memref<?xi32>
      %scratch_f = memref.alloca(%c3) : memref<?xf32>
      %scratch_f_tensor = bufferization.to_tensor %scratch_f : memref<?xf32>
      %out_view = polygeist.submap(%out, %i, %c3) {map = #map} :
          (tensor<?xf32>, index, index) -> tensor<?xf32>
      %copied_out = linalg.generic {
          indexing_maps = [#identity, #identity],
          iterator_types = ["parallel"]}
          ins(%scratch_f_tensor : tensor<?xf32>)
          outs(%out_view : tensor<?xf32>) {
      ^bb0(%in: f32, %old: f32):
        linalg.yield %in : f32
      } -> tensor<?xf32>
      %new_out = polygeist.submapInverse(%out, %copied_out, %i, %c3)
          {map = #map} : (tensor<?xf32>, tensor<?xf32>, index, index)
          -> tensor<?xf32>
      %idx_view = polygeist.submap(%idx, %i, %c3) {map = #map} :
          (tensor<?xi32>, index, index) -> tensor<?xi32>
      %copied_idx = linalg.generic {
          indexing_maps = [#identity, #identity],
          iterator_types = ["parallel"]}
          ins(%scratch_i_tensor : tensor<?xi32>)
          outs(%idx_view : tensor<?xi32>) {
      ^bb0(%in: i32, %old: i32):
        linalg.yield %in : i32
      } -> tensor<?xi32>
      %new_idx = polygeist.submapInverse(%idx, %copied_idx, %i, %c3)
          {map = #map} : (tensor<?xi32>, tensor<?xi32>, index, index)
          -> tensor<?xi32>
      affine.yield %new_out, %new_idx : tensor<?xf32>, tensor<?xi32>
    }
    return
  }
}

// CHECK-LABEL: func.func @aten_max_pool1d_cpu
// CHECK: affine.for
// CHECK-NOT: kernel.launch @thrustUpsample{{Trilinear}}Backward3DHalfPixel_f32_memref
