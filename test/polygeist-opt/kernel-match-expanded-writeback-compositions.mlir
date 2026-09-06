// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#pool_out = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#pool_window = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#avg_input = affine_map<(d0, d1, d2, d3, d4) -> (d2 + d0 * 9 + d1 * 3)>
#avg_output = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 7 + d0 * 42)>
#avg_identity = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#twice = affine_map<(d0) -> (d0 * 2)>
#twice_plus_two = affine_map<(d0) -> (d0 * 2 + 2)>

module {
  func.func @count_nonzero_expanded(%arg0: memref<?x64xf32>, %arg1: memref<?xi32>) {
    %zero = arith.constant 0 : i32
    %zero_f = arith.constant 0.0 : f32
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %input = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %output = bufferization.to_tensor %arg1 : memref<?xi32>
    %output_slice = polygeist.submap(%output, %c32) {map = #map} : (tensor<?xi32>, index) -> tensor<?xi32>
    %initialized = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%output_slice : tensor<?xi32>) {
    ^bb0(%out: i32):
      linalg.yield %zero : i32
    } -> tensor<?xi32>
    %initialized_output = polygeist.submapInverse(%output, %initialized, %c32) {map = #map} : (tensor<?xi32>, tensor<?xi32>, index) -> tensor<?xi32>
    %matrix = polygeist.submap(%input, %c32, %c64) {map = #map1} : (tensor<?x64xf32>, index, index) -> tensor<?x?xf32>
    %expanded_output = polygeist.submap(%initialized_output, %c32, %c64) {map = #map2} : (tensor<?xi32>, index, index) -> tensor<?x?xi32>
    %reduced = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%matrix : tensor<?x?xf32>) outs(%expanded_output : tensor<?x?xi32>) {
    ^bb0(%in: f32, %out: i32):
      %nz = arith.cmpf une, %in, %zero_f : f32
      %one = arith.extui %nz : i1 to i32
      %sum = arith.addi %out, %one : i32
      linalg.yield %sum : i32
    } -> tensor<?x?xi32>
    %written = polygeist.submapInverse(%initialized_output, %reduced, %c32, %c64) {map = #map2} : (tensor<?xi32>, tensor<?x?xi32>, index, index) -> tensor<?xi32>
    %result = bufferization.to_memref %written : memref<?xi32>
    memref.copy %result, %arg1 : memref<?xi32> to memref<?xi32>
    return
  }

  func.func @max_pool_expanded(%arg0: memref<?x8x16x16xf32>, %arg1: memref<?x8x8x8xf32>) {
    %neg_inf = arith.constant -3.40282347E+38 : f32
    %c8 = arith.constant 8 : index
    %c2 = arith.constant 2 : index
    %input = bufferization.to_tensor %arg0 : memref<?x8x16x16xf32>
    %output = bufferization.to_tensor %arg1 : memref<?x8x8x8xf32>
    %output_slice = polygeist.submap(%output, %c2, %c8, %c8, %c8) {map = #pool_out} : (tensor<?x8x8x8xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %initialized = linalg.generic {indexing_maps = [#pool_out], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%output_slice : tensor<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %neg_inf : f32
    } -> tensor<?x?x?x?xf32>
    %initialized_output = polygeist.submapInverse(%output, %initialized, %c2, %c8, %c8, %c8) {map = #pool_out} : (tensor<?x8x8x8xf32>, tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x8x8x8xf32>
    %windows = polygeist.submap(%input, %c2, %c8, %c8, %c8, %c2, %c2) {map = #pool_window} : (tensor<?x8x16x16xf32>, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?xf32>
    %expanded_output = polygeist.submap(%initialized_output, %c2, %c8, %c8, %c8, %c2, %c2) {map = #pool_window} : (tensor<?x8x8x8xf32>, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?xf32>
    %reduced = linalg.generic {indexing_maps = [#pool_window, #pool_window], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%windows : tensor<?x?x?x?x?x?xf32>) outs(%expanded_output : tensor<?x?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %gt = arith.cmpf ogt, %in, %out : f32
      %max = arith.select %gt, %in, %out : f32
      linalg.yield %max : f32
    } -> tensor<?x?x?x?x?x?xf32>
    %written = polygeist.submapInverse(%initialized_output, %reduced, %c2, %c8, %c8, %c8, %c2, %c2) {map = #pool_window} : (tensor<?x8x8x8xf32>, tensor<?x?x?x?x?x?xf32>, index, index, index, index, index, index) -> tensor<?x8x8x8xf32>
    %result = bufferization.to_memref %written : memref<?x8x8x8xf32>
    memref.copy %result, %arg1 : memref<?x8x8x8xf32> to memref<?x8x8x8xf32>
    return
  }

  func.func @cumprod_expanded(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>) {
    %one = arith.constant 1.0 : f32
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %input = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %output = bufferization.to_tensor %arg1 : memref<?x64xf32>
    %final_storage = tensor.empty(%c32) : tensor<?xf32>
    %final_slice = polygeist.submap(%final_storage, %c32) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %initialized = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%final_slice : tensor<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %one : f32
    } -> tensor<?xf32>
    %initialized_final = polygeist.submapInverse(%final_storage, %initialized, %c32) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    %input_matrix = polygeist.submap(%input, %c32, %c64) {map = #map1} : (tensor<?x64xf32>, index, index) -> tensor<?x?xf32>
    %output_matrix = polygeist.submap(%output, %c32, %c64) {map = #map1} : (tensor<?x64xf32>, index, index) -> tensor<?x?xf32>
    %expanded_final = polygeist.submap(%initialized_final, %c32, %c64) {map = #map2} : (tensor<?xf32>, index, index) -> tensor<?x?xf32>
    %scan:2 = linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%input_matrix : tensor<?x?xf32>) outs(%output_matrix, %expanded_final : tensor<?x?xf32>, tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32, %out_0: f32):
      %product = arith.mulf %out_0, %in : f32
      linalg.yield %product, %product : f32, f32
    } -> (tensor<?x?xf32>, tensor<?x?xf32>)
    %written = polygeist.submapInverse(%output, %scan#0, %c32, %c64) {map = #map1} : (tensor<?x64xf32>, tensor<?x?xf32>, index, index) -> tensor<?x64xf32>
    %result = bufferization.to_memref %written : memref<?x64xf32>
    memref.copy %result, %arg1 : memref<?x64xf32> to memref<?x64xf32>
    return
  }

  func.func @aten_sum(%arg0: memref<?x64xf64>, %arg1: memref<?xf64>) {
    %zero = arith.constant 0.0 : f64
    %c64 = arith.constant 64 : index
    %c65536 = arith.constant 65536 : index
    %input = bufferization.to_tensor %arg0 : memref<?x64xf64>
    %output = bufferization.to_tensor %arg1 : memref<?xf64>
    %output_slice = polygeist.submap(%output, %c65536) {map = #map} : (tensor<?xf64>, index) -> tensor<?xf64>
    %initialized = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%output_slice : tensor<?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %zero : f64
    } -> tensor<?xf64>
    %initialized_output = polygeist.submapInverse(%output, %initialized, %c65536) {map = #map} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %matrix = polygeist.submap(%input, %c65536, %c64) {map = #map1} : (tensor<?x64xf64>, index, index) -> tensor<?x?xf64>
    %expanded_output = polygeist.submap(%initialized_output, %c65536, %c64) {map = #map2} : (tensor<?xf64>, index, index) -> tensor<?x?xf64>
    %reduced = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "reduction"]} ins(%matrix : tensor<?x?xf64>) outs(%expanded_output : tensor<?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %sum = arith.addf %out, %in : f64
      linalg.yield %sum : f64
    } -> tensor<?x?xf64>
    %written = polygeist.submapInverse(%initialized_output, %reduced, %c65536, %c64) {map = #map2} : (tensor<?xf64>, tensor<?x?xf64>, index, index) -> tensor<?xf64>
    %result = bufferization.to_memref %written : memref<?xf64>
    memref.copy %result, %arg1 : memref<?xf64> to memref<?xf64>
    return
  }

  func.func @aten_avg_pool2d_backward_cpu(%arg0: memref<?xf32>, %arg1: memref<?xf32>) {
    %zero = arith.constant 0.0 : f32
    %four = arith.constant 4.0 : f32
    %c6 = arith.constant 6 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c84 = arith.constant 84 : index
    %input = bufferization.to_tensor %arg0 : memref<?xf32>
    %output = bufferization.to_tensor %arg1 : memref<?xf32>
    %flat_output = polygeist.submap(%output, %c84) {map = #map} : (tensor<?xf32>, index) -> tensor<?xf32>
    %initialized = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%flat_output : tensor<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %zero : f32
    } -> tensor<?xf32>
    %initialized_output = polygeist.submapInverse(%output, %initialized, %c84) {map = #map} : (tensor<?xf32>, tensor<?xf32>, index) -> tensor<?xf32>
    %grad_windows = polygeist.submap(%input, %c2, %c3, %c3, %c6, %c6) {map = #avg_input} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %output_windows = polygeist.submap(%initialized_output, %c2, %c3, %c3, %c6, %c6) {map = #avg_output} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %distributed = linalg.generic {indexing_maps = [#avg_identity, #avg_identity], iterator_types = ["parallel", "reduction", "reduction", "parallel", "parallel"]} ins(%grad_windows : tensor<?x?x?x?x?xf32>) outs(%output_windows : tensor<?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %row = linalg.index 2 : index
      %addend = arith.divf %in, %four : f32
      %sum = arith.addf %out, %addend : f32
      %input_row = linalg.index 4 : index
      %begin = affine.apply #twice(%row)
      %after_begin = arith.cmpi sge, %input_row, %begin : index
      %end = affine.apply #twice_plus_two(%row)
      %before_end = arith.cmpi slt, %input_row, %end : index
      %inside = arith.andi %after_begin, %before_end : i1
      %value = arith.select %inside, %sum, %out : f32
      linalg.yield %value : f32
    } -> tensor<?x?x?x?x?xf32>
    %written = polygeist.submapInverse(%initialized_output, %distributed, %c2, %c3, %c3, %c6, %c6) {map = #avg_output} : (tensor<?xf32>, tensor<?x?x?x?x?xf32>, index, index, index, index, index) -> tensor<?xf32>
    %result = bufferization.to_memref %written : memref<?xf32>
    memref.copy %result, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}

// CHECK-LABEL: func.func @count_nonzero_expanded
// CHECK: %written = kernel.launch @cubSegmentedCountNonzero2D_f32_tensor
// CHECK-SAME: -> tensor<?xi32>
// CHECK-NEXT: %result = bufferization.to_memref %written
// CHECK-LABEL: func.func @max_pool_expanded
// CHECK: kernel.launch @cudnnMaxPoolFwd_batched
// CHECK-SAME: -> tensor<?x?x?x?xf32>
// CHECK: %written = tensor.cast
// CHECK-NEXT: %result = bufferization.to_memref %written
// CHECK-LABEL: func.func @cumprod_expanded
// CHECK: %scan:2 = kernel.launch @cubSegmentedInclusiveProduct2D_f32_tensor
// CHECK-SAME: -> (tensor<?x?xf32>, tensor<?xf32>)
// CHECK-NEXT: %written = polygeist.submapInverse
// CHECK-LABEL: func.func @aten_sum
// CHECK: kernel.launch @cubSegmentedSum_f64_memref
// CHECK-SAME: polygeist.fixed_extents = array<i64: 65536, 64>
// CHECK-NEXT: return
// CHECK-LABEL: func.func @aten_avg_pool2d_backward_cpu
// CHECK: kernel.launch @cudnnAveragePool_f32_flat2
// CHECK-NEXT: return
