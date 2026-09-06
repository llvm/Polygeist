// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | sed '/^\/\/ CHECK/d' | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d2 + d0 * 9 + d1 * 3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 7 + d0 * 42)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map8 = affine_map<(d0) -> (d0 * 2)>
#map9 = affine_map<(d0) -> (d0 * 2 + 2)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d2 + d0 * 32, d3 + d1 * 32)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map14 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d0, d1, d4)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>
#map17 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map19 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map20 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map21 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#map22 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d5 + d2, d6 + d3)>
#map23 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>

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
    %output_slice = polygeist.submap(%output, %c2, %c8, %c8, %c8) {map = #map3} : (tensor<?x8x8x8xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %initialized = linalg.generic {indexing_maps = [#map3], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%output_slice : tensor<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %neg_inf : f32
    } -> tensor<?x?x?x?xf32>
    %initialized_output = polygeist.submapInverse(%output, %initialized, %c2, %c8, %c8, %c8) {map = #map3} : (tensor<?x8x8x8xf32>, tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x8x8x8xf32>
    %windows = polygeist.submap(%input, %c2, %c8, %c8, %c8, %c2, %c2) {map = #map4} : (tensor<?x8x16x16xf32>, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?xf32>
    %expanded_output = polygeist.submap(%initialized_output, %c2, %c8, %c8, %c8, %c2, %c2) {map = #map4} : (tensor<?x8x8x8xf32>, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?xf32>
    %reduced = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%windows : tensor<?x?x?x?x?x?xf32>) outs(%expanded_output : tensor<?x?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %gt = arith.cmpf ogt, %in, %out : f32
      %max = arith.select %gt, %in, %out : f32
      linalg.yield %max : f32
    } -> tensor<?x?x?x?x?x?xf32>
    %written = polygeist.submapInverse(%initialized_output, %reduced, %c2, %c8, %c8, %c8, %c2, %c2) {map = #map4} : (tensor<?x8x8x8xf32>, tensor<?x?x?x?x?x?xf32>, index, index, index, index, index, index) -> tensor<?x8x8x8xf32>
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
    %grad_windows = polygeist.submap(%input, %c2, %c3, %c3, %c6, %c6) {map = #map5} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %output_windows = polygeist.submap(%initialized_output, %c2, %c3, %c3, %c6, %c6) {map = #map6} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %distributed = linalg.generic {indexing_maps = [#map7, #map7], iterator_types = ["parallel", "reduction", "reduction", "parallel", "parallel"]} ins(%grad_windows : tensor<?x?x?x?x?xf32>) outs(%output_windows : tensor<?x?x?x?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %row = linalg.index 2 : index
      %addend = arith.divf %in, %four : f32
      %sum = arith.addf %out, %addend : f32
      %input_row = linalg.index 4 : index
      %begin = affine.apply #map8(%row)
      %after_begin = arith.cmpi sge, %input_row, %begin : index
      %end = affine.apply #map9(%row)
      %before_end = arith.cmpi slt, %input_row, %end : index
      %inside = arith.andi %after_begin, %before_end : i1
      %value = arith.select %inside, %sum, %out : f32
      linalg.yield %value : f32
    } -> tensor<?x?x?x?x?xf32>
    %written = polygeist.submapInverse(%initialized_output, %distributed, %c2, %c3, %c3, %c6, %c6) {map = #map6} : (tensor<?xf32>, tensor<?x?x?x?x?xf32>, index, index, index, index, index) -> tensor<?xf32>
    %result = bufferization.to_memref %written : memref<?xf32>
    memref.copy %result, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }

  func.func @kron_expanded(%arg0: memref<?x128xf32>, %arg1: memref<?x32xf32>, %arg2: memref<?x4096xf32>) {
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %x = bufferization.to_tensor %arg0 : memref<?x128xf32>
    %y = bufferization.to_tensor %arg1 : memref<?x32xf32>
    %out = bufferization.to_tensor %arg2 : memref<?x4096xf32>
    %expanded_x = polygeist.submap(%x, %c256, %c128, %c32, %c32) {map = #map10} : (tensor<?x128xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %expanded_y = polygeist.submap(%y, %c256, %c128, %c32, %c32) {map = #map11} : (tensor<?x32xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %expanded_out = polygeist.submap(%out, %c256, %c128, %c32, %c32) {map = #map12} : (tensor<?x4096xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %product = linalg.generic {indexing_maps = [#map13, #map13, #map13], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_x, %expanded_y : tensor<?x?x?x?xf32>, tensor<?x?x?x?xf32>) outs(%expanded_out : tensor<?x?x?x?xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %value = arith.mulf %lhs, %rhs : f32
      linalg.yield %value : f32
    } -> tensor<?x?x?x?xf32>
    %written = polygeist.submapInverse(%out, %product, %c256, %c128, %c32, %c32) {map = #map12} : (tensor<?x4096xf32>, tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x4096xf32>
    %result = bufferization.to_memref %written : memref<?x4096xf32>
    memref.copy %result, %arg2 : memref<?x4096xf32> to memref<?x4096xf32>
    return
  }

  func.func @aten_conv_tbc_cpu(%arg0: memref<?x16x32xf32>, %arg1: memref<?x32x64xf32>, %arg2: memref<?x16x64xf32>) {
    %zero = arith.constant 0.0 : f32
    %c32 = arith.constant 32 : index
    %c3 = arith.constant 3 : index
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c4094 = arith.constant 4094 : index
    %input = bufferization.to_tensor %arg0 : memref<?x16x32xf32>
    %filter = bufferization.to_tensor %arg1 : memref<?x32x64xf32>
    %output = bufferization.to_tensor %arg2 : memref<?x16x64xf32>
    %output_view = polygeist.submap(%output, %c4094, %c16, %c64) {map = #map14} : (tensor<?x16x64xf32>, index, index, index) -> tensor<?x?x?xf32>
    %initialized = linalg.generic {indexing_maps = [#map14], iterator_types = ["parallel", "parallel", "parallel"]} outs(%output_view : tensor<?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %zero : f32
    } -> tensor<?x?x?xf32>
    %initialized_output = polygeist.submapInverse(%output, %initialized, %c4094, %c16, %c64) {map = #map14} : (tensor<?x16x64xf32>, tensor<?x?x?xf32>, index, index, index) -> tensor<?x16x64xf32>
    %input_windows = polygeist.submap(%input, %c4094, %c16, %c64, %c3, %c32) {map = #map15} : (tensor<?x16x32xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %filter_view = polygeist.submap(%filter, %c4094, %c16, %c64, %c3, %c32) {map = #map16} : (tensor<?x32x64xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %expanded_output = polygeist.submap(%initialized_output, %c4094, %c16, %c64, %c3, %c32) {map = #map17} : (tensor<?x16x64xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %convolution = linalg.generic {indexing_maps = [#map18, #map18, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%input_windows, %filter_view : tensor<?x?x?x?x?xf32>, tensor<?x?x?x?x?xf32>) outs(%expanded_output : tensor<?x?x?x?x?xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %out, %product : f32
      linalg.yield %sum : f32
    } -> tensor<?x?x?x?x?xf32>
    %written = polygeist.submapInverse(%initialized_output, %convolution, %c4094, %c16, %c64, %c3, %c32) {map = #map17} : (tensor<?x16x64xf32>, tensor<?x?x?x?x?xf32>, index, index, index, index, index) -> tensor<?x16x64xf32>
    %result = bufferization.to_memref %written : memref<?x16x64xf32>
    memref.copy %result, %arg2 : memref<?x16x64xf32> to memref<?x16x64xf32>
    return
  }

  func.func @aten_conv_transpose2d(%arg0: memref<?x16x128x128xf32>, %arg1: memref<?x32x3x3xf32>, %arg2: memref<?x32x130x130xf32>) {
    %zero = arith.constant 0.0 : f32
    %c3 = arith.constant 3 : index
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %c130 = arith.constant 130 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %input = bufferization.to_tensor %arg0 : memref<?x16x128x128xf32>
    %filter = bufferization.to_tensor %arg1 : memref<?x32x3x3xf32>
    %output = bufferization.to_tensor %arg2 : memref<?x32x130x130xf32>
    %output_view = polygeist.submap(%output, %c2, %c32, %c130, %c130) {map = #map19} : (tensor<?x32x130x130xf32>, index, index, index, index) -> tensor<?x?x?x?xf32>
    %initialized = linalg.generic {indexing_maps = [#map19], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} outs(%output_view : tensor<?x?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %zero : f32
    } -> tensor<?x?x?x?xf32>
    %initialized_output = polygeist.submapInverse(%output, %initialized, %c2, %c32, %c130, %c130) {map = #map19} : (tensor<?x32x130x130xf32>, tensor<?x?x?x?xf32>, index, index, index, index) -> tensor<?x32x130x130xf32>
    %input_view = polygeist.submap(%input, %c2, %c16, %c128, %c128, %c32, %c3, %c3) {map = #map20} : (tensor<?x16x128x128xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf32>
    %filter_view = polygeist.submap(%filter, %c2, %c16, %c128, %c128, %c32, %c3, %c3) {map = #map21} : (tensor<?x32x3x3xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf32>
    %expanded_output = polygeist.submap(%initialized_output, %c2, %c16, %c128, %c128, %c32, %c3, %c3) {map = #map22} : (tensor<?x32x130x130xf32>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf32>
    %convolution = linalg.generic {indexing_maps = [#map23, #map23, #map23], iterator_types = ["parallel", "reduction", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%input_view, %filter_view : tensor<?x?x?x?x?x?x?xf32>, tensor<?x?x?x?x?x?x?xf32>) outs(%expanded_output : tensor<?x?x?x?x?x?x?xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %product = arith.mulf %lhs, %rhs : f32
      %sum = arith.addf %out, %product : f32
      linalg.yield %sum : f32
    } -> tensor<?x?x?x?x?x?x?xf32>
    %written = polygeist.submapInverse(%initialized_output, %convolution, %c2, %c16, %c128, %c128, %c32, %c3, %c3) {map = #map22} : (tensor<?x32x130x130xf32>, tensor<?x?x?x?x?x?x?xf32>, index, index, index, index, index, index, index) -> tensor<?x32x130x130xf32>
    %result = bufferization.to_memref %written : memref<?x32x130x130xf32>
    memref.copy %result, %arg2 : memref<?x32x130x130xf32> to memref<?x32x130x130xf32>
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
// CHECK-LABEL: func.func @kron_expanded
// CHECK: kernel.launch @cutensorKroneckerProduct2D_f32_memref
// CHECK-NEXT: return
// CHECK-LABEL: func.func @aten_conv_tbc_cpu
// CHECK: kernel.launch @cudnnConvolutionTBC_f32_memref
// CHECK-NEXT: return
// CHECK-LABEL: func.func @aten_conv_transpose2d
// CHECK: kernel.launch @cudnnConvolutionTranspose2D_f32_memref
// CHECK-NEXT: return
