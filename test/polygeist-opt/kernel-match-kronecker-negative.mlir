// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | FileCheck %s

#wrong_output = affine_map<(d0, d1, d2, d3) ->
    (d2 + d0 * 8, d3 + d1 * 12)>
#lhs = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#rhs = affine_map<(d0, d1, d2, d3) -> (d2, d3)>
#output = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

module {
  func.func @wrong_physical_flattening(
      %x: memref<?x12xf32>, %y: memref<?x10xf32>,
      %output_memref: memref<?x120xf32>) {
    %c10 = arith.constant 10 : index
    %c8 = arith.constant 8 : index
    %c12 = arith.constant 12 : index
    %c16 = arith.constant 16 : index
    %x_tensor = bufferization.to_tensor %x : memref<?x12xf32>
    %y_tensor = bufferization.to_tensor %y : memref<?x10xf32>
    %output_tensor = bufferization.to_tensor %output_memref
        : memref<?x120xf32>
    %x_slice = tensor.extract_slice %x_tensor[0, 0] [%c16, %c12] [1, 1]
        : tensor<?x12xf32> to tensor<?x?xf32>
    %y_slice = tensor.extract_slice %y_tensor[0, 0] [%c8, %c10] [1, 1]
        : tensor<?x10xf32> to tensor<?x?xf32>
    %output_view = polygeist.submap(
        %output_tensor, %c16, %c12, %c8, %c10) {map = #wrong_output}
        : (tensor<?x120xf32>, index, index, index, index)
          -> tensor<?x?x?x?xf32>
    %product = linalg.generic {
        indexing_maps = [#lhs, #rhs, #output],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%x_slice, %y_slice : tensor<?x?xf32>, tensor<?x?xf32>)
        outs(%output_view : tensor<?x?x?x?xf32>) {
    ^bb0(%x_value: f32, %y_value: f32, %old: f32):
      %value = arith.mulf %x_value, %y_value : f32
      linalg.yield %value : f32
    } -> tensor<?x?x?x?xf32>
    %updated = polygeist.submapInverse(
        %output_tensor, %product, %c16, %c12, %c8, %c10)
        {map = #wrong_output}
        : (tensor<?x120xf32>, tensor<?x?x?x?xf32>, index, index,
           index, index) -> tensor<?x120xf32>
    %updated_memref = bufferization.to_memref %updated : memref<?x120xf32>
    memref.copy %updated_memref, %output_memref
        : memref<?x120xf32> to memref<?x120xf32>
    return
  }
}

// CHECK-LABEL: func.func @wrong_physical_flattening
// CHECK-NOT: kernel.launch @cutensorKroneckerProduct2D_f32_memref
// CHECK: linalg.generic
// CHECK: polygeist.submapInverse
// CHECK: memref.copy
