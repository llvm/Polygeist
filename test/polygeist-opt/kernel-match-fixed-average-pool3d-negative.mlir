// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | FileCheck %s

#identity5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
// Deliberately wrong: spatial stride is one, not the fixed stride two that
// the cuDNN operation tag and physical extents require.
#wrong_window = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) ->
    (d0, d1, d5 + d2, d6 + d3, d7 + d4)>
#identity8 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) ->
    (d0, d1, d2, d3, d4, d5, d6, d7)>
#project5 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) ->
    (d0, d1, d2, d3, d4)>

module {
  func.func @wrong_stride(
      %input: memref<?x3x8x8x8xf32>,
      %output: memref<?x3x4x4x4xf32>) {
    %zero = arith.constant 0.0 : f32
    %eight = arith.constant 8.0 : f32
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %in = bufferization.to_tensor %input : memref<?x3x8x8x8xf32>
    %out = bufferization.to_tensor %output : memref<?x3x4x4x4xf32>
    %slice = tensor.extract_slice %out[0, 0, 0, 0, 0]
        [%c2, %c3, %c4, %c4, %c4] [1, 1, 1, 1, 1]
        : tensor<?x3x4x4x4xf32> to tensor<?x?x?x?x?xf32>
    %init = linalg.generic {
        indexing_maps = [#identity5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel",
                          "parallel"]}
        outs(%slice : tensor<?x?x?x?x?xf32>) {
    ^bb0(%old: f32):
      linalg.yield %zero : f32
    } -> tensor<?x?x?x?x?xf32>
    %windows = polygeist.submap(
        %in, %c2, %c3, %c4, %c4, %c4, %c2, %c2, %c2)
        {map = #wrong_window}
        : (tensor<?x3x8x8x8xf32>, index, index, index, index, index,
           index, index, index) -> tensor<?x?x?x?x?x?x?x?xf32>
    %pooled = linalg.generic {
        indexing_maps = [#identity8, #project5],
        iterator_types = ["parallel", "parallel", "parallel", "parallel",
                          "parallel", "reduction", "reduction", "reduction"]}
        ins(%windows : tensor<?x?x?x?x?x?x?x?xf32>)
        outs(%init : tensor<?x?x?x?x?xf32>) {
    ^bb0(%value: f32, %sum: f32):
      %scaled = arith.divf %value, %eight : f32
      %next = arith.addf %sum, %scaled : f32
      linalg.yield %next : f32
    } -> tensor<?x?x?x?x?xf32>
    %updated = tensor.insert_slice %pooled into %out[0, 0, 0, 0, 0]
        [%c2, %c3, %c4, %c4, %c4] [1, 1, 1, 1, 1]
        : tensor<?x?x?x?x?xf32> into tensor<?x3x4x4x4xf32>
    %result = bufferization.to_memref %updated : memref<?x3x4x4x4xf32>
    memref.copy %result, %output
        : memref<?x3x4x4x4xf32> to memref<?x3x4x4x4xf32>
    return
  }
}

// CHECK-LABEL: func.func @wrong_stride
// CHECK-NOT: kernel.launch @cudnnAveragePool_f32_r5
// CHECK: linalg.generic
// CHECK: linalg.generic
