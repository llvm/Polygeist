// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s > %t.matched.mlir
// RUN: sed '/^\/\/ CHECK/d' %t.matched.mlir | FileCheck %s
// RUN: /usr/bin/python3 %S/../../scripts/correctness/inject_kernel_library.py %t.matched.mlir %S/../../generic_solver/kernel_library_phase2.mlir -o %t.abi.mlir
// RUN: polygeist-opt %t.abi.mlir --lower-kernel-launch-to-cublas --canonicalize -o /dev/null

#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d0, d1, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d3, d4, d2)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>
module {
  func.func @anonymous_operation(%arg0: memref<?x8x16xf32>, %arg1: memref<?x16x24xf32>, %arg2: memref<?x8x24xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c16 = arith.constant 16 : index
    %c3 = arith.constant 3 : index
    %c24 = arith.constant 24 : index
    %c8 = arith.constant 8 : index
    %c30 = arith.constant 30 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x8x16xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x16x24xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?x8x24xf32>
    %extracted_slice = tensor.extract_slice %2[0, 0, 0] [%c30, %c8, %c24] [1, 1, 1] : tensor<?x8x24xf32> to tensor<?x?x?xf32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?x?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?x?x?xf32>
    %4 = polygeist.submap(%0, %c30, %c8, %c24, %c3, %c16) {map = #map1} : (tensor<?x8x16xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %extracted_slice_0 = tensor.extract_slice %1[0, 0, 0] [%c3, %c16, %c24] [1, 1, 1] : tensor<?x16x24xf32> to tensor<?x?x?xf32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction", "reduction"], library_call = ""} ins(%4, %extracted_slice_0 : tensor<?x?x?x?x?xf32>, tensor<?x?x?xf32>) outs(%3 : tensor<?x?x?xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %7 = arith.mulf %in, %in_1 : f32
      %8 = arith.addf %out, %7 : f32
      linalg.yield %8 : f32
    } -> tensor<?x?x?xf32>
    %inserted_slice = tensor.insert_slice %5 into %2[0, 0, 0] [%c30, %c8, %c24] [1, 1, 1] : tensor<?x?x?xf32> into tensor<?x8x24xf32>
    %6 = bufferization.to_memref %inserted_slice : memref<?x8x24xf32>
    memref.copy %6, %arg2 : memref<?x8x24xf32> to memref<?x8x24xf32>
    return
  }
}


// CHECK: kernel.launch @cudnnConvolutionTBC_f32_memref
