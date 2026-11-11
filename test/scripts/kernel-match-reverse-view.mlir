// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | sed '/^\/\/ CHECK/d' | FileCheck %s

#map = affine_map<(d0, d1) -> (-d0 + 31, -d1 + 63)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @reverse_axes(%arg0: memref<?x64xf32>, %arg1: memref<?x64xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %0 = bufferization.to_tensor %arg0 : memref<?x64xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?x64xf32>
    %2 = polygeist.submap(%0, %c32, %c64) {map = #map} : (tensor<?x64xf32>, index, index) -> tensor<?x?xf32>
    %extracted_slice = tensor.extract_slice %1[0, 0] [%c32, %c64] [1, 1] : tensor<?x64xf32> to tensor<?x?xf32>
    %3 = linalg.generic {doc = "", indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"], library_call = ""} ins(%2 : tensor<?x?xf32>) outs(%extracted_slice : tensor<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<?x?xf32>
    %inserted_slice = tensor.insert_slice %3 into %1[0, 0] [%c32, %c64] [1, 1] : tensor<?x?xf32> into tensor<?x64xf32>
    %4 = bufferization.to_memref %inserted_slice : memref<?x64xf32>
    memref.copy %4, %arg1 : memref<?x64xf32> to memref<?x64xf32>
    return
  }
}

// CHECK-NOT: kernel.launch
// CHECK: linalg.generic
// CHECK-NOT: kernel.launch
