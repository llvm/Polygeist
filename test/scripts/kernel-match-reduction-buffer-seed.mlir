// RUN: /usr/bin/python3 %S/../../scripts/correctness/kernel_match_rewrite.py %s | sed '/^\/\/ CHECK/d' | FileCheck %s

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @seeded_extremum(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4095 = arith.constant 4095 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %extracted = tensor.extract %0[%c0] : tensor<?xf32>
    affine.store %extracted, %arg1[0] : memref<?xf32>
    %subview = memref.subview %arg0[1] [%c4095] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: 1>>
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [], strides: [] : memref<?xf32> to memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%subview : memref<?xf32, strided<[1], offset: 1>>) outs(%reinterpret_cast : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.cmpf ogt, %in, %out : f32
      %2 = arith.select %1, %in, %out : f32
      linalg.yield %2 : f32
    }
    return
  }
}

// CHECK-NOT: kernel.launch @cudnnReduceMax
// CHECK: linalg.generic
// CHECK-NOT: kernel.launch @cudnnReduceMax
