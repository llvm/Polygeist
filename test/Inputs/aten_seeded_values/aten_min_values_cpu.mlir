#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @anonymous(%arg0: memref<?x64xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c63 = arith.constant 63 : index
    %subview = memref.subview %arg0[0, 0] [%c32, 1] [1, 1] : memref<?x64xf32> to memref<?xf32, strided<[64]>>
    %subview_0 = memref.subview %arg1[0] [%c32] [1] : memref<?xf32> to memref<?xf32, strided<[1]>>
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%subview : memref<?xf32, strided<[64]>>) outs(%subview_0 : memref<?xf32, strided<[1]>>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    %subview_1 = memref.subview %arg0[0, 1] [%c32, %c63] [1, 1] : memref<?x64xf32> to memref<?x?xf32, strided<[64, 1], offset: 1>>
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [%c32], strides: [1] : memref<?xf32> to memref<32xf32>
    linalg.generic {indexing_maps = [#map1, #map2], iterator_types = ["parallel", "reduction"]} ins(%subview_1 : memref<?x?xf32, strided<[64, 1], offset: 1>>) outs(%reinterpret_cast : memref<32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.cmpf olt, %in, %out : f32
      %1 = arith.select %0, %in, %out : f32
      linalg.yield %1 : f32
    }
    return
  }
}

