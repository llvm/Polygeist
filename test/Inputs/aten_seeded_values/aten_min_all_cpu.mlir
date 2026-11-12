#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @anonymous(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c4095 = arith.constant 4095 : index
    %0 = affine.load %arg0[0] : memref<?xf32>
    affine.store %0, %arg1[0] : memref<?xf32>
    %subview = memref.subview %arg0[1] [%c4095] [1] : memref<?xf32> to memref<?xf32, strided<[1], offset: 1>>
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [], strides: [] : memref<?xf32> to memref<f32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%subview : memref<?xf32, strided<[1], offset: 1>>) outs(%reinterpret_cast : memref<f32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.cmpf olt, %in, %out : f32
      %2 = arith.select %1, %in, %out : f32
      linalg.yield %2 : f32
    }
    return
  }
}

