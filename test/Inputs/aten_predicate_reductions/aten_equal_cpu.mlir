#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @anonymous(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %false = arith.constant false
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    affine.store %c1_i32, %arg2[0] : memref<?xi32>
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [0], sizes: [], strides: [] : memref<?xi32> to memref<i32>
    linalg.generic {indexing_maps = [#map, #map, #map1], iterator_types = ["reduction"]} ins(%arg0, %arg1 : memref<?xf32>, memref<?xf32>) outs(%reinterpret_cast : memref<i32>) {
    ^bb0(%in: f32, %in_0: f32, %out: i32):
      %0 = arith.cmpi ne, %out, %c0_i32 : i32
      %1 = arith.cmpf oeq, %in, %in_0 : f32
      %2 = arith.select %0, %1, %false : i1
      %3 = arith.extui %2 : i1 to i32
      linalg.yield %3 : i32
    }
    return
  }
}

