#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module {
  func.func @anonymous(%arg0: memref<?xf32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0_i32 = arith.constant 0 : i32
    affine.store %c0_i32, %arg1[0] : memref<?xi32>
    %reinterpret_cast = memref.reinterpret_cast %arg1 to offset: [0], sizes: [], strides: [] : memref<?xi32> to memref<i32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["reduction"]} ins(%arg0 : memref<?xf32>) outs(%reinterpret_cast : memref<i32>) {
    ^bb0(%in: f32, %out: i32):
      %0 = arith.cmpf une, %in, %cst : f32
      %1 = arith.extui %0 : i1 to i32
      %2 = arith.addi %out, %1 : i32
      linalg.yield %2 : i32
    }
    return
  }
}

