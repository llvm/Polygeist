#map = affine_map<(d0) -> (d0)>
module {
  func.func @anonymous(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: f32, %arg4: f32, %arg5: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = bufferization.to_tensor %arg2 : memref<?xf32>
    %3 = bufferization.to_tensor %arg5 : memref<?xf32>
    %4 = arith.cmpf oeq, %arg3, %cst : f32
    %5 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel"], library_call = ""} ins(%1, %2, %0 : tensor<?xf32>, tensor<?xf32>, tensor<?xf32>) outs(%3 : tensor<?xf32>) {
    ^bb0(%in: f32, %in_0: f32, %in_1: f32, %out: f32):
      %7 = arith.mulf %arg4, %in : f32
      %8 = arith.mulf %7, %in_0 : f32
      %9 = arith.mulf %arg3, %in_1 : f32
      %10 = arith.mulf %arg4, %in : f32
      %11 = arith.mulf %10, %in_0 : f32
      %12 = arith.addf %9, %11 : f32
      %13 = arith.select %4, %8, %12 : f32
      linalg.yield %13 : f32
    } -> tensor<?xf32>
    %6 = bufferization.to_memref %5 : memref<?xf32>
    memref.copy %6, %arg5 : memref<?xf32> to memref<?xf32>
    return
  }
}

