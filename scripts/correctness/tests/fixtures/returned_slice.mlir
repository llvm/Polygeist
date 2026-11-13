#map = affine_map<(d0)[s0] -> (s0, d0 + 1)>
#map1 = affine_map<(d0) -> (d0)>
#map2 = affine_map<(d0) -> ()>
#map3 = affine_map<(d0)[s0] -> (s0, d0)>
module {
  func.func @row_transform(%arg0: memref<?x5xf32>, %arg1: memref<?x5xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg1 : memref<?x5xf32>
    %1 = bufferization.to_tensor %arg0 : memref<?x5xf32>
    %2 = affine.for %arg2 = 0 to 3 iter_args(%arg3 = %0) -> (tensor<?x5xf32>) {
      %extracted = tensor.extract %1[%arg2, %c0] : tensor<?x5xf32>
      %alloca = memref.alloca() : memref<f32>
      %4 = bufferization.to_tensor %alloca : memref<f32>
      %inserted = tensor.insert %extracted into %4[] : tensor<f32>
      %5 = polygeist.submap(%1, %arg2, %c4) {map = #map} : (tensor<?x5xf32>, index, index) -> tensor<?xf32>
      %6 = linalg.generic {doc = "", indexing_maps = [#map1, #map2], iterator_types = ["reduction"], library_call = ""} ins(%5 : tensor<?xf32>) outs(%inserted : tensor<f32>) {
      ^bb0(%in: f32, %out: f32):
        %15 = arith.cmpf ogt, %in, %out : f32
        %16 = arith.select %15, %in, %out : f32
        linalg.yield %16 : f32
      } -> tensor<f32>
      %extracted_0 = tensor.extract %6[] : tensor<f32>
      %alloca_1 = memref.alloca() : memref<f32>
      %7 = bufferization.to_tensor %alloca_1 : memref<f32>
      %inserted_2 = tensor.insert %cst into %7[] : tensor<f32>
      %8 = polygeist.submap(%1, %arg2, %c5) {map = #map3} : (tensor<?x5xf32>, index, index) -> tensor<?xf32>
      %9 = polygeist.submap(%arg3, %arg2, %c5) {map = #map3} : (tensor<?x5xf32>, index, index) -> tensor<?xf32>
      %10:2 = linalg.generic {doc = "", indexing_maps = [#map1, #map1, #map2], iterator_types = ["reduction"], library_call = ""} ins(%8 : tensor<?xf32>) outs(%9, %inserted_2 : tensor<?xf32>, tensor<f32>) {
      ^bb0(%in: f32, %out: f32, %out_4: f32):
        %15 = arith.subf %in, %extracted_0 : f32
        %16 = math.exp %15 : f32
        %17 = arith.addf %out_4, %16 : f32
        linalg.yield %16, %17 : f32, f32
      } -> (tensor<?xf32>, tensor<f32>)
      %11 = polygeist.submapInverse(%arg3, %10#0, %arg2, %c5) {map = #map3} : (tensor<?x5xf32>, tensor<?xf32>, index, index) -> tensor<?x5xf32>
      %extracted_3 = tensor.extract %10#1[] : tensor<f32>
      %12 = polygeist.submap(%11, %arg2, %c5) {map = #map3} : (tensor<?x5xf32>, index, index) -> tensor<?xf32>
      %13 = linalg.generic {doc = "", indexing_maps = [#map1], iterator_types = ["parallel"], library_call = ""} outs(%12 : tensor<?xf32>) {
      ^bb0(%out: f32):
        %15 = arith.divf %out, %extracted_3 : f32
        linalg.yield %15 : f32
      } -> tensor<?xf32>
      %14 = polygeist.submapInverse(%11, %13, %arg2, %c5) {map = #map3} : (tensor<?x5xf32>, tensor<?xf32>, index, index) -> tensor<?x5xf32>
      affine.yield %14 : tensor<?x5xf32>
    }
    %3 = bufferization.to_memref %2 : memref<?x5xf32>
    memref.copy %3, %arg1 : memref<?x5xf32> to memref<?x5xf32>
    return
  }
}

