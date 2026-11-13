// RUN: polygeist-opt %s --canonicalize -o /dev/null

#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d2 + d0 * 9 + d1 * 3)>
#map2 = affine_map<(d0, d1, d2) -> (d2 + d1 * 7 + d0 * 42)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4)>
#map5 = affine_map<(d0) -> (d0 * 2)>
#map6 = affine_map<(d0) -> (d0 * 2 + 2)>
module {
  func.func @masked_scatter(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 4.000000e+00 : f32
    %c6 = arith.constant 6 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf32>
    %1 = bufferization.to_tensor %arg1 : memref<?xf32>
    %2 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel"], library_call = ""} outs(%1 : tensor<?xf32>) {
    ^bb0(%out: f32):
      linalg.yield %cst : f32
    } -> tensor<?xf32>
    %3 = polygeist.submap(%0, %c2, %c3, %c3, %c6, %c6) {map = #map1} : (tensor<?xf32>, index, index, index, index, index) -> tensor<?x?x?x?x?xf32>
    %4 = polygeist.submap(%2, %c2, %c6, %c6) {map = #map2} : (tensor<?xf32>, index, index, index) -> tensor<2x6x6xf32>
    %5 = linalg.generic {doc = "", indexing_maps = [#map3, #map4], iterator_types = ["parallel", "reduction", "reduction", "parallel", "parallel"], library_call = ""} ins(%3 : tensor<?x?x?x?x?xf32>) outs(%4 : tensor<2x6x6xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = linalg.index 1 : index
      %9 = linalg.index 2 : index
      %10 = arith.divf %in, %cst_0 : f32
      %11 = arith.addf %out, %10 : f32
      %12 = linalg.index 4 : index
      %13 = affine.apply #map5(%9)
      %14 = arith.cmpi sge, %12, %13 : index
      %15 = affine.apply #map6(%9)
      %16 = arith.cmpi slt, %12, %15 : index
      %17 = arith.andi %14, %16 : i1
      %18 = arith.select %17, %11, %out : f32
      %19 = linalg.index 3 : index
      %20 = affine.apply #map5(%8)
      %21 = arith.cmpi sge, %19, %20 : index
      %22 = affine.apply #map6(%8)
      %23 = arith.cmpi slt, %19, %22 : index
      %24 = arith.andi %21, %23 : i1
      %25 = arith.select %24, %18, %out : f32
      linalg.yield %25 : f32
    } -> tensor<2x6x6xf32>
    %6 = polygeist.submapInverse(%2, %5, %c2, %c6, %c6) {map = #map2} : (tensor<?xf32>, tensor<2x6x6xf32>, index, index, index) -> tensor<?xf32>
    %7 = bufferization.to_memref %6 : memref<?xf32>
    memref.copy %7, %arg1 : memref<?xf32> to memref<?xf32>
    return
  }
}
