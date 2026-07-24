#map = affine_map<(d0, d1, d2, d3) -> (d0, d3, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 * 25 + d3 + d0 * 125 + d1 * 5)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 * 4 + d2)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d4 * 4 + d1)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d4, d2)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d0 * 64 + d1 * 16 + d2 * 4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_integrate_value_3d_scratch_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg2 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = bufferization.to_tensor %arg0 : memref<?xf64>
    %3 = tensor.empty() : tensor<2x5x4x4xf64>
    %4 = tensor.empty() : tensor<2x5x5x4xf64>
    %5 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%4 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %6 = polygeist.submap(%2, %c2, %c5, %c4, %c5, %c5) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %7 = polygeist.submap(%1, %c2, %c5, %c4, %c5, %c5) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %8 = linalg.generic {doc = "", indexing_maps = [#map3, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%6, %7 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%5 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %17 = arith.mulf %in, %in_0 : f64
      %18 = arith.addf %out, %17 : f64
      linalg.yield %18 : f64
    } -> tensor<2x5x5x4xf64>
    %9 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%3 : tensor<2x5x4x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x4x4xf64>
    %10 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c5) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %11 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%8, %10 : tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>) outs(%9 : tensor<2x5x4x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %17 = arith.mulf %in, %in_0 : f64
      %18 = arith.addf %out, %17 : f64
      linalg.yield %18 : f64
    } -> tensor<2x5x4x4xf64>
    %12 = polygeist.submap(%1, %c2, %c4, %c4, %c4, %c5) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %13 = polygeist.submap(%0, %c2, %c4, %c4, %c4, %c5) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %14 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%11, %12 : tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>) outs(%13 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %17 = arith.mulf %in, %in_0 : f64
      %18 = arith.addf %out, %17 : f64
      linalg.yield %18 : f64
    } -> tensor<?x?x?x?x?xf64>
    %15 = polygeist.submapInverse(%0, %14, %c2, %c4, %c4, %c4, %c5) {map = #map7} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %16 = bufferization.to_memref %15 : memref<?xf64>
    memref.copy %16, %arg2 : memref<?xf64> to memref<?xf64>
    return
  }
}
