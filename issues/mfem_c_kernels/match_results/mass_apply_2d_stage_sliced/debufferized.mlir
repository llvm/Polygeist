#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d2 + d0 * 25 + d1 * 5)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 5)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 5)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d2 + d0 * 16 + d1 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_mass_apply_2d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = bufferization.to_tensor %arg4 : memref<?xf64>
    %5 = tensor.empty() : tensor<2x5x4xf64>
    %6 = tensor.empty() : tensor<2x5x5xf64>
    %7 = tensor.empty() : tensor<2x4x5xf64>
    %8 = polygeist.submap(%7, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %9 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%8 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %10 = polygeist.submapInverse(%7, %9, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x5xf64>
    %11 = polygeist.submap(%0, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %12 = polygeist.submap(%3, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %13 = polygeist.submap(%10, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %14 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%11, %12 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%13 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %42 = arith.mulf %in, %in_0 : f64
      %43 = arith.addf %out, %42 : f64
      linalg.yield %43 : f64
    } -> tensor<?x?x?x?xf64>
    %15 = polygeist.submapInverse(%10, %14, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5xf64>
    %16 = polygeist.submap(%6, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %17 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%16 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %18 = polygeist.submapInverse(%6, %17, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x5xf64>
    %19 = polygeist.submap(%0, %c2, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %20 = polygeist.submap(%15, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %21 = polygeist.submap(%18, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %22 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%19, %20 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%21 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %42 = arith.mulf %in, %in_0 : f64
      %43 = arith.addf %out, %42 : f64
      linalg.yield %43 : f64
    } -> tensor<?x?x?x?xf64>
    %23 = polygeist.submapInverse(%18, %22, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5xf64>
    %24 = polygeist.submap(%2, %c2, %c5, %c5) {map = #map7} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %25 = polygeist.submap(%23, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %26 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%24 : tensor<?x?x?xf64>) outs(%25 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %42 = arith.mulf %out, %in : f64
      linalg.yield %42 : f64
    } -> tensor<?x?x?xf64>
    %27 = polygeist.submapInverse(%23, %26, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x5xf64>
    %28 = polygeist.submap(%5, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %29 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%28 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %30 = polygeist.submapInverse(%5, %29, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x4xf64>
    %31 = polygeist.submap(%1, %c2, %c5, %c4, %c5) {map = #map8} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %32 = polygeist.submap(%27, %c2, %c5, %c4, %c5) {map = #map9} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %33 = polygeist.submap(%30, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %34 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%31, %32 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%33 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %42 = arith.mulf %in, %in_0 : f64
      %43 = arith.addf %out, %42 : f64
      linalg.yield %43 : f64
    } -> tensor<?x?x?x?xf64>
    %35 = polygeist.submapInverse(%30, %34, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4xf64>
    %36 = polygeist.submap(%1, %c2, %c4, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %37 = polygeist.submap(%35, %c2, %c4, %c4, %c5) {map = #map6} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %38 = polygeist.submap(%4, %c2, %c4, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %39 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%36, %37 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%38 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %42 = arith.mulf %in, %in_0 : f64
      %43 = arith.addf %out, %42 : f64
      linalg.yield %43 : f64
    } -> tensor<?x?x?x?xf64>
    %40 = polygeist.submapInverse(%4, %39, %c2, %c4, %c4, %c5) {map = #map11} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %41 = bufferization.to_memref %40 : memref<?xf64>
    memref.copy %41, %arg4 : memref<?xf64> to memref<?xf64>
    return
  }
}
