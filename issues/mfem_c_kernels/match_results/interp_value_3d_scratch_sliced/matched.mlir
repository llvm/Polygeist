#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 64 + d1 * 16 + d2 * 4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d3 * 25 + d1 + d0 * 125 + d2 * 5)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d3 * 25 + d1 + d0 * 125 + d2 * 5)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_interp_value_3d_scratch_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?xf64>
    %3 = tensor.empty() : tensor<2x4x5x5xf64>
    %4 = tensor.empty() : tensor<2x4x4x5xf64>
    %5 = polygeist.submap(%4, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %6 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%5 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %7 = polygeist.submapInverse(%4, %6, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x5xf64>
    %8 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %9 = polygeist.submap(%0, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %10 = polygeist.submap(%7, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v8_contract_11_tc0 = tensor.cast %8 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v9_contract_11_tc1 = tensor.cast %9 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v10_contract_11_tc2 = tensor.cast %10 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v11_tdyn = kernel.launch @cutensornetContraction2_f64(%v8_contract_11_tc0, %v9_contract_11_tc1, %v10_contract_11_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %11 = tensor.cast %v11_tdyn : tensor<*xf64> to tensor<?x?x?x?x?xf64>
    %12 = polygeist.submapInverse(%7, %11, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x5xf64>
    %13 = polygeist.submap(%3, %c2, %c4, %c5, %c5) {map = #map5} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %14 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%13 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %15 = polygeist.submapInverse(%3, %14, %c2, %c4, %c5, %c5) {map = #map5} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %16 = polygeist.submap(%12, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %17 = polygeist.submap(%1, %c2, %c4, %c5, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %18 = polygeist.submap(%15, %c2, %c4, %c5, %c5, %c4) {map = #map7} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %19 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%16, %17 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%18 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %30 = arith.mulf %in, %in_0 : f64
      %31 = arith.addf %out, %30 : f64
      linalg.yield %31 : f64
    } -> tensor<?x?x?x?x?xf64>
    %20 = polygeist.submapInverse(%15, %19, %c2, %c4, %c5, %c5, %c4) {map = #map7} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %21 = polygeist.submap(%2, %c2, %c5, %c5, %c5) {map = #map8} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %22 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%21 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %23 = polygeist.submapInverse(%2, %22, %c2, %c5, %c5, %c5) {map = #map8} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %24 = polygeist.submap(%20, %c2, %c5, %c5, %c5, %c4) {map = #map9} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %25 = polygeist.submap(%1, %c2, %c5, %c5, %c5, %c4) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %26 = polygeist.submap(%23, %c2, %c5, %c5, %c5, %c4) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %27 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%24, %25 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%26 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %30 = arith.mulf %in, %in_0 : f64
      %31 = arith.addf %out, %30 : f64
      linalg.yield %31 : f64
    } -> tensor<?x?x?x?x?xf64>
    %28 = polygeist.submapInverse(%23, %27, %c2, %c5, %c5, %c5, %c4) {map = #map11} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %29 = bufferization.to_memref %28 : memref<?xf64>
    memref.copy %29, %arg2 : memref<?xf64> to memref<?xf64>
    return
  }
}
