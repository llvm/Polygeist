#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d2 * 5 + d1 + d0 * 50)>
#map9 = affine_map<(d0, d1, d2) -> (d2 * 5 + d1 + d0 * 50 + 25)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d2 * 5 + d1 + d0 * 50 + 25)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_interp_grad_2d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = tensor.empty() : tensor<2x4x5xf64>
    %5 = tensor.empty() : tensor<2x4x5xf64>
    %6 = polygeist.submap(%5, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %7 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%6 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %8 = polygeist.submapInverse(%5, %7, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x5xf64>
    %9 = polygeist.submap(%0, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %10 = polygeist.submap(%1, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %11 = polygeist.submap(%8, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v9_contract_12_tc0 = tensor.cast %9 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v10_contract_12_tc1 = tensor.cast %10 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v11_contract_12_tc2 = tensor.cast %11 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v12_tdyn = kernel.launch @cutensornetContraction2_f64(%v9_contract_12_tc0, %v10_contract_12_tc1, %v11_contract_12_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %12 = tensor.cast %v12_tdyn : tensor<*xf64> to tensor<?x?x?x?xf64>
    %13 = polygeist.submapInverse(%8, %12, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5xf64>
    %14 = polygeist.submap(%4, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %15 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%14 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %16 = polygeist.submapInverse(%4, %15, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x5xf64>
    %17 = polygeist.submap(%0, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %18 = polygeist.submap(%2, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %19 = polygeist.submap(%16, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v17_contract_20_tc0 = tensor.cast %17 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v18_contract_20_tc1 = tensor.cast %18 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v19_contract_20_tc2 = tensor.cast %19 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v20_tdyn = kernel.launch @cutensornetContraction2_f64(%v17_contract_20_tc0, %v18_contract_20_tc1, %v19_contract_20_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %20 = tensor.cast %v20_tdyn : tensor<*xf64> to tensor<?x?x?x?xf64>
    %21 = polygeist.submapInverse(%16, %20, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5xf64>
    %22 = polygeist.submap(%3, %c2, %c5, %c5) {map = #map5} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %23 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%22 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %24 = polygeist.submapInverse(%3, %23, %c2, %c5, %c5) {map = #map5} : (tensor<?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?xf64>
    %25 = polygeist.submap(%21, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %26 = polygeist.submap(%1, %c2, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %27 = polygeist.submap(%24, %c2, %c5, %c5, %c4) {map = #map8} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %28 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%25, %26 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%27 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %39 = arith.mulf %in, %in_0 : f64
      %40 = arith.addf %out, %39 : f64
      linalg.yield %40 : f64
    } -> tensor<?x?x?x?xf64>
    %29 = polygeist.submapInverse(%24, %28, %c2, %c5, %c5, %c4) {map = #map8} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %30 = polygeist.submap(%29, %c2, %c5, %c5) {map = #map9} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %31 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%30 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %32 = polygeist.submapInverse(%29, %31, %c2, %c5, %c5) {map = #map9} : (tensor<?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?xf64>
    %33 = polygeist.submap(%13, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %34 = polygeist.submap(%2, %c2, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %35 = polygeist.submap(%32, %c2, %c5, %c5, %c4) {map = #map10} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %36 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%33, %34 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%35 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %39 = arith.mulf %in, %in_0 : f64
      %40 = arith.addf %out, %39 : f64
      linalg.yield %40 : f64
    } -> tensor<?x?x?x?xf64>
    %37 = polygeist.submapInverse(%32, %36, %c2, %c5, %c5, %c4) {map = #map10} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %38 = bufferization.to_memref %37 : memref<?xf64>
    memref.copy %38, %arg3 : memref<?xf64> to memref<?xf64>
    return
  }
}
