#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 3 + d0 * 24)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4 + d0 * 24 + 12)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 5)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 25 + d1 * 5)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 5)>
#map14 = affine_map<(d0, d1, d2) -> (d2 + d1 * 3 + d0 * 24)>
#map15 = affine_map<(d0, d1, d2) -> (d2 + d1 * 4 + d0 * 24 + 12)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_curlcurl_apply_2d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg6 : memref<?xf64>
    %1 = bufferization.to_tensor %arg5 : memref<?xf64>
    %2 = bufferization.to_tensor %arg4 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = bufferization.to_tensor %arg2 : memref<?xf64>
    %5 = bufferization.to_tensor %arg1 : memref<?xf64>
    %6 = bufferization.to_tensor %arg0 : memref<?xf64>
    %7 = tensor.empty() : tensor<2x3x4xf64>
    %8 = tensor.empty() : tensor<2x4x3xf64>
    %9 = tensor.empty() : tensor<2x5x4xf64>
    %10 = tensor.empty() : tensor<2x5x3xf64>
    %11 = tensor.empty() : tensor<2x5x5xf64>
    %12 = tensor.empty() : tensor<2x5x5xf64>
    %13 = tensor.empty() : tensor<2x3x5xf64>
    %14 = tensor.empty() : tensor<2x4x5xf64>
    %16 = polygeist.submap(%6, %c2, %c4, %c5, %c3) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %17 = polygeist.submap(%1, %c2, %c4, %c5, %c3) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v17_contract_18_tc0 = tensor.cast %17 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v16_contract_18_tc1 = tensor.cast %16 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v14_contract_18_tc2 = tensor.cast %14 : tensor<2x4x5xf64> to tensor<*xf64>

    %v18_tdyn = kernel.launch @cutensornetContraction2_f64(%v17_contract_18_tc0, %v16_contract_18_tc1, %v14_contract_18_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %18 = tensor.cast %v18_tdyn : tensor<*xf64> to tensor<2x4x5xf64>
    %20 = polygeist.submap(%4, %c2, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v18_contract_21_tc0 = tensor.cast %18 : tensor<2x4x5xf64> to tensor<*xf64>

    %v20_contract_21_tc1 = tensor.cast %20 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v12_contract_21_tc2 = tensor.cast %12 : tensor<2x5x5xf64> to tensor<*xf64>

    %v21_tdyn = kernel.launch @cutensornetContraction2_f64(%v18_contract_21_tc0, %v20_contract_21_tc1, %v12_contract_21_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %21 = tensor.cast %v21_tdyn : tensor<*xf64> to tensor<2x5x5xf64>
    %23 = polygeist.submap(%4, %c2, %c3, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %24 = polygeist.submap(%1, %c2, %c3, %c5, %c4) {map = #map8} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v24_contract_25_tc0 = tensor.cast %24 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v23_contract_25_tc1 = tensor.cast %23 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v13_contract_25_tc2 = tensor.cast %13 : tensor<2x3x5xf64> to tensor<*xf64>

    %v25_tdyn = kernel.launch @cutensornetContraction2_f64(%v24_contract_25_tc0, %v23_contract_25_tc1, %v13_contract_25_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %25 = tensor.cast %v25_tdyn : tensor<*xf64> to tensor<2x3x5xf64>
    %27 = polygeist.submap(%6, %c2, %c5, %c5, %c3) {map = #map9} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v25_contract_28_tc0 = tensor.cast %25 : tensor<2x3x5xf64> to tensor<*xf64>

    %v27_contract_28_tc1 = tensor.cast %27 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v11_contract_28_tc2 = tensor.cast %11 : tensor<2x5x5xf64> to tensor<*xf64>

    %v28_tdyn = kernel.launch @cutensornetContraction2_f64(%v25_contract_28_tc0, %v27_contract_28_tc1, %v11_contract_28_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %28 = tensor.cast %v28_tdyn : tensor<*xf64> to tensor<2x5x5xf64>
    %29 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%10 : tensor<2x5x3xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x3xf64>
    %30 = polygeist.submap(%5, %c2, %c5, %c3, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %31 = polygeist.submap(%2, %c2, %c5, %c3, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %32 = linalg.generic {doc = "", indexing_maps = [#map3, #map12, #map12, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%31, %28, %21, %30 : tensor<?x?x?x?xf64>, tensor<2x5x5xf64>, tensor<2x5x5xf64>, tensor<?x?x?x?xf64>) outs(%29 : tensor<2x5x3xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %50 = arith.subf %in_0, %in_1 : f64
      %51 = arith.mulf %in, %50 : f64
      %52 = arith.mulf %51, %in_2 : f64
      %53 = arith.addf %out, %52 : f64
      linalg.yield %53 : f64
    } -> tensor<2x5x3xf64>
    %33 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%8 : tensor<2x4x3xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x4x3xf64>
    %34 = polygeist.submap(%3, %c2, %c4, %c3, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %35 = linalg.generic {doc = "", indexing_maps = [#map6, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%32, %34 : tensor<2x5x3xf64>, tensor<?x?x?x?xf64>) outs(%33 : tensor<2x4x3xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %50 = arith.mulf %in, %in_0 : f64
      %51 = arith.subf %out, %50 : f64
      linalg.yield %51 : f64
    } -> tensor<2x4x3xf64>
    %36 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%9 : tensor<2x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x4xf64>
    %37 = polygeist.submap(%3, %c2, %c5, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %38 = polygeist.submap(%2, %c2, %c5, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %39 = linalg.generic {doc = "", indexing_maps = [#map3, #map12, #map12, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%38, %28, %21, %37 : tensor<?x?x?x?xf64>, tensor<2x5x5xf64>, tensor<2x5x5xf64>, tensor<?x?x?x?xf64>) outs(%36 : tensor<2x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %50 = arith.subf %in_0, %in_1 : f64
      %51 = arith.mulf %in, %50 : f64
      %52 = arith.mulf %51, %in_2 : f64
      %53 = arith.addf %out, %52 : f64
      linalg.yield %53 : f64
    } -> tensor<2x5x4xf64>
    %41 = polygeist.submap(%5, %c2, %c3, %c4, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v39_contract_42_tc0 = tensor.cast %39 : tensor<2x5x4xf64> to tensor<*xf64>

    %v41_contract_42_tc1 = tensor.cast %41 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v7_contract_42_tc2 = tensor.cast %7 : tensor<2x3x4xf64> to tensor<*xf64>

    %v42_tdyn = kernel.launch @cutensornetContraction2_f64(%v39_contract_42_tc0, %v41_contract_42_tc1, %v7_contract_42_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d3, d1)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %42 = tensor.cast %v42_tdyn : tensor<*xf64> to tensor<2x3x4xf64>
    %43 = polygeist.submap(%0, %c2, %c4, %c3) {map = #map14} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %v35_tc0 = tensor.cast %35 : tensor<2x4x3xf64> to tensor<?x?x?xf64>

    %44 = kernel.launch @cublasDaxpby(%v35_tc0, %43) : (tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
    %45 = polygeist.submapInverse(%0, %44, %c2, %c4, %c3) {map = #map14} : (tensor<?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?xf64>
    %46 = polygeist.submap(%45, %c2, %c3, %c4) {map = #map15} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %v42_tc0 = tensor.cast %42 : tensor<2x3x4xf64> to tensor<?x?x?xf64>

    %47 = kernel.launch @cublasDaxpby(%v42_tc0, %46) : (tensor<?x?x?xf64>, tensor<?x?x?xf64>) -> tensor<?x?x?xf64>
    %48 = polygeist.submapInverse(%45, %47, %c2, %c3, %c4) {map = #map15} : (tensor<?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?xf64>
    %49 = bufferization.to_memref %48 : memref<?xf64>
    memref.copy %49, %arg6 : memref<?xf64> to memref<?xf64>
    return
  }
}
