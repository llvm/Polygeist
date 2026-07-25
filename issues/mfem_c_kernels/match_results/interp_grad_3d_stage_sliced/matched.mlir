#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 64 + d1 * 16 + d2 * 4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d3 * 25 + d1 + d0 * 375 + d2 * 5)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d3 * 25 + d1 + d0 * 375 + d2 * 5)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d3 * 25 + d1 + d0 * 375 + d2 * 5 + 125)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d3 * 25 + d1 + d0 * 375 + d2 * 5 + 125)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d3 * 25 + d1 + d0 * 375 + d2 * 5 + 250)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d3 * 25 + d1 + d0 * 375 + d2 * 5 + 250)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_interp_grad_3d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg3 : memref<?xf64>
    %1 = bufferization.to_tensor %arg2 : memref<?xf64>
    %2 = bufferization.to_tensor %arg1 : memref<?xf64>
    %3 = bufferization.to_tensor %arg0 : memref<?xf64>
    %4 = tensor.empty() : tensor<2x4x5x5xf64>
    %5 = tensor.empty() : tensor<2x4x5x5xf64>
    %6 = tensor.empty() : tensor<2x4x5x5xf64>
    %7 = tensor.empty() : tensor<2x4x4x5xf64>
    %8 = tensor.empty() : tensor<2x4x4x5xf64>
    %10 = polygeist.submap(%3, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %11 = polygeist.submap(%2, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v8_contract_12_tc2 = tensor.cast %8 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v12_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%10, %11, %v8_contract_12_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %12 = tensor.cast %v12_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %14 = polygeist.submap(%3, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %15 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v7_contract_16_tc2 = tensor.cast %7 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v16_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%14, %15, %v7_contract_16_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %16 = tensor.cast %v16_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %18 = polygeist.submap(%2, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v16_contract_19_tc0 = tensor.cast %16 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v6_contract_19_tc2 = tensor.cast %6 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v19_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v16_contract_19_tc0, %18, %v6_contract_19_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %19 = tensor.cast %v19_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %21 = polygeist.submap(%1, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v12_contract_22_tc0 = tensor.cast %12 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v5_contract_22_tc2 = tensor.cast %5 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v22_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v12_contract_22_tc0, %21, %v5_contract_22_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %22 = tensor.cast %v22_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %24 = polygeist.submap(%2, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v12_contract_25_tc0 = tensor.cast %12 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v4_contract_25_tc2 = tensor.cast %4 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v25_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v12_contract_25_tc0, %24, %v4_contract_25_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %25 = tensor.cast %v25_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %26 = polygeist.submap(%0, %c2, %c5, %c5, %c5) {map = #map7} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %27 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%26 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %28 = polygeist.submapInverse(%0, %27, %c2, %c5, %c5, %c5) {map = #map7} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %29 = polygeist.submap(%2, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %30 = polygeist.submap(%28, %c2, %c5, %c5, %c5, %c4) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v19_contract_31_tc0 = tensor.cast %19 : tensor<2x4x5x5xf64> to tensor<*xf64>

    %v29_contract_31_tc1 = tensor.cast %29 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v30_contract_31_tc2 = tensor.cast %30 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v31_tdyn = kernel.launch @cutensornetContraction2_f64(%v19_contract_31_tc0, %v29_contract_31_tc1, %v30_contract_31_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %31 = tensor.cast %v31_tdyn : tensor<*xf64> to tensor<?x?x?x?x?xf64>
    %32 = polygeist.submapInverse(%28, %31, %c2, %c5, %c5, %c5, %c4) {map = #map9} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %33 = polygeist.submap(%32, %c2, %c5, %c5, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %34 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%33 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %35 = polygeist.submapInverse(%32, %34, %c2, %c5, %c5, %c5) {map = #map11} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %36 = polygeist.submap(%2, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %37 = polygeist.submap(%35, %c2, %c5, %c5, %c5, %c4) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v22_contract_38_tc0 = tensor.cast %22 : tensor<2x4x5x5xf64> to tensor<*xf64>

    %v36_contract_38_tc1 = tensor.cast %36 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v37_contract_38_tc2 = tensor.cast %37 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v38_tdyn = kernel.launch @cutensornetContraction2_f64(%v22_contract_38_tc0, %v36_contract_38_tc1, %v37_contract_38_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %38 = tensor.cast %v38_tdyn : tensor<*xf64> to tensor<?x?x?x?x?xf64>
    %39 = polygeist.submapInverse(%35, %38, %c2, %c5, %c5, %c5, %c4) {map = #map12} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %40 = polygeist.submap(%39, %c2, %c5, %c5, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %41 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%40 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %42 = polygeist.submapInverse(%39, %41, %c2, %c5, %c5, %c5) {map = #map13} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %43 = polygeist.submap(%1, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %44 = polygeist.submap(%42, %c2, %c5, %c5, %c5, %c4) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v25_contract_45_tc0 = tensor.cast %25 : tensor<2x4x5x5xf64> to tensor<*xf64>

    %v43_contract_45_tc1 = tensor.cast %43 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v44_contract_45_tc2 = tensor.cast %44 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v45_tdyn = kernel.launch @cutensornetContraction2_f64(%v25_contract_45_tc0, %v43_contract_45_tc1, %v44_contract_45_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %45 = tensor.cast %v45_tdyn : tensor<*xf64> to tensor<?x?x?x?x?xf64>
    %46 = polygeist.submapInverse(%42, %45, %c2, %c5, %c5, %c5, %c4) {map = #map14} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %47 = bufferization.to_memref %46 : memref<?xf64>
    memref.copy %47, %arg3 : memref<?xf64> to memref<?xf64>
    return
  }
}
