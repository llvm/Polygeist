#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 64 + d1 * 16 + d2 * 4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 125 + d1 * 25 + d2 * 5)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 5)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 5)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 5)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 64 + d1 * 16 + d2 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_abs_l1_mass_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg4 : memref<?xf64>
    %1 = bufferization.to_tensor %arg3 : memref<?xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?xf64>
    %3 = bufferization.to_tensor %arg1 : memref<?xf64>
    %4 = bufferization.to_tensor %arg0 : memref<?xf64>
    %5 = tensor.empty() : tensor<2x5x4x4xf64>
    %6 = tensor.empty() : tensor<2x5x5x4xf64>
    %7 = tensor.empty() : tensor<2x5x5x5xf64>
    %8 = tensor.empty() : tensor<2x4x5x5xf64>
    %9 = tensor.empty() : tensor<2x4x4x5xf64>
    %extracted_slice = tensor.extract_slice %9[0, 0, 0, 0] [%c2, %c4, %c4, %c5] [1, 1, 1, 1] : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>
    %10 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %inserted_slice = tensor.insert_slice %10 into %9[0, 0, 0, 0] [%c2, %c4, %c4, %c5] [1, 1, 1, 1] : tensor<?x?x?x?xf64> into tensor<2x4x4x5xf64>
    %11 = polygeist.submap(%4, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %12 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %inserted_slice_contract_13_tc2 = tensor.cast %inserted_slice : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v13_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%11, %12, %inserted_slice_contract_13_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %13 = tensor.cast %v13_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %extracted_slice_0 = tensor.extract_slice %8[0, 0, 0, 0] [%c2, %c4, %c5, %c5] [1, 1, 1, 1] : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>
    %extracted_slice_1 = tensor.extract_slice %13[0, 0, 0, 0] [%c2, %c4, %c4, %c5] [1, 1, 1, 1] : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>
    %15 = polygeist.submap(%4, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %16 = kernel.launch @cutensornetContraction2_f64_r5r4r4(%15, %extracted_slice_1, %extracted_slice_0) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %extracted_slice_2 = tensor.extract_slice %7[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>
    %18 = polygeist.submap(%4, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %19 = kernel.launch @cutensornetContraction2_f64_r5r4r4(%18, %16, %extracted_slice_2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %20 = polygeist.submap(%2, %c2, %c5, %c5, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %21 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%20 : tensor<?x?x?x?xf64>) outs(%19 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %33 = arith.mulf %out, %in : f64
      linalg.yield %33 : f64
    } -> tensor<?x?x?x?xf64>
    %extracted_slice_3 = tensor.extract_slice %6[0, 0, 0, 0] [%c2, %c5, %c5, %c4] [1, 1, 1, 1] : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>
    %23 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %24 = kernel.launch @cutensornetContraction2_f64_r5r4r4(%23, %21, %extracted_slice_3) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %extracted_slice_4 = tensor.extract_slice %5[0, 0, 0, 0] [%c2, %c5, %c4, %c4] [1, 1, 1, 1] : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>
    %26 = polygeist.submap(%3, %c2, %c5, %c4, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %27 = kernel.launch @cutensornetContraction2_f64_r5r4r4(%26, %24, %extracted_slice_4) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %28 = polygeist.submap(%3, %c2, %c4, %c4, %c4, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %29 = polygeist.submap(%0, %c2, %c4, %c4, %c4) {map = #map14} : (tensor<?xf64>, index, index, index, index) -> tensor<2x4x4x4xf64>
    %30 = linalg.generic {doc = "", indexing_maps = [#map3, #map8, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%28, %27 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%29 : tensor<2x4x4x4xf64>) {
    ^bb0(%in: f64, %in_5: f64, %out: f64):
      %33 = arith.mulf %in, %in_5 : f64
      %34 = arith.addf %out, %33 : f64
      linalg.yield %34 : f64
    } -> tensor<2x4x4x4xf64>
    %31 = polygeist.submapInverse(%0, %30, %c2, %c4, %c4, %c4) {map = #map14} : (tensor<?xf64>, tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?xf64>
    %32 = bufferization.to_memref %31 : memref<?xf64>
    memref.copy %32, %arg4 : memref<?xf64> to memref<?xf64>
    return
  }
}

