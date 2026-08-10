#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 64 + d1 * 16 + d2 * 4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 5)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d2 * 5 + d0 * 750)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 125)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 250)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 375)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 500)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 625)>
#map17 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 5)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 5)>
#map19 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 64 + d1 * 16 + d2 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_abs_l1_diffusion_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
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
    %7 = tensor.empty() : tensor<2x4x4x4xf64>
    %8 = tensor.empty() : tensor<2x4x4x4xf64>
    %9 = tensor.empty() : tensor<2x4x4x4xf64>
    %10 = tensor.empty() : tensor<2x5x4x4xf64>
    %11 = tensor.empty() : tensor<2x5x4x4xf64>
    %12 = tensor.empty() : tensor<2x5x4x4xf64>
    %13 = tensor.empty() : tensor<2x5x5x4xf64>
    %14 = tensor.empty() : tensor<2x5x5x4xf64>
    %15 = tensor.empty() : tensor<2x5x5x4xf64>
    %16 = tensor.empty() : tensor<2x5x5x5xf64>
    %17 = tensor.empty() : tensor<2x5x5x5xf64>
    %18 = tensor.empty() : tensor<2x5x5x5xf64>
    %19 = tensor.empty() : tensor<2x4x5x5xf64>
    %20 = tensor.empty() : tensor<2x4x5x5xf64>
    %21 = tensor.empty() : tensor<2x4x5x5xf64>
    %22 = tensor.empty() : tensor<2x4x4x5xf64>
    %23 = tensor.empty() : tensor<2x4x4x5xf64>
    %extracted_slice = tensor.extract_slice %23[0, 0, 0, 0] [%c2, %c4, %c4, %c5] [1, 1, 1, 1] : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>
    %24 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %inserted_slice = tensor.insert_slice %24 into %23[0, 0, 0, 0] [%c2, %c4, %c4, %c5] [1, 1, 1, 1] : tensor<?x?x?x?xf64> into tensor<2x4x4x5xf64>
    %25 = polygeist.submap(%6, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %26 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %inserted_slice_contract_27_tc2 = tensor.cast %inserted_slice : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v27_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%26, %25, %inserted_slice_contract_27_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %27 = tensor.cast %v27_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %extracted_slice_0 = tensor.extract_slice %22[0, 0, 0, 0] [%c2, %c4, %c4, %c5] [1, 1, 1, 1] : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>
    %28 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_0 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %inserted_slice_1 = tensor.insert_slice %28 into %22[0, 0, 0, 0] [%c2, %c4, %c4, %c5] [1, 1, 1, 1] : tensor<?x?x?x?xf64> into tensor<2x4x4x5xf64>
    %29 = polygeist.submap(%5, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %30 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %inserted_slice_1_contract_31_tc2 = tensor.cast %inserted_slice_1 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v31_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%30, %29, %inserted_slice_1_contract_31_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %31 = tensor.cast %v31_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %extracted_slice_2 = tensor.extract_slice %21[0, 0, 0, 0] [%c2, %c4, %c5, %c5] [1, 1, 1, 1] : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>
    %extracted_slice_3 = tensor.extract_slice %31[0, 0, 0, 0] [%c2, %c4, %c4, %c5] [1, 1, 1, 1] : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>
    %33 = polygeist.submap(%6, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %34 = kernel.launch @cutensornetContraction2_f64_r4r5r4(%extracted_slice_3, %33, %extracted_slice_2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %extracted_slice_4 = tensor.extract_slice %20[0, 0, 0, 0] [%c2, %c4, %c5, %c5] [1, 1, 1, 1] : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>
    %extracted_slice_5 = tensor.extract_slice %27[0, 0, 0, 0] [%c2, %c4, %c4, %c5] [1, 1, 1, 1] : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>
    %36 = polygeist.submap(%5, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %37 = kernel.launch @cutensornetContraction2_f64_r4r5r4(%extracted_slice_5, %36, %extracted_slice_4) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %extracted_slice_6 = tensor.extract_slice %19[0, 0, 0, 0] [%c2, %c4, %c5, %c5] [1, 1, 1, 1] : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>
    %extracted_slice_7 = tensor.extract_slice %27[0, 0, 0, 0] [%c2, %c4, %c4, %c5] [1, 1, 1, 1] : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>
    %39 = polygeist.submap(%6, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %40 = kernel.launch @cutensornetContraction2_f64_r4r5r4(%extracted_slice_7, %39, %extracted_slice_6) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %extracted_slice_8 = tensor.extract_slice %18[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>
    %42 = polygeist.submap(%6, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %43 = kernel.launch @cutensornetContraction2_f64_r4r5r4(%34, %42, %extracted_slice_8) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %extracted_slice_9 = tensor.extract_slice %17[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>
    %45 = polygeist.submap(%6, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %46 = kernel.launch @cutensornetContraction2_f64_r4r5r4(%37, %45, %extracted_slice_9) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %extracted_slice_10 = tensor.extract_slice %16[0, 0, 0, 0] [%c2, %c5, %c5, %c5] [1, 1, 1, 1] : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>
    %48 = polygeist.submap(%5, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %49 = kernel.launch @cutensornetContraction2_f64_r4r5r4(%40, %48, %extracted_slice_10) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %extracted_slice_11 = tensor.extract_slice %15[0, 0, 0, 0] [%c2, %c5, %c5, %c4] [1, 1, 1, 1] : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>
    %50 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_11 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %51 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %52 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %53 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %54 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %55 = linalg.generic {doc = "", indexing_maps = [#map3, #map13, #map3, #map13, #map3, #map13, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%52, %43, %53, %46, %54, %49, %51 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%50 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_20: f64, %in_21: f64, %in_22: f64, %in_23: f64, %in_24: f64, %in_25: f64, %out: f64):
      %90 = arith.mulf %in, %in_20 : f64
      %91 = arith.mulf %in_21, %in_22 : f64
      %92 = arith.addf %90, %91 : f64
      %93 = arith.mulf %in_23, %in_24 : f64
      %94 = arith.addf %92, %93 : f64
      %95 = arith.mulf %94, %in_25 : f64
      %96 = arith.addf %out, %95 : f64
      linalg.yield %96 : f64
    } -> tensor<?x?x?x?xf64>
    %extracted_slice_12 = tensor.extract_slice %14[0, 0, 0, 0] [%c2, %c5, %c5, %c4] [1, 1, 1, 1] : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>
    %56 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_12 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %57 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %58 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %59 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %60 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %61 = linalg.generic {doc = "", indexing_maps = [#map3, #map13, #map3, #map13, #map3, #map13, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%58, %43, %59, %46, %60, %49, %57 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%56 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_20: f64, %in_21: f64, %in_22: f64, %in_23: f64, %in_24: f64, %in_25: f64, %out: f64):
      %90 = arith.mulf %in, %in_20 : f64
      %91 = arith.mulf %in_21, %in_22 : f64
      %92 = arith.addf %90, %91 : f64
      %93 = arith.mulf %in_23, %in_24 : f64
      %94 = arith.addf %92, %93 : f64
      %95 = arith.mulf %94, %in_25 : f64
      %96 = arith.addf %out, %95 : f64
      linalg.yield %96 : f64
    } -> tensor<?x?x?x?xf64>
    %extracted_slice_13 = tensor.extract_slice %13[0, 0, 0, 0] [%c2, %c5, %c5, %c4] [1, 1, 1, 1] : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>
    %62 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%extracted_slice_13 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %63 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %64 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %65 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %66 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map16} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %67 = linalg.generic {doc = "", indexing_maps = [#map3, #map13, #map3, #map13, #map3, #map13, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%64, %43, %65, %46, %66, %49, %63 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%62 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_20: f64, %in_21: f64, %in_22: f64, %in_23: f64, %in_24: f64, %in_25: f64, %out: f64):
      %90 = arith.mulf %in, %in_20 : f64
      %91 = arith.mulf %in_21, %in_22 : f64
      %92 = arith.addf %90, %91 : f64
      %93 = arith.mulf %in_23, %in_24 : f64
      %94 = arith.addf %92, %93 : f64
      %95 = arith.mulf %94, %in_25 : f64
      %96 = arith.addf %out, %95 : f64
      linalg.yield %96 : f64
    } -> tensor<?x?x?x?xf64>
    %extracted_slice_14 = tensor.extract_slice %12[0, 0, 0, 0] [%c2, %c5, %c4, %c4] [1, 1, 1, 1] : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>
    %69 = polygeist.submap(%4, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %70 = kernel.launch @cutensornetContraction2_f64_r4r5r4(%55, %69, %extracted_slice_14) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %extracted_slice_15 = tensor.extract_slice %11[0, 0, 0, 0] [%c2, %c5, %c4, %c4] [1, 1, 1, 1] : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>
    %72 = polygeist.submap(%3, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %73 = kernel.launch @cutensornetContraction2_f64_r4r5r4(%61, %72, %extracted_slice_15) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %extracted_slice_16 = tensor.extract_slice %10[0, 0, 0, 0] [%c2, %c5, %c4, %c4] [1, 1, 1, 1] : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>
    %75 = polygeist.submap(%4, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %76 = kernel.launch @cutensornetContraction2_f64_r4r5r4(%67, %75, %extracted_slice_16) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %extracted_slice_17 = tensor.extract_slice %9[0, 0, 0, 0] [%c2, %c4, %c4, %c4] [1, 1, 1, 1] : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>
    %78 = polygeist.submap(%4, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %79 = kernel.launch @cutensornetContraction2_f64_r4r5r4(%70, %78, %extracted_slice_17) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %extracted_slice_18 = tensor.extract_slice %8[0, 0, 0, 0] [%c2, %c4, %c4, %c4] [1, 1, 1, 1] : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>
    %81 = polygeist.submap(%4, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %82 = kernel.launch @cutensornetContraction2_f64_r4r5r4(%73, %81, %extracted_slice_18) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %extracted_slice_19 = tensor.extract_slice %7[0, 0, 0, 0] [%c2, %c4, %c4, %c4] [1, 1, 1, 1] : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>
    %84 = polygeist.submap(%3, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %85 = kernel.launch @cutensornetContraction2_f64_r4r5r4(%76, %84, %extracted_slice_19) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %86 = polygeist.submap(%0, %c2, %c4, %c4, %c4) {map = #map19} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %87 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%79, %82, %85 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%86 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_20: f64, %in_21: f64, %out: f64):
      %90 = arith.addf %in, %in_20 : f64
      %91 = arith.addf %90, %in_21 : f64
      %92 = arith.addf %out, %91 : f64
      linalg.yield %92 : f64
    } -> tensor<?x?x?x?xf64>
    %88 = polygeist.submapInverse(%0, %87, %c2, %c4, %c4, %c4) {map = #map19} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %89 = bufferization.to_memref %88 : memref<?xf64>
    memref.copy %89, %arg6 : memref<?xf64> to memref<?xf64>
    return
  }
}

