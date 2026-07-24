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
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 375 + d1 * 25 + d2 * 5)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 375 + d1 * 25 + d2 * 5 + 125)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 375 + d1 * 25 + d2 * 5 + 250)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 5)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 5)>
#map16 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 64 + d1 * 16 + d2 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_convection_apply_3d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg5 : memref<?xf64>
    %1 = bufferization.to_tensor %arg4 : memref<?xf64>
    %2 = bufferization.to_tensor %arg3 : memref<?xf64>
    %3 = bufferization.to_tensor %arg2 : memref<?xf64>
    %4 = bufferization.to_tensor %arg1 : memref<?xf64>
    %5 = bufferization.to_tensor %arg0 : memref<?xf64>
    %6 = tensor.empty() : tensor<2x4x4x4xf64>
    %7 = tensor.empty() : tensor<2x5x4x4xf64>
    %8 = tensor.empty() : tensor<2x5x5x4xf64>
    %9 = tensor.empty() : tensor<2x5x5x5xf64>
    %10 = tensor.empty() : tensor<2x5x5x5xf64>
    %11 = tensor.empty() : tensor<2x5x5x5xf64>
    %12 = tensor.empty() : tensor<2x4x5x5xf64>
    %13 = tensor.empty() : tensor<2x4x5x5xf64>
    %14 = tensor.empty() : tensor<2x4x5x5xf64>
    %15 = tensor.empty() : tensor<2x4x4x5xf64>
    %16 = tensor.empty() : tensor<2x4x4x5xf64>
    %18 = polygeist.submap(%5, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %19 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v16_contract_20_tc2 = tensor.cast %16 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v20_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%19, %18, %v16_contract_20_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %20 = tensor.cast %v20_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %22 = polygeist.submap(%4, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %23 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v15_contract_24_tc2 = tensor.cast %15 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v24_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%23, %22, %v15_contract_24_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %24 = tensor.cast %v24_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %26 = polygeist.submap(%5, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v24_contract_27_tc0 = tensor.cast %24 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v14_contract_27_tc2 = tensor.cast %14 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v27_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v24_contract_27_tc0, %26, %v14_contract_27_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %27 = tensor.cast %v27_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %29 = polygeist.submap(%4, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v20_contract_30_tc0 = tensor.cast %20 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v13_contract_30_tc2 = tensor.cast %13 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v30_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v20_contract_30_tc0, %29, %v13_contract_30_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %30 = tensor.cast %v30_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %32 = polygeist.submap(%5, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v20_contract_33_tc0 = tensor.cast %20 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v12_contract_33_tc2 = tensor.cast %12 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v33_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v20_contract_33_tc0, %32, %v12_contract_33_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %33 = tensor.cast %v33_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %35 = polygeist.submap(%5, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v27_contract_36_tc0 = tensor.cast %27 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v11_contract_36_tc2 = tensor.cast %11 : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>

    %v36_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v27_contract_36_tc0, %35, %v11_contract_36_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %36 = tensor.cast %v36_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x5xf64>
    %38 = polygeist.submap(%5, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v30_contract_39_tc0 = tensor.cast %30 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v10_contract_39_tc2 = tensor.cast %10 : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>

    %v39_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v30_contract_39_tc0, %38, %v10_contract_39_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %39 = tensor.cast %v39_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x5xf64>
    %41 = polygeist.submap(%4, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v33_contract_42_tc0 = tensor.cast %33 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v9_contract_42_tc2 = tensor.cast %9 : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>

    %v42_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v33_contract_42_tc0, %41, %v9_contract_42_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %42 = tensor.cast %v42_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x5xf64>
    %43 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%8 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %44 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %45 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %46 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %47 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %48 = linalg.generic {doc = "", indexing_maps = [#map3, #map13, #map3, #map13, #map3, #map13, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%45, %36, %46, %39, %47, %42, %44 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%43 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
      %59 = arith.mulf %in, %in_0 : f64
      %60 = arith.mulf %in_1, %in_2 : f64
      %61 = arith.addf %59, %60 : f64
      %62 = arith.mulf %in_3, %in_4 : f64
      %63 = arith.addf %61, %62 : f64
      %64 = arith.mulf %63, %in_5 : f64
      %65 = arith.addf %out, %64 : f64
      linalg.yield %65 : f64
    } -> tensor<2x5x5x4xf64>
    %50 = polygeist.submap(%3, %c2, %c5, %c4, %c4, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v48_contract_51_tc0 = tensor.cast %48 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v7_contract_51_tc2 = tensor.cast %7 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v51_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v48_contract_51_tc0, %50, %v7_contract_51_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %51 = tensor.cast %v51_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %53 = polygeist.submap(%3, %c2, %c4, %c4, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v51_contract_54_tc0 = tensor.cast %51 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v6_contract_54_tc2 = tensor.cast %6 : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>

    %v54_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v51_contract_54_tc0, %53, %v6_contract_54_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %54 = tensor.cast %v54_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x4xf64>
    %55 = polygeist.submap(%0, %c2, %c4, %c4, %c4) {map = #map16} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v54_tc0 = tensor.cast %54 : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>

    %56 = kernel.launch @cudnnAddTensor_batched(%v54_tc0, %55) : (tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>
    %57 = polygeist.submapInverse(%0, %56, %c2, %c4, %c4, %c4) {map = #map16} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %58 = bufferization.to_memref %57 : memref<?xf64>
    memref.copy %58, %arg5 : memref<?xf64> to memref<?xf64>
    return
  }
}
