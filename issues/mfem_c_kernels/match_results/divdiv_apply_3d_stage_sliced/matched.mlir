#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 12 + d2 * 4 + d0 * 108)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 3)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 3)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 12 + d2 * 3 + d0 * 108 + 36)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 9 + d2 * 3 + d0 * 108 + 72)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 5)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 125 + d1 * 25 + d2 * 5)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map17 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 5)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 5)>
#map19 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d1 * 12 + d2 * 4 + d0 * 108)>
#map20 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d1 * 12 + d2 * 3 + d0 * 108 + 36)>
#map21 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d1 * 9 + d2 * 3 + d0 * 108 + 72)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_divdiv_apply_3d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg6 : memref<?xf64>
    %1 = bufferization.to_tensor %arg5 : memref<?xf64>
    %2 = bufferization.to_tensor %arg4 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = bufferization.to_tensor %arg2 : memref<?xf64>
    %5 = bufferization.to_tensor %arg1 : memref<?xf64>
    %6 = bufferization.to_tensor %arg0 : memref<?xf64>
    %7 = tensor.empty() : tensor<2x5x4x4xf64>
    %8 = tensor.empty() : tensor<2x5x4x4xf64>
    %9 = tensor.empty() : tensor<2x5x4x4xf64>
    %10 = tensor.empty() : tensor<2x5x5x4xf64>
    %11 = tensor.empty() : tensor<2x5x5x4xf64>
    %12 = tensor.empty() : tensor<2x5x5x4xf64>
    %13 = tensor.empty() : tensor<2x5x5x5xf64>
    %14 = tensor.empty() : tensor<2x5x5x5xf64>
    %15 = tensor.empty() : tensor<2x5x5x5xf64>
    %16 = tensor.empty() : tensor<2x4x5x5xf64>
    %17 = tensor.empty() : tensor<2x4x5x5xf64>
    %18 = tensor.empty() : tensor<2x4x5x5xf64>
    %19 = tensor.empty() : tensor<2x4x4x5xf64>
    %20 = tensor.empty() : tensor<2x4x4x5xf64>
    %21 = tensor.empty() : tensor<2x4x4x5xf64>
    %23 = polygeist.submap(%4, %c2, %c3, %c3, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %24 = polygeist.submap(%1, %c2, %c3, %c3, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v21_contract_25_tc2 = tensor.cast %21 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v25_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%24, %23, %v21_contract_25_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %25 = tensor.cast %v25_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %27 = polygeist.submap(%6, %c2, %c3, %c5, %c5, %c3) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v25_contract_28_tc0 = tensor.cast %25 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v18_contract_28_tc2 = tensor.cast %18 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v28_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v25_contract_28_tc0, %27, %v18_contract_28_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %28 = tensor.cast %v28_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %30 = polygeist.submap(%6, %c2, %c5, %c5, %c5, %c3) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v28_contract_31_tc0 = tensor.cast %28 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v15_contract_31_tc2 = tensor.cast %15 : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>

    %v31_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v28_contract_31_tc0, %30, %v15_contract_31_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %31 = tensor.cast %v31_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x5xf64>
    %33 = polygeist.submap(%6, %c2, %c3, %c4, %c5, %c3) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %34 = polygeist.submap(%1, %c2, %c3, %c4, %c5, %c3) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v20_contract_35_tc2 = tensor.cast %20 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v35_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%34, %33, %v20_contract_35_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %35 = tensor.cast %v35_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %37 = polygeist.submap(%4, %c2, %c3, %c5, %c5, %c4) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v35_contract_38_tc0 = tensor.cast %35 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v17_contract_38_tc2 = tensor.cast %17 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v38_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v35_contract_38_tc0, %37, %v17_contract_38_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %38 = tensor.cast %v38_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %40 = polygeist.submap(%6, %c2, %c5, %c5, %c5, %c3) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v38_contract_41_tc0 = tensor.cast %38 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v14_contract_41_tc2 = tensor.cast %14 : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>

    %v41_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v38_contract_41_tc0, %40, %v14_contract_41_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %41 = tensor.cast %v41_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x5xf64>
    %43 = polygeist.submap(%6, %c2, %c4, %c3, %c5, %c3) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %44 = polygeist.submap(%1, %c2, %c4, %c3, %c5, %c3) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v19_contract_45_tc2 = tensor.cast %19 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v45_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%44, %43, %v19_contract_45_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %45 = tensor.cast %v45_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %47 = polygeist.submap(%6, %c2, %c4, %c5, %c5, %c3) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v45_contract_48_tc0 = tensor.cast %45 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v16_contract_48_tc2 = tensor.cast %16 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v48_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v45_contract_48_tc0, %47, %v16_contract_48_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %48 = tensor.cast %v48_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %50 = polygeist.submap(%4, %c2, %c5, %c5, %c5, %c4) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v48_contract_51_tc0 = tensor.cast %48 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v13_contract_51_tc2 = tensor.cast %13 : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>

    %v51_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v48_contract_51_tc0, %50, %v13_contract_51_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %51 = tensor.cast %v51_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x5xf64>
    %52 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%12 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %53 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %54 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %55 = linalg.generic {doc = "", indexing_maps = [#map3, #map16, #map16, #map16, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%54, %31, %41, %51, %53 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%52 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
      %86 = arith.addf %in_0, %in_1 : f64
      %87 = arith.addf %86, %in_2 : f64
      %88 = arith.mulf %in, %87 : f64
      %89 = arith.mulf %88, %in_3 : f64
      %90 = arith.addf %out, %89 : f64
      linalg.yield %90 : f64
    } -> tensor<2x5x5x4xf64>
    %56 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%11 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %57 = polygeist.submap(%5, %c2, %c5, %c5, %c3, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %58 = polygeist.submap(%2, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %59 = linalg.generic {doc = "", indexing_maps = [#map3, #map16, #map16, #map16, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%58, %31, %41, %51, %57 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%56 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
      %86 = arith.addf %in_0, %in_1 : f64
      %87 = arith.addf %86, %in_2 : f64
      %88 = arith.mulf %in, %87 : f64
      %89 = arith.mulf %88, %in_3 : f64
      %90 = arith.addf %out, %89 : f64
      linalg.yield %90 : f64
    } -> tensor<2x5x5x4xf64>
    %60 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%10 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %61 = polygeist.submap(%5, %c2, %c5, %c5, %c3, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %62 = polygeist.submap(%2, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %63 = linalg.generic {doc = "", indexing_maps = [#map3, #map16, #map16, #map16, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%62, %31, %41, %51, %61 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%60 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
      %86 = arith.addf %in_0, %in_1 : f64
      %87 = arith.addf %86, %in_2 : f64
      %88 = arith.mulf %in, %87 : f64
      %89 = arith.mulf %88, %in_3 : f64
      %90 = arith.addf %out, %89 : f64
      linalg.yield %90 : f64
    } -> tensor<2x5x5x4xf64>
    %65 = polygeist.submap(%5, %c2, %c5, %c3, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v55_contract_66_tc0 = tensor.cast %55 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v9_contract_66_tc2 = tensor.cast %9 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v66_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v55_contract_66_tc0, %65, %v9_contract_66_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %66 = tensor.cast %v66_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %68 = polygeist.submap(%3, %c2, %c5, %c4, %c3, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v59_contract_69_tc0 = tensor.cast %59 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v8_contract_69_tc2 = tensor.cast %8 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v69_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v59_contract_69_tc0, %68, %v8_contract_69_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %69 = tensor.cast %v69_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %71 = polygeist.submap(%5, %c2, %c5, %c3, %c3, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v63_contract_72_tc0 = tensor.cast %63 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v7_contract_72_tc2 = tensor.cast %7 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v72_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v63_contract_72_tc0, %71, %v7_contract_72_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %72 = tensor.cast %v72_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %73 = polygeist.submap(%5, %c2, %c3, %c3, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %74 = polygeist.submap(%0, %c2, %c3, %c3, %c4, %c5) {map = #map19} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %75 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%66, %73 : tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>) outs(%74 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %86 = arith.mulf %in, %in_0 : f64
      %87 = arith.addf %out, %86 : f64
      linalg.yield %87 : f64
    } -> tensor<?x?x?x?x?xf64>
    %76 = polygeist.submapInverse(%0, %75, %c2, %c3, %c3, %c4, %c5) {map = #map19} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %77 = polygeist.submap(%5, %c2, %c3, %c4, %c3, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %78 = polygeist.submap(%76, %c2, %c3, %c4, %c3, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %79 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%69, %77 : tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>) outs(%78 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %86 = arith.mulf %in, %in_0 : f64
      %87 = arith.addf %out, %86 : f64
      linalg.yield %87 : f64
    } -> tensor<?x?x?x?x?xf64>
    %80 = polygeist.submapInverse(%76, %79, %c2, %c3, %c4, %c3, %c5) {map = #map20} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %81 = polygeist.submap(%3, %c2, %c4, %c3, %c3, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %82 = polygeist.submap(%80, %c2, %c4, %c3, %c3, %c5) {map = #map21} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %83 = linalg.generic {doc = "", indexing_maps = [#map8, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%72, %81 : tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>) outs(%82 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %86 = arith.mulf %in, %in_0 : f64
      %87 = arith.addf %out, %86 : f64
      linalg.yield %87 : f64
    } -> tensor<?x?x?x?x?xf64>
    %84 = polygeist.submapInverse(%80, %83, %c2, %c4, %c3, %c3, %c5) {map = #map21} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %85 = bufferization.to_memref %84 : memref<?xf64>
    memref.copy %85, %arg6 : memref<?xf64> to memref<?xf64>
    return
  }
}
