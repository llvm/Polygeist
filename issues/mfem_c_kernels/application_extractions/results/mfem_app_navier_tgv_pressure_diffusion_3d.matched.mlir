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
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_navier_tgv_pressure_diffusion_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
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
    %25 = polygeist.submap(%6, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %26 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v23_contract_27_tc2 = tensor.cast %23 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v27_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%26, %25, %v23_contract_27_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %27 = tensor.cast %v27_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %29 = polygeist.submap(%5, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %30 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v22_contract_31_tc2 = tensor.cast %22 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v31_tdyn = kernel.launch @cutensornetContraction2_f64_r5r5r4(%30, %29, %v22_contract_31_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>]} : (tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %31 = tensor.cast %v31_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x5xf64>
    %33 = polygeist.submap(%6, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v31_contract_34_tc0 = tensor.cast %31 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v21_contract_34_tc2 = tensor.cast %21 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v34_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v31_contract_34_tc0, %33, %v21_contract_34_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %34 = tensor.cast %v34_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %36 = polygeist.submap(%5, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v27_contract_37_tc0 = tensor.cast %27 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v20_contract_37_tc2 = tensor.cast %20 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v37_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v27_contract_37_tc0, %36, %v20_contract_37_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %37 = tensor.cast %v37_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %39 = polygeist.submap(%6, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v27_contract_40_tc0 = tensor.cast %27 : tensor<2x4x4x5xf64> to tensor<?x?x?x?xf64>

    %v19_contract_40_tc2 = tensor.cast %19 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v40_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v27_contract_40_tc0, %39, %v19_contract_40_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %40 = tensor.cast %v40_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x5x5xf64>
    %42 = polygeist.submap(%6, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v34_contract_43_tc0 = tensor.cast %34 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v18_contract_43_tc2 = tensor.cast %18 : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>

    %v43_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v34_contract_43_tc0, %42, %v18_contract_43_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %43 = tensor.cast %v43_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x5xf64>
    %45 = polygeist.submap(%6, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v37_contract_46_tc0 = tensor.cast %37 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v17_contract_46_tc2 = tensor.cast %17 : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>

    %v46_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v37_contract_46_tc0, %45, %v17_contract_46_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %46 = tensor.cast %v46_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x5xf64>
    %48 = polygeist.submap(%5, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v40_contract_49_tc0 = tensor.cast %40 : tensor<2x4x5x5xf64> to tensor<?x?x?x?xf64>

    %v16_contract_49_tc2 = tensor.cast %16 : tensor<2x5x5x5xf64> to tensor<?x?x?x?xf64>

    %v49_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v40_contract_49_tc0, %48, %v16_contract_49_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %49 = tensor.cast %v49_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x5x5xf64>
    %50 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%15 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %51 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %52 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %53 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %54 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %55 = linalg.generic {doc = "", indexing_maps = [#map3, #map13, #map3, #map13, #map3, #map13, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%52, %43, %53, %46, %54, %49, %51 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%50 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
      %90 = arith.mulf %in, %in_0 : f64
      %91 = arith.mulf %in_1, %in_2 : f64
      %92 = arith.addf %90, %91 : f64
      %93 = arith.mulf %in_3, %in_4 : f64
      %94 = arith.addf %92, %93 : f64
      %95 = arith.mulf %94, %in_5 : f64
      %96 = arith.addf %out, %95 : f64
      linalg.yield %96 : f64
    } -> tensor<2x5x5x4xf64>
    %56 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%14 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %57 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %58 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %59 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %60 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %61 = linalg.generic {doc = "", indexing_maps = [#map3, #map13, #map3, #map13, #map3, #map13, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%58, %43, %59, %46, %60, %49, %57 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%56 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
      %90 = arith.mulf %in, %in_0 : f64
      %91 = arith.mulf %in_1, %in_2 : f64
      %92 = arith.addf %90, %91 : f64
      %93 = arith.mulf %in_3, %in_4 : f64
      %94 = arith.addf %92, %93 : f64
      %95 = arith.mulf %94, %in_5 : f64
      %96 = arith.addf %out, %95 : f64
      linalg.yield %96 : f64
    } -> tensor<2x5x5x4xf64>
    %62 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%13 : tensor<2x5x5x4xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<2x5x5x4xf64>
    %63 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %64 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %65 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %66 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map16} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %67 = linalg.generic {doc = "", indexing_maps = [#map3, #map13, #map3, #map13, #map3, #map13, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%64, %43, %65, %46, %66, %49, %63 : tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>) outs(%62 : tensor<2x5x5x4xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
      %90 = arith.mulf %in, %in_0 : f64
      %91 = arith.mulf %in_1, %in_2 : f64
      %92 = arith.addf %90, %91 : f64
      %93 = arith.mulf %in_3, %in_4 : f64
      %94 = arith.addf %92, %93 : f64
      %95 = arith.mulf %94, %in_5 : f64
      %96 = arith.addf %out, %95 : f64
      linalg.yield %96 : f64
    } -> tensor<2x5x5x4xf64>
    %69 = polygeist.submap(%4, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v55_contract_70_tc0 = tensor.cast %55 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v12_contract_70_tc2 = tensor.cast %12 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v70_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v55_contract_70_tc0, %69, %v12_contract_70_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %70 = tensor.cast %v70_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %72 = polygeist.submap(%3, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v61_contract_73_tc0 = tensor.cast %61 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v11_contract_73_tc2 = tensor.cast %11 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v73_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v61_contract_73_tc0, %72, %v11_contract_73_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %73 = tensor.cast %v73_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %75 = polygeist.submap(%4, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v67_contract_76_tc0 = tensor.cast %67 : tensor<2x5x5x4xf64> to tensor<?x?x?x?xf64>

    %v10_contract_76_tc2 = tensor.cast %10 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v76_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v67_contract_76_tc0, %75, %v10_contract_76_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %76 = tensor.cast %v76_tdyn : tensor<?x?x?x?xf64> to tensor<2x5x4x4xf64>
    %78 = polygeist.submap(%4, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v70_contract_79_tc0 = tensor.cast %70 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v9_contract_79_tc2 = tensor.cast %9 : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>

    %v79_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v70_contract_79_tc0, %78, %v9_contract_79_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %79 = tensor.cast %v79_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x4xf64>
    %81 = polygeist.submap(%4, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v73_contract_82_tc0 = tensor.cast %73 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v8_contract_82_tc2 = tensor.cast %8 : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>

    %v82_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v73_contract_82_tc0, %81, %v8_contract_82_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %82 = tensor.cast %v82_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x4xf64>
    %84 = polygeist.submap(%3, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v76_contract_85_tc0 = tensor.cast %76 : tensor<2x5x4x4xf64> to tensor<?x?x?x?xf64>

    %v7_contract_85_tc2 = tensor.cast %7 : tensor<2x4x4x4xf64> to tensor<?x?x?x?xf64>

    %v85_tdyn = kernel.launch @cutensornetContraction2_f64_r4r5r4(%v76_contract_85_tc0, %84, %v7_contract_85_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d1, d2)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d3, d1, d2)>]} : (tensor<?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?xf64>) -> tensor<?x?x?x?xf64>

    %85 = tensor.cast %v85_tdyn : tensor<?x?x?x?xf64> to tensor<2x4x4x4xf64>
    %86 = polygeist.submap(%0, %c2, %c4, %c4, %c4) {map = #map19} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %87 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%79, %82, %85 : tensor<2x4x4x4xf64>, tensor<2x4x4x4xf64>, tensor<2x4x4x4xf64>) outs(%86 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64):
      %90 = arith.addf %in, %in_0 : f64
      %91 = arith.addf %90, %in_1 : f64
      %92 = arith.addf %out, %91 : f64
      linalg.yield %92 : f64
    } -> tensor<?x?x?x?xf64>
    %88 = polygeist.submapInverse(%0, %87, %c2, %c4, %c4, %c4) {map = #map19} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %89 = bufferization.to_memref %88 : memref<?xf64>
    memref.copy %89, %arg6 : memref<?xf64> to memref<?xf64>
    return
  }
}
