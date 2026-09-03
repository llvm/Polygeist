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
#map12 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 50 + d1 * 5)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 50 + d1 * 5 + 25)>
#map14 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4)>
#map15 = affine_map<(d0) -> (d0)>
#map16 = affine_map<(d0) -> (0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_ex9p_mass_convection_2d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: f64, %arg8: f64, %arg9: memref<?xf64>, %arg10: memref<?xf64>, %arg11: memref<?xf64>, %arg12: memref<?xf64>, %arg13: memref<?xf64>, %arg14: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c32 = arith.constant 32 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = bufferization.to_tensor %arg4 : memref<?xf64>
    %5 = bufferization.to_tensor %arg5 : memref<?xf64>
    %6 = bufferization.to_tensor %arg6 : memref<?xf64>
    %7 = bufferization.to_tensor %arg9 : memref<?xf64>
    %8 = bufferization.to_tensor %arg10 : memref<?xf64>
    %9 = bufferization.to_tensor %arg11 : memref<?xf64>
    %10 = bufferization.to_tensor %arg12 : memref<?xf64>
    %11 = bufferization.to_tensor %arg13 : memref<?xf64>
    %12 = bufferization.to_tensor %arg14 : memref<?xf64>
    %13 = tensor.empty() : tensor<2x5x4xf64>
    %14 = tensor.empty() : tensor<2x5x5xf64>
    %15 = tensor.empty() : tensor<2x4x5xf64>
    %16 = polygeist.submap(%15, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %17 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%16 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %18 = polygeist.submapInverse(%15, %17, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x5xf64>
    %19 = polygeist.submap(%0, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %20 = polygeist.submap(%5, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %21 = polygeist.submap(%18, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v19_contract_22_tc0 = tensor.cast %19 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v20_contract_22_tc1 = tensor.cast %20 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v21_contract_22_tc2 = tensor.cast %21 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v22_tdyn = kernel.launch @cutensornetContraction2_f64(%v19_contract_22_tc0, %v20_contract_22_tc1, %v21_contract_22_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %22 = tensor.cast %v22_tdyn : tensor<*xf64> to tensor<?x?x?x?xf64>
    %23 = polygeist.submapInverse(%18, %22, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5xf64>
    %24 = polygeist.submap(%14, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %25 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%24 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %26 = polygeist.submapInverse(%14, %25, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x5xf64>
    %27 = polygeist.submap(%0, %c2, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %28 = polygeist.submap(%23, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %29 = polygeist.submap(%26, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %30 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%27, %28 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%29 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %138 = arith.mulf %in, %in_0 : f64
      %139 = arith.addf %out, %138 : f64
      linalg.yield %139 : f64
    } -> tensor<?x?x?x?xf64>
    %31 = polygeist.submapInverse(%26, %30, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5xf64>
    %32 = polygeist.submap(%3, %c2, %c5, %c5) {map = #map7} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %33 = polygeist.submap(%31, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %34 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%32 : tensor<?x?x?xf64>) outs(%33 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %138 = arith.mulf %out, %in : f64
      linalg.yield %138 : f64
    } -> tensor<?x?x?xf64>
    %35 = polygeist.submapInverse(%31, %34, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x5xf64>
    %36 = polygeist.submap(%13, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %37 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%36 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %38 = polygeist.submapInverse(%13, %37, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x4xf64>
    %39 = polygeist.submap(%2, %c2, %c5, %c4, %c5) {map = #map8} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %40 = polygeist.submap(%35, %c2, %c5, %c4, %c5) {map = #map9} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %41 = polygeist.submap(%38, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %42 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%39, %40 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%41 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %138 = arith.mulf %in, %in_0 : f64
      %139 = arith.addf %out, %138 : f64
      linalg.yield %139 : f64
    } -> tensor<?x?x?x?xf64>
    %43 = polygeist.submapInverse(%38, %42, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4xf64>
    %44 = polygeist.submap(%2, %c2, %c4, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %45 = polygeist.submap(%43, %c2, %c4, %c4, %c5) {map = #map6} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %46 = polygeist.submap(%7, %c2, %c4, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %47 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%44, %45 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%46 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %138 = arith.mulf %in, %in_0 : f64
      %139 = arith.addf %out, %138 : f64
      linalg.yield %139 : f64
    } -> tensor<?x?x?x?xf64>
    %48 = polygeist.submapInverse(%7, %47, %c2, %c4, %c4, %c5) {map = #map11} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %49 = bufferization.to_memref %48 : memref<?xf64>
    memref.copy %49, %arg9 : memref<?xf64> to memref<?xf64>
    %50 = tensor.empty() : tensor<2x4x4xf64>
    %51 = tensor.empty() : tensor<2x5x4xf64>
    %52 = tensor.empty() : tensor<2x5x5xf64>
    %53 = tensor.empty() : tensor<2x5x5xf64>
    %54 = tensor.empty() : tensor<2x4x5xf64>
    %55 = tensor.empty() : tensor<2x4x5xf64>
    %56 = polygeist.submap(%55, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %57 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%56 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %58 = polygeist.submapInverse(%55, %57, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x5xf64>
    %59 = polygeist.submap(%5, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %60 = polygeist.submap(%0, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %61 = polygeist.submap(%58, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v59_contract_62_tc0 = tensor.cast %59 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v60_contract_62_tc1 = tensor.cast %60 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v61_contract_62_tc2 = tensor.cast %61 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v62_tdyn = kernel.launch @cutensornetContraction2_f64(%v59_contract_62_tc0, %v60_contract_62_tc1, %v61_contract_62_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %62 = tensor.cast %v62_tdyn : tensor<*xf64> to tensor<?x?x?x?xf64>
    %63 = polygeist.submapInverse(%58, %62, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5xf64>
    %64 = polygeist.submap(%54, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %65 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%64 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %66 = polygeist.submapInverse(%54, %65, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x5xf64>
    %67 = polygeist.submap(%5, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %68 = polygeist.submap(%1, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %69 = polygeist.submap(%66, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v67_contract_70_tc0 = tensor.cast %67 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v68_contract_70_tc1 = tensor.cast %68 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v69_contract_70_tc2 = tensor.cast %69 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v70_tdyn = kernel.launch @cutensornetContraction2_f64(%v67_contract_70_tc0, %v68_contract_70_tc1, %v69_contract_70_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %70 = tensor.cast %v70_tdyn : tensor<*xf64> to tensor<?x?x?x?xf64>
    %71 = polygeist.submapInverse(%66, %70, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5xf64>
    %72 = polygeist.submap(%53, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %73 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%72 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %74 = polygeist.submapInverse(%53, %73, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x5xf64>
    %75 = polygeist.submap(%71, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %76 = polygeist.submap(%0, %c2, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %77 = polygeist.submap(%74, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %78 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%75, %76 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%77 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %138 = arith.mulf %in, %in_0 : f64
      %139 = arith.addf %out, %138 : f64
      linalg.yield %139 : f64
    } -> tensor<?x?x?x?xf64>
    %79 = polygeist.submapInverse(%74, %78, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5xf64>
    %80 = polygeist.submap(%52, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %81 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%80 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %82 = polygeist.submapInverse(%52, %81, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x5xf64>
    %83 = polygeist.submap(%63, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %84 = polygeist.submap(%1, %c2, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %85 = polygeist.submap(%82, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %86 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%83, %84 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%85 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %138 = arith.mulf %in, %in_0 : f64
      %139 = arith.addf %out, %138 : f64
      linalg.yield %139 : f64
    } -> tensor<?x?x?x?xf64>
    %87 = polygeist.submapInverse(%82, %86, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5xf64>
    %88 = polygeist.submap(%51, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %89 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%88 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %90 = polygeist.submapInverse(%51, %89, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x4xf64>
    %91 = polygeist.submap(%4, %c2, %c5, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %92 = polygeist.submap(%79, %c2, %c5, %c4, %c5) {map = #map9} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %93 = polygeist.submap(%4, %c2, %c5, %c4, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %94 = polygeist.submap(%87, %c2, %c5, %c4, %c5) {map = #map9} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %95 = polygeist.submap(%2, %c2, %c5, %c4, %c5) {map = #map8} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %96 = polygeist.submap(%90, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %97 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%91, %92, %93, %94, %95 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%96 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
      %138 = arith.mulf %in, %in_0 : f64
      %139 = arith.mulf %in_1, %in_2 : f64
      %140 = arith.addf %138, %139 : f64
      %141 = arith.mulf %140, %in_3 : f64
      %142 = arith.addf %out, %141 : f64
      linalg.yield %142 : f64
    } -> tensor<?x?x?x?xf64>
    %98 = polygeist.submapInverse(%90, %97, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4xf64>
    %99 = polygeist.submap(%50, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %100 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%99 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %101 = polygeist.submapInverse(%50, %100, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x4xf64>
    %102 = polygeist.submap(%98, %c2, %c4, %c4, %c5) {map = #map6} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %103 = polygeist.submap(%2, %c2, %c4, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %104 = polygeist.submap(%101, %c2, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %105 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%102, %103 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%104 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %138 = arith.mulf %in, %in_0 : f64
      %139 = arith.addf %out, %138 : f64
      linalg.yield %139 : f64
    } -> tensor<?x?x?x?xf64>
    %106 = polygeist.submapInverse(%101, %105, %c2, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4xf64>
    %107 = polygeist.submap(%106, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %108 = polygeist.submap(%8, %c2, %c4, %c4) {map = #map14} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %109 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%107 : tensor<?x?x?xf64>) outs(%108 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %138 = arith.addf %out, %in : f64
      linalg.yield %138 : f64
    } -> tensor<?x?x?xf64>
    %110 = polygeist.submapInverse(%8, %109, %c2, %c4, %c4) {map = #map14} : (tensor<?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?xf64>
    %111 = polygeist.submap(%11, %c32) {map = #map15} : (tensor<?xf64>, index) -> tensor<?xf64>
    %112 = polygeist.submap(%110, %c32) {map = #map15} : (tensor<?xf64>, index) -> tensor<?xf64>
    %v113_beta = arith.constant 1.0 : f64

    %113 = kernel.launch @cublasDaxpby(%111, %112, %arg7, %v113_beta) : (tensor<?xf64>, tensor<?xf64>, f64, f64) -> tensor<?xf64>
    %114 = polygeist.submapInverse(%110, %113, %c32) {map = #map15} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %115 = bufferization.to_memref %114 : memref<?xf64>
    memref.copy %115, %arg10 : memref<?xf64> to memref<?xf64>
    %116 = polygeist.submap(%48, %c32) {map = #map15} : (tensor<?xf64>, index) -> tensor<?xf64>
    %117 = polygeist.submap(%9, %c32) {map = #map15} : (tensor<?xf64>, index) -> tensor<?xf64>
    %118 = linalg.generic {doc = "", indexing_maps = [#map15, #map15], iterator_types = ["parallel"], library_call = ""} ins(%116 : tensor<?xf64>) outs(%117 : tensor<?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %138 = arith.mulf %arg7, %in : f64
      %139 = arith.subf %out, %138 : f64
      linalg.yield %139 : f64
    } -> tensor<?xf64>
    %119 = polygeist.submapInverse(%9, %118, %c32) {map = #map15} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %120 = bufferization.to_memref %119 : memref<?xf64>
    memref.copy %120, %arg11 : memref<?xf64> to memref<?xf64>
    %121 = polygeist.submap(%6, %c32) {map = #map15} : (tensor<?xf64>, index) -> tensor<?xf64>
    %122 = polygeist.submap(%119, %c32) {map = #map15} : (tensor<?xf64>, index) -> tensor<?xf64>
    %123 = polygeist.submap(%10, %c32) {map = #map15} : (tensor<?xf64>, index) -> tensor<?xf64>
    %124 = linalg.generic {doc = "", indexing_maps = [#map15, #map15, #map15], iterator_types = ["parallel"], library_call = ""} ins(%121, %122 : tensor<?xf64>, tensor<?xf64>) outs(%123 : tensor<?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %138 = arith.mulf %in, %in_0 : f64
      linalg.yield %138 : f64
    } -> tensor<?xf64>
    %125 = polygeist.submapInverse(%10, %124, %c32) {map = #map15} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %126 = bufferization.to_memref %125 : memref<?xf64>
    memref.copy %126, %arg12 : memref<?xf64> to memref<?xf64>
    %inserted = tensor.insert %cst into %12[%c0] : tensor<?xf64>
    %127 = polygeist.submap(%119, %c32) {map = #map15} : (tensor<?xf64>, index) -> tensor<?xf64>
    %128 = polygeist.submap(%125, %c32) {map = #map15} : (tensor<?xf64>, index) -> tensor<?xf64>
    %129 = polygeist.submap(%inserted, %c32) {map = #map16} : (tensor<?xf64>, index) -> tensor<?xf64>
    %130 = linalg.generic {doc = "", indexing_maps = [#map15, #map15, #map15], iterator_types = ["reduction"], library_call = ""} ins(%127, %128 : tensor<?xf64>, tensor<?xf64>) outs(%129 : tensor<?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %138 = arith.mulf %in, %in_0 : f64
      %139 = arith.addf %out, %138 : f64
      linalg.yield %139 : f64
    } -> tensor<?xf64>
    %131 = polygeist.submapInverse(%inserted, %130, %c32) {map = #map16} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %132 = bufferization.to_memref %131 : memref<?xf64>
    memref.copy %132, %arg14 : memref<?xf64> to memref<?xf64>
    %133 = polygeist.submap(%125, %c32) {map = #map15} : (tensor<?xf64>, index) -> tensor<?xf64>
    %134 = polygeist.submap(%11, %c32) {map = #map15} : (tensor<?xf64>, index) -> tensor<?xf64>
    %v135_alpha = arith.constant 1.0 : f64

    %135 = kernel.launch @cublasDaxpby(%133, %134, %v135_alpha, %arg8) : (tensor<?xf64>, tensor<?xf64>, f64, f64) -> tensor<?xf64>
    %136 = polygeist.submapInverse(%11, %135, %c32) {map = #map15} : (tensor<?xf64>, tensor<?xf64>, index) -> tensor<?xf64>
    %137 = bufferization.to_memref %136 : memref<?xf64>
    memref.copy %137, %arg13 : memref<?xf64> to memref<?xf64>
    return
  }
}
