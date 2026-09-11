#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 50 + d1 * 5)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 50 + d1 * 5 + 25)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 5)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 5)>
#map12 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_convection_apply_2d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = bufferization.to_tensor %arg4 : memref<?xf64>
    %5 = bufferization.to_tensor %arg5 : memref<?xf64>
    %6 = tensor.empty() : tensor<2x4x4xf64>
    %7 = tensor.empty() : tensor<2x5x4xf64>
    %8 = tensor.empty() : tensor<2x5x5xf64>
    %9 = tensor.empty() : tensor<2x5x5xf64>
    %10 = tensor.empty() : tensor<2x4x5xf64>
    %11 = tensor.empty() : tensor<2x4x5xf64>
    %12 = polygeist.submap(%11, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %13 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%12 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %14 = polygeist.submapInverse(%11, %13, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x5xf64>
    %15 = polygeist.submap(%4, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %16 = polygeist.submap(%0, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %17 = polygeist.submap(%14, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v15_contract_18_tc0 = tensor.cast %15 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v16_contract_18_tc1 = tensor.cast %16 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v17_contract_18_tc2 = tensor.cast %17 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v18_tdyn = kernel.launch @cutensornetContraction2_f64(%v15_contract_18_tc0, %v16_contract_18_tc1, %v17_contract_18_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %18 = tensor.cast %v18_tdyn : tensor<*xf64> to tensor<?x?x?x?xf64>
    %19 = polygeist.submapInverse(%14, %18, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5xf64>
    %20 = polygeist.submap(%10, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %21 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%20 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %22 = polygeist.submapInverse(%10, %21, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x5xf64>
    %23 = polygeist.submap(%4, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %24 = polygeist.submap(%1, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %25 = polygeist.submap(%22, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v23_contract_26_tc0 = tensor.cast %23 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v24_contract_26_tc1 = tensor.cast %24 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v25_contract_26_tc2 = tensor.cast %25 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v26_tdyn = kernel.launch @cutensornetContraction2_f64(%v23_contract_26_tc0, %v24_contract_26_tc1, %v25_contract_26_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %26 = tensor.cast %v26_tdyn : tensor<*xf64> to tensor<?x?x?x?xf64>
    %27 = polygeist.submapInverse(%22, %26, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5xf64>
    %28 = polygeist.submap(%9, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %29 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%28 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %30 = polygeist.submapInverse(%9, %29, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x5xf64>
    %31 = polygeist.submap(%27, %c2, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %32 = polygeist.submap(%0, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %33 = polygeist.submap(%30, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %34 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%31, %32 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%33 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %68 = arith.mulf %in, %in_0 : f64
      %69 = arith.addf %out, %68 : f64
      linalg.yield %69 : f64
    } -> tensor<?x?x?x?xf64>
    %35 = polygeist.submapInverse(%30, %34, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5xf64>
    %36 = polygeist.submap(%8, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %37 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%36 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %38 = polygeist.submapInverse(%8, %37, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x5xf64>
    %39 = polygeist.submap(%19, %c2, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %40 = polygeist.submap(%1, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %41 = polygeist.submap(%38, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %42 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%39, %40 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%41 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %68 = arith.mulf %in, %in_0 : f64
      %69 = arith.addf %out, %68 : f64
      linalg.yield %69 : f64
    } -> tensor<?x?x?x?xf64>
    %43 = polygeist.submapInverse(%38, %42, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5xf64>
    %44 = polygeist.submap(%7, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %45 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%44 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %46 = polygeist.submapInverse(%7, %45, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x4xf64>
    %47 = polygeist.submap(%3, %c2, %c5, %c4, %c5) {map = #map7} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %48 = polygeist.submap(%35, %c2, %c5, %c4, %c5) {map = #map8} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %49 = polygeist.submap(%3, %c2, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %50 = polygeist.submap(%43, %c2, %c5, %c4, %c5) {map = #map8} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %51 = polygeist.submap(%2, %c2, %c5, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %52 = polygeist.submap(%46, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %53 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%47, %48, %49, %50, %51 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%52 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
      %68 = arith.mulf %in, %in_0 : f64
      %69 = arith.mulf %in_1, %in_2 : f64
      %70 = arith.addf %68, %69 : f64
      %71 = arith.mulf %70, %in_3 : f64
      %72 = arith.addf %out, %71 : f64
      linalg.yield %72 : f64
    } -> tensor<?x?x?x?xf64>
    %54 = polygeist.submapInverse(%46, %53, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4xf64>
    %55 = polygeist.submap(%6, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %56 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%55 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %57 = polygeist.submapInverse(%6, %56, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x4xf64>
    %58 = polygeist.submap(%54, %c2, %c4, %c4, %c5) {map = #map5} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %59 = polygeist.submap(%2, %c2, %c4, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %60 = polygeist.submap(%57, %c2, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %61 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%58, %59 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%60 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %68 = arith.mulf %in, %in_0 : f64
      %69 = arith.addf %out, %68 : f64
      linalg.yield %69 : f64
    } -> tensor<?x?x?x?xf64>
    %62 = polygeist.submapInverse(%57, %61, %c2, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4xf64>
    %63 = polygeist.submap(%62, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %64 = polygeist.submap(%5, %c2, %c4, %c4) {map = #map12} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %65 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%63 : tensor<?x?x?xf64>) outs(%64 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %68 = arith.addf %out, %in : f64
      linalg.yield %68 : f64
    } -> tensor<?x?x?xf64>
    %66 = polygeist.submapInverse(%5, %65, %c2, %c4, %c4) {map = #map12} : (tensor<?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?xf64>
    %67 = bufferization.to_memref %66 : memref<?xf64>
    memref.copy %67, %arg5 : memref<?xf64> to memref<?xf64>
    return
  }
}
