#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 64 + d1 * 16 + d2 * 4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 125 + d1 * 25 + d2 * 5)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 5)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 5)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 5)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d0 * 64 + d1 * 16 + d2 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_abs_l1_mass_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = bufferization.to_tensor %arg4 : memref<?xf64>
    %5 = tensor.empty() : tensor<2x5x4x4xf64>
    %6 = tensor.empty() : tensor<2x5x5x4xf64>
    %7 = tensor.empty() : tensor<2x5x5x5xf64>
    %8 = tensor.empty() : tensor<2x4x5x5xf64>
    %9 = tensor.empty() : tensor<2x4x4x5xf64>
    %10 = polygeist.submap(%9, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %11 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%10 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %12 = polygeist.submapInverse(%9, %11, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x5xf64>
    %13 = polygeist.submap(%0, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %14 = polygeist.submap(%3, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %15 = polygeist.submap(%12, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v13_contract_16_tc0 = tensor.cast %13 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v14_contract_16_tc1 = tensor.cast %14 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v15_contract_16_tc2 = tensor.cast %15 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v16_tdyn = kernel.launch @cutensornetContraction2_f64(%v13_contract_16_tc0, %v14_contract_16_tc1, %v15_contract_16_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %16 = tensor.cast %v16_tdyn : tensor<*xf64> to tensor<?x?x?x?x?xf64>
    %17 = polygeist.submapInverse(%12, %16, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x5xf64>
    %18 = polygeist.submap(%8, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %19 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%18 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %20 = polygeist.submapInverse(%8, %19, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %21 = polygeist.submap(%0, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %22 = polygeist.submap(%17, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %23 = polygeist.submap(%20, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %24 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%21, %22 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%23 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %60 = arith.mulf %in, %in_0 : f64
      %61 = arith.addf %out, %60 : f64
      linalg.yield %61 : f64
    } -> tensor<?x?x?x?x?xf64>
    %25 = polygeist.submapInverse(%20, %24, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %26 = polygeist.submap(%7, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %27 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%26 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %28 = polygeist.submapInverse(%7, %27, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %29 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %30 = polygeist.submap(%25, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %31 = polygeist.submap(%28, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %32 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%29, %30 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%31 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %60 = arith.mulf %in, %in_0 : f64
      %61 = arith.addf %out, %60 : f64
      linalg.yield %61 : f64
    } -> tensor<?x?x?x?x?xf64>
    %33 = polygeist.submapInverse(%28, %32, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x5xf64>
    %34 = polygeist.submap(%2, %c2, %c5, %c5, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %35 = polygeist.submap(%33, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %36 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%34 : tensor<?x?x?x?xf64>) outs(%35 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %60 = arith.mulf %out, %in : f64
      linalg.yield %60 : f64
    } -> tensor<?x?x?x?xf64>
    %37 = polygeist.submapInverse(%33, %36, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %38 = polygeist.submap(%6, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %39 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%38 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %40 = polygeist.submapInverse(%6, %39, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x4xf64>
    %41 = polygeist.submap(%1, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %42 = polygeist.submap(%37, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %43 = polygeist.submap(%40, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %44 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%41, %42 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%43 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %60 = arith.mulf %in, %in_0 : f64
      %61 = arith.addf %out, %60 : f64
      linalg.yield %61 : f64
    } -> tensor<?x?x?x?x?xf64>
    %45 = polygeist.submapInverse(%40, %44, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x4xf64>
    %46 = polygeist.submap(%5, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %47 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%46 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %48 = polygeist.submapInverse(%5, %47, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4x4xf64>
    %49 = polygeist.submap(%1, %c2, %c5, %c4, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %50 = polygeist.submap(%45, %c2, %c5, %c4, %c4, %c5) {map = #map6} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %51 = polygeist.submap(%48, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %52 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%49, %50 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%51 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %60 = arith.mulf %in, %in_0 : f64
      %61 = arith.addf %out, %60 : f64
      linalg.yield %61 : f64
    } -> tensor<?x?x?x?x?xf64>
    %53 = polygeist.submapInverse(%48, %52, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x4x4xf64>
    %54 = polygeist.submap(%1, %c2, %c4, %c4, %c4, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %55 = polygeist.submap(%53, %c2, %c4, %c4, %c4, %c5) {map = #map8} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %56 = polygeist.submap(%4, %c2, %c4, %c4, %c4, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %57 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%54, %55 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%56 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %60 = arith.mulf %in, %in_0 : f64
      %61 = arith.addf %out, %60 : f64
      linalg.yield %61 : f64
    } -> tensor<?x?x?x?x?xf64>
    %58 = polygeist.submapInverse(%4, %57, %c2, %c4, %c4, %c4, %c5) {map = #map14} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %59 = bufferization.to_memref %58 : memref<?xf64>
    memref.copy %59, %arg4 : memref<?xf64> to memref<?xf64>
    return
  }
}
