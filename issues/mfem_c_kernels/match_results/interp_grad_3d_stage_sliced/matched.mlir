#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 64 + d1 * 16 + d2 * 4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d3 * 25 + d1 + d0 * 375 + d2 * 5)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d3 * 25 + d1 + d0 * 375 + d2 * 5)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d3 * 25 + d1 + d0 * 375 + d2 * 5 + 125)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d3 * 25 + d1 + d0 * 375 + d2 * 5 + 125)>
#map13 = affine_map<(d0, d1, d2, d3) -> (d3 * 25 + d1 + d0 * 375 + d2 * 5 + 250)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d3 * 25 + d1 + d0 * 375 + d2 * 5 + 250)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_interp_grad_3d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = tensor.empty() : tensor<2x4x5x5xf64>
    %5 = tensor.empty() : tensor<2x4x5x5xf64>
    %6 = tensor.empty() : tensor<2x4x5x5xf64>
    %7 = tensor.empty() : tensor<2x4x4x5xf64>
    %8 = tensor.empty() : tensor<2x4x4x5xf64>
    %9 = polygeist.submap(%8, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %10 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%9 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %11 = polygeist.submapInverse(%8, %10, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x5xf64>
    %12 = polygeist.submap(%0, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %13 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %14 = polygeist.submap(%11, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v12_contract_15_tc0 = tensor.cast %12 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v13_contract_15_tc1 = tensor.cast %13 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v14_contract_15_tc2 = tensor.cast %14 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v15_tdyn = kernel.launch @cutensornetContraction2_f64(%v12_contract_15_tc0, %v13_contract_15_tc1, %v14_contract_15_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %15 = tensor.cast %v15_tdyn : tensor<*xf64> to tensor<?x?x?x?x?xf64>
    %16 = polygeist.submapInverse(%11, %15, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x5xf64>
    %17 = polygeist.submap(%7, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %18 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%17 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %19 = polygeist.submapInverse(%7, %18, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x5xf64>
    %20 = polygeist.submap(%0, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %21 = polygeist.submap(%2, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %22 = polygeist.submap(%19, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v20_contract_23_tc0 = tensor.cast %20 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v21_contract_23_tc1 = tensor.cast %21 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v22_contract_23_tc2 = tensor.cast %22 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v23_tdyn = kernel.launch @cutensornetContraction2_f64(%v20_contract_23_tc0, %v21_contract_23_tc1, %v22_contract_23_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %23 = tensor.cast %v23_tdyn : tensor<*xf64> to tensor<?x?x?x?x?xf64>
    %24 = polygeist.submapInverse(%19, %23, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x5xf64>
    %25 = polygeist.submap(%6, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %26 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%25 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %27 = polygeist.submapInverse(%6, %26, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %28 = polygeist.submap(%24, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %29 = polygeist.submap(%1, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %30 = polygeist.submap(%27, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %31 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%28, %29 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%30 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %74 = arith.mulf %in, %in_0 : f64
      %75 = arith.addf %out, %74 : f64
      linalg.yield %75 : f64
    } -> tensor<?x?x?x?x?xf64>
    %32 = polygeist.submapInverse(%27, %31, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %33 = polygeist.submap(%5, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %34 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%33 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %35 = polygeist.submapInverse(%5, %34, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %36 = polygeist.submap(%16, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %37 = polygeist.submap(%2, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %38 = polygeist.submap(%35, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %39 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%36, %37 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%38 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %74 = arith.mulf %in, %in_0 : f64
      %75 = arith.addf %out, %74 : f64
      linalg.yield %75 : f64
    } -> tensor<?x?x?x?x?xf64>
    %40 = polygeist.submapInverse(%35, %39, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %41 = polygeist.submap(%4, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %42 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%41 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %43 = polygeist.submapInverse(%4, %42, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %44 = polygeist.submap(%16, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %45 = polygeist.submap(%1, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %46 = polygeist.submap(%43, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %47 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%44, %45 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%46 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %74 = arith.mulf %in, %in_0 : f64
      %75 = arith.addf %out, %74 : f64
      linalg.yield %75 : f64
    } -> tensor<?x?x?x?x?xf64>
    %48 = polygeist.submapInverse(%43, %47, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %49 = polygeist.submap(%3, %c2, %c5, %c5, %c5) {map = #map7} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %50 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%49 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %51 = polygeist.submapInverse(%3, %50, %c2, %c5, %c5, %c5) {map = #map7} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %52 = polygeist.submap(%32, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %53 = polygeist.submap(%1, %c2, %c5, %c5, %c5, %c4) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %54 = polygeist.submap(%51, %c2, %c5, %c5, %c5, %c4) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %55 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%52, %53 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%54 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %74 = arith.mulf %in, %in_0 : f64
      %75 = arith.addf %out, %74 : f64
      linalg.yield %75 : f64
    } -> tensor<?x?x?x?x?xf64>
    %56 = polygeist.submapInverse(%51, %55, %c2, %c5, %c5, %c5, %c4) {map = #map10} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %57 = polygeist.submap(%56, %c2, %c5, %c5, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %58 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%57 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %59 = polygeist.submapInverse(%56, %58, %c2, %c5, %c5, %c5) {map = #map11} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %60 = polygeist.submap(%40, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %61 = polygeist.submap(%1, %c2, %c5, %c5, %c5, %c4) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %62 = polygeist.submap(%59, %c2, %c5, %c5, %c5, %c4) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %63 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%60, %61 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%62 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %74 = arith.mulf %in, %in_0 : f64
      %75 = arith.addf %out, %74 : f64
      linalg.yield %75 : f64
    } -> tensor<?x?x?x?x?xf64>
    %64 = polygeist.submapInverse(%59, %63, %c2, %c5, %c5, %c5, %c4) {map = #map12} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %65 = polygeist.submap(%64, %c2, %c5, %c5, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %66 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%65 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %67 = polygeist.submapInverse(%64, %66, %c2, %c5, %c5, %c5) {map = #map13} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %68 = polygeist.submap(%48, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %69 = polygeist.submap(%2, %c2, %c5, %c5, %c5, %c4) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %70 = polygeist.submap(%67, %c2, %c5, %c5, %c5, %c4) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %71 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%68, %69 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%70 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %74 = arith.mulf %in, %in_0 : f64
      %75 = arith.addf %out, %74 : f64
      linalg.yield %75 : f64
    } -> tensor<?x?x?x?x?xf64>
    %72 = polygeist.submapInverse(%67, %71, %c2, %c5, %c5, %c5, %c4) {map = #map14} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %73 = bufferization.to_memref %72 : memref<?xf64>
    memref.copy %73, %arg3 : memref<?xf64> to memref<?xf64>
    return
  }
}
