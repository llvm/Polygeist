#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 64 + d1 * 16 + d2 * 4)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d2 * 5 + d0 * 750)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 125)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 250)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 5)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 375)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 500)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 25 + d0 * 750 + d2 * 5 + 625)>
#map17 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 5)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 5)>
#map19 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 64 + d1 * 16 + d2 * 4)>
#map20 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 125 + d1 * 25 + d2 * 5)>
#map21 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d0 * 64 + d1 * 16 + d2 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_ex35p_h1_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = bufferization.to_tensor %arg4 : memref<?xf64>
    %5 = bufferization.to_tensor %arg5 : memref<?xf64>
    %6 = bufferization.to_tensor %arg6 : memref<?xf64>
    %7 = bufferization.to_tensor %arg7 : memref<?xf64>
    %8 = tensor.empty() : tensor<2x4x4x4xf64>
    %9 = tensor.empty() : tensor<2x4x4x4xf64>
    %10 = tensor.empty() : tensor<2x4x4x4xf64>
    %11 = tensor.empty() : tensor<2x5x4x4xf64>
    %12 = tensor.empty() : tensor<2x5x4x4xf64>
    %13 = tensor.empty() : tensor<2x5x4x4xf64>
    %14 = tensor.empty() : tensor<2x5x5x4xf64>
    %15 = tensor.empty() : tensor<2x5x5x4xf64>
    %16 = tensor.empty() : tensor<2x5x5x4xf64>
    %17 = tensor.empty() : tensor<2x5x5x5xf64>
    %18 = tensor.empty() : tensor<2x5x5x5xf64>
    %19 = tensor.empty() : tensor<2x5x5x5xf64>
    %20 = tensor.empty() : tensor<2x4x5x5xf64>
    %21 = tensor.empty() : tensor<2x4x5x5xf64>
    %22 = tensor.empty() : tensor<2x4x5x5xf64>
    %23 = tensor.empty() : tensor<2x4x4x5xf64>
    %24 = tensor.empty() : tensor<2x4x4x5xf64>
    %25 = polygeist.submap(%24, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %26 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%25 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %27 = polygeist.submapInverse(%24, %26, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x5xf64>
    %28 = polygeist.submap(%6, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %29 = polygeist.submap(%0, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %30 = polygeist.submap(%27, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v28_contract_31_tc0 = tensor.cast %28 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v29_contract_31_tc1 = tensor.cast %29 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v30_contract_31_tc2 = tensor.cast %30 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v31_tdyn = kernel.launch @cutensornetContraction2_f64(%v28_contract_31_tc0, %v29_contract_31_tc1, %v30_contract_31_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %31 = tensor.cast %v31_tdyn : tensor<*xf64> to tensor<?x?x?x?x?xf64>
    %32 = polygeist.submapInverse(%27, %31, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x5xf64>
    %33 = polygeist.submap(%23, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %34 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%33 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %35 = polygeist.submapInverse(%23, %34, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x5xf64>
    %36 = polygeist.submap(%6, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %37 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %38 = polygeist.submap(%35, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v36_contract_39_tc0 = tensor.cast %36 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v37_contract_39_tc1 = tensor.cast %37 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v38_contract_39_tc2 = tensor.cast %38 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v39_tdyn = kernel.launch @cutensornetContraction2_f64(%v36_contract_39_tc0, %v37_contract_39_tc1, %v38_contract_39_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %39 = tensor.cast %v39_tdyn : tensor<*xf64> to tensor<?x?x?x?x?xf64>
    %40 = polygeist.submapInverse(%35, %39, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x5xf64>
    %41 = polygeist.submap(%22, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %42 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%41 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %43 = polygeist.submapInverse(%22, %42, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %44 = polygeist.submap(%40, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %45 = polygeist.submap(%0, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %46 = polygeist.submap(%43, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %47 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%44, %45 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%46 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %48 = polygeist.submapInverse(%43, %47, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %49 = polygeist.submap(%21, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %50 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%49 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %51 = polygeist.submapInverse(%21, %50, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %52 = polygeist.submap(%32, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %53 = polygeist.submap(%1, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %54 = polygeist.submap(%51, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %55 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%52, %53 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%54 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %56 = polygeist.submapInverse(%51, %55, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %57 = polygeist.submap(%20, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %58 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%57 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %59 = polygeist.submapInverse(%20, %58, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %60 = polygeist.submap(%32, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %61 = polygeist.submap(%0, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %62 = polygeist.submap(%59, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %63 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%60, %61 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%62 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %64 = polygeist.submapInverse(%59, %63, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %65 = polygeist.submap(%19, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %66 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%65 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %67 = polygeist.submapInverse(%19, %66, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %68 = polygeist.submap(%48, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %69 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %70 = polygeist.submap(%67, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %71 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%68, %69 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%70 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %72 = polygeist.submapInverse(%67, %71, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x5xf64>
    %73 = polygeist.submap(%18, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %74 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%73 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %75 = polygeist.submapInverse(%18, %74, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %76 = polygeist.submap(%56, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %77 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %78 = polygeist.submap(%75, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %79 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%76, %77 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%78 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %80 = polygeist.submapInverse(%75, %79, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x5xf64>
    %81 = polygeist.submap(%17, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %82 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%81 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %83 = polygeist.submapInverse(%17, %82, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %84 = polygeist.submap(%64, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %85 = polygeist.submap(%1, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %86 = polygeist.submap(%83, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %87 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%84, %85 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%86 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %88 = polygeist.submapInverse(%83, %87, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x5xf64>
    %89 = polygeist.submap(%16, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %90 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%89 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %91 = polygeist.submapInverse(%16, %90, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x4xf64>
    %92 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %93 = polygeist.submap(%72, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %94 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %95 = polygeist.submap(%80, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %96 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %97 = polygeist.submap(%88, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %98 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %99 = polygeist.submap(%91, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %100 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%92, %93, %94, %95, %96, %97, %98 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%99 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.mulf %in_1, %in_2 : f64
      %239 = arith.addf %237, %238 : f64
      %240 = arith.mulf %in_3, %in_4 : f64
      %241 = arith.addf %239, %240 : f64
      %242 = arith.mulf %241, %in_5 : f64
      %243 = arith.addf %out, %242 : f64
      linalg.yield %243 : f64
    } -> tensor<?x?x?x?x?xf64>
    %101 = polygeist.submapInverse(%91, %100, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x4xf64>
    %102 = polygeist.submap(%15, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %103 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%102 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %104 = polygeist.submapInverse(%15, %103, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x4xf64>
    %105 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %106 = polygeist.submap(%72, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %107 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %108 = polygeist.submap(%80, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %109 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %110 = polygeist.submap(%88, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %111 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %112 = polygeist.submap(%104, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %113 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%105, %106, %107, %108, %109, %110, %111 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%112 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.mulf %in_1, %in_2 : f64
      %239 = arith.addf %237, %238 : f64
      %240 = arith.mulf %in_3, %in_4 : f64
      %241 = arith.addf %239, %240 : f64
      %242 = arith.mulf %241, %in_5 : f64
      %243 = arith.addf %out, %242 : f64
      linalg.yield %243 : f64
    } -> tensor<?x?x?x?x?xf64>
    %114 = polygeist.submapInverse(%104, %113, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x4xf64>
    %115 = polygeist.submap(%14, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %116 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%115 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %117 = polygeist.submapInverse(%14, %116, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x4xf64>
    %118 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %119 = polygeist.submap(%72, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %120 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %121 = polygeist.submap(%80, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %122 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map16} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %123 = polygeist.submap(%88, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %124 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %125 = polygeist.submap(%117, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %126 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%118, %119, %120, %121, %122, %123, %124 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%125 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.mulf %in_1, %in_2 : f64
      %239 = arith.addf %237, %238 : f64
      %240 = arith.mulf %in_3, %in_4 : f64
      %241 = arith.addf %239, %240 : f64
      %242 = arith.mulf %241, %in_5 : f64
      %243 = arith.addf %out, %242 : f64
      linalg.yield %243 : f64
    } -> tensor<?x?x?x?x?xf64>
    %127 = polygeist.submapInverse(%117, %126, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x4xf64>
    %128 = polygeist.submap(%13, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %129 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%128 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %130 = polygeist.submapInverse(%13, %129, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4x4xf64>
    %131 = polygeist.submap(%101, %c2, %c5, %c4, %c4, %c5) {map = #map5} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %132 = polygeist.submap(%2, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %133 = polygeist.submap(%130, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %134 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%131, %132 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%133 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %135 = polygeist.submapInverse(%130, %134, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x4x4xf64>
    %136 = polygeist.submap(%12, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %137 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%136 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %138 = polygeist.submapInverse(%12, %137, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4x4xf64>
    %139 = polygeist.submap(%114, %c2, %c5, %c4, %c4, %c5) {map = #map5} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %140 = polygeist.submap(%3, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %141 = polygeist.submap(%138, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %142 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%139, %140 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%141 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %143 = polygeist.submapInverse(%138, %142, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x4x4xf64>
    %144 = polygeist.submap(%11, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %145 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%144 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %146 = polygeist.submapInverse(%11, %145, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4x4xf64>
    %147 = polygeist.submap(%127, %c2, %c5, %c4, %c4, %c5) {map = #map5} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %148 = polygeist.submap(%2, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %149 = polygeist.submap(%146, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %150 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%147, %148 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%149 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %151 = polygeist.submapInverse(%146, %150, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x4x4xf64>
    %152 = polygeist.submap(%10, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %153 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%152 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %154 = polygeist.submapInverse(%10, %153, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x4xf64>
    %155 = polygeist.submap(%135, %c2, %c4, %c4, %c4, %c5) {map = #map7} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %156 = polygeist.submap(%2, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %157 = polygeist.submap(%154, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %158 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%155, %156 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%157 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %159 = polygeist.submapInverse(%154, %158, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x4xf64>
    %160 = polygeist.submap(%9, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %161 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%160 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %162 = polygeist.submapInverse(%9, %161, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x4xf64>
    %163 = polygeist.submap(%143, %c2, %c4, %c4, %c4, %c5) {map = #map7} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %164 = polygeist.submap(%2, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %165 = polygeist.submap(%162, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %166 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%163, %164 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%165 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %167 = polygeist.submapInverse(%162, %166, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x4xf64>
    %168 = polygeist.submap(%8, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %169 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%168 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %170 = polygeist.submapInverse(%8, %169, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x4xf64>
    %171 = polygeist.submap(%151, %c2, %c4, %c4, %c4, %c5) {map = #map7} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %172 = polygeist.submap(%3, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %173 = polygeist.submap(%170, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %174 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%171, %172 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%173 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %175 = polygeist.submapInverse(%170, %174, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x4xf64>
    %176 = polygeist.submap(%159, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %177 = polygeist.submap(%167, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %178 = polygeist.submap(%175, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %179 = polygeist.submap(%7, %c2, %c4, %c4, %c4) {map = #map19} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %180 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%176, %177, %178 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%179 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64):
      %237 = arith.addf %in, %in_0 : f64
      %238 = arith.addf %237, %in_1 : f64
      %239 = arith.addf %out, %238 : f64
      linalg.yield %239 : f64
    } -> tensor<?x?x?x?xf64>
    %181 = polygeist.submapInverse(%7, %180, %c2, %c4, %c4, %c4) {map = #map19} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %182 = tensor.empty() : tensor<2x5x4x4xf64>
    %183 = tensor.empty() : tensor<2x5x5x4xf64>
    %184 = tensor.empty() : tensor<2x5x5x5xf64>
    %185 = tensor.empty() : tensor<2x4x5x5xf64>
    %186 = tensor.empty() : tensor<2x4x4x5xf64>
    %187 = polygeist.submap(%186, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %188 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%187 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %189 = polygeist.submapInverse(%186, %188, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x5xf64>
    %190 = polygeist.submap(%0, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %191 = polygeist.submap(%6, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %192 = polygeist.submap(%189, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v190_contract_193_tc0 = tensor.cast %190 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v191_contract_193_tc1 = tensor.cast %191 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v192_contract_193_tc2 = tensor.cast %192 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v193_tdyn = kernel.launch @cutensornetContraction2_f64(%v190_contract_193_tc0, %v191_contract_193_tc1, %v192_contract_193_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %193 = tensor.cast %v193_tdyn : tensor<*xf64> to tensor<?x?x?x?x?xf64>
    %194 = polygeist.submapInverse(%189, %193, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x5xf64>
    %195 = polygeist.submap(%185, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %196 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%195 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %197 = polygeist.submapInverse(%185, %196, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %198 = polygeist.submap(%0, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %199 = polygeist.submap(%194, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %200 = polygeist.submap(%197, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %201 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%198, %199 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%200 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %202 = polygeist.submapInverse(%197, %201, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %203 = polygeist.submap(%184, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %204 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%203 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %205 = polygeist.submapInverse(%184, %204, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %206 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %207 = polygeist.submap(%202, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %208 = polygeist.submap(%205, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %209 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%206, %207 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%208 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %210 = polygeist.submapInverse(%205, %209, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x5xf64>
    %211 = polygeist.submap(%5, %c2, %c5, %c5, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %212 = polygeist.submap(%210, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %213 = linalg.generic {doc = "", indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%211 : tensor<?x?x?x?xf64>) outs(%212 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %out: f64):
      %237 = arith.mulf %out, %in : f64
      linalg.yield %237 : f64
    } -> tensor<?x?x?x?xf64>
    %214 = polygeist.submapInverse(%210, %213, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %215 = polygeist.submap(%183, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %216 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%215 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %217 = polygeist.submapInverse(%183, %216, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x4xf64>
    %218 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %219 = polygeist.submap(%214, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %220 = polygeist.submap(%217, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %221 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%218, %219 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%220 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %222 = polygeist.submapInverse(%217, %221, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x4xf64>
    %223 = polygeist.submap(%182, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %224 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%223 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %225 = polygeist.submapInverse(%182, %224, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4x4xf64>
    %226 = polygeist.submap(%2, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %227 = polygeist.submap(%222, %c2, %c5, %c4, %c4, %c5) {map = #map5} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %228 = polygeist.submap(%225, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %229 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%226, %227 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%228 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %230 = polygeist.submapInverse(%225, %229, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x4x4xf64>
    %231 = polygeist.submap(%2, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %232 = polygeist.submap(%230, %c2, %c4, %c4, %c4, %c5) {map = #map7} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %233 = polygeist.submap(%181, %c2, %c4, %c4, %c4, %c5) {map = #map21} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %234 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%231, %232 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%233 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %237 = arith.mulf %in, %in_0 : f64
      %238 = arith.addf %out, %237 : f64
      linalg.yield %238 : f64
    } -> tensor<?x?x?x?x?xf64>
    %235 = polygeist.submapInverse(%181, %234, %c2, %c4, %c4, %c4, %c5) {map = #map21} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %236 = bufferization.to_memref %235 : memref<?xf64>
    memref.copy %236, %arg7 : memref<?xf64> to memref<?xf64>
    return
  }
}
