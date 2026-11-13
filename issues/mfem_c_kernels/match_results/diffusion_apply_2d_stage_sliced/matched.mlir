#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 16 + d1 * 4)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 4)>
#map3 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 4)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 5 + d0 * 75)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 75 + d1 * 5 + 25)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d3 + d2 * 5)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d3 + d0 * 75 + d1 * 5 + 50)>
#map12 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 5)>
#map13 = affine_map<(d0, d1, d2) -> (d2 + d0 * 16 + d1 * 4)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_pa_diffusion_apply_2d_stage_sliced(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
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
    %6 = bufferization.to_tensor %arg6 : memref<?xf64>
    %7 = tensor.empty() : tensor<2x4x4xf64>
    %8 = tensor.empty() : tensor<2x4x4xf64>
    %9 = tensor.empty() : tensor<2x5x4xf64>
    %10 = tensor.empty() : tensor<2x5x4xf64>
    %11 = tensor.empty() : tensor<2x5x5xf64>
    %12 = tensor.empty() : tensor<2x5x5xf64>
    %13 = tensor.empty() : tensor<2x4x5xf64>
    %14 = tensor.empty() : tensor<2x4x5xf64>
    %15 = polygeist.submap(%14, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %16 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%15 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %17 = polygeist.submapInverse(%14, %16, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x5xf64>
    %18 = polygeist.submap(%5, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %19 = polygeist.submap(%0, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %20 = polygeist.submap(%17, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v18_contract_21_tc0 = tensor.cast %18 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v19_contract_21_tc1 = tensor.cast %19 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v20_contract_21_tc2 = tensor.cast %20 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v21_tdyn = kernel.launch @cutensornetContraction2_f64(%v18_contract_21_tc0, %v19_contract_21_tc1, %v20_contract_21_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %21 = tensor.cast %v21_tdyn : tensor<*xf64> to tensor<?x?x?x?xf64>
    %22 = polygeist.submapInverse(%17, %21, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5xf64>
    %23 = polygeist.submap(%13, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %24 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%23 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %25 = polygeist.submapInverse(%13, %24, %c2, %c4, %c5) {map = #map} : (tensor<2x4x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x5xf64>
    %26 = polygeist.submap(%5, %c2, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %27 = polygeist.submap(%1, %c2, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %28 = polygeist.submap(%25, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %v26_contract_29_tc0 = tensor.cast %26 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v27_contract_29_tc1 = tensor.cast %27 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v28_contract_29_tc2 = tensor.cast %28 : tensor<?x?x?x?xf64> to tensor<*xf64>

    %v29_tdyn = kernel.launch @cutensornetContraction2_f64(%v26_contract_29_tc0, %v27_contract_29_tc1, %v28_contract_29_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %29 = tensor.cast %v29_tdyn : tensor<*xf64> to tensor<?x?x?x?xf64>
    %30 = polygeist.submapInverse(%25, %29, %c2, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5xf64>
    %31 = polygeist.submap(%12, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %32 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%31 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %33 = polygeist.submapInverse(%12, %32, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x5xf64>
    %34 = polygeist.submap(%30, %c2, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %35 = polygeist.submap(%0, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %36 = polygeist.submap(%33, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %37 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%34, %35 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%36 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %91 = arith.mulf %in, %in_0 : f64
      %92 = arith.addf %out, %91 : f64
      linalg.yield %92 : f64
    } -> tensor<?x?x?x?xf64>
    %38 = polygeist.submapInverse(%33, %37, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5xf64>
    %39 = polygeist.submap(%11, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, index, index, index) -> tensor<?x?x?xf64>
    %40 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%39 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %41 = polygeist.submapInverse(%11, %40, %c2, %c5, %c5) {map = #map} : (tensor<2x5x5xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x5xf64>
    %42 = polygeist.submap(%22, %c2, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %43 = polygeist.submap(%1, %c2, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %44 = polygeist.submap(%41, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %45 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%42, %43 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%44 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %91 = arith.mulf %in, %in_0 : f64
      %92 = arith.addf %out, %91 : f64
      linalg.yield %92 : f64
    } -> tensor<?x?x?x?xf64>
    %46 = polygeist.submapInverse(%41, %45, %c2, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5xf64>
    %47 = polygeist.submap(%10, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %48 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%47 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %49 = polygeist.submapInverse(%10, %48, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x4xf64>
    %50 = polygeist.submap(%4, %c2, %c5, %c4, %c5) {map = #map7} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %51 = polygeist.submap(%38, %c2, %c5, %c4, %c5) {map = #map8} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %52 = polygeist.submap(%4, %c2, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %53 = polygeist.submap(%46, %c2, %c5, %c4, %c5) {map = #map8} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %54 = polygeist.submap(%3, %c2, %c5, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %55 = polygeist.submap(%49, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %56 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%50, %51, %52, %53, %54 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%55 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
      %91 = arith.mulf %in, %in_0 : f64
      %92 = arith.mulf %in_1, %in_2 : f64
      %93 = arith.addf %91, %92 : f64
      %94 = arith.mulf %93, %in_3 : f64
      %95 = arith.addf %out, %94 : f64
      linalg.yield %95 : f64
    } -> tensor<?x?x?x?xf64>
    %57 = polygeist.submapInverse(%49, %56, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4xf64>
    %58 = polygeist.submap(%9, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %59 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%58 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %60 = polygeist.submapInverse(%9, %59, %c2, %c5, %c4) {map = #map} : (tensor<2x5x4xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x5x4xf64>
    %61 = polygeist.submap(%4, %c2, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %62 = polygeist.submap(%38, %c2, %c5, %c4, %c5) {map = #map8} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %63 = polygeist.submap(%4, %c2, %c5, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %64 = polygeist.submap(%46, %c2, %c5, %c4, %c5) {map = #map8} : (tensor<2x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %65 = polygeist.submap(%2, %c2, %c5, %c4, %c5) {map = #map10} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %66 = polygeist.submap(%60, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %67 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%61, %62, %63, %64, %65 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%66 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
      %91 = arith.mulf %in, %in_0 : f64
      %92 = arith.mulf %in_1, %in_2 : f64
      %93 = arith.addf %91, %92 : f64
      %94 = arith.mulf %93, %in_3 : f64
      %95 = arith.addf %out, %94 : f64
      linalg.yield %95 : f64
    } -> tensor<?x?x?x?xf64>
    %68 = polygeist.submapInverse(%60, %67, %c2, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4xf64>
    %69 = polygeist.submap(%8, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %70 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%69 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %71 = polygeist.submapInverse(%8, %70, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x4xf64>
    %72 = polygeist.submap(%57, %c2, %c4, %c4, %c5) {map = #map5} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %73 = polygeist.submap(%2, %c2, %c4, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %74 = polygeist.submap(%71, %c2, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %75 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%72, %73 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%74 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %91 = arith.mulf %in, %in_0 : f64
      %92 = arith.addf %out, %91 : f64
      linalg.yield %92 : f64
    } -> tensor<?x?x?x?xf64>
    %76 = polygeist.submapInverse(%71, %75, %c2, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4xf64>
    %77 = polygeist.submap(%7, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %78 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} outs(%77 : tensor<?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?xf64>
    %79 = polygeist.submapInverse(%7, %78, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<2x4x4xf64>
    %80 = polygeist.submap(%68, %c2, %c4, %c4, %c5) {map = #map5} : (tensor<2x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %81 = polygeist.submap(%3, %c2, %c4, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %82 = polygeist.submap(%79, %c2, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %83 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%80, %81 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%82 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %91 = arith.mulf %in, %in_0 : f64
      %92 = arith.addf %out, %91 : f64
      linalg.yield %92 : f64
    } -> tensor<?x?x?x?xf64>
    %84 = polygeist.submapInverse(%79, %83, %c2, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4xf64>
    %85 = polygeist.submap(%76, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %86 = polygeist.submap(%84, %c2, %c4, %c4) {map = #map} : (tensor<2x4x4xf64>, index, index, index) -> tensor<?x?x?xf64>
    %87 = polygeist.submap(%6, %c2, %c4, %c4) {map = #map13} : (tensor<?xf64>, index, index, index) -> tensor<?x?x?xf64>
    %88 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel", "parallel"], library_call = ""} ins(%85, %86 : tensor<?x?x?xf64>, tensor<?x?x?xf64>) outs(%87 : tensor<?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %91 = arith.addf %in, %in_0 : f64
      %92 = arith.addf %out, %91 : f64
      linalg.yield %92 : f64
    } -> tensor<?x?x?xf64>
    %89 = polygeist.submapInverse(%6, %88, %c2, %c4, %c4) {map = #map13} : (tensor<?xf64>, tensor<?x?x?xf64>, index, index, index) -> tensor<?xf64>
    %90 = bufferization.to_memref %89 : memref<?xf64>
    memref.copy %90, %arg6 : memref<?xf64> to memref<?xf64>
    return
  }
}
