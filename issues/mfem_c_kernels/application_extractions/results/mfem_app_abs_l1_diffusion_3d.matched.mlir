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
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_abs_l1_diffusion_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
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
    %24 = polygeist.submap(%23, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %25 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%24 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %26 = polygeist.submapInverse(%23, %25, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x5xf64>
    %27 = polygeist.submap(%5, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %28 = polygeist.submap(%0, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %29 = polygeist.submap(%26, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v27_contract_30_tc0 = tensor.cast %27 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v28_contract_30_tc1 = tensor.cast %28 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v29_contract_30_tc2 = tensor.cast %29 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v30_tdyn = kernel.launch @cutensornetContraction2_f64(%v27_contract_30_tc0, %v28_contract_30_tc1, %v29_contract_30_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %30 = tensor.cast %v30_tdyn : tensor<*xf64> to tensor<?x?x?x?x?xf64>
    %31 = polygeist.submapInverse(%26, %30, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x5xf64>
    %32 = polygeist.submap(%22, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %33 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%32 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %34 = polygeist.submapInverse(%22, %33, %c2, %c4, %c4, %c5) {map = #map} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x5xf64>
    %35 = polygeist.submap(%5, %c2, %c4, %c4, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %36 = polygeist.submap(%1, %c2, %c4, %c4, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %37 = polygeist.submap(%34, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v35_contract_38_tc0 = tensor.cast %35 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v36_contract_38_tc1 = tensor.cast %36 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v37_contract_38_tc2 = tensor.cast %37 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v38_tdyn = kernel.launch @cutensornetContraction2_f64(%v35_contract_38_tc0, %v36_contract_38_tc1, %v37_contract_38_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %38 = tensor.cast %v38_tdyn : tensor<*xf64> to tensor<?x?x?x?x?xf64>
    %39 = polygeist.submapInverse(%34, %38, %c2, %c4, %c4, %c5, %c4) {map = #map3} : (tensor<2x4x4x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x5xf64>
    %40 = polygeist.submap(%21, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %41 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%40 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %42 = polygeist.submapInverse(%21, %41, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %43 = polygeist.submap(%39, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %44 = polygeist.submap(%0, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %45 = polygeist.submap(%42, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %46 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%43, %44 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%45 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<?x?x?x?x?xf64>
    %47 = polygeist.submapInverse(%42, %46, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %48 = polygeist.submap(%20, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %49 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%48 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %50 = polygeist.submapInverse(%20, %49, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %51 = polygeist.submap(%31, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %52 = polygeist.submap(%1, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %53 = polygeist.submap(%50, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %54 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%51, %52 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%53 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<?x?x?x?x?xf64>
    %55 = polygeist.submapInverse(%50, %54, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %56 = polygeist.submap(%19, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %57 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%56 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %58 = polygeist.submapInverse(%19, %57, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %59 = polygeist.submap(%31, %c2, %c4, %c5, %c5, %c4) {map = #map5} : (tensor<2x4x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %60 = polygeist.submap(%0, %c2, %c4, %c5, %c5, %c4) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %61 = polygeist.submap(%58, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %62 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%59, %60 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%61 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<?x?x?x?x?xf64>
    %63 = polygeist.submapInverse(%58, %62, %c2, %c4, %c5, %c5, %c4) {map = #map3} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %64 = polygeist.submap(%18, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %65 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%64 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %66 = polygeist.submapInverse(%18, %65, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %67 = polygeist.submap(%47, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %68 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %69 = polygeist.submap(%66, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %70 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%67, %68 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%69 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<?x?x?x?x?xf64>
    %71 = polygeist.submapInverse(%66, %70, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x5xf64>
    %72 = polygeist.submap(%17, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %73 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%72 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %74 = polygeist.submapInverse(%17, %73, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %75 = polygeist.submap(%55, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %76 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %77 = polygeist.submap(%74, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %78 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%75, %76 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%77 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<?x?x?x?x?xf64>
    %79 = polygeist.submapInverse(%74, %78, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x5xf64>
    %80 = polygeist.submap(%16, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %81 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%80 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %82 = polygeist.submapInverse(%16, %81, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %83 = polygeist.submap(%63, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %84 = polygeist.submap(%1, %c2, %c5, %c5, %c5, %c4) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %85 = polygeist.submap(%82, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %86 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%83, %84 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%85 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<?x?x?x?x?xf64>
    %87 = polygeist.submapInverse(%82, %86, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x5xf64>
    %88 = polygeist.submap(%15, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %89 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%88 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %90 = polygeist.submapInverse(%15, %89, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x4xf64>
    %91 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %92 = polygeist.submap(%71, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %93 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %94 = polygeist.submap(%79, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %95 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %96 = polygeist.submap(%87, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %97 = polygeist.submap(%3, %c2, %c5, %c5, %c4, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %98 = polygeist.submap(%90, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %99 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%91, %92, %93, %94, %95, %96, %97 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%98 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.mulf %in_1, %in_2 : f64
      %184 = arith.addf %182, %183 : f64
      %185 = arith.mulf %in_3, %in_4 : f64
      %186 = arith.addf %184, %185 : f64
      %187 = arith.mulf %186, %in_5 : f64
      %188 = arith.addf %out, %187 : f64
      linalg.yield %188 : f64
    } -> tensor<?x?x?x?x?xf64>
    %100 = polygeist.submapInverse(%90, %99, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x4xf64>
    %101 = polygeist.submap(%14, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %102 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%101 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %103 = polygeist.submapInverse(%14, %102, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x4xf64>
    %104 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %105 = polygeist.submap(%71, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %106 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %107 = polygeist.submap(%79, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %108 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %109 = polygeist.submap(%87, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %110 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %111 = polygeist.submap(%103, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %112 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%104, %105, %106, %107, %108, %109, %110 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%111 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.mulf %in_1, %in_2 : f64
      %184 = arith.addf %182, %183 : f64
      %185 = arith.mulf %in_3, %in_4 : f64
      %186 = arith.addf %184, %185 : f64
      %187 = arith.mulf %186, %in_5 : f64
      %188 = arith.addf %out, %187 : f64
      linalg.yield %188 : f64
    } -> tensor<?x?x?x?x?xf64>
    %113 = polygeist.submapInverse(%103, %112, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x4xf64>
    %114 = polygeist.submap(%13, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %115 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%114 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %116 = polygeist.submapInverse(%13, %115, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x4xf64>
    %117 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %118 = polygeist.submap(%71, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %119 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %120 = polygeist.submap(%79, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %121 = polygeist.submap(%4, %c2, %c5, %c5, %c4, %c5) {map = #map16} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %122 = polygeist.submap(%87, %c2, %c5, %c5, %c4, %c5) {map = #map10} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %123 = polygeist.submap(%2, %c2, %c5, %c5, %c4, %c5) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %124 = polygeist.submap(%116, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %125 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%117, %118, %119, %120, %121, %122, %123 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%124 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.mulf %in_1, %in_2 : f64
      %184 = arith.addf %182, %183 : f64
      %185 = arith.mulf %in_3, %in_4 : f64
      %186 = arith.addf %184, %185 : f64
      %187 = arith.mulf %186, %in_5 : f64
      %188 = arith.addf %out, %187 : f64
      linalg.yield %188 : f64
    } -> tensor<?x?x?x?x?xf64>
    %126 = polygeist.submapInverse(%116, %125, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x4xf64>
    %127 = polygeist.submap(%12, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %128 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%127 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %129 = polygeist.submapInverse(%12, %128, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4x4xf64>
    %130 = polygeist.submap(%100, %c2, %c5, %c4, %c4, %c5) {map = #map5} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %131 = polygeist.submap(%2, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %132 = polygeist.submap(%129, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %133 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%130, %131 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%132 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<?x?x?x?x?xf64>
    %134 = polygeist.submapInverse(%129, %133, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x4x4xf64>
    %135 = polygeist.submap(%11, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %136 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%135 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %137 = polygeist.submapInverse(%11, %136, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4x4xf64>
    %138 = polygeist.submap(%113, %c2, %c5, %c4, %c4, %c5) {map = #map5} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %139 = polygeist.submap(%3, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %140 = polygeist.submap(%137, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %141 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%138, %139 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%140 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<?x?x?x?x?xf64>
    %142 = polygeist.submapInverse(%137, %141, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x4x4xf64>
    %143 = polygeist.submap(%10, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %144 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%143 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %145 = polygeist.submapInverse(%10, %144, %c2, %c5, %c4, %c4) {map = #map} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4x4xf64>
    %146 = polygeist.submap(%126, %c2, %c5, %c4, %c4, %c5) {map = #map5} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %147 = polygeist.submap(%2, %c2, %c5, %c4, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %148 = polygeist.submap(%145, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %149 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%146, %147 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%148 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<?x?x?x?x?xf64>
    %150 = polygeist.submapInverse(%145, %149, %c2, %c5, %c4, %c4, %c5) {map = #map3} : (tensor<2x5x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x4x4xf64>
    %151 = polygeist.submap(%9, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %152 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%151 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %153 = polygeist.submapInverse(%9, %152, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x4xf64>
    %154 = polygeist.submap(%134, %c2, %c4, %c4, %c4, %c5) {map = #map7} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %155 = polygeist.submap(%2, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %156 = polygeist.submap(%153, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %157 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%154, %155 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%156 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<?x?x?x?x?xf64>
    %158 = polygeist.submapInverse(%153, %157, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x4xf64>
    %159 = polygeist.submap(%8, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %160 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%159 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %161 = polygeist.submapInverse(%8, %160, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x4xf64>
    %162 = polygeist.submap(%142, %c2, %c4, %c4, %c4, %c5) {map = #map7} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %163 = polygeist.submap(%2, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %164 = polygeist.submap(%161, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %165 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%162, %163 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%164 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<?x?x?x?x?xf64>
    %166 = polygeist.submapInverse(%161, %165, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x4xf64>
    %167 = polygeist.submap(%7, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %168 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%167 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %169 = polygeist.submapInverse(%7, %168, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x4x4xf64>
    %170 = polygeist.submap(%150, %c2, %c4, %c4, %c4, %c5) {map = #map7} : (tensor<2x5x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %171 = polygeist.submap(%3, %c2, %c4, %c4, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %172 = polygeist.submap(%169, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %173 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%170, %171 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%172 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %182 = arith.mulf %in, %in_0 : f64
      %183 = arith.addf %out, %182 : f64
      linalg.yield %183 : f64
    } -> tensor<?x?x?x?x?xf64>
    %174 = polygeist.submapInverse(%169, %173, %c2, %c4, %c4, %c4, %c5) {map = #map3} : (tensor<2x4x4x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x4x4xf64>
    %175 = polygeist.submap(%158, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %176 = polygeist.submap(%166, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %177 = polygeist.submap(%174, %c2, %c4, %c4, %c4) {map = #map} : (tensor<2x4x4x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %178 = polygeist.submap(%6, %c2, %c4, %c4, %c4) {map = #map19} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %179 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%175, %176, %177 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%178 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %out: f64):
      %182 = arith.addf %in, %in_0 : f64
      %183 = arith.addf %182, %in_1 : f64
      %184 = arith.addf %out, %183 : f64
      linalg.yield %184 : f64
    } -> tensor<?x?x?x?xf64>
    %180 = polygeist.submapInverse(%6, %179, %c2, %c4, %c4, %c4) {map = #map19} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %181 = bufferization.to_memref %180 : memref<?xf64>
    memref.copy %181, %arg6 : memref<?xf64> to memref<?xf64>
    return
  }
}
