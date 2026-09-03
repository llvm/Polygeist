#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 12 + d2 * 4 + d0 * 108)>
#map2 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 4)>
#map3 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>
#map4 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map5 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
#map6 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 3)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2, d3)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 3)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 12 + d2 * 3 + d0 * 108 + 36)>
#map10 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 3)>
#map11 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 4)>
#map12 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 9 + d2 * 3 + d0 * 108 + 72)>
#map13 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 4)>
#map14 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d0 * 125 + d1 * 25 + d2 * 5)>
#map15 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
#map16 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d3 * 5)>
#map17 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d2 * 5)>
#map18 = affine_map<(d0, d1, d2, d3, d4) -> (d4 + d1 * 5)>
#map19 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d1 * 12 + d2 * 4 + d0 * 108)>
#map20 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d1 * 12 + d2 * 3 + d0 * 108 + 36)>
#map21 = affine_map<(d0, d1, d2, d3, d4) -> (d3 + d1 * 9 + d2 * 3 + d0 * 108 + 72)>
#map22 = affine_map<(d0, d1, d2, d3) -> (d0, 0, d1, d2, d3)>
#map23 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4 + d1 * 3)>
#map24 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5 + d2 * 3)>
#map25 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6 + d4 * 12 + d5 * 4 + d0 * 108)>
#map26 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6 + d3 * 4)>
#map27 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, 0, d1, d2, d3)>
#map28 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3, d4, d5, d6)>
#map29 = affine_map<(d0, d1, d2, d3) -> (d0, 1, d1, d2, d3)>
#map30 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5 + d2 * 4)>
#map31 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6 + d4 * 12 + d5 * 3 + d0 * 108 + 36)>
#map32 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6 + d3 * 3)>
#map33 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, 1, d1, d2, d3)>
#map34 = affine_map<(d0, d1, d2, d3) -> (d0, 2, d1, d2, d3)>
#map35 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4 + d1 * 4)>
#map36 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6 + d4 * 9 + d5 * 3 + d0 * 108 + 72)>
#map37 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, 2, d1, d2, d3)>
#map38 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 25 + d2 * 5 + d0 * 750)>
#map39 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 25 + d2 * 5 + d0 * 750 + 125)>
#map40 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 25 + d2 * 5 + d0 * 750 + 250)>
#map41 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 25 + d2 * 5 + d0 * 750 + 375)>
#map42 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 25 + d2 * 5 + d0 * 750 + 500)>
#map43 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 25 + d2 * 5 + d0 * 750 + 625)>
#map44 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4 + d1 * 5)>
#map45 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d5 + d2 * 5)>
#map46 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, 0, d4, d5, d6)>
#map47 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d6 + d3 * 5)>
#map48 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 12 + d2 * 4 + d0 * 108)>
#map49 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
#map50 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, 1, d4, d5, d6)>
#map51 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 12 + d2 * 3 + d0 * 108 + 36)>
#map52 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, 2, d4, d5, d6)>
#map53 = affine_map<(d0, d1, d2, d3) -> (d3 + d1 * 9 + d2 * 3 + d0 * 108 + 72)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @mfem_app_ex35p_hdiv_3d(%arg0: memref<?xf64>, %arg1: memref<?xf64>, %arg2: memref<?xf64>, %arg3: memref<?xf64>, %arg4: memref<?xf64>, %arg5: memref<?xf64>, %arg6: memref<?xf64>, %arg7: memref<?xf64>, %arg8: memref<?xf64>, %arg9: memref<?xf64>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f64
    %c5 = arith.constant 5 : index
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %0 = bufferization.to_tensor %arg0 : memref<?xf64>
    %1 = bufferization.to_tensor %arg1 : memref<?xf64>
    %2 = bufferization.to_tensor %arg2 : memref<?xf64>
    %3 = bufferization.to_tensor %arg3 : memref<?xf64>
    %4 = bufferization.to_tensor %arg4 : memref<?xf64>
    %5 = bufferization.to_tensor %arg5 : memref<?xf64>
    %6 = bufferization.to_tensor %arg6 : memref<?xf64>
    %7 = bufferization.to_tensor %arg7 : memref<?xf64>
    %8 = bufferization.to_tensor %arg8 : memref<?xf64>
    %9 = bufferization.to_tensor %arg9 : memref<?xf64>
    %10 = tensor.empty() : tensor<2x5x3x3xf64>
    %11 = tensor.empty() : tensor<2x5x4x3xf64>
    %12 = tensor.empty() : tensor<2x5x3x4xf64>
    %13 = tensor.empty() : tensor<2x5x5x3xf64>
    %14 = tensor.empty() : tensor<2x5x5x3xf64>
    %15 = tensor.empty() : tensor<2x5x5x4xf64>
    %16 = tensor.empty() : tensor<2x5x5x5xf64>
    %17 = tensor.empty() : tensor<2x5x5x5xf64>
    %18 = tensor.empty() : tensor<2x5x5x5xf64>
    %19 = tensor.empty() : tensor<2x4x5x5xf64>
    %20 = tensor.empty() : tensor<2x3x5x5xf64>
    %21 = tensor.empty() : tensor<2x3x5x5xf64>
    %22 = tensor.empty() : tensor<2x4x3x5xf64>
    %23 = tensor.empty() : tensor<2x3x4x5xf64>
    %24 = tensor.empty() : tensor<2x3x3x5xf64>
    %25 = polygeist.submap(%24, %c2, %c3, %c3, %c5) {map = #map} : (tensor<2x3x3x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %26 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%25 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %27 = polygeist.submapInverse(%24, %26, %c2, %c3, %c3, %c5) {map = #map} : (tensor<2x3x3x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x3x3x5xf64>
    %28 = polygeist.submap(%8, %c2, %c3, %c3, %c5, %c4) {map = #map1} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %29 = polygeist.submap(%4, %c2, %c3, %c3, %c5, %c4) {map = #map2} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %30 = polygeist.submap(%27, %c2, %c3, %c3, %c5, %c4) {map = #map3} : (tensor<2x3x3x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v28_contract_31_tc0 = tensor.cast %28 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v29_contract_31_tc1 = tensor.cast %29 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v30_contract_31_tc2 = tensor.cast %30 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v31_tdyn = kernel.launch @cutensornetContraction2_f64(%v28_contract_31_tc0, %v29_contract_31_tc1, %v30_contract_31_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %31 = tensor.cast %v31_tdyn : tensor<*xf64> to tensor<?x?x?x?x?xf64>
    %32 = polygeist.submapInverse(%27, %31, %c2, %c3, %c3, %c5, %c4) {map = #map3} : (tensor<2x3x3x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x3x3x5xf64>
    %33 = polygeist.submap(%21, %c2, %c3, %c5, %c5) {map = #map} : (tensor<2x3x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %34 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%33 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %35 = polygeist.submapInverse(%21, %34, %c2, %c3, %c5, %c5) {map = #map} : (tensor<2x3x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x3x5x5xf64>
    %36 = polygeist.submap(%32, %c2, %c3, %c5, %c5, %c3) {map = #map5} : (tensor<2x3x3x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %37 = polygeist.submap(%0, %c2, %c3, %c5, %c5, %c3) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %38 = polygeist.submap(%35, %c2, %c3, %c5, %c5, %c3) {map = #map3} : (tensor<2x3x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %39 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%36, %37 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%38 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %238 = arith.mulf %in, %in_0 : f64
      %239 = arith.addf %out, %238 : f64
      linalg.yield %239 : f64
    } -> tensor<?x?x?x?x?xf64>
    %40 = polygeist.submapInverse(%35, %39, %c2, %c3, %c5, %c5, %c3) {map = #map3} : (tensor<2x3x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x3x5x5xf64>
    %41 = polygeist.submap(%18, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %42 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%41 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %43 = polygeist.submapInverse(%18, %42, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %44 = polygeist.submap(%40, %c2, %c5, %c5, %c5, %c3) {map = #map7} : (tensor<2x3x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %45 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c3) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %46 = polygeist.submap(%43, %c2, %c5, %c5, %c5, %c3) {map = #map3} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %47 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%44, %45 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%46 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %238 = arith.mulf %in, %in_0 : f64
      %239 = arith.addf %out, %238 : f64
      linalg.yield %239 : f64
    } -> tensor<?x?x?x?x?xf64>
    %48 = polygeist.submapInverse(%43, %47, %c2, %c5, %c5, %c5, %c3) {map = #map3} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x5xf64>
    %49 = polygeist.submap(%23, %c2, %c3, %c4, %c5) {map = #map} : (tensor<2x3x4x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %50 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%49 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %51 = polygeist.submapInverse(%23, %50, %c2, %c3, %c4, %c5) {map = #map} : (tensor<2x3x4x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x3x4x5xf64>
    %52 = polygeist.submap(%8, %c2, %c3, %c4, %c5, %c3) {map = #map9} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %53 = polygeist.submap(%0, %c2, %c3, %c4, %c5, %c3) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %54 = polygeist.submap(%51, %c2, %c3, %c4, %c5, %c3) {map = #map3} : (tensor<2x3x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v52_contract_55_tc0 = tensor.cast %52 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v53_contract_55_tc1 = tensor.cast %53 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v54_contract_55_tc2 = tensor.cast %54 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v55_tdyn = kernel.launch @cutensornetContraction2_f64(%v52_contract_55_tc0, %v53_contract_55_tc1, %v54_contract_55_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %55 = tensor.cast %v55_tdyn : tensor<*xf64> to tensor<?x?x?x?x?xf64>
    %56 = polygeist.submapInverse(%51, %55, %c2, %c3, %c4, %c5, %c3) {map = #map3} : (tensor<2x3x4x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x3x4x5xf64>
    %57 = polygeist.submap(%20, %c2, %c3, %c5, %c5) {map = #map} : (tensor<2x3x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %58 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%57 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %59 = polygeist.submapInverse(%20, %58, %c2, %c3, %c5, %c5) {map = #map} : (tensor<2x3x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x3x5x5xf64>
    %60 = polygeist.submap(%56, %c2, %c3, %c5, %c5, %c4) {map = #map5} : (tensor<2x3x4x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %61 = polygeist.submap(%4, %c2, %c3, %c5, %c5, %c4) {map = #map11} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %62 = polygeist.submap(%59, %c2, %c3, %c5, %c5, %c4) {map = #map3} : (tensor<2x3x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %63 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%60, %61 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%62 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %238 = arith.mulf %in, %in_0 : f64
      %239 = arith.addf %out, %238 : f64
      linalg.yield %239 : f64
    } -> tensor<?x?x?x?x?xf64>
    %64 = polygeist.submapInverse(%59, %63, %c2, %c3, %c5, %c5, %c4) {map = #map3} : (tensor<2x3x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x3x5x5xf64>
    %65 = polygeist.submap(%17, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %66 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%65 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %67 = polygeist.submapInverse(%17, %66, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %68 = polygeist.submap(%64, %c2, %c5, %c5, %c5, %c3) {map = #map7} : (tensor<2x3x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %69 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c3) {map = #map8} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %70 = polygeist.submap(%67, %c2, %c5, %c5, %c5, %c3) {map = #map3} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %71 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%68, %69 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%70 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %238 = arith.mulf %in, %in_0 : f64
      %239 = arith.addf %out, %238 : f64
      linalg.yield %239 : f64
    } -> tensor<?x?x?x?x?xf64>
    %72 = polygeist.submapInverse(%67, %71, %c2, %c5, %c5, %c5, %c3) {map = #map3} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x5xf64>
    %73 = polygeist.submap(%22, %c2, %c4, %c3, %c5) {map = #map} : (tensor<2x4x3x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %74 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%73 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %75 = polygeist.submapInverse(%22, %74, %c2, %c4, %c3, %c5) {map = #map} : (tensor<2x4x3x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x3x5xf64>
    %76 = polygeist.submap(%8, %c2, %c4, %c3, %c5, %c3) {map = #map12} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %77 = polygeist.submap(%0, %c2, %c4, %c3, %c5, %c3) {map = #map10} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %78 = polygeist.submap(%75, %c2, %c4, %c3, %c5, %c3) {map = #map3} : (tensor<2x4x3x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %v76_contract_79_tc0 = tensor.cast %76 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v77_contract_79_tc1 = tensor.cast %77 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v78_contract_79_tc2 = tensor.cast %78 : tensor<?x?x?x?x?xf64> to tensor<*xf64>

    %v79_tdyn = kernel.launch @cutensornetContraction2_f64(%v76_contract_79_tc0, %v77_contract_79_tc1, %v78_contract_79_tc2) {contraction_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>, affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>]} : (tensor<*xf64>, tensor<*xf64>, tensor<*xf64>) -> tensor<*xf64>

    %79 = tensor.cast %v79_tdyn : tensor<*xf64> to tensor<?x?x?x?x?xf64>
    %80 = polygeist.submapInverse(%75, %79, %c2, %c4, %c3, %c5, %c3) {map = #map3} : (tensor<2x4x3x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x3x5xf64>
    %81 = polygeist.submap(%19, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %82 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%81 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %83 = polygeist.submapInverse(%19, %82, %c2, %c4, %c5, %c5) {map = #map} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x4x5x5xf64>
    %84 = polygeist.submap(%80, %c2, %c4, %c5, %c5, %c3) {map = #map5} : (tensor<2x4x3x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %85 = polygeist.submap(%0, %c2, %c4, %c5, %c5, %c3) {map = #map6} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %86 = polygeist.submap(%83, %c2, %c4, %c5, %c5, %c3) {map = #map3} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %87 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%84, %85 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%86 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %238 = arith.mulf %in, %in_0 : f64
      %239 = arith.addf %out, %238 : f64
      linalg.yield %239 : f64
    } -> tensor<?x?x?x?x?xf64>
    %88 = polygeist.submapInverse(%83, %87, %c2, %c4, %c5, %c5, %c3) {map = #map3} : (tensor<2x4x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x4x5x5xf64>
    %89 = polygeist.submap(%16, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %90 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%89 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %91 = polygeist.submapInverse(%16, %90, %c2, %c5, %c5, %c5) {map = #map} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x5xf64>
    %92 = polygeist.submap(%88, %c2, %c5, %c5, %c5, %c4) {map = #map7} : (tensor<2x4x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %93 = polygeist.submap(%4, %c2, %c5, %c5, %c5, %c4) {map = #map13} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %94 = polygeist.submap(%91, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %95 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%92, %93 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%94 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %238 = arith.mulf %in, %in_0 : f64
      %239 = arith.addf %out, %238 : f64
      linalg.yield %239 : f64
    } -> tensor<?x?x?x?x?xf64>
    %96 = polygeist.submapInverse(%91, %95, %c2, %c5, %c5, %c5, %c4) {map = #map3} : (tensor<2x5x5x5xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x5xf64>
    %97 = polygeist.submap(%15, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %98 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%97 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %99 = polygeist.submapInverse(%15, %98, %c2, %c5, %c5, %c4) {map = #map} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x4xf64>
    %100 = polygeist.submap(%6, %c2, %c5, %c5, %c4, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %101 = polygeist.submap(%48, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %102 = polygeist.submap(%72, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %103 = polygeist.submap(%96, %c2, %c5, %c5, %c4, %c5) {map = #map15} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %104 = polygeist.submap(%5, %c2, %c5, %c5, %c4, %c5) {map = #map16} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %105 = polygeist.submap(%99, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %106 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%100, %101, %102, %103, %104 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%105 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
      %238 = arith.addf %in_0, %in_1 : f64
      %239 = arith.addf %238, %in_2 : f64
      %240 = arith.mulf %in, %239 : f64
      %241 = arith.mulf %240, %in_3 : f64
      %242 = arith.addf %out, %241 : f64
      linalg.yield %242 : f64
    } -> tensor<?x?x?x?x?xf64>
    %107 = polygeist.submapInverse(%99, %106, %c2, %c5, %c5, %c4, %c5) {map = #map3} : (tensor<2x5x5x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x4xf64>
    %108 = polygeist.submap(%14, %c2, %c5, %c5, %c3) {map = #map} : (tensor<2x5x5x3xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %109 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%108 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %110 = polygeist.submapInverse(%14, %109, %c2, %c5, %c5, %c3) {map = #map} : (tensor<2x5x5x3xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x3xf64>
    %111 = polygeist.submap(%6, %c2, %c5, %c5, %c3, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %112 = polygeist.submap(%48, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %113 = polygeist.submap(%72, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %114 = polygeist.submap(%96, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %115 = polygeist.submap(%2, %c2, %c5, %c5, %c3, %c5) {map = #map16} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %116 = polygeist.submap(%110, %c2, %c5, %c5, %c3, %c5) {map = #map3} : (tensor<2x5x5x3xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %117 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%111, %112, %113, %114, %115 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%116 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
      %238 = arith.addf %in_0, %in_1 : f64
      %239 = arith.addf %238, %in_2 : f64
      %240 = arith.mulf %in, %239 : f64
      %241 = arith.mulf %240, %in_3 : f64
      %242 = arith.addf %out, %241 : f64
      linalg.yield %242 : f64
    } -> tensor<?x?x?x?x?xf64>
    %118 = polygeist.submapInverse(%110, %117, %c2, %c5, %c5, %c3, %c5) {map = #map3} : (tensor<2x5x5x3xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x3xf64>
    %119 = polygeist.submap(%13, %c2, %c5, %c5, %c3) {map = #map} : (tensor<2x5x5x3xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %120 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%119 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %121 = polygeist.submapInverse(%13, %120, %c2, %c5, %c5, %c3) {map = #map} : (tensor<2x5x5x3xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x5x3xf64>
    %122 = polygeist.submap(%6, %c2, %c5, %c5, %c3, %c5) {map = #map14} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %123 = polygeist.submap(%48, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %124 = polygeist.submap(%72, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %125 = polygeist.submap(%96, %c2, %c5, %c5, %c3, %c5) {map = #map15} : (tensor<2x5x5x5xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %126 = polygeist.submap(%2, %c2, %c5, %c5, %c3, %c5) {map = #map16} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %127 = polygeist.submap(%121, %c2, %c5, %c5, %c3, %c5) {map = #map3} : (tensor<2x5x5x3xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %128 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4, #map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%122, %123, %124, %125, %126 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%127 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %out: f64):
      %238 = arith.addf %in_0, %in_1 : f64
      %239 = arith.addf %238, %in_2 : f64
      %240 = arith.mulf %in, %239 : f64
      %241 = arith.mulf %240, %in_3 : f64
      %242 = arith.addf %out, %241 : f64
      linalg.yield %242 : f64
    } -> tensor<?x?x?x?x?xf64>
    %129 = polygeist.submapInverse(%121, %128, %c2, %c5, %c5, %c3, %c5) {map = #map3} : (tensor<2x5x5x3xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x5x3xf64>
    %130 = polygeist.submap(%12, %c2, %c5, %c3, %c4) {map = #map} : (tensor<2x5x3x4xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %131 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%130 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %132 = polygeist.submapInverse(%12, %131, %c2, %c5, %c3, %c4) {map = #map} : (tensor<2x5x3x4xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x3x4xf64>
    %133 = polygeist.submap(%107, %c2, %c5, %c3, %c4, %c5) {map = #map5} : (tensor<2x5x5x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %134 = polygeist.submap(%2, %c2, %c5, %c3, %c4, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %135 = polygeist.submap(%132, %c2, %c5, %c3, %c4, %c5) {map = #map3} : (tensor<2x5x3x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %136 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%133, %134 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%135 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %238 = arith.mulf %in, %in_0 : f64
      %239 = arith.addf %out, %238 : f64
      linalg.yield %239 : f64
    } -> tensor<?x?x?x?x?xf64>
    %137 = polygeist.submapInverse(%132, %136, %c2, %c5, %c3, %c4, %c5) {map = #map3} : (tensor<2x5x3x4xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x3x4xf64>
    %138 = polygeist.submap(%11, %c2, %c5, %c4, %c3) {map = #map} : (tensor<2x5x4x3xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %139 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%138 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %140 = polygeist.submapInverse(%11, %139, %c2, %c5, %c4, %c3) {map = #map} : (tensor<2x5x4x3xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x4x3xf64>
    %141 = polygeist.submap(%118, %c2, %c5, %c4, %c3, %c5) {map = #map5} : (tensor<2x5x5x3xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %142 = polygeist.submap(%5, %c2, %c5, %c4, %c3, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %143 = polygeist.submap(%140, %c2, %c5, %c4, %c3, %c5) {map = #map3} : (tensor<2x5x4x3xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %144 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%141, %142 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%143 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %238 = arith.mulf %in, %in_0 : f64
      %239 = arith.addf %out, %238 : f64
      linalg.yield %239 : f64
    } -> tensor<?x?x?x?x?xf64>
    %145 = polygeist.submapInverse(%140, %144, %c2, %c5, %c4, %c3, %c5) {map = #map3} : (tensor<2x5x4x3xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x4x3xf64>
    %146 = polygeist.submap(%10, %c2, %c5, %c3, %c3) {map = #map} : (tensor<2x5x3x3xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %147 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%146 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %148 = polygeist.submapInverse(%10, %147, %c2, %c5, %c3, %c3) {map = #map} : (tensor<2x5x3x3xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x5x3x3xf64>
    %149 = polygeist.submap(%129, %c2, %c5, %c3, %c3, %c5) {map = #map5} : (tensor<2x5x5x3xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %150 = polygeist.submap(%2, %c2, %c5, %c3, %c3, %c5) {map = #map17} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %151 = polygeist.submap(%148, %c2, %c5, %c3, %c3, %c5) {map = #map3} : (tensor<2x5x3x3xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %152 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%149, %150 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%151 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %238 = arith.mulf %in, %in_0 : f64
      %239 = arith.addf %out, %238 : f64
      linalg.yield %239 : f64
    } -> tensor<?x?x?x?x?xf64>
    %153 = polygeist.submapInverse(%148, %152, %c2, %c5, %c3, %c3, %c5) {map = #map3} : (tensor<2x5x3x3xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<2x5x3x3xf64>
    %154 = polygeist.submap(%137, %c2, %c3, %c3, %c4, %c5) {map = #map7} : (tensor<2x5x3x4xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %155 = polygeist.submap(%2, %c2, %c3, %c3, %c4, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %156 = polygeist.submap(%9, %c2, %c3, %c3, %c4, %c5) {map = #map19} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %157 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%154, %155 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%156 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %238 = arith.mulf %in, %in_0 : f64
      %239 = arith.addf %out, %238 : f64
      linalg.yield %239 : f64
    } -> tensor<?x?x?x?x?xf64>
    %158 = polygeist.submapInverse(%9, %157, %c2, %c3, %c3, %c4, %c5) {map = #map19} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %159 = polygeist.submap(%145, %c2, %c3, %c4, %c3, %c5) {map = #map7} : (tensor<2x5x4x3xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %160 = polygeist.submap(%2, %c2, %c3, %c4, %c3, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %161 = polygeist.submap(%158, %c2, %c3, %c4, %c3, %c5) {map = #map20} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %162 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%159, %160 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%161 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %238 = arith.mulf %in, %in_0 : f64
      %239 = arith.addf %out, %238 : f64
      linalg.yield %239 : f64
    } -> tensor<?x?x?x?x?xf64>
    %163 = polygeist.submapInverse(%158, %162, %c2, %c3, %c4, %c3, %c5) {map = #map20} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %164 = polygeist.submap(%153, %c2, %c4, %c3, %c3, %c5) {map = #map7} : (tensor<2x5x3x3xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %165 = polygeist.submap(%5, %c2, %c4, %c3, %c3, %c5) {map = #map18} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %166 = polygeist.submap(%163, %c2, %c4, %c3, %c3, %c5) {map = #map21} : (tensor<?xf64>, index, index, index, index, index) -> tensor<?x?x?x?x?xf64>
    %167 = linalg.generic {doc = "", indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"], library_call = ""} ins(%164, %165 : tensor<?x?x?x?x?xf64>, tensor<?x?x?x?x?xf64>) outs(%166 : tensor<?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %out: f64):
      %238 = arith.mulf %in, %in_0 : f64
      %239 = arith.addf %out, %238 : f64
      linalg.yield %239 : f64
    } -> tensor<?x?x?x?x?xf64>
    %168 = polygeist.submapInverse(%163, %167, %c2, %c4, %c3, %c3, %c5) {map = #map21} : (tensor<?xf64>, tensor<?x?x?x?x?xf64>, index, index, index, index, index) -> tensor<?xf64>
    %169 = tensor.empty() : tensor<2x3x5x5x5xf64>
    %170 = polygeist.submap(%169, %c2, %c5, %c5, %c5) {map = #map22} : (tensor<2x3x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %171 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%170 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %172 = polygeist.submapInverse(%169, %171, %c2, %c5, %c5, %c5) {map = #map22} : (tensor<2x3x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x3x5x5x5xf64>
    %173 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c3, %c3, %c4) {map = #map23} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %174 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c3, %c3, %c4) {map = #map24} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %175 = polygeist.submap(%8, %c2, %c5, %c5, %c5, %c3, %c3, %c4) {map = #map25} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %176 = polygeist.submap(%1, %c2, %c5, %c5, %c5, %c3, %c3, %c4) {map = #map26} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %177 = polygeist.submap(%172, %c2, %c5, %c5, %c5, %c3, %c3, %c4) {map = #map27} : (tensor<2x3x5x5x5xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %178 = linalg.generic {doc = "", indexing_maps = [#map28, #map28, #map28, #map28, #map28], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"], library_call = ""} ins(%173, %174, %175, %176 : tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>) outs(%177 : tensor<?x?x?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %238 = arith.mulf %in_1, %in_2 : f64
      %239 = arith.mulf %238, %in_0 : f64
      %240 = arith.mulf %239, %in : f64
      %241 = arith.addf %out, %240 : f64
      linalg.yield %241 : f64
    } -> tensor<?x?x?x?x?x?x?xf64>
    %179 = polygeist.submapInverse(%172, %178, %c2, %c5, %c5, %c5, %c3, %c3, %c4) {map = #map27} : (tensor<2x3x5x5x5xf64>, tensor<?x?x?x?x?x?x?xf64>, index, index, index, index, index, index, index) -> tensor<2x3x5x5x5xf64>
    %180 = polygeist.submap(%179, %c2, %c5, %c5, %c5) {map = #map29} : (tensor<2x3x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %181 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%180 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %182 = polygeist.submapInverse(%179, %181, %c2, %c5, %c5, %c5) {map = #map29} : (tensor<2x3x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x3x5x5x5xf64>
    %183 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c3, %c4, %c3) {map = #map23} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %184 = polygeist.submap(%1, %c2, %c5, %c5, %c5, %c3, %c4, %c3) {map = #map30} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %185 = polygeist.submap(%8, %c2, %c5, %c5, %c5, %c3, %c4, %c3) {map = #map31} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %186 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c3, %c4, %c3) {map = #map32} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %187 = polygeist.submap(%182, %c2, %c5, %c5, %c5, %c3, %c4, %c3) {map = #map33} : (tensor<2x3x5x5x5xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %188 = linalg.generic {doc = "", indexing_maps = [#map28, #map28, #map28, #map28, #map28], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"], library_call = ""} ins(%183, %184, %185, %186 : tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>) outs(%187 : tensor<?x?x?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %238 = arith.mulf %in_1, %in_2 : f64
      %239 = arith.mulf %238, %in_0 : f64
      %240 = arith.mulf %239, %in : f64
      %241 = arith.addf %out, %240 : f64
      linalg.yield %241 : f64
    } -> tensor<?x?x?x?x?x?x?xf64>
    %189 = polygeist.submapInverse(%182, %188, %c2, %c5, %c5, %c5, %c3, %c4, %c3) {map = #map33} : (tensor<2x3x5x5x5xf64>, tensor<?x?x?x?x?x?x?xf64>, index, index, index, index, index, index, index) -> tensor<2x3x5x5x5xf64>
    %190 = polygeist.submap(%189, %c2, %c5, %c5, %c5) {map = #map34} : (tensor<2x3x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %191 = linalg.generic {doc = "", indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} outs(%190 : tensor<?x?x?x?xf64>) {
    ^bb0(%out: f64):
      linalg.yield %cst : f64
    } -> tensor<?x?x?x?xf64>
    %192 = polygeist.submapInverse(%189, %191, %c2, %c5, %c5, %c5) {map = #map34} : (tensor<2x3x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x3x5x5x5xf64>
    %193 = polygeist.submap(%1, %c2, %c5, %c5, %c5, %c4, %c3, %c3) {map = #map35} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %194 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c4, %c3, %c3) {map = #map24} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %195 = polygeist.submap(%8, %c2, %c5, %c5, %c5, %c4, %c3, %c3) {map = #map36} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %196 = polygeist.submap(%0, %c2, %c5, %c5, %c5, %c4, %c3, %c3) {map = #map32} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %197 = polygeist.submap(%192, %c2, %c5, %c5, %c5, %c4, %c3, %c3) {map = #map37} : (tensor<2x3x5x5x5xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %198 = linalg.generic {doc = "", indexing_maps = [#map28, #map28, #map28, #map28, #map28], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"], library_call = ""} ins(%193, %194, %195, %196 : tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>) outs(%197 : tensor<?x?x?x?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %238 = arith.mulf %in_1, %in_2 : f64
      %239 = arith.mulf %238, %in_0 : f64
      %240 = arith.mulf %239, %in : f64
      %241 = arith.addf %out, %240 : f64
      linalg.yield %241 : f64
    } -> tensor<?x?x?x?x?x?x?xf64>
    %199 = polygeist.submapInverse(%192, %198, %c2, %c5, %c5, %c5, %c4, %c3, %c3) {map = #map37} : (tensor<2x3x5x5x5xf64>, tensor<?x?x?x?x?x?x?xf64>, index, index, index, index, index, index, index) -> tensor<2x3x5x5x5xf64>
    %200 = polygeist.submap(%7, %c2, %c5, %c5, %c5) {map = #map38} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %201 = polygeist.submap(%7, %c2, %c5, %c5, %c5) {map = #map39} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %202 = polygeist.submap(%7, %c2, %c5, %c5, %c5) {map = #map40} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %203 = polygeist.submap(%7, %c2, %c5, %c5, %c5) {map = #map39} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %204 = polygeist.submap(%7, %c2, %c5, %c5, %c5) {map = #map41} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %205 = polygeist.submap(%7, %c2, %c5, %c5, %c5) {map = #map42} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %206 = polygeist.submap(%7, %c2, %c5, %c5, %c5) {map = #map40} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %207 = polygeist.submap(%7, %c2, %c5, %c5, %c5) {map = #map42} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %208 = polygeist.submap(%7, %c2, %c5, %c5, %c5) {map = #map43} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %209 = polygeist.submap(%199, %c2, %c5, %c5, %c5) {map = #map22} : (tensor<2x3x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %210 = polygeist.submap(%199, %c2, %c5, %c5, %c5) {map = #map29} : (tensor<2x3x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %211 = polygeist.submap(%199, %c2, %c5, %c5, %c5) {map = #map34} : (tensor<2x3x5x5x5xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %212:3 = linalg.generic {doc = "", indexing_maps = [#map, #map, #map, #map, #map, #map, #map, #map, #map, #map, #map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], library_call = ""} ins(%200, %201, %202, %203, %204, %205, %206, %207, %208 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) outs(%209, %210, %211 : tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %in_3: f64, %in_4: f64, %in_5: f64, %in_6: f64, %in_7: f64, %out: f64, %out_8: f64, %out_9: f64):
      %238 = arith.mulf %in, %out : f64
      %239 = arith.mulf %in_0, %out_8 : f64
      %240 = arith.addf %238, %239 : f64
      %241 = arith.mulf %in_1, %out_9 : f64
      %242 = arith.addf %240, %241 : f64
      %243 = arith.mulf %in_2, %out : f64
      %244 = arith.mulf %in_3, %out_8 : f64
      %245 = arith.addf %243, %244 : f64
      %246 = arith.mulf %in_4, %out_9 : f64
      %247 = arith.addf %245, %246 : f64
      %248 = arith.mulf %in_5, %out : f64
      %249 = arith.mulf %in_6, %out_8 : f64
      %250 = arith.addf %248, %249 : f64
      %251 = arith.mulf %in_7, %out_9 : f64
      %252 = arith.addf %250, %251 : f64
      linalg.yield %242, %247, %252 : f64, f64, f64
    } -> (tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>, tensor<?x?x?x?xf64>)
    %213 = polygeist.submapInverse(%199, %212#0, %c2, %c5, %c5, %c5) {map = #map22} : (tensor<2x3x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x3x5x5x5xf64>
    %214 = polygeist.submapInverse(%213, %212#1, %c2, %c5, %c5, %c5) {map = #map29} : (tensor<2x3x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x3x5x5x5xf64>
    %215 = polygeist.submapInverse(%214, %212#2, %c2, %c5, %c5, %c5) {map = #map34} : (tensor<2x3x5x5x5xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<2x3x5x5x5xf64>
    %216 = polygeist.submap(%2, %c2, %c3, %c3, %c4, %c5, %c5, %c5) {map = #map44} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %217 = polygeist.submap(%2, %c2, %c3, %c3, %c4, %c5, %c5, %c5) {map = #map45} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %218 = polygeist.submap(%215, %c2, %c3, %c3, %c4, %c5, %c5, %c5) {map = #map46} : (tensor<2x3x5x5x5xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %219 = polygeist.submap(%3, %c2, %c3, %c3, %c4, %c5, %c5, %c5) {map = #map47} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %220 = polygeist.submap(%168, %c2, %c3, %c3, %c4) {map = #map48} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %221 = linalg.generic {doc = "", indexing_maps = [#map28, #map28, #map28, #map28, #map49], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"], library_call = ""} ins(%216, %217, %218, %219 : tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>) outs(%220 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %238 = arith.mulf %in_1, %in_2 : f64
      %239 = arith.mulf %238, %in_0 : f64
      %240 = arith.mulf %239, %in : f64
      %241 = arith.addf %out, %240 : f64
      linalg.yield %241 : f64
    } -> tensor<?x?x?x?xf64>
    %222 = polygeist.submapInverse(%168, %221, %c2, %c3, %c3, %c4) {map = #map48} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %223 = polygeist.submap(%2, %c2, %c3, %c4, %c3, %c5, %c5, %c5) {map = #map44} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %224 = polygeist.submap(%3, %c2, %c3, %c4, %c3, %c5, %c5, %c5) {map = #map45} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %225 = polygeist.submap(%215, %c2, %c3, %c4, %c3, %c5, %c5, %c5) {map = #map50} : (tensor<2x3x5x5x5xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %226 = polygeist.submap(%2, %c2, %c3, %c4, %c3, %c5, %c5, %c5) {map = #map47} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %227 = polygeist.submap(%222, %c2, %c3, %c4, %c3) {map = #map51} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %228 = linalg.generic {doc = "", indexing_maps = [#map28, #map28, #map28, #map28, #map49], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"], library_call = ""} ins(%223, %224, %225, %226 : tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>) outs(%227 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %238 = arith.mulf %in_1, %in_2 : f64
      %239 = arith.mulf %238, %in_0 : f64
      %240 = arith.mulf %239, %in : f64
      %241 = arith.addf %out, %240 : f64
      linalg.yield %241 : f64
    } -> tensor<?x?x?x?xf64>
    %229 = polygeist.submapInverse(%222, %228, %c2, %c3, %c4, %c3) {map = #map51} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %230 = polygeist.submap(%3, %c2, %c4, %c3, %c3, %c5, %c5, %c5) {map = #map44} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %231 = polygeist.submap(%2, %c2, %c4, %c3, %c3, %c5, %c5, %c5) {map = #map45} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %232 = polygeist.submap(%215, %c2, %c4, %c3, %c3, %c5, %c5, %c5) {map = #map52} : (tensor<2x3x5x5x5xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %233 = polygeist.submap(%2, %c2, %c4, %c3, %c3, %c5, %c5, %c5) {map = #map47} : (tensor<?xf64>, index, index, index, index, index, index, index) -> tensor<?x?x?x?x?x?x?xf64>
    %234 = polygeist.submap(%229, %c2, %c4, %c3, %c3) {map = #map53} : (tensor<?xf64>, index, index, index, index) -> tensor<?x?x?x?xf64>
    %235 = linalg.generic {doc = "", indexing_maps = [#map28, #map28, #map28, #map28, #map49], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"], library_call = ""} ins(%230, %231, %232, %233 : tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>, tensor<?x?x?x?x?x?x?xf64>) outs(%234 : tensor<?x?x?x?xf64>) {
    ^bb0(%in: f64, %in_0: f64, %in_1: f64, %in_2: f64, %out: f64):
      %238 = arith.mulf %in_1, %in_2 : f64
      %239 = arith.mulf %238, %in_0 : f64
      %240 = arith.mulf %239, %in : f64
      %241 = arith.addf %out, %240 : f64
      linalg.yield %241 : f64
    } -> tensor<?x?x?x?xf64>
    %236 = polygeist.submapInverse(%229, %235, %c2, %c4, %c3, %c3) {map = #map53} : (tensor<?xf64>, tensor<?x?x?x?xf64>, index, index, index, index) -> tensor<?xf64>
    %237 = bufferization.to_memref %236 : memref<?xf64>
    memref.copy %237, %arg9 : memref<?xf64> to memref<?xf64>
    return
  }
}
